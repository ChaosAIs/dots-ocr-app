"""
Planning tools for the Planner Agent.

These tools handle:
- Query decomposition into sub-questions
- Agent classification for tasks
- Execution plan creation
"""

import json
import logging
import uuid
from typing import Annotated, Literal, Optional

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agents.state.models import (
    SubTask,
    ExecutionPlan,
    SchemaGroup,
    AgentType,
    AggregationType
)
from agents.config import SCHEMA_CLASSIFICATIONS, AGGREGATION_KEYWORDS

logger = logging.getLogger(__name__)


@tool
def identify_sub_questions(
    query: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Identify distinct sub-questions or calculations within a complex query.

    Analyzes the user query to extract individual questions that can be
    processed independently or with dependencies.

    OPTIMIZATION: Multiple aggregations on the same data (sum, count, avg, min, max)
    are combined into a single SQL task since they can all be calculated in one query.
    For example: "sum inventory, count products, average inventory" becomes ONE task
    with aggregations=["sum", "count", "avg"].

    Args:
        query: The user's natural language query

    Returns:
        JSON array of identified sub-questions with:
        - id: Unique identifier
        - text: The sub-question text
        - type: Type of question (aggregation, search, extraction)
        - aggregation: Detected aggregation type if applicable (or list for combined)
        - target_fields: Detected field names
        - dependencies: IDs of questions this depends on
    """
    try:
        query_lower = query.lower()
        sub_questions = []

        # Split by common conjunctions and delimiters
        delimiters = [" and ", ", and ", ", ", " then ", " finally ", " also "]
        parts = [query]

        for delimiter in delimiters:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(delimiter))
            parts = new_parts

        # First pass: analyze each part
        raw_questions = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part or len(part) < 5:
                continue

            # Detect aggregation type
            aggregation = None
            for keyword, agg_type in AGGREGATION_KEYWORDS.items():
                if keyword in part.lower():
                    aggregation = agg_type
                    break

            # Detect question type
            if aggregation in ["sum", "count", "avg", "min", "max"]:
                q_type = "aggregation"
            elif any(word in part.lower() for word in ["find", "search", "show", "list"]):
                q_type = "search"
            elif any(word in part.lower() for word in ["extract", "get", "retrieve"]):
                q_type = "extraction"
            else:
                q_type = "aggregation" if aggregation else "search"

            # Extract potential field names (simple heuristic)
            target_fields = []
            common_fields = ["amount", "total", "price", "quantity", "inventory", "stock",
                            "sales", "revenue", "cost", "count", "date", "name", "product"]
            for field in common_fields:
                if field in part.lower():
                    target_fields.append(field)

            raw_questions.append({
                "text": part,
                "type": q_type,
                "aggregation": aggregation,
                "target_fields": target_fields,
            })

        # OPTIMIZATION: Combine multiple aggregations into a single task
        # This avoids redundant SQL calls for sum/count/avg on the same data
        aggregation_questions = [q for q in raw_questions if q["type"] == "aggregation"]
        other_questions = [q for q in raw_questions if q["type"] != "aggregation"]

        if len(aggregation_questions) > 1:
            # Check if aggregations can be combined (same target data)
            # Combine sum, count, avg into single query - they don't conflict
            combinable_aggs = {"sum", "count", "avg"}
            all_target_fields = set()
            combined_aggregations = []
            combined_texts = []

            for q in aggregation_questions:
                agg = q["aggregation"]
                if agg in combinable_aggs:
                    combined_aggregations.append(agg)
                    combined_texts.append(q["text"])
                    all_target_fields.update(q["target_fields"])
                else:
                    # min/max might need separate handling
                    other_questions.append(q)

            if len(combined_aggregations) > 1:
                # Create single combined task
                # Remove duplicates while preserving order
                unique_aggs = list(dict.fromkeys(combined_aggregations))
                combined_question = {
                    "id": "q_1",
                    "text": f"Combined: {'; '.join(combined_texts)}",
                    "type": "aggregation",
                    "aggregation": unique_aggs[0] if len(unique_aggs) == 1 else "combined",
                    "aggregations": unique_aggs,  # List of all aggregations to perform
                    "target_fields": list(all_target_fields),
                    "dependencies": [],
                    "combined": True,
                    "original_queries": combined_texts
                }
                sub_questions.append(combined_question)
                logger.info(f"Combined {len(combined_aggregations)} aggregations into single task: {unique_aggs}")
            elif len(combined_aggregations) == 1:
                # Just one aggregation, add as normal
                q = aggregation_questions[0]
                sub_questions.append({
                    "id": "q_1",
                    "text": q["text"],
                    "type": q["type"],
                    "aggregation": q["aggregation"],
                    "target_fields": q["target_fields"],
                    "dependencies": []
                })
        elif len(aggregation_questions) == 1:
            q = aggregation_questions[0]
            sub_questions.append({
                "id": "q_1",
                "text": q["text"],
                "type": q["type"],
                "aggregation": q["aggregation"],
                "target_fields": q["target_fields"],
                "dependencies": []
            })

        # Add non-aggregation questions
        for i, q in enumerate(other_questions, start=len(sub_questions) + 1):
            sub_questions.append({
                "id": f"q_{i}",
                "text": q["text"],
                "type": q["type"],
                "aggregation": q["aggregation"],
                "target_fields": q["target_fields"],
                "dependencies": []
            })

        logger.info(f"Identified {len(sub_questions)} sub-questions from query (after optimization)")
        return json.dumps(sub_questions, indent=2)

    except Exception as e:
        logger.error(f"Error identifying sub-questions: {e}")
        return json.dumps({"error": str(e), "sub_questions": []})


@tool
def classify_task_agent(
    task_description: str,
    schema_type: str,
    schema_fields: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Classify which agent should handle a task based on schema and data type.

    Routing Rules:
    - Tabular data with structured fields → sql_agent
    - Unstructured documents (invoices, receipts without tabular data) → vector_agent
    - Entity relationship queries → graph_agent
    - Mixed or unclear → generic_doc_agent

    Args:
        task_description: What the task needs to accomplish
        schema_type: Type of document schema
        schema_fields: JSON array of available fields in the schema

    Returns:
        JSON with recommended agent and reasoning
    """
    try:
        fields = json.loads(schema_fields) if schema_fields else []
        schema_type_lower = schema_type.lower() if schema_type else "unknown"
        task_lower = task_description.lower()

        # Check for relationship/graph queries
        graph_keywords = ["relationship", "connected", "related", "link", "path", "between"]
        if any(kw in task_lower for kw in graph_keywords):
            return json.dumps({
                "agent": AgentType.GRAPH_AGENT.value,
                "reasoning": "Query involves entity relationships, suitable for graph search",
                "confidence": 0.8
            })

        # Check schema classifications
        for category, config in SCHEMA_CLASSIFICATIONS.items():
            if schema_type_lower in [t.lower() for t in config["types"]]:
                if category == "structured":
                    return json.dumps({
                        "agent": AgentType.SQL_AGENT.value,
                        "reasoning": f"Schema type '{schema_type}' is structured/tabular",
                        "confidence": 0.9
                    })
                elif category == "semi_structured":
                    # Check if enough fields for SQL
                    min_fields = config.get("min_fields_for_sql", 3)
                    if len(fields) >= min_fields:
                        return json.dumps({
                            "agent": AgentType.SQL_AGENT.value,
                            "reasoning": f"Schema '{schema_type}' has {len(fields)} extracted fields",
                            "confidence": 0.8
                        })
                    else:
                        return json.dumps({
                            "agent": config.get("fallback_agent", AgentType.VECTOR_AGENT.value),
                            "reasoning": f"Schema '{schema_type}' has limited structure ({len(fields)} fields)",
                            "confidence": 0.7
                        })
                elif category == "unstructured":
                    return json.dumps({
                        "agent": AgentType.VECTOR_AGENT.value,
                        "reasoning": f"Schema type '{schema_type}' is unstructured",
                        "confidence": 0.85
                    })

        # Default to generic agent for unknown types
        return json.dumps({
            "agent": AgentType.GENERIC_DOC_AGENT.value,
            "reasoning": f"Schema type '{schema_type}' is unknown or mixed",
            "confidence": 0.6
        })

    except json.JSONDecodeError:
        return json.dumps({
            "agent": AgentType.GENERIC_DOC_AGENT.value,
            "reasoning": "Could not parse schema fields, using generic agent",
            "confidence": 0.5
        })
    except Exception as e:
        logger.error(f"Error classifying task agent: {e}")
        return json.dumps({
            "agent": AgentType.GENERIC_DOC_AGENT.value,
            "reasoning": f"Error during classification: {e}",
            "confidence": 0.4
        })


@tool
def create_execution_plan(
    sub_tasks: str,
    schema_groups: str,
    execution_strategy: Literal["parallel", "sequential", "mixed"],
    reasoning: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Create the final execution plan and hand off to retrieval team.

    The plan should:
    - Combine tasks for documents with similar schemas
    - Split tasks for documents with different schemas
    - Set proper dependencies between tasks
    - Route each task to the appropriate agent

    IMPORTANT: This tool stores the execution_plan in the state so it can be
    accessed by downstream agents. The state is passed by reference.

    Args:
        sub_tasks: JSON array of SubTask definitions
        schema_groups: JSON array of SchemaGroup definitions
        execution_strategy: How tasks should be executed (parallel/sequential/mixed)
        reasoning: Explanation of planning decisions

    Returns:
        JSON string with the execution plan details
    """
    try:
        # Parse inputs - handle potential malformed JSON from LLM
        def safe_json_parse(json_str: str, param_name: str):
            """Parse JSON with robust error handling for LLM outputs."""
            if not json_str or not json_str.strip():
                logger.warning(f"[create_execution_plan] Empty {param_name}, using empty list")
                return []

            # Clean up common LLM JSON issues
            cleaned = json_str.strip()

            # Remove markdown code blocks if present
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines if they're code block markers
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            # Try to find and extract just the JSON array
            # Handle case where LLM adds extra text after JSON
            start_idx = cleaned.find("[")
            if start_idx != -1:
                # Find matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                for i, char in enumerate(cleaned[start_idx:], start_idx):
                    if char == "[":
                        bracket_count += 1
                    elif char == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    cleaned = cleaned[start_idx:end_idx]

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"[create_execution_plan] JSON parse error for {param_name}: {e}")
                logger.error(f"[create_execution_plan] Raw input (first 500 chars): {json_str[:500]}")
                raise

        parsed_tasks_raw = safe_json_parse(sub_tasks, "sub_tasks")
        parsed_groups_raw = safe_json_parse(schema_groups, "schema_groups")

        # Convert to proper models
        parsed_tasks = []
        for t in parsed_tasks_raw:
            # Ensure required fields
            if "task_id" not in t:
                t["task_id"] = t.get("id", f"task_{uuid.uuid4().hex[:8]}")
            if "target_agent" not in t:
                t["target_agent"] = AgentType.SQL_AGENT.value
            if "schema_type" not in t:
                t["schema_type"] = "unknown"
            # Handle LLM using 'text' instead of 'description'
            if "description" not in t and "text" in t:
                t["description"] = t.pop("text")
            if "description" not in t:
                t["description"] = t.get("original_query_part", "Task from query")
            # Ensure document_ids is set
            if "document_ids" not in t:
                t["document_ids"] = []

            # Handle combined aggregations from identify_sub_questions
            # The 'aggregations' field contains list like ['sum', 'count', 'avg']
            if "aggregations" in t and isinstance(t["aggregations"], list):
                # Store the list in aggregation_types
                t["aggregation_types"] = t.pop("aggregations")
                # Set aggregation_type to "combined"
                t["aggregation_type"] = "combined"
            elif "aggregation_type" in t:
                # Handle case where LLM sends comma-separated string like "sum,count,avg"
                agg_type = t["aggregation_type"]
                if isinstance(agg_type, str) and "," in agg_type:
                    t["aggregation_types"] = [a.strip() for a in agg_type.split(",")]
                    t["aggregation_type"] = "combined"
                elif isinstance(agg_type, list):
                    t["aggregation_types"] = agg_type
                    t["aggregation_type"] = "combined"

            parsed_tasks.append(SubTask(**t))

        parsed_groups = []
        for g in parsed_groups_raw:
            # Handle nested documents
            if "documents" in g and isinstance(g["documents"], list):
                # Keep documents as dicts for now
                pass
            parsed_groups.append(SchemaGroup(**g))

        # Calculate document statistics
        docs_by_schema = {}
        total_docs = 0
        for group in parsed_groups:
            doc_count = group.document_count or len(group.documents)
            docs_by_schema[group.schema_type] = doc_count
            total_docs += doc_count

        # Create the execution plan
        plan = ExecutionPlan(
            sub_tasks=parsed_tasks,
            schema_groups=parsed_groups,
            execution_strategy=execution_strategy,
            total_documents=total_docs,
            documents_by_schema=docs_by_schema,
            reasoning=reasoning
        )

        logger.info(
            f"Created execution plan: {len(parsed_tasks)} tasks, "
            f"{total_docs} documents, strategy={execution_strategy}"
        )

        # Return full plan data so workflow wrapper can extract and store it
        # Include serialized plan for downstream processing
        def serialize_documents(docs):
            """Serialize document list, handling both dicts and DocumentSource objects."""
            result = []
            for doc in docs:
                if hasattr(doc, 'model_dump'):
                    result.append(doc.model_dump())
                elif hasattr(doc, 'dict'):
                    result.append(doc.dict())
                elif isinstance(doc, dict):
                    result.append(doc)
                else:
                    result.append({"document_id": str(doc)})
            return result

        plan_dict = {
            "sub_tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "target_agent": t.target_agent,
                    "schema_type": t.schema_type,
                    "document_ids": t.document_ids,
                    "aggregation_type": t.aggregation_type,
                    "aggregation_types": t.aggregation_types,  # List for combined aggregations
                    "dependencies": t.dependencies
                }
                for t in parsed_tasks
            ],
            "schema_groups": [
                {
                    "group_id": g.group_id,
                    "schema_type": g.schema_type,
                    "common_fields": g.common_fields,
                    "documents": serialize_documents(g.documents),
                    "can_combine": g.can_combine,
                    "document_count": g.document_count
                }
                for g in parsed_groups
            ],
            "execution_strategy": execution_strategy,
            "total_documents": total_docs,
            "documents_by_schema": docs_by_schema,
            "reasoning": reasoning
        }

        return json.dumps({
            "success": True,
            "message": f"Execution plan created with {len(parsed_tasks)} tasks across {total_docs} documents",
            "execution_plan": plan_dict,
            "strategy": execution_strategy,
            "task_count": len(parsed_tasks),
            "document_count": total_docs,
            "tasks": [{"task_id": t.task_id, "description": t.description, "target_agent": t.target_agent} for t in parsed_tasks],
            "reasoning": reasoning
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in create_execution_plan: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to parse execution plan: {e}"
        })
    except Exception as e:
        logger.error(f"Error creating execution plan: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })
