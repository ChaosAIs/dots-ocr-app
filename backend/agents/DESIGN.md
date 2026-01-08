# Agentic AI Flow Design for Multi-Query Analytics

## Overview

This document describes the design for a dynamic, self-orchestrating multi-agent system using **LangGraph 1.0+** and **LangChain 2025** patterns. The system intelligently handles complex analytical queries by decomposing them into sub-tasks, routing to appropriate document sources, and coordinating specialized retrieval agents.

### Key Principles

1. **No Hardcoded Workflows** - All routing decisions made by LLM-powered agents
2. **Schema-Driven** - Planner uses actual document schema metadata for intelligent task grouping
3. **Self-Correcting** - Reviewer agent enables automatic refinement loops
4. **Graceful Degradation** - Fallback to existing pipeline if agents fail
5. **Observable** - Full logging and metrics for each agent decision

---

## Architecture

### High-Level Flow

```
                              ┌─────────────────────────────────────┐
                              │         USER QUERY INPUT            │
                              │  "sum total inventory, count        │
                              │   products, average inventory"      │
                              └─────────────────┬───────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        PLANNER AGENT (Orchestrator)                                 │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │  RESPONSIBILITIES:                                                            │  │
│  │  1. Query Analysis & Decomposition                                            │  │
│  │  2. Document Source Routing (using existing routing feature)                  │  │
│  │  3. Schema-Aware Task Classification                                          │  │
│  │  4. Smart Grouping (combine similar schemas / split different ones)           │  │
│  │  5. Execution Plan Generation with Dependencies                               │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                        PLANNING DECISION FLOW                               │    │
│  │                                                                             │    │
│  │   Query Analysis ──► Get Relevant Documents ──► Analyze Schemas             │    │
│  │         │                    │                        │                     │    │
│  │         ▼                    ▼                        ▼                     │    │
│  │   Identify Sub-      Filter by Schema          Group by Schema Type         │    │
│  │   Questions          Type & Relevance          (tabular/invoice/etc)        │    │
│  │         │                    │                        │                     │    │
│  │         └────────────────────┴────────────────────────┘                     │    │
│  │                              │                                              │    │
│  │                              ▼                                              │    │
│  │                    Create Execution Plan                                    │    │
│  │                    ┌─────────────────────────────────────────┐              │    │
│  │                    │ Similar Schemas → Combine into 1 task   │              │    │
│  │                    │ Different Schemas → Split into N tasks  │              │    │
│  │                    │ Unstructured → Vector/Graph search      │              │    │
│  │                    └─────────────────────────────────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                    Command(goto="retrieval_team")
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL TEAM (Sub-Supervisor)                           │
│                                                                                     │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│   │   SQL AGENT     │  │  VECTOR AGENT   │  │   GRAPH AGENT   │  │  GENERIC     │  │
│   │                 │  │                 │  │                 │  │  DOC AGENT   │  │
│   │ • Tabular data  │  │ • Semantic      │  │ • Entity        │  │              │  │
│   │ • Structured    │  │   similarity    │  │   relationships │  │ • Mixed docs │  │
│   │   aggregations  │  │ • Unstructured  │  │ • Knowledge     │  │ • Hybrid     │  │
│   │ • Schema-aware  │  │   documents     │  │   graph queries │  │   search     │  │
│   │   SQL gen       │  │ • RAG retrieval │  │ • Neo4j Cypher  │  │ • Fallback   │  │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘  └──────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                              Command(goto="reviewer", graph=Command.PARENT)
                                                │
                                                ▼
                    ┌───────────────────────────────────────────────────────┐
                    │                  REVIEWER AGENT                       │
                    │  • Validate completeness & quality                    │
                    │  • Cross-validate results across agents               │
                    │  • Decision: APPROVE / REFINE / ESCALATE              │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         │                                             │
              Command(goto="retrieval_team")              Command(goto="summary_agent")
              + refinement guidance                                    │
                         │                                             ▼
                  [Feedback Loop]                    ┌─────────────────────────────┐
                                                     │      SUMMARY AGENT          │
                                                     │  • Aggregate all outputs    │
                                                     │  • Synthesize response      │
                                                     │  • Format for user          │
                                                     └─────────────────────────────┘
```

---

## State Models

### Core Pydantic Models

```python
from typing import Annotated, Literal, Sequence, TypedDict, Optional, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import operator

# Document source with schema information
class DocumentSource(BaseModel):
    """Represents a document with its schema metadata."""
    document_id: str
    filename: str
    schema_type: str  # e.g., "tabular", "invoice", "receipt", "inventory_report"
    schema_definition: Dict[str, Any]  # Field definitions from DataSchema
    relevance_score: float
    data_location: Literal["header", "line_items", "summary"]

# Schema group for combining similar documents
class SchemaGroup(BaseModel):
    """Groups documents with compatible schemas for efficient processing."""
    group_id: str
    schema_type: str
    common_fields: List[str]  # Fields shared across documents in group
    documents: List[DocumentSource]
    can_combine: bool  # Whether documents can be processed together in single query

# Enhanced sub-task with document context
class SubTask(BaseModel):
    """Individual task for retrieval agents."""
    task_id: str
    description: str
    original_query_part: str
    target_agent: Literal["sql_agent", "vector_agent", "graph_agent", "generic_doc_agent"]

    # Document source context
    document_ids: List[str]
    schema_group: Optional[SchemaGroup] = None
    schema_type: str

    # Execution hints
    aggregation_type: Optional[Literal["sum", "count", "avg", "min", "max", "group_by"]] = None
    target_fields: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)

    # Dependencies
    dependencies: List[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"

# Execution plan
class ExecutionPlan(BaseModel):
    """Complete execution plan generated by Planner Agent."""
    sub_tasks: List[SubTask]
    schema_groups: List[SchemaGroup]
    execution_strategy: Literal["parallel", "sequential", "mixed"]

    # Document routing metadata
    total_documents: int
    documents_by_schema: Dict[str, int]  # schema_type -> count

    reasoning: str

# Agent output with source tracking
class AgentOutput(BaseModel):
    """Output from any retrieval agent."""
    task_id: str
    agent_name: str
    status: Literal["success", "partial", "failed"]
    data: Any

    # Source tracking
    documents_used: List[str]
    schema_type: str
    query_executed: Optional[str] = None  # SQL or Cypher query

    confidence: float
    row_count: Optional[int] = None
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

# Review decision
class ReviewDecision(BaseModel):
    """Decision from Reviewer Agent."""
    decision: Literal["approve", "refine", "escalate"]
    approved_task_ids: List[str] = Field(default_factory=list)
    refinement_requests: List[Dict[str, Any]] = Field(default_factory=list)
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    reasoning: str
```

### Main Graph State

```python
class AnalyticsAgentState(TypedDict):
    """Main state for the analytics workflow graph."""

    # Input
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    workspace_id: str
    chat_history: List[Dict[str, str]]

    # Document context (populated by planner)
    available_documents: List[DocumentSource]
    schema_groups: List[SchemaGroup]

    # Planning
    execution_plan: Optional[ExecutionPlan]

    # Execution - using operator.add as reducer for list accumulation
    agent_outputs: Annotated[List[AgentOutput], operator.add]

    # Review
    review_iteration: int
    max_iterations: int
    review_decision: Optional[ReviewDecision]

    # Output
    final_response: Optional[str]
    data_sources: List[str]

    # Control
    active_agent: Optional[str]
```

---

## Agent Definitions

### 1. Planner Agent (Orchestrator)

The Planner Agent is the central orchestrator responsible for:
- Query analysis and decomposition
- Document source routing
- Schema-aware task classification
- Smart grouping of similar schemas
- Execution plan generation

#### Tools

| Tool | Purpose |
|------|---------|
| `get_relevant_documents` | Route to relevant documents using existing document routing feature |
| `analyze_document_schemas` | Analyze schemas to identify commonalities and differences |
| `group_documents_by_schema` | Group documents by compatible schemas |
| `identify_sub_questions` | Extract distinct questions from complex queries |
| `classify_task_agent` | Determine which agent should handle each task |
| `create_execution_plan` | Generate final execution plan and hand off |

#### Implementation

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from typing import Annotated

@tool
def get_relevant_documents(
    query: str,
    workspace_id: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Get relevant documents for the query using existing document routing feature.

    Integrates with unified_query_analyzer and document filtering.
    Returns documents with their schema types and relevance scores.
    """
    # Integration point: Call existing document routing service
    pass

@tool
def analyze_document_schemas(
    document_ids: str,  # JSON array
    state: Annotated[dict, InjectedState]
) -> str:
    """Analyze schemas of selected documents to identify commonalities."""
    # Integration point: Call schema_service
    pass

@tool
def group_documents_by_schema(
    documents: str,  # JSON array of DocumentSource
    state: Annotated[dict, InjectedState]
) -> str:
    """Group documents by compatible schemas for efficient processing.

    Documents with similar schemas can be combined into single SQL queries.
    Documents with different schemas must be processed separately.
    """
    import json
    docs = json.loads(documents)

    # Group by schema_type
    groups = {}
    for doc in docs:
        schema_type = doc["schema_type"]
        if schema_type not in groups:
            groups[schema_type] = {
                "group_id": f"group_{schema_type}",
                "schema_type": schema_type,
                "documents": [],
                "common_fields": set()
            }
        groups[schema_type]["documents"].append(doc)

        # Track common fields
        doc_fields = set(doc.get("schema_definition", {}).get("fields", []))
        if not groups[schema_type]["common_fields"]:
            groups[schema_type]["common_fields"] = doc_fields
        else:
            groups[schema_type]["common_fields"] &= doc_fields

    # Determine if groups can be combined
    result = []
    for schema_type, group in groups.items():
        result.append({
            "group_id": group["group_id"],
            "schema_type": schema_type,
            "common_fields": list(group["common_fields"]),
            "documents": group["documents"],
            "can_combine": len(group["documents"]) > 1 and len(group["common_fields"]) > 0,
            "document_count": len(group["documents"])
        })

    return json.dumps(result, indent=2)

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
    - Unstructured documents → vector_agent
    - Entity relationship queries → graph_agent
    - Mixed or unclear → generic_doc_agent
    """
    structured_types = ["tabular", "spreadsheet", "csv", "inventory_report", "financial_report"]
    semi_structured_types = ["invoice", "receipt", "purchase_order", "bank_statement"]

    if schema_type.lower() in structured_types:
        return json.dumps({"agent": "sql_agent", "reasoning": "Tabular/structured data"})
    elif schema_type.lower() in semi_structured_types:
        fields = json.loads(schema_fields) if schema_fields else []
        if len(fields) > 3:
            return json.dumps({"agent": "sql_agent", "reasoning": "Has extracted fields"})
        return json.dumps({"agent": "vector_agent", "reasoning": "Limited structure"})
    return json.dumps({"agent": "generic_doc_agent", "reasoning": "Unknown type"})

@tool
def create_execution_plan(
    sub_tasks: str,
    schema_groups: str,
    execution_strategy: Literal["parallel", "sequential", "mixed"],
    reasoning: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create the final execution plan and hand off to retrieval team."""
    import json

    parsed_tasks = [SubTask(**t) for t in json.loads(sub_tasks)]
    parsed_groups = [SchemaGroup(**g) for g in json.loads(schema_groups)]

    plan = ExecutionPlan(
        sub_tasks=parsed_tasks,
        schema_groups=parsed_groups,
        execution_strategy=execution_strategy,
        total_documents=sum(len(g.documents) for g in parsed_groups),
        documents_by_schema={g.schema_type: len(g.documents) for g in parsed_groups},
        reasoning=reasoning
    )

    return Command(
        goto="retrieval_team",
        update={
            "execution_plan": plan,
            "schema_groups": parsed_groups,
            "messages": [ToolMessage(
                content=f"Plan created: {len(parsed_tasks)} tasks",
                tool_call_id=tool_call_id
            )]
        }
    )

def create_planner_agent(llm: ChatOpenAI):
    """Create the Planner/Orchestrator agent."""

    planner_prompt = """You are the Planner Agent (Orchestrator) responsible for:
1. Analyzing user queries to identify sub-questions
2. Routing to relevant document sources
3. Analyzing document schemas for smart task grouping
4. Creating optimized execution plans

## Workflow:

### Step 1: Query Analysis
- Identify all distinct questions/calculations
- Detect aggregation types (sum, count, avg, etc.)

### Step 2: Document Source Routing
- Use get_relevant_documents to find matching documents
- Consider schema types and relevance scores

### Step 3: Schema Analysis & Grouping
- Use group_documents_by_schema to create efficient groups
- COMBINE: Same schema + common fields → single task
- SPLIT: Different schemas → separate tasks

### Step 4: Task Classification
- Tabular → sql_agent
- Unstructured → vector_agent
- Relationships → graph_agent
- Mixed → generic_doc_agent

### Step 5: Create Execution Plan
- Set dependencies between tasks
- Choose strategy: parallel/sequential/mixed

Always use create_execution_plan to finalize."""

    return create_react_agent(
        model=llm,
        tools=[
            get_relevant_documents,
            analyze_document_schemas,
            group_documents_by_schema,
            identify_sub_questions,
            classify_task_agent,
            create_execution_plan
        ],
        prompt=planner_prompt,
        name="planner_agent"
    )
```

---

### 2. SQL Agent

Handles structured/tabular data queries with schema-aware SQL generation.

#### Tools

| Tool | Purpose |
|------|---------|
| `generate_schema_aware_sql` | Generate SQL aware of document schema grouping |
| `execute_sql_with_retry` | Execute SQL with automatic error correction |
| `report_sql_result` | Report results back to supervisor |

#### Implementation

```python
@tool
def generate_schema_aware_sql(
    task_description: str,
    document_ids: str,
    schema_group: str,
    aggregation_type: Optional[str],
    target_fields: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Generate SQL query aware of document schema grouping.

    For combined documents (same schema):
    - Generate single query with document_id IN (...)

    For single document:
    - Generate focused query for that document
    """
    # Integration with LLMSQLGeneratorV2
    pass

@tool
def execute_sql_with_retry(
    sql_query: str,
    max_retries: int,
    state: Annotated[dict, InjectedState]
) -> str:
    """Execute SQL with automatic error correction retry."""
    # Integration with SQLQueryExecutor
    pass

@tool
def report_sql_result(
    task_id: str,
    data: str,
    documents_used: str,
    schema_type: str,
    sql_executed: str,
    row_count: int,
    confidence: float,
    issues: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Report SQL agent results."""
    output = AgentOutput(
        task_id=task_id,
        agent_name="sql_agent",
        status="success" if confidence > 0.7 else "partial",
        data=json.loads(data),
        documents_used=json.loads(documents_used),
        schema_type=schema_type,
        query_executed=sql_executed,
        confidence=confidence,
        row_count=row_count,
        issues=json.loads(issues) if issues else []
    )

    return Command(
        goto="retrieval_supervisor",
        update={"agent_outputs": [output]}
    )

def create_sql_agent(llm: ChatOpenAI):
    prompt = """You are the SQL Retrieval Agent for structured/tabular data.

## Capabilities:
- Generate SQL for aggregations (SUM, COUNT, AVG, MIN, MAX)
- Handle grouped queries (GROUP BY)
- Process multiple documents with same schema in single query
- Automatic error correction with retry

## Query Pattern:
```sql
WITH documents_data AS (...)
SELECT SUM(CAST(header_data->>'amount' AS NUMERIC))
FROM documents_data dd
WHERE dd.document_id IN ('doc1', 'doc2')
```

Always report results with the SQL executed."""

    return create_react_agent(
        model=llm,
        tools=[generate_schema_aware_sql, execute_sql_with_retry, report_sql_result],
        prompt=prompt,
        name="sql_agent"
    )
```

---

### 3. Vector Agent

Handles semantic search on unstructured documents.

#### Tools

| Tool | Purpose |
|------|---------|
| `semantic_search` | Perform semantic similarity search via Qdrant |
| `report_vector_result` | Report results back to supervisor |

#### Implementation

```python
@tool
def semantic_search(
    query: str,
    document_ids: str,
    top_k: int,
    filters: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Perform semantic similarity search via Qdrant."""
    # Integration with existing Qdrant client
    pass

@tool
def report_vector_result(
    task_id: str,
    documents: str,
    confidence: float,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Report vector search results."""
    output = AgentOutput(
        task_id=task_id,
        agent_name="vector_agent",
        status="success" if confidence > 0.7 else "partial",
        data=json.loads(documents),
        documents_used=[],
        schema_type="unstructured",
        confidence=confidence,
        issues=[]
    )

    return Command(
        goto="retrieval_supervisor",
        update={"agent_outputs": [output]}
    )

def create_vector_agent(llm: ChatOpenAI):
    prompt = """You are the Vector Search Agent for semantic document retrieval.

## Capabilities:
- Semantic similarity search
- Unstructured document retrieval
- RAG-based question answering

Always report results with confidence scores."""

    return create_react_agent(
        model=llm,
        tools=[semantic_search, report_vector_result],
        prompt=prompt,
        name="vector_agent"
    )
```

---

### 4. Graph Agent

Handles entity relationship queries via Neo4j.

#### Tools

| Tool | Purpose |
|------|---------|
| `cypher_query` | Execute Cypher queries against Neo4j |
| `report_graph_result` | Report results back to supervisor |

#### Implementation

```python
@tool
def cypher_query(
    query: str,
    entity_hints: str,
    max_depth: int,
    state: Annotated[dict, InjectedState]
) -> str:
    """Execute Cypher query against Neo4j knowledge graph."""
    # Integration with Neo4j
    pass

@tool
def report_graph_result(
    task_id: str,
    entities: str,
    relationships: str,
    confidence: float,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Report graph search results."""
    output = AgentOutput(
        task_id=task_id,
        agent_name="graph_agent",
        status="success" if confidence > 0.7 else "partial",
        data={"entities": json.loads(entities), "relationships": json.loads(relationships)},
        documents_used=[],
        schema_type="graph",
        confidence=confidence,
        issues=[]
    )

    return Command(
        goto="retrieval_supervisor",
        update={"agent_outputs": [output]}
    )

def create_graph_agent(llm: ChatOpenAI):
    prompt = """You are the Graph Search Agent for entity relationship queries.

## Capabilities:
- Execute Cypher queries
- Find entity relationships
- Multi-hop traversal

Report entities and relationships found."""

    return create_react_agent(
        model=llm,
        tools=[cypher_query, report_graph_result],
        prompt=prompt,
        name="graph_agent"
    )
```

---

### 5. Generic Document Agent (NEW)

Hybrid/fallback agent for mixed or unknown document types.

#### Tools

| Tool | Purpose |
|------|---------|
| `hybrid_document_search` | Combine vector and SQL approaches |
| `extract_and_query` | Extract structured data then query |
| `fallback_rag_search` | Standard RAG fallback |
| `report_generic_result` | Report results back to supervisor |

#### Implementation

```python
@tool
def hybrid_document_search(
    query: str,
    document_ids: str,
    search_strategy: Literal["vector_first", "sql_first", "parallel"],
    state: Annotated[dict, InjectedState]
) -> str:
    """Perform hybrid search combining vector and SQL approaches."""
    pass

@tool
def extract_and_query(
    query: str,
    document_ids: str,
    extraction_fields: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Extract structured data from documents then query it."""
    pass

@tool
def fallback_rag_search(
    query: str,
    document_ids: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Fallback to standard RAG when other methods fail."""
    pass

@tool
def report_generic_result(
    task_id: str,
    data: str,
    documents_used: str,
    confidence: float,
    method_used: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Report generic document agent results."""
    output = AgentOutput(
        task_id=task_id,
        agent_name="generic_doc_agent",
        status="success" if confidence > 0.6 else "partial",
        data=json.loads(data),
        documents_used=json.loads(documents_used),
        schema_type="mixed",
        confidence=confidence,
        issues=[] if confidence > 0.6 else [f"Low confidence using {method_used}"]
    )

    return Command(
        goto="retrieval_supervisor",
        update={"agent_outputs": [output]}
    )

def create_generic_doc_agent(llm: ChatOpenAI):
    prompt = """You are the Generic Document Agent for mixed/unknown document types.

## Strategy Order:
1. Try hybrid_document_search first
2. If structured data detected, use extract_and_query
3. Fall back to fallback_rag_search if other methods fail

Always report results with confidence and method used."""

    return create_react_agent(
        model=llm,
        tools=[hybrid_document_search, extract_and_query, fallback_rag_search, report_generic_result],
        prompt=prompt,
        name="generic_doc_agent"
    )
```

---

### 6. Retrieval Team Supervisor

Coordinates all retrieval agents using `langgraph-supervisor`.

```python
from langgraph_supervisor import create_supervisor

def create_retrieval_team(llm: ChatOpenAI):
    """Create retrieval team with all four specialized agents."""

    sql_agent = create_sql_agent(llm)
    vector_agent = create_vector_agent(llm)
    graph_agent = create_graph_agent(llm)
    generic_doc_agent = create_generic_doc_agent(llm)

    retrieval_supervisor = create_supervisor(
        agents=[sql_agent, vector_agent, graph_agent, generic_doc_agent],
        model=llm,
        prompt="""You are the Retrieval Team Supervisor coordinating:
1. sql_agent: Structured/tabular data
2. vector_agent: Semantic search
3. graph_agent: Entity relationships
4. generic_doc_agent: Mixed/unknown types

## Workflow:
1. Review execution_plan from state
2. Delegate each sub_task to target_agent
3. Monitor completion
4. Hand off to reviewer_agent when done

## Fallback:
If an agent fails → try generic_doc_agent""",
        supervisor_name="retrieval_supervisor",
        output_mode="full_history"
    )

    return retrieval_supervisor
```

---

### 7. Reviewer Agent

Quality control and decision-making for retrieval results.

#### Tools

| Tool | Purpose |
|------|---------|
| `validate_completeness` | Check all tasks have outputs |
| `check_data_quality` | Evaluate quality and consistency |
| `approve_and_continue` | Approve and hand off to summary |
| `request_refinement` | Send back for refinement |

#### Implementation

```python
@tool
def validate_completeness(
    execution_plan: str,
    agent_outputs: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Check if all planned tasks have corresponding outputs."""
    plan = json.loads(execution_plan)
    outputs = json.loads(agent_outputs)

    completed_ids = {o["task_id"] for o in outputs}
    planned_ids = {t["task_id"] for t in plan["sub_tasks"]}

    missing = planned_ids - completed_ids
    if missing:
        return f"INCOMPLETE: Missing tasks: {missing}"
    return "COMPLETE: All tasks have results"

@tool
def check_data_quality(
    agent_outputs: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Evaluate data quality and consistency."""
    outputs = json.loads(agent_outputs)

    issues = []
    for output in outputs:
        if output["confidence"] < 0.5:
            issues.append(f"Low confidence: {output['task_id']}")

    if issues:
        return f"QUALITY ISSUES: {issues}"
    return "QUALITY OK"

@tool
def approve_and_continue(
    reasoning: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Approve outputs and hand off to summary agent."""
    decision = ReviewDecision(
        decision="approve",
        approved_task_ids=[o.task_id for o in state.get("agent_outputs", [])],
        reasoning=reasoning
    )

    return Command(
        goto="summary_agent",
        update={"review_decision": decision}
    )

@tool
def request_refinement(
    task_ids: str,
    guidance: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Request refinement from retrieval team."""
    iteration = state.get("review_iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration >= max_iter:
        # Force approval after max iterations
        return Command(
            goto="summary_agent",
            update={
                "review_decision": ReviewDecision(
                    decision="approve",
                    reasoning=f"Max iterations reached"
                )
            }
        )

    return Command(
        goto="retrieval_team",
        update={
            "review_iteration": iteration + 1,
            "review_decision": ReviewDecision(
                decision="refine",
                refinement_requests=[{"task_ids": json.loads(task_ids), "guidance": guidance}],
                reasoning=f"Refinement requested"
            )
        }
    )

def create_reviewer_agent(llm: ChatOpenAI):
    prompt = """You are the Reviewer Agent for quality validation.

## Responsibilities:
1. Check all tasks have outputs
2. Validate quality and confidence
3. Decide: APPROVE or REFINE

## Guidelines:
- APPROVE if confidence > 0.7
- REFINE if data missing or low confidence
- Always approve after max iterations"""

    return create_react_agent(
        model=llm,
        tools=[validate_completeness, check_data_quality, approve_and_continue, request_refinement],
        prompt=prompt,
        name="reviewer_agent"
    )
```

---

### 8. Summary Agent

Synthesizes final response from approved outputs.

#### Tools

| Tool | Purpose |
|------|---------|
| `aggregate_results` | Combine all agent outputs |
| `format_response` | Format final user response |

#### Implementation

```python
@tool
def aggregate_results(
    agent_outputs: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Aggregate all agent outputs into unified structure."""
    outputs = json.loads(agent_outputs)
    aggregated = {o["task_id"]: o["data"] for o in outputs}
    return json.dumps(aggregated, indent=2)

@tool
def format_response(
    aggregated_data: str,
    original_query: str,
    data_sources: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Format the final response for user consumption."""
    return Command(
        goto="__end__",
        update={
            "final_response": aggregated_data,
            "data_sources": json.loads(data_sources)
        }
    )

def create_summary_agent(llm: ChatOpenAI):
    prompt = """You are the Summary Agent for final response synthesis.

## Output Format:
## Summary:
- **Metric 1:** value
- **Metric 2:** value

**Data Sources:**
- source1.csv
- source2.pdf"""

    return create_react_agent(
        model=llm,
        tools=[aggregate_results, format_response],
        prompt=prompt,
        name="summary_agent"
    )
```

---

## Workflow Assembly

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

def create_analytics_workflow(llm: ChatOpenAI):
    """Create the complete multi-agent analytics workflow."""

    # Create agents
    planner = create_planner_agent(llm)
    retrieval_team = create_retrieval_team(llm)
    reviewer = create_reviewer_agent(llm)
    summarizer = create_summary_agent(llm)

    # Build graph
    builder = StateGraph(AnalyticsAgentState)

    # Add nodes
    builder.add_node("planner_agent", planner)
    builder.add_node("retrieval_team", retrieval_team.compile(name="retrieval_team"))
    builder.add_node("reviewer_agent", reviewer)
    builder.add_node("summary_agent", summarizer)

    # Define edges
    builder.add_edge(START, "planner_agent")
    builder.add_edge("planner_agent", "retrieval_team")
    builder.add_edge("retrieval_team", "reviewer_agent")

    # Conditional routing from reviewer
    def route_after_review(state: AnalyticsAgentState) -> Literal["summary_agent", "retrieval_team"]:
        decision = state.get("review_decision")
        if decision and decision.decision == "refine":
            return "retrieval_team"
        return "summary_agent"

    builder.add_conditional_edges(
        "reviewer_agent",
        route_after_review,
        {"summary_agent": "summary_agent", "retrieval_team": "retrieval_team"}
    )

    builder.add_edge("summary_agent", END)

    # Compile with checkpointing
    return builder.compile(
        checkpointer=InMemorySaver(),
        store=InMemoryStore()
    )
```

---

## File Structure

```
backend/analytics_service/agentic/
├── __init__.py
├── config.py                       # Configuration constants
├── workflow.py                     # Main graph assembly
│
├── state/
│   ├── __init__.py
│   ├── models.py                   # Pydantic models
│   └── schema.py                   # AnalyticsAgentState
│
├── agents/
│   ├── __init__.py
│   ├── planner.py                  # Planner/Orchestrator agent
│   ├── sql_agent.py                # SQL retrieval agent
│   ├── vector_agent.py             # Vector search agent
│   ├── graph_agent.py              # Graph search agent
│   ├── generic_doc_agent.py        # Hybrid/fallback agent
│   ├── reviewer.py                 # Quality control agent
│   └── summary.py                  # Response synthesis agent
│
├── tools/
│   ├── __init__.py
│   ├── routing_tools.py            # Document routing tools
│   ├── planning_tools.py           # Query decomposition tools
│   ├── grouping_tools.py           # Schema grouping tools
│   ├── sql_tools.py                # SQL generation/execution
│   ├── vector_tools.py             # Semantic search tools
│   ├── graph_tools.py              # Cypher query tools
│   ├── generic_tools.py            # Hybrid search tools
│   ├── review_tools.py             # Validation tools
│   └── summary_tools.py            # Aggregation/formatting
│
└── integration/
    ├── __init__.py
    ├── document_router.py          # Bridge to existing routing
    ├── schema_service_bridge.py    # Bridge to schema_service
    ├── sql_generator_bridge.py     # Bridge to LLMSQLGeneratorV2
    ├── vector_bridge.py            # Bridge to Qdrant
    ├── graph_bridge.py             # Bridge to Neo4j
    └── formatter_bridge.py         # Bridge to LLMResultFormatter
```

---

## Configuration

```python
# backend/analytics_service/agentic/config.py

AGENTIC_CONFIG = {
    # Workflow settings
    "max_review_iterations": 3,
    "default_confidence_threshold": 0.7,
    "enable_streaming": True,

    # Agent timeouts (seconds)
    "planner_timeout": 30,
    "retrieval_timeout": 60,
    "reviewer_timeout": 20,
    "summary_timeout": 30,

    # Feature flags
    "enable_graph_search": True,
    "enable_vector_search": True,
    "fallback_to_legacy": True,

    # Checkpointing
    "checkpoint_backend": "memory",  # "memory", "postgres", "redis"
}

AGENT_LLM_CONFIG = {
    "planner": {"model": "gpt-4o", "temperature": 0.1},
    "sql_agent": {"model": "gpt-4o", "temperature": 0.0},
    "vector_agent": {"model": "gpt-4o-mini", "temperature": 0.0},
    "graph_agent": {"model": "gpt-4o-mini", "temperature": 0.0},
    "generic_doc_agent": {"model": "gpt-4o", "temperature": 0.1},
    "retrieval_supervisor": {"model": "gpt-4o", "temperature": 0.1},
    "reviewer": {"model": "gpt-4o", "temperature": 0.1},
    "summary": {"model": "gpt-4o", "temperature": 0.3},
}
```

---

## Dependencies

```txt
# requirements.txt additions
langgraph>=1.0.5
langgraph-supervisor>=0.1.0
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-core>=0.3.0
pydantic>=2.0
```

---

## Example Execution

### Input Query

```
"sum the total inventory for all products, count how many products we have, finally average inventory for product"
```

### Step 1: Planner Agent

```json
{
  "sub_questions": [
    {"id": "q1", "text": "sum total inventory", "type": "aggregation"},
    {"id": "q2", "text": "count products", "type": "aggregation"},
    {"id": "q3", "text": "average inventory", "type": "aggregation"}
  ],
  "relevant_documents": [
    {"id": "doc1", "name": "ProductInventory.csv", "schema_type": "tabular"}
  ],
  "schema_groups": [
    {
      "group_id": "group_tabular",
      "schema_type": "tabular",
      "documents": ["doc1"],
      "common_fields": ["product_name", "stock_quantity"],
      "can_combine": true
    }
  ],
  "execution_plan": {
    "sub_tasks": [
      {
        "task_id": "sum_inventory",
        "target_agent": "sql_agent",
        "aggregation_type": "sum"
      },
      {
        "task_id": "count_products",
        "target_agent": "sql_agent",
        "aggregation_type": "count"
      },
      {
        "task_id": "avg_inventory",
        "target_agent": "sql_agent",
        "aggregation_type": "avg"
      }
    ],
    "execution_strategy": "parallel"
  }
}
```

### Step 2: SQL Agent Execution

```json
{
  "agent_outputs": [
    {"task_id": "sum_inventory", "data": {"total": 2224.23}, "confidence": 0.95},
    {"task_id": "count_products", "data": {"count": 8}, "confidence": 0.95},
    {"task_id": "avg_inventory", "data": {"average": 278.03}, "confidence": 0.95}
  ]
}
```

### Step 3: Reviewer Validation

```json
{
  "decision": "approve",
  "quality_scores": {
    "sum_inventory": 0.95,
    "count_products": 0.95,
    "avg_inventory": 0.95
  },
  "reasoning": "All tasks completed. Cross-validation: 2224.23/8 = 278.03 ✓"
}
```

### Step 4: Final Response

```markdown
## Summary:
- **Total Inventory for All Products:** 2,224.23
- **Number of Products:** 8
- **Average Inventory per Product:** 278.03

**Data Sources:**
- ProductInventory.csv
```

---

## Complex Multi-Schema Example

### Input Query

```
"What is the total sales from all CSV reports, and summarize payment terms from vendor contracts"
```

### Planner Analysis

```json
{
  "schema_groups": [
    {
      "group_id": "group_tabular",
      "schema_type": "tabular",
      "documents": ["Sales2023.csv", "Sales2024.xlsx"],
      "can_combine": true
    },
    {
      "group_id": "group_document",
      "schema_type": "document",
      "documents": ["VendorContract_A.pdf", "VendorContract_B.pdf"],
      "can_combine": false
    }
  ],
  "execution_plan": {
    "sub_tasks": [
      {
        "task_id": "total_sales",
        "target_agent": "sql_agent",
        "document_ids": ["Sales2023.csv", "Sales2024.xlsx"]
      },
      {
        "task_id": "payment_terms",
        "target_agent": "vector_agent",
        "document_ids": ["VendorContract_A.pdf", "VendorContract_B.pdf"]
      }
    ],
    "execution_strategy": "parallel"
  }
}
```

### Final Response

```markdown
## Summary:

### Sales Analysis
- **Total Sales (CSV Reports):** $125,000.00
  - Sources: Sales2023.csv, Sales2024.xlsx

### Vendor Payment Terms
- **VendorContract_A.pdf:** Net 30, 2% discount for early payment
- **VendorContract_B.pdf:** Net 45, milestone-based payments

**Data Sources:**
- Sales2023.csv
- Sales2024.xlsx
- VendorContract_A.pdf
- VendorContract_B.pdf
```

---

## LangGraph 1.0+ Patterns Used

| Pattern | Usage | Reference |
|---------|-------|-----------|
| StateGraph with TypedDict | Main workflow state | [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview) |
| Annotated reducers | `agent_outputs: Annotated[List, operator.add]` | [State Management](https://sparkco.ai/blog/mastering-langgraph-state-management-in-2025) |
| create_react_agent | Individual agent creation | [ReAct Agent Guide](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/) |
| create_supervisor | Retrieval team orchestration | [langgraph-supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) |
| Command object | Control flow + state updates | [Command Blog](https://blog.langchain.com/command-a-new-tool-for-multi-agent-architectures-in-langgraph/) |
| Command.PARENT | Cross-subgraph navigation | [Multi-Agent Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/multi_agent.md) |
| InjectedState | Access state from tools | [Agents Reference](https://reference.langchain.com/python/langgraph/agents/) |
| Checkpointing | Durable execution | [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview) |

---

## Integration Points with Existing Code

| New Component | Existing Service | Integration Method |
|--------------|------------------|-------------------|
| Document routing tools | `unified_query_analyzer.py` | Call existing analyzer |
| Schema analysis tools | `schema_service.py` | Use SchemaService class |
| SQL generation | `LLMSQLGeneratorV2` | Bridge adapter |
| SQL execution | `SQLQueryExecutor` | Bridge adapter |
| Vector search | Qdrant client | Bridge adapter |
| Graph search | Neo4j client | Bridge adapter |
| Result formatting | `LLMResultFormatter` | Bridge adapter |

---

## Future Enhancements

1. **Persistent Checkpointing** - PostgreSQL/Redis backend for production
2. **Human-in-the-Loop** - Add approval steps for sensitive queries
3. **Agent Memory** - Long-term memory across sessions
4. **Parallel Subgraph Execution** - Run multiple schema groups simultaneously
5. **Cost Optimization** - Route simpler tasks to smaller models
6. **Observability** - LangSmith integration for debugging

---

## References

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Multi-Agent Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/multi_agent.md)
- [langgraph-supervisor Library](https://github.com/langchain-ai/langgraph-supervisor-py)
- [LangGraph Multi-Agent Guide 2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025)
- [Command Object for Multi-Agent](https://blog.langchain.com/command-a-new-tool-for-multi-agent-architectures-in-langgraph/)
- [ReAct Agent from Scratch](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
