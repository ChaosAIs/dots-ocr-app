"""
Document routing tools for the Planner Agent.

These tools handle:
- Finding relevant documents for a query
- Analyzing document schemas
- Grouping documents by schema compatibility

Uses shared document filtering logic from shared.document_filter
to ensure consistency with non-agent flow.
"""

import json
import logging
import os
import sys
from typing import Annotated, List, Dict, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agents.state.models import DocumentSource, SchemaGroup

logger = logging.getLogger(__name__)

# Ensure backend path is in sys.path for imports
_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_PATH)

# Import shared document filtering logic
from shared.document_filter import (
    extract_query_hints,
    topic_based_prefilter,
    calculate_relevance_score,
    apply_adaptive_threshold,
    extract_fields_from_header_data,
    ADAPTIVE_THRESHOLD_RATIO,
)

# Import database session factory
try:
    from db.database import get_db_session
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("[routing_tools] Database not available")


@tool
def get_relevant_documents(
    query: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Get relevant documents for the query.

    Uses shared document filtering logic (same as non-agent flow) to ensure
    consistent document selection. Applies topic pre-filtering before
    relevance scoring.

    Args:
        query: The user's natural language query

    Returns:
        JSON array of relevant documents with schema information
    """
    try:
        logger.info(f"[get_relevant_documents] Query: '{query[:50]}...'")

        if state is None:
            logger.error("[get_relevant_documents] State is None!")
            return json.dumps({"error": "State is None", "documents": []})

        # Get documents from state
        available_docs = state.get("available_documents", [])

        if not available_docs:
            logger.error("[get_relevant_documents] No documents in state")
            return json.dumps({
                "error": "No accessible documents found in state",
                "documents": []
            })

        logger.info(f"[get_relevant_documents] Found {len(available_docs)} documents in state")

        # Convert DocumentSource objects to dicts
        doc_dicts = []
        for doc in available_docs:
            if hasattr(doc, 'model_dump'):
                doc_dict = doc.model_dump()
            elif hasattr(doc, 'dict'):
                doc_dict = doc.dict()
            elif isinstance(doc, dict):
                doc_dict = doc
            else:
                doc_dict = {"document_id": str(doc)}
            doc_dicts.append(doc_dict)

        # Extract document IDs
        doc_ids = [d.get("document_id") for d in doc_dicts if d.get("document_id")]

        if not doc_ids:
            return json.dumps({"error": "No document IDs found", "documents": []})

        # Step 1: Extract query hints for topic pre-filtering
        hints = extract_query_hints(query)
        topics = hints.get('topics', [])
        doc_type_hints = hints.get('document_type_hints', [])

        logger.info(f"[get_relevant_documents] Query hints: topics={topics}, types={doc_type_hints}")

        # Step 2: Apply topic pre-filter if available
        filtered_doc_ids = doc_ids
        if (topics or doc_type_hints) and DB_AVAILABLE:
            prefiltered_ids, prefiltered_names = topic_based_prefilter(
                topics=topics,
                document_type_hints=doc_type_hints,
                accessible_document_ids=doc_ids,
                db_session_factory=get_db_session
            )
            if prefiltered_ids:
                filtered_doc_ids = prefiltered_ids
                logger.info(
                    f"[get_relevant_documents] Topic pre-filter: {len(doc_ids)} -> {len(filtered_doc_ids)} documents"
                )

        # Step 3: Enrich documents with schema info and calculate relevance
        enriched_docs = []

        if DB_AVAILABLE:
            try:
                from sqlalchemy import text

                with get_db_session() as db:
                    placeholders = ", ".join([f"'{doc_id}'" for doc_id in filtered_doc_ids])
                    result = db.execute(text(f"""
                        SELECT
                            dd.document_id::text,
                            d.original_filename as filename,
                            dd.schema_type,
                            COALESCE(dd.header_data, '{{}}') as header_data,
                            COALESCE(dd.extraction_metadata, '{{}}') as extraction_metadata
                        FROM documents_data dd
                        JOIN documents d ON dd.document_id = d.id
                        WHERE dd.document_id::text IN ({placeholders})
                    """))

                    rows = result.fetchall()
                    db_docs = {str(row[0]): row for row in rows}

                    for doc_id in filtered_doc_ids:
                        row = db_docs.get(doc_id)

                        if row:
                            header_data = row[3] if row[3] else {}
                            if isinstance(header_data, str):
                                try:
                                    header_data = json.loads(header_data)
                                except:
                                    header_data = {}

                            extraction_metadata = row[4] if row[4] else {}
                            if isinstance(extraction_metadata, str):
                                try:
                                    extraction_metadata = json.loads(extraction_metadata)
                                except:
                                    extraction_metadata = {}

                            # Extract fields
                            fields = extract_fields_from_header_data(header_data)

                            # Calculate relevance using shared function
                            relevance = calculate_relevance_score(
                                query=query,
                                schema_type=row[2] or "",
                                header_data=header_data,
                                fields=fields,
                                doc_metadata=extraction_metadata,
                                document_type_hints=doc_type_hints
                            )

                            # Essential metadata only
                            essential_metadata = {
                                "column_headers": header_data.get("column_headers", []),
                                "total_rows": header_data.get("total_rows"),
                                "total_columns": header_data.get("total_columns"),
                            }

                            enriched_docs.append({
                                "document_id": str(row[0]),
                                "filename": row[1] or f"doc_{doc_id[:8]}",
                                "schema_type": row[2] or "unknown",
                                "schema_definition": {"fields": fields, "metadata": essential_metadata},
                                "relevance_score": relevance,
                                "data_location": "header"
                            })
                        else:
                            enriched_docs.append({
                                "document_id": doc_id,
                                "filename": f"doc_{doc_id[:8]}",
                                "schema_type": "unknown",
                                "schema_definition": {"fields": []},
                                "relevance_score": 0.2,
                                "data_location": "header"
                            })

            except Exception as db_error:
                logger.warning(f"[get_relevant_documents] Database error: {db_error}")
                # Use basic docs without enrichment
                for doc_id in filtered_doc_ids:
                    enriched_docs.append({
                        "document_id": doc_id,
                        "filename": f"doc_{doc_id[:8]}",
                        "schema_type": "unknown",
                        "schema_definition": {"fields": []},
                        "relevance_score": 0.3,
                        "data_location": "header"
                    })

        # Step 4: Sort by relevance and apply adaptive threshold
        enriched_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        enriched_docs = apply_adaptive_threshold(enriched_docs)

        logger.info(f"[get_relevant_documents] Returning {len(enriched_docs)} documents")
        for doc in enriched_docs[:5]:
            logger.info(f"[get_relevant_documents]   - {doc['filename']}: score={doc['relevance_score']:.2f}")

        return json.dumps(enriched_docs, indent=2)

    except Exception as e:
        logger.error(f"[get_relevant_documents] Error: {e}")
        return json.dumps({"error": str(e), "documents": []})


@tool
def analyze_document_schemas(
    document_ids: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Analyze schemas of selected documents to identify commonalities and differences.

    Args:
        document_ids: JSON array of document IDs to analyze

    Returns:
        Schema analysis with common fields, groupability, and recommendations
    """
    try:
        doc_ids = json.loads(document_ids)

        available_docs = state.get("available_documents", [])
        if isinstance(available_docs, str):
            available_docs = json.loads(available_docs)

        # Filter to requested documents
        docs_to_analyze = []
        for doc in available_docs:
            doc_dict = doc if isinstance(doc, dict) else doc.dict()
            if doc_dict.get("document_id") in doc_ids:
                docs_to_analyze.append(doc_dict)

        if not docs_to_analyze:
            return json.dumps({
                "error": "No matching documents found",
                "schemas": [],
                "common_fields": [],
                "groupable": False
            })

        # Analyze schemas
        all_fields_by_doc = {}
        schema_types = set()

        for doc in docs_to_analyze:
            doc_id = doc["document_id"]
            schema_def = doc.get("schema_definition", {})
            fields = set(schema_def.get("fields", []))
            all_fields_by_doc[doc_id] = fields
            schema_types.add(doc.get("schema_type", "unknown"))

        # Find common fields
        if all_fields_by_doc:
            common_fields = set.intersection(*all_fields_by_doc.values())
        else:
            common_fields = set()

        # Determine if groupable
        groupable = (
            len(schema_types) == 1 and
            len(common_fields) >= 2 and
            "unknown" not in schema_types
        )

        # Generate recommendations
        recommended_groups = []
        if groupable:
            recommended_groups.append({
                "group_id": f"group_{list(schema_types)[0]}",
                "schema_type": list(schema_types)[0],
                "document_ids": doc_ids,
                "can_combine": True,
                "reason": f"All documents share schema type and {len(common_fields)} common fields"
            })
        else:
            by_type = {}
            for doc in docs_to_analyze:
                st = doc.get("schema_type", "unknown")
                if st not in by_type:
                    by_type[st] = []
                by_type[st].append(doc["document_id"])

            for st, ids in by_type.items():
                recommended_groups.append({
                    "group_id": f"group_{st}",
                    "schema_type": st,
                    "document_ids": ids,
                    "can_combine": len(ids) > 1,
                    "reason": f"Grouped by schema type: {st}"
                })

        result = {
            "schemas": [
                {
                    "document_id": doc["document_id"],
                    "schema_type": doc.get("schema_type"),
                    "fields": list(all_fields_by_doc.get(doc["document_id"], []))
                }
                for doc in docs_to_analyze
            ],
            "common_fields": list(common_fields),
            "groupable": groupable,
            "schema_types": list(schema_types),
            "recommended_groups": recommended_groups
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        logger.error(f"Error analyzing schemas: {e}")
        return json.dumps({"error": str(e)})


@tool
def group_documents_by_schema(
    documents: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Group documents by compatible schemas for efficient processing.

    Args:
        documents: JSON array of document sources with schema info

    Returns:
        JSON array of schema groups with minimal document info to avoid JSON size issues
    """
    try:
        docs = json.loads(documents)

        # Group by schema_type
        groups: Dict[str, Dict[str, Any]] = {}

        for doc in docs:
            schema_type = doc.get("schema_type", "unknown")

            if schema_type not in groups:
                groups[schema_type] = {
                    "group_id": f"group_{schema_type}_{len(groups)}",
                    "schema_type": schema_type,
                    "documents": [],
                    "all_fields": []
                }

            # Only include essential document fields to keep JSON small
            minimal_doc = {
                "document_id": doc.get("document_id"),
                "filename": doc.get("filename"),
                "schema_type": doc.get("schema_type")
            }
            groups[schema_type]["documents"].append(minimal_doc)

            schema_def = doc.get("schema_definition", {})
            fields = set(schema_def.get("fields", []))
            groups[schema_type]["all_fields"].append(fields)

        # Process each group
        result = []
        for schema_type, group in groups.items():
            all_field_sets = group["all_fields"]

            if all_field_sets:
                common_fields = set.intersection(*all_field_sets) if len(all_field_sets) > 1 else all_field_sets[0]
            else:
                common_fields = set()

            can_combine = (
                len(group["documents"]) > 1 and
                len(common_fields) > 0 and
                schema_type != "unknown"
            )

            # Only include top 10 common fields to keep JSON small
            common_fields_list = list(common_fields)[:10]

            result.append({
                "group_id": group["group_id"],
                "schema_type": schema_type,
                "common_fields": common_fields_list,
                "document_ids": [d["document_id"] for d in group["documents"]],
                "document_names": [d["filename"] for d in group["documents"]],
                "can_combine": can_combine,
                "document_count": len(group["documents"])
            })

        logger.info(f"Created {len(result)} schema groups from {len(docs)} documents")
        return json.dumps(result)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        logger.error(f"Error grouping documents: {e}")
        return json.dumps({"error": str(e)})
