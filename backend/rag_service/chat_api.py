"""
WebSocket API for real-time chat with the RAG agent.
Provides streaming responses for better UX.

Enhanced with intent-based routing:
- DOCUMENT_SEARCH → RAG Service (vector search)
- DATA_ANALYTICS → SQL Query Executor (documents_data table)
- HYBRID → Both services combined
- GENERAL → Direct response
"""

import asyncio
import json
import logging
import os
import random
from typing import List, Dict, Optional
from uuid import UUID
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .rag_agent import stream_agent_response, create_agent_executor
from .vectorstore import get_collection_info
from .indexer import get_indexed_count
from .timing_metrics import TimingMetrics, set_current_metrics, clear_current_metrics
from db.database import get_db_session
from chat_service.conversation_manager import ConversationManager
from chat_service.context_analyzer import ContextAnalyzer
from chat_service.chat_orchestrator import ChatOrchestrator, OrchestratorResult
from analytics_service.intent_classifier import QueryIntent
from analytics_service.query_cache_service import (
    get_query_cache_service,
    QueryCacheService,
    CacheSearchResult
)
from analytics_service.unified_query_analyzer import (
    get_unified_query_analyzer,
    QueryAnalysisResult,
    QueryIntent,
    DissatisfactionType
)
from analytics_service.intent_classifier import IntentClassifier, IntentClassification
from db.user_document_repository import UserDocumentRepository
from db.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

# Load configuration from environment
CHAT_ENABLE_CONTEXT_ANALYSIS = os.getenv("CHAT_ENABLE_CONTEXT_ANALYSIS", "true").lower() == "true"
CHAT_ENABLE_METADATA_TRACKING = os.getenv("CHAT_ENABLE_METADATA_TRACKING", "true").lower() == "true"
CHAT_MAX_ENTITIES_PER_TYPE = int(os.getenv("CHAT_MAX_ENTITIES_PER_TYPE", "10"))
CHAT_MAX_TOPICS = int(os.getenv("CHAT_MAX_TOPICS", "5"))

# Analytics/Orchestrator configuration
CHAT_ENABLE_INTENT_ROUTING = os.getenv("CHAT_ENABLE_INTENT_ROUTING", "true").lower() == "true"
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"

# Query Cache configuration
QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"

# Unified Preprocessing - combines context analysis, cache analysis, and intent classification into ONE LLM call
UNIFIED_PREPROCESSING_ENABLED = os.getenv("UNIFIED_PREPROCESSING_ENABLED", "true").lower() == "true"

logger.info(f"Chat Context Configuration: ENABLE_CONTEXT_ANALYSIS={CHAT_ENABLE_CONTEXT_ANALYSIS}, ENABLE_METADATA_TRACKING={CHAT_ENABLE_METADATA_TRACKING}")
logger.info(f"Chat Intent Routing: ENABLE_INTENT_ROUTING={CHAT_ENABLE_INTENT_ROUTING}, ANALYTICS_ENABLED={ANALYTICS_ENABLED}")
logger.info(f"Query Cache: QUERY_CACHE_ENABLED={QUERY_CACHE_ENABLED}")
logger.info(f"Unified Preprocessing: UNIFIED_PREPROCESSING_ENABLED={UNIFIED_PREPROCESSING_ENABLED}")

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    sources: List[str] = []


class StatusResponse(BaseModel):
    """Status response model."""

    status: str
    indexed_files: int
    collection_info: dict


class ChatConfigResponse(BaseModel):
    """Chat configuration response model."""

    iterative_reasoning_enabled: bool


@router.get("/config", response_model=ChatConfigResponse)
async def get_chat_config():
    """Get chat configuration settings for the frontend."""
    try:
        # Read ITERATIVE_REASONING_ENABLED from environment
        # This controls the default checkbox state for "Enable Graph Knowledge Reasoning"
        iterative_reasoning_enabled = os.getenv("ITERATIVE_REASONING_ENABLED", "true").lower() == "true"

        return ChatConfigResponse(
            iterative_reasoning_enabled=iterative_reasoning_enabled
        )
    except Exception as e:
        logger.error(f"Error getting chat config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def get_chat_status():
    """Get the status of the RAG service."""
    try:
        collection_info = get_collection_info()
        indexed_count = get_indexed_count()

        return StatusResponse(
            status="ready",
            indexed_files=indexed_count,
            collection_info=collection_info,
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a chat message and get a response (non-streaming).
    Use WebSocket for streaming responses.
    """
    try:
        # Convert history to dict format
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        # Collect full response
        full_response = ""
        async for chunk in stream_agent_response(request.message, history):
            full_response += chunk

        return ChatResponse(response=full_response, sources=[])

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming chat responses with session persistence.

    Protocol:
    - Client sends JSON: {"message": "...", "user_id": "..."}
    - Server streams JSON: {"type": "token", "content": "..."} for each token
    - Server sends JSON: {"type": "end"} when complete
    - Server sends JSON: {"type": "error", "message": "..."} on error

    All messages are persisted to the database under the given session_id.

    THREAD SAFETY NOTE: Database sessions are now created per-message instead of
    per-connection to avoid holding connections during long-running LLM operations.
    This prevents connection pool exhaustion when multiple chat sessions are active.
    """
    await websocket.accept()
    logger.debug(f"WebSocket connection established for chat session: {session_id}")

    # Verify session exists using a short-lived DB session
    try:
        with get_db_session() as db:
            conv_manager = ConversationManager(db)
            session = conv_manager.get_session(session_id)
            if not session:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Session {session_id} not found",
                })
                await websocket.close()
                return
    except Exception as e:
        logger.error(f"Error verifying session: {e}")
        await websocket.send_json({
            "type": "error",
            "message": "Failed to verify session",
        })
        await websocket.close()
        return

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                request = json.loads(data)

                # Handle ping/pong for keepalive
                if request.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                message = request.get("message", "")
                user_id = request.get("user_id")
                is_retry = request.get("is_retry", False)  # Flag for retry action
                workspace_ids = request.get("workspace_ids", [])  # Optional workspace filter
                document_ids = request.get("document_ids", [])  # Optional document filter
                iterative_reasoning_enabled = request.get("iterative_reasoning_enabled", None)  # Optional toggle from UI checkbox
                agent_mode_enabled = request.get("agent_mode_enabled", False)  # Optional toggle for new agentic workflow

                # Debug logging for agent mode
                logger.info("=" * 80)
                logger.info(f"[ChatAPI] ========== REQUEST RECEIVED ==========")
                logger.info(f"[ChatAPI] Message: {message[:100]}...")
                logger.info(f"[ChatAPI] Agent Mode Enabled: {agent_mode_enabled} (type: {type(agent_mode_enabled).__name__})")
                logger.info(f"[ChatAPI] Iterative Reasoning Enabled: {iterative_reasoning_enabled}")
                logger.info(f"[ChatAPI] Document IDs: {len(document_ids)} documents")
                logger.info(f"[ChatAPI] Workspace IDs: {workspace_ids}")
                logger.info("=" * 80)

                if not message:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty message received",
                    })
                    continue

                logger.info(f"Received chat message for session {session_id}: {message[:100]}...")

                # Initialize timing metrics for this query
                import uuid as uuid_module
                timing_metrics = TimingMetrics(
                    query_id=f"{session_id[:8]}-{str(uuid_module.uuid4())[:8]}",
                    query_preview=message[:100]
                )
                set_current_metrics(timing_metrics)

                # Save user message to database (short-lived session)
                try:
                    with get_db_session() as db:
                        conv_manager = ConversationManager(db)
                        conv_manager.add_message(
                            session_id=UUID(session_id),
                            role="user",
                            content=message
                        )
                except Exception as e:
                    logger.error(f"Error saving user message: {e}")

                # Get conversation history from database (short-lived session)
                history = []
                context_info = {}
                session_context = {}
                enhanced_message = message
                document_context_changed = False  # Track if document/workspace selection changed
                # Normalize current selections to sorted lists for comparison
                curr_workspace_ids = sorted(workspace_ids) if workspace_ids else []
                curr_document_ids = sorted(document_ids) if document_ids else []

                try:
                    with get_db_session() as db:
                        conv_manager = ConversationManager(db)
                        context = conv_manager.get_conversation_context(UUID(session_id))
                        # Convert to format expected by RAG agent
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in context.get("short_term_messages", [])
                        ]

                        # Get session metadata for context
                        session_context = context.get("session_metadata", {})

                    # Check if document/workspace selection has changed from previous message in THIS session
                    # This is important to force a new document search when user changes filters mid-conversation
                    # Note: For NEW sessions, the initial selection comes from user preferences (users table)
                    #       and we don't need to force a new search for the first message
                    prev_workspace_ids = session_context.get("_prev_workspace_ids", [])
                    prev_document_ids = session_context.get("_prev_document_ids", [])

                    # Only detect change if there was a previous selection stored in this session
                    # This means user has sent at least one message before and is now changing the filter
                    if prev_workspace_ids or prev_document_ids:
                        if curr_workspace_ids != prev_workspace_ids or curr_document_ids != prev_document_ids:
                            document_context_changed = True
                            logger.info(f"[Document Context] Selection changed mid-conversation!")
                            logger.info(f"[Document Context]   Previous: workspaces={prev_workspace_ids[:3]}{'...' if len(prev_workspace_ids) > 3 else ''}, docs={prev_document_ids[:3]}{'...' if len(prev_document_ids) > 3 else ''}")
                            logger.info(f"[Document Context]   Current: workspaces={curr_workspace_ids[:3]}{'...' if len(curr_workspace_ids) > 3 else ''}, docs={curr_document_ids[:3]}{'...' if len(curr_document_ids) > 3 else ''}")

                except Exception as e:
                    logger.error(f"Error loading conversation history: {e}")

                # ====== UNIFIED QUERY ANALYSIS (SINGLE LLM CALL) ======
                # This performs ALL query analysis in ONE LLM call:
                # 1. Context Resolution (pronouns, references)
                # 2. Cache Analysis (dissatisfaction, bypass, cacheable)
                # 3. Intent Classification (routing decision)
                # 4. Query Enhancement (entities, topics, document types for routing)
                # NOTE: Run in thread to avoid blocking async event loop
                query_analysis: Optional[QueryAnalysisResult] = None

                if UNIFIED_PREPROCESSING_ENABLED and CHAT_ENABLE_CONTEXT_ANALYSIS:
                    try:
                        # Get previous response for dissatisfaction detection
                        prev_response = None
                        if history:
                            for msg in reversed(history):
                                if msg.get("role") == "assistant":
                                    prev_response = msg.get("content", "")[:500]
                                    break

                        # Get available schemas for intent classification (short-lived session)
                        available_schemas = None
                        try:
                            with get_db_session() as db:
                                from chat_service.chat_orchestrator import ChatOrchestrator
                                temp_orchestrator = ChatOrchestrator(db)
                                available_schemas = temp_orchestrator._get_available_schemas()
                        except Exception:
                            pass

                        # Single unified LLM call for ALL analysis
                        # Run in thread to avoid blocking async event loop
                        with timing_metrics.measure("unified_query_analysis"):
                            analyzer = get_unified_query_analyzer()

                            def run_analysis():
                                return analyzer.analyze(
                                    message=message,
                                    chat_history=history,
                                    previous_response=prev_response,
                                    available_schemas=available_schemas
                                )

                            query_analysis = await asyncio.to_thread(run_analysis)

                        # Record the internal processing time
                        timing_metrics.record("unified_query_analysis_llm", query_analysis.processing_time_ms)

                        # Extract results from unified analysis
                        enhanced_message = query_analysis.resolved_message or message
                        context_info = {
                            "entities": query_analysis.entities,
                            "topics": query_analysis.topics,
                            "document_type_hints": query_analysis.document_type_hints,
                            "enhanced_query": query_analysis.enhanced_query,
                            "has_pronouns": enhanced_message != message,
                        }

                        # Log summary
                        logger.info(f"[UnifiedQueryAnalysis] Message: '{message[:60]}...'")
                        logger.info(f"[UnifiedQueryAnalysis] Resolved: '{enhanced_message[:60]}...'")
                        logger.info(f"[UnifiedQueryAnalysis] Cache key: '{query_analysis.cache_key[:60]}...'")
                        logger.info(f"[UnifiedQueryAnalysis] Intent: {query_analysis.intent.value} (confidence: {query_analysis.intent_confidence:.2f})")
                        logger.info(f"[UnifiedQueryAnalysis] Cacheable: {query_analysis.is_cacheable}, Bypass: {query_analysis.bypass_cache}")
                        logger.info(f"[UnifiedQueryAnalysis] Topics: {query_analysis.topics}")
                        logger.info(f"[UnifiedQueryAnalysis] Entities: {query_analysis.entities}")
                        logger.info(f"[UnifiedQueryAnalysis] Doc types: {query_analysis.document_type_hints}")
                        logger.info(f"[UnifiedQueryAnalysis] Processing time: {query_analysis.processing_time_ms:.2f}ms")

                    except Exception as e:
                        logger.error(f"[UnifiedQueryAnalysis] Error: {e}", exc_info=True)
                        query_analysis = None
                        enhanced_message = message

                elif CHAT_ENABLE_CONTEXT_ANALYSIS:
                    # Fallback to individual context analysis if unified preprocessing disabled
                    logger.info(f"[Context Analysis] Analyzing message: '{message[:100]}...' with {len(history)} history messages")
                    try:
                        analyzer = ContextAnalyzer()

                        # Run in thread to avoid blocking async event loop
                        def run_context_analysis():
                            return analyzer.analyze_message(message, history)

                        analysis = await asyncio.to_thread(run_context_analysis)

                        # Use resolved message if pronouns were detected
                        enhanced_message = analysis.get("resolved_message", message)

                        # Store context info for metadata update
                        context_info = {
                            "entities": analysis.get("entities", {}),
                            "topics": analysis.get("topics", []),
                            "has_pronouns": analysis.get("has_pronouns", False),
                            "detected_pronouns": analysis.get("detected_pronouns", [])
                        }

                        if analysis.get("has_pronouns"):
                            logger.info(f"[Context Analysis] Pronouns detected: {context_info['detected_pronouns']}")
                            logger.info(f"[Context Analysis] Original: '{message}' → Enhanced: '{enhanced_message}'")
                        else:
                            logger.info(f"[Context Analysis] No pronouns detected in message")

                        logger.info(f"[Context Analysis] Extracted entities: {context_info['entities']}")
                        logger.info(f"[Context Analysis] Extracted topics: {context_info['topics']}")

                    except Exception as e:
                        logger.error(f"[Context Analysis] Error analyzing context: {e}", exc_info=True)
                        enhanced_message = message
                else:
                    logger.warning(f"[Context Analysis] Context analysis is DISABLED")

                # Stream response and accumulate content
                full_response = ""
                streaming_error = None

                try:
                    # Track last progress message time for minimum delay
                    import time
                    last_progress_time = [0]  # Use list to allow modification in nested function
                    last_progress_message = [""]  # Track last message for stage-specific delays

                    # Stage-specific minimum display times (in seconds)
                    # Set to 0 for maximum performance - no artificial delays
                    STAGE_MIN_DISPLAY_TIMES = {
                        # All delays disabled for performance
                        "Understanding your question...": 0,
                        "Analyzing key topics in your question...": 0,
                        "Finding relevant documents...": 0,
                        "Analyzing data structure...": 0,
                        "Building database query...": 0,
                        "Running database query...": 0,
                        "Found": 0,
                        "Writing your report...": 0,
                        "default": 0
                    }

                    def get_min_display_time(message: str) -> float:
                        """Get minimum display time for a progress message."""
                        # Check for exact match first
                        if message in STAGE_MIN_DISPLAY_TIMES:
                            return STAGE_MIN_DISPLAY_TIMES[message]
                        # Check for prefix match (e.g., "Found 43 records...")
                        for prefix, delay in STAGE_MIN_DISPLAY_TIMES.items():
                            if message.startswith(prefix):
                                return delay
                        return STAGE_MIN_DISPLAY_TIMES["default"]

                    # Create progress callback for WebSocket updates
                    async def progress_callback(message: str, percent: int = None):
                        """Send progress updates to frontend via WebSocket with stage-specific delays."""
                        try:
                            # Get minimum display time for the PREVIOUS message
                            if last_progress_time[0] > 0:
                                min_display = get_min_display_time(last_progress_message[0])
                                current_time = time.time()
                                elapsed = current_time - last_progress_time[0]
                                if elapsed < min_display:
                                    await asyncio.sleep(min_display - elapsed)

                            payload = {"type": "progress", "message": message}
                            if percent is not None:
                                payload["percent"] = percent
                            await websocket.send_json(payload)
                            last_progress_time[0] = time.time()
                            last_progress_message[0] = message
                        except Exception as e:
                            logger.error(f"Error sending progress update: {e}")

                    # Get accessible document IDs for the current user
                    # This ensures chat only searches documents the user has access to
                    # NOTE: Uses short-lived DB session to avoid holding connection during streaming
                    accessible_doc_ids = None
                    access_control_start = time.time()
                    logger.info(f"[Access Control] Checking document access for user_id: {user_id}")
                    if user_id:
                        try:
                            with get_db_session() as db:
                                user_doc_repo = UserDocumentRepository(db)

                                # OPTIMIZATION: When specific document_ids are provided by frontend,
                                # only validate those IDs instead of fetching ALL accessible documents.
                                # This is much more efficient when user has selected specific documents.
                                if document_ids and len(document_ids) > 0:
                                    logger.info(f"[Access Control] Validating {len(document_ids)} selected document(s) for user access")
                                    try:
                                        # Convert document_ids to UUID list
                                        selected_doc_ids = [UUID(doc_id) for doc_id in document_ids]
                                        # Only query for the specific documents user selected
                                        accessible_doc_ids = user_doc_repo.filter_accessible_document_ids(
                                            user_id=UUID(user_id),
                                            document_ids=selected_doc_ids,
                                            permission='read'
                                        )
                                        # Check for stale/inaccessible document IDs
                                        selected_set = set(selected_doc_ids)
                                        stale_ids = selected_set - accessible_doc_ids
                                        if stale_ids:
                                            logger.warning(f"[Access Control] {len(stale_ids)} selected documents are not accessible: {[str(sid) for sid in list(stale_ids)[:3]]}...")
                                        logger.info(f"[Access Control] User has access to {len(accessible_doc_ids)} of {len(document_ids)} selected documents")
                                    except Exception as doc_error:
                                        logger.warning(f"[Access Control] Failed to validate selected documents: {doc_error}")
                                        # Fall back to full access check if validation fails
                                        accessible_doc_ids = user_doc_repo.get_user_accessible_document_ids(
                                            user_id=UUID(user_id),
                                            permission='read'
                                        )
                                        if document_ids:
                                            selected_doc_ids = set(UUID(doc_id) for doc_id in document_ids)
                                            accessible_doc_ids = accessible_doc_ids.intersection(selected_doc_ids)
                                # When workspace_ids are provided (but no document_ids), get workspace docs directly
                                elif workspace_ids and len(workspace_ids) > 0:
                                    logger.info(f"[Access Control] Filtering by {len(workspace_ids)} workspace(s): {workspace_ids[:3]}...")
                                    try:
                                        doc_repo = DocumentRepository(db)
                                        workspace_doc_ids = doc_repo.get_document_ids_by_workspaces(
                                            [UUID(ws_id) for ws_id in workspace_ids]
                                        )
                                        if workspace_doc_ids:
                                            # Validate that user has access to these workspace documents
                                            accessible_doc_ids = user_doc_repo.filter_accessible_document_ids(
                                                user_id=UUID(user_id),
                                                document_ids=list(workspace_doc_ids),
                                                permission='read'
                                            )
                                            logger.info(f"[Access Control] User has access to {len(accessible_doc_ids)} of {len(workspace_doc_ids)} workspace documents")
                                        else:
                                            logger.info(f"[Access Control] Selected workspaces have no documents")
                                            accessible_doc_ids = set()
                                    except Exception as ws_error:
                                        logger.warning(f"[Access Control] Failed to filter by workspaces: {ws_error}")
                                        # Fall back to getting all accessible documents
                                        accessible_doc_ids = user_doc_repo.get_user_accessible_document_ids(
                                            user_id=UUID(user_id),
                                            permission='read'
                                        )
                                else:
                                    # No specific filter - get all accessible documents (original behavior)
                                    logger.info(f"[Access Control] No filter applied - fetching all accessible documents")
                                    accessible_doc_ids = user_doc_repo.get_user_accessible_document_ids(
                                        user_id=UUID(user_id),
                                        permission='read'
                                    )
                                    logger.info(f"[Access Control] User has access to {len(accessible_doc_ids) if accessible_doc_ids else 0} documents")

                                # IMPORTANT: If the set is empty, it means user has NO access to any documents
                                # This should block all document searches
                                if accessible_doc_ids is not None and len(accessible_doc_ids) == 0:
                                    logger.warning(f"[Access Control] User {user_id} has NO accessible documents with current filter - chat will return no documents")
                        except Exception as e:
                            import traceback
                            logger.warning(f"[Access Control] Failed to get accessible documents for user {user_id}: {e}")
                            logger.warning(f"[Access Control] Exception traceback: {traceback.format_exc()}")
                            # If we can't get access list, use empty set to block all access (secure by default)
                            accessible_doc_ids = set()
                    else:
                        logger.warning(f"[Access Control] No user_id provided - cannot enforce access control")
                    timing_metrics.record("access_control", (time.time() - access_control_start) * 1000)

                    # Use enhanced message with resolved pronouns and pass session context
                    chunk_count = 0
                    logger.info(f"[Streaming] Starting to stream response for message: '{enhanced_message[:100]}...' (is_retry={is_retry})")

                    # Convert accessible_doc_ids set to list for orchestrator
                    accessible_doc_ids_list = list(accessible_doc_ids) if accessible_doc_ids else []

                    # ====== QUERY CACHE INTEGRATION ======
                    cache_hit = False
                    cache_result = None
                    cache_key_question = None
                    source_document_ids_for_cache = []
                    # Cache variables from unified query analysis (set defaults for scope)
                    cache_bypass = query_analysis.bypass_cache if query_analysis else False
                    cache_is_cacheable = query_analysis.is_cacheable if query_analysis else True
                    cache_key = query_analysis.cache_key if query_analysis else ""
                    cache_analysis_method = query_analysis.analysis_method if query_analysis else "none"

                    # Fallback: Direct keyword check for cache bypass using centralized patterns
                    # (in case LLM analysis misses it)
                    if not cache_bypass:
                        import re
                        from analytics_service.unified_query_analyzer import CACHE_CONTROL_PATTERNS
                        message_lower = enhanced_message.lower()
                        keyword_bypass = any(re.search(pattern, message_lower, re.IGNORECASE) for pattern in CACHE_CONTROL_PATTERNS)
                        if keyword_bypass:
                            logger.info("[QueryCache] Keyword bypass detected in query (LLM missed it)")
                            cache_bypass = True

                    # Skip cache when agent mode is enabled OR cache bypass keywords detected
                    use_cache = QUERY_CACHE_ENABLED and accessible_doc_ids_list and not agent_mode_enabled and not cache_bypass
                    if agent_mode_enabled:
                        logger.info("[QueryCache] Cache SKIPPED - Agent Mode is enabled")
                    elif cache_bypass:
                        logger.info("[QueryCache] Cache SKIPPED - User requested fresh/latest data")

                    if use_cache:
                        cache_start = time.time()
                        try:
                            logger.info("=" * 80)
                            logger.info("[QueryCache] ========== QUERY CACHE PROCESS START ==========")
                            logger.info("=" * 80)
                            logger.info(f"[QueryCache] Query: {enhanced_message[:100]}...")
                            logger.info(f"[QueryCache] Workspace IDs: {curr_workspace_ids[:3]}{'...' if len(curr_workspace_ids) > 3 else ''}")
                            logger.info(f"[QueryCache] Accessible documents: {len(accessible_doc_ids_list)}")
                            logger.info("-" * 80)

                            cache_service = get_query_cache_service()

                            # Use cache analysis from unified query analysis (NO additional LLM call needed)
                            # The new QueryAnalysisResult contains all cache-related fields directly
                            if query_analysis is not None:
                                logger.info("[QueryCache] STEP 1: Using cache analysis from unified query analysis (NO additional LLM call)")
                                cache_bypass = query_analysis.bypass_cache
                                cache_is_cacheable = query_analysis.is_cacheable
                                cache_key = query_analysis.cache_key
                                cache_is_dissatisfied = query_analysis.is_dissatisfied
                                cache_dissatisfaction_type = query_analysis.dissatisfaction_type
                                cache_invalidate_previous = query_analysis.invalidate_previous
                                cache_analysis_method = query_analysis.analysis_method
                            else:
                                # Fallback: Run separate cache analysis (LLM call) - rarely needed
                                logger.info("[QueryCache] STEP 1: Running separate cache analysis (LLM call)...")
                                await progress_callback("Analyzing your question...")
                                # Use heuristic fallback since we don't have unified analysis
                                from analytics_service.unified_query_analyzer import strip_cache_control_words
                                cache_bypass = False
                                cache_is_cacheable = True
                                cache_key = strip_cache_control_words(enhanced_message)
                                cache_is_dissatisfied = False
                                cache_dissatisfaction_type = DissatisfactionType.NONE
                                cache_invalidate_previous = False
                                cache_analysis_method = "fallback"

                            logger.info("-" * 80)
                            logger.info("[QueryCache] CACHE ANALYSIS RESULT:")
                            logger.info(f"[QueryCache]   • Method: {cache_analysis_method}")
                            logger.info(f"[QueryCache]   • Dissatisfied: {cache_is_dissatisfied} ({cache_dissatisfaction_type.value})")
                            logger.info(f"[QueryCache]   • Bypass cache: {cache_bypass}")
                            logger.info(f"[QueryCache]   • Is cacheable: {cache_is_cacheable}")
                            logger.info(f"[QueryCache]   • Cache key: {cache_key[:80]}...")
                            logger.info("-" * 80)

                            # Determine cache workspace ID from current workspace selection
                            # Use first workspace ID or "default" if none selected
                            cache_workspace_id = str(curr_workspace_ids[0]) if curr_workspace_ids else "default"
                            logger.info(f"[QueryCache] Using workspace ID for cache: {cache_workspace_id}")

                            # Step 2: Check for bypass (user dissatisfaction or cache control words)
                            if cache_bypass:
                                logger.info("[QueryCache] STEP 2: BYPASS CACHE (user request or dissatisfaction)")
                                logger.info(f"[QueryCache]   • Dissatisfaction type: {cache_dissatisfaction_type.value}")
                                logger.info(f"[QueryCache]   • Should invalidate previous: {cache_invalidate_previous}")
                                # Optionally invalidate previous cache entry
                                if cache_invalidate_previous and history:
                                    # Find the previous question to invalidate
                                    for msg in reversed(history):
                                        if msg.get("role") == "user":
                                            prev_question = msg.get("content", "")
                                            if prev_question:
                                                cache_service.invalidate_question(prev_question, cache_workspace_id)
                                                logger.info(f"[QueryCache]   • Invalidated previous cache entry: {prev_question[:50]}...")
                                            break
                                logger.info("[QueryCache] → Proceeding to fresh query (cache bypassed)")
                            elif cache_is_cacheable:
                                # Step 3: Try cache lookup
                                logger.info("[QueryCache] STEP 2: CACHE LOOKUP (question is cacheable)")
                                cache_key_question = cache_key or enhanced_message
                                accessible_doc_ids_str = [str(doc_id) for doc_id in accessible_doc_ids_list]

                                logger.info(f"[QueryCache]   • Cache key question: {cache_key_question[:80]}...")
                                logger.info(f"[QueryCache]   • User accessible docs: {len(accessible_doc_ids_str)}")

                                cache_result = cache_service.lookup(
                                    question=cache_key_question,
                                    workspace_id=cache_workspace_id,
                                    user_accessible_doc_ids=accessible_doc_ids_str
                                )

                                logger.info("-" * 80)
                                logger.info("[QueryCache] CACHE LOOKUP RESULT:")
                                logger.info(f"[QueryCache]   • Cache hit: {cache_result.cache_hit}")
                                logger.info(f"[QueryCache]   • Similarity score: {cache_result.similarity_score:.3f}")
                                logger.info(f"[QueryCache]   • Candidates checked: {cache_result.candidates_checked}")
                                logger.info(f"[QueryCache]   • Search time: {cache_result.search_time_ms:.2f}ms")
                                logger.info(f"[QueryCache]   • Permission granted: {cache_result.permission_granted}")

                                if cache_result.cache_hit:
                                    # CACHE HIT! Stream the cached answer
                                    cache_hit = True
                                    logger.info("-" * 80)
                                    logger.info("[QueryCache] ★★★ CACHE HIT! ★★★")
                                    logger.info(f"[QueryCache]   • Entry ID: {cache_result.entry.id}")
                                    logger.info(f"[QueryCache]   • Cached question: {cache_result.entry.question[:80]}...")
                                    logger.info(f"[QueryCache]   • Answer length: {len(cache_result.entry.answer)} chars")
                                    logger.info(f"[QueryCache]   • Source docs: {cache_result.entry.source_document_ids[:3]}...")
                                    logger.info(f"[QueryCache]   • Confidence: {cache_result.entry.confidence_score:.2f}")
                                    logger.info(f"[QueryCache]   • Hit count: {cache_result.entry.hit_count}")
                                    logger.info("[QueryCache] → Streaming cached response to user...")

                                    # Stream the cached response with realistic LLM-like typing effect
                                    await progress_callback("Found a matching answer...")
                                    cached_answer = cache_result.entry.answer

                                    # Stream cached response in small chunks to simulate LLM streaming
                                    # Use variable chunk sizes (2-8 chars) for more natural effect
                                    pos = 0
                                    answer_len = len(cached_answer)

                                    while pos < answer_len:
                                        # Variable chunk size: 2-8 characters, but respect word boundaries when possible
                                        base_chunk_size = random.randint(2, 8)
                                        end_pos = min(pos + base_chunk_size, answer_len)

                                        # Try to extend to word boundary if close
                                        if end_pos < answer_len and cached_answer[end_pos] != ' ':
                                            # Look for next space within 5 chars
                                            for lookahead in range(1, 6):
                                                if end_pos + lookahead < answer_len and cached_answer[end_pos + lookahead] == ' ':
                                                    end_pos = end_pos + lookahead + 1
                                                    break

                                        chunk = cached_answer[pos:end_pos]
                                        if chunk:
                                            chunk_count += 1
                                            full_response += chunk
                                            await websocket.send_json({
                                                "type": "token",
                                                "content": chunk,
                                            })
                                            # Variable delay: 5-20ms for natural typing feel
                                            await asyncio.sleep(random.uniform(0.005, 0.020))

                                        pos = end_pos

                                    # Add subtle cache indicator at the end
                                    cache_indicator = "\n\n---\n_⚡ Retrieved from cache_"
                                    full_response += cache_indicator
                                    await websocket.send_json({
                                        "type": "token",
                                        "content": cache_indicator,
                                    })

                                    logger.info(f"[QueryCache] Streamed cached response: {len(full_response)} chars, {chunk_count} chunks")
                                else:
                                    logger.info("[QueryCache] CACHE MISS - no matching entry found")
                                    logger.info("[QueryCache] → Proceeding to fresh query...")
                            else:
                                logger.info("[QueryCache] STEP 2: SKIP CACHE (question not cacheable)")
                                logger.info("[QueryCache] → Proceeding to fresh query (no caching)...")

                            logger.info("=" * 80)
                            logger.info(f"[QueryCache] ========== QUERY CACHE PROCESS END (hit={cache_hit}) ==========")
                            logger.info("=" * 80)

                        except Exception as cache_error:
                            logger.error(f"[QueryCache] ========== CACHE ERROR ==========")
                            logger.error(f"[QueryCache] Error during cache operations: {cache_error}", exc_info=True)
                            logger.error(f"[QueryCache] → Continuing without cache...")
                            # Continue without cache on error
                        finally:
                            timing_metrics.record("cache_lookup", (time.time() - cache_start) * 1000)

                    # Skip full pipeline if cache hit
                    if cache_hit:
                        timing_metrics.record("cache_hit_response", 0)  # Mark cache hit
                        # Save cached response to conversation (short-lived session)
                        try:
                            with get_db_session() as db:
                                conv_manager = ConversationManager(db)
                                conv_manager.add_message(
                                    session_id=session_id,
                                    role="assistant",
                                    content=full_response,
                                    metadata={"source": "cache", "cache_entry_id": cache_result.entry.id if cache_result else None}
                                )
                        except Exception as e:
                            logger.error(f"Error saving cache hit response: {e}")
                        # Log timing summary for cache hit
                        timing_metrics.log_summary()
                        clear_current_metrics()
                        # Send end message and continue to next message
                        await websocket.send_json({"type": "end"})
                        continue

                    # ====== AGENTIC WORKFLOW MODE ======
                    # When agent_mode_enabled is True, use the new multi-agent agentic workflow
                    # instead of the standard intent-based routing
                    logger.info("=" * 80)
                    logger.info("[AgenticMode] ========== CHECKING AGENTIC WORKFLOW CONDITIONS ==========")
                    logger.info(f"[AgenticMode] agent_mode_enabled = {agent_mode_enabled} (type: {type(agent_mode_enabled).__name__})")
                    logger.info(f"[AgenticMode] accessible_doc_ids_list = {len(accessible_doc_ids_list) if accessible_doc_ids_list else 0} documents")
                    logger.info(f"[AgenticMode] Condition result: agent_mode_enabled AND accessible_doc_ids_list = {bool(agent_mode_enabled and accessible_doc_ids_list)}")
                    logger.info("=" * 80)

                    if agent_mode_enabled and accessible_doc_ids_list:
                        logger.info("=" * 80)
                        logger.info("[AgenticMode] ========== AGENTIC WORKFLOW ACTIVATED ==========")
                        logger.info(f"[AgenticMode] Query: {enhanced_message[:100]}...")
                        logger.info(f"[AgenticMode] Workspace IDs: {curr_workspace_ids[:3]}{'...' if len(curr_workspace_ids) > 3 else ''}")
                        logger.info(f"[AgenticMode] Accessible documents: {len(accessible_doc_ids_list)}")
                        logger.info("=" * 80)

                        try:
                            # Import the agents module - try multiple import paths for flexibility
                            execute_analytics_query_async = None
                            import_error = None

                            # Try absolute import first (when running with proper PYTHONPATH)
                            try:
                                from backend.agents import execute_analytics_query_async
                                logger.info("[AgenticMode] Successfully imported from backend.agents")
                            except ImportError as e1:
                                import_error = e1
                                # Try relative import (when running from backend directory)
                                try:
                                    from agents import execute_analytics_query_async
                                    logger.info("[AgenticMode] Successfully imported from agents (relative)")
                                except ImportError as e2:
                                    import_error = e2
                                    # Try adding parent directory to path
                                    try:
                                        import sys
                                        import os
                                        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                        if backend_dir not in sys.path:
                                            sys.path.insert(0, backend_dir)
                                            logger.info(f"[AgenticMode] Added {backend_dir} to sys.path")
                                        from agents import execute_analytics_query_async
                                        logger.info("[AgenticMode] Successfully imported after path adjustment")
                                    except ImportError as e3:
                                        import_error = e3

                            if execute_analytics_query_async is None:
                                raise ImportError(f"Could not import execute_analytics_query_async: {import_error}")

                            # Get workspace ID (use first workspace or "default")
                            agentic_workspace_id = str(curr_workspace_ids[0]) if curr_workspace_ids else "default"

                            # Get LLM from existing service (vLLM or Ollama based on config)
                            from .llm_service import get_llm_service
                            llm_service = get_llm_service()
                            agentic_llm = llm_service.get_chat_model(temperature=0.1)
                            logger.info(f"[AgenticMode] Using LLM: {type(agentic_llm).__name__}")

                            await progress_callback("Planning query execution...")

                            # Send initial status message to show auto-flowing status UI
                            await websocket.send_json({
                                "type": "status",
                                "agent": "initializing",
                                "phase": "initializing",
                                "message": "Initializing agent workflow",
                                "sub_messages": [
                                    f"Preparing to analyze {len(accessible_doc_ids_list) if accessible_doc_ids_list else 0} documents",
                                    "Setting up execution environment"
                                ],
                                "metrics": {"document_count": len(accessible_doc_ids_list) if accessible_doc_ids_list else 0},
                                "iteration": 0
                            })

                            # Execute agentic workflow with streaming
                            # iterative_reasoning_enabled controls Graph RAG agent
                            graph_search_enabled = iterative_reasoning_enabled if iterative_reasoning_enabled is not None else True
                            logger.info(f"[AgenticMode] Graph Search Enabled: {graph_search_enabled}")

                            # Convert document IDs to strings for the agentic workflow
                            accessible_doc_ids_str_list = [str(doc_id) for doc_id in accessible_doc_ids_list] if accessible_doc_ids_list else []
                            logger.info(f"[AgenticMode] Passing {len(accessible_doc_ids_str_list)} document IDs to workflow")

                            async for event in execute_analytics_query_async(
                                user_query=enhanced_message,
                                workspace_id=agentic_workspace_id,
                                llm=agentic_llm,
                                chat_history=history,
                                max_iterations=3,
                                stream_progress=True,
                                enable_graph_search=graph_search_enabled,
                                accessible_doc_ids=accessible_doc_ids_str_list
                            ):
                                event_type = event.get("type")

                                if event_type == "progress":
                                    agent_name = event.get("agent", "")
                                    iteration = event.get("iteration", 0)
                                    status_details = event.get("status_details", {})

                                    # Map agent names to user-friendly progress messages (fallback)
                                    agent_progress_messages = {
                                        "planner_agent": "Analyzing your question and creating execution plan...",
                                        "retrieval_team": f"Retrieving data from documents (iteration {iteration + 1})...",
                                        "reviewer_agent": "Reviewing data quality...",
                                        "summary_agent": "Generating final response..."
                                    }
                                    progress_msg = agent_progress_messages.get(agent_name, f"Processing with {agent_name}...")
                                    await progress_callback(progress_msg)

                                    # Send detailed status as auto-flowing chat-like messages
                                    if status_details:
                                        await websocket.send_json({
                                            "type": "status",
                                            "agent": agent_name,
                                            "phase": status_details.get("phase", ""),
                                            "message": status_details.get("message", ""),
                                            "sub_messages": status_details.get("sub_messages", []),
                                            "metrics": status_details.get("metrics", {}),
                                            "iteration": iteration
                                        })

                                elif event_type == "result":
                                    # Stream the final response
                                    response_text = event.get("response", "")
                                    if response_text:
                                        await progress_callback("Writing your report...")
                                        # Stream response in chunks
                                        chunk_size = 20
                                        for i in range(0, len(response_text), chunk_size):
                                            chunk = response_text[i:i + chunk_size]
                                            chunk_count += 1
                                            full_response += chunk
                                            await websocket.send_json({
                                                "type": "token",
                                                "content": chunk,
                                            })
                                            await asyncio.sleep(0.01)

                                        # Add data sources section if available and not already in response
                                        # (Summary Agent may have already included data sources)
                                        data_sources = event.get("data_sources", [])
                                        if data_sources and "Data Sources:" not in full_response:
                                            sources_text = "\n\n---\n**Data Sources:**\n"
                                            for source in data_sources:
                                                if isinstance(source, dict):
                                                    filename = source.get('filename', source.get('document_id', 'Unknown'))
                                                else:
                                                    filename = str(source)
                                                sources_text += f"- {filename}\n"
                                            full_response += sources_text
                                            await websocket.send_json({
                                                "type": "token",
                                                "content": sources_text,
                                            })

                                    logger.info(f"[AgenticMode] Streamed agentic response ({len(full_response)} chars)")

                                elif event_type == "error":
                                    error_msg = event.get("error", "Unknown error in agentic workflow")
                                    logger.error(f"[AgenticMode] Error: {error_msg}")
                                    # Send friendly error response
                                    error_response = f"I encountered an issue while processing your request with the agentic workflow. Please try again.\n\nError details: {error_msg}"
                                    for chunk in [error_response[i:i+50] for i in range(0, len(error_response), 50)]:
                                        full_response += chunk
                                        await websocket.send_json({
                                            "type": "token",
                                            "content": chunk,
                                        })

                            # Save assistant response to database
                            if full_response:
                                try:
                                    with get_db_session() as db:
                                        conv_manager = ConversationManager(db)
                                        conv_manager.add_message(
                                            session_id=UUID(session_id),
                                            role="assistant",
                                            content=full_response,
                                            metadata={"source": "agentic_workflow"}
                                        )
                                except Exception as e:
                                    logger.error(f"Error saving agentic response: {e}")

                            # Update session metadata
                            try:
                                with get_db_session() as db:
                                    conv_manager = ConversationManager(db)
                                    session = conv_manager.get_session(UUID(session_id))
                                    if session:
                                        current_metadata = session.session_metadata or {}
                                        current_metadata["_prev_workspace_ids"] = curr_workspace_ids
                                        current_metadata["_prev_document_ids"] = curr_document_ids
                                        conv_manager.update_session_metadata(UUID(session_id), current_metadata)
                            except Exception as e:
                                logger.error(f"Error updating session metadata: {e}")

                            logger.info("=" * 80)
                            logger.info(f"[AgenticMode] ========== AGENTIC WORKFLOW COMPLETE ==========")
                            logger.info("=" * 80)

                            # Send end message and continue to next message
                            timing_metrics.log_summary()
                            clear_current_metrics()
                            await websocket.send_json({"type": "end"})
                            continue

                        except ImportError as ie:
                            logger.error(f"[AgenticMode] Import error - agents module not available: {ie}")
                            # Fall back to standard routing
                            logger.info("[AgenticMode] Falling back to standard routing due to import error")
                        except Exception as e:
                            logger.error(f"[AgenticMode] Error in agentic workflow: {e}", exc_info=True)
                            # Fall back to standard routing on error
                            logger.info("[AgenticMode] Falling back to standard routing due to error")

                    # ====== INTENT-BASED ROUTING ======
                    use_analytics_path = False
                    use_rag_path = True  # Default to RAG
                    analytics_result = None
                    classification = None

                    if CHAT_ENABLE_INTENT_ROUTING and ANALYTICS_ENABLED and accessible_doc_ids_list:
                        try:
                            # Initialize orchestrator (short-lived session)
                            with get_db_session() as db:
                                orchestrator = ChatOrchestrator(db)

                                # Get intent classification
                                # If unified query analysis was done, use its result; otherwise do separate classification
                                if query_analysis is not None:
                                    # Use pre-computed intent from unified query analysis (NO additional LLM call)
                                    logger.info("[Intent Routing] Using intent from unified query analysis (NO additional LLM call)")
                                    # Create IntentClassification from QueryAnalysisResult
                                    classification = IntentClassification(
                                        intent=query_analysis.intent,
                                        confidence=query_analysis.intent_confidence,
                                        reasoning="",
                                        requires_extracted_data=query_analysis.requires_extracted_data,
                                        suggested_schemas=[],
                                        detected_entities=query_analysis.entities,
                                        detected_metrics=[],
                                        detected_time_range=query_analysis.time_range
                                    )
                                else:
                                    # Fallback: Run separate intent classification (LLM call)
                                    await progress_callback("Understanding your question...")
                                    classification = orchestrator.classify_query(enhanced_message, history)

                                logger.info(f"[Intent Routing] Query: '{enhanced_message[:50]}...' -> Intent: {classification.intent.value}, Confidence: {classification.confidence:.2f}")

                                # Determine routing based on classification
                                use_analytics_path = orchestrator.should_use_analytics(classification)
                                use_rag_path = orchestrator.should_use_rag(classification)

                                # OVERRIDE: If user has explicitly selected specific documents, always use RAG
                                # This ensures document search is performed when user wants to query their selected docs
                                if document_ids and len(document_ids) > 0 and not use_rag_path:
                                    logger.info(f"[Intent Routing] Override: User selected {len(document_ids)} specific document(s), forcing RAG path")
                                    use_rag_path = True

                                logger.info(f"[Intent Routing] Routing decision: analytics={use_analytics_path}, rag={use_rag_path}")

                                # Execute analytics query if needed
                                if use_analytics_path:
                                    # ====== DOCUMENT RELEVANCE FILTERING FOR ANALYTICS ======
                                    # Reuse the same document routing logic as RAG path to filter
                                    # to relevant documents before querying structured data
                                    from .document_router import DocumentRouter
                                    from .llm_service import get_llm_service

                                    # Get query metadata for document routing
                                    # Use data from unified query analysis if available (NO additional LLM call)
                                    if query_analysis is not None:
                                        await progress_callback("Using pre-analyzed topics...")
                                        query_metadata = {
                                            "entities": query_analysis.entities,
                                            "topics": query_analysis.topics,
                                            "document_type_hints": query_analysis.document_type_hints,
                                            "intent": query_analysis.intent.value,
                                        }
                                        logger.info(f"[Intent Routing] Using unified query analysis for document routing (NO additional LLM call)")
                                    else:
                                        # Fallback: run separate query analysis
                                        from .rag_agent import _analyze_query_with_llm
                                        await progress_callback("Analyzing key topics in your question...")
                                        fallback_analysis = _analyze_query_with_llm(enhanced_message)
                                        query_metadata = fallback_analysis.get("metadata", {})

                                    # Convert UUIDs to strings for the router (router expects List[str])
                                    accessible_doc_ids_str = [str(doc_id) for doc_id in accessible_doc_ids_list]

                                    # Route to relevant documents
                                    await progress_callback("Finding relevant documents...")
                                    llm_service = get_llm_service()
                                    router = DocumentRouter(llm_service=llm_service)
                                    routed_document_ids = router.route_query(
                                        query_metadata,
                                        original_query=enhanced_message,
                                        accessible_document_ids=accessible_doc_ids_str
                                    )

                                    # Use routed document IDs if available, otherwise fall back to all accessible
                                    # Router returns List[str], need to convert back to List[UUID] for orchestrator
                                    if routed_document_ids and len(routed_document_ids) > 0:
                                        analytics_doc_ids = [UUID(doc_id) for doc_id in routed_document_ids]
                                        logger.info(f"[Intent Routing] Analytics using {len(analytics_doc_ids)} routed documents (filtered from {len(accessible_doc_ids_list)} accessible)")
                                    else:
                                        analytics_doc_ids = accessible_doc_ids_list
                                        logger.info(f"[Intent Routing] Analytics using all {len(analytics_doc_ids)} accessible documents (no routing filter applied)")

                                    # Execute analytics query with progress callback
                                    analytics_result = await orchestrator.execute_analytics_query_async(
                                        query=enhanced_message,
                                        classification=classification,
                                        accessible_doc_ids=analytics_doc_ids,
                                        progress_callback=progress_callback
                                    )
                                    logger.info(f"[Intent Routing] Analytics result: {analytics_result.get('summary', {})}")

                                    # For pure analytics queries, stream the report directly from LLM
                                    if not use_rag_path and analytics_result.get('data'):
                                        # Check if streaming generator is available
                                        stream_generator = analytics_result.get('_stream_generator')
                                        field_mappings = analytics_result.get('_field_mappings')

                                        if stream_generator and hasattr(stream_generator, 'stream_summary_report'):
                                            # Stream the report directly from LLM
                                            await progress_callback("Writing your report...")
                                            logger.info(f"[Intent Routing] Streaming report directly from LLM...")
                                            # Extract additional context for large result handling
                                            summary = analytics_result.get('summary', {})
                                            schema_type = summary.get('schema_type', 'unknown')
                                            sql_explanation = summary.get('explanation', '')
                                            async for chunk in stream_generator.stream_summary_report(
                                                enhanced_message,
                                                analytics_result.get('data', []),
                                                field_mappings or {},
                                                schema_type=schema_type,
                                                sql_explanation=sql_explanation
                                            ):
                                                if chunk:
                                                    chunk_count += 1
                                                    full_response += chunk
                                                    await websocket.send_json({
                                                        "type": "token",
                                                        "content": chunk,
                                                    })

                                            # Append document sources at the end of streamed report
                                            metadata = analytics_result.get('metadata', {})
                                            document_sources = metadata.get('document_sources', [])
                                            if document_sources:
                                                sources_text = "\n\n---\n**Data Sources:**\n"
                                                for source in document_sources:
                                                    filename = source.get('filename', 'Unknown')
                                                    sources_text += f"- {filename}\n"
                                                full_response += sources_text
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "content": sources_text,
                                                })

                                            logger.info(f"[Intent Routing] Streamed analytics report ({len(full_response)} chars)")
                                        else:
                                            # Fallback: use format_analytics_response with simulated streaming
                                            await progress_callback("Preparing your answer...")
                                            formatted_response = orchestrator.format_analytics_response(
                                                query=enhanced_message,
                                                analytics_result=analytics_result,
                                                classification=classification
                                            )
                                            # Stream the formatted response with realistic timing
                                            words = formatted_response.split(' ')
                                            current_chunk = ""
                                            for i, word in enumerate(words):
                                                current_chunk += word + ' '
                                                is_sentence_end = word.endswith(('.', '!', '?', ':', '\n'))
                                                if len(current_chunk) >= 20 or is_sentence_end or i == len(words) - 1:
                                                    chunk_count += 1
                                                    full_response += current_chunk
                                                    await websocket.send_json({
                                                        "type": "token",
                                                        "content": current_chunk,
                                                    })
                                                    current_chunk = ""
                                                    await asyncio.sleep(0.015)
                                            logger.info(f"[Intent Routing] Streamed analytics response ({len(full_response)} chars)")
                                    elif not use_rag_path and not analytics_result.get('data'):
                                        # Analytics returned no data - fall back to RAG to search documents
                                        logger.info(f"[Intent Routing] Analytics returned no data, falling back to RAG for document search")
                                        use_rag_path = True

                        except Exception as e:
                            logger.error(f"[Intent Routing] Error during intent classification/analytics: {e}", exc_info=True)
                            # Fall back to RAG on error
                            use_rag_path = True
                            use_analytics_path = False

                    # ====== RAG PATH (if needed) ======
                    if use_rag_path:
                        rag_start = time.time()
                        # Prepare context injection for hybrid queries
                        hybrid_context = None
                        if use_analytics_path and analytics_result and classification:
                            try:
                                with get_db_session() as db:
                                    orchestrator = ChatOrchestrator(db)
                                    hybrid_context = orchestrator._summarize_analytics_for_context(analytics_result)
                                logger.info(f"[Intent Routing] Injecting analytics context for hybrid query")
                            except Exception as e:
                                logger.warning(f"[Intent Routing] Failed to prepare hybrid context: {e}")

                        try:
                            # Get preprocessing data to pass to agent (avoid redundant LLM call in rag_agent)
                            # These come from the unified query analysis which already did entity/topic extraction
                            preprocessing_topics = context_info.get('topics', []) if context_info else []
                            preprocessing_entities = context_info.get('entities', []) if context_info else []
                            preprocessing_doc_types = context_info.get('document_type_hints', []) if context_info else []
                            preprocessing_enhanced_query = context_info.get('enhanced_query', '') if context_info else ''

                            async for chunk in stream_agent_response(
                                enhanced_message,
                                history,
                                progress_callback=progress_callback,
                                session_context=session_context,
                                is_retry=is_retry,
                                accessible_doc_ids=accessible_doc_ids,
                                analytics_context=hybrid_context,  # Pass analytics context for hybrid queries
                                document_context_changed=document_context_changed,  # Force new search if selection changed
                                preprocessing_topics=preprocessing_topics,  # Pass topics from unified query analysis
                                preprocessing_entities=preprocessing_entities,  # Pass entities from unified query analysis
                                preprocessing_doc_types=preprocessing_doc_types,  # Pass document type hints
                                preprocessing_enhanced_query=preprocessing_enhanced_query,  # Pass enhanced query
                                iterative_reasoning_enabled=iterative_reasoning_enabled  # Pass UI checkbox toggle for iterative vs simple flow
                            ):
                                chunk_count += 1
                                full_response += chunk
                                if chunk_count == 1:
                                    logger.info(f"[Streaming] Received first chunk (length: {len(chunk)})")
                                await websocket.send_json({
                                    "type": "token",
                                    "content": chunk,
                                })

                            logger.info(f"[Streaming] Received {chunk_count} chunks, total response length: {len(full_response)} chars")

                            # Check if we got any response
                            if chunk_count == 0 or not full_response.strip():
                                logger.warning("[Streaming] No chunks received from agent - sending friendly message")
                                # Send a user-friendly message as a normal response (not error)
                                friendly_message = "I couldn't find relevant information to answer your question. This could be because:\n\n• The documents in the selected workspace don't contain information related to your query\n• The question might need to be rephrased for better results\n• The data you're looking for hasn't been uploaded yet\n\nPlease try rephrasing your question or check if the relevant documents are available in your workspace."
                                full_response = friendly_message
                                # Stream the friendly message to the client
                                for chunk in [friendly_message[i:i+50] for i in range(0, len(friendly_message), 50)]:
                                    await websocket.send_json({
                                        "type": "token",
                                        "content": chunk,
                                    })
                                # Don't set streaming_error - this is a valid response

                        except WebSocketDisconnect as stream_error:
                            # Client disconnected - expected behavior (user navigated away, network issue, etc.)
                            streaming_error = stream_error
                            logger.info(f"[Streaming] Client disconnected after {chunk_count} chunks - this is normal if user navigated away")
                            # Don't raise - just stop streaming gracefully
                        except Exception as stream_error:
                            streaming_error = stream_error
                            error_msg = str(stream_error) if str(stream_error) else f"{type(stream_error).__name__}"
                            logger.error(f"Error during streaming: {error_msg}", exc_info=True)
                            # Send a user-friendly error message as a normal response
                            friendly_error = f"I encountered an issue while processing your request. Please try again, or rephrase your question.\n\nIf the problem persists, the system might be temporarily unavailable."
                            if not full_response:  # Only set if no partial response was sent
                                full_response = friendly_error
                                try:
                                    for chunk in [friendly_error[i:i+50] for i in range(0, len(friendly_error), 50)]:
                                        await websocket.send_json({
                                            "type": "token",
                                            "content": chunk,
                                        })
                                except Exception:
                                    pass  # WebSocket might be closed

                    # Record RAG pipeline time if RAG path was used
                    if use_rag_path:
                        timing_metrics.record("rag_pipeline_total", (time.time() - rag_start) * 1000)

                    # Save assistant response to database ONLY if no error occurred
                    # Don't save error messages to chat history - let user retry
                    if full_response and not streaming_error:
                        try:
                            with get_db_session() as db:
                                conv_manager = ConversationManager(db)
                                conv_manager.add_message(
                                    session_id=UUID(session_id),
                                    role="assistant",
                                    content=full_response
                                )
                        except Exception as e:
                            logger.error(f"Error saving assistant message: {e}")

                        # ====== BACKGROUND CACHE STORAGE ======
                        # Store the response in cache for future similar questions
                        # This is non-blocking (fire-and-forget)
                        # Use variables from unified query analysis (cache_is_cacheable, cache_key)
                        should_cache = (
                            QUERY_CACHE_ENABLED and
                            cache_is_cacheable and
                            not cache_hit and  # Don't re-cache cached responses
                            accessible_doc_ids_list
                        )
                        if should_cache:
                            try:
                                # Check if response indicates "not found" or failure - DON'T cache these
                                response_lower = full_response.lower()
                                NOT_FOUND_INDICATORS = [
                                    "couldn't find",
                                    "could not find",
                                    "no documents found",
                                    "no results found",
                                    "unable to find",
                                    "i couldn't find any",
                                    "i could not find any",
                                    "none are specifically related",
                                    "no matching documents",
                                    "no relevant documents",
                                    "don't have any information",
                                    "do not have any information",
                                    "no information available",
                                    "no data found",
                                    "i don't have access to",
                                    "i do not have access to",
                                ]

                                is_not_found_response = any(indicator in response_lower for indicator in NOT_FOUND_INDICATORS)

                                if is_not_found_response:
                                    logger.info(f"[QueryCache] SKIP CACHE STORAGE: Response indicates 'not found' - should not be cached")
                                    logger.info(f"[QueryCache]   • Response preview: {full_response[:100]}...")
                                else:
                                    cache_service = get_query_cache_service()
                                    # Use cache_key from unified query analysis (already cleaned)
                                    cache_question = cache_key if cache_key else enhanced_message
                                    source_doc_ids = [str(doc_id) for doc_id in accessible_doc_ids_list]

                                    # Determine intent for TTL
                                    cache_intent = "general"
                                    if classification:
                                        cache_intent = classification.intent.value

                                    # Determine cache workspace ID
                                    store_workspace_id = str(curr_workspace_ids[0]) if curr_workspace_ids else "default"

                                    # Store in background (non-blocking)
                                    cache_service.store_async(
                                        question=cache_question,
                                        answer=full_response,
                                        workspace_id=store_workspace_id,
                                        source_document_ids=source_doc_ids,
                                        intent=cache_intent,
                                        metadata={
                                            "original_question": enhanced_message,
                                            "session_id": session_id,
                                            "analysis_method": cache_analysis_method if query_analysis else "fallback"
                                        }
                                    )
                                    logger.info(f"[QueryCache] Background cache storage initiated for: {cache_question[:50]}...")
                            except Exception as cache_store_error:
                                logger.warning(f"[QueryCache] Failed to store response in cache: {cache_store_error}")
                                # Non-critical, continue without caching

                    elif streaming_error:
                        logger.info(f"[WebSocket] Not saving assistant message due to streaming error - user can retry")

                    # Update session metadata with context information (short-lived session)
                    # ALWAYS save document context (workspace/document selection) for change detection
                    try:
                        with get_db_session() as db:
                            conv_manager = ConversationManager(db)
                            session = conv_manager.get_session(UUID(session_id))
                            if session:
                                current_metadata = session.session_metadata or {}

                                # Context tracking (entities, topics) - only if enabled
                                if CHAT_ENABLE_METADATA_TRACKING and context_info:
                                    # Merge entities (list of proper nouns from unified analyzer)
                                    entities = context_info.get("entities", [])
                                    if entities:
                                        if "entities" not in current_metadata:
                                            current_metadata["entities"] = []
                                        current_metadata["entities"].extend(entities)
                                        current_metadata["entities"] = list(set(current_metadata["entities"]))[-CHAT_MAX_ENTITIES_PER_TYPE:]

                                    # Update topics (keep last N)
                                    current_topics = current_metadata.get("topics", [])
                                    new_topics = context_info.get("topics", [])
                                    if new_topics:
                                        current_topics.extend(new_topics)
                                        current_metadata["topics"] = list(set(current_topics))[-CHAT_MAX_TOPICS:]

                                    # Update last message info for quick context display
                                    current_metadata["last_message_preview"] = message[:100]
                                    current_metadata["last_response_preview"] = full_response[:100] if full_response else ""

                                # Save current workspace/document selection to session metadata for two purposes:
                                # 1. Detect mid-conversation filter changes (force new search when user changes selection)
                                # 2. Restore document context when reopening an existing chat session
                                # Note: For NEW sessions, initial selection comes from user preferences in users table
                                current_metadata["_prev_workspace_ids"] = curr_workspace_ids
                                current_metadata["_prev_document_ids"] = curr_document_ids

                                # Update metadata in database
                                conv_manager.update_session_metadata(UUID(session_id), current_metadata)

                                if document_context_changed:
                                    logger.info(f"[Document Context] Selection changed - saved to session: workspaces={len(curr_workspace_ids)}, docs={len(curr_document_ids)}")
                                logger.debug(f"Updated session metadata: {current_metadata}")

                    except Exception as e:
                        logger.error(f"Error updating session metadata: {e}")

                    # Signal end of response (or error if streaming failed)
                    logger.info(f"[WebSocket] Preparing to send end message (streaming_error={streaming_error is not None})")
                    logger.info(f"[WebSocket] WebSocket state: client_state={websocket.client_state}, application_state={websocket.application_state}")

                    try:
                        # Check if WebSocket is still open
                        from starlette.websockets import WebSocketState
                        if websocket.client_state != WebSocketState.CONNECTED:
                            logger.error(f"[WebSocket] Cannot send end message - WebSocket not connected (state={websocket.client_state})")
                        else:
                            if streaming_error:
                                error_msg = str(streaming_error) if str(streaming_error) else f"{type(streaming_error).__name__}"
                                logger.info(f"[WebSocket] Sending error message: {error_msg}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Streaming error: {error_msg}",
                                })
                            else:
                                logger.info(f"[WebSocket] Sending end message")
                                await websocket.send_json({"type": "end"})
                            logger.info(f"[WebSocket] End/error message sent successfully")
                    except Exception as ws_error:
                        logger.error(f"[WebSocket] Error sending end/error message: {ws_error}", exc_info=True)

                    # Log final timing summary
                    timing_metrics.log_summary()
                    clear_current_metrics()

                except Exception as e:
                    error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
                    logger.error(f"Error in message processing: {error_msg}", exc_info=True)
                    # Send a user-friendly response instead of raw error
                    try:
                        friendly_error = "I'm sorry, but I encountered an unexpected issue while processing your question. Please try again.\n\nIf you continue to experience problems, try:\n• Refreshing the page\n• Rephrasing your question\n• Selecting a different workspace"
                        for chunk in [friendly_error[i:i+50] for i in range(0, len(friendly_error), 50)]:
                            await websocket.send_json({
                                "type": "token",
                                "content": chunk,
                            })
                        await websocket.send_json({"type": "end"})
                    except Exception as ws_error:
                        logger.error(f"WebSocket error: {ws_error}")

            except json.JSONDecodeError:
                # Send a user-friendly response for invalid input
                try:
                    friendly_error = "I couldn't understand your message format. Please try sending your question again."
                    for chunk in [friendly_error[i:i+50] for i in range(0, len(friendly_error), 50)]:
                        await websocket.send_json({
                            "type": "token",
                            "content": chunk,
                        })
                    await websocket.send_json({"type": "end"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/reindex")
async def trigger_reindex():
    """Trigger a re-indexing of all documents."""
    from .indexer import index_existing_documents

    try:
        count = index_existing_documents()
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        logger.error(f"Error during reindex: {e}")
        raise HTTPException(status_code=500, detail=str(e))

