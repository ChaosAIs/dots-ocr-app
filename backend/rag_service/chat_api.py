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
from typing import List, Dict, Optional
from uuid import UUID
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .rag_agent import stream_agent_response, create_agent_executor
from .vectorstore import get_collection_info
from .indexer import get_indexed_count
from db.database import get_db_session
from chat_service.conversation_manager import ConversationManager
from chat_service.context_analyzer import ContextAnalyzer
from chat_service.chat_orchestrator import ChatOrchestrator, OrchestratorResult
from analytics_service.intent_classifier import QueryIntent
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

logger.info(f"Chat Context Configuration: ENABLE_CONTEXT_ANALYSIS={CHAT_ENABLE_CONTEXT_ANALYSIS}, ENABLE_METADATA_TRACKING={CHAT_ENABLE_METADATA_TRACKING}")
logger.info(f"Chat Intent Routing: ENABLE_INTENT_ROUTING={CHAT_ENABLE_INTENT_ROUTING}, ANALYTICS_ENABLED={ANALYTICS_ENABLED}")

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
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for chat session: {session_id}")

    # Get database session using context manager
    with get_db_session() as db:
        conv_manager = ConversationManager(db)

        # Verify session exists
        try:
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

                    if not message:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Empty message received",
                        })
                        continue

                    logger.info(f"Received chat message for session {session_id}: {message[:100]}...")

                    # Save user message to database
                    try:
                        conv_manager.add_message(
                            session_id=UUID(session_id),
                            role="user",
                            content=message
                        )
                        db.commit()
                    except Exception as e:
                        logger.error(f"Error saving user message: {e}")
                        db.rollback()

                    # Get conversation history from database
                    history = []
                    context_info = {}
                    session_context = {}
                    enhanced_message = message

                    try:
                        context = conv_manager.get_conversation_context(UUID(session_id))
                        # Convert to format expected by RAG agent
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in context.get("short_term_messages", [])
                        ]

                        # Get session metadata for context
                        session_context = context.get("session_metadata", {})

                        # ENHANCEMENT: Analyze context for pronouns and entities
                        if CHAT_ENABLE_CONTEXT_ANALYSIS:
                            logger.info(f"[Context Analysis] Analyzing message: '{message[:100]}...' with {len(history)} history messages")
                            try:
                                analyzer = ContextAnalyzer()
                                analysis = analyzer.analyze_message(message, history)

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
                            logger.warning(f"[Context Analysis] Context analysis is DISABLED (CHAT_ENABLE_CONTEXT_ANALYSIS={CHAT_ENABLE_CONTEXT_ANALYSIS})")

                    except Exception as e:
                        logger.error(f"Error loading conversation history: {e}")

                    # Stream response and accumulate content
                    full_response = ""
                    streaming_error = None

                    try:
                        # Create progress callback for WebSocket updates
                        async def progress_callback(message: str, percent: int = None):
                            """Send progress updates to frontend via WebSocket."""
                            try:
                                payload = {"type": "progress", "message": message}
                                if percent is not None:
                                    payload["percent"] = percent
                                await websocket.send_json(payload)
                            except Exception as e:
                                logger.error(f"Error sending progress update: {e}")

                        # Get accessible document IDs for the current user
                        # This ensures chat only searches documents the user has access to
                        accessible_doc_ids = None
                        logger.info(f"[Access Control] Checking document access for user_id: {user_id}")
                        if user_id:
                            try:
                                user_doc_repo = UserDocumentRepository(db)
                                accessible_doc_ids = user_doc_repo.get_user_accessible_document_ids(
                                    user_id=UUID(user_id),
                                    permission='read'
                                )
                                # IMPORTANT: If the set is empty, it means user has NO access to any documents
                                # This should block all document searches
                                if accessible_doc_ids is not None:
                                    if len(accessible_doc_ids) == 0:
                                        logger.warning(f"[Access Control] User {user_id} has NO document access permissions - chat will return no documents")
                                    else:
                                        logger.info(f"[Access Control] User {user_id} has access to {len(accessible_doc_ids)} documents: {list(accessible_doc_ids)[:5]}...")

                                        # Apply document filter if document_ids are provided (takes precedence over workspace filter)
                                        if document_ids and len(document_ids) > 0:
                                            logger.info(f"[Document Filter] Filtering by {len(document_ids)} document(s)")
                                            try:
                                                # Convert document_ids to UUID set
                                                selected_doc_ids = set(UUID(doc_id) for doc_id in document_ids)
                                                # Intersect with accessible documents
                                                original_count = len(accessible_doc_ids)
                                                accessible_doc_ids = accessible_doc_ids.intersection(selected_doc_ids)
                                                logger.info(f"[Document Filter] Filtered from {original_count} to {len(accessible_doc_ids)} documents")
                                            except Exception as doc_error:
                                                logger.warning(f"[Document Filter] Failed to filter by documents: {doc_error}")
                                                # Fall back to workspace filter if document filter fails
                                                if workspace_ids and len(workspace_ids) > 0:
                                                    try:
                                                        doc_repo = DocumentRepository(db)
                                                        workspace_doc_ids = doc_repo.get_document_ids_by_workspaces(
                                                            [UUID(ws_id) for ws_id in workspace_ids]
                                                        )
                                                        original_count = len(accessible_doc_ids)
                                                        accessible_doc_ids = accessible_doc_ids.intersection(workspace_doc_ids)
                                                        logger.info(f"[Workspace Filter] Fallback - filtered from {original_count} to {len(accessible_doc_ids)} documents")
                                                    except Exception as ws_error:
                                                        logger.warning(f"[Workspace Filter] Fallback also failed: {ws_error}")
                                        # Apply workspace filter if workspace_ids are provided but no document_ids
                                        elif workspace_ids and len(workspace_ids) > 0:
                                            logger.info(f"[Workspace Filter] Filtering by {len(workspace_ids)} workspace(s): {workspace_ids}")
                                            try:
                                                doc_repo = DocumentRepository(db)
                                                workspace_doc_ids = doc_repo.get_document_ids_by_workspaces(
                                                    [UUID(ws_id) for ws_id in workspace_ids]
                                                )
                                                # Intersect with accessible documents
                                                original_count = len(accessible_doc_ids)
                                                accessible_doc_ids = accessible_doc_ids.intersection(workspace_doc_ids)
                                                logger.info(f"[Workspace Filter] Filtered from {original_count} to {len(accessible_doc_ids)} documents (workspace has {len(workspace_doc_ids)} docs)")
                                            except Exception as ws_error:
                                                logger.warning(f"[Workspace Filter] Failed to filter by workspaces: {ws_error}")
                                                # Continue with all accessible documents if workspace filter fails
                                        else:
                                            logger.info(f"[Workspace Filter] No workspace/document filter applied - searching all accessible documents")
                                else:
                                    logger.warning(f"[Access Control] get_user_accessible_document_ids returned None for user {user_id}")
                            except Exception as e:
                                import traceback
                                logger.warning(f"[Access Control] Failed to get accessible documents for user {user_id}: {e}")
                                logger.warning(f"[Access Control] Exception traceback: {traceback.format_exc()}")
                                # If we can't get access list, use empty set to block all access (secure by default)
                                accessible_doc_ids = set()
                        else:
                            logger.warning(f"[Access Control] No user_id provided - cannot enforce access control")

                        # Use enhanced message with resolved pronouns and pass session context
                        chunk_count = 0
                        logger.info(f"[Streaming] Starting to stream response for message: '{enhanced_message[:100]}...' (is_retry={is_retry})")

                        # Convert accessible_doc_ids set to list for orchestrator
                        accessible_doc_ids_list = list(accessible_doc_ids) if accessible_doc_ids else []

                        # ====== INTENT-BASED ROUTING ======
                        use_analytics_path = False
                        use_rag_path = True  # Default to RAG
                        analytics_result = None
                        classification = None

                        if CHAT_ENABLE_INTENT_ROUTING and ANALYTICS_ENABLED and accessible_doc_ids_list:
                            try:
                                # Initialize orchestrator
                                orchestrator = ChatOrchestrator(db)

                                # Classify the query intent
                                await progress_callback("Understanding your question...")
                                classification = orchestrator.classify_query(enhanced_message, history)

                                logger.info(f"[Intent Routing] Query: '{enhanced_message[:50]}...' -> Intent: {classification.intent.value}, Confidence: {classification.confidence:.2f}")

                                # Determine routing based on classification
                                use_analytics_path = orchestrator.should_use_analytics(classification)
                                use_rag_path = orchestrator.should_use_rag(classification)

                                logger.info(f"[Intent Routing] Routing decision: analytics={use_analytics_path}, rag={use_rag_path}")

                                # Execute analytics query if needed
                                if use_analytics_path:
                                    await progress_callback("Finding relevant documents...")

                                    # ====== DOCUMENT RELEVANCE FILTERING FOR ANALYTICS ======
                                    # Reuse the same document routing logic as RAG path to filter
                                    # to relevant documents before querying structured data
                                    from .document_router import DocumentRouter
                                    from .llm_service import get_llm_service
                                    from .rag_agent import _analyze_query_with_llm

                                    # Get query metadata for document routing
                                    query_analysis = _analyze_query_with_llm(enhanced_message)
                                    query_metadata = query_analysis.get("metadata", {})

                                    # Convert UUIDs to strings for the router (router expects List[str])
                                    accessible_doc_ids_str = [str(doc_id) for doc_id in accessible_doc_ids_list]

                                    # Route to relevant documents
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

                                    # For pure analytics queries, stream the formatted response directly
                                    if not use_rag_path and analytics_result.get('data'):
                                        await progress_callback("Preparing your answer...")
                                        formatted_response = orchestrator.format_analytics_response(
                                            query=enhanced_message,
                                            analytics_result=analytics_result,
                                            classification=classification
                                        )
                                        # Stream the formatted response with realistic timing
                                        # Use word-based chunking for more natural streaming
                                        words = formatted_response.split(' ')
                                        current_chunk = ""
                                        for i, word in enumerate(words):
                                            current_chunk += word + ' '
                                            # Send chunk every 3-5 words or at sentence boundaries
                                            is_sentence_end = word.endswith(('.', '!', '?', ':', '\n'))
                                            if len(current_chunk) >= 20 or is_sentence_end or i == len(words) - 1:
                                                chunk_count += 1
                                                full_response += current_chunk
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "content": current_chunk,
                                                })
                                                current_chunk = ""
                                                # Small delay for streaming effect (10-30ms)
                                                await asyncio.sleep(0.015)
                                        logger.info(f"[Intent Routing] Streamed analytics response directly ({len(full_response)} chars)")
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
                            # Prepare context injection for hybrid queries
                            hybrid_context = None
                            if use_analytics_path and analytics_result and classification:
                                try:
                                    orchestrator = ChatOrchestrator(db)
                                    hybrid_context = orchestrator._summarize_analytics_for_context(analytics_result)
                                    logger.info(f"[Intent Routing] Injecting analytics context for hybrid query")
                                except Exception as e:
                                    logger.warning(f"[Intent Routing] Failed to prepare hybrid context: {e}")

                            try:
                                async for chunk in stream_agent_response(
                                    enhanced_message,
                                    history,
                                    progress_callback=progress_callback,
                                    session_context=session_context,
                                    is_retry=is_retry,
                                    accessible_doc_ids=accessible_doc_ids,
                                    analytics_context=hybrid_context  # Pass analytics context for hybrid queries
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

                        # Save assistant response to database ONLY if no error occurred
                        # Don't save error messages to chat history - let user retry
                        if full_response and not streaming_error:
                            try:
                                conv_manager.add_message(
                                    session_id=UUID(session_id),
                                    role="assistant",
                                    content=full_response
                                )
                                db.commit()
                            except Exception as e:
                                logger.error(f"Error saving assistant message: {e}")
                                db.rollback()
                        elif streaming_error:
                            logger.info(f"[WebSocket] Not saving assistant message due to streaming error - user can retry")

                        # Update session metadata with context information
                        if CHAT_ENABLE_METADATA_TRACKING and context_info:
                            try:
                                session = conv_manager.get_session(UUID(session_id))
                                if session:
                                    current_metadata = session.session_metadata or {}

                                    # Merge entities (keep last N of each type)
                                    for entity_type, values in context_info.get("entities", {}).items():
                                        if values:  # Only process if there are values
                                            if entity_type not in current_metadata:
                                                current_metadata[entity_type] = []
                                            current_metadata[entity_type].extend(values)
                                            # Deduplicate and keep last N
                                            current_metadata[entity_type] = list(set(current_metadata[entity_type]))[-CHAT_MAX_ENTITIES_PER_TYPE:]

                                    # Update topics (keep last N)
                                    current_topics = current_metadata.get("topics", [])
                                    new_topics = context_info.get("topics", [])
                                    if new_topics:
                                        current_topics.extend(new_topics)
                                        current_metadata["topics"] = list(set(current_topics))[-CHAT_MAX_TOPICS:]

                                    # Determine main topic from the most frequent topic
                                    if current_metadata.get("topics"):
                                        # Use the first topic as main topic (most recent/relevant)
                                        current_metadata["main_topic"] = current_metadata["topics"][0]

                                    # Update last message info for quick context display
                                    current_metadata["last_message_preview"] = message[:100]
                                    current_metadata["last_response_preview"] = full_response[:100]

                                    # Update metadata in database
                                    conv_manager.update_session_metadata(UUID(session_id), current_metadata)
                                    db.commit()

                                    logger.debug(f"Updated session metadata: {current_metadata}")

                            except Exception as e:
                                logger.error(f"Error updating session metadata: {e}")
                                db.rollback()

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

