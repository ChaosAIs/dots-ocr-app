"""
WebSocket API for real-time chat with the RAG agent.
Provides streaming responses for better UX.
"""

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

logger = logging.getLogger(__name__)

# Load configuration from environment
CHAT_ENABLE_CONTEXT_ANALYSIS = os.getenv("CHAT_ENABLE_CONTEXT_ANALYSIS", "true").lower() == "true"
CHAT_ENABLE_METADATA_TRACKING = os.getenv("CHAT_ENABLE_METADATA_TRACKING", "true").lower() == "true"
CHAT_MAX_ENTITIES_PER_TYPE = int(os.getenv("CHAT_MAX_ENTITIES_PER_TYPE", "10"))
CHAT_MAX_TOPICS = int(os.getenv("CHAT_MAX_TOPICS", "5"))

logger.info(f"Chat Context Configuration: ENABLE_CONTEXT_ANALYSIS={CHAT_ENABLE_CONTEXT_ANALYSIS}, ENABLE_METADATA_TRACKING={CHAT_ENABLE_METADATA_TRACKING}")

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
                    message = request.get("message", "")
                    user_id = request.get("user_id")

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

                        # Use enhanced message with resolved pronouns and pass session context
                        chunk_count = 0
                        logger.info(f"[Streaming] Starting to stream response for message: '{enhanced_message[:100]}...'")
                        try:
                            async for chunk in stream_agent_response(
                                enhanced_message,
                                history,
                                progress_callback=progress_callback,
                                session_context=session_context
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
                                logger.warning("[Streaming] No chunks received from agent - this may indicate an issue")
                                full_response = "I apologize, but I encountered an issue generating a response. Please try again."
                                streaming_error = Exception("No response chunks received from agent")

                        except Exception as stream_error:
                            streaming_error = stream_error
                            error_msg = str(stream_error) if str(stream_error) else f"{type(stream_error).__name__}"
                            logger.error(f"Error during streaming: {error_msg}", exc_info=True)
                            # Don't raise here - we want to send error message to client

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
                        try:
                            await websocket.send_json({
                                "type": "error",
                                "message": error_msg,
                            })
                        except Exception as ws_error:
                            logger.error(f"WebSocket error: {ws_error}")

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format",
                    })

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

