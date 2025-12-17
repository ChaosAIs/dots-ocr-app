"""
WebSocket API for real-time chat with the RAG agent.
Provides streaming responses for better UX.
"""

import json
import logging
from typing import List, Dict
from uuid import UUID
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .rag_agent import stream_agent_response, create_agent_executor
from .vectorstore import get_collection_info
from .indexer import get_indexed_count
from db.database import get_db_session
from chat_service.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

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
                    try:
                        context = conv_manager.get_conversation_context(UUID(session_id))
                        # Convert to format expected by RAG agent
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in context.get("short_term_messages", [])
                        ]
                    except Exception as e:
                        logger.error(f"Error loading conversation history: {e}")

                    # Stream response and accumulate content
                    full_response = ""
                    try:
                        async for chunk in stream_agent_response(message, history):
                            full_response += chunk
                            await websocket.send_json({
                                "type": "token",
                                "content": chunk,
                            })

                        # Save assistant response to database
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

                        # Signal end of response
                        await websocket.send_json({"type": "end"})

                    except Exception as e:
                        logger.error(f"Error streaming response: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e),
                        })

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

