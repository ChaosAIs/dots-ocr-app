"""
WebSocket API for real-time chat with the RAG agent.
Provides streaming responses for better UX.
"""

import json
import logging
from typing import List, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from .rag_agent import stream_agent_response, create_agent_executor
from .vectorstore import get_collection_info
from .indexer import get_indexed_count

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


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses.

    Protocol:
    - Client sends JSON: {"message": "...", "history": [...]}
    - Server streams JSON: {"type": "token", "content": "..."} for each token
    - Server sends JSON: {"type": "end"} when complete
    - Server sends JSON: {"type": "error", "message": "..."} on error
    """
    await websocket.accept()
    logger.info("WebSocket connection established for chat")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                request = json.loads(data)
                message = request.get("message", "")
                history = request.get("history", [])

                if not message:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty message received",
                    })
                    continue

                logger.info(f"Received chat message: {message[:100]}...")

                # Stream response
                try:
                    async for chunk in stream_agent_response(message, history):
                        await websocket.send_json({
                            "type": "token",
                            "content": chunk,
                        })

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
        logger.info("WebSocket disconnected")
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

