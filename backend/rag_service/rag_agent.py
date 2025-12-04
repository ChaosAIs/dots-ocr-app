"""
LangGraph-based RAG Agent with document search tool.
Uses Ollama with qwen3:30b-a3b model for generation.
"""

import os
import logging
from typing import Annotated, TypedDict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .vectorstore import get_retriever

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"


class AgentState(TypedDict):
    """State for the RAG agent."""

    messages: Annotated[List[BaseMessage], add_messages]


# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about documents.
You have access to a search tool that can retrieve relevant information from indexed documents.

When answering questions:
1. Use the search_documents tool to find relevant information from the documents
2. Base your answers on the retrieved context
3. If you can't find relevant information, say so honestly
4. Cite the source document when possible
5. Provide clear, concise answers

Always search for relevant information before answering questions about the documents.
Do not provide thinking or reflection tags in your response. Provide direct, clear answers."""


def _keyword_boost_search(query: str, vector_results: list, boost_k: int = 5) -> list:
    """
    Boost search results by adding documents that contain query keywords.
    This helps when semantic search doesn't find exact keyword matches.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    # Extract keywords from query (simple word tokenization)
    keywords = [w.lower() for w in query.split() if len(w) > 2]

    if not keywords:
        return vector_results

    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            check_compatibility=False
        )

        # Get all unique sources first
        all_points = client.scroll(
            collection_name="dots_ocr_documents",
            limit=1000,
            with_payload=["metadata"],
            with_vectors=False,
        )
        points, _ = all_points

        # Find sources that match keywords (case-insensitive)
        matching_sources = set()
        for p in points:
            source = p.payload.get("metadata", {}).get("source", "").lower()
            for keyword in keywords:
                if keyword in source:
                    matching_sources.add(p.payload.get("metadata", {}).get("source", ""))
                    break

        if not matching_sources:
            return vector_results

        logger.info(f"Keyword boost found matching sources: {matching_sources}")

        # Fetch documents from matching sources
        keyword_docs = []
        for source in matching_sources:
            results = client.scroll(
                collection_name="dots_ocr_documents",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                ),
                limit=boost_k,
                with_payload=True,
                with_vectors=False,
            )
            points, _ = results
            for p in points:
                doc = Document(
                    page_content=p.payload.get("page_content", ""),
                    metadata=p.payload.get("metadata", {}),
                )
                keyword_docs.append(doc)

        # Combine: keyword matches first, then vector results
        # Remove duplicates based on content
        seen_content = set()
        combined = []

        for doc in keyword_docs + vector_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(doc)

        return combined[:len(vector_results) + boost_k]

    except Exception as e:
        logger.warning(f"Keyword boost search failed: {e}")
        return vector_results


@tool
def search_documents(query: str) -> str:
    """
    Search the indexed documents for relevant information.

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Relevant document chunks as a formatted string.
    """
    try:
        # Increase k to retrieve more documents for better recall
        # The LLM will filter for relevance from these results
        retriever = get_retriever(k=15, fetch_k=50)
        docs = retriever.invoke(query)

        # Boost with keyword matching for better recall
        docs = _keyword_boost_search(query, docs, boost_k=10)

        if not docs:
            return "No relevant documents found for this query."

        # Format results - include all unique sources for better coverage
        results = []
        seen_sources = set()

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            heading = doc.metadata.get("heading_path", "")
            content = doc.page_content[:800]  # Limit content length

            result = f"[Document {i}: {source}]"
            if heading:
                result += f"\nSection: {heading}"
            result += f"\n{content}"
            results.append(result)
            seen_sources.add(source)

        # Log which sources were found
        logger.info(f"Search query '{query[:50]}...' found {len(docs)} docs from sources: {seen_sources}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error searching documents: {str(e)}"


# Define tools
tools = [search_documents]


def create_llm():
    """Create the Ollama LLM instance."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.2,
        num_ctx=8192,
    )


def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made a tool call, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return END


def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current state."""
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def create_agent_executor():
    """
    Create the LangGraph agent executor.

    Returns:
        Compiled LangGraph workflow.
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    # Compile
    return workflow.compile()


async def stream_agent_response(query: str, conversation_history: List[dict] = None):
    """
    Stream the agent response for a query.

    Args:
        query: The user's question.
        conversation_history: Optional list of previous messages.

    Yields:
        Chunks of the response text.
    """
    agent = create_agent_executor()

    # Build message history
    messages = []
    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # Add current query
    messages.append(HumanMessage(content=query))

    # Stream the response
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content

