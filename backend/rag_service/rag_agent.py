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
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .vectorstore import (
    get_retriever,
    get_retriever_with_sources,
    search_file_summaries,
    search_chunk_summaries,
    get_chunks_by_ids,
)

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b")
# Smaller, faster model for query analysis (default: qwen2.5:7b)
OLLAMA_QUERY_MODEL = os.getenv("OLLAMA_QUERY_MODEL", "qwen2.5:7b")
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


# Prompt for query analysis - concise output to avoid repetition
QUERY_ANALYSIS_PROMPT = """Analyze the user's question and create an optimized search query.

User Question: {query}

Create a concise, focused search query (50-100 words max) that:
- Captures the main topic and user intent
- Includes key concepts and relevant terms
- Avoids repetition

Output ONLY the search query text, nothing else."""


def _analyze_query_with_llm(query: str) -> str:
    """
    Use LLM to analyze the user's query and generate an enhanced search query.

    This function:
    1. Sends the user query to LLM for analysis
    2. LLM clarifies the query purpose and user intent
    3. Extracts keywords, topics, and related concepts
    4. Generates an enhanced query optimized for vector search

    Args:
        query: The original user query

    Returns:
        Enhanced search query string optimized for embedding-based search
    """
    try:
        # Use smaller, faster model for query analysis
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_QUERY_MODEL,
            temperature=0.1,  # Low temperature for consistent analysis
            num_ctx=2048,
            num_predict=256,  # Limit output tokens to prevent endless generation
        )

        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        response = llm.invoke([HumanMessage(content=prompt)])

        enhanced_query = response.content.strip()

        # Clean up any thinking tags if present
        if "</think>" in enhanced_query:
            enhanced_query = enhanced_query.split("</think>")[-1].strip()

        # Truncate if still too long (safety measure)
        if len(enhanced_query) > 500:
            enhanced_query = enhanced_query[:500]
            logger.warning("Query analysis output truncated to 500 chars")

        logger.info(f"Query analysis: '{query[:50]}...' -> '{enhanced_query[:100]}...'")

        return enhanced_query if enhanced_query else query

    except Exception as e:
        logger.warning(f"Query analysis failed, using original query: {e}")
        return query


def _find_relevant_files(query: str, k: int = 5) -> List[str]:
    """
    Find relevant files by searching file summaries first.

    This implements the two-phase retrieval strategy:
    1. Search file summaries to identify relevant documents
    2. Return list of source names that are relevant to the query

    Args:
        query: The search query.
        k: Maximum number of relevant files to return.

    Returns:
        List of source names that are relevant to the query.
    """
    try:
        # Search file summaries using the query
        summary_docs = search_file_summaries(query, k=k)

        if not summary_docs:
            logger.info("No file summaries found, will search all chunks")
            return []

        # Extract unique source names from the summary results
        relevant_sources = []
        seen_sources = set()

        for doc in summary_docs:
            source = doc.metadata.get("source", "")
            if source and source not in seen_sources:
                relevant_sources.append(source)
                seen_sources.add(source)

        logger.info(f"Found {len(relevant_sources)} relevant files from summaries: {relevant_sources}")
        return relevant_sources

    except Exception as e:
        logger.warning(f"Error searching file summaries: {e}")
        return []


def _find_relevant_chunks_from_summaries(
    query: str,
    source_names: List[str] = None,
    k: int = 15,
) -> List[str]:
    """
    Find relevant chunks by searching chunk summaries.

    Args:
        query: The search query.
        source_names: Optional list of source names to filter.
        k: Maximum number of chunk summaries to return.

    Returns:
        List of chunk_ids that are relevant to the query.
    """
    try:
        # Search chunk summaries
        chunk_summary_docs = search_chunk_summaries(query, k=k, source_names=source_names)

        if not chunk_summary_docs:
            logger.info("No chunk summaries found")
            return []

        # Extract unique chunk_ids from the summary results
        # Each summary may cover multiple chunks (combined small chunks)
        chunk_ids = []
        seen_ids = set()

        for doc in chunk_summary_docs:
            # Support both list format (new) and single string format (legacy)
            doc_chunk_ids = doc.metadata.get("chunk_ids", [])
            if not doc_chunk_ids:
                # Fallback to single chunk_id for legacy data
                single_id = doc.metadata.get("chunk_id", "")
                if single_id:
                    doc_chunk_ids = [single_id]

            for chunk_id in doc_chunk_ids:
                if chunk_id and chunk_id not in seen_ids:
                    chunk_ids.append(chunk_id)
                    seen_ids.add(chunk_id)

        logger.info(f"Found {len(chunk_ids)} relevant chunks from chunk summaries")
        return chunk_ids

    except Exception as e:
        logger.warning(f"Error searching chunk summaries: {e}")
        return []


@tool
def search_documents(query: str) -> str:
    """
    Search the indexed documents for relevant information using three-phase retrieval.

    Three-phase retrieval strategy:
    1. First search file summaries to identify relevant documents
    2. Then search chunk summaries to find relevant chunks within those files
    3. Finally retrieve full chunk content for the relevant chunks

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Relevant document chunks as a formatted string.
    """
    try:
        # Step 1: Use LLM to analyze and enhance the query
        enhanced_query = _analyze_query_with_llm(query)

        # Step 2: Find relevant files by searching file summaries first
        relevant_sources = _find_relevant_files(enhanced_query, k=5)

        # Step 3: Search chunk summaries to find relevant chunks
        chunk_ids = _find_relevant_chunks_from_summaries(
            enhanced_query,
            source_names=relevant_sources if relevant_sources else None,
            k=15,
        )

        docs = []

        # Step 4: Retrieve full chunk content
        if chunk_ids:
            # Get full chunks by their IDs
            docs = get_chunks_by_ids(chunk_ids, source_names=relevant_sources if relevant_sources else None)
            logger.info(f"Retrieved {len(docs)} full chunks by chunk_ids")

        # Fallback: If no chunks found via chunk summaries, use direct chunk search
        if not docs:
            logger.info("No chunks found via chunk summaries, falling back to direct chunk search")
            if relevant_sources:
                retriever = get_retriever_with_sources(
                    k=15, fetch_k=50, source_names=relevant_sources
                )
                logger.info(f"Searching chunks filtered by sources: {relevant_sources}")
            else:
                retriever = get_retriever(k=15, fetch_k=50)
                logger.info("No relevant files from summaries, searching all chunks")

            docs = retriever.invoke(enhanced_query)

            if not docs and relevant_sources:
                # Try without filter as last resort
                logger.info("No results with source filter, trying without filter")
                retriever = get_retriever(k=15, fetch_k=50)
                docs = retriever.invoke(enhanced_query)

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
        logger.info(
            f"Original query: '{query[:50]}...' | "
            f"Enhanced query: '{enhanced_query[:50]}...' | "
            f"Relevant sources from summaries: {relevant_sources} | "
            f"Chunk IDs from chunk summaries: {len(chunk_ids)} | "
            f"Found {len(docs)} chunks from sources: {seen_sources}"
        )

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

