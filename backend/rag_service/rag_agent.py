"""
LangGraph-based RAG Agent with document search tool.
Supports multiple LLM backends: Ollama and vLLM.
"""

import os
import logging
from typing import Annotated, TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .vectorstore import (
    get_retriever,
    get_retriever_with_sources,
    get_chunks_by_ids,
)
from .llm_service import get_llm_service

# Import GraphRAG components
try:
    from .graph_rag import GraphRAG, GRAPH_RAG_ENABLED
    # If the graph_rag library is available, set the GRAPH_RAG_ENABLED flag = True
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GRAPH_RAG_ENABLED = False

logger = logging.getLogger(__name__)

# Legacy aliases for backward compatibility (used by other modules that import these)
RAG_OLLAMA_HOST = os.getenv("RAG_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "localhost"))
RAG_OLLAMA_PORT = os.getenv("RAG_OLLAMA_PORT", os.getenv("OLLAMA_PORT", "11434"))
RAG_OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:latest"))
RAG_OLLAMA_QUERY_MODEL = os.getenv("RAG_OLLAMA_QUERY_MODEL", os.getenv("OLLAMA_QUERY_MODEL", "qwen2.5:latest"))
RAG_OLLAMA_BASE_URL = f"http://{RAG_OLLAMA_HOST}:{RAG_OLLAMA_PORT}"
OLLAMA_HOST = RAG_OLLAMA_HOST
OLLAMA_PORT = RAG_OLLAMA_PORT
OLLAMA_MODEL = RAG_OLLAMA_MODEL
OLLAMA_QUERY_MODEL = RAG_OLLAMA_QUERY_MODEL
OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL


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


# Prompt for query analysis with scope extraction
QUERY_ANALYSIS_WITH_SCOPES_PROMPT = """Analyze the user's question and extract search information.

User Question: {query}

Provide your analysis in JSON format:
{{
    "enhanced_query": "An optimized search query (50-100 words) capturing the main topic and intent",
    "query_scopes": ["scope1", "scope2", "scope3", ...]
}}

Requirements:
1. ENHANCED_QUERY: Create a focused search query with key concepts and relevant terms
2. QUERY_SCOPES: List 3-8 topic keywords/areas the user is asking about (e.g., "authentication", "JWT", "security")

Output ONLY valid JSON, nothing else."""


# Prompt for LLM-based scope matching
SCOPE_MATCHING_PROMPT = """You are analyzing document relevance based on topic scopes.

The user is searching for information about these topics:
QUERY SCOPES: {query_scopes}

Here are candidate documents with their topic scopes:
{candidates}

For each document, determine if its scopes are semantically relevant to the query scopes.
Consider:
- Exact matches (e.g., "JWT" matches "JWT")
- Semantic equivalence (e.g., "login" matches "authentication")
- Related concepts (e.g., "password" relates to "security")
- Hierarchical relationships (e.g., "OAuth" is a type of "authentication")

Respond in JSON format:
{{
    "relevant_documents": [
        {{"source": "doc_name", "relevance_score": 0.0-1.0, "reason": "brief explanation"}}
    ]
}}

Only include documents with relevance_score >= 0.5.
Output ONLY valid JSON, nothing else."""


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
        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
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


def _analyze_query_with_scopes(query: str) -> tuple:
    """
    Use LLM to analyze the query and extract both enhanced query and topic scopes.

    Args:
        query: The original user query

    Returns:
        Tuple of (enhanced_query, query_scopes)
    """
    import json
    import re

    try:
        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
            temperature=0.1,
            num_ctx=2048,
            num_predict=512,
        )

        prompt = QUERY_ANALYSIS_WITH_SCOPES_PROMPT.format(query=query)
        response = llm.invoke([HumanMessage(content=prompt)])

        content = response.content.strip()

        # Clean up thinking tags
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Try to extract JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1).strip()

        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1:
            content = content[json_start:json_end + 1]

        result = json.loads(content)
        enhanced_query = result.get("enhanced_query", query)
        query_scopes = result.get("query_scopes", [])

        # Validate scopes
        if not isinstance(query_scopes, list):
            query_scopes = []
        query_scopes = [str(s).strip().lower() for s in query_scopes if s]

        logger.info(
            f"Query analysis with scopes: '{query[:50]}...' -> "
            f"query='{enhanced_query[:50]}...', scopes={query_scopes}"
        )

        return enhanced_query, query_scopes

    except Exception as e:
        logger.warning(f"Query analysis with scopes failed: {e}")
        # Fallback to basic query analysis
        enhanced_query = _analyze_query_with_llm(query)
        return enhanced_query, []


def _match_scopes_with_llm(
    query_scopes: List[str],
    candidate_docs: List,
) -> List[str]:
    """
    Use LLM to match query scopes with document scopes.

    Args:
        query_scopes: List of topic scopes from the query.
        candidate_docs: List of Document objects with scopes in metadata.

    Returns:
        List of source names that are relevant based on scope matching.
    """
    import json
    import re

    if not query_scopes or not candidate_docs:
        # If no scopes, return all candidates
        return [doc.metadata.get("source", "") for doc in candidate_docs if doc.metadata.get("source")]

    try:
        # Build candidates string for prompt
        candidates_str = ""
        for doc in candidate_docs:
            source = doc.metadata.get("source", "unknown")
            scopes = doc.metadata.get("scopes", [])
            if not scopes:
                # If no scopes stored, extract from summary
                scopes = ["general"]
            candidates_str += f"- {source}: {scopes}\n"

        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
            temperature=0.1,
            num_ctx=4096,
            num_predict=1024,
        )

        prompt = SCOPE_MATCHING_PROMPT.format(
            query_scopes=query_scopes,
            candidates=candidates_str
        )
        response = llm.invoke([HumanMessage(content=prompt)])

        content = response.content.strip()

        # Clean up thinking tags
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Try to extract JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1).strip()

        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1:
            content = content[json_start:json_end + 1]

        result = json.loads(content)
        relevant_docs = result.get("relevant_documents", [])

        # Extract source names from relevant documents
        relevant_sources = []
        for doc_info in relevant_docs:
            source = doc_info.get("source", "")
            score = doc_info.get("relevance_score", 0)
            if source and score >= 0.5:
                relevant_sources.append(source)
                logger.debug(f"Scope match: {source} (score={score})")

        logger.info(f"LLM scope matching: {len(relevant_sources)}/{len(candidate_docs)} documents matched")
        return relevant_sources

    except Exception as e:
        logger.warning(f"LLM scope matching failed: {e}")
        # Fallback: return all candidates
        return [doc.metadata.get("source", "") for doc in candidate_docs if doc.metadata.get("source")]


def _find_relevant_files_with_scopes(
    enhanced_query: str,
    query_scopes: List[str],
    k: int = 5,
) -> List[str]:
    """
    Find relevant files using vector search + LLM scope matching.

    This implements the enhanced retrieval strategy:
    1. Vector search on file summaries to get candidates
    2. LLM-based scope matching to filter candidates

    Note: If FILE_SUMMARY_ENABLED is False, returns empty list to skip
    file summary search and go directly to chunk search.

    Args:
        enhanced_query: The enhanced search query.
        query_scopes: List of topic scopes from the query.
        k: Maximum number of relevant files to return.

    Returns:
        List of source names that are relevant to the query.
    """
    # Skip file summary search if disabled
    if not FILE_SUMMARY_ENABLED:
        logger.debug("File summary search disabled, skipping file-level filtering")
        return []

    try:
        # Step 1: Get candidate documents via vector search
        candidate_docs = search_file_summaries_with_scopes(enhanced_query, k=k * 2)

        if not candidate_docs:
            logger.info("No file summaries found, will search all chunks")
            return []

        # Step 2: If we have query scopes, use LLM to filter candidates
        if query_scopes:
            relevant_sources = _match_scopes_with_llm(query_scopes, candidate_docs)
        else:
            # No scopes, use all candidates
            relevant_sources = [
                doc.metadata.get("source", "")
                for doc in candidate_docs
                if doc.metadata.get("source")
            ]

        # Limit to k results
        relevant_sources = relevant_sources[:k]

        logger.info(f"Found {len(relevant_sources)} relevant files with scope matching: {relevant_sources}")
        return relevant_sources

    except Exception as e:
        logger.warning(f"Error in find_relevant_files_with_scopes: {e}")
        return []


def _find_relevant_files(query: str, k: int = 5) -> List[str]:
    """
    Find relevant files by searching file summaries first.

    This implements the two-phase retrieval strategy:
    1. Search file summaries to identify relevant documents
    2. Return list of source names that are relevant to the query

    Note: If FILE_SUMMARY_ENABLED is False, returns empty list to skip
    file summary search and go directly to chunk search.

    Args:
        query: The search query.
        k: Maximum number of relevant files to return.

    Returns:
        List of source names that are relevant to the query.
    """
    # Skip file summary search if disabled
    if not FILE_SUMMARY_ENABLED:
        logger.debug("File summary search disabled, skipping file-level filtering")
        return []

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


# GraphRAG query mode: "auto", "local", "global", "hybrid", "naive", or "agent"
# "agent" enables Graph-R1 style iterative reasoning
GRAPHRAG_QUERY_MODE = os.getenv("GRAPHRAG_QUERY_MODE", "auto").lower()


def _get_graphrag_context(query: str) -> str:
    """
    Get additional context from GraphRAG if enabled.

    Supports multiple query modes:
    - auto: Let the system detect the best mode based on query
    - local: Entity-focused retrieval
    - global: Relationship-focused retrieval
    - hybrid: Combined entity and relationship retrieval with graph expansion
    - naive: Simple vector search fallback
    - agent: Graph-R1 style iterative reasoning (multi-step agent loop)

    Args:
        query: The search query.

    Returns:
        Formatted GraphRAG context string, or empty string if disabled/unavailable.
    """
    logger.debug(
        f"[GraphRAG Query] Starting - GRAPHRAG_AVAILABLE={GRAPHRAG_AVAILABLE}, "
        f"GRAPH_RAG_ENABLED={GRAPH_RAG_ENABLED}, mode={GRAPHRAG_QUERY_MODE}"
    )

    if not GRAPHRAG_AVAILABLE:
        logger.debug("[GraphRAG Query] Skipping - GraphRAG module not available")
        return ""

    if not GRAPH_RAG_ENABLED:
        logger.debug("[GraphRAG Query] Skipping - GRAPH_RAG_ENABLED=false in .env")
        return ""

    try:
        import asyncio
        import concurrent.futures

        logger.debug(f"[GraphRAG Query] Initializing GraphRAG for query: '{query[:100]}...'")
        graphrag = GraphRAG()

        # Determine query mode
        mode = GRAPHRAG_QUERY_MODE if GRAPHRAG_QUERY_MODE != "auto" else None

        # Helper function to run the async query
        async def _run_query():
            return await graphrag.query(query, mode=mode)

        # Run async query in sync context - handle case when called from async context (FastAPI)
        try:
            # Check if we're already in an event loop (e.g., FastAPI request)
            asyncio.get_running_loop()
            # We're in an existing event loop - run in a new thread with its own event loop
            logger.debug("[GraphRAG Query] Running in thread pool (existing event loop detected)")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run_query())
                context = future.result()
        except RuntimeError:
            # No running event loop - use asyncio.run directly
            logger.debug("[GraphRAG Query] Running with asyncio.run (no existing event loop)")
            context = asyncio.run(_run_query())

        logger.debug(
            f"[GraphRAG Query] Raw result - mode={context.mode.value}, "
            f"entities={len(context.entities)}, relationships={len(context.relationships)}, "
            f"chunks={len(context.chunks)}, enhanced_query='{context.enhanced_query[:50]}...'"
        )

        # For AGENT mode, the answer is in the chunks (as agent_chunk)
        if context.mode.value == "agent" and context.chunks:
            agent_chunk = context.chunks[0]
            answer = agent_chunk.get("page_content", "")
            if answer:
                logger.info(
                    f"[GraphRAG Query] Agent mode success - "
                    f"steps={agent_chunk.get('metadata', {}).get('steps', 0)}"
                )
                return f"[Agent Reasoning Result]\n{answer}"

        if context.entities or context.relationships:
            formatted = graphrag.format_context(context)
            logger.info(
                f"[GraphRAG Query] Success - mode={context.mode.value}, "
                f"entities={len(context.entities)}, relationships={len(context.relationships)}"
            )
            logger.debug(f"[GraphRAG Query] Formatted context length: {len(formatted)} chars")
            return formatted
        else:
            logger.debug("[GraphRAG Query] No entities or relationships found")

    except Exception as e:
        logger.warning(f"[GraphRAG Query] Failed: {e}", exc_info=True)

    return ""


@tool
def search_documents(query: str) -> str:
    """
    Search the indexed documents for relevant information using enhanced retrieval.

    Enhanced retrieval strategy:
    1. LLM analyzes query and extracts topic scopes
    2. Vector search on file summaries to get candidates
    3. LLM-based scope matching to filter relevant files
    4. Direct chunk search within relevant files
    5. GraphRAG context for entity/relationship enrichment

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Relevant document chunks as a formatted string.
    """
    try:
        # Step 1: Use LLM to analyze query and extract scopes
        enhanced_query, query_scopes = _analyze_query_with_scopes(query)

        # Step 2: Find relevant files using vector search + LLM scope matching
        relevant_sources = _find_relevant_files_with_scopes(
            enhanced_query, query_scopes, k=5
        )

        # Fallback to basic file search if scope-based search returns nothing
        if not relevant_sources:
            relevant_sources = _find_relevant_files(enhanced_query, k=5)

        docs = []

        # Step 3: Direct chunk search (skip chunk summaries - simplified flow)
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

        # Step 4: Get GraphRAG context (entities and relationships)
        graphrag_context = _get_graphrag_context(enhanced_query)
        if graphrag_context:
            results.append(f"[Knowledge Graph Context]\n{graphrag_context}")

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
            f"Query scopes: {query_scopes} | "
            f"Relevant sources: {relevant_sources} | "
            f"Found {len(docs)} chunks from sources: {seen_sources} | "
            f"GraphRAG context: {'yes' if graphrag_context else 'no'}"
        )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error searching documents: {str(e)}"


# Define tools
tools = [search_documents]


def create_llm():
    """Create the LLM instance based on configured backend."""
    llm_service = get_llm_service()
    return llm_service.get_chat_model(
        temperature=0.2,
        num_ctx=8192,
    )


def is_vllm_backend() -> bool:
    """Check if using vLLM backend (which doesn't support tool calling without special flags)."""
    from .llm_service import RAG_LLM_BACKEND
    return RAG_LLM_BACKEND == "vllm"


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

    # For vLLM backend: use RAG-first approach (search then respond)
    # vLLM doesn't support tool calling without --enable-auto-tool-choice flag
    if is_vllm_backend():
        # Extract the user's query from the last human message
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if user_query:
            # Always search for context first
            context = search_documents.invoke({"query": user_query})

            # Add context to the system message
            context_message = f"\n\nRelevant document context:\n{context}"
            augmented_messages = list(messages)
            if isinstance(augmented_messages[0], SystemMessage):
                augmented_messages[0] = SystemMessage(
                    content=augmented_messages[0].content + context_message
                )

            response = llm.invoke(augmented_messages)
        else:
            response = llm.invoke(messages)
    else:
        # For Ollama and other backends: use tool binding
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

