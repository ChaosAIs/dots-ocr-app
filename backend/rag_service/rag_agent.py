"""
LangGraph-based RAG Agent with document search tool.
Supports multiple LLM backends: Ollama and vLLM.
"""

import os
import logging
import json
from typing import Annotated, TypedDict, List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .vectorstore import get_retriever, get_retriever_with_sources
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

IMPORTANT INSTRUCTIONS:
- Provide ONLY your final answer to the user's question
- Do NOT include any JSON objects, XML tags, or structured data in your response
- Do NOT echo or repeat the context format
- Do NOT include <query>, <answer>, <think>, or any other XML tags
- Do NOT include any metadata or processing information
- Provide a direct, natural language answer only

Always search for relevant information before answering questions about the documents."""


# Prompt for query analysis - concise output to avoid repetition
QUERY_ANALYSIS_PROMPT = """Analyze the user's question and create an optimized search query.

User Question: {query}

Create a concise, focused search query (50-100 words max) that:
- Captures the main topic and user intent
- Includes key concepts and relevant terms
- Avoids repetition

Output ONLY the search query text, nothing else."""





def _analyze_query_with_llm(query: str) -> Dict[str, Any]:
    """
    Use LLM to analyze and enhance the user's query, extracting both enhanced query text
    and structured metadata for document routing.

    This function performs a SINGLE LLM call to:
    1. Generate an enhanced query optimized for vector search
    2. Extract structured metadata (entities, topics, document types, intent)
    3. Enable intelligent document routing based on metadata matching

    Args:
        query: The original user query

    Returns:
        Dict with:
        - enhanced_query: str - Enhanced query text for vector search
        - metadata: dict - Structured metadata for document routing
            - entities: List[str] - Named entities (people, orgs, products, tech)
            - topics: List[str] - Subject areas and domains
            - document_type_hints: List[str] - Likely document types
            - intent: str - Brief description of user intent
    """
    try:
        # Import the new combined prompt
        from .graph_rag.prompts import QUERY_ENHANCEMENT_WITH_METADATA_PROMPT

        # Use smaller, faster model for query analysis
        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
            temperature=0.1,  # Low temperature for consistent analysis
            num_ctx=4096,     # Increased context for JSON output
            num_predict=512,  # Increased for structured output
        )

        prompt = QUERY_ENHANCEMENT_WITH_METADATA_PROMPT.format(query=query)
        response = llm.invoke([HumanMessage(content=prompt)])

        response_text = response.content.strip()

        # Clean up any thinking tags if present
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if len(lines) > 2:
                response_text = "\n".join(lines[1:-1])

        # Parse JSON response
        try:
            result = json.loads(response_text)

            # Validate required fields
            enhanced_query = result.get("enhanced_query", query)
            metadata = {
                "entities": result.get("entities", []),
                "topics": result.get("topics", []),
                "document_type_hints": result.get("document_type_hints", []),
                "intent": result.get("intent", ""),
            }

            logger.info(f"[Query Analysis] Original: '{query[:50]}...'")
            logger.info(f"[Query Analysis] Enhanced: '{enhanced_query[:100]}...'")
            logger.info(f"[Query Analysis] Entities: {metadata['entities']}")
            logger.info(f"[Query Analysis] Topics: {metadata['topics']}")

            return {
                "enhanced_query": enhanced_query,
                "metadata": metadata,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response, using fallback: {e}")
            logger.debug(f"Response text: {response_text[:200]}")
            # Fallback: use response as enhanced query, empty metadata
            return {
                "enhanced_query": response_text[:500] if response_text else query,
                "metadata": {
                    "entities": [],
                    "topics": [],
                    "document_type_hints": [],
                    "intent": "",
                }
            }

    except Exception as e:
        logger.warning(f"Query analysis failed, using original query: {e}")
        # Fallback: return original query with empty metadata
        return {
            "enhanced_query": query,
            "metadata": {
                "entities": [],
                "topics": [],
                "document_type_hints": [],
                "intent": "",
            }
        }





# GraphRAG query mode: "auto", "local", "global", or "hybrid"
# Following Graph-R1 paper design with iterative reasoning built-in
GRAPHRAG_QUERY_MODE = os.getenv("GRAPHRAG_QUERY_MODE", "auto").lower()


def _get_graphrag_context(query: str) -> str:
    """
    Get additional context from GraphRAG if enabled.

    Following Graph-R1 paper design:
    - Combines graph-based retrieval with Qdrant vector search
    - Supports LOCAL, GLOBAL, HYBRID query modes
    - Iterative reasoning controlled by max_steps parameter

    Supported query modes:
    - auto: Let the system detect the best mode based on query (recommended)
    - local: Entity-focused retrieval
    - global: Relationship-focused retrieval
    - hybrid: Combined entity and relationship retrieval with graph expansion (default)

    Args:
        query: The search query.

    Returns:
        Formatted GraphRAG context string combining graph and vector results,
        or empty string if disabled/unavailable.
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

        # Format context combining graph results and vector search results
        if context.entities or context.relationships or context.chunks:
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
    Search the indexed documents for relevant information using intelligent routing.

    Retrieval strategy:
    1. LLM analyzes query and extracts metadata (entities, topics, intent)
    2. Route to relevant documents based on metadata matching
    3. Vector search on filtered document chunks
    4. GraphRAG context for entity/relationship enrichment

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Relevant document chunks as a formatted string.
    """
    try:
        # Step 1: Enhance query and extract metadata (SINGLE LLM CALL)
        query_analysis = _analyze_query_with_llm(query)
        enhanced_query = query_analysis["enhanced_query"]
        query_metadata = query_analysis["metadata"]

        logger.info(f"[Search] Original query: '{query[:50]}...'")
        logger.info(f"[Search] Enhanced query: '{enhanced_query[:100]}...'")
        logger.info(f"[Search] Extracted entities: {query_metadata.get('entities', [])}")
        logger.info(f"[Search] Extracted topics: {query_metadata.get('topics', [])}")

        # Step 2: Route to relevant documents based on metadata
        from .document_router import DocumentRouter
        router = DocumentRouter()
        relevant_sources = router.route_query(query_metadata)

        # Step 3: Create retriever with optional source filtering
        if relevant_sources:
            retriever = get_retriever_with_sources(
                k=15,
                fetch_k=50,
                source_names=relevant_sources
            )
        else:
            retriever = get_retriever(k=15, fetch_k=50)

        # Step 4: Search with enhanced query
        docs = retriever.invoke(enhanced_query)

        if not docs:
            return "No relevant documents found for this query."

        # Format results - include all unique sources for better coverage
        results = []
        seen_sources = set()

        # Step 5: Get GraphRAG context (entities and relationships)
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
            f"[Search] Found {len(docs)} chunks from {len(seen_sources)} sources: {seen_sources} | "
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


def _clean_llm_response(response_content: str) -> str:
    """
    Clean up LLM response to remove any JSON, XML, or structured data artifacts.

    Args:
        response_content: Raw LLM response content

    Returns:
        Cleaned response with only the natural language answer
    """
    import re

    # Remove JSON objects (anything between { and })
    cleaned = re.sub(r'\{[^}]*"[^"]*"[^}]*\}', '', response_content)

    # Remove XML tags and their content if they appear at the start
    cleaned = re.sub(r'^<[^>]+>.*?</[^>]+>\s*', '', cleaned, flags=re.DOTALL)

    # Remove any remaining XML tags
    cleaned = re.sub(r'<[^>]+>.*?</[^>]+>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)

    # Remove thinking tags content
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)

    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned


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

            # Clean up the response to remove any artifacts
            if hasattr(response, 'content'):
                cleaned_content = _clean_llm_response(response.content)
                response.content = cleaned_content
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
    # We need to filter events to only stream the final agent response,
    # not intermediate LLM calls from tools (query analysis, GraphRAG, etc.)

    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
    ):
        # Only stream events from the main chat model in the "agent" node
        if event["event"] == "on_chat_model_stream":
            # Check the parent_ids to see if this is nested in a tool call
            # Tool calls (search_documents, etc.) will have more parent IDs
            # The main agent response will have fewer parent IDs (graph + agent node only)
            parent_ids = event.get("parent_ids", [])
            name = event.get("name", "")

            # Debug logging to understand the event structure
            logger.debug(
                f"[Stream Event] name={name}, parent_ids_count={len(parent_ids)}, "
                f"event={event['event']}"
            )

            # If there are multiple parent IDs (>2), this is likely a nested call from a tool
            # Main agent has at most 2 levels: graph root + agent node
            if len(parent_ids) <= 2:
                content = event["data"]["chunk"].content
                if content:
                    logger.debug(f"[Stream] Yielding content from {name} (parent_ids={len(parent_ids)})")
                    yield content
            else:
                # Skip tool-generated content
                logger.debug(f"[Stream] Skipping content from {name} (parent_ids={len(parent_ids)})")


