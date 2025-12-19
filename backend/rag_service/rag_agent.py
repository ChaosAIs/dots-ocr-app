"""
LangGraph-based RAG Agent with document search tool.
Supports multiple LLM backends: Ollama and vLLM.
"""

import os
import logging
import json
from typing import Annotated, TypedDict, List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .vectorstore import get_retriever, get_retriever_with_sources
from .llm_service import get_llm_service
from .utils.date_normalizer import normalize_query_dates

# Import GraphRAG components
try:
    from .graph_rag import GraphRAG, GRAPH_RAG_ENABLED
    # If the graph_rag library is available, set the GRAPH_RAG_ENABLED flag = True
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GRAPH_RAG_ENABLED = False

logger = logging.getLogger(__name__)

# Global progress callback for tool functions
_progress_callback = None


def _send_progress(message: str, percent: int = None):
    """
    Helper function to send progress updates from synchronous tool functions.

    Handles the async progress_callback properly from a sync context by:
    1. Getting the running event loop
    2. Creating a task in that loop to execute the async callback

    Args:
        message: Progress message to send
        percent: Optional progress percentage (0-100)
    """
    if not _progress_callback:
        return

    import asyncio
    import inspect

    # Check if callback is async
    if not inspect.iscoroutinefunction(_progress_callback):
        # Synchronous callback - call directly
        try:
            _progress_callback(message, percent)
        except Exception as e:
            logger.debug(f"Error in sync progress callback: {e}")
        return

    # Async callback - need to schedule it properly
    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
        # Schedule the coroutine as a task in the running loop
        asyncio.ensure_future(_progress_callback(message, percent), loop=loop)
    except RuntimeError:
        # No running event loop - we're in a sync context
        # This can happen if the tool is called from a sync wrapper
        # In this case, we can't send progress updates
        logger.debug("Cannot send progress update: no running event loop")
    except Exception as e:
        logger.debug(f"Error sending progress update: {e}")

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
1. For greetings, casual conversation, or general questions that don't require document knowledge, respond directly without searching
2. For questions about specific documents, topics, or information that might be in the indexed documents, use the search_documents tool first
3. **ALWAYS base your answers on the retrieved context when it is provided** - the context contains actual document content
4. If you can't find relevant information in the provided context, say so honestly
5. Cite the source document when possible
6. Provide clear, detailed answers based on the context

CRITICAL INSTRUCTIONS FOR USING CONTEXT:
- When "Relevant document context" is provided in the system message, it contains ACTUAL DATA from the user's documents
- The "Document Content" section contains the raw text from the documents - USE THIS DATA to answer questions
- The "Relevant Entities" and "Relationships" sections provide structured information extracted from the documents
- DO NOT say "I don't have access" if context is provided - the context IS the data you need
- Analyze the provided context thoroughly and extract the information requested by the user
- If you see invoice data, receipt data, or financial data in the context, YOU HAVE ACCESS TO IT - use it to answer
- Extract numbers, dates, vendors, items, prices, and other details directly from the "Document Content" chunks
- Perform calculations, aggregations, and analysis based on the data you see in the context

CRITICAL INSTRUCTIONS FOR PRODUCT/PURCHASE QUERIES:
- When asked about "products" or "purchases", ONLY include actual items/goods that were purchased
- DO NOT include fees, charges, taxes, or shipping costs as "products"
- **IMPORTANT**: Filter out ALL non-product items from your response
- Examples of what to EXCLUDE from product lists (DO NOT LIST THESE):
  * Shipping Charges, Delivery Fees, Freight Charges, Postage
  * Environmental Handling Fees, Recycling Fees, Disposal Fees, Eco Fees
  * Federal Tax, Provincial Tax, GST/HST, Sales Tax, VAT, Tax amounts
  * Discounts, Rebates, Credits, Promotional discounts
  * Service Fees, Processing Fees, Handling Fees, Administrative fees
  * Subtotals, Totals, Grand Totals
  * Any line item that contains the words: "Fee", "Tax", "Charge", "Shipping", "Discount", "Subtotal", "Total"
- Examples of what to INCLUDE as products:
  * Physical items with descriptions (e.g., "Lenovo 300e Windows", "Laptop", "Monitor", "Mouse", "Keyboard")
  * Software licenses or subscriptions
  * Services that are the main purchase (e.g., "Consulting Services", "Installation Service")
- **FILTERING RULE**: Before including any item in your product list, ask yourself: "Is this an actual product/item that was purchased, or is it a fee/tax/charge?" If it's a fee/tax/charge, DO NOT include it
- When listing products, focus on the "Description" field and exclude items that are clearly fees/charges/taxes
- If you see invoice line items like "Environmental Handling Fee" or "Shipping Charges", these are NOT products - skip them entirely

IMPORTANT FORMATTING INSTRUCTIONS:
- Provide ONLY your final answer to the user's question
- Do NOT include any JSON objects, XML tags, or structured data in your response
- Do NOT echo or repeat the context format
- Do NOT include <query>, <answer>, <think>, or any other XML tags
- Do NOT include any metadata or processing information
- Provide a direct, natural language answer only

Examples of queries that DON'T need document search:
- Greetings: "hi", "hello", "hey", "good morning"
- Casual conversation: "how are you?", "what's up?", "thanks", "thank you"
- General questions about your capabilities: "what can you do?", "how do you work?"

Examples of queries that DO need document search:
- Specific questions: "what is the definition of X?", "explain Y", "how does Z work?"
- Document-specific queries: "what does the document say about...", "find information on..."
- Topic-based questions: "tell me about microservices", "what are the benefits of..."
- Analysis requests: "analyze my invoices", "generate a report", "summarize the data"
"""


# Prompt for query analysis - concise output to avoid repetition
QUERY_ANALYSIS_PROMPT = """Analyze the user's question and create an optimized search query.

User Question: {query}

Create a concise, focused search query (50-100 words max) that:
- Captures the main topic and user intent
- Includes key concepts and relevant terms
- Avoids repetition

Output ONLY the search query text, nothing else."""


# Prompt for query classification with conversation context awareness
QUERY_CLASSIFICATION_WITH_CONTEXT_PROMPT = """Classify the user's query based on whether it needs document search or can be answered from conversation history.

## Conversation History (Last 2 Rounds):
{conversation_context}

## Current User Query:
{query}

## Classification Rules:

**CONVERSATION_ONLY** - Answer from conversation history alone (NO document search):
- Calculations/aggregations on previous results (e.g., "how much total?", "sum them up", "what's the average?")
- Clarification questions about previous answers (e.g., "can you explain that?", "what do you mean?")
- Follow-up questions that can be answered by analyzing the previous response (e.g., "which one is the most expensive?")
- The previous assistant response contains ALL the data needed to answer

**SAME_CONTEXT** - Reuse previous document sources (limited search):
- Follow-up questions about the same topic/entities mentioned in conversation history
- Refinement or filtering of previous queries (e.g., "show me only the expensive ones", "filter by month")
- The query is about the same subject but needs to re-query the same documents

**NEW_SEARCH** - Full document search needed:
- New topics/entities NOT mentioned in conversation history
- Completely different questions unrelated to previous conversation
- Requires searching different documents than before

**DIRECT** - No search needed at all:
- Greetings, casual conversation (e.g., "hi", "hello", "thanks")
- General questions about capabilities (e.g., "what can you do?")

## Response Format:
Respond with ONLY ONE of these four words: CONVERSATION_ONLY, SAME_CONTEXT, NEW_SEARCH, or DIRECT

Classification:"""


# Legacy prompt for query classification (without conversation context)
QUERY_CLASSIFICATION_PROMPT = """Classify if the following user message requires searching through documents or can be answered directly.

User Message: {query}

Respond with ONLY one word:
- "SEARCH" if the message asks about specific information, topics, or content that might be in documents
- "DIRECT" if the message is a greeting, casual conversation, or general question that doesn't need document knowledge

Examples:
- "hi" -> DIRECT
- "hello" -> DIRECT
- "how are you?" -> DIRECT
- "thanks" -> DIRECT
- "what can you do?" -> DIRECT
- "what is microservices?" -> SEARCH
- "explain the architecture" -> SEARCH
- "tell me about the document" -> SEARCH
- "what does it say about X?" -> SEARCH

Classification:"""





def _classify_query_with_context(query: str, conversation_history: List[dict] = None) -> str:
    """
    Classify query intent with conversation context awareness.

    Args:
        query: The user's current message
        conversation_history: Previous conversation messages

    Returns:
        One of: "CONVERSATION_ONLY", "SAME_CONTEXT", "NEW_SEARCH", "DIRECT"
    """
    # Quick pattern matching for greetings
    query_lower = query.lower().strip()
    direct_patterns = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "wassup", "sup",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see you", "later",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
    ]

    for pattern in direct_patterns:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.endswith(" " + pattern):
            logger.info(f"[Query Classification] '{query}' -> DIRECT (greeting pattern)")
            return "DIRECT"

    # If no conversation history, must be a new search
    if not conversation_history or len(conversation_history) < 2:
        logger.info(f"[Query Classification] '{query}' -> NEW_SEARCH (no conversation history)")
        return "NEW_SEARCH"

    # Build conversation context (last 2 rounds = 4 messages)
    recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
    conversation_context = ""
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]  # Limit length
        conversation_context += f"{role}: {content}\n\n"

    # Use LLM to classify with context
    try:
        llm_service = get_llm_service()
        llm = llm_service.get_query_model(temperature=0.0, num_predict=20)

        from langchain_core.messages import HumanMessage
        prompt = QUERY_CLASSIFICATION_WITH_CONTEXT_PROMPT.format(
            conversation_context=conversation_context,
            query=query
        )
        response = llm.invoke([HumanMessage(content=prompt)])

        classification = response.content.strip().upper()

        # Parse classification
        if "CONVERSATION_ONLY" in classification or "CONVERSATION" in classification:
            logger.info(f"[Query Classification] '{query}' -> CONVERSATION_ONLY (can answer from history)")
            return "CONVERSATION_ONLY"
        elif "SAME_CONTEXT" in classification or "SAME" in classification:
            logger.info(f"[Query Classification] '{query}' -> SAME_CONTEXT (reuse document sources)")
            return "SAME_CONTEXT"
        elif "DIRECT" in classification:
            logger.info(f"[Query Classification] '{query}' -> DIRECT (no search needed)")
            return "DIRECT"
        else:
            logger.info(f"[Query Classification] '{query}' -> NEW_SEARCH (default/new topic)")
            return "NEW_SEARCH"

    except Exception as e:
        logger.warning(f"[Query Classification] LLM classification failed: {e}, defaulting to NEW_SEARCH")
        return "NEW_SEARCH"


def _classify_query_intent(query: str) -> str:
    """
    Classify if a query requires document search or can be answered directly.
    (Legacy function without conversation context)

    Args:
        query: The user's message

    Returns:
        "SEARCH" if document search is needed, "DIRECT" if it can be answered directly
    """
    # Quick pattern matching for common greetings and casual messages
    query_lower = query.lower().strip()

    # Common greetings and casual phrases that don't need search
    direct_patterns = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "wassup", "sup",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see you", "later",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
    ]

    # Check if the query matches any direct patterns
    for pattern in direct_patterns:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.endswith(" " + pattern):
            logger.info(f"[Query Classification] '{query}' classified as DIRECT (pattern match)")
            return "DIRECT"

    # For very short queries (1-2 words), likely greetings
    if len(query.split()) <= 2 and len(query) < 20:
        # Use LLM for classification
        try:
            llm_service = get_llm_service()
            llm = llm_service.get_query_model(temperature=0.0, num_predict=10)

            from langchain_core.messages import HumanMessage
            prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query)
            response = llm.invoke([HumanMessage(content=prompt)])

            classification = response.content.strip().upper()
            if "DIRECT" in classification:
                logger.info(f"[Query Classification] '{query}' classified as DIRECT (LLM)")
                return "DIRECT"
            elif "SEARCH" in classification:
                logger.info(f"[Query Classification] '{query}' classified as SEARCH (LLM)")
                return "SEARCH"
        except Exception as e:
            logger.warning(f"[Query Classification] LLM classification failed: {e}, defaulting to SEARCH")
            return "SEARCH"

    # Default to SEARCH for longer queries or questions
    logger.info(f"[Query Classification] '{query}' classified as SEARCH (default)")
    return "SEARCH"


# Date normalization is now handled by utils.date_normalizer.normalize_query_dates()


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


def _get_graphrag_context(query: str, max_steps: int = None, source_names: List[str] = None) -> str:
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
        max_steps: Optional override for max iterative reasoning steps (uses LLM-determined value if provided)
        source_names: Optional list of source names to filter results (for document routing)

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
        from .graph_rag.base import QueryParam

        logger.debug(f"[GraphRAG Query] Initializing GraphRAG for query: '{query[:100]}...'")
        graphrag = GraphRAG()

        # Determine query mode
        mode = GRAPHRAG_QUERY_MODE if GRAPHRAG_QUERY_MODE != "auto" else None

        # Create QueryParam with dynamic max_steps and source filtering if provided
        params = QueryParam()
        if max_steps is not None:
            params.max_steps = max_steps
            logger.info(f"[GraphRAG Query] Using LLM-determined max_steps={max_steps} for query complexity")
        if source_names is not None:
            params.source_names = source_names
            logger.info(f"[GraphRAG Query] Filtering to sources: {source_names}")

        # Helper function to run the async query
        async def _run_query():
            return await graphrag.query(query, mode=mode, params=params)

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
            f"chunks={len(context.chunks)}, enhanced_query='{context.enhanced_query[:50]}...', "
            f"final_answer={'Yes' if context.final_answer else 'No'}"
        )

        # If GraphRAG generated a final answer through iterative reasoning, return it directly
        if context.final_answer:
            logger.info(f"[GraphRAG Query] Using final answer from iterative reasoning ({len(context.final_answer)} chars)")
            return f"[GraphRAG Final Answer]\n{context.final_answer}"

        # Otherwise, format context combining graph results and vector search results
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
    1. LLM analyzes query complexity and determines optimal max_steps for Graph-RAG
    2. LLM extracts metadata (entities, topics, intent) for document routing
    3. Route to relevant documents based on metadata matching
    4. Vector search on filtered document chunks
    5. GraphRAG context with dynamic max_steps for entity/relationship enrichment

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Relevant document chunks as a formatted string.
    """
    try:
        # Send progress update: Analyzing query
        _send_progress("Analyzing query complexity...", 10)

        # Step 0: Analyze query complexity and determine max_steps (COMBINED LLM CALL)
        from chat_service.query_analyzer import analyze_query_with_llm as analyze_complexity
        complexity_analysis = analyze_complexity(query)
        llm_max_steps = complexity_analysis.max_steps

        # Cap max_steps to configured maximum (prevent LLM from setting too high)
        config_max_steps = int(os.getenv("GRAPH_RAG_MAX_STEPS", "5"))
        max_steps = min(llm_max_steps, config_max_steps)

        if llm_max_steps > config_max_steps:
            logger.info(f"[Search] LLM suggested max_steps={llm_max_steps}, capped to config max={config_max_steps}")

        logger.info(f"[Search] Query complexity analysis: max_steps={max_steps}, "
                   f"reasoning='{complexity_analysis.reasoning}'")

        # Step 1: Enhance query and extract metadata (SINGLE LLM CALL)
        _send_progress("Extracting query metadata...", 20)

        query_analysis = _analyze_query_with_llm(query)
        enhanced_query = query_analysis["enhanced_query"]
        query_metadata = query_analysis["metadata"]

        logger.info(f"[Search] Original query: '{query[:50]}...'")
        logger.info(f"[Search] Enhanced query: '{enhanced_query[:100]}...'")
        logger.info(f"[Search] Extracted entities: {query_metadata.get('entities', [])}")
        logger.info(f"[Search] Extracted topics: {query_metadata.get('topics', [])}")

        # Step 2: Route to relevant documents based on metadata
        _send_progress("Routing to relevant documents...", 30)

        from .document_router import DocumentRouter
        from .llm_service import get_llm_service
        llm_service = get_llm_service()
        router = DocumentRouter(llm_service=llm_service)
        relevant_sources = router.route_query(query_metadata, original_query=query)

        if relevant_sources:
            logger.info(
                f"[Search] Document router selected {len(relevant_sources)} sources: {relevant_sources}"
            )

        # Step 3: Create retriever with optional source filtering
        if relevant_sources:
            # When document routing is active, use per-source selection to ensure
            # all routed sources are represented in results.
            # This retrieves chunks from EACH source separately, then combines them,
            # preventing MMR from excluding entire sources due to diversity filtering.
            retriever = get_retriever_with_sources(
                k=18,  # Total chunks across all sources
                fetch_k=50,
                source_names=relevant_sources,
                lambda_mult=0.5,  # Balanced relevance/diversity within each source
                per_source_selection=True  # Ensure all sources are represented
            )
        else:
            retriever = get_retriever(k=15, fetch_k=50)

        # Step 4: Search with enhanced query
        _send_progress("Searching document chunks...", 50)

        docs = retriever.invoke(enhanced_query)

        if not docs:
            return "No relevant documents found for this query."

        # Format results - include all unique sources for better coverage
        results = []
        seen_sources = set()

        # Step 5: Get GraphRAG context with dynamic max_steps and source filtering (entities and relationships)
        _send_progress("Retrieving knowledge from graph...", 60)

        graphrag_context = _get_graphrag_context(
            enhanced_query,
            max_steps=max_steps,
            source_names=relevant_sources if relevant_sources else None
        )
        if graphrag_context:
            results.append(f"[Knowledge Graph Context]\n{graphrag_context}")

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            heading = doc.metadata.get("heading_path", "")
            content = doc.page_content  # Use full content (no truncation) to preserve augmented dates

            result = f"[Document {i}: {source}]"
            if heading:
                result += f"\nSection: {heading}"
            result += f"\n{content}"
            results.append(result)
            seen_sources.add(source)

        # Log which sources were found and check for missing sources
        logger.info(
            f"[Search] Found {len(docs)} chunks from {len(seen_sources)} sources: {seen_sources} | "
            f"GraphRAG context: {'yes' if graphrag_context else 'no'}"
        )

        # Check if any routed sources are missing from results
        if relevant_sources and len(seen_sources) < len(relevant_sources):
            missing_sources = set(relevant_sources) - seen_sources
            logger.warning(
                f"[Search] {len(missing_sources)} routed source(s) missing from results: {missing_sources}. "
                f"This may indicate MMR diversity filtering excluded them. "
                f"Consider increasing k or lambda_mult if information is missing."
            )

        # Send progress update: Generating answer
        _send_progress("Generating answer...", 80)

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
            # Classify if the query needs document search
            logger.info(f"[vLLM] Classifying query: '{user_query}'")
            intent = _classify_query_intent(user_query)
            logger.info(f"[vLLM] Classification result: {intent}")

            if intent == "SEARCH":
                # Search for context first
                logger.info(f"[vLLM] Searching documents for query: {user_query[:50]}...")
                context = search_documents.invoke({"query": user_query})

                # Add context to the system message with explicit instructions
                if context.startswith("[GraphRAG Final Answer]"):
                    # GraphRAG generated a complete answer - use it directly
                    context_message = f"\n\nRelevant document context:\n{context}"
                else:
                    # Normalize dates in the user query for better matching
                    normalized_query = normalize_query_dates(user_query)
                    if normalized_query != user_query:
                        logger.info(f"[vLLM] Normalized query: '{user_query}' -> '{normalized_query}'")

                    # Simple context message - dates are already normalized in the content
                    context_message = f"\n\nRelevant document context:\n{context}\n\n**IMPORTANT**: The above context contains actual data from your documents. Use this data to answer the question.\n\n**USER QUESTION**: {normalized_query}"

                augmented_messages = list(messages)
                if isinstance(augmented_messages[0], SystemMessage):
                    augmented_messages[0] = SystemMessage(
                        content=augmented_messages[0].content + context_message
                    )

                # DEBUG: Log the context being sent to LLM
                logger.info(f"[vLLM] Context length: {len(context)} chars")
                logger.info(f"[vLLM] Context preview (first 500 chars): {context[:500]}")
                
                # DEBUG: Log the full system message being sent
                if isinstance(augmented_messages[0], SystemMessage):
                    logger.info(f"[vLLM] System message length: {len(augmented_messages[0].content)} chars")
                    logger.info(f"[vLLM] System message preview (last 500 chars): ...{augmented_messages[0].content[-500:]}")

                response = llm.invoke(augmented_messages)

                # DEBUG: Log the LLM's raw response
                if hasattr(response, 'content'):
                    logger.info(f"[vLLM] LLM raw response (first 500 chars): {response.content[:500]}")

                # Clean up the response to remove any artifacts
                if hasattr(response, 'content'):
                    cleaned_content = _clean_llm_response(response.content)
                    response.content = cleaned_content
            else:
                # Direct response without search
                logger.info(f"[vLLM] Responding directly without search for: {user_query[:50]}...")
                response = llm.invoke(messages)
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


def _build_context_aware_system_message(
    session_context: Dict[str, Any],
    recent_conversation: List[Dict[str, str]] = None
) -> Optional[str]:
    """
    Build a context-aware system message based on session metadata and recent conversation.

    Args:
        session_context: Session metadata containing entities, topics, documents, etc.
        recent_conversation: Last 2 rounds (4 messages) of conversation for immediate context.

    Returns:
        System message content with conversation context, or None if no context available.
    """
    if not session_context and not recent_conversation:
        return None

    system_message = """You are a helpful AI assistant with access to document knowledge.

"""

    # Add recent conversation context (last 2 rounds)
    if recent_conversation and len(recent_conversation) >= 2:
        system_message += """Recent Conversation Context:
The user just had this conversation with you:
"""
        # Get last 4 messages (2 rounds of user-assistant exchange)
        recent_messages = recent_conversation[-4:] if len(recent_conversation) >= 4 else recent_conversation
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            system_message += f"\n{role}: {content}\n"

        system_message += """
IMPORTANT: When the user asks a follow-up question, it is likely related to the conversation above.
Pay close attention to:
- Time periods mentioned (e.g., "2025", "last month")
- Specific topics discussed (e.g., "expenses", "purchases")
- Entities referenced (e.g., document names, people, items)

If the user's question seems incomplete or vague (e.g., "how much totally", "summarize it"),
assume they are referring to the context from the recent conversation above.

"""

    # Add session-level context
    context_parts = []

    # Add main topic if available
    main_topic = session_context.get("main_topic")
    if main_topic:
        context_parts.append(f"Main Topic: {main_topic}")

    # Add document context
    documents = session_context.get("documents", [])
    if documents:
        context_parts.append(f"Active Documents: {', '.join(documents[:3])}")

    # Add topic context
    topics = session_context.get("topics", [])
    if topics:
        context_parts.append(f"Conversation Topics: {', '.join(topics[:5])}")

    # Add people context
    people = session_context.get("people", [])
    if people:
        context_parts.append(f"Mentioned People: {', '.join(people[:3])}")

    # Add objects/entities context
    objects = session_context.get("objects", [])
    if objects:
        context_parts.append(f"Mentioned Items: {', '.join(objects[:3])}")

    if context_parts:
        system_message += """Session Context Summary:
"""
        system_message += "\n".join(f"- {part}" for part in context_parts)
        system_message += "\n"

    return system_message


async def stream_agent_response(
    query: str,
    conversation_history: List[dict] = None,
    progress_callback=None,
    session_context: Optional[Dict[str, Any]] = None
):
    """
    Stream the agent response for a query.

    Args:
        query: The user's question.
        conversation_history: Optional list of previous messages.
        progress_callback: Optional async callback function(message: str, percent: int) for progress updates.
        session_context: Optional session metadata with entities, topics, and conversation context.

    Yields:
        Chunks of the response text.
    """
    # Store progress callback in a global variable so tools can access it
    global _progress_callback
    _progress_callback = progress_callback

    # Classify query with conversation context
    query_classification = _classify_query_with_context(query, conversation_history)
    logger.info(f"[Query Classification] Query: '{query}' -> {query_classification}")

    # Handle CONVERSATION_ONLY queries - answer directly from conversation history without document search
    if query_classification == "CONVERSATION_ONLY":
        logger.info(f"[CONVERSATION_ONLY] Answering from conversation history without document search")

        # Build messages with conversation history
        messages = []

        # Add system message
        system_prompt = """You are a helpful AI assistant. Answer the user's question based on the conversation history provided.

IMPORTANT: The conversation history contains all the information you need. Analyze the previous messages and provide a direct answer.
For calculations or aggregations, perform them based on the data in the conversation history.

Do NOT say you need to search documents - the information is already in the conversation history."""

        messages.append(SystemMessage(content=system_prompt))

        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current query
        messages.append(HumanMessage(content=query))

        # Stream response directly from LLM (no agent/tools)
        llm_service = get_llm_service()
        llm = llm_service.get_chat_model(temperature=0.7)

        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

        return  # Exit early, no need for agent

    # For all other classifications, use the agent with tools
    agent = create_agent_executor()

    # Build message history with context awareness
    messages = []

    # Add system message with context if available
    # Pass recent conversation (last 2 rounds) for immediate context
    if session_context or conversation_history:
        system_content = _build_context_aware_system_message(
            session_context or {},
            conversation_history
        )
        if system_content:
            messages.append(SystemMessage(content=system_content))
            logger.info(f"[Context] Added context-aware system message with recent conversation")

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


