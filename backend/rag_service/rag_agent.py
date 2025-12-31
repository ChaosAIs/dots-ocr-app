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

from .vectorstore import unified_vector_search, UnifiedSearchResult
from .llm_service import get_llm_service
from .utils.date_normalizer import normalize_query_dates
from .context_builder import build_token_aware_context, get_context_builder

# Set up early logger for GraphRAG import debugging
_early_logger = logging.getLogger(__name__)

# Import GraphRAG components
try:
    from .graph_rag import GraphRAG, GRAPH_RAG_QUERY_ENABLED
    # If the graph_rag library is available, set the GRAPHRAG_AVAILABLE flag = True
    GRAPHRAG_AVAILABLE = True
    _early_logger.info(f"[RAG Agent INIT] GraphRAG imported successfully, GRAPH_RAG_QUERY_ENABLED={GRAPH_RAG_QUERY_ENABLED}")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    GRAPH_RAG_QUERY_ENABLED = False
    # Log import failure for debugging
    import traceback
    error_msg = f"[RAG Agent INIT] GraphRAG import failed: {e}"
    traceback_msg = traceback.format_exc()
    _early_logger.error(error_msg)
    _early_logger.error(f"[RAG Agent INIT] Traceback:\n{traceback_msg}")
    # Also print to ensure visibility
    print(error_msg)
    print(f"[RAG Agent INIT] Traceback:\n{traceback_msg}")
except Exception as e:
    # Catch any other exception type
    GRAPHRAG_AVAILABLE = False
    GRAPH_RAG_QUERY_ENABLED = False
    import traceback
    error_msg = f"[RAG Agent INIT] GraphRAG import failed with {type(e).__name__}: {e}"
    traceback_msg = traceback.format_exc()
    _early_logger.error(error_msg)
    _early_logger.error(f"[RAG Agent INIT] Traceback:\n{traceback_msg}")
    print(error_msg)
    print(f"[RAG Agent INIT] Traceback:\n{traceback_msg}")

logger = logging.getLogger(__name__)

# Log GraphRAG status at module load time
logger.info(f"[RAG Agent INIT] Final status: GRAPHRAG_AVAILABLE={GRAPHRAG_AVAILABLE}, GRAPH_RAG_QUERY_ENABLED={GRAPH_RAG_QUERY_ENABLED}")

# Maximum input token limit for chat history (from .env)
MAX_INPUT_TOKEN_LIMIT = int(os.getenv("MAX_INPUT_TOKEN_LIMIT", "4096"))

# Global progress callback for tool functions
_progress_callback = None

# Global accessible document IDs for access control in tool functions
_accessible_doc_ids = None

# Global analytics context for hybrid queries (pre-computed SQL aggregation results)
_analytics_context = None


def _estimate_token_count(text: str) -> int:
    """
    Estimate the token count for a given text.

    Uses a simple heuristic: ~4 characters per token on average.
    This is a reasonable approximation for most LLMs.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Approximate: 1 token ≈ 4 characters (works well for English/code)
    # For CJK languages, this may underestimate slightly
    return len(text) // 4 + 1


def _truncate_history_to_token_limit(
    conversation_history: List[dict],
    max_tokens: int = None
) -> List[dict]:
    """
    Truncate conversation history to fit within token limit.

    Keeps the most recent messages that fit within the token budget.
    Always preserves at least the last message if possible.

    Args:
        conversation_history: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum token limit. Defaults to MAX_INPUT_TOKEN_LIMIT.

    Returns:
        Truncated list of messages (most recent) that fit within limit.
    """
    if not conversation_history:
        return []

    if max_tokens is None:
        max_tokens = MAX_INPUT_TOKEN_LIMIT

    # Calculate tokens for each message (from most recent to oldest)
    messages_with_tokens = []
    for msg in reversed(conversation_history):
        content = msg.get("content", "")
        token_count = _estimate_token_count(content)
        messages_with_tokens.append((msg, token_count))

    # Select messages from most recent until we hit the limit
    selected_messages = []
    total_tokens = 0

    for msg, tokens in messages_with_tokens:
        if total_tokens + tokens <= max_tokens:
            selected_messages.append(msg)
            total_tokens += tokens
        else:
            # Stop adding older messages once we exceed the limit
            break

    # Reverse back to chronological order
    selected_messages.reverse()

    if len(selected_messages) < len(conversation_history):
        logger.info(
            f"[Token Limit] Truncated history from {len(conversation_history)} to "
            f"{len(selected_messages)} messages (~{total_tokens} tokens, limit={max_tokens})"
        )

    return selected_messages


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
7. Return the final answer only with markdown format, without any internal notes or reasoning steps

CRITICAL INSTRUCTIONS FOR USING CONTEXT:
- When "Relevant document context" is provided in the system message, it contains ACTUAL DATA from the user's documents
- The "Document Content" section contains the raw text from the documents - USE THIS DATA to answer questions
- The "Relevant Entities" and "Relationships" sections provide structured information extracted from the documents
- DO NOT say "I don't have access" if context is provided - the context IS the data you need
- Analyze the provided context thoroughly and extract the information requested by the user
- If you see invoice data, receipt data, or financial data in the context, YOU HAVE ACCESS TO IT - use it to answer
- Extract numbers, dates, vendors, items, prices, and other details directly from the "Document Content" chunks

CRITICAL INSTRUCTIONS FOR ANALYTICS/AGGREGATION QUERIES (HYBRID MODE):
- When "PRE-COMPUTED ANALYTICS DATA" is provided in the context, it contains ACCURATE statistics computed by SQL queries
- For questions about totals, sums, counts, breakdowns, or aggregations → ALWAYS use the PRE-COMPUTED ANALYTICS DATA
- DO NOT try to manually calculate totals by adding up values from individual document chunks
- The document chunks in hybrid mode are for general context only (e.g., document names, vendor info, product descriptions)
- The analytics data is the AUTHORITATIVE source for numerical statistics - trust it over manual calculations
- Present the analytics data in a clear, formatted way (tables, bullet points) following the hierarchical structure provided
- IMPORTANT: Copy ALL group and sub-group values from the analytics data exactly as shown
- DO NOT report $0.00 for any group that has a non-zero value in the analytics data
- If you see different values in document chunks, IGNORE THEM - the analytics data is computed from the COMPLETE dataset
- The analytics data contains the complete breakdown by all dimensions (primary, secondary, etc.) - use these exact values

CRITICAL INSTRUCTIONS FOR PRODUCT/PURCHASE QUERIES:
- When asked about "products" or "purchases", ONLY include actual items/goods that were purchased and its price/quantity/tax/discount/shipping fee
- Examples of what to INCLUDE as products:
  * Physical items with descriptions (e.g., "Lenovo 300e Windows", "Laptop", "Monitor", "Mouse", "Keyboard")
  * Software licenses or subscriptions
  * Services that are the main purchase (e.g., "Consulting Services", "Installation Service")

CRITICAL INSTRUCTIONS FOR AMOUNT/PRICE QUERIES:
- When asked about the price, cost, or amount of a SPECIFIC item, respond ONLY with that item's individual amount
- DO NOT return the invoice total, subtotal, or grand total when the user asks about a specific item's price
- Match the item name carefully - if the user asks "how much was the laptop?", find the line item for the laptop and return ONLY that line item's amount
- If an invoice has multiple items:
  * "How much was X?" → Return ONLY the amount for item X (NOT the invoice total)
  * "What is the total?" → Return the invoice total/grand total
  * "How much did I spend on X?" → Return ONLY the amount for item X
  * "What was the cost of X?" → Return ONLY the amount for item X
- NEVER confuse:
  * Individual line item amounts (what was paid for a single product/service)
  * Invoice subtotals (sum before tax/fees)
  * Invoice totals (final amount including everything)
- If the user does not explicitly ask for "total", "subtotal", "grand total", or "all items", assume they want the specific item amount
- When providing an amount, be explicit about what it represents (e.g., "The laptop cost $500" NOT "The total was $1500")

CRITICAL INSTRUCTIONS FOR MULTIPLE ITEMS DETECTION:
- If the context appears to list multiple items merged on one line (e.g., "Keyboard, Mouse, Webcam" or "Laptop / Monitor / Cable"), DO NOT treat them as a single item
- When you detect comma-separated or slash-separated item names that appear to be multiple distinct products:
  * First, check other parts of the context for detailed individual item entries
  * If detailed entries exist, use those individual items with their respective prices/quantities
  * List each item separately with its own details (price, quantity, etc.)
- NEVER output merged item names as one entry (e.g., don't say "Keyboard, Mouse, Webcam costs $150")
- Instead, if you find individual details elsewhere in the context, list them separately:
  * "Keyboard: $50"
  * "Mouse: $30"
  * "Webcam: $70"
- If the context ONLY has the merged summary line without individual details, respond:
  "The context shows multiple items listed together (e.g., 'Keyboard, Mouse, Webcam') but individual item details are not available. Please check the original document for itemized pricing."
- Common patterns that indicate merged items (DO NOT treat as single item):
  * Items separated by commas: "Item A, Item B, Item C"
  * Items separated by slashes: "Item A / Item B / Item C"
  * Items separated by "and": "Item A and Item B and Item C"
  * Items with "etc." or ellipsis: "Keyboards, Mice, etc."

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





def _get_last_assistant_message(history: List[dict]) -> Optional[str]:
    """
    Get the last assistant message from history.

    Args:
        history: Conversation history

    Returns:
        Last assistant message content, or None if not found
    """
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def _is_error_or_incomplete_response(response: Optional[str]) -> bool:
    """
    Check if a response indicates an error or incomplete answer.

    Args:
        response: Assistant response content

    Returns:
        True if the response appears to be an error or incomplete
    """
    if not response:
        return True

    response_lower = response.lower()

    # Error indicators
    error_indicators = [
        "error", "failed", "could not", "unable to",
        "no relevant documents", "no information found",
        "i don't have", "i cannot find", "i couldn't find",
        "no data", "not found", "unavailable",
        "apologize", "sorry, i"
    ]

    # Check for error indicators
    for indicator in error_indicators:
        if indicator in response_lower:
            return True

    # Very short response might be incomplete (less than 100 chars)
    if len(response.strip()) < 100:
        return True

    return False


# Configuration for auto-fallback from CONVERSATION_ONLY to document search
CHAT_AUTO_FALLBACK_ENABLED = os.getenv("CHAT_AUTO_FALLBACK_ENABLED", "true").lower() == "true"


def _is_no_data_response(response: str) -> bool:
    """
    Detect if LLM response indicates no useful data was found in conversation history.

    This is used to trigger automatic fallback to document search when
    CONVERSATION_ONLY mode fails to find relevant information.

    Args:
        response: The LLM response from conversation-only mode

    Returns:
        True if the response indicates no useful data was found
    """
    if not response:
        return True

    response_lower = response.lower()

    # Patterns that indicate the LLM couldn't find the requested information
    no_data_patterns = [
        # Direct "no data" statements
        "no information available",
        "no data found",
        "no products found",
        "no records",
        "no items found",
        "no purchases",
        "no transactions",
        # Cannot find patterns
        "cannot find any",
        "couldn't find any",
        "could not find any",
        "can't find any",
        "unable to find",
        # Don't have patterns
        "don't have information",
        "do not have information",
        "don't have any information",
        "don't have details",
        "no relevant information",
        "no specific information",
        # Not mentioned patterns
        "not mentioned",
        "wasn't discussed",
        "haven't been mentioned",
        "not available in the conversation",
        "not in the conversation",
        "not discussed in",
        # Request more info patterns
        "please provide more details",
        "check the date again",
        "could you clarify",
        "need more information",
        # Appears/seems patterns
        "it seems there is no",
        "there doesn't appear to be",
        "there appears to be no",
        "it appears that no",
    ]

    for pattern in no_data_patterns:
        if pattern in response_lower:
            logger.info(f"[No-Data Detection] Pattern matched: '{pattern}'")
            return True

    return False


def _is_follow_up_query(query: str) -> bool:
    """
    Detect if a query is a follow-up question that references previous conversation.

    Args:
        query: The user's query

    Returns:
        True if the query appears to be a follow-up question
    """
    query_lower = query.lower().strip()

    # Patterns that indicate follow-up questions
    follow_up_patterns = [
        # Reference to previous answer/response
        "double check", "check again", "verify", "are you sure", "you are not correct",
        "you're not correct", "that's not right", "that is not right", "incorrect",
        "wrong", "mistake", "error", "re-check", "recheck",
        # Pronouns referencing previous context
        "what about it", "tell me more", "more details", "explain more",
        "what else", "anything else", "is there more",
        # Implicit references
        "how much", "how many", "what is the total", "summarize it",
        "can you clarify", "i don't understand", "what do you mean",
        # Continuation
        "and also", "also", "what about", "regarding that",
        "on that note", "speaking of", "related to that",
    ]

    for pattern in follow_up_patterns:
        if pattern in query_lower:
            return True

    # Very short queries are often follow-ups
    if len(query.split()) <= 5 and any(word in query_lower for word in ["it", "this", "that", "them", "those"]):
        return True

    return False


def _rewrite_query_with_llm(
    original_query: str,
    conversation_history: List[dict]
) -> str:
    """
    Use LLM to rewrite a follow-up query by incorporating conversation context.

    Args:
        original_query: The user's original follow-up query
        conversation_history: Previous conversation messages

    Returns:
        Rewritten query with full context
    """
    try:
        # Build conversation context string (last 4 messages for context)
        recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history

        context_str = ""
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long messages
            content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            context_str += f"{role}: {content}\n\n"

        prompt = f"""You are an expert query rewriter. The user asked a follow-up question that references previous conversation context.

CONVERSATION HISTORY:
{context_str}

CURRENT USER QUERY: {original_query}

## THINK CAREFULLY - Analysis Steps:

**Step 1: Understand the Context**
- What was the previous conversation about?
- What specific information was discussed?
- What entities, topics, or documents were mentioned?

**Step 2: Understand the Follow-up Intent**
- What is the user REALLY asking for?
- Are they asking for verification, more details, corrections, or clarification?
- What specific aspect do they want to know about?

**Step 3: Identify Key Elements to Include**
- Specific entities (names, products, dates) from the conversation
- Topics and subjects that were discussed
- Document names or types that were referenced
- Any specific values, amounts, or details mentioned

**Step 4: Construct Comprehensive Query**
- Create a detailed, self-contained query
- Include ALL relevant context from the conversation
- Be specific about what information is needed

TASK: Rewrite the user's query to be a standalone, self-contained search query that includes all necessary context from the conversation.

RULES:
1. The rewritten query should be DETAILED and SPECIFIC
2. Include ALL relevant entities, topics, dates, and document names from the conversation
3. If the user is asking for verification/correction, clearly state what needs to be verified
4. If the user is asking for more details, specify what details are needed
5. Keep the rewritten query comprehensive but focused (under 100 words)
6. Return ONLY the rewritten query, nothing else

REWRITTEN QUERY:"""

        # Use the query model for fast rewriting
        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
            temperature=0.1,
            num_ctx=2048,
            num_predict=150,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()

        # Clean up response
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]

        # Remove any "REWRITTEN QUERY:" prefix if present
        if rewritten.lower().startswith("rewritten query:"):
            rewritten = rewritten[16:].strip()

        logger.info(f"[Query Rewrite] '{original_query[:50]}...' -> '{rewritten[:100]}...'")
        return rewritten

    except Exception as e:
        logger.warning(f"[Query Rewrite] Failed to rewrite query: {e}")
        return original_query


def _build_enhanced_fallback_query(
    original_query: str,
    conversation_history: List[dict]
) -> str:
    """
    Build an enhanced query for fallback document search by combining
    the original query with context from conversation history.

    For follow-up questions, uses LLM to rewrite the query with full context.
    For other queries, extracts entities and time periods from history.

    Args:
        original_query: The user's original query
        conversation_history: Previous conversation messages

    Returns:
        Enhanced query string for document search
    """
    if not conversation_history:
        return original_query

    # Check if this is a follow-up question that needs LLM rewriting
    if _is_follow_up_query(original_query) and len(conversation_history) >= 2:
        logger.info(f"[Query Enhancement] Detected follow-up query, using LLM rewriting")
        return _rewrite_query_with_llm(original_query, conversation_history)

    # For non-follow-up queries, extract context from recent conversation history
    context_parts = []

    # Look at last few user messages for additional context
    user_messages = [
        msg["content"] for msg in conversation_history
        if msg.get("role") == "user"
    ][-3:]  # Last 3 user messages

    # Extract potential entities and time periods from user messages
    for user_msg in user_messages:
        msg_lower = user_msg.lower()

        # Extract year references
        import re
        years = re.findall(r'\b(20\d{2})\b', user_msg)
        for year in years:
            if year not in context_parts:
                context_parts.append(year)

        # Extract month references
        months = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        for month in months:
            if month in msg_lower and month not in context_parts:
                context_parts.append(month)

    # Build enhanced query
    if context_parts:
        # Combine original query with extracted context
        context_str = " ".join(context_parts)
        enhanced_query = f"{original_query} {context_str}"
        logger.info(f"[Fallback Query] Enhanced: '{original_query}' -> '{enhanced_query}'")
        return enhanced_query

    return original_query


def _classify_query_with_context(
    query: str,
    conversation_history: List[dict] = None,
    is_retry: bool = False
) -> str:
    """
    Classify query intent with conversation context awareness.

    Args:
        query: The user's current message
        conversation_history: Previous conversation messages
        is_retry: If True, user clicked retry button - always trigger new search

    Returns:
        One of: "CONVERSATION_ONLY", "SAME_CONTEXT", "NEW_SEARCH", "DIRECT"
    """
    # Rule 1: Retry always triggers new search
    if is_retry:
        logger.info(f"[Query Classification] '{query}' -> NEW_SEARCH (retry action)")
        return "NEW_SEARCH"

    # Rule 2: Quick pattern matching for greetings
    query_lower = query.lower().strip()
    direct_patterns = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "wassup", "sup",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see you", "later",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
        "cool", "awesome", "nice", "thanks", "thank you", "thx", "ty",
        "great", "good job", "well done", "excellent", "fantastic", "amazing", "wow", "How are you"
    ]

    for pattern in direct_patterns:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.endswith(" " + pattern):
            logger.info(f"[Query Classification] '{query}' -> DIRECT (greeting pattern)")
            return "DIRECT"

    # Rule 3: If no conversation history or too short, must be a new search
    # Need at least 2 Q&A rounds (4 messages) for CONVERSATION_ONLY
    if not conversation_history or len(conversation_history) < 4:
        logger.info(f"[Query Classification] '{query}' -> NEW_SEARCH (insufficient history: {len(conversation_history) if conversation_history else 0} messages)")
        return "NEW_SEARCH"

    # Rule 4: Check if last assistant response was an error/failure
    last_assistant = _get_last_assistant_message(conversation_history)
    if _is_error_or_incomplete_response(last_assistant):
        logger.info(f"[Query Classification] '{query}' -> NEW_SEARCH (previous response was error/incomplete)")
        return "NEW_SEARCH"

    # Rule 5: Use full conversation history with token limit for LLM classification
    # Truncate to fit within token budget, keeping most recent messages
    truncated_history = _truncate_history_to_token_limit(conversation_history)

    conversation_context = ""
    for msg in truncated_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
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
        f"GRAPH_RAG_QUERY_ENABLED={GRAPH_RAG_QUERY_ENABLED}, mode={GRAPHRAG_QUERY_MODE}"
    )

    if not GRAPHRAG_AVAILABLE:
        logger.debug("[GraphRAG Query] Skipping - GraphRAG module not available")
        return ""

    if not GRAPH_RAG_QUERY_ENABLED:
        logger.debug("[GraphRAG Query] Skipping - GRAPH_RAG_QUERY_ENABLED=false in .env")
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
    4. Vector search on filtered document chunks (filtered by user access control)
    5. GraphRAG context with dynamic max_steps for entity/relationship enrichment

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Relevant document chunks as a formatted string.
    """
    try:
        # Get accessible document IDs from global context (set by stream_agent_response)
        global _accessible_doc_ids
        accessible_doc_ids = _accessible_doc_ids

        # Log access control status
        # SECURITY: If accessible_doc_ids is empty set, user has NO document access - return early
        if accessible_doc_ids is not None:
            if len(accessible_doc_ids) == 0:
                logger.warning("[Search] Access control: user has NO accessible documents - blocking search")
                return "You don't have access to any documents. Please contact your administrator to grant document access permissions."
            else:
                logger.info(f"[Search] Access control: filtering to {len(accessible_doc_ids)} accessible documents")
        else:
            # accessible_doc_ids is None - this could mean:
            # 1. No user_id was provided (anonymous/unauthenticated)
            # 2. An error occurred while fetching permissions
            # For security, we should block access in this case
            logger.warning("[Search] Access control: accessible_doc_ids is None - blocking search for security")
            return "Unable to verify your document access permissions. Please try again or contact your administrator."

        # Check if iterative reasoning is enabled
        iterative_enabled = os.getenv("ITERATIVE_REASONING_ENABLED", "true").lower() == "true"

        # Send progress update: Analyzing query
        _send_progress("Analyzing your question...")

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
        _send_progress("Identifying key topics and entities...")

        query_analysis = _analyze_query_with_llm(query)
        enhanced_query = query_analysis["enhanced_query"]
        query_metadata = query_analysis["metadata"]

        logger.info(f"[Search] Original query: '{query[:50]}...'")
        logger.info(f"[Search] Enhanced query: '{enhanced_query[:100]}...'")
        logger.info(f"[Search] Extracted entities: {query_metadata.get('entities', [])}")
        logger.info(f"[Search] Extracted topics: {query_metadata.get('topics', [])}")

        # Step 2: Convert accessible_doc_ids to list for document router
        # The router now works with document_ids directly (simplified from source names)
        accessible_document_ids = None
        if accessible_doc_ids is not None:
            # Convert set/frozenset to list of strings
            accessible_document_ids = [str(doc_id) for doc_id in accessible_doc_ids]
            logger.info(f"[Access Control] Passing {len(accessible_document_ids)} accessible document IDs to router")

        # Step 2.5: Route to relevant documents based on metadata (with access control)
        _send_progress("Finding relevant documents...")

        from .document_router import DocumentRouter
        from .llm_service import get_llm_service
        llm_service = get_llm_service()
        router = DocumentRouter(llm_service=llm_service)
        # Pass accessible_document_ids directly - router now returns document IDs
        routed_document_ids = router.route_query(
            query_metadata,
            original_query=query,
            accessible_document_ids=accessible_document_ids
        )

        if routed_document_ids:
            logger.info(
                f"[Search] Document router selected {len(routed_document_ids)} document IDs: {routed_document_ids}"
            )

        # Step 3: Use Iterative Reasoning Engine (unified for GraphRAG and vector-only)
        if iterative_enabled:
            return _search_with_iterative_reasoning(
                query=query,
                max_steps=max_steps,
                relevant_sources=None,  # Deprecated - use routed_document_ids
                graphrag_enabled=GRAPHRAG_AVAILABLE and GRAPH_RAG_QUERY_ENABLED,
                accessible_doc_ids=accessible_doc_ids,
                routed_document_ids=routed_document_ids
            )
        else:
            # Simple vector-only search (no iterative reasoning)
            return _simple_vector_search(
                query=query,
                enhanced_query=enhanced_query,
                accessible_doc_ids=accessible_doc_ids,
                routed_document_ids=routed_document_ids
            )

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error searching documents: {str(e)}"


def _search_with_iterative_reasoning(
    query: str,
    max_steps: int,
    relevant_sources: List[str] = None,  # DEPRECATED - use routed_document_ids
    graphrag_enabled: bool = False,
    accessible_doc_ids: Optional[set] = None,
    routed_document_ids: Optional[List[str]] = None
) -> str:
    """
    Search using the unified Iterative Reasoning Engine.

    This provides multi-round query refinement for both GraphRAG and vector-only modes.

    Args:
        query: User's search query
        max_steps: Maximum reasoning iterations
        relevant_sources: DEPRECATED - use routed_document_ids
        graphrag_enabled: Whether GraphRAG (Neo4j) is available and enabled
        accessible_doc_ids: Optional set of document IDs the user can access (for access control)
        routed_document_ids: List of document IDs from router (filtered by access control)

    Returns:
        Formatted search results string
    """
    import asyncio
    import concurrent.futures
    from .iterative_reasoning import IterativeReasoningEngine, ReasoningResult

    # Debug: Log the decision factors for GraphRAG
    logger.info(f"[Search] GraphRAG Decision Debug:")
    logger.info(f"[Search]   - GRAPHRAG_AVAILABLE={GRAPHRAG_AVAILABLE}")
    logger.info(f"[Search]   - GRAPH_RAG_QUERY_ENABLED={GRAPH_RAG_QUERY_ENABLED}")
    logger.info(f"[Search]   - graphrag_enabled parameter={graphrag_enabled}")
    logger.info(f"[Search]   - routed_document_ids count={len(routed_document_ids) if routed_document_ids else 'None'}")
    logger.info(f"[Search] Using Iterative Reasoning Engine (graphrag={graphrag_enabled}, max_steps={max_steps})")

    # Create reasoning engine with routed document IDs
    engine = IterativeReasoningEngine(
        graphrag_enabled=graphrag_enabled,
        max_steps=max_steps,
        source_names=None,  # DEPRECATED - not used
        accessible_doc_ids=accessible_doc_ids,
        routed_document_ids=routed_document_ids
    )

    # Define async reasoning function
    async def _run_reasoning():
        # Create async progress callback
        # Note: percent is optional because iterative_reasoning.py only passes message
        async def progress_callback(message: str, percent: int = None):
            _send_progress(message, percent)

        return await engine.reason(query, progress_callback=progress_callback)

    # Run async reasoning in sync context
    try:
        # Check if we're already in an event loop (e.g., FastAPI request)
        asyncio.get_running_loop()
        # We're in an existing event loop - run in a new thread
        logger.debug("[Search] Running iterative reasoning in thread pool")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _run_reasoning())
            result: ReasoningResult = future.result()
    except RuntimeError:
        # No running event loop - use asyncio.run directly
        logger.debug("[Search] Running iterative reasoning with asyncio.run")
        result: ReasoningResult = asyncio.run(_run_reasoning())

    # Format results
    return _format_reasoning_result(result)


def _format_reasoning_result(result) -> str:
    """
    Format the ReasoningResult into a string for the LLM.

    Uses token-aware context building to ensure results fit within
    LLM context window limits while prioritizing the most relevant content.

    Args:
        result: ReasoningResult from IterativeReasoningEngine

    Returns:
        Formatted string with answer and/or context
    """
    from .iterative_reasoning import ReasoningResult

    results = []

    # If we have a final answer from iterative reasoning, prioritize it
    if result.final_answer:
        logger.info(f"[Search] Using final answer from iterative reasoning ({len(result.final_answer)} chars)")
        results.append(f"[Iterative Reasoning Answer]\n{result.final_answer}")

    # Build graph context string from entities and relationships
    graph_context = _format_graph_context(result.entities, result.relationships)

    # Get the initial query for relevance scoring
    initial_query = getattr(result, 'initial_query', None)
    queries_made = getattr(result, 'queries_made', [])

    # Log what query will be used for scoring
    logger.info("[Search] Query relevance scoring debug:")
    logger.info(f"  Initial query (for scoring): '{initial_query}'")
    logger.info(f"  Queries made during retrieval: {queries_made}")
    logger.info(f"  Number of chunks to score: {len(result.chunks)}")
    logger.info(f"  Number of parent chunks to score: {len(getattr(result, 'parent_chunks', []))}")

    # Use token-aware context builder for chunks and parent chunks
    try:
        built_context = build_token_aware_context(
            chunks=result.chunks,
            parent_chunks=getattr(result, 'parent_chunks', []),
            graph_context=graph_context,
            query=initial_query,  # Pass query for centralized relevance scoring
        )

        # Log context building stats
        logger.info(
            f"[Search] Iterative reasoning context built: {built_context.total_tokens} tokens, "
            f"{built_context.chunks_included}/{len(result.chunks)} chunks, "
            f"{built_context.parent_chunks_included}/{len(getattr(result, 'parent_chunks', []))} parents "
            f"({built_context.parent_chunks_summarized} summarized), "
            f"graph_tokens={built_context.graph_context_tokens}"
        )

        if built_context.warnings:
            for warning in built_context.warnings:
                logger.warning(f"[Search] Context builder warning: {warning}")

        # Add the formatted context from the builder
        if built_context.formatted_context:
            results.append(built_context.formatted_context)

    except Exception as e:
        # Fallback to simple formatting if context builder fails
        logger.warning(f"[Search] Context builder failed for iterative reasoning, using simple format: {e}")

        # Format graph context manually
        if graph_context:
            results.append(graph_context)

        # Add document chunks as context (simple fallback)
        seen_sources = set()
        for i, chunk in enumerate(result.chunks[:20], 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            heading = chunk.get("metadata", {}).get("heading_path", "")
            content = chunk.get("page_content", chunk.get("content", ""))

            chunk_result = f"[Document {i}: {source}]"
            if heading:
                chunk_result += f"\nSection: {heading}"
            chunk_result += f"\n{content}"
            results.append(chunk_result)
            seen_sources.add(source)

        logger.info(
            f"[Search] Fallback formatting: {len(result.chunks)} chunks from {len(seen_sources)} sources"
        )

    # Log summary
    logger.info(
        f"[Search] Iterative reasoning complete: {result.steps_taken} steps, "
        f"{len(result.queries_made)} queries, {len(result.chunks)} chunks, "
        f"{len(getattr(result, 'parent_chunks', []))} parent chunks, "
        f"answer={'yes' if result.final_answer else 'no'}"
    )

    if not results:
        return "No relevant documents found for this query."

    _send_progress("Preparing your answer...")

    return "\n\n---\n\n".join(results)


def _format_graph_context(entities: list, relationships: list) -> str:
    """
    Format entities and relationships into a graph context string.

    This helper function creates a formatted string from GraphRAG entities
    and relationships that can be passed to the token-aware context builder.

    Args:
        entities: List of entity dictionaries from GraphRAG
        relationships: List of relationship dictionaries from GraphRAG

    Returns:
        Formatted graph context string, or empty string if no graph data
    """
    if not entities and not relationships:
        return ""

    parts = []

    # Format entities
    if entities:
        entity_lines = ["[Knowledge Graph Entities]"]
        for entity in entities[:15]:
            name = entity.get("name", entity.get("entity_name", "Unknown"))
            entity_type = entity.get("entity_type", "")
            desc = entity.get("description", "")[:200]
            entity_lines.append(f"- **{name}** ({entity_type}): {desc}")
        if len(entities) > 15:
            entity_lines.append(f"... and {len(entities) - 15} more entities")
        parts.append("\n".join(entity_lines))

    # Format relationships
    if relationships:
        rel_lines = ["[Knowledge Graph Relationships]"]
        for rel in relationships[:15]:
            src = rel.get("src_name", rel.get("src_entity_id", "?"))
            tgt = rel.get("tgt_name", rel.get("tgt_entity_id", "?"))
            desc = rel.get("description", "related to")
            rel_lines.append(f"- {src} → {tgt}: {desc}")
        if len(relationships) > 15:
            rel_lines.append(f"... and {len(relationships) - 15} more relationships")
        parts.append("\n".join(rel_lines))

    return "\n\n".join(parts)


def _simple_vector_search(
    query: str,
    enhanced_query: str,
    accessible_doc_ids: Optional[set] = None,
    routed_document_ids: Optional[List[str]] = None
) -> str:
    """
    Simple vector-only search (used when iterative reasoning is disabled).

    This function performs a straightforward vector search without:
    - Iterative reasoning (multi-step query refinement)
    - GraphRAG context (graph-based knowledge retrieval)

    Uses unified_vector_search() for consistent behavior and token-aware
    context building to ensure results fit within LLM context limits.

    Args:
        query: Original user query
        enhanced_query: LLM-enhanced query
        accessible_doc_ids: Optional set of document IDs the user can access (for access control)
        routed_document_ids: List of document IDs from router (filtered by access control)

    Returns:
        Formatted search results string
    """
    logger.info("[Search] Using simple vector search (no iterative reasoning)")

    # Determine document IDs to search - prefer routed, fallback to accessible
    if routed_document_ids and len(routed_document_ids) > 0:
        doc_id_list = routed_document_ids
        logger.info(f"[Search] Vector search filtering to {len(doc_id_list)} routed documents")
    elif accessible_doc_ids and len(accessible_doc_ids) > 0:
        doc_id_list = list(accessible_doc_ids)
        logger.info(f"[Search] Vector search using {len(doc_id_list)} accessible documents (no routing)")
    else:
        doc_id_list = []

    # Use unified vector search
    _send_progress("Searching through your documents...")

    search_result = unified_vector_search(
        query=enhanced_query,
        document_ids=doc_id_list,
        k=18,
        fetch_k=50,
        lambda_mult=0.5,
        per_document_selection=True,
        include_parent_chunks=True  # Uses env variable if None
    )

    # Check if search returned results
    if search_result.total_count == 0:
        if search_result.retrieval_stats.get("blocked"):
            return "No documents accessible for this query."
        return "No relevant documents found for this query."

    # Use token-aware context builder to format results within token limits
    _send_progress("Gathering relevant information...")

    try:
        built_context = build_token_aware_context(
            chunks=search_result.chunks,
            parent_chunks=search_result.parent_chunks,
            graph_context=None,  # No GraphRAG in simple vector search
            query=enhanced_query,
        )

        # Log context building stats
        logger.info(
            f"[Search] Context built: {built_context.total_tokens} tokens, "
            f"{built_context.chunks_included}/{len(search_result.chunks)} chunks, "
            f"{built_context.parent_chunks_included}/{len(search_result.parent_chunks)} parents "
            f"({built_context.parent_chunks_summarized} summarized)"
        )

        if built_context.warnings:
            for warning in built_context.warnings:
                logger.warning(f"[Search] Context builder warning: {warning}")

        _send_progress("Preparing your answer...")

        return built_context.formatted_context

    except Exception as e:
        # Fallback to simple formatting if context builder fails
        logger.warning(f"[Search] Context builder failed, using simple format: {e}")

        results = []
        doc_num = 1

        # Format regular chunks
        for chunk in search_result.chunks:
            source = chunk["metadata"].get("source", "Unknown")
            heading = chunk["metadata"].get("heading_path", "")
            content = chunk["page_content"]

            result = f"[Document {doc_num}: {source}]"
            if heading:
                result += f"\nSection: {heading}"
            result += f"\n{content}"
            results.append(result)
            doc_num += 1

        # Format parent chunks - use summary if available
        for chunk in search_result.parent_chunks:
            source = chunk["metadata"].get("source", "Unknown")
            heading = chunk["metadata"].get("heading_path", "")
            # Prefer summary over full content if available
            content = chunk["metadata"].get("parent_summary", chunk["page_content"])

            result = f"[Document {doc_num} (Context): {source}]"
            if heading:
                result += f"\nSection: {heading}"
            result += f"\n{content}"
            results.append(result)
            doc_num += 1

        logger.info(
            f"[Search] Simple vector search: {len(search_result.chunks)} chunks + "
            f"{len(search_result.parent_chunks)} parent chunks from {len(search_result.sources)} sources"
        )

        _send_progress("Preparing your answer...")

        return "\n\n---\n\n".join(results)


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
                # Check if we have pre-computed analytics context
                # For analytics-focused queries, skip document chunk retrieval entirely
                # as chunks are just partial data that confuses the LLM
                global _analytics_context

                if _analytics_context:
                    # ANALYTICS-FOCUSED HYBRID MODE:
                    # Skip document chunk search - use analytics data + document summary only
                    logger.info(f"[vLLM] Analytics context available - skipping chunk retrieval for cleaner response")

                    # Normalize the query
                    normalized_query = normalize_query_dates(user_query)
                    if normalized_query != user_query:
                        logger.info(f"[vLLM] Normalized query: '{user_query}' -> '{normalized_query}'")

                    # Build context with ONLY analytics data (no confusing chunks)
                    context_message = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  📊 PRE-COMPUTED ANALYTICS DATA - USE THESE EXACT VALUES IN YOUR RESPONSE                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣

{_analytics_context}

╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  INSTRUCTIONS:                                                                                       ║
║  1. Format the above data in a clear, readable way (tables, bullet points, etc.)                    ║
║  2. Use the EXACT values shown above - they are computed from the complete dataset                  ║
║  3. Include all groups and sub-groups shown - do not skip any categories                            ║
║  4. Present totals by year and by category as requested                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

**USER QUESTION**: {normalized_query}
"""
                else:
                    # STANDARD RAG MODE: Search for document chunks
                    # Extract conversation history from messages for context-aware search
                    conversation_history = []
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            conversation_history.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            conversation_history.append({"role": "assistant", "content": msg.content})

                    # Enhance query with conversation context (excluding current query)
                    # The last message is the current query, so we use history up to that point
                    history_for_context = conversation_history[:-1] if len(conversation_history) > 1 else []
                    enhanced_query = _build_enhanced_fallback_query(user_query, history_for_context)

                    if enhanced_query != user_query:
                        logger.info(f"[vLLM] Enhanced query with context: '{user_query[:30]}...' -> '{enhanced_query[:50]}...'")

                    # Search for context first
                    logger.info(f"[vLLM] Searching documents for query: {enhanced_query[:50]}...")
                    context = search_documents.invoke({"query": enhanced_query})

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
                logger.info(f"[vLLM] Context message length: {len(context_message)} chars")
                logger.info(f"[vLLM] Context preview (first 500 chars): {context_message[:500]}")

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
    conversation_history: List[Dict[str, str]] = None
) -> Optional[str]:
    """
    Build a context-aware system message based on session metadata and conversation history.

    Uses the full conversation history (truncated to token limit) for better context awareness.
    The token limit is controlled by MAX_INPUT_TOKEN_LIMIT environment variable.

    Args:
        session_context: Session metadata containing entities, topics, documents, etc.
        conversation_history: Full conversation history (will be truncated to token limit).

    Returns:
        System message content with conversation context, or None if no context available.
    """
    if not session_context and not conversation_history:
        return None

    system_message = """You are a helpful AI assistant with access to document knowledge.

"""

    # Add conversation context (full history truncated to token limit)
    if conversation_history and len(conversation_history) >= 2:
        # Truncate history to fit within token budget, keeping most recent messages
        truncated_history = _truncate_history_to_token_limit(conversation_history)

        if truncated_history:
            system_message += """Conversation History:
The following is the conversation history with the user:
"""
            for msg in truncated_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                system_message += f"\n{role}: {content}\n"

            system_message += """
IMPORTANT: When the user asks a follow-up question, it is likely related to the conversation above.
Pay close attention to:
- Time periods mentioned (e.g., "2025", "last month")
- Specific topics discussed (e.g., "expenses", "purchases")
- Entities referenced (e.g., document names, people, items)

If the user's question seems incomplete or vague (e.g., "how much totally", "summarize it"),
assume they are referring to the context from the conversation history above.

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
    session_context: Optional[Dict[str, Any]] = None,
    is_retry: bool = False,
    accessible_doc_ids: Optional[set] = None,
    analytics_context: Optional[str] = None,
    document_context_changed: bool = False
):
    """
    Stream the agent response for a query.

    Args:
        query: The user's question.
        conversation_history: Optional list of previous messages.
        progress_callback: Optional async callback function(message: str, percent: int) for progress updates.
        session_context: Optional session metadata with entities, topics, and conversation context.
        is_retry: If True, user clicked retry button - always trigger new document search.
        accessible_doc_ids: Optional set of document IDs the user has access to (for access control filtering).
        analytics_context: Optional pre-computed analytics context for hybrid queries (from SQL aggregation).
        document_context_changed: If True, the document/workspace selection has changed - force new search.

    Yields:
        Chunks of the response text.
    """
    # Store progress callback and accessible doc IDs in global variables so tools can access them
    global _progress_callback, _accessible_doc_ids, _analytics_context
    _progress_callback = progress_callback
    _accessible_doc_ids = accessible_doc_ids
    _analytics_context = analytics_context

    # Log analytics context if provided (for hybrid queries)
    if analytics_context:
        logger.info(f"[Analytics Context] Received pre-computed analytics context ({len(analytics_context)} chars) for hybrid query")

    # Log access control info
    if accessible_doc_ids is not None:
        if len(accessible_doc_ids) == 0:
            logger.warning(f"[Access Control] User has NO accessible documents - searches will be blocked")
        else:
            logger.info(f"[Access Control] Filtering search to {len(accessible_doc_ids)} accessible documents")
    else:
        logger.warning("[Access Control] accessible_doc_ids is None - searches will be blocked for security")

    # Force NEW_SEARCH if document context has changed (user changed workspace/document selection)
    force_new_search = document_context_changed
    if document_context_changed:
        logger.info(f"[Document Context] Document/workspace selection changed - forcing new document search")

    # Classify query with conversation context and retry flag
    query_classification = _classify_query_with_context(query, conversation_history, is_retry=is_retry or force_new_search)
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

        # Phase 1: Get response from LLM using conversation history
        llm_service = get_llm_service()
        llm = llm_service.get_chat_model(temperature=0.7)

        # If auto-fallback is enabled, collect full response first to check for "no data"
        if CHAT_AUTO_FALLBACK_ENABLED:
            logger.info(f"[CONVERSATION_ONLY] Auto-fallback enabled, collecting response first")

            # Collect full response
            full_response = ""
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content

            # Check if response indicates no useful data
            if _is_no_data_response(full_response):
                logger.info(f"[CONVERSATION_ONLY] Response indicates no data, triggering fallback to document search")

                # Send progress update if callback available
                if progress_callback:
                    await progress_callback("Searching documents for more information...")

                # Phase 2: Fallback to document search with enhanced query
                enhanced_query = _build_enhanced_fallback_query(query, conversation_history)
                logger.info(f"[Fallback] Using enhanced query for document search: '{enhanced_query}'")

                # Use the agent with tools for document search
                agent = create_agent_executor()

                # Build message history for agent
                agent_messages = []

                # Add system message with context if available
                if session_context or conversation_history:
                    system_content = _build_context_aware_system_message(
                        session_context or {},
                        conversation_history
                    )
                    if system_content:
                        agent_messages.append(SystemMessage(content=system_content))

                if conversation_history:
                    for msg in conversation_history:
                        if msg["role"] == "user":
                            agent_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            agent_messages.append(AIMessage(content=msg["content"]))

                # Add enhanced query
                agent_messages.append(HumanMessage(content=enhanced_query))

                # Stream response from agent
                async for event in agent.astream_events(
                    {"messages": agent_messages},
                    version="v2",
                ):
                    if event["event"] == "on_chat_model_stream":
                        parent_ids = event.get("parent_ids", [])
                        if len(parent_ids) <= 2:
                            content = event["data"]["chunk"].content
                            if content:
                                yield content

                return  # Exit after fallback search

            else:
                # Response is good, stream it to the user
                logger.info(f"[CONVERSATION_ONLY] Response has useful data, streaming to user")
                yield full_response
                return  # Exit early

        else:
            # Auto-fallback disabled, stream directly
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


