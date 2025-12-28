"""
Context analyzer for understanding conversation context, pronouns, and references.
Uses LLM for intelligent topic and entity extraction.
"""
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

# Environment variable to enable/disable LLM-based analysis
CONTEXT_ANALYZER_USE_LLM = os.getenv("CONTEXT_ANALYZER_USE_LLM", "true").lower() == "true"

# LLM prompt for context analysis
CONTEXT_ANALYSIS_PROMPT = """Analyze the following conversation and current message to extract context information.

## Current User Message:
{current_message}

## Recent Conversation History (last few messages):
{conversation_history}

## Task:
Extract the following information from the conversation:

1. **Topics**: Main subjects or themes being discussed (e.g., "expenses", "invoices", "sales report", "customer orders")
2. **Entities**: Named entities mentioned:
   - documents: File names or document references (e.g., "report.pdf", "invoice #123")
   - people: Names of people mentioned
   - organizations: Company or organization names
   - products: Product or service names
   - dates: Specific dates or time periods mentioned
3. **Pronouns**: Any pronouns in the current message that refer to previous context
4. **Resolved Message**: If the current message contains pronouns or references, rewrite it with the actual referents

Respond with ONLY a JSON object in this exact format:
{{
  "topics": ["topic1", "topic2"],
  "entities": {{
    "documents": ["doc1.pdf"],
    "people": ["John Smith"],
    "organizations": [],
    "products": [],
    "dates": ["2025", "January"]
  }},
  "detected_pronouns": ["it", "they"],
  "resolved_message": "the rewritten message with pronouns resolved, or original if no pronouns",
  "main_intent": "brief description of what the user is asking about"
}}

Important:
- Only include entities that are actually mentioned in the conversation
- Topics should be specific to the conversation content, not generic
- If no pronouns need resolving, set resolved_message to the original current message
- Keep topics concise (1-3 words each)
- Maximum 5 topics and 5 entities per category

Your JSON response:"""


class ContextAnalyzer:
    """
    Analyzes conversation context to understand:
    - Pronoun references (he, she, it, they, we, our, etc.)
    - Entity tracking across messages
    - Topic and scope detection

    Uses LLM for intelligent analysis when enabled, falls back to heuristics otherwise.
    """

    # Pronouns to track (used for fallback heuristic detection)
    PRONOUNS = {
        'subject': ['he', 'she', 'it', 'they', 'we'],
        'object': ['him', 'her', 'it', 'them', 'us'],
        'possessive': ['his', 'her', 'its', 'their', 'our'],
        'demonstrative': ['this', 'that', 'these', 'those']
    }

    def __init__(self, use_llm: Optional[bool] = None):
        """
        Initialize context analyzer.

        Args:
            use_llm: Override for LLM usage. If None, uses CONTEXT_ANALYZER_USE_LLM env var.
        """
        self.use_llm = use_llm if use_llm is not None else CONTEXT_ANALYZER_USE_LLM
        self.entity_cache: Dict[str, Any] = {}
        self.topic_cache: Set[str] = set()
        logger.info(f"ContextAnalyzer initialized with use_llm={self.use_llm}")

    def analyze_message(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a message in the context of conversation history.

        Args:
            message: Current user message
            conversation_history: List of previous messages

        Returns:
            Analysis result with:
            - has_pronouns: Whether message contains pronouns
            - detected_pronouns: List of pronouns found
            - entities: Tracked entities from history
            - topics: Detected topics
            - resolved_message: Message with pronouns resolved (if possible)
        """
        if self.use_llm:
            try:
                return self._analyze_with_llm(message, conversation_history)
            except Exception as e:
                logger.warning(f"LLM analysis failed, falling back to heuristics: {e}")
                return self._analyze_with_heuristics(message, conversation_history)
        else:
            return self._analyze_with_heuristics(message, conversation_history)

    def _analyze_with_llm(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze context using LLM for intelligent extraction.
        """
        from rag_service.llm_service import get_llm_service
        from langchain_core.messages import HumanMessage

        # Build conversation history string (last 6 messages for context)
        history_str = self._format_conversation_history(conversation_history[-6:])

        # Create prompt
        prompt = CONTEXT_ANALYSIS_PROMPT.format(
            current_message=message,
            conversation_history=history_str if history_str else "(No previous messages)"
        )

        # Get LLM response
        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
            temperature=0.0,  # Deterministic for extraction
            num_ctx=2048,
            num_predict=500
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        logger.debug(f"LLM context analysis response: {result[:500]}")

        # Parse LLM response
        return self._parse_llm_response(result, message)

    def _format_conversation_history(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Format conversation history for LLM prompt."""
        if not conversation_history:
            return ""

        lines = []
        for msg in conversation_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _parse_llm_response(
        self,
        response: str,
        original_message: str
    ) -> Dict[str, Any]:
        """Parse LLM JSON response into analysis result."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            # Extract fields with defaults
            topics = data.get("topics", [])
            entities = data.get("entities", {})
            detected_pronouns = data.get("detected_pronouns", [])
            resolved_message = data.get("resolved_message", original_message)

            # Ensure entities has expected structure
            if not isinstance(entities, dict):
                entities = {}

            # Normalize entities to expected format
            normalized_entities = {
                "documents": entities.get("documents", [])[:5],
                "people": entities.get("people", [])[:5],
                "topics": topics[:5],  # Also store topics in entities for compatibility
                "objects": entities.get("products", [])[:5] + entities.get("organizations", [])[:5],
                "dates": entities.get("dates", [])[:5]
            }

            logger.info(f"[LLM Context Analysis] Topics: {topics}, Entities: {list(normalized_entities.keys())}, Pronouns: {detected_pronouns}")

            return {
                "has_pronouns": len(detected_pronouns) > 0,
                "detected_pronouns": detected_pronouns,
                "entities": normalized_entities,
                "topics": topics[:5],
                "resolved_message": resolved_message,
                "original_message": original_message,
                "main_intent": data.get("main_intent", "")
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}, response: {response[:200]}")
            # Fall back to heuristics if JSON parsing fails
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    def _analyze_with_heuristics(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze context using rule-based heuristics (fallback method).
        """
        # Detect pronouns
        detected_pronouns = self._detect_pronouns(message)

        # Extract entities from history
        entities = self._extract_entities(conversation_history)

        # Detect topics
        topics = self._detect_topics(conversation_history, message)

        # Attempt to resolve pronouns
        resolved_message = self._resolve_pronouns(
            message,
            detected_pronouns,
            entities,
            conversation_history
        )

        return {
            "has_pronouns": len(detected_pronouns) > 0,
            "detected_pronouns": detected_pronouns,
            "entities": entities,
            "topics": list(topics),
            "resolved_message": resolved_message,
            "original_message": message
        }

    def _detect_pronouns(self, message: str) -> List[str]:
        """Detect pronouns in message."""
        message_lower = message.lower()
        words = re.findall(r'\b\w+\b', message_lower)

        detected = []
        for category, pronouns in self.PRONOUNS.items():
            for pronoun in pronouns:
                if pronoun in words:
                    detected.append(pronoun)

        return list(set(detected))

    def _extract_entities(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Extract entities from conversation history using pattern matching.
        """
        entities = {
            "documents": [],
            "people": [],
            "topics": [],
            "objects": [],
            "dates": []
        }

        # Look for patterns in last 10 messages
        for msg in conversation_history[-10:]:
            content = msg.get("content", "")

            # Find document names (e.g., "document.pdf", "file.docx")
            doc_pattern = r'\b[\w-]+\.(pdf|docx?|xlsx?|txt|md|csv)\b'
            docs = re.findall(doc_pattern, content, re.IGNORECASE)
            entities["documents"].extend(docs)

            # Find date patterns
            date_patterns = [
                r'\b\d{4}\b',  # Years like 2025
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b',
            ]
            for pattern in date_patterns:
                dates = re.findall(pattern, content, re.IGNORECASE)
                entities["dates"].extend(dates)

            # Find capitalized multi-word phrases (potential names/topics)
            cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
            entities["people"].extend(cap_words)

            # Find single capitalized words (potential topics)
            single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
            entities["topics"].extend(single_caps)

        # Deduplicate and limit
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]

        return entities

    def _detect_topics(
        self,
        conversation_history: List[Dict[str, Any]],
        current_message: str = ""
    ) -> Set[str]:
        """Detect main topics from conversation history and current message."""
        topics = set()

        # Domain-specific topic keywords
        topic_keywords = {
            # Document/file related
            'document': ['document', 'file', 'pdf', 'report', 'invoice', 'receipt'],
            'financial': ['expense', 'cost', 'price', 'amount', 'total', 'payment', 'budget', 'spending'],
            'sales': ['sales', 'revenue', 'order', 'purchase', 'customer', 'product'],
            'analysis': ['analysis', 'summary', 'compare', 'comparison', 'trend', 'statistics'],
            'search': ['search', 'find', 'query', 'lookup', 'filter'],
            'time': ['daily', 'weekly', 'monthly', 'yearly', 'annual', 'quarter'],
        }

        # Combine history and current message
        all_content = current_message.lower()
        for msg in conversation_history[-10:]:
            all_content += " " + msg.get("content", "").lower()

        # Find matching topics
        for topic_name, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in all_content:
                    topics.add(topic_name)
                    break

        # Also extract specific terms that appear frequently
        words = re.findall(r'\b[a-z]{4,}\b', all_content)
        word_freq = {}
        for word in words:
            if word not in ['this', 'that', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where', 'which', 'about', 'would', 'could', 'should', 'their', 'there', 'these', 'those']:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add top frequent words as topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:3]:
            if freq >= 2:  # Word appears at least twice
                topics.add(word)

        return topics

    def _resolve_pronouns(
        self,
        message: str,
        pronouns: List[str],
        entities: Dict[str, List[str]],
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Attempt to resolve pronouns to their referents.
        """
        if not pronouns:
            return message

        resolved = message

        # Simple resolution: replace 'it' with last mentioned document
        if 'it' in pronouns and entities.get("documents"):
            last_doc = entities["documents"][0]
            resolved = re.sub(r'\bit\b', f'"{last_doc}"', resolved, flags=re.IGNORECASE)

        # Replace 'this'/'that' with last mentioned topic
        if ('this' in pronouns or 'that' in pronouns) and entities.get("topics"):
            last_topic = entities["topics"][0]
            resolved = re.sub(r'\b(this|that)\b', f'"{last_topic}"', resolved, flags=re.IGNORECASE)

        # Replace 'they'/'them' with last mentioned people
        if ('they' in pronouns or 'them' in pronouns) and entities.get("people"):
            people = ", ".join(entities["people"][:2])
            resolved = re.sub(r'\b(they|them)\b', f'"{people}"', resolved, flags=re.IGNORECASE)

        return resolved


# Async version for use in async contexts
async def analyze_context_async(
    message: str,
    conversation_history: List[Dict[str, Any]],
    use_llm: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Async wrapper for context analysis.

    Args:
        message: Current user message
        conversation_history: List of previous messages
        use_llm: Override for LLM usage

    Returns:
        Analysis result dictionary
    """
    analyzer = ContextAnalyzer(use_llm=use_llm)
    return analyzer.analyze_message(message, conversation_history)
