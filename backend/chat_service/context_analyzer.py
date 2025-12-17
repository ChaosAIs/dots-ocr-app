"""
Context analyzer for understanding conversation context, pronouns, and references.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """
    Analyzes conversation context to understand:
    - Pronoun references (he, she, it, they, we, our, etc.)
    - Entity tracking across messages
    - Topic and scope detection
    """
    
    # Pronouns to track
    PRONOUNS = {
        'subject': ['he', 'she', 'it', 'they', 'we'],
        'object': ['him', 'her', 'it', 'them', 'us'],
        'possessive': ['his', 'her', 'its', 'their', 'our'],
        'demonstrative': ['this', 'that', 'these', 'those']
    }
    
    def __init__(self):
        """Initialize context analyzer."""
        self.entity_cache: Dict[str, Any] = {}
        self.topic_cache: Set[str] = set()
    
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
        # Detect pronouns
        detected_pronouns = self._detect_pronouns(message)
        
        # Extract entities from history
        entities = self._extract_entities(conversation_history)
        
        # Detect topics
        topics = self._detect_topics(conversation_history)
        
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
        Extract entities from conversation history.
        Simple implementation - looks for capitalized words and common patterns.
        """
        entities = {
            "documents": [],
            "people": [],
            "topics": [],
            "objects": []
        }
        
        # Look for document references
        for msg in conversation_history[-10:]:  # Last 10 messages
            content = msg.get("content", "")
            
            # Find document names (e.g., "document.pdf", "file.docx")
            doc_pattern = r'\b[\w-]+\.(pdf|docx?|xlsx?|txt|md)\b'
            docs = re.findall(doc_pattern, content, re.IGNORECASE)
            entities["documents"].extend([d[0] for d in docs])
            
            # Find capitalized words (potential names/topics)
            cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            entities["topics"].extend(cap_words)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]  # Keep top 5
        
        return entities
    
    def _detect_topics(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> Set[str]:
        """Detect main topics from conversation history."""
        topics = set()
        
        # Common topic keywords
        topic_keywords = [
            'document', 'file', 'ocr', 'conversion', 'indexing',
            'search', 'query', 'chat', 'analysis', 'summary'
        ]
        
        for msg in conversation_history[-10:]:
            content = msg.get("content", "").lower()
            for keyword in topic_keywords:
                if keyword in content:
                    topics.add(keyword)
        
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
        This is a simple implementation - can be enhanced with NLP models.
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
        
        return resolved

