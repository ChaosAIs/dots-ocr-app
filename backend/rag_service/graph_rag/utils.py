"""
Utility functions for GraphRAG.

This module provides helper functions for:
- Parsing LLM extraction output
- Entity/relationship deduplication
- Text processing
"""

import re
import hashlib
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import asdict

from .base import Entity, Relationship

logger = logging.getLogger(__name__)


def generate_entity_id(name: str, entity_type: str, workspace_id: str = "default") -> str:
    """Generate a unique ID for an entity based on name and type."""
    key = f"{workspace_id}:{entity_type}:{name.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()


def generate_relationship_id(
    src_name: str, tgt_name: str, workspace_id: str = "default"
) -> str:
    """Generate a unique ID for a relationship."""
    # Sort names to ensure consistent ID regardless of direction
    names = sorted([src_name.lower().strip(), tgt_name.lower().strip()])
    key = f"{workspace_id}:{names[0]}:{names[1]}"
    return hashlib.md5(key.encode()).hexdigest()


def parse_entity_line(line: str, source_chunk_id: str = "") -> Optional[Entity]:
    """
    Parse an entity line from LLM output.

    Expected format:
    (entity|<entity_type>|<entity_name>|<entity_description>|<importance_score>)
    """
    line = line.strip()
    if not line.startswith("(entity|"):
        return None

    # Remove parentheses and split
    content = line[1:-1] if line.endswith(")") else line[1:]
    parts = content.split("|")

    if len(parts) < 5:
        logger.warning(f"Invalid entity format: {line}")
        return None

    try:
        entity_type = parts[1].strip()
        name = parts[2].strip()
        description = parts[3].strip()
        key_score = int(parts[4].strip()) if parts[4].strip().isdigit() else 50

        entity_id = generate_entity_id(name, entity_type)

        return Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=description,
            source_chunk_id=source_chunk_id,
            key_score=key_score,
        )
    except Exception as e:
        logger.warning(f"Failed to parse entity: {line}, error: {e}")
        return None


def parse_relationship_line(
    line: str, source_chunk_id: str = ""
) -> Optional[Relationship]:
    """
    Parse a relationship line from LLM output.

    Expected format:
    (relationship|<source_entity>|<target_entity>|<description>|<keywords>|<weight>)
    """
    line = line.strip()
    if not line.startswith("(relationship|"):
        return None

    # Remove parentheses and split
    content = line[1:-1] if line.endswith(")") else line[1:]
    parts = content.split("|")

    if len(parts) < 6:
        logger.warning(f"Invalid relationship format: {line}")
        return None

    try:
        src_name = parts[1].strip()
        tgt_name = parts[2].strip()
        description = parts[3].strip()
        keywords = parts[4].strip()

        # Parse weight - handle malformed values (e.g., "uncertain, film, director" instead of float)
        weight_str = parts[5].strip() if len(parts) > 5 else ""
        try:
            weight = float(weight_str) if weight_str else 1.0
        except ValueError:
            # If weight is not a valid float, default to 1.0
            logger.debug(f"Invalid weight value '{weight_str}', using default 1.0")
            weight = 1.0

        rel_id = generate_relationship_id(src_name, tgt_name)
        src_id = generate_entity_id(src_name, "unknown")
        tgt_id = generate_entity_id(tgt_name, "unknown")

        return Relationship(
            id=rel_id,
            src_entity_id=src_id,
            tgt_entity_id=tgt_id,
            description=description,
            keywords=keywords,
            weight=weight,
            source_chunk_id=source_chunk_id,
            metadata={"src_name": src_name, "tgt_name": tgt_name},
        )
    except Exception as e:
        logger.warning(f"Failed to parse relationship: {line}, error: {e}")
        return None


def parse_extraction_output(
    output: str, source_chunk_id: str = ""
) -> Tuple[List[Entity], List[Relationship]]:
    """
    Parse the complete LLM extraction output.

    Returns:
        Tuple of (entities, relationships)
    """
    entities = []
    relationships = []

    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("(entity|"):
            entity = parse_entity_line(line, source_chunk_id)
            if entity:
                entities.append(entity)
        elif line.startswith("(relationship|"):
            rel = parse_relationship_line(line, source_chunk_id)
            if rel:
                relationships.append(rel)

    return entities, relationships


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    """
    Deduplicate entities by ID, keeping the one with highest key_score.
    """
    entity_map: Dict[str, Entity] = {}

    for entity in entities:
        if entity.id not in entity_map:
            entity_map[entity.id] = entity
        else:
            # Keep the one with higher key_score
            if entity.key_score > entity_map[entity.id].key_score:
                entity_map[entity.id] = entity

    return list(entity_map.values())


def deduplicate_relationships(relationships: List[Relationship]) -> List[Relationship]:
    """
    Deduplicate relationships by ID, keeping the one with highest weight.
    """
    rel_map: Dict[str, Relationship] = {}

    for rel in relationships:
        if rel.id not in rel_map:
            rel_map[rel.id] = rel
        else:
            # Keep the one with higher weight
            if rel.weight > rel_map[rel.id].weight:
                rel_map[rel.id] = rel

    return list(rel_map.values())


def merge_entity_descriptions(entities: List[Entity]) -> Dict[str, str]:
    """
    Merge descriptions for entities with the same ID.

    Returns:
        Dict mapping entity_id to merged description
    """
    descriptions: Dict[str, List[str]] = {}

    for entity in entities:
        if entity.id not in descriptions:
            descriptions[entity.id] = []
        if entity.description and entity.description not in descriptions[entity.id]:
            descriptions[entity.id].append(entity.description)

    return {
        entity_id: " | ".join(descs)
        for entity_id, descs in descriptions.items()
    }


def entity_to_dict(entity: Entity) -> Dict[str, Any]:
    """Convert an Entity to a dictionary for storage."""
    return asdict(entity)


def relationship_to_dict(rel: Relationship) -> Dict[str, Any]:
    """Convert a Relationship to a dictionary for storage."""
    return asdict(rel)


def extract_entity_names_from_query(query: str) -> List[str]:
    """
    Extract potential entity names from a query using simple heuristics.

    This is a fallback when LLM-based extraction is not available.
    """
    # Remove common stop words and punctuation
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "until", "while", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "about", "tell", "me", "explain",
    }

    # Split query into words
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\b', query)

    # Filter out stop words and short words
    potential_entities = [
        word for word in words
        if word.lower() not in stop_words and len(word) > 2
    ]

    # Also look for capitalized phrases (potential proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

    # Combine and deduplicate
    all_entities = list(set(potential_entities + capitalized))

    return all_entities


def truncate_text(text: str, max_tokens: int = 4000) -> str:
    """
    Truncate text to approximately max_tokens.

    Uses a simple word-based approximation (1 token ≈ 0.75 words).
    """
    max_words = int(max_tokens * 0.75)
    words = text.split()

    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + "..."


def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing common artifacts.
    """
    # Remove markdown code blocks
    response = re.sub(r'```[a-z]*\n?', '', response)
    response = re.sub(r'```', '', response)

    # Remove leading/trailing whitespace
    response = response.strip()

    return response

