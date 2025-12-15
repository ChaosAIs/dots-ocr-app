"""
Entity Extractor for GraphRAG.

This module implements entity and relationship extraction from text chunks
using LLM with an iterative gleaning loop for thorough extraction.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional

from .base import Entity, Relationship
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_CONTINUE_EXTRACTION_PROMPT,
    ENTITY_CHECK_CONTINUE_PROMPT,
)
from .utils import (
    parse_extraction_output,
    deduplicate_entities,
    deduplicate_relationships,
    clean_llm_response,
)

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts entities and relationships from text using LLM.

    Implements the Graph-R1 gleaning loop pattern for thorough extraction:
    1. Initial extraction
    2. Continue extraction (find missed entities)
    3. Check if more iterations needed
    4. Repeat until complete or max iterations reached
    """

    def __init__(
        self,
        llm_client=None,
        max_gleaning: int = None,
        cache_storage=None,
        min_entity_score: int = None,
    ):
        """
        Initialize the entity extractor.

        Args:
            llm_client: LLM client for extraction (uses configured RAG LLM backend)
            max_gleaning: Maximum gleaning iterations (default from env)
            cache_storage: Optional LLM cache storage
            min_entity_score: Minimum importance score for entities (default from env)
        """
        self.llm_client = llm_client
        self.max_gleaning = max_gleaning or int(
            os.getenv("GRAPH_RAG_MAX_GLEANING", "0")
        )
        self.min_entity_score = min_entity_score or int(
            os.getenv("GRAPH_RAG_MIN_ENTITY_SCORE", "60")
        )
        self.cache_storage = cache_storage
        self._cached_llm_client = None

    def _get_llm_client(self):
        """Get or create the LLM client using the configured RAG LLM backend."""
        if self.llm_client is not None:
            return self.llm_client

        if self._cached_llm_client is None:
            # Use the same LLM service as the RAG agent
            from ..llm_service import get_llm_service

            llm_service = get_llm_service()
            # Use query model for entity extraction (lower temperature for consistency)
            self._cached_llm_client = llm_service.get_query_model(
                temperature=0.1,
                num_ctx=4096,
                num_predict=2048,
            )
            logger.info(f"EntityExtractor using LLM service: {type(llm_service).__name__}")

        return self._cached_llm_client

    async def _call_llm(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Call the LLM with optional conversation history."""
        llm = self._get_llm_client()

        # Check cache first
        if self.cache_storage:
            cached = await self.cache_storage.get_cached_response(prompt)
            if cached:
                logger.debug("Using cached LLM response")
                return cached

        # Build messages
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        # Call LLM
        try:
            response = await llm.ainvoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)
            result = clean_llm_response(result)

            # Cache the response
            if self.cache_storage:
                await self.cache_storage.cache_response(prompt, result)

            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def _initial_extraction(
        self, text: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship], str]:
        """
        Perform initial entity extraction with importance filtering.

        Returns:
            Tuple of (filtered_entities, relationships, raw_response)
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        response = await self._call_llm(prompt)

        entities, relationships = parse_extraction_output(response, chunk_id)

        # Filter entities by importance score
        original_count = len(entities)
        entities = [e for e in entities if e.key_score >= self.min_entity_score]
        filtered_count = original_count - len(entities)

        if filtered_count > 0:
            logger.info(
                f"Initial extraction: {len(entities)} entities (filtered {filtered_count} "
                f"low-importance entities with score < {self.min_entity_score}), "
                f"{len(relationships)} relationships"
            )
        else:
            logger.info(
                f"Initial extraction: {len(entities)} entities, {len(relationships)} relationships"
            )

        return entities, relationships, response

    async def _continue_extraction(
        self, history: List[Dict[str, str]], chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship], str]:
        """
        Continue extraction to find missed entities with importance filtering.

        Returns:
            Tuple of (filtered_entities, relationships, raw_response)
        """
        response = await self._call_llm(ENTITY_CONTINUE_EXTRACTION_PROMPT, history)

        entities, relationships = parse_extraction_output(response, chunk_id)

        # Filter entities by importance score
        original_count = len(entities)
        entities = [e for e in entities if e.key_score >= self.min_entity_score]
        filtered_count = original_count - len(entities)

        if filtered_count > 0:
            logger.info(
                f"Continue extraction: {len(entities)} entities (filtered {filtered_count} "
                f"low-importance), {len(relationships)} relationships"
            )
        else:
            logger.info(
                f"Continue extraction: {len(entities)} entities, {len(relationships)} relationships"
            )

        return entities, relationships, response

    async def _check_should_continue(self, history: List[Dict[str, str]]) -> bool:
        """Check if more extraction iterations are needed."""
        response = await self._call_llm(ENTITY_CHECK_CONTINUE_PROMPT, history)
        response = response.strip().upper()

        should_continue = response.startswith("YES")
        logger.debug(f"Should continue extraction: {should_continue}")

        return should_continue

    def _build_history(
        self,
        text: str,
        initial_response: str,
        additional_responses: List[str] = None,
    ) -> List[Dict[str, str]]:
        """Build conversation history for gleaning loop."""
        history = [
            {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=text)},
            {"role": "assistant", "content": initial_response},
        ]

        if additional_responses:
            for response in additional_responses:
                history.append(
                    {"role": "user", "content": ENTITY_CONTINUE_EXTRACTION_PROMPT}
                )
                history.append({"role": "assistant", "content": response})

        return history

    async def extract_with_gleaning(
        self,
        text: str,
        chunk_id: str = "",
        max_gleaning: int = None,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships with iterative gleaning.

        This implements the Graph-R1 gleaning loop pattern:
        1. Initial extraction
        2. Continue extraction (find missed entities)
        3. Check if more iterations needed
        4. Repeat until complete or max iterations reached

        Args:
            text: Text to extract from
            chunk_id: Source chunk ID for tracking
            max_gleaning: Override max gleaning iterations

        Returns:
            Tuple of (deduplicated_entities, deduplicated_relationships)
        """
        max_iterations = max_gleaning or self.max_gleaning
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        additional_responses: List[str] = []

        # Step 1: Initial extraction
        entities, relationships, initial_response = await self._initial_extraction(
            text, chunk_id
        )
        all_entities.extend(entities)
        all_relationships.extend(relationships)

        # Step 2-4: Gleaning loop
        for iteration in range(max_iterations):
            # Build conversation history
            history = self._build_history(text, initial_response, additional_responses)

            # Check if we should continue
            should_continue = await self._check_should_continue(history)
            if not should_continue:
                logger.info(f"Gleaning complete after {iteration} iterations")
                break

            # Continue extraction
            entities, relationships, response = await self._continue_extraction(
                history, chunk_id
            )

            if not entities and not relationships:
                logger.info("No new entities found, stopping gleaning")
                break

            all_entities.extend(entities)
            all_relationships.extend(relationships)
            additional_responses.append(response)

            logger.info(
                f"Gleaning iteration {iteration + 1}: "
                f"+{len(entities)} entities, +{len(relationships)} relationships"
            )

        # Deduplicate results
        final_entities = deduplicate_entities(all_entities)
        final_relationships = deduplicate_relationships(all_relationships)

        logger.info(
            f"Final extraction: {len(final_entities)} entities, "
            f"{len(final_relationships)} relationships"
        )

        return final_entities, final_relationships

    async def extract_simple(
        self,
        text: str,
        chunk_id: str = "",
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Simple extraction without gleaning loop.

        Use this for faster extraction when thoroughness is not critical.
        """
        entities, relationships, _ = await self._initial_extraction(text, chunk_id)
        return deduplicate_entities(entities), deduplicate_relationships(relationships)

    async def extract_batch(
        self,
        chunks: List[Dict[str, Any]],
        use_gleaning: bool = True,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: List of dicts with 'id' and 'content' keys
            use_gleaning: Whether to use gleaning loop

        Returns:
            Tuple of (all_entities, all_relationships)
        """
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            content = chunk.get("page_content", "")

            if not content.strip():
                continue

            logger.info(f"Processing chunk {i + 1}/{len(chunks)}: {chunk_id}")

            try:
                if use_gleaning:
                    entities, relationships = await self.extract_with_gleaning(
                        content, chunk_id
                    )
                else:
                    entities, relationships = await self.extract_simple(
                        content, chunk_id
                    )

                all_entities.extend(entities)
                all_relationships.extend(relationships)

            except Exception as e:
                logger.error(f"Failed to extract from chunk {chunk_id}: {e}")
                continue

        # Final deduplication across all chunks
        final_entities = deduplicate_entities(all_entities)
        final_relationships = deduplicate_relationships(all_relationships)

        logger.info(
            f"Batch extraction complete: {len(final_entities)} entities, "
            f"{len(final_relationships)} relationships from {len(chunks)} chunks"
        )

        return final_entities, final_relationships

