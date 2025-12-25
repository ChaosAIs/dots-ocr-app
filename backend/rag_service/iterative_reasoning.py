"""
Iterative Reasoning Engine - Unified reasoning for GraphRAG and vector-only retrieval.

This module implements the Graph-R1 style iterative reasoning loop that works
with both GraphRAG (Neo4j + Qdrant) and vector-only (Qdrant) backends.

The reasoning process:
1. Generate initial query from user question
2. Retrieve knowledge (graph or vector)
3. Apply deep reasoning (5-stage analysis)
4. Decide: answer if sufficient, or generate new query
5. Repeat until answer found or max_steps reached
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .unified_retriever import UnifiedRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# Environment configuration
ITERATIVE_REASONING_ENABLED = os.getenv("ITERATIVE_REASONING_ENABLED", "true").lower() == "true"
DEFAULT_MAX_STEPS = int(os.getenv("ITERATIVE_REASONING_DEFAULT_MAX_STEPS", "3"))


@dataclass
class ReasoningResult:
    """
    Result from iterative reasoning.

    Contains the final answer (if generated) and all accumulated knowledge.
    """
    final_answer: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    parent_chunks: List[Dict[str, Any]] = field(default_factory=list)  # Parent context chunks
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    sources: set = field(default_factory=set)
    queries_made: List[str] = field(default_factory=list)
    steps_taken: int = 0
    reasoning_mode: str = "iterative"  # "iterative" or "single-shot"
    initial_query: str = ""  # Original user question for query relevance scoring

    def has_answer(self) -> bool:
        """Check if a final answer was generated."""
        return bool(self.final_answer)


class IterativeReasoningEngine:
    """
    Unified iterative reasoning engine.

    Implements the Graph-R1 style think-query-retrieve-rethink loop
    that works with both GraphRAG and vector-only backends.
    """

    def __init__(
        self,
        graphrag_enabled: bool = False,
        max_steps: int = None,
        source_names: Optional[List[str]] = None,
        workspace_id: str = "default",
        accessible_doc_ids: Optional[set] = None,
        routed_document_ids: Optional[List[str]] = None
    ):
        """
        Initialize the iterative reasoning engine.

        Args:
            graphrag_enabled: Whether to use GraphRAG or vector-only retrieval
            max_steps: Maximum reasoning iterations (defaults to ITERATIVE_REASONING_DEFAULT_MAX_STEPS)
            source_names: DEPRECATED - use routed_document_ids instead
            workspace_id: Workspace ID for multi-tenant isolation
            accessible_doc_ids: Optional set of document IDs the user can access (for access control)
            routed_document_ids: List of document IDs from router (already filtered by access control).
                                 When provided, this takes precedence over accessible_doc_ids.
        """
        self.graphrag_enabled = graphrag_enabled
        self.max_steps = max_steps or DEFAULT_MAX_STEPS
        self.source_names = source_names  # DEPRECATED
        self.workspace_id = workspace_id
        self.accessible_doc_ids = accessible_doc_ids
        self.routed_document_ids = routed_document_ids

        # Determine which document IDs to use for filtering
        # Priority: routed_document_ids > accessible_doc_ids
        effective_doc_ids = None
        if routed_document_ids is not None and len(routed_document_ids) > 0:
            # Use routed document IDs (already filtered by router + access control)
            effective_doc_ids = set(routed_document_ids)
            logger.info(f"[IterativeReasoning] Using {len(routed_document_ids)} routed document IDs for filtering")
        elif accessible_doc_ids is not None:
            # Fallback to full accessible doc IDs
            effective_doc_ids = accessible_doc_ids
            logger.info(f"[IterativeReasoning] Using {len(accessible_doc_ids)} accessible document IDs for filtering")

        # Initialize retriever with effective document IDs
        self.retriever = UnifiedRetriever(
            graphrag_enabled=graphrag_enabled,
            source_names=None,  # DEPRECATED - not used
            workspace_id=workspace_id,
            accessible_doc_ids=effective_doc_ids
        )

        # LLM service (initialized lazily)
        self._llm_service = None

    def _get_llm_service(self):
        """Lazily initialize LLM service."""
        if self._llm_service is None:
            from .llm_service import get_llm_service
            self._llm_service = get_llm_service()
        return self._llm_service

    async def reason(
        self,
        question: str,
        progress_callback=None
    ) -> ReasoningResult:
        """
        Execute iterative reasoning to answer the question.

        This is the main entry point for the reasoning engine. It:
        1. Generates an initial query
        2. Iteratively retrieves and reasons
        3. Returns when answer is found or max_steps reached

        Args:
            question: User's original question
            progress_callback: Optional callback for progress updates

        Returns:
            ReasoningResult with answer and accumulated knowledge
        """
        if not ITERATIVE_REASONING_ENABLED:
            logger.info("[IterativeReasoning] Disabled, using single-shot retrieval")
            return await self._single_shot_retrieval(question)

        logger.info("=" * 70)
        logger.info("[IterativeReasoning] STARTING ITERATIVE REASONING")
        logger.info("=" * 70)
        logger.info(f"[IterativeReasoning] Question: {question}")
        logger.info(f"[IterativeReasoning] Max steps: {self.max_steps}")
        logger.info(f"[IterativeReasoning] GraphRAG enabled: {self.graphrag_enabled}")
        logger.info(f"[IterativeReasoning] Routed document IDs: {len(self.routed_document_ids) if self.routed_document_ids else 'None'}")
        logger.info(f"[IterativeReasoning] Retriever type: {type(self.retriever).__name__}")
        logger.info(f"[IterativeReasoning] Retriever.graphrag_enabled: {self.retriever.graphrag_enabled}")
        logger.info("-" * 70)

        # Import prompts
        from .graph_rag.agent_prompts import (
            AGENT_THINK_PROMPT,
            INITIAL_QUERY_PROMPT,
            KNOWLEDGE_FORMAT_TEMPLATE,
            NO_KNOWLEDGE_TEMPLATE,
        )

        # State tracking
        step = 0
        retrieved_knowledge = []
        queries_made = []
        is_complete = False
        final_answer = ""

        # Accumulated results
        all_entities = []
        all_relationships = []
        all_chunks = []
        all_parent_chunks = []
        all_sources = set()
        seen_entity_ids = set()
        seen_chunk_ids = set()
        seen_parent_chunk_ids = set()

        # Step 1: Generate initial query (LLM-enhanced search query)
        if progress_callback:
            await progress_callback("Generating initial search query...", 10)

        logger.info("[IterativeReasoning] Generating initial query...")
        current_query = await self._generate_initial_query(question, INITIAL_QUERY_PROMPT)

        # Log query comparison for debugging
        logger.info("-" * 70)
        logger.info("[IterativeReasoning] QUERY COMPARISON:")
        logger.info(f"  Original question: '{question}'")
        logger.info(f"  Generated query:   '{current_query}'")
        logger.info(f"  Query changed:     {question != current_query}")
        logger.info("-" * 70)

        # Iterative reasoning loop
        while step < self.max_steps and not is_complete:
            step += 1

            logger.info("=" * 50)
            logger.info(f"[IterativeReasoning] ITERATION {step}/{self.max_steps}")
            logger.info("=" * 50)
            logger.info(f"[IterativeReasoning] Current query: '{current_query}'")

            if progress_callback:
                progress_pct = 20 + (step * 60 // self.max_steps)
                await progress_callback(f"Searching (round {step}/{self.max_steps})...", progress_pct)

            # Step 2: Retrieve knowledge
            logger.info("[IterativeReasoning] Retrieving knowledge...")
            result = await self.retriever.retrieve(current_query, top_k=20)

            # Track this retrieval step
            knowledge_step = {
                "query": current_query,
                "mode": result.retrieval_mode,
                "entities": result.entities,
                "relationships": result.relationships,
                "chunks": result.chunks,
                "parent_chunks": result.parent_chunks,
            }
            retrieved_knowledge.append(knowledge_step)
            queries_made.append(current_query)

            logger.info(f"[IterativeReasoning] Retrieved:")
            logger.info(f"  - Entities: {len(result.entities)}")
            logger.info(f"  - Relationships: {len(result.relationships)}")
            logger.info(f"  - Chunks: {len(result.chunks)}")
            logger.info(f"  - Parent chunks: {len(result.parent_chunks)}")
            logger.info(f"  - Sources: {result.sources}")

            # Accumulate results (deduplicate)
            for entity in result.entities:
                entity_id = entity.get("id")
                if entity_id and entity_id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity_id)
                elif not entity_id:
                    all_entities.append(entity)

            for chunk in result.chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
                elif not chunk_id:
                    all_chunks.append(chunk)

            # Accumulate parent chunks (deduplicate)
            for parent in result.parent_chunks:
                parent_id = parent.get("chunk_id", parent.get("metadata", {}).get("chunk_id", ""))
                if parent_id and parent_id not in seen_parent_chunk_ids:
                    all_parent_chunks.append(parent)
                    seen_parent_chunk_ids.add(parent_id)
                elif not parent_id:
                    all_parent_chunks.append(parent)

            all_relationships.extend(result.relationships)
            all_sources.update(result.sources)

            # Step 3: Think - decide to continue or answer
            logger.info("-" * 50)
            logger.info("[IterativeReasoning] THINKING: Should I continue or answer?")

            if progress_callback:
                await progress_callback(f"Analyzing results (round {step})...", progress_pct + 10)

            # Format knowledge summary for LLM
            knowledge_summary = self._format_knowledge_summary(
                retrieved_knowledge,
                KNOWLEDGE_FORMAT_TEMPLATE,
                NO_KNOWLEDGE_TEMPLATE
            )

            # Format previous queries for anti-duplication
            previous_queries_str = ", ".join(f'"{q}"' for q in queries_made) if queries_made else "None"

            think_prompt = AGENT_THINK_PROMPT.format(
                question=question,
                step=step,
                max_steps=self.max_steps,
                retrieved_knowledge=knowledge_summary,
                previous_queries=previous_queries_str,
            )

            # Get LLM response
            # 4000 tokens allows for comprehensive answers with multiple products/items
            from langchain_core.messages import HumanMessage
            llm = self._get_llm_service().get_chat_model(temperature=0.2, num_predict=4000)
            think_response_msg = await llm.ainvoke([HumanMessage(content=think_prompt)])
            think_response = think_response_msg.content.strip()

            logger.info(f"[IterativeReasoning] LLM response preview: {think_response[:200]}...")
            logger.info(f"[IterativeReasoning] LLM response length: {len(think_response)} chars")

            # Parse response for <answer> or <query> tags
            answer_match = re.search(r'<answer>(.*?)</answer>', think_response, re.DOTALL)
            next_query_match = re.search(r'<query>(.*?)</query>', think_response, re.DOTALL)

            has_answer_tag = '<answer>' in think_response
            has_query_tag = '<query>' in think_response
            logger.info(f"[IterativeReasoning] Tags: <answer>={has_answer_tag}, <query>={has_query_tag}")

            # Handle answer
            if answer_match:
                final_answer = answer_match.group(1).strip()
                is_complete = True
                logger.info("[IterativeReasoning] DECISION: ANSWER found")
                logger.info(f"[IterativeReasoning] Answer preview: {final_answer[:200]}...")
            elif has_answer_tag:
                # Handle unclosed <answer> tag
                answer_start = think_response.find('<answer>')
                if answer_start != -1:
                    final_answer = think_response[answer_start + 8:].strip()
                    is_complete = True
                    logger.info("[IterativeReasoning] DECISION: ANSWER found (unclosed tag)")
            elif next_query_match:
                next_query = next_query_match.group(1).strip()
                if next_query and next_query not in queries_made:
                    current_query = next_query
                    logger.info(f"[IterativeReasoning] DECISION: CONTINUE with new query: '{next_query}'")
                else:
                    is_complete = True
                    logger.info("[IterativeReasoning] DECISION: TERMINATE (duplicate query)")
            elif has_query_tag:
                # Handle unclosed <query> tag
                query_start = think_response.find('<query>')
                if query_start != -1:
                    next_query = think_response[query_start + 7:].strip()
                    if next_query and next_query not in queries_made:
                        current_query = next_query
                        logger.info(f"[IterativeReasoning] DECISION: CONTINUE with new query (unclosed): '{next_query}'")
                    else:
                        is_complete = True
            else:
                # No valid tags found
                is_complete = True
                logger.info("[IterativeReasoning] DECISION: TERMINATE (no valid tags)")

        # Final step check
        if step >= self.max_steps and not final_answer:
            logger.info(f"[IterativeReasoning] MAX STEPS REACHED ({self.max_steps})")
            # Generate answer from accumulated knowledge if none was provided
            if progress_callback:
                await progress_callback("Generating final answer...", 90)
            final_answer = await self._generate_final_answer(question, retrieved_knowledge)

        # Log summary
        logger.info("=" * 70)
        logger.info("[IterativeReasoning] REASONING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"[IterativeReasoning] Summary:")
        logger.info(f"  - Steps taken: {step}")
        logger.info(f"  - Queries made: {queries_made}")
        logger.info(f"  - Total entities: {len(all_entities)}")
        logger.info(f"  - Total relationships: {len(all_relationships)}")
        logger.info(f"  - Total chunks: {len(all_chunks)}")
        logger.info(f"  - Total parent chunks: {len(all_parent_chunks)}")
        logger.info(f"  - Sources: {all_sources}")
        logger.info(f"  - Final answer: {'Yes' if final_answer else 'No'} ({len(final_answer)} chars)")
        logger.info("=" * 70)

        return ReasoningResult(
            final_answer=final_answer,
            chunks=all_chunks,
            parent_chunks=all_parent_chunks,
            entities=all_entities,
            relationships=all_relationships,
            sources=all_sources,
            queries_made=queries_made,
            steps_taken=step,
            reasoning_mode="iterative",
            initial_query=question,
        )

    async def _single_shot_retrieval(self, question: str) -> ReasoningResult:
        """
        Fallback single-shot retrieval when iterative reasoning is disabled.

        Args:
            question: User's question

        Returns:
            ReasoningResult with chunks from single retrieval
        """
        logger.info("[IterativeReasoning] Single-shot retrieval mode")

        result = await self.retriever.retrieve(question, top_k=20)

        return ReasoningResult(
            final_answer="",  # No answer generated, let caller handle
            chunks=result.chunks,
            parent_chunks=result.parent_chunks,
            entities=result.entities,
            relationships=result.relationships,
            sources=result.sources,
            queries_made=[question],
            steps_taken=1,
            reasoning_mode="single-shot",
            initial_query=question,
        )

    async def _generate_initial_query(
        self,
        question: str,
        prompt_template: str
    ) -> str:
        """
        Generate initial search query from user question.

        This method uses an LLM to transform the user's natural language question
        into an optimized search query with:
        - Entity-focused terms
        - Relevant keywords and synonyms
        - Specific context for better retrieval

        Args:
            question: User's original question
            prompt_template: INITIAL_QUERY_PROMPT template

        Returns:
            Generated query string (enhanced for retrieval)
        """
        logger.info("[IterativeReasoning] === QUERY GENERATION START ===")
        logger.info(f"[IterativeReasoning] Original question: '{question}'")

        prompt = prompt_template.format(question=question)
        logger.debug(f"[IterativeReasoning] Using prompt template (first 200 chars): {prompt[:200]}...")

        from langchain_core.messages import HumanMessage
        llm = self._get_llm_service().get_query_model(temperature=0.1, num_predict=200)

        logger.info("[IterativeReasoning] Invoking LLM for query generation...")
        response_msg = await llm.ainvoke([HumanMessage(content=prompt)])
        response = response_msg.content.strip()

        logger.info(f"[IterativeReasoning] LLM raw response: '{response}'")

        # Extract query from <query>...</query> tags
        query_match = re.search(r'<query>(.*?)</query>', response, re.DOTALL)
        if query_match:
            generated_query = query_match.group(1).strip()
            logger.info(f"[IterativeReasoning] Extracted query from tags: '{generated_query}'")
            logger.info("[IterativeReasoning] === QUERY GENERATION END (success) ===")
            return generated_query
        else:
            # Fallback to original question
            logger.warning(f"[IterativeReasoning] No <query> tags found in response, using original question")
            logger.info("[IterativeReasoning] === QUERY GENERATION END (fallback) ===")
            return question

    async def _generate_final_answer(
        self,
        question: str,
        retrieved_knowledge: List[Dict]
    ) -> str:
        """
        Generate final answer when max_steps reached without explicit answer.

        Args:
            question: Original user question
            retrieved_knowledge: All accumulated knowledge from iterations

        Returns:
            Generated answer string
        """
        # Build context from all retrieved knowledge
        context_parts = []
        for step_knowledge in retrieved_knowledge:
            for chunk in step_knowledge.get("chunks", []):
                content = chunk.get("page_content", chunk.get("content", ""))
                source = chunk.get("metadata", {}).get("source", "Unknown")
                if content:
                    context_parts.append(f"[Source: {source}]\n{content}")

        context = "\n\n---\n\n".join(context_parts[:15])  # Limit context size

        prompt = f"""Based on the following retrieved information, answer the question.

## Question
{question}

## Retrieved Information
{context}

## Instructions
Provide a comprehensive answer based on the information above.
If the information is insufficient, acknowledge what's missing.
Always cite sources when possible.

Your answer:"""

        from langchain_core.messages import HumanMessage
        # 4000 tokens allows for comprehensive fallback answers with multiple products
        llm = self._get_llm_service().get_chat_model(temperature=0.3, num_predict=4000)
        response_msg = await llm.ainvoke([HumanMessage(content=prompt)])

        return response_msg.content.strip()

    def _format_knowledge_summary(
        self,
        retrieved_knowledge: List[Dict],
        format_template: str,
        no_knowledge_template: str
    ) -> str:
        """
        Format retrieved knowledge for the AGENT_THINK_PROMPT.

        Args:
            retrieved_knowledge: List of knowledge from each step
            format_template: KNOWLEDGE_FORMAT_TEMPLATE
            no_knowledge_template: NO_KNOWLEDGE_TEMPLATE

        Returns:
            Formatted knowledge summary string
        """
        if not retrieved_knowledge:
            return "No knowledge retrieved yet."

        summaries = []
        for i, knowledge in enumerate(retrieved_knowledge, 1):
            entities = knowledge.get("entities", [])
            relationships = knowledge.get("relationships", [])
            chunks = knowledge.get("chunks", [])
            query = knowledge.get("query", "")

            if entities or relationships or chunks:
                entities_str = self._format_entities(entities)
                relationships_str = self._format_relationships(relationships)
                chunks_str = self._format_chunks(chunks)

                # Extract sources
                sources = set()
                for chunk in chunks:
                    source = chunk.get("metadata", {}).get("source", "")
                    if source:
                        sources.add(source)
                sources_str = ", ".join(sources) if sources else "None"

                summaries.append(
                    format_template.format(
                        step=i,
                        query=query,
                        entities=entities_str,
                        relationships=relationships_str,
                        chunks=chunks_str,
                        entity_count=len(entities),
                        relationship_count=len(relationships),
                        chunk_count=len(chunks),
                        sources=sources_str,
                    )
                )
            else:
                summaries.append(no_knowledge_template.format(step=i, query=query))

        return "\n".join(summaries)

    def _format_entities(self, entities: List[Dict]) -> str:
        """Format entity list for LLM prompt."""
        if not entities:
            return "None"

        lines = []
        for e in entities[:10]:
            name = e.get("name", e.get("entity_name", "Unknown"))
            entity_type = e.get("entity_type", "Unknown")
            desc = e.get("description", "")[:200]
            score = e.get("_score", 0)
            lines.append(f"- **{name}** ({entity_type}): {desc} [score: {score:.2f}]")

        if len(entities) > 10:
            lines.append(f"... and {len(entities) - 10} more entities")

        return "\n".join(lines) if lines else "None"

    def _format_relationships(self, relationships: List[Dict]) -> str:
        """Format relationship list for LLM prompt."""
        if not relationships:
            return "None"

        lines = []
        for r in relationships[:10]:
            src = r.get("src_name", r.get("src_entity_id", "?"))
            tgt = r.get("tgt_name", r.get("tgt_entity_id", "?"))
            desc = r.get("description", "")[:150]
            keywords = r.get("keywords", "")
            score = r.get("_score", 0)
            lines.append(f"- **{src}** → **{tgt}**: {desc} [{keywords}] [score: {score:.2f}]")

        if len(relationships) > 10:
            lines.append(f"... and {len(relationships) - 10} more relationships")

        return "\n".join(lines) if lines else "None"

    def _format_chunks(self, chunks: List[Dict]) -> str:
        """Format document chunks for LLM prompt, grouped by source."""
        if not chunks:
            return "None"

        chunks_per_source = int(os.getenv("ITERATIVE_REASONING_CHUNKS_PER_SOURCE", "3"))

        # Group chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "Unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

        lines = []
        total_shown = 0

        for source, source_chunks in chunks_by_source.items():
            # Sort by score
            sorted_chunks = sorted(
                source_chunks,
                key=lambda c: c.get("_score", 0),
                reverse=True
            )

            lines.append(f"\n**From: {source}** ({len(source_chunks)} chunks)")

            for idx, chunk in enumerate(sorted_chunks[:chunks_per_source], 1):
                content = chunk.get("page_content", chunk.get("content", ""))
                # Increased from 500 to 800 to capture complete product information
                if len(content) > 800:
                    content = content[:800] + "..."
                score = chunk.get("_score", 0)
                lines.append(f"  Chunk {idx} (score: {score:.2f}):\n  {content}\n")
                total_shown += 1

            if len(sorted_chunks) > chunks_per_source:
                lines.append(f"  ... and {len(sorted_chunks) - chunks_per_source} more chunks\n")

        return "\n".join(lines) if lines else "None"
