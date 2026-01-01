"""
GraphRAG Orchestrator - Main entry point for graph-based retrieval.

Following Graph-R1 paper design:
1. Query mode detection (LOCAL, GLOBAL, HYBRID)
2. Entity/relationship retrieval based on mode (using vector similarity)
3. Iterative think-query-retrieve-rethink cycle (optional, controlled by max_steps)
4. Context building from graph and vector stores
5. Always combine graph results with Qdrant vector search results
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import QueryMode, QueryParam
from .query_mode_detector import QueryModeDetector
from .utils import extract_entity_names_from_query

logger = logging.getLogger(__name__)

# Feature flags
GRAPH_RAG_QUERY_ENABLED = os.getenv("GRAPH_RAG_QUERY_ENABLED", "false").lower() == "true"
DEFAULT_MODE = os.getenv("GRAPH_RAG_DEFAULT_MODE", "auto")
GRAPH_RAG_VECTOR_SEARCH_ENABLED = os.getenv("GRAPH_RAG_VECTOR_SEARCH_ENABLED", "true").lower() == "true"


@dataclass
class GraphRAGContext:
    """Context retrieved from GraphRAG."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    mode: QueryMode
    enhanced_query: str
    final_answer: str = ""  # Answer from iterative reasoning (if available)


class GraphRAG:
    """
    Main GraphRAG orchestrator.

    Handles query processing with graph-enhanced retrieval:
    - Detects query mode (or uses specified mode)
    - Retrieves relevant entities and relationships using vector similarity
    - Builds context from graph traversal
    - Returns enhanced context for LLM response generation
    """

    def __init__(
        self,
        workspace_id: str = "default",
        query_mode_detector: QueryModeDetector = None,
    ):
        """
        Initialize GraphRAG.

        Args:
            workspace_id: Workspace ID for multi-tenant isolation
            query_mode_detector: Optional custom query mode detector
        """
        self.workspace_id = workspace_id
        self.query_mode_detector = query_mode_detector or QueryModeDetector()
        self._storage_initialized = False
        self._graph_storage = None
        self._embedding_service = None
        self._llm_service = None
        self._vector_search_enabled = GRAPH_RAG_VECTOR_SEARCH_ENABLED
        self._graphrag_status_cache = {}  # Cache for document graphrag status checks

    def _check_documents_graphrag_status(self, document_ids: set) -> Dict[str, Any]:
        """
        Check if any of the referenced documents have pending GraphRAG indexing.

        When GraphRAG indexing is still pending/processing for some documents,
        vector search should be enabled as a fallback even if GRAPH_RAG_VECTOR_SEARCH_ENABLED=false.

        Args:
            document_ids: Set of document IDs to check

        Returns:
            Dict with status information:
                - all_completed: True if all documents have completed GraphRAG
                - pending_count: Number of documents with pending GraphRAG
                - should_enable_vector_search: True if vector search should be enabled
        """
        if not document_ids:
            return {
                "all_completed": True,
                "pending_count": 0,
                "should_enable_vector_search": False
            }

        # Convert to list of strings for caching
        doc_id_list = sorted([str(doc_id) for doc_id in document_ids])
        cache_key = ",".join(doc_id_list[:10])  # Use first 10 IDs as cache key

        # Check cache first (valid for current request)
        if cache_key in self._graphrag_status_cache:
            return self._graphrag_status_cache[cache_key]

        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            with get_db_session() as db:
                repo = DocumentRepository(db)
                status = repo.check_graphrag_status_by_ids(doc_id_list)

                result = {
                    "all_completed": status["all_completed"],
                    "pending_count": status["pending_count"],
                    "pending_doc_ids": status["pending_doc_ids"],
                    "should_enable_vector_search": not status["all_completed"]
                }

                # Cache the result
                self._graphrag_status_cache[cache_key] = result

                if not status["all_completed"]:
                    logger.info(
                        f"[GraphRAG] Documents with pending GraphRAG indexing detected: "
                        f"{status['pending_count']} pending, {status['completed_count']} completed. "
                        f"Vector search will be ENABLED as fallback."
                    )

                return result

        except Exception as e:
            logger.warning(f"[GraphRAG] Failed to check document GraphRAG status: {e}")
            # On error, return safe defaults (enable vector search as fallback)
            return {
                "all_completed": False,
                "pending_count": 0,
                "should_enable_vector_search": True  # Enable vector search as safe fallback
            }

    async def _init_storage(self):
        """Initialize Neo4j storage backend lazily."""
        if self._storage_initialized:
            return

        logger.info("[GraphRAG Query] Initializing Neo4j storage...")

        try:
            from ..storage import Neo4jStorage

            self._graph_storage = Neo4jStorage(self.workspace_id)

            self._storage_initialized = True
            logger.info(f"[GraphRAG Query] Neo4j storage initialized successfully (storage={self._graph_storage is not None})")

        except ImportError as e:
            logger.error(f"[GraphRAG Query] Failed to import Neo4jStorage: {e}")
            import traceback
            logger.error(f"[GraphRAG Query] Import traceback:\n{traceback.format_exc()}")
            self._storage_initialized = True  # Mark as initialized to prevent retries
            raise
        except Exception as e:
            logger.error(f"[GraphRAG Query] Failed to initialize Neo4j storage: {e}")
            import traceback
            logger.error(f"[GraphRAG Query] Init traceback:\n{traceback.format_exc()}")
            self._storage_initialized = True  # Mark as initialized to prevent retries
            raise

    async def _init_embedding_service(self):
        """Initialize embedding service for query embedding."""
        if self._embedding_service is not None:
            return

        if not self._vector_search_enabled:
            return

        try:
            from ..local_qwen_embedding import LocalQwen3Embedding
            self._embedding_service = LocalQwen3Embedding()
            logger.info("[GraphRAG Query] Embedding service initialized")
        except Exception as e:
            logger.warning(f"[GraphRAG Query] Failed to init embedding service: {e}")
            self._vector_search_enabled = False

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for query.

        Args:
            query: Query string

        Returns:
            Embedding vector or None if not available
        """
        if not self._vector_search_enabled:
            return None

        await self._init_embedding_service()

        if not self._embedding_service:
            return None

        try:
            return self._embedding_service.embed_query(query)
        except Exception as e:
            logger.warning(f"[GraphRAG Query] Failed to embed query: {e}")
            return None

    def _get_chunks_for_entities(
        self,
        entities: List[Dict[str, Any]],
        accessible_doc_ids: set = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve source chunks from Qdrant for the given entities.

        Entities have source_chunk_ids that link back to the original chunks
        in the vector database. This method retrieves those chunks to provide
        the full text context for LLM response generation.

        Args:
            entities: List of entity dictionaries with source_chunk_ids
            accessible_doc_ids: Optional set of document IDs for access control filtering.
                               If provided, only chunks with document_id in this set are returned.

        Returns:
            List of chunk dictionaries with page_content and metadata
        """
        from ..vectorstore import get_chunks_by_ids

        # Collect all unique source_chunk_ids from entities
        chunk_ids = set()
        for entity in entities:
            source_chunk_ids = entity.get("source_chunk_ids", [])
            if isinstance(source_chunk_ids, list):
                chunk_ids.update(source_chunk_ids)
            elif isinstance(source_chunk_ids, str):
                # Handle case where it's stored as single string
                chunk_ids.add(source_chunk_ids)

        if not chunk_ids:
            logger.debug("[GraphRAG Query] No source_chunk_ids found in entities")
            return []

        logger.debug(f"[GraphRAG Query] Retrieving {len(chunk_ids)} chunks for entities")

        # Convert accessible_doc_ids set to list for Qdrant filter
        accessible_document_ids = None
        if accessible_doc_ids is not None:
            accessible_document_ids = [str(doc_id) for doc_id in accessible_doc_ids]
            logger.debug(f"[GraphRAG Query] Applying access control: {len(accessible_document_ids)} document IDs")

        # Retrieve chunks from Qdrant with access control
        try:
            docs = get_chunks_by_ids(
                list(chunk_ids),
                accessible_document_ids=accessible_document_ids
            )
            chunks = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                }
                for doc in docs
            ]
            logger.debug(f"[GraphRAG Query] Retrieved {len(chunks)} chunks from Qdrant")
            return chunks
        except Exception as e:
            logger.warning(f"[GraphRAG Query] Error retrieving chunks: {e}")
            return []

    async def _get_vector_search_chunks(
        self,
        query: str,
        top_k: int = 60,
        source_names: List[str] = None,
        accessible_doc_ids: set = None
    ) -> List[Dict[str, Any]]:
        """
        Perform direct Qdrant vector search for the query.

        This ensures we always have vector search results even if graph knowledge
        is not ready or incomplete.

        IMPORTANT: Vector search is dynamically enabled when any of the accessible
        documents have pending GraphRAG indexing, even if GRAPH_RAG_VECTOR_SEARCH_ENABLED=false.
        This ensures users can still search documents while graph indexing is in progress.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            source_names: Optional list of source names to filter results (for document routing)
            accessible_doc_ids: Optional set of document IDs for user access control filtering

        Returns:
            List of chunk dictionaries with page_content and metadata
        """
        # Check if vector search should be enabled
        # 1. If GRAPH_RAG_VECTOR_SEARCH_ENABLED=true, always enabled
        # 2. If disabled, check if any documents have pending GraphRAG indexing
        vector_search_enabled = GRAPH_RAG_VECTOR_SEARCH_ENABLED

        if not vector_search_enabled and accessible_doc_ids:
            # Check if any documents have pending GraphRAG indexing
            status = self._check_documents_graphrag_status(accessible_doc_ids)
            if status["should_enable_vector_search"]:
                vector_search_enabled = True
                logger.info(
                    f"[GraphRAG] Vector search ENABLED dynamically: "
                    f"{status['pending_count']} documents have pending GraphRAG indexing"
                )

        if not vector_search_enabled:
            logger.debug("[GraphRAG] Qdrant vector search is disabled (all documents have completed GraphRAG)")
            return []

        try:
            from ..vectorstore import get_vectorstore
            from qdrant_client import models

            # Get vectorstore instance
            vectorstore = get_vectorstore()

            # Build search kwargs with optional filtering
            search_kwargs = {"k": top_k}

            # Build filter conditions
            filter_conditions = []

            # Add source filter if specified (for document routing)
            if source_names and len(source_names) > 0:
                filter_conditions.append(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source_name),
                            )
                            for source_name in source_names
                        ]
                    )
                )
                logger.debug(f"[GraphRAG] Filtering vector search to sources: {source_names}")

            # Add document ID filter for access control
            if accessible_doc_ids is not None and len(accessible_doc_ids) > 0:
                doc_id_strings = [str(doc_id) for doc_id in accessible_doc_ids]
                filter_conditions.append(
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchAny(any=doc_id_strings),
                    )
                )
                logger.info(f"[GraphRAG] Access control: filtering to {len(doc_id_strings)} accessible document IDs")

            # Combine filters - always wrap in Filter object for Qdrant API compatibility
            if len(filter_conditions) > 0:
                # Check if we have any raw FieldCondition objects that need wrapping
                must_conditions = []
                for cond in filter_conditions:
                    if isinstance(cond, models.Filter):
                        # Nested Filter - add to must list
                        must_conditions.append(cond)
                    else:
                        # FieldCondition - wrap in Filter and add
                        must_conditions.append(models.Filter(must=[cond]))

                # Combine all conditions with AND logic
                if len(must_conditions) == 1:
                    search_kwargs["filter"] = must_conditions[0]
                else:
                    search_kwargs["filter"] = models.Filter(must=must_conditions)

            # Perform vector search using similarity_search with filter
            docs = vectorstore.similarity_search(query, **search_kwargs)
            chunks = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                }
                for doc in docs
            ]
            logger.debug(f"[GraphRAG] Qdrant vector search found {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.warning(f"[GraphRAG] Qdrant vector search failed: {e}")
            return []

    def _merge_chunks(self, *chunk_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge multiple chunk lists and deduplicate by chunk_id.

        Args:
            *chunk_lists: Variable number of chunk lists to merge

        Returns:
            Deduplicated list of chunks
        """
        seen_ids = set()
        merged = []

        for chunk_list in chunk_lists:
            for chunk in chunk_list:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    merged.append(chunk)
                elif not chunk_id:
                    # Include chunks without IDs (shouldn't happen, but be safe)
                    merged.append(chunk)

        logger.debug(f"[GraphRAG] Merged {sum(len(cl) for cl in chunk_lists)} chunks into {len(merged)} unique chunks")
        return merged

    async def query(
        self,
        query: str,
        mode: str = None,
        params: QueryParam = None,
    ) -> GraphRAGContext:
        """
        Process a query and retrieve graph-enhanced context.

        Following Graph-R1 paper design:
        - Supports LOCAL, GLOBAL, HYBRID modes
        - Always combines graph results with Qdrant vector search
        - Iterative "think-query-retrieve-rethink" reasoning (when params.max_steps > 1)

        Args:
            query: User query string
            mode: Optional mode override ("local", "global", "hybrid", "auto")
            params: Optional query parameters

        Returns:
            GraphRAGContext with entities, relationships, and chunks
        """
        if not GRAPH_RAG_QUERY_ENABLED:
            logger.debug("GraphRAG is disabled, returning empty context")
            return GraphRAGContext(
                entities=[],
                relationships=[],
                chunks=[],
                mode=QueryMode.HYBRID,
                enhanced_query=query,
            )

        logger.info("=" * 80)
        logger.info("[GraphRAG Query] ========== GRAPH RAG QUERY START ==========")
        logger.info("=" * 80)
        logger.info(f"[GraphRAG Query] Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"[GraphRAG Query] Mode requested: {mode or 'auto'}")
        logger.info("-" * 80)

        await self._init_storage()

        params = params or QueryParam()

        # Check if iterative reasoning is enabled (Graph-R1 paper design)
        if params.max_steps > 1:
            logger.info("[GraphRAG Query] STEP 1: Iterative reasoning mode detected")
            logger.info(f"[GraphRAG Query]   - Max steps: {params.max_steps}")
            return await self._iterative_reasoning_query(query, mode, params)

        # Single-step retrieval (original behavior)
        logger.info("[GraphRAG Query] STEP 1: Detecting query mode...")
        # Determine query mode
        if mode and mode != "auto":
            query_mode = QueryMode(mode.lower())
            enhanced_query = query
            logger.info(f"[GraphRAG Query]   - Mode: {query_mode.value} (explicitly set)")
        else:
            query_mode, enhanced_query = await self.query_mode_detector.detect_mode(query)
            logger.info(f"[GraphRAG Query]   - Mode: {query_mode.value} (auto-detected)")
            if enhanced_query != query:
                logger.info(f"[GraphRAG Query]   - Enhanced query: {enhanced_query[:80]}...")

        logger.info("-" * 80)
        logger.info(f"[GraphRAG Query] STEP 2: Executing {query_mode.value.upper()} retrieval...")

        # Retrieve based on mode (all modes now include vector search)
        if query_mode == QueryMode.LOCAL:
            context = await self._local_retrieval(enhanced_query, params)
        elif query_mode == QueryMode.GLOBAL:
            context = await self._global_retrieval(enhanced_query, params)
        else:  # HYBRID (default)
            context = await self._hybrid_retrieval(enhanced_query, params)

        context.mode = query_mode
        context.enhanced_query = enhanced_query

        logger.info("-" * 80)
        logger.info("[GraphRAG Query] ========== GRAPH RAG QUERY COMPLETE ==========")
        logger.info("[GraphRAG Query] FINAL SUMMARY:")
        logger.info(f"[GraphRAG Query]   - Query Mode: {query_mode.value}")
        logger.info(f"[GraphRAG Query]   - Entities Retrieved: {len(context.entities)}")
        logger.info(f"[GraphRAG Query]   - Relationships Retrieved: {len(context.relationships)}")
        logger.info(f"[GraphRAG Query]   - Chunks Retrieved: {len(context.chunks)}")
        logger.info("=" * 80)

        return context

    async def _iterative_reasoning_query(
        self,
        question: str,
        mode: str = None,
        params: QueryParam = None,
    ) -> GraphRAGContext:
        """
        Iterative "think-query-retrieve-rethink" reasoning cycle (Graph-R1 paper).

        This implements the core Graph-R1 reasoning loop:
        1. THINK: Analyze current knowledge and decide next action
        2. QUERY: Generate a retrieval query if more info needed
        3. RETRIEVE: Use LOCAL/GLOBAL/HYBRID to get graph + vector results
        4. RETHINK: Evaluate if sufficient knowledge gathered
        5. REPEAT: Continue up to max_steps or until answer found

        Args:
            question: User's original question
            mode: Optional mode override for retrieval steps
            params: Query parameters (max_steps controls iterations)

        Returns:
            GraphRAGContext with accumulated entities, relationships, and chunks
        """
        from .agent_prompts import (
            AGENT_THINK_PROMPT,
            INITIAL_QUERY_PROMPT,
            KNOWLEDGE_FORMAT_TEMPLATE,
            NO_KNOWLEDGE_TEMPLATE,
        )
        from .reward_scorer import RewardScorer
        import re

        # Initialize LLM service first (needed by reward scorer)
        if not self._llm_service:
            from ..llm_service import get_llm_service
            self._llm_service = get_llm_service()

        # Initialize reward scorer for quality-based termination (AFTER LLM service)
        enable_reward_scoring = os.getenv("GRAPH_RAG_ENABLE_REWARD_SCORING", "true").lower() == "true"
        reward_scorer = RewardScorer(llm_service=self._llm_service) if enable_reward_scoring else None

        # Reward thresholds from environment
        format_reward_high_threshold = float(os.getenv("GRAPH_RAG_FORMAT_REWARD_HIGH_THRESHOLD", "0.8"))
        format_reward_low_threshold = float(os.getenv("GRAPH_RAG_FORMAT_REWARD_LOW_THRESHOLD", "0.3"))
        answer_reward_threshold = float(os.getenv("GRAPH_RAG_ANSWER_REWARD_THRESHOLD", "0.6"))

        logger.info("=" * 70)
        logger.info("[GraphRAG] 🚀 STARTING ITERATIVE REASONING (Graph-R1)")
        logger.info("=" * 70)
        logger.info(f"[GraphRAG] Question: {question}")
        logger.info(f"[GraphRAG] Max steps: {params.max_steps}")
        logger.info(f"[GraphRAG] Reward-based termination: {enable_reward_scoring}")
        logger.info("-" * 70)

        # State tracking
        step = 0
        max_steps = params.max_steps
        retrieved_knowledge = []
        queries_made = []
        is_complete = False
        final_answer = ""  # Store the final answer from LLM
        reward_history = []  # Track reward scores for each step

        # Accumulated results
        all_entities = []
        all_relationships = []
        all_chunks = []
        seen_entity_ids = set()
        modes_used = []

        # Step 1: Generate initial query
        logger.info("[GraphRAG] 📝 PHASE 1: Generating initial search query...")
        initial_query_prompt = INITIAL_QUERY_PROMPT.format(question=question)

        # Get LLM chat model and invoke it
        from langchain_core.messages import HumanMessage
        llm = self._llm_service.get_query_model(temperature=0.1, num_predict=200)
        initial_response_msg = await llm.ainvoke([HumanMessage(content=initial_query_prompt)])
        initial_response = initial_response_msg.content.strip()

        # Extract query from <query>...</query> tags
        query_match = re.search(r'<query>(.*?)</query>', initial_response, re.DOTALL)
        current_query = query_match.group(1).strip() if query_match else question

        logger.info(f"[GraphRAG] Initial query: '{current_query}'")
        logger.info("-" * 70)

        # Iterative reasoning loop
        while step < max_steps and not is_complete:
            step += 1

            logger.info("=" * 50)
            logger.info(f"[GraphRAG] 🔄 ITERATION {step}/{max_steps}")
            logger.info("=" * 50)
            logger.info(f"[GraphRAG] Current query: '{current_query}'")

            # Step 2: Retrieve knowledge using LOCAL/GLOBAL/HYBRID
            logger.info(f"[GraphRAG] 🔍 RETRIEVING knowledge...")

            # Determine retrieval mode for this step
            if mode and mode != "auto":
                retrieval_mode = QueryMode(mode.lower())
                enhanced_query = current_query
            else:
                retrieval_mode, enhanced_query = await self.query_mode_detector.detect_mode(current_query)

            logger.info(f"[GraphRAG] Using {retrieval_mode.value.upper()} mode for retrieval")

            # Perform retrieval
            if retrieval_mode == QueryMode.LOCAL:
                context = await self._local_retrieval(enhanced_query, params)
            elif retrieval_mode == QueryMode.GLOBAL:
                context = await self._global_retrieval(enhanced_query, params)
            else:  # HYBRID
                context = await self._hybrid_retrieval(enhanced_query, params)

            # Track this retrieval step
            knowledge_step = {
                "query": current_query,
                "mode": retrieval_mode.value,
                "entities": context.entities,
                "relationships": context.relationships,
                "chunks": context.chunks,
            }
            retrieved_knowledge.append(knowledge_step)
            queries_made.append(current_query)

            if retrieval_mode.value not in modes_used:
                modes_used.append(retrieval_mode.value)

            logger.info(f"[GraphRAG] Retrieved:")
            logger.info(f"  - Entities: {len(context.entities)}")
            logger.info(f"  - Relationships: {len(context.relationships)}")
            logger.info(f"  - Chunks: {len(context.chunks)}")

            # Accumulate results (deduplicate entities)
            for entity in context.entities:
                entity_id = entity.get("id")
                if entity_id and entity_id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity_id)
                elif not entity_id:
                    all_entities.append(entity)

            all_relationships.extend(context.relationships)
            all_chunks.extend(context.chunks)

            # Step 3: Think - decide to continue or answer
            logger.info("-" * 50)
            logger.info(f"[GraphRAG] 🤔 THINKING: Should I continue or answer?")

            # Format knowledge summary for LLM
            knowledge_summary = self._format_knowledge_summary(retrieved_knowledge)

            # Format previous queries for anti-duplication
            previous_queries_str = ", ".join(f'"{q}"' for q in queries_made) if queries_made else "None"

            think_prompt = AGENT_THINK_PROMPT.format(
                question=question,
                step=step,
                max_steps=max_steps,
                retrieved_knowledge=knowledge_summary,
                previous_queries=previous_queries_str,
            )

            # Get LLM chat model and invoke it
            # Increased num_predict for enhanced deep reasoning prompt
            # 4000 tokens allows for comprehensive answers with multiple products/items
            llm = self._llm_service.get_chat_model(temperature=0.2, num_predict=4000)
            think_response_msg = await llm.ainvoke([HumanMessage(content=think_prompt)])
            think_response = think_response_msg.content.strip()

            logger.info(f"[GraphRAG] LLM response preview: {think_response[:200]}...")
            logger.info(f"[GraphRAG] LLM response length: {len(think_response)} chars")

            # Compute reward scores for quality assessment (Graph-R1 alignment)
            if enable_reward_scoring and reward_scorer:
                format_score, format_details = reward_scorer.compute_format_reward(think_response)

                logger.info("=" * 50)
                logger.info("[GraphRAG] 📊 REWARD SCORING")
                logger.info("=" * 50)
                logger.info(f"[GraphRAG] Format Reward: {format_score:.2f}")
                logger.info(f"[GraphRAG] Format Details: {format_details['breakdown']}")

                # Store reward data
                reward_data = {
                    'step': step,
                    'format_score': format_score,
                    'format_details': format_details,
                }
                reward_history.append(reward_data)

            # Debug: Check if tags exist in response
            has_answer_tag = '<answer>' in think_response
            has_query_tag = '<query>' in think_response
            logger.info(f"[GraphRAG] Tags present: <answer>={has_answer_tag}, <query>={has_query_tag}")

            # Parse response for <answer> or <query> tags
            answer_match = re.search(r'<answer>(.*?)</answer>', think_response, re.DOTALL)
            next_query_match = re.search(r'<query>(.*?)</query>', think_response, re.DOTALL)

            logger.info(f"[GraphRAG] Regex matches: answer_match={answer_match is not None}, next_query_match={next_query_match is not None}")

            # Extract answer if present
            answer_text = None
            if not answer_match and has_answer_tag:
                # Handle case where LLM starts <answer> but doesn't close it properly
                answer_start = think_response.find('<answer>')
                if answer_start != -1:
                    answer_text = think_response[answer_start + 8:].strip()  # 8 = len('<answer>')
            elif answer_match:
                answer_text = answer_match.group(1).strip()

            # Reward-based termination logic (Graph-R1 alignment)
            if enable_reward_scoring and reward_scorer and reward_history:
                format_score = reward_history[-1]['format_score']

                # Decision 1: High format score + answer found = Check answer quality
                if format_score >= format_reward_high_threshold and answer_text:
                    answer_score, answer_details = await reward_scorer.compute_answer_reward(
                        answer_text, question
                    )
                    logger.info(f"[GraphRAG] Answer Quality Score: {answer_score:.2f}")
                    reward_history[-1]['answer_score'] = answer_score
                    reward_history[-1]['answer_details'] = answer_details

                    if answer_score >= answer_reward_threshold:
                        final_answer = answer_text
                        is_complete = True
                        logger.info(f"[GraphRAG] ✅ DECISION: TERMINATE")
                        logger.info(f"[GraphRAG] Reason: High format score ({format_score:.2f}) + good answer quality ({answer_score:.2f})")
                        logger.info(f"[GraphRAG] Final answer preview: {final_answer[:200]}...")
                    else:
                        logger.info(f"[GraphRAG] 🔄 DECISION: CONTINUE")
                        logger.info(f"[GraphRAG] Reason: Answer quality too low ({answer_score:.2f} < {answer_reward_threshold})")
                        # Force continuation despite <answer> tag
                        answer_text = None

                # Decision 2: Low format score = Likely hallucination, terminate and regenerate
                elif format_score < format_reward_low_threshold:
                    is_complete = True
                    logger.info(f"[GraphRAG] ⚠️ DECISION: TERMINATE")
                    logger.info(f"[GraphRAG] Reason: Low format score ({format_score:.2f}) - likely hallucination")
                    logger.info(f"[GraphRAG] Will generate final answer from accumulated knowledge...")
                    # Don't use the LLM's answer, will generate from knowledge
                    answer_text = None

                # Decision 3: Medium format score + answer = Check quality
                elif answer_text and format_score >= 0.5:
                    answer_score, answer_details = await reward_scorer.compute_answer_reward(
                        answer_text, question
                    )
                    logger.info(f"[GraphRAG] Answer Quality Score: {answer_score:.2f}")
                    reward_history[-1]['answer_score'] = answer_score
                    reward_history[-1]['answer_details'] = answer_details

                    if answer_score >= answer_reward_threshold:
                        final_answer = answer_text
                        is_complete = True
                        logger.info(f"[GraphRAG] ✅ DECISION: TERMINATE")
                        logger.info(f"[GraphRAG] Reason: Acceptable format ({format_score:.2f}) + good answer ({answer_score:.2f})")
                    else:
                        logger.info(f"[GraphRAG] 🔄 DECISION: CONTINUE")
                        logger.info(f"[GraphRAG] Reason: Answer quality too low ({answer_score:.2f})")
                        # Force continuation
                        answer_text = None

            # Fallback to original logic if reward scoring disabled or no answer yet
            if not enable_reward_scoring or not reward_scorer:
                if answer_text:
                    final_answer = answer_text
                    is_complete = True
                    logger.info(f"[GraphRAG] ✅ DECISION: TERMINATE - Answer found")
                    logger.info(f"[GraphRAG] Termination reason: LLM provided <answer> tag")
                    logger.info(f"[GraphRAG] Final answer preview: {final_answer[:200]}...")
            elif not next_query_match and has_query_tag:
                # Handle case where LLM starts <query> but doesn't close it properly
                query_start = think_response.find('<query>')
                if query_start != -1:
                    next_query = think_response[query_start + 7:].strip()  # 7 = len('<query>')
                    if next_query and next_query not in queries_made:
                        current_query = next_query
                        logger.info(f"[GraphRAG] 🔄 DECISION: CONTINUE - New query generated (unclosed tag)")
                        logger.info(f"[GraphRAG] Next query: '{next_query}'")
                    else:
                        # Duplicate or empty query, terminate
                        is_complete = True
                        logger.info(f"[GraphRAG] ⚠️ DECISION: TERMINATE - Duplicate or empty query")
                        logger.info(f"[GraphRAG] Termination reason: Query already used or empty")
            elif next_query_match:
                # LLM wants to continue with a new query
                next_query = next_query_match.group(1).strip()
                if next_query not in queries_made:
                    current_query = next_query
                    logger.info(f"[GraphRAG] 🔄 DECISION: CONTINUE - New query generated")
                    logger.info(f"[GraphRAG] Next query: '{next_query}'")
                else:
                    # Duplicate query, terminate
                    is_complete = True
                    logger.info(f"[GraphRAG] ⚠️ DECISION: TERMINATE - Duplicate query")
                    logger.info(f"[GraphRAG] Termination reason: Query already used")
            else:
                # No valid tags, terminate
                is_complete = True
                logger.info(f"[GraphRAG] ⚠️ DECISION: TERMINATE - No valid tags found")
                logger.info(f"[GraphRAG] Termination reason: Missing <answer> or <query> tags")

        # Check if max steps reached
        if step >= max_steps and not is_complete:
            logger.info(f"[GraphRAG] ⚠️ MAX STEPS REACHED ({max_steps})")

        # Log reward summary if available
        if enable_reward_scoring and reward_history:
            logger.info("=" * 70)
            logger.info("[GraphRAG] 📊 REWARD SUMMARY")
            logger.info("=" * 70)
            for i, reward_data in enumerate(reward_history, 1):
                logger.info(f"Step {i}:")
                logger.info(f"  Format Score: {reward_data.get('format_score', 'N/A'):.2f}")
                if 'answer_score' in reward_data:
                    logger.info(f"  Answer Score: {reward_data['answer_score']:.2f}")
            logger.info("=" * 70)

        # Final summary
        logger.info("=" * 70)
        logger.info("[GraphRAG] 🏁 ITERATIVE REASONING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"[GraphRAG] Summary:")
        logger.info(f"  - Total iterations: {step}")
        logger.info(f"  - Queries made: {queries_made}")
        logger.info(f"  - Retrieval modes used: {modes_used}")
        logger.info(f"  - Total unique entities: {len(all_entities)}")
        logger.info(f"  - Total relationships: {len(all_relationships)}")
        logger.info(f"  - Total chunks: {len(all_chunks)}")
        if final_answer:
            logger.info(f"  - Final answer generated: Yes ({len(final_answer)} chars)")
        logger.info("=" * 70)

        # Return accumulated context
        return GraphRAGContext(
            entities=all_entities,
            relationships=all_relationships,
            chunks=all_chunks,
            mode=QueryMode.HYBRID,  # Iterative mode uses all modes
            enhanced_query=f"Iterative reasoning: {queries_made}",
            final_answer=final_answer,
        )

    def _format_knowledge_summary(self, retrieved_knowledge: List[Dict]) -> str:
        """Format retrieved knowledge for LLM prompt with enhanced template."""
        from .agent_prompts import KNOWLEDGE_FORMAT_TEMPLATE, NO_KNOWLEDGE_TEMPLATE

        if not retrieved_knowledge:
            return "No knowledge retrieved yet."

        summaries = []
        for i, knowledge in enumerate(retrieved_knowledge, 1):
            entities = knowledge.get("entities", [])
            relationships = knowledge.get("relationships", [])
            chunks = knowledge.get("chunks", [])
            query = knowledge.get("query", "")

            if entities or relationships or chunks:
                entities_str = self._format_entities_for_prompt(entities)
                relationships_str = self._format_relationships_for_prompt(relationships)
                chunks_str = self._format_chunks_for_prompt(chunks)

                # Extract unique sources from chunks
                sources = set()
                for chunk in chunks:
                    source = chunk.get("metadata", {}).get("source", "")
                    if source:
                        sources.add(source)
                sources_str = ", ".join(sources) if sources else "None"

                summaries.append(
                    KNOWLEDGE_FORMAT_TEMPLATE.format(
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
                summaries.append(NO_KNOWLEDGE_TEMPLATE.format(step=i, query=query))

        return "\n".join(summaries)

    def _format_entities_for_prompt(self, entities: List[Dict]) -> str:
        """Format entity list for LLM prompt."""
        if not entities:
            return "None"

        lines = []
        for e in entities[:10]:  # Limit to top 10
            name = e.get("name", e.get("entity_name", "Unknown"))
            entity_type = e.get("entity_type", "Unknown")
            desc = e.get("description", "")[:200]
            score = e.get("_score", 0)
            lines.append(f"- **{name}** ({entity_type}): {desc} [score: {score:.2f}]")

        if len(entities) > 10:
            lines.append(f"... and {len(entities) - 10} more entities")

        return "\n".join(lines) if lines else "None"

    def _format_relationships_for_prompt(self, relationships: List[Dict]) -> str:
        """Format relationship list for LLM prompt."""
        if not relationships:
            return "None"

        lines = []
        for r in relationships[:10]:  # Limit to top 10
            src = r.get("src_name", r.get("src_entity_id", "?"))
            tgt = r.get("tgt_name", r.get("tgt_entity_id", "?"))
            desc = r.get("description", "")[:150]
            keywords = r.get("keywords", "")
            score = r.get("_score", 0)
            lines.append(f"- **{src}** → **{tgt}**: {desc} [{keywords}] [score: {score:.2f}]")

        if len(relationships) > 10:
            lines.append(f"... and {len(relationships) - 10} more relationships")

        return "\n".join(lines) if lines else "None"

    def _format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """Format document chunks for LLM prompt, grouped by source document.

        Shows top N chunks (by relevance score) from EACH source document.
        N is configured by GRAPH_RAG_CHUNKS_PER_SOURCE in .env (default: 3).
        ALL sources are included to ensure complete coverage.
        """
        if not chunks:
            return "None"

        # Get chunks per source from environment (default: 3)
        chunks_per_source = int(os.getenv("GRAPH_RAG_CHUNKS_PER_SOURCE", "3"))

        # Group chunks by source document
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "Unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

        logger.info(f"[GraphRAG] Formatting chunks from {len(chunks_by_source)} sources (showing top {chunks_per_source} chunks per source)")
        logger.info(f"[GraphRAG] Sources: {list(chunks_by_source.keys())}")

        lines = []
        total_chunks_shown = 0

        # Format chunks grouped by source - INCLUDE ALL SOURCES (no max_sources limit)
        for source_idx, (source, source_chunks) in enumerate(chunks_by_source.items(), 1):
            # Sort chunks by relevance score (descending) within each source
            sorted_chunks = sorted(
                source_chunks,
                key=lambda c: c.get("_score", 0),
                reverse=True
            )

            lines.append(f"\n**From Document: {source}** ({len(source_chunks)} chunks available)")

            # Show top N chunks by score
            for chunk_idx, chunk in enumerate(sorted_chunks[:chunks_per_source], 1):
                content = chunk.get("page_content", chunk.get("content", ""))
                # Truncate very long chunks but keep enough for invoice/product data
                # Increased from 500 to 800 to capture complete product information
                if len(content) > 800:
                    content = content[:800] + "..."
                score = chunk.get("_score", 0)
                lines.append(f"  Chunk {chunk_idx} (score: {score:.2f}):\n  {content}\n")
                total_chunks_shown += 1

            if len(sorted_chunks) > chunks_per_source:
                lines.append(f"  ... and {len(sorted_chunks) - chunks_per_source} more chunks from this source\n")

        logger.info(f"[GraphRAG] Formatted {total_chunks_shown} chunks from {len(chunks_by_source)} sources for LLM prompt")

        return "\n".join(lines) if lines else "None"

    async def _local_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        LOCAL mode: Entity-focused retrieval from Neo4j + Qdrant vector search.

        Following Graph-R1 paper design:
        1. Find semantically similar entities via Neo4j vector search
        2. Fall back to text matching if vector search unavailable
        3. Retrieve source chunks for matched entities from Qdrant
        4. ALWAYS include direct Qdrant vector search results for the query
        """
        logger.info("-" * 60)
        logger.info("[GraphRAG Query] LOCAL MODE RETRIEVAL")
        logger.info("-" * 60)

        # Ensure storage is initialized (may be called directly by UnifiedRetriever)
        await self._init_storage()

        entities = []

        # Convert accessible_doc_ids to list for Neo4j methods
        accessible_doc_ids_list = None
        if params.accessible_doc_ids is not None:
            accessible_doc_ids_list = [str(doc_id) for doc_id in params.accessible_doc_ids]
            logger.info(f"[GraphRAG Query]   - Access control: {len(accessible_doc_ids_list)} accessible docs")

        # Try vector search first
        logger.info("[GraphRAG Query] Step 2.1: Entity vector search in Neo4j...")
        if self._vector_search_enabled:
            query_embedding = await self._get_query_embedding(query)
            if query_embedding:
                logger.info("[GraphRAG Query]   - Using vector similarity search")
                try:
                    entities = await self._graph_storage.vector_search_entities(
                        query_embedding=query_embedding,
                        limit=params.top_k,
                        min_score=0.5,
                        source_names=params.source_names,
                        accessible_doc_ids=accessible_doc_ids_list,
                    )
                    logger.info(f"[GraphRAG Query]   - Vector search found: {len(entities)} entities")
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Vector search failed: {e}")
                    entities = []

        # Fall back to text matching if vector search didn't find results
        if not entities:
            entity_names = extract_entity_names_from_query(query)
            logger.info(f"[GraphRAG Query] Step 2.2: Fallback text search for: {entity_names[:5]}...")

            for name in entity_names[:10]:  # Limit to top 10 names
                try:
                    found = await self._graph_storage.get_nodes_by_name(
                        name,
                        limit=5,
                        accessible_doc_ids=accessible_doc_ids_list
                    )
                    # Filter by source_names if document routing is enabled (fallback if no access control)
                    if params.source_names and len(params.source_names) > 0 and not accessible_doc_ids_list:
                        found = [e for e in found if e.get("source_doc") in params.source_names]
                    for entity in found:
                        if entity not in entities:
                            entities.append(entity)
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Error finding entity '{name}': {e}")

            logger.info(f"[GraphRAG Query]   - Text search found: {len(entities)} entities")

        # Limit entities to top_k
        entities = entities[:params.top_k]

        # Retrieve source chunks for the found entities (with access control)
        logger.info("[GraphRAG Query] Step 2.3: Retrieving source chunks from Qdrant...")
        entity_chunks = self._get_chunks_for_entities(
            entities,
            accessible_doc_ids=params.accessible_doc_ids
        )
        logger.info(f"[GraphRAG Query]   - Entity chunks retrieved: {len(entity_chunks)}")

        # ALWAYS include direct Qdrant vector search results (with optional source and access control filtering)
        logger.info("[GraphRAG Query] Step 2.4: Direct vector search in Qdrant...")
        vector_chunks = await self._get_vector_search_chunks(
            query,
            params.top_k,
            source_names=params.source_names,
            accessible_doc_ids=params.accessible_doc_ids
        )
        logger.info(f"[GraphRAG Query]   - Vector chunks retrieved: {len(vector_chunks)}")

        # Combine and deduplicate chunks
        chunks = self._merge_chunks(entity_chunks, vector_chunks)
        logger.info(f"[GraphRAG Query] LOCAL MODE RESULT: {len(entities)} entities, {len(chunks)} chunks")

        return GraphRAGContext(
            entities=entities,
            relationships=[],
            chunks=chunks,
            mode=QueryMode.LOCAL,
            enhanced_query=query,
        )

    async def _global_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        GLOBAL mode: Relationship-focused retrieval from Neo4j.

        Uses vector similarity search for relationships when available.

        1. Generate query embedding
        2. Find semantically similar relationships via vector search
        3. Get connected entities from matched relationships
        4. Fall back to text-based entity search if needed
        5. Retrieve source chunks for context
        """
        logger.info("-" * 60)
        logger.info("[GraphRAG Query] GLOBAL MODE RETRIEVAL")
        logger.info("-" * 60)

        # Ensure storage is initialized (may be called directly by UnifiedRetriever)
        await self._init_storage()

        entities = []
        relationships = []

        # Convert accessible_doc_ids to list for Neo4j methods
        accessible_doc_ids_list = None
        if params.accessible_doc_ids is not None:
            accessible_doc_ids_list = [str(doc_id) for doc_id in params.accessible_doc_ids]
            logger.info(f"[GraphRAG Query]   - Access control: {len(accessible_doc_ids_list)} accessible docs")

        # Try vector search for relationships first
        logger.info("[GraphRAG Query] Step 2.1: Relationship vector search in Neo4j...")
        if self._vector_search_enabled:
            query_embedding = await self._get_query_embedding(query)
            if query_embedding:
                logger.info("[GraphRAG Query]   - Using vector similarity search")
                try:
                    rel_results = await self._graph_storage.vector_search_relationships(
                        query_embedding=query_embedding,
                        limit=params.top_k,
                        min_score=0.5,
                        source_names=params.source_names,
                        accessible_doc_ids=accessible_doc_ids_list,
                    )
                    for rel in rel_results:
                        relationships.append({
                            "src_name": rel.get("src_name", ""),
                            "tgt_name": rel.get("tgt_name", ""),
                            "description": rel.get("description", ""),
                            "keywords": rel.get("keywords", ""),
                            "_score": rel.get("_score", 0),
                        })
                    logger.info(f"[GraphRAG Query]   - Vector search found: {len(relationships)} relationships")
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Relationship vector search failed: {e}")

        # Fall back to entity-based relationship search
        if not relationships:
            entity_names = extract_entity_names_from_query(query)
            logger.info(f"[GraphRAG Query] Step 2.2: Fallback entity-based search for: {entity_names[:5]}...")

            for name in entity_names[:5]:  # Limit traversal
                try:
                    found = await self._graph_storage.get_nodes_by_name(
                        name,
                        limit=3,
                        accessible_doc_ids=accessible_doc_ids_list
                    )
                    # Filter by source_names if document routing is enabled (fallback if no access control)
                    if params.source_names and len(params.source_names) > 0 and not accessible_doc_ids_list:
                        found = [e for e in found if e.get("source_doc") in params.source_names]
                    for entity in found:
                        if entity not in entities:
                            entities.append(entity)
                        # Get edges for this entity
                        entity_id = entity.get("id")
                        if entity_id:
                            edges = await self._graph_storage.get_node_edges(entity_id)
                            # Filter edges by document_id if access control is enabled
                            if accessible_doc_ids_list:
                                edges = [
                                    (src_id, tgt_id, edge_data)
                                    for src_id, tgt_id, edge_data in edges
                                    if edge_data.get("document_id") in accessible_doc_ids_list
                                ]
                            # Otherwise filter by source_doc
                            elif params.source_names and len(params.source_names) > 0:
                                edges = [
                                    (src_id, tgt_id, edge_data)
                                    for src_id, tgt_id, edge_data in edges
                                    if edge_data.get("source_doc") in params.source_names
                                ]
                            for src_id, tgt_id, edge_data in edges:
                                rel = {
                                    "src_entity_id": src_id,
                                    "tgt_entity_id": tgt_id,
                                    "description": edge_data.get("description", ""),
                                    "keywords": edge_data.get("keywords", ""),
                                    "source_doc": edge_data.get("source_doc"),
                                }
                                if rel not in relationships:
                                    relationships.append(rel)
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Error finding entity '{name}': {e}")

            logger.info(f"[GraphRAG Query]   - Text search found: {len(entities)} entities, {len(relationships)} relationships")

        # Limit to top_k
        entities = entities[:params.top_k]
        relationships = relationships[:params.top_k]

        # Retrieve source chunks for the found entities (with access control)
        logger.info("[GraphRAG Query] Step 2.3: Retrieving source chunks from Qdrant...")
        entity_chunks = self._get_chunks_for_entities(
            entities,
            accessible_doc_ids=params.accessible_doc_ids
        )
        logger.info(f"[GraphRAG Query]   - Entity chunks retrieved: {len(entity_chunks)}")

        # ALWAYS include direct Qdrant vector search results (with optional source and access control filtering)
        logger.info("[GraphRAG Query] Step 2.4: Direct vector search in Qdrant...")
        vector_chunks = await self._get_vector_search_chunks(
            query,
            params.top_k,
            source_names=params.source_names,
            accessible_doc_ids=params.accessible_doc_ids
        )
        logger.info(f"[GraphRAG Query]   - Vector chunks retrieved: {len(vector_chunks)}")

        # Combine and deduplicate chunks
        chunks = self._merge_chunks(entity_chunks, vector_chunks)
        logger.info(f"[GraphRAG Query] GLOBAL MODE RESULT: {len(entities)} entities, {len(relationships)} relationships, {len(chunks)} chunks")

        return GraphRAGContext(
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            mode=QueryMode.GLOBAL,
            enhanced_query=query,
        )

    async def _hybrid_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        HYBRID mode: Combined entity and relationship retrieval from Neo4j.

        Uses vector similarity for both entities and relationships, then expands via graph.

        1. Vector search for entities and relationships
        2. Expand via graph traversal from matched entities
        3. Retrieve source chunks for context
        """
        logger.info("-" * 60)
        logger.info("[GraphRAG Query] HYBRID MODE RETRIEVAL")
        logger.info("-" * 60)

        # Ensure storage is initialized (may be called directly by UnifiedRetriever)
        await self._init_storage()

        entities = []
        relationships = []
        seen_entity_ids = set()

        # Convert accessible_doc_ids to list for Neo4j methods
        accessible_doc_ids_list = None
        if params.accessible_doc_ids is not None:
            accessible_doc_ids_list = [str(doc_id) for doc_id in params.accessible_doc_ids]
            logger.info(f"[GraphRAG Query]   - Access control: {len(accessible_doc_ids_list)} accessible docs")

        # Try vector search for both entities and relationships
        logger.info("[GraphRAG Query] Step 2.1: Entity + Relationship vector search in Neo4j...")
        if self._vector_search_enabled:
            query_embedding = await self._get_query_embedding(query)
            if query_embedding:
                logger.info("[GraphRAG Query]   - Using vector similarity search")

                # Vector search for entities
                try:
                    entity_results = await self._graph_storage.vector_search_entities(
                        query_embedding=query_embedding,
                        limit=params.top_k // 2,
                        min_score=0.5,
                        source_names=params.source_names,
                        accessible_doc_ids=accessible_doc_ids_list,
                    )
                    for entity in entity_results:
                        entity_id = entity.get("id")
                        if entity_id and entity_id not in seen_entity_ids:
                            entities.append(entity)
                            seen_entity_ids.add(entity_id)
                    logger.info(f"[GraphRAG Query]   - Entity vector search found: {len(entities)} entities")
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Entity vector search failed: {e}")

                # Vector search for relationships
                try:
                    rel_results = await self._graph_storage.vector_search_relationships(
                        query_embedding=query_embedding,
                        limit=params.top_k // 2,
                        min_score=0.5,
                        source_names=params.source_names,
                        accessible_doc_ids=accessible_doc_ids_list,
                    )
                    for rel in rel_results:
                        relationships.append({
                            "src_name": rel.get("src_name", ""),
                            "tgt_name": rel.get("tgt_name", ""),
                            "description": rel.get("description", ""),
                            "keywords": rel.get("keywords", ""),
                            "_score": rel.get("_score", 0),
                            "source_doc": rel.get("source_doc"),
                        })
                    logger.info(f"[GraphRAG Query]   - Relationship vector search found: {len(relationships)} relationships")
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Relationship vector search failed: {e}")

        # Expand via graph traversal from matched entities
        logger.info("[GraphRAG Query] Step 2.2: Expanding via graph traversal...")
        expansion_count = 0
        for entity in list(entities):  # Copy list to avoid modification during iteration
            entity_id = entity.get("id")
            if entity_id:
                try:
                    edges = await self._graph_storage.get_node_edges(entity_id)
                    # Filter edges by document_id if access control is enabled
                    if accessible_doc_ids_list:
                        edges = [
                            (src_id, tgt_id, edge_data)
                            for src_id, tgt_id, edge_data in edges
                            if edge_data.get("document_id") in accessible_doc_ids_list
                        ]
                    # Otherwise filter by source_doc if document routing is enabled
                    elif params.source_names and len(params.source_names) > 0:
                        edges = [
                            (src_id, tgt_id, edge_data)
                            for src_id, tgt_id, edge_data in edges
                            if edge_data.get("source_doc") in params.source_names
                        ]
                    for src_id, tgt_id, edge_data in edges:
                        # Add relationship if not already present
                        rel = {
                            "src_entity_id": src_id,
                            "tgt_entity_id": tgt_id,
                            "description": edge_data.get("description", ""),
                            "keywords": edge_data.get("keywords", ""),
                            "source_doc": edge_data.get("source_doc"),
                        }
                        if rel not in relationships:
                            relationships.append(rel)

                        # Add neighbor entity (only if from accessible documents)
                        neighbor_id = tgt_id if src_id == entity_id else src_id
                        if neighbor_id not in seen_entity_ids:
                            neighbor = await self._graph_storage.get_node(neighbor_id)
                            if neighbor:
                                # Filter neighbor by document_id if access control is enabled
                                if accessible_doc_ids_list:
                                    if neighbor.get("document_id") not in accessible_doc_ids_list:
                                        continue
                                # Otherwise filter by source_doc
                                elif params.source_names and len(params.source_names) > 0:
                                    if neighbor.get("source_doc") not in params.source_names:
                                        continue
                                entities.append(neighbor)
                                seen_entity_ids.add(neighbor_id)
                                expansion_count += 1
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Error expanding entity {entity_id}: {e}")

        logger.info(f"[GraphRAG Query]   - Graph expansion added: {expansion_count} neighbor entities")

        # Fall back to text search if no results from vector search
        if not entities and not relationships:
            entity_names = extract_entity_names_from_query(query)
            logger.info(f"[GraphRAG Query] Step 2.3: Fallback text search for: {entity_names[:5]}...")

            for name in entity_names[:5]:
                try:
                    found = await self._graph_storage.get_nodes_by_name(
                        name,
                        limit=3,
                        accessible_doc_ids=accessible_doc_ids_list
                    )
                    # Filter by source_names if document routing is enabled (fallback if no access control)
                    if params.source_names and len(params.source_names) > 0 and not accessible_doc_ids_list:
                        found = [e for e in found if e.get("source_doc") in params.source_names]
                    for entity in found:
                        entity_id = entity.get("id")
                        if entity_id and entity_id not in seen_entity_ids:
                            entities.append(entity)
                            seen_entity_ids.add(entity_id)
                except Exception as e:
                    logger.warning(f"[GraphRAG Query]   - Error in text search for '{name}': {e}")

            logger.info(f"[GraphRAG Query]   - Text search found: {len(entities)} entities")

        # Filter entities and relationships by document_id if access control is enabled
        if accessible_doc_ids_list:
            before_filter_e = len(entities)
            before_filter_r = len(relationships)
            entities = [
                e for e in entities
                if e.get("document_id") in accessible_doc_ids_list
            ]
            relationships = [
                r for r in relationships
                if r.get("document_id") in accessible_doc_ids_list
            ]
            if len(entities) < before_filter_e or len(relationships) < before_filter_r:
                logger.info(
                    f"[GraphRAG] Filtered by document_id: entities {before_filter_e} → {len(entities)}, "
                    f"relationships {before_filter_r} → {len(relationships)} "
                    f"(access control: {len(accessible_doc_ids_list)} docs)"
                )
        # Otherwise filter by source_names if document routing is enabled (fallback)
        elif params.source_names and len(params.source_names) > 0:
            before_filter_e = len(entities)
            before_filter_r = len(relationships)
            entities = [
                e for e in entities
                if e.get("source_doc") in params.source_names
            ]
            relationships = [
                r for r in relationships
                if r.get("source_doc") in params.source_names
            ]
            if len(entities) < before_filter_e or len(relationships) < before_filter_r:
                logger.info(
                    f"[GraphRAG] Filtered by source: entities {before_filter_e} → {len(entities)}, "
                    f"relationships {before_filter_r} → {len(relationships)} "
                    f"(sources: {params.source_names})"
                )

        # Limit to top_k
        entities = entities[:params.top_k]
        relationships = relationships[:params.top_k]

        # Retrieve source chunks for the found entities (with access control)
        logger.info("[GraphRAG Query] Step 2.4: Retrieving source chunks from Qdrant...")
        entity_chunks = self._get_chunks_for_entities(
            entities,
            accessible_doc_ids=params.accessible_doc_ids
        )
        logger.info(f"[GraphRAG Query]   - Entity chunks retrieved: {len(entity_chunks)}")

        # ALWAYS include direct Qdrant vector search results (with optional source and access control filtering)
        logger.info("[GraphRAG Query] Step 2.5: Direct vector search in Qdrant...")
        vector_chunks = await self._get_vector_search_chunks(
            query,
            params.top_k,
            source_names=params.source_names,
            accessible_doc_ids=params.accessible_doc_ids
        )
        logger.info(f"[GraphRAG Query]   - Vector chunks retrieved: {len(vector_chunks)}")

        # Combine and deduplicate chunks
        chunks = self._merge_chunks(entity_chunks, vector_chunks)
        logger.info(f"[GraphRAG Query] HYBRID MODE RESULT: {len(entities)} entities, {len(relationships)} relationships, {len(chunks)} chunks")

        return GraphRAGContext(
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            mode=QueryMode.HYBRID,
            enhanced_query=query,
        )

    def format_context(self, context: GraphRAGContext) -> str:
        """
        Format GraphRAG context for LLM consumption.

        Args:
            context: GraphRAGContext from query()

        Returns:
            Formatted string with entities, relationships, and chunks
        """
        parts = []

        # Prioritize chunks (actual document content) over entities/relationships
        # Chunks contain the raw invoice data which is most important for answering questions
        if context.chunks:
            parts.append("## Document Content (from 2025 invoices)\n")
            for i, chunk in enumerate(context.chunks[:15], 1):  # Increased from 5 to 15
                # Use page_content (from Qdrant Document) or content as fallback
                content = chunk.get("page_content", chunk.get("content", ""))
                if content:
                    # Don't truncate chunks - include full content for invoice data
                    parts.append(f"### Chunk {i}\n{content}\n---\n")

        if context.entities:
            parts.append("\n## Relevant Entities (extracted from invoices)\n")
            for entity in context.entities[:20]:  # Increased from 10 to 20
                name = entity.get("name", "Unknown")
                entity_type = entity.get("entity_type", "")
                description = entity.get("description", "")
                parts.append(f"- **{name}** ({entity_type}): {description}\n")

        if context.relationships:
            parts.append("\n## Relationships (connections in invoice data)\n")
            for rel in context.relationships[:20]:  # Increased from 10 to 20
                src = rel.get("src_name", "?")
                tgt = rel.get("tgt_name", "?")
                desc = rel.get("description", "related to")
                parts.append(f"- {src} → {tgt}: {desc}\n")

        return "".join(parts) if parts else ""

