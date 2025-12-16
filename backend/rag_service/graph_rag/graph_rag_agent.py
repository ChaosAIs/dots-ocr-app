"""
Graph-R1 Style Agent for Iterative Knowledge Graph Reasoning.

This module implements an agent-based approach for iterative reasoning
over the knowledge graph, following the Graph-R1 design principles:
1. Thinking: Agent decides whether to continue reasoning or terminate
2. Query Generation: Agent formulates retrieval queries based on current state
3. Graph Retrieval: Agent retrieves from Knowledge HyperGraph
4. Answering: Agent generates final response when sufficient knowledge gathered
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .base import QueryMode, QueryParam
from .agent_prompts import (
    AGENT_THINK_PROMPT,
    INITIAL_QUERY_PROMPT,
    KNOWLEDGE_FORMAT_TEMPLATE,
    NO_KNOWLEDGE_TEMPLATE,
    AGENT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State for the GraphRAG reasoning agent."""
    question: str
    step: int = 0
    max_steps: int = 5
    retrieved_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    queries_made: List[str] = field(default_factory=list)
    is_complete: bool = False
    final_answer: Optional[str] = None
    # Reward tracking for Graph-R1 alignment
    reward_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_knowledge_summary(self) -> str:
        """Format all retrieved knowledge for the prompt."""
        if not self.retrieved_knowledge:
            return "No knowledge retrieved yet."
        
        summaries = []
        for i, knowledge in enumerate(self.retrieved_knowledge, 1):
            entities = knowledge.get("entities", [])
            relationships = knowledge.get("relationships", [])
            
            if entities or relationships:
                entities_str = self._format_entities(entities)
                relationships_str = self._format_relationships(relationships)
                summaries.append(
                    KNOWLEDGE_FORMAT_TEMPLATE.format(
                        step=i,
                        entities=entities_str,
                        relationships=relationships_str,
                    )
                )
            else:
                summaries.append(NO_KNOWLEDGE_TEMPLATE.format(step=i))
        
        return "\n".join(summaries)
    
    def _format_entities(self, entities: List[Dict]) -> str:
        """Format entity list for display."""
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
    
    def _format_relationships(self, relationships: List[Dict]) -> str:
        """Format relationship list for display."""
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


@dataclass
class AgentConfig:
    """Configuration for the GraphRAG agent."""
    max_steps: int = 5
    min_entities_for_answer: int = 3
    min_score_threshold: float = 0.5
    top_k_per_step: int = 30
    enable_query_refinement: bool = True
    # Default retrieval mode for each step (auto = auto-detect per step)
    default_retrieval_mode: str = "auto"

    # Graph-R1 reward-based termination settings
    enable_reward_based_termination: bool = True
    format_reward_high_threshold: float = 0.8  # High confidence to terminate
    format_reward_low_threshold: float = 0.3   # Low quality, likely hallucination
    answer_reward_threshold: float = 0.6       # Minimum answer quality


# Type for retrieval callback function
from typing import Callable, Awaitable
RetrievalCallback = Callable[[str, QueryParam], Awaitable[Any]]


class GraphRAGAgent:
    """
    Agent-based GraphRAG with iterative reasoning.

    Implements the Graph-R1 style think-query-retrieve-answer loop.

    The agent is a WRAPPER on top of existing retrieval modes:
    - Each reasoning step uses LOCAL/GLOBAL/HYBRID for actual retrieval
    - Agent handles multi-round conversation and termination logic
    - Original retrieval modes are preserved and used as building blocks
    """

    def __init__(
        self,
        llm_service,
        retrieval_callback: RetrievalCallback,
        mode_detector=None,
        config: AgentConfig = None,
    ):
        """
        Initialize the GraphRAG agent.

        Args:
            llm_service: LLM service for agent reasoning
            retrieval_callback: Callback to parent GraphRAG for retrieval
                                (uses LOCAL/GLOBAL/HYBRID modes)
            mode_detector: Optional query mode detector for auto-detection
            config: Agent configuration
        """
        self.llm_service = llm_service
        self.retrieval_callback = retrieval_callback
        self.mode_detector = mode_detector
        self.config = config or AgentConfig()

        # Initialize reward scorer for Graph-R1 alignment
        from .reward_scorer import RewardScorer
        self.reward_scorer = RewardScorer()

        logger.info(
            f"[GraphRAG Agent] Initialized with max_steps={self.config.max_steps}, "
            f"default_retrieval_mode={self.config.default_retrieval_mode}, "
            f"reward_based_termination={self.config.enable_reward_based_termination}"
        )

    async def query(
        self,
        question: str,
        params: QueryParam = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute iterative reasoning query over the knowledge graph.

        Args:
            question: The user's question
            params: Query parameters

        Returns:
            Tuple of (answer, metadata)
        """
        params = params or QueryParam()
        state = AgentState(
            question=question,
            max_steps=self.config.max_steps,
        )

        logger.info("=" * 70)
        logger.info("[GraphRAG Agent] 🚀 STARTING ITERATIVE REASONING")
        logger.info("=" * 70)
        logger.info(f"[GraphRAG Agent] Question: {question}")
        logger.info(f"[GraphRAG Agent] Config: max_steps={self.config.max_steps}, "
                   f"default_mode={self.config.default_retrieval_mode}")
        logger.info("-" * 70)

        # Step 1: Generate initial query
        logger.info("[GraphRAG Agent] 📝 PHASE 1: Generating initial search query...")
        initial_query = await self._generate_initial_query(question)
        logger.info(f"[GraphRAG Agent] Initial query generated: '{initial_query}'")
        logger.info("-" * 70)

        # Iterative reasoning loop
        while state.step < state.max_steps and not state.is_complete:
            state.step += 1

            logger.info("=" * 50)
            logger.info(f"[GraphRAG Agent] 🔄 ITERATION {state.step}/{state.max_steps}")
            logger.info("=" * 50)

            # Determine query to use
            current_query = initial_query if state.step == 1 else state.queries_made[-1]
            logger.info(f"[GraphRAG Agent] Current query: '{current_query}'")

            # Step 2: Retrieve knowledge from graph
            logger.info(f"[GraphRAG Agent] 🔍 RETRIEVING knowledge from graph...")
            knowledge = await self._retrieve_knowledge(current_query, params)
            state.retrieved_knowledge.append(knowledge)
            state.queries_made.append(current_query)

            entity_count = len(knowledge.get('entities', []))
            rel_count = len(knowledge.get('relationships', []))
            chunk_count = len(knowledge.get('chunks', []))
            mode_used = knowledge.get('mode_used', 'unknown')

            logger.info(f"[GraphRAG Agent] Retrieved via {mode_used.upper()} mode:")
            logger.info(f"  - Entities: {entity_count}")
            logger.info(f"  - Relationships: {rel_count}")
            logger.info(f"  - Chunks: {chunk_count}")

            # Log some entity names for visibility
            if knowledge.get('entities'):
                entity_names = [e.get('name', e.get('id', 'unknown'))[:30]
                               for e in knowledge['entities'][:5]]
                logger.info(f"  - Sample entities: {entity_names}")

            # Step 3: Think - decide to continue or answer
            logger.info("-" * 50)
            logger.info(f"[GraphRAG Agent] 🤔 THINKING: Should I continue or answer?")
            response = await self._think_and_decide(state)

            # Log the LLM response (truncated for readability)
            response_preview = response[:500] + "..." if len(response) > 500 else response
            logger.info(f"[GraphRAG Agent] LLM Response:\n{response_preview}")
            logger.info("-" * 50)

            # Parse response for answer or next query
            answer = self._extract_answer(response)
            next_query = self._extract_query(response)

            logger.info(f"[GraphRAG Agent] 📊 PARSING DECISION:")
            logger.info(f"  - Found <answer> tag: {answer is not None}")
            logger.info(f"  - Found <query> tag: {next_query is not None}")

            # Reward-based termination logic (Graph-R1 alignment)
            if self.config.enable_reward_based_termination and state.reward_history:
                format_score = state.reward_history[-1]['format_score']

                # Decision 1: High format score + answer found = TERMINATE
                if format_score >= self.config.format_reward_high_threshold and answer:
                    state.is_complete = True
                    state.final_answer = answer
                    logger.info(f"[GraphRAG Agent] ✅ DECISION: TERMINATE")
                    logger.info(f"[GraphRAG Agent] Reason: High format score ({format_score:.2f}) + answer found")

                    # Compute answer quality
                    answer_score, answer_details = self.reward_scorer.compute_answer_reward_heuristic(
                        answer, state.question
                    )
                    logger.info(f"[GraphRAG Agent] Answer Quality Score: {answer_score:.2f}")
                    state.reward_history[-1]['answer_score'] = answer_score
                    state.reward_history[-1]['answer_details'] = answer_details

                    answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
                    logger.info(f"[GraphRAG Agent] Answer preview: {answer_preview}")

                # Decision 2: Low format score = TERMINATE (hallucination)
                elif format_score < self.config.format_reward_low_threshold:
                    state.is_complete = True
                    logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE")
                    logger.info(f"[GraphRAG Agent] Reason: Low format score ({format_score:.2f}) - likely hallucination")
                    logger.info(f"[GraphRAG Agent] Generating final answer from accumulated knowledge...")
                    state.final_answer = await self._generate_final_answer(state)

                # Decision 3: Medium format score + answer = Check quality
                elif answer and format_score >= 0.5:
                    answer_score, answer_details = self.reward_scorer.compute_answer_reward_heuristic(
                        answer, state.question
                    )
                    logger.info(f"[GraphRAG Agent] Answer Quality Score: {answer_score:.2f}")

                    state.reward_history[-1]['answer_score'] = answer_score
                    state.reward_history[-1]['answer_details'] = answer_details

                    if answer_score >= self.config.answer_reward_threshold:
                        state.is_complete = True
                        state.final_answer = answer
                        logger.info(f"[GraphRAG Agent] ✅ DECISION: TERMINATE")
                        logger.info(f"[GraphRAG Agent] Reason: Acceptable format ({format_score:.2f}) + good answer ({answer_score:.2f})")
                    else:
                        logger.info(f"[GraphRAG Agent] 🔄 DECISION: CONTINUE")
                        logger.info(f"[GraphRAG Agent] Reason: Answer quality too low ({answer_score:.2f})")
                        # Continue to next iteration

                # Decision 4: Has next query = CONTINUE
                elif next_query:
                    if next_query not in state.queries_made:
                        state.queries_made.append(next_query)
                        logger.info(f"[GraphRAG Agent] 🔄 DECISION: CONTINUE")
                        logger.info(f"[GraphRAG Agent] Next query: '{next_query}'")
                    else:
                        # Duplicate query, terminate
                        state.is_complete = True
                        logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE")
                        logger.info(f"[GraphRAG Agent] Reason: Duplicate query detected")
                        state.final_answer = await self._generate_final_answer(state)

                # Decision 5: No valid tags = TERMINATE
                else:
                    state.is_complete = True
                    logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE")
                    logger.info(f"[GraphRAG Agent] Reason: No valid tags found")
                    state.final_answer = await self._generate_final_answer(state)

            else:
                # Fallback to original rule-based logic
                if answer:
                    state.is_complete = True
                    state.final_answer = answer
                    logger.info(f"[GraphRAG Agent] ✅ DECISION: TERMINATE - Answer found")
                    answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
                    logger.info(f"[GraphRAG Agent] Answer preview: {answer_preview}")
                elif next_query:
                    if next_query not in state.queries_made:
                        state.queries_made.append(next_query)
                        logger.info(f"[GraphRAG Agent] 🔄 DECISION: CONTINUE - New query generated")
                        logger.info(f"[GraphRAG Agent] Next query: '{next_query}'")
                    else:
                        # Duplicate query, terminate
                        state.is_complete = True
                        logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE - Duplicate query")
                        state.final_answer = await self._generate_final_answer(state)
                else:
                    # No valid tags, terminate
                    state.is_complete = True
                    logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE - No valid tags found")
                    state.final_answer = await self._generate_final_answer(state)

        # Check if max steps reached
        if state.step >= state.max_steps and not state.is_complete:
            logger.info(f"[GraphRAG Agent] ⚠️ MAX STEPS REACHED ({state.max_steps})")
            logger.info(f"[GraphRAG Agent] Termination reason: Exceeded maximum iteration limit")

        # If loop ended without answer, generate one
        if not state.final_answer:
            logger.info(f"[GraphRAG Agent] Generating final answer from accumulated knowledge...")
            state.final_answer = await self._generate_final_answer(state)

        # Collect all entities, relationships, and chunks from all retrieval steps
        all_entities = []
        all_relationships = []
        all_chunks = []
        modes_used = []
        seen_entity_ids = set()

        for knowledge in state.retrieved_knowledge:
            # Deduplicate entities
            for entity in knowledge.get("entities", []):
                entity_id = entity.get("id")
                if entity_id and entity_id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity_id)
                elif not entity_id:
                    all_entities.append(entity)

            # Add relationships
            all_relationships.extend(knowledge.get("relationships", []))

            # Add chunks
            all_chunks.extend(knowledge.get("chunks", []))

            # Track modes used
            mode_used = knowledge.get("mode_used")
            if mode_used and mode_used not in modes_used:
                modes_used.append(mode_used)

        metadata = {
            "steps": state.step,
            "queries_made": state.queries_made,
            "modes_used": modes_used,
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "mode": "agent_iterative",
            # Include actual data for parent GraphRAG
            "all_entities": all_entities,
            "all_relationships": all_relationships,
            "all_chunks": all_chunks,
            # Graph-R1 reward history
            "reward_history": state.reward_history,
        }

        # Log reward summary if available
        if state.reward_history:
            logger.info("=" * 70)
            logger.info("[GraphRAG Agent] 📊 REWARD SUMMARY")
            logger.info("=" * 70)
            for i, reward_data in enumerate(state.reward_history, 1):
                logger.info(f"Step {i}:")
                logger.info(f"  Format Score: {reward_data.get('format_score', 'N/A'):.2f}")
                if 'answer_score' in reward_data:
                    logger.info(f"  Answer Score: {reward_data['answer_score']:.2f}")
            logger.info("=" * 70)

        # Final summary log
        logger.info("=" * 70)
        logger.info("[GraphRAG Agent] 🏁 AGENT QUERY COMPLETE")
        logger.info("=" * 70)
        logger.info(f"[GraphRAG Agent] Summary:")
        logger.info(f"  - Total iterations: {state.step}")
        logger.info(f"  - Queries made: {state.queries_made}")
        logger.info(f"  - Retrieval modes used: {modes_used}")
        logger.info(f"  - Total unique entities: {len(all_entities)}")
        logger.info(f"  - Total relationships: {len(all_relationships)}")
        logger.info(f"  - Total chunks: {len(all_chunks)}")
        logger.info(f"  - Is complete: {state.is_complete}")
        final_preview = state.final_answer[:300] + "..." if len(state.final_answer) > 300 else state.final_answer
        logger.info(f"[GraphRAG Agent] Final answer preview:\n{final_preview}")
        logger.info("=" * 70)

        return state.final_answer, metadata

    async def _generate_initial_query(self, question: str) -> str:
        """Generate the initial search query from the question."""
        logger.info(f"[GraphRAG Agent] Generating initial query from question...")
        prompt = INITIAL_QUERY_PROMPT.format(question=question)

        response = await self.llm_service.generate(
            prompt=prompt,
            max_tokens=200,
        )

        logger.info(f"[GraphRAG Agent] LLM initial query response: {response[:200]}...")

        # Extract query from response
        query = self._extract_query(response)
        if query:
            logger.info(f"[GraphRAG Agent] Extracted initial query: '{query}'")
        else:
            logger.info(f"[GraphRAG Agent] No <query> tag found, using original question as query")
        return query if query else question  # Fallback to original question

    async def _retrieve_knowledge(
        self,
        query: str,
        params: QueryParam,
    ) -> Dict[str, Any]:
        """
        Retrieve entities and relationships using the parent GraphRAG's retrieval.

        This delegates to the original LOCAL/GLOBAL/HYBRID modes for actual
        graph retrieval. The agent is just a wrapper for multi-round reasoning.
        """
        # Determine retrieval mode for this step
        retrieval_mode = self.config.default_retrieval_mode
        logger.info(f"[GraphRAG Agent] Retrieval: configured mode = '{retrieval_mode}'")

        if retrieval_mode == "auto" and self.mode_detector:
            # Auto-detect best mode for this specific query
            logger.info(f"[GraphRAG Agent] Auto-detecting best retrieval mode for query...")
            detected_mode, _ = await self.mode_detector.detect_mode(query)
            # Don't use AGENT mode recursively - use HYBRID as default
            if detected_mode == QueryMode.AGENT:
                retrieval_mode = "hybrid"
                logger.info(f"[GraphRAG Agent] Detected AGENT mode, using HYBRID instead")
            else:
                retrieval_mode = detected_mode.value
            logger.info(f"[GraphRAG Agent] Auto-detected mode: {retrieval_mode.upper()}")
        elif retrieval_mode == "auto":
            # No mode detector, default to HYBRID for comprehensive retrieval
            retrieval_mode = "hybrid"
            logger.info(f"[GraphRAG Agent] No mode detector, defaulting to HYBRID")

        # Call parent GraphRAG's retrieval (LOCAL/GLOBAL/HYBRID)
        logger.info(f"[GraphRAG Agent] Calling GraphRAG.{retrieval_mode}_retrieval()...")
        context = await self.retrieval_callback(query, retrieval_mode, params)

        logger.info(f"[GraphRAG Agent] Retrieval complete via {retrieval_mode.upper()}")

        return {
            "entities": context.entities,
            "relationships": context.relationships,
            "chunks": context.chunks,
            "query": query,
            "mode_used": retrieval_mode,
        }

    async def _think_and_decide(self, state: AgentState) -> str:
        """Agent thinks about current state and decides next action."""
        logger.info(f"[GraphRAG Agent] Preparing think prompt with {len(state.retrieved_knowledge)} knowledge items...")

        knowledge_summary = state.get_knowledge_summary()
        logger.info(f"[GraphRAG Agent] Knowledge summary length: {len(knowledge_summary)} chars")

        prompt = AGENT_THINK_PROMPT.format(
            question=state.question,
            step=state.step,
            max_steps=state.max_steps,
            retrieved_knowledge=knowledge_summary,
        )

        system_prompt = AGENT_SYSTEM_PROMPT.format(max_steps=state.max_steps)

        logger.info(f"[GraphRAG Agent] Sending to LLM for thinking...")
        logger.info(f"[GraphRAG Agent] Prompt length: {len(prompt)} chars")

        response = await self.llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1500,
        )

        logger.info(f"[GraphRAG Agent] LLM response received: {len(response)} chars")

        # Compute reward scores for Graph-R1 alignment
        if self.config.enable_reward_based_termination:
            format_score, format_details = self.reward_scorer.compute_format_reward(response)

            logger.info("=" * 50)
            logger.info("[GraphRAG Agent] 📊 REWARD SCORING")
            logger.info("=" * 50)
            logger.info(f"[GraphRAG Agent] Format Reward: {format_score:.2f}")
            logger.info(f"[GraphRAG Agent] Format Details: {format_details['breakdown']}")

            # Store in state for later analysis
            state.reward_history.append({
                'step': state.step,
                'format_score': format_score,
                'format_details': format_details,
            })

        return response

    async def _generate_final_answer(self, state: AgentState) -> str:
        """Generate final answer from accumulated knowledge."""
        logger.info(f"[GraphRAG Agent] 📝 Generating FINAL ANSWER from accumulated knowledge...")
        logger.info(f"[GraphRAG Agent] Knowledge items accumulated: {len(state.retrieved_knowledge)}")

        prompt = f"""Based on all the retrieved knowledge, provide a comprehensive answer to the question.

## Question
{state.question}

## All Retrieved Knowledge
{state.get_knowledge_summary()}

## Instructions
Synthesize the information from all retrieval steps to provide a complete, accurate answer.
If the retrieved knowledge is insufficient, clearly state what information is missing.

Your answer:"""

        logger.info(f"[GraphRAG Agent] Sending final answer prompt to LLM...")

        response = await self.llm_service.generate(
            prompt=prompt,
            max_tokens=2000,
        )

        logger.info(f"[GraphRAG Agent] Final answer generated: {len(response)} chars")

        return response

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from response if present.

        Tries multiple patterns to handle various LLM output formats.
        """
        # Try standard format first
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try with newlines inside tags
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Check if response contains answer-like conclusion without tags
        # This is a fallback for LLMs that don't follow format
        if not self._extract_query(response):
            # No query tag found - check if this looks like a final answer
            answer_indicators = [
                "the answer is", "in conclusion", "to summarize",
                "based on the knowledge", "based on the retrieved",
                "therefore", "thus", "in summary"
            ]
            lower_response = response.lower()
            for indicator in answer_indicators:
                if indicator in lower_response:
                    logger.warning(
                        f"[GraphRAG Agent] LLM didn't use tags, "
                        f"inferring answer from response with indicator: {indicator}"
                    )
                    return response.strip()

        return None

    def _extract_query(self, response: str) -> Optional[str]:
        """Extract query from response if present.

        Tries multiple patterns to handle various LLM output formats.
        """
        # Try standard format first
        match = re.search(r"<query>(.*?)</query>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try with newlines inside tags
        match = re.search(r"<query>\s*(.*?)\s*</query>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _has_valid_tags(self, response: str) -> bool:
        """Check if response contains valid answer or query tags."""
        return bool(
            re.search(r"<answer>.*?</answer>", response, re.DOTALL) or
            re.search(r"<query>.*?</query>", response, re.DOTALL)
        )

