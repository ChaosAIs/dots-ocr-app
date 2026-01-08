"""
Reviewer Agent implementation.

The Reviewer Agent handles quality control:
- Validates completeness of results
- Checks data quality and consistency
- Decides to approve or request refinement
"""

import json
import logging
from typing import Optional, Dict, Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agents.tools.review_tools import (
    validate_completeness,
    check_data_quality,
    approve_and_continue,
    request_refinement
)
from agents.prompts.reviewer_prompts import REVIEWER_SYSTEM_PROMPT
from agents.state.models import ReviewDecision, ReviewDecisionType

logger = logging.getLogger(__name__)


def create_reviewer_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the Reviewer Agent.

    The Reviewer Agent validates retrieval results and decides
    whether to approve them for summarization or request refinement.

    This implementation uses a simple StateGraph instead of create_react_agent
    to ensure proper state passing from the parent workflow. The ReAct agent
    wrapper doesn't correctly pass InjectedState to tools.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or REVIEWER_SYSTEM_PROMPT

    def review_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main review node that validates outputs and makes a decision.

        This node directly calls the review tools with the state,
        ensuring proper state access.
        """
        logger.info("[ReviewerAgent] Starting review process...")

        # Step 1: Validate completeness
        logger.info("[ReviewerAgent] Step 1: Validating completeness...")
        completeness_result = validate_completeness.invoke({"state": state})
        try:
            completeness = json.loads(completeness_result)
        except (json.JSONDecodeError, TypeError):
            completeness = {"is_complete": False, "error": "Failed to parse completeness result"}

        logger.info(f"[ReviewerAgent] Completeness: {completeness.get('message', 'N/A')}")

        # Step 2: Check data quality
        logger.info("[ReviewerAgent] Step 2: Checking data quality...")
        quality_result = check_data_quality.invoke({"state": state})
        try:
            quality = json.loads(quality_result)
        except (json.JSONDecodeError, TypeError):
            quality = {"meets_threshold": False, "error": "Failed to parse quality result"}

        logger.info(f"[ReviewerAgent] Quality: {quality.get('assessment', 'N/A')}")

        # Step 3: Make decision based on results
        iteration = state.get("review_iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        # Decision logic
        is_complete = completeness.get("is_complete", False)
        meets_threshold = quality.get("meets_threshold", False)
        avg_score = quality.get("average_score", 0)
        issues = quality.get("issues", [])
        failed_tasks = completeness.get("failed_tasks", [])

        # Force approval if max iterations reached
        if iteration >= max_iterations:
            logger.info(f"[ReviewerAgent] Max iterations ({max_iterations}) reached, forcing approval")
            decision = ReviewDecision(
                decision=ReviewDecisionType.APPROVE,
                reasoning=f"Max iterations ({max_iterations}) reached. Proceeding with available data.",
                approved_task_ids=list(completeness.get("total_completed", 0)),
                refinement_requests=[]
            )
            approve_result = approve_and_continue.invoke({
                "reasoning": decision.reasoning,
                "state": state
            })
            return {
                "review_decision": decision,
                "review_iteration": iteration + 1
            }

        # Decide: approve or refine
        if is_complete and meets_threshold:
            # All good - approve
            logger.info("[ReviewerAgent] Quality checks passed, approving...")
            decision = ReviewDecision(
                decision=ReviewDecisionType.APPROVE,
                reasoning=f"All {completeness.get('total_completed', 0)} tasks completed with average quality score {avg_score:.2f}",
                approved_task_ids=[],
                refinement_requests=[]
            )
            approve_result = approve_and_continue.invoke({
                "reasoning": decision.reasoning,
                "state": state
            })
        elif is_complete and avg_score >= 0.5:
            # Complete but quality could be better - still approve if acceptable
            logger.info("[ReviewerAgent] Quality acceptable, approving with notes...")
            decision = ReviewDecision(
                decision=ReviewDecisionType.APPROVE,
                reasoning=f"Tasks completed with acceptable quality (score: {avg_score:.2f}). Issues: {len(issues)}",
                approved_task_ids=[],
                refinement_requests=[]
            )
            approve_result = approve_and_continue.invoke({
                "reasoning": decision.reasoning,
                "state": state
            })
        elif failed_tasks or (not is_complete and iteration < max_iterations - 1):
            # Need refinement
            tasks_to_refine = failed_tasks if failed_tasks else completeness.get("missing_tasks", [])
            if not tasks_to_refine:
                # No specific tasks to refine, just approve
                logger.info("[ReviewerAgent] No specific tasks to refine, approving...")
                decision = ReviewDecision(
                    decision=ReviewDecisionType.APPROVE,
                    reasoning=f"No specific refinement targets identified. Score: {avg_score:.2f}",
                    approved_task_ids=[],
                    refinement_requests=[]
                )
                approve_result = approve_and_continue.invoke({
                    "reasoning": decision.reasoning,
                    "state": state
                })
            else:
                logger.info(f"[ReviewerAgent] Requesting refinement for tasks: {tasks_to_refine}")
                guidance = f"Improve results. Issues found: {'; '.join(issues[:3]) if issues else 'Low confidence scores'}"
                decision = ReviewDecision(
                    decision=ReviewDecisionType.REFINE,
                    reasoning=f"Refinement needed for {len(tasks_to_refine)} tasks (iteration {iteration + 1}/{max_iterations})",
                    approved_task_ids=[],
                    refinement_requests=[{"task_id": tid, "guidance": guidance} for tid in tasks_to_refine]
                )
                refine_result = request_refinement.invoke({
                    "task_ids": json.dumps(tasks_to_refine),
                    "guidance": guidance,
                    "state": state
                })
        else:
            # Default: approve with what we have
            logger.info("[ReviewerAgent] Defaulting to approval...")
            decision = ReviewDecision(
                decision=ReviewDecisionType.APPROVE,
                reasoning=f"Proceeding with available results. Quality score: {avg_score:.2f}",
                approved_task_ids=[],
                refinement_requests=[]
            )
            approve_result = approve_and_continue.invoke({
                "reasoning": decision.reasoning,
                "state": state
            })

        # Log decision (handle both enum and string cases due to use_enum_values config)
        decision_value = decision.decision.value if hasattr(decision.decision, 'value') else decision.decision
        logger.info(f"[ReviewerAgent] Decision: {decision_value}")
        logger.info(f"[ReviewerAgent] Reason: {decision.reasoning}")

        return {
            "review_decision": decision,
            "review_iteration": iteration + 1
        }

    # Build a simple single-node graph
    builder = StateGraph(dict)
    builder.add_node("review", review_node)
    builder.set_entry_point("review")
    builder.add_edge("review", END)

    agent = builder.compile()

    logger.info("Created Reviewer Agent with direct state access (no ReAct wrapper)")

    return agent
