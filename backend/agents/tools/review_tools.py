"""
Review tools for the Reviewer Agent.

These tools handle:
- Validating completeness of results
- Checking data quality
- Approving or requesting refinement
"""

import json
import logging
from typing import Annotated, List

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agents.state.models import AgentType
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)


@tool
def validate_completeness(
    state: Annotated[dict, InjectedState]
) -> str:
    """Check if all planned tasks have corresponding outputs.

    Validates that every task in the execution plan has a result
    from one of the retrieval agents.

    This tool reads execution_plan and agent_outputs directly from state.

    Returns:
        Validation result with missing tasks if any
    """
    try:
        # Read directly from state
        plan = state.get("execution_plan")
        outputs = state.get("agent_outputs", [])

        logger.info(f"[validate_completeness] Plan present: {plan is not None}, Outputs: {len(outputs)}")

        # Handle case where plan/outputs might be model objects
        if hasattr(plan, 'model_dump'):
            plan = plan.model_dump()
        elif hasattr(plan, 'dict'):
            plan = plan.dict()
        elif plan is None:
            plan = {}

        # Normalize outputs
        normalized_outputs = []
        for o in outputs:
            if hasattr(o, 'model_dump'):
                normalized_outputs.append(o.model_dump())
            elif hasattr(o, 'dict'):
                normalized_outputs.append(o.dict())
            elif isinstance(o, dict):
                normalized_outputs.append(o)
        outputs = normalized_outputs

        # Get planned task IDs
        planned_tasks = plan.get("sub_tasks", [])
        planned_ids = {t.get("task_id") or t.get("id") for t in planned_tasks}

        # Get completed task IDs
        completed_ids = {o.get("task_id") for o in outputs if o.get("task_id")}

        # Find missing
        missing_ids = planned_ids - completed_ids

        # Find failed tasks
        failed_ids = {
            o.get("task_id") for o in outputs
            if o.get("status") == "failed"
        }

        is_complete = len(missing_ids) == 0

        result = {
            "is_complete": is_complete,
            "total_planned": len(planned_ids),
            "total_completed": len(completed_ids),
            "missing_tasks": list(missing_ids),
            "failed_tasks": list(failed_ids),
            "completion_rate": len(completed_ids) / len(planned_ids) if planned_ids else 0
        }

        if is_complete and not failed_ids:
            result["message"] = "COMPLETE: All tasks have results"
        elif missing_ids:
            result["message"] = f"INCOMPLETE: Missing results for {len(missing_ids)} tasks: {list(missing_ids)}"
        elif failed_ids:
            result["message"] = f"PARTIAL: {len(failed_ids)} tasks failed: {list(failed_ids)}"

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error validating completeness: {e}")
        return json.dumps({
            "is_complete": False,
            "error": str(e),
            "message": f"ERROR: Could not validate completeness: {e}"
        })


@tool
def check_data_quality(
    state: Annotated[dict, InjectedState]
) -> str:
    """Evaluate data quality and consistency across agent outputs.

    Checks for:
    - Low confidence scores
    - Empty or null data
    - Inconsistencies between related results
    - Error indicators

    This tool reads agent_outputs directly from state - no arguments needed.

    Returns:
        Quality assessment with issues and scores
    """
    try:
        # Read directly from state
        outputs = state.get("agent_outputs", [])

        logger.info(f"[check_data_quality] Found {len(outputs)} outputs to review")

        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs else []

        # Convert Pydantic models to dicts if needed
        normalized_outputs = []
        for o in outputs:
            if hasattr(o, 'model_dump'):
                normalized_outputs.append(o.model_dump())
            elif hasattr(o, 'dict'):
                normalized_outputs.append(o.dict())
            elif isinstance(o, dict):
                normalized_outputs.append(o)
            else:
                logger.warning(f"Unexpected output type in review: {type(o)}")
                continue
        outputs = normalized_outputs

        issues = []
        quality_scores = {}
        confidence_threshold = AGENTIC_CONFIG.get("default_confidence_threshold", 0.7)
        low_threshold = AGENTIC_CONFIG.get("low_confidence_threshold", 0.5)

        for output in outputs:
            task_id = output.get("task_id", "unknown")
            confidence = output.get("confidence", 0)
            status = output.get("status", "unknown")
            data = output.get("data")
            output_issues = output.get("issues", [])

            # Calculate quality score
            score = confidence

            # Penalize for issues
            if output_issues:
                score *= 0.9
                issues.extend([f"{task_id}: {issue}" for issue in output_issues])

            # Penalize for failed status
            if status == "failed":
                score = 0.0
                issues.append(f"{task_id}: Task failed")
            elif status == "partial":
                score *= 0.8

            # Check for empty data
            if data is None or (isinstance(data, (list, dict)) and len(data) == 0):
                score *= 0.5
                issues.append(f"{task_id}: Empty or null data")

            # Check confidence thresholds
            if confidence < low_threshold:
                issues.append(f"{task_id}: Very low confidence ({confidence:.2f})")
            elif confidence < confidence_threshold:
                issues.append(f"{task_id}: Below threshold confidence ({confidence:.2f})")

            quality_scores[task_id] = round(score, 2)

        # Calculate overall quality
        avg_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0

        # Determine overall assessment
        if avg_score >= confidence_threshold and not issues:
            assessment = "QUALITY OK: All outputs meet quality thresholds"
        elif avg_score >= low_threshold:
            assessment = f"QUALITY ACCEPTABLE: Average score {avg_score:.2f}, {len(issues)} issues"
        else:
            assessment = f"QUALITY ISSUES: Average score {avg_score:.2f}, {len(issues)} issues"

        return json.dumps({
            "assessment": assessment,
            "average_score": round(avg_score, 2),
            "quality_scores": quality_scores,
            "issues": issues,
            "issue_count": len(issues),
            "meets_threshold": avg_score >= confidence_threshold
        })

    except Exception as e:
        logger.error(f"Error checking data quality: {e}")
        return json.dumps({
            "assessment": f"ERROR: Could not check quality: {e}",
            "average_score": 0,
            "issues": [str(e)],
            "meets_threshold": False
        })


@tool
def approve_and_continue(
    reasoning: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Approve outputs and hand off to summary agent.

    Called when all results pass quality checks and are ready
    for final summarization.

    Args:
        reasoning: Explanation of why results are approved

    Returns:
        JSON string with approval decision
    """
    try:
        # Get agent outputs from state
        agent_outputs = state.get("agent_outputs", [])

        # Extract task IDs
        approved_ids = []
        quality_scores = {}

        for output in agent_outputs:
            # Convert Pydantic model to dict if needed
            if hasattr(output, 'model_dump'):
                output = output.model_dump()
            elif hasattr(output, 'dict'):
                output = output.dict()

            task_id = output.get("task_id", "") if isinstance(output, dict) else getattr(output, 'task_id', "")
            if task_id:
                approved_ids.append(task_id)
                confidence = output.get("confidence", 0.7) if isinstance(output, dict) else getattr(output, 'confidence', 0.7)
                quality_scores[task_id] = confidence

        logger.info(f"Reviewer approved {len(approved_ids)} tasks: {reasoning}")

        return json.dumps({
            "success": True,
            "decision": "approve",
            "approved_task_ids": approved_ids,
            "quality_scores": quality_scores,
            "reasoning": reasoning,
            "next_step": "summary_agent",
            "message": f"Review approved: {len(approved_ids)} tasks passed. {reasoning}"
        })

    except Exception as e:
        logger.error(f"Error in approve_and_continue: {e}")
        return json.dumps({
            "success": True,
            "decision": "approve",
            "approved_task_ids": [],
            "reasoning": f"Approved with error: {e}",
            "next_step": "summary_agent",
            "message": f"Review approved with warning: {e}",
            "error": str(e)
        })


@tool
def request_refinement(
    task_ids: str,
    guidance: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Request refinement from retrieval team for specific tasks.

    Called when some results need improvement. Creates refinement
    requests and sends back to the retrieval team.

    Args:
        task_ids: JSON array of task IDs that need refinement
        guidance: Instructions for how to improve results

    Returns:
        JSON string with refinement decision
    """
    try:
        ids_to_refine = json.loads(task_ids) if task_ids else []

        # Get current iteration
        iteration = state.get("review_iteration", 0)
        max_iter = state.get("max_iterations", AGENTIC_CONFIG.get("max_review_iterations", 3))

        # Check if we've hit max iterations
        if iteration >= max_iter:
            logger.warning(f"Max iterations ({max_iter}) reached, forcing approval")

            # Force approval
            agent_outputs = state.get("agent_outputs", [])
            approved_ids = [
                (o.task_id if hasattr(o, 'task_id') else o.get("task_id", ""))
                for o in agent_outputs
            ]

            return json.dumps({
                "success": True,
                "decision": "approve",
                "approved_task_ids": approved_ids,
                "reasoning": f"Max iterations ({max_iter}) reached. Proceeding with available data. Original guidance: {guidance}",
                "next_step": "summary_agent",
                "message": "Max iterations reached. Proceeding with available results."
            })

        # Get execution plan to find target agents
        execution_plan = state.get("execution_plan")
        if hasattr(execution_plan, 'dict'):
            execution_plan = execution_plan.dict()

        # Build refinement requests
        refinement_requests = []
        task_agent_map = {}

        if execution_plan:
            for task in execution_plan.get("sub_tasks", []):
                task_id = task.get("task_id")
                if task_id in ids_to_refine:
                    task_agent_map[task_id] = task.get("target_agent", AgentType.GENERIC_DOC_AGENT.value)

        for task_id in ids_to_refine:
            target_agent = task_agent_map.get(task_id, AgentType.GENERIC_DOC_AGENT.value)
            refinement_requests.append({
                "task_id": task_id,
                "target_agent": target_agent,
                "issue": "Results did not meet quality threshold",
                "guidance": guidance,
                "new_parameters": {}
            })

        logger.info(f"Reviewer requesting refinement for tasks {ids_to_refine}: {guidance}")

        return json.dumps({
            "success": True,
            "decision": "refine",
            "task_ids": ids_to_refine,
            "refinement_requests": refinement_requests,
            "iteration": iteration + 1,
            "max_iterations": max_iter,
            "reasoning": f"Refinement requested for {len(ids_to_refine)} tasks (iteration {iteration + 1}/{max_iter})",
            "next_step": "retrieval_team",
            "message": f"Refinement requested for {len(ids_to_refine)} tasks: {guidance}"
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in request_refinement: {e}")
        return json.dumps({
            "success": True,
            "decision": "approve",
            "approved_task_ids": [],
            "reasoning": f"Could not parse refinement request, proceeding: {e}",
            "next_step": "summary_agent",
            "message": f"Refinement parsing failed, proceeding with approval: {e}",
            "error": str(e)
        })
    except Exception as e:
        logger.error(f"Error in request_refinement: {e}")
        return json.dumps({
            "success": True,
            "decision": "approve",
            "approved_task_ids": [],
            "reasoning": f"Error during refinement request, proceeding: {e}",
            "next_step": "summary_agent",
            "message": f"Refinement error, proceeding with approval: {e}",
            "error": str(e)
        })
