"""Retry decorator for workflow nodes."""
import asyncio
import logging
import traceback
from functools import wraps

from ..constants import LlmFailureAction, NODE_TO_RESULT_KEY
from ..models.schemas import AgentType
from ..utils.llm_client import UpstreamExhaustedError
from .helpers import (
    reset_unattended_gate_budget,
    _retry_backoff_delay,
    WorkflowSupersededError,
    WorkflowTerminatedError,
)

logger = logging.getLogger(__name__)

MAX_NODE_RETRIES = 3


def with_retry(node_name: str, max_retries: int = MAX_NODE_RETRIES):
    """Decorator to add retry logic to workflow nodes.

    When UpstreamExhaustedError exhausts all retries (LLM timeout/5xx),
    blocks the workflow and waits for user intervention.

    Uses a while loop (not for) so that on 'retry' the attempt counter
    truly resets to 0.  Python's for-loop rebinds the variable each
    iteration, making in-loop assignment ineffective.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, state: 'WorkflowState') -> 'WorkflowState':
            reset_unattended_gate_budget(self, node_name)

            # Cooperative pause gate: if the workflow was paused, block here
            # before doing any work for this node. signal_resume() releases it.
            pause_event = getattr(self, '_pause_event', None)
            if pause_event is not None and not pause_event.is_set():
                logger.info(
                    f"[{getattr(self, 'session_id', '?')}] Node {node_name} "
                    f"pausing at entry, awaiting resume"
                )
                await pause_event.wait()
                logger.info(
                    f"[{getattr(self, 'session_id', '?')}] Node {node_name} "
                    f"resumed"
                )

            # Check skip_nodes
            if node_name in state.get("skip_nodes", []):
                logger.info(f"[{self.session_id}] Skipping node {node_name} (not activated by PI)")
                if hasattr(self, '_send_message'):
                    await self._send_message(
                        "status",
                        f"{node_name} skipped this round (not needed)",
                        None
                    )
                result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
                return {**state, "failed_node": None, result_key: {"findings": [], "status": "skipped"}}

            last_error = None
            attempt = 0  # manual counter — while loop, not for

            while attempt <= max_retries:
                try:
                    if attempt > 0:
                        await self._send_message(
                            "status",
                            f"Retrying {node_name} (attempt {attempt + 1}/{max_retries + 1})...",
                            AgentType.PI
                        )
                        await asyncio.sleep(_retry_backoff_delay(attempt))

                    result = await func(self, state)
                    if isinstance(result, dict):
                        result["failed_node"] = None
                    return result

                except UpstreamExhaustedError as e:
                    last_error = e
                    action = await self._wait_for_llm_intervention(
                        e, node_name, state["analysis_state"]
                    )
                    if action == LlmFailureAction.RETRY:
                        attempt = 0
                        continue  # re-enter while loop
                    if action == LlmFailureAction.TERMINATE:
                        # User chose to abort the whole analysis at the gate.
                        raise WorkflowTerminatedError(
                            f"User terminated analysis at node {node_name} "
                            f"after LLM failure"
                        ) from last_error
                    # skip
                    result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
                    return {
                        **state,
                        "failed_node": node_name,
                        result_key: {
                            "error": str(last_error),
                            "findings": [],
                            "status": "failed"
                        }
                    }

                except WorkflowTerminatedError:
                    # User chose to abort (e.g. TERMINATE at a HITL gate, or a
                    # disabled-agent terminate). Must propagate, not be retried
                    # by the generic handler below.
                    raise

                except WorkflowSupersededError:
                    # A snapshot restore replaced the state this run is built on.
                    # Retrying the node would re-enter a graph position that no
                    # longer exists, on a checkpointer connection the restore
                    # already closed. Propagate to the run loop, which unwinds
                    # without touching the state the restore just persisted.
                    raise

                except Exception as e:
                    last_error = e
                    error_detail = traceback.format_exc()
                    logger.error(
                        f"[{self.session_id}] Node {node_name} failed "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {error_detail}"
                    )
                    await self._send_message(
                        "error",
                        f"Node {node_name} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}",
                        AgentType.PI
                    )
                    attempt += 1

            # All retries exhausted for non-LLM error
            logger.error(
                f"[{self.session_id}] Node {node_name} failed after {max_retries + 1} attempts"
            )
            result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
            return {
                **state,
                "failed_node": node_name,
                result_key: {"error": str(last_error), "findings": [], "status": "failed"}
            }
        return wrapper
    return decorator
