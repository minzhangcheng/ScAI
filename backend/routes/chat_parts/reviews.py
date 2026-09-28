"""Chat-owned supervision and bounded PI review on the live kernel."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from backend.agents.supervisor_agent import SupervisorAgent, SupervisorEvalResult
from backend.agents.supervisor.shared import _serialize_record
from backend.models.schemas import AgentType
from backend.models.schema_parts.analysis import AnalysisStep, CodeExecution
from backend.workflow.callbacks import CallbacksMixin

logger = logging.getLogger(__name__)

CHAT_EXECUTION_STEPS = 60
CHAT_CORRECTION_CYCLES = 2
CHAT_CORRECTION_STEPS = 20


class ChatReviewCallbacks:
    """Adapt the workflow's figure fallback without constructing a workflow.

    Keep PI -> bio self-review -> Supervisor -> unavailable behavior, including
    unavailable-vision handling and the producer-owned revision loop.
    """

    _figure_review_callback = CallbacksMixin._figure_review_callback
    _vision_figure_review = CallbacksMixin._vision_figure_review
    _bio_self_figure_review = CallbacksMixin._bio_self_figure_review
    _supervisor_figure_review = CallbacksMixin._supervisor_figure_review

    def __init__(self, state, provider, pi_agent, send, pi_callback, sandbox):
        self.session_id = state.id
        self._current_analysis_state = state
        self._provider = provider
        self.pi_agent = pi_agent
        self.bioinformatics_agent = None
        self._send_message = send
        self.supervisor_agent = None
        self.observations = []
        self._pi_callback = pi_callback
        self._sandbox = sandbox

    @property
    def _vision_settings(self):
        return {name: self._provider.get_vision_setting(name)
                for name in ("pi", "bioinformatics", "supervisor")}

    def ensure_agent_ready(self, name):
        if name != "supervisor" or not self._provider.get_model(name):
            self.supervisor_agent = None
            return False
        if self.supervisor_agent is None:
            async def notify(kind, content):
                await self._send_message(kind, content, AgentType.SUPERVISOR)

            self.supervisor_agent = SupervisorAgent(
                settings_provider=self._provider, session_id=self.session_id,
                pi_callback=self._pi_callback, message_callback=notify,
            )
            self.supervisor_agent.set_sandbox(self._sandbox)
        return True

    async def supervise(self, step, figures):
        if not self.ensure_agent_ready("supervisor"):
            return SupervisorEvalResult()
        try:
            return await self.supervisor_agent.evaluate_step(step, figures or [])
        except Exception:
            logger.warning("Chat Supervisor review unavailable", exc_info=True)
            self.observations.append({
                "step_id": step.id, "review_unavailable": True,
                "reason": "Supervisor evaluation failed; no verdict obtained.",
            })
            return SupervisorEvalResult()

    def supervision_records(self):
        if self.supervisor_agent is None:
            return []
        return [_serialize_record(item, self._current_analysis_state.current_round)
                for item in self.supervisor_agent.supervision_log]


def _merge_records(previous, current):
    """Preserve each receipt once, including failed and superseded attempts."""
    records = {}
    for item in [*(previous or []), *(current or [])]:
        if not isinstance(item, dict):
            item = item.model_dump(mode="json")
        records[item.get("id") or json.dumps(item, sort_keys=True)] = item
    return list(records.values())


def _persist_cycle_evidence(state, current, review_id, scope):
    """Retain Chat receipts without assigning them to an ordinary round plan."""
    execution_ids = set(current.get("attempt_execution_ids") or [])
    existing_code = {item.id for item in state.code_history}
    existing_steps = {item.id for item in state.analysis_log}
    step_ids = []
    for item in current.get("code_history") or []:
        receipt = CodeExecution.model_validate(item)
        if receipt.id not in execution_ids:
            continue
        if receipt.executor_run_kind not in (None, "chat") or receipt.executor_run_scope_id not in (None, scope):
            raise ValueError("Chat execution receipt has a conflicting owner")
        if receipt.id not in existing_code:
            state.code_history.append(receipt.model_copy(update={
                "executor_run_kind": "chat", "executor_run_scope_id": scope,
                "chat_review_id": review_id, "plan_id": None, "plan_step_ids": [],
            }))
            existing_code.add(receipt.id)
    for item in current.get("analysis_log") or []:
        step = AnalysisStep.model_validate(item)
        if step.id in existing_steps:
            continue
        state.analysis_log.append(step.model_copy(update={
            "executor_run_kind": "chat", "executor_run_scope_id": scope,
            "chat_review_id": review_id, "plan_id": None, "plan_step_ids": [],
        }))
        existing_steps.add(step.id)
        step_ids.append(step.id)
    return step_ids


async def run_reviewed_analysis(
    *, state, instruction, bio_agent, reviews, persist, collect_figures,
    execution_context,
    source_decision=None,
):
    """Review and correct within one Chat ownership reservation and kernel."""
    record = {
        "id": f"chat_review_{uuid.uuid4().hex}",
        "created_at": datetime.now().isoformat(),
        "request": instruction, "status": "running", "cycles": [],
        "execution_budget": CHAT_EXECUTION_STEPS,
        "correction_limit": CHAT_CORRECTION_CYCLES,
        "branch_id": getattr(state, "restore_branch_id", None) or "main",
    }
    if source_decision is not None:
        record["adata_source_decision"] = dict(source_decision)
    state.chat_analysis_reviews.append(record)
    remaining = CHAT_EXECUTION_STEPS
    result = {}
    next_instruction = instruction + (
        "\n\nUse generate_publication_figure for figures intended for the final "
        "Chat answer. Raw execute_code autosaves are diagnostic artifacts, not "
        "reviewed final figures. Supervisor advice is available during execution; "
        "the PI will review the executed evidence before the final answer."
    )
    scopes = []
    try:
        for cycle in range(CHAT_CORRECTION_CYCLES + 1):
            scope = f"chat_{uuid.uuid4().hex}"
            scopes.append(scope)
            allowance = remaining if cycle == 0 else min(remaining, CHAT_CORRECTION_STEPS)
            attempt = {"run_scope_id": scope, "instruction": next_instruction,
                       "step_budget": allowance, "status": "running"}
            record["cycles"].append(attempt)
            await persist()
            current = await bio_agent.run_autonomous_analysis(
                state, next_instruction, max_steps=allowance,
                run_kind="chat", run_scope_id=scope, resume_from_checkpoint=False,
                adata_source={
                    **(source_decision or {}),
                    "owner_id": record["id"],
                    "preparation_id": record.get("adata_source_preparation_id"),
                } if source_decision is not None else None,
            )
            prepared_id = (current.get("resolved_adata_source_context") or {}).get(
                "preparation_id"
            )
            if prepared_id:
                record["adata_source_preparation_id"] = prepared_id
            # Missing accounting cannot grant an unbounded correction budget.
            used = current.get("steps_executed", allowance)
            if not isinstance(used, int) or isinstance(used, bool) or used < 0:
                used = allowance
            remaining = max(0, remaining - used)
            attempt.update({
                "status": "executed", "steps_executed": used,
                "execution_ids": list(current.get("attempt_execution_ids") or []),
                "executor_status": current.get("status"),
                "findings": current.get("findings") or [],
            })
            attempt["analysis_step_ids"] = _persist_cycle_evidence(
                state, current, record["id"], scope,
            )
            merged = dict(current)
            for key in ("code_history", "analysis_log"):
                merged[key] = _merge_records(result.get(key), current.get(key))
            result = merged
            result["figures"] = collect_figures(scopes)
            result["figure_review_records"] = [
                figure.model_dump(mode="json", exclude={"png_base64"})
                for figure in state.publication_figures
                if figure.producer_run_scope_id in scopes
            ]
            result["score_definitions"] = [s.model_dump(mode="json") for s in state.score_definitions]
            result["scoring_plans"] = [s.model_dump(mode="json") for s in state.scoring_plans]
            record["supervision"] = reviews.supervision_records()
            record["supervision_observations"] = list(reviews.observations)
            record["supervisor_enabled"] = bool(reviews._provider.get_model("supervisor"))
            record["remaining_steps"] = remaining
            record["execution_incomplete"] = current.get("status") == "max_steps_reached"
            if record["execution_incomplete"]:
                record["unfinished_work"] = (
                    "The bioinformatics executor reached its step limit before "
                    "reporting completion. Continue in a new Chat request using "
                    "the saved execution receipts and available kernel results."
                )
            else:
                record.pop("unfinished_work", None)
            record["status"] = "reviewing"
            await persist()
            await reviews._send_message("status", "PI reviewing Chat execution evidence...", AgentType.PI)
            evidence = execution_context(result) + "\nChat review history:\n" + json.dumps(record)
            decision = await reviews.pi_agent.review_chat_execution(
                instruction, evidence, remaining, CHAT_CORRECTION_CYCLES - cycle,
            )
            attempt["review"] = decision
            attempt["status"] = "reviewed" if not decision.get("review_unavailable") else "review_unavailable"
            await reviews._send_message("status", {
                "chat_review_id": record["id"], "cycle": cycle,
                "review": decision,
            }, AgentType.PI)
            if decision.get("decision") != "revise":
                record["status"] = decision.get("decision", "unavailable")
                break
            if remaining == 0 or cycle == CHAT_CORRECTION_CYCLES:
                record["status"] = "budget_exhausted"
                record["unexecuted_revision"] = decision["revision_instruction"]
                break
            next_instruction = (
                f"Original Chat request and scope:\n{instruction}\n\n"
                f"PI-requested correction:\n{decision['revision_instruction']}\n\n"
                "The same kernel retains prior results. Do not repeat completed work. "
                "Keep negative results and errors visible; state which earlier claims "
                "are corrected. Use generate_publication_figure for final figures so "
                "the visual review and artifact record are retained.\n\n"
                "Prior executed evidence:\n" + execution_context(result)
            )
        result["chat_review"] = record
        result["steps_executed"] = CHAT_EXECUTION_STEPS - remaining
        return result
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise
    except Exception:
        record["status"] = "interrupted"
        raise
    finally:
        record["supervision"] = reviews.supervision_records()
        record["supervision_observations"] = list(reviews.observations)
        try:
            await persist()
        except Exception:
            if record["status"] not in {"cancelled", "interrupted"}:
                raise
            logger.warning("Could not persist interrupted Chat review", exc_info=True)
