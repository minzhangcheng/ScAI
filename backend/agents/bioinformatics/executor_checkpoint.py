"""Executor checkpoint validation and payload construction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Optional


#: One central recognizer for every legacy or scoped bioinformatics executor
#: transcript.  Export/import/lifecycle code must call this instead of
#: guessing `_bioinfo_checkpoint.json` or broad `*_checkpoint.json` patterns.
_EXECUTOR_CHECKPOINT_SUFFIX_RE = re.compile(
    r"(?:"
    r"bioinfo"
    r"|bioinfo_(?:ordinary|supplementary|figure_followup|chat)_[0-9a-f]{24}"
    r")_checkpoint\.json(?:\.quarantine(?:\.\d+)?)?$"
)


def is_executor_checkpoint_name(
    name: str,
    *,
    session_id: Optional[str] = None,
) -> bool:
    """True for a legacy or scoped executor transcript filename."""
    text = str(name or "")
    if session_id is not None:
        prefix = f"{session_id}_"
        if not text.startswith(prefix):
            return False
        text = text[len(prefix):]
    else:
        marker = "_bioinfo"
        # Session IDs may themselves contain ``_bioinfo``.  The executor
        # namespace is the final such segment immediately before the exact
        # recognized suffix, so split from the right.
        split_at = text.rfind(marker)
        if split_at <= 0:
            return False
        text = text[split_at + 1:]
    return bool(_EXECUTOR_CHECKPOINT_SUFFIX_RE.fullmatch(text))


def executor_checkpoint_prefix(
    *,
    run_kind: str,
    scope_digest: str,
    figure_followup: bool = False,
) -> str:
    """Build the prefix consumed by ``CheckpointManager``.

    ``CheckpointManager`` writes ``<session>_<prefix>_checkpoint.json``; the
    resulting filename is the accepted namespaced contract.
    """
    kind = "figure_followup" if figure_followup else str(run_kind or "ordinary")
    digest = re.sub(r"[^0-9a-f]", "", str(scope_digest or "").lower())[:24]
    if len(digest) < 24:
        digest = digest.ljust(24, "0")
    return f"bioinfo_{kind}_{digest}"


def legacy_checkpoint_prefix() -> str:
    return "bioinfo"


@dataclass(frozen=True)
class ExecutorCheckpointIdentity:
    """Complete Host-built owner identity for one executor transcript."""

    run_kind: str
    round_number: int
    planning_stage: str
    current_lineage_id: str
    origin_lineage_id: str
    reconcile_from_lineage_id: str
    active_plan_id: str
    revision_of_plan_id: str
    run_scope_id: str
    settlement_scope_id: str
    source_scope_id: str
    preparation_id: str
    selection_signature: str
    application_token_digest: str
    supplementary_request_id: str
    figure_followup: bool = False

    def fields(self) -> dict[str, Any]:
        return {
            "identity_version": 1,
            "run_kind": self.run_kind,
            "round_number": int(self.round_number),
            "planning_stage": self.planning_stage,
            "current_lineage_id": self.current_lineage_id,
            "origin_lineage_id": self.origin_lineage_id,
            "reconcile_from_lineage_id": self.reconcile_from_lineage_id,
            "active_plan_id": self.active_plan_id,
            "revision_of_plan_id": self.revision_of_plan_id,
            "run_scope_id": self.run_scope_id,
            "settlement_scope_id": self.settlement_scope_id,
            "source_scope_id": self.source_scope_id,
            "preparation_id": self.preparation_id,
            "selection_signature": self.selection_signature,
            "application_token_digest": self.application_token_digest,
            "supplementary_request_id": self.supplementary_request_id,
            "figure_followup": bool(self.figure_followup),
        }

    @property
    def scope_digest(self) -> str:
        # The filename selects the stable Host-owned run.  Volatile plan/source
        # facts remain in the payload and are validated after source
        # preparation; including them here would make a direct same-source plan
        # revision unable to find the transcript it is explicitly allowed to
        # inherit.
        payload = json.dumps(
            {
                "identity_version": 1,
                "run_kind": self.run_kind,
                "run_scope_id": self.run_scope_id,
                "supplementary_request_id": self.supplementary_request_id,
                "figure_followup": bool(self.figure_followup),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def checkpoint_fields(self) -> dict[str, Any]:
        return {"_scope_digest": self.scope_digest, **self.fields()}


def build_executor_checkpoint_identity(
    state: Any,
    *,
    run_kind: str,
    planning_stage: str,
    run_scope_id: str,
    settlement_scope_id: str,
    source_selection: Any = None,
    preparation: Any = None,
    supplementary_request_id: Optional[str] = None,
    figure_followup: bool = False,
) -> ExecutorCheckpointIdentity:
    """Build the sole checkpoint identity from durable Host-owned facts."""
    plan = None if run_kind in {"supplementary", "chat"} else getattr(state, "current_plan", None)
    if hasattr(source_selection, "model_dump"):
        source_payload = source_selection.model_dump(mode="json")
    elif isinstance(source_selection, dict):
        source_payload = dict(source_selection)
    else:
        source_payload = {"mode": "current"}
    selection_digest = str(
        getattr(preparation, "selection_signature", "") or ""
    ) or hashlib.sha256(
        json.dumps(
            source_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    token = str(getattr(preparation, "application_token", "") or "")
    current_lineage = str(getattr(state, "restore_branch_id", None) or "main")
    origin_lineage = str(
        getattr(preparation, "origin_session_lineage_id", None)
        or getattr(state, "restore_source_lineage_id", None)
        or current_lineage
    )
    reconcile_from = str(
        getattr(preparation, "reconcile_from_session_lineage_id", None) or ""
    )
    return ExecutorCheckpointIdentity(
        run_kind=str(run_kind or "ordinary"),
        round_number=int(getattr(state, "current_round", 0) or 0),
        planning_stage=str(planning_stage or "full"),
        current_lineage_id=current_lineage,
        origin_lineage_id=origin_lineage,
        reconcile_from_lineage_id=reconcile_from,
        active_plan_id=str(getattr(plan, "id", "") or ""),
        revision_of_plan_id=str(getattr(plan, "revision_of", "") or ""),
        run_scope_id=str(run_scope_id or ""),
        settlement_scope_id=str(settlement_scope_id or ""),
        source_scope_id=str(getattr(preparation, "scope_id", "") or ""),
        preparation_id=str(getattr(preparation, "preparation_id", "") or ""),
        selection_signature=selection_digest,
        application_token_digest=(
            hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
        ),
        supplementary_request_id=str(supplementary_request_id or ""),
        figure_followup=bool(figure_followup),
    )


def find_executor_checkpoint_paths(
    states_dir: Any,
    session_id: Optional[str] = None,
) -> list[Any]:
    """Return recognized executor transcript files under *states_dir*."""
    from pathlib import Path as _Path

    root = _Path(states_dir)
    if not root.is_dir():
        return []
    result = []
    for entry in sorted(root.iterdir()):
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        name = entry.name
        if ".quarantine" in name:
            continue
        if is_executor_checkpoint_name(name, session_id=session_id):
            result.append(entry)
    return result


def latest_executor_checkpoint_path(
    states_dir: Any,
    session_id: Optional[str] = None,
) -> Optional[Any]:
    """Return the newest recognized executor transcript, if any."""
    files = find_executor_checkpoint_paths(states_dir, session_id=session_id)
    if not files:
        return None
    try:
        return max(files, key=lambda path: path.stat().st_mtime_ns)
    except OSError:
        return files[-1]


def rebind_executor_checkpoint_name(
    name: str,
    source_session_id: str,
    destination_session_id: str,
) -> str:
    """Keep the scoped suffix and swap the session-id prefix on import/restore."""
    text = str(name)
    prefix = f"{source_session_id}_bioinfo"
    if text.startswith(prefix):
        return f"{destination_session_id}{text[len(source_session_id):]}"
    return text


@dataclass
class PendingToolCheckpoint:
    """A handle to checkpoint the executor at the tool call it is inside.

    The executor's checkpoint closure is local to its run loop, and the cell
    annotation decision is taken by the *workflow*, several frames above it,
    inside a callback the tool handler awaits. Without a handle the workflow
    would have to reconstruct the executor's position from outside, which is
    the mismatch the save-chain repair's section 3.11 records: two records of
    the same run that nothing binds together.

    The synthetic tool result is not decoration. ``tool_calls_are_paired``
    rejects a checkpoint whose last assistant message asks for a tool call that
    has no result, and ``restore_executor_state`` then throws the transcript
    away and starts fresh -- so a restored session would lose the executor's
    whole account of the round it is resuming into.
    """

    tool_call_id: str
    tool_name: str
    save: Callable[..., Awaitable[None]]

    async def checkpoint_now(self, step_name: str, *, result_text: str) -> None:
        await self.save(step_name, extra_messages=[{
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": result_text,
        }])


def validate_checkpoint(
    checkpoint: dict,
    *,
    current_round: int,
    planning_stage: str,
    committed_revision: int | None = None,
    exact_revision: bool = False,
    expected_identity: dict | None = None,
) -> list[str]:
    errors: list[str] = []
    recorded_revision = checkpoint.get("runtime_revision")
    behind_or_mismatched = (
        committed_revision is not None
        and recorded_revision is not None
        and (
            int(recorded_revision) < int(committed_revision)
            or (
                # The namespace was just moved back onto this generation
                # (B2-2), so "ahead" no longer means a live kernel that has
                # carried on -- it means a position from work the restore
                # discarded, and resuming there would skip the steps that
                # rebuilt it.
                exact_revision
                and int(recorded_revision) != int(committed_revision)
            )
        )
    )
    if behind_or_mismatched:
        # Behind the committed generation means the namespace moved past this
        # position: something restored or replaced the runtime after the
        # checkpoint was written, and resuming would replay against variables
        # that have since been rebuilt.
        #
        # Ahead is not an error and must not become one. The executor
        # checkpoints every five steps while the dump waits out the save floor,
        # so a live kernel is normally ahead of its newest generation -- that is
        # what §3.1 describes, and refusing it would break every ordinary
        # resume. Absent is not an error either: every checkpoint written before
        # this field existed has none, and absence is not evidence of mismatch.
        errors.append(
            f"runtime_revision does not belong to the committed generation: "
            f"checkpoint={recorded_revision}, committed={committed_revision}"
        )
    step_number = checkpoint.get("step_number")
    if not isinstance(step_number, int) or step_number < 0:
        errors.append(f"step_number invalid: {step_number}")
    messages = checkpoint.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages invalid: expected list with >= 2 items")
    if not isinstance(checkpoint.get("findings"), list):
        errors.append("findings invalid: expected list")
    if not isinstance(checkpoint.get("figures"), list):
        errors.append("figures invalid: expected list")
    checkpoint_round = checkpoint.get("round_number")
    if checkpoint_round is not None and checkpoint_round != current_round:
        errors.append(
            f"round_number mismatch: checkpoint={checkpoint_round}, "
            f"current={current_round}"
        )
    checkpoint_stage = checkpoint.get("planning_stage")
    if checkpoint_stage is not None and checkpoint_stage != planning_stage:
        errors.append(
            f"planning_stage mismatch: checkpoint={checkpoint_stage}, "
            f"current={planning_stage}"
        )
    if expected_identity:
        inherited_direct_revision = bool(
            expected_identity.get("active_plan_id")
            and expected_identity.get("revision_of_plan_id")
            and str(checkpoint.get("active_plan_id") or "")
            == str(expected_identity.get("revision_of_plan_id") or "")
        )
        for key, expected in expected_identity.items():
            if expected in (None, ""):
                continue
            if inherited_direct_revision and key in {
                "active_plan_id", "revision_of_plan_id",
            }:
                continue
            recorded = checkpoint.get(key)
            if str("" if recorded is None else recorded) != str(expected):
                errors.append(
                    f"{key} mismatch: checkpoint={recorded!r}, "
                    f"current={expected!r}"
                )
    return errors


def build_checkpoint_payload(
    *,
    step_name: str,
    state: Any,
    planning_stage: str,
    step_number: int,
    total_steps: int,
    attempt_start_step: int,
    active_plan_id: str | None,
    messages: list,
    findings: list,
    figures: list,
    code_history: list,
    analysis_log: list,
    analysis_context: dict,
    budget_exhausted: bool,
    runtime_revision: int | None = None,
    checkpoint_scope_digest: str | None = None,
    checkpoint_run_kind: str | None = None,
    checkpoint_run_scope_id: str | None = None,
    checkpoint_settlement_scope_id: str | None = None,
    checkpoint_source_scope_id: str | None = None,
    checkpoint_preparation_id: str | None = None,
    checkpoint_selection_signature: str | None = None,
    checkpoint_token_digest: str | None = None,
    checkpoint_request_id: str | None = None,
    checkpoint_figure_followup: bool = False,
    attempt_execution_ids: list | None = None,
    pending_dispatch: dict | None = None,
    checkpoint_identity: dict | None = None,
    annotation_retry_required: bool = False,
    annotation_retry_attempts: int = 0,
    annotation_failure_reason: str = "",
) -> dict:
    payload = {
        # Which runtime generation this position belongs to. Spec §3.11: the
        # two fields below cannot see a namespace, so without this a checkpoint
        # written against one kernel validated cleanly against another.
        "runtime_revision": runtime_revision,
        "_scope_digest": checkpoint_scope_digest,
        "run_kind": checkpoint_run_kind,
        "run_scope_id": checkpoint_run_scope_id,
        "settlement_scope_id": checkpoint_settlement_scope_id,
        "source_scope_id": checkpoint_source_scope_id,
        "preparation_id": checkpoint_preparation_id,
        "selection_signature": checkpoint_selection_signature,
        "application_token_digest": checkpoint_token_digest,
        "supplementary_request_id": checkpoint_request_id,
        "figure_followup": bool(checkpoint_figure_followup),
        "attempt_execution_ids": sorted({
            str(value) for value in (attempt_execution_ids or []) if str(value)
        }),
        "pending_dispatch": dict(pending_dispatch) if pending_dispatch else None,
        "step_name": step_name,
        "session_id": state.id,
        "round_number": state.current_round,
        "planning_stage": planning_stage,
        "step_number": step_number,
        "total_steps": total_steps,
        "attempt_start_step": attempt_start_step,
        "active_plan_id": active_plan_id,
        "messages": messages,
        "findings": findings,
        "figures": figures,
        "code_history": code_history,
        "analysis_log": analysis_log,
        "analysis_context": analysis_context,
        "annotation_retry_required": annotation_retry_required,
        "annotation_retry_attempts": annotation_retry_attempts,
        "annotation_failure_reason": annotation_failure_reason,
        "budget_exhausted": budget_exhausted,
        "completed": False,
    }
    if checkpoint_identity:
        payload.update(dict(checkpoint_identity))
    return payload



async def load_checkpoint(
    agent,
    checkpoint_manager,
    state,
    *,
    resume_from_checkpoint: bool,
    planning_stage: str,
    bindings,
    expected_identity: dict | None = None,
    legacy_checkpoint_manager: Any = None,
    allow_legacy_current: bool = False,
) -> tuple[dict | None, bool]:
    logger = getattr(bindings, "logger")
    # Try to resume from checkpoint if enabled
    checkpoint = None
    checkpoint_valid = False
    if resume_from_checkpoint:
        try:
            checkpoint = await checkpoint_manager.load()
            loaded_manager = checkpoint_manager
            if checkpoint is None and legacy_checkpoint_manager is not None:
                checkpoint = await legacy_checkpoint_manager.load()
                loaded_manager = legacy_checkpoint_manager
            if checkpoint and not checkpoint.get('completed', False):
                legacy_payload = not checkpoint.get("_scope_digest")
                validation_errors = validate_checkpoint(
                    checkpoint,
                    current_round=state.current_round,
                    planning_stage=planning_stage,
                    committed_revision=(
                        getattr(state, "runtime_persistence", None) or {}
                    ).get("committed_revision"),
                    # The notebook projection carries the restored generation
                    # marker across process restarts and onto the paired
                    # AnalysisState selected by the run route.
                    exact_revision=bool(
                        (getattr(state, "runtime_persistence", None) or {})
                        .get("pair_restore")
                    ),
                    expected_identity=(
                        None
                        if legacy_payload and allow_legacy_current
                        else expected_identity
                    ),
                )

                if not validation_errors:
                    checkpoint_valid = True
                    logger.info(f"[Bioinformatics] Resuming from checkpoint at step {checkpoint['step_number']}")
                    await agent._notify("status", f"Resuming analysis from step {checkpoint['step_number']}...")
                else:
                    logger.warning(f"[Bioinformatics] Checkpoint validation failed: {validation_errors}, starting fresh")
                    checkpoint = None
        except Exception as e:
            logger.exception("[Bioinformatics] Failed to load checkpoint")
            checkpoint = None
            await agent._notify("status", "Previous checkpoint corrupted, starting fresh analysis...")


    return checkpoint, checkpoint_valid
