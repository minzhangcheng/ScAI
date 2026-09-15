"""Supplementary task preparation, execution, and merge support."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from backend.services.cohort_identity import active_cohort_identity
from backend.services.sample_design import (
    infer_execution_modalities,
    samples_per_group_by_modality,
)
from backend.services.statistical_gates import run_gates as run_statistical_gates

from .shared import (
    AnalysisStep,
    CodeExecution,
    compact_traceback,
    execution_supports_evidence,
    truncate_for_llm,
)

_VALID_STATUSES = ("completed", "partial", "failed")

logger = logging.getLogger(__name__)


def _validated_supplementary_findings(
    findings: List[Any], executions: List[CodeExecution], state: Any,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Retain findings with objective execution links for later PI review.

    Execution IDs embedded in prose are not a reliable binding contract.  The
    host therefore supplies the bounded set of successful executions from this
    pass and leaves scientific sufficiency to the PI.
    """
    evidence_by_id = {
        str(record.id): record
        for record in executions
        if execution_supports_evidence(record)
    }
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    modality_names = {
        frozenset({"single_cell"}): "single_cell",
        frozenset({"bulk_rna_seq"}): "bulk_rna_seq",
        frozenset({"single_cell", "bulk_rna_seq"}): "cross_modal",
    }
    candidate_ids = sorted(evidence_by_id)[:40]
    modalities = frozenset(
        modality
        for execution in evidence_by_id.values()
        for modality in getattr(execution, "analysis_modalities", [])
    )
    modality = modality_names.get(modalities)
    plan = getattr(state, "current_plan", None)
    plan_id = str(getattr(plan, "id", "") or "") or None
    lineage_id = getattr(state, "restore_branch_id", None) or "main"
    cohort = active_cohort_identity(state)
    candidate_steps = sorted({
        int(step_id)
        for execution in evidence_by_id.values()
        for step_id in (getattr(execution, "plan_step_ids", None) or [])
    })
    for index, value in enumerate(findings or [], start=1):
        text = str(value or "").strip()
        if not text:
            continue
        record = {
            "id": str(uuid4()),
            "text": text,
            "modality": modality,
            "evidence_execution_ids": [],
            "candidate_evidence_execution_ids": candidate_ids,
            "claimed_evidence_execution_ids": [],
            "candidate_plan_step_ids": candidate_steps,
            "confirmed_plan_step_ids": [],
            "evidence_status": "unverified",
            "semantic_status": "unverified",
            "round_number": int(getattr(state, "current_round", 0) or 0),
            "lineage_id": lineage_id,
            "plan_id": plan_id,
            "analysis_generation": int(
                (getattr(state, "sandbox_state", None) or {}).get(
                    "analysis_generation", 0
                ) or 0
            ),
            "cohort_generation": int(
                cohort.get("cohort_generation", 0) or 0
            ),
            "cohort_digest": str(cohort.get("cohort_digest") or "") or None,
        }
        if modality is not None:
            report = run_statistical_gates(
                [record],
                score_definitions=list(
                    getattr(state, "score_definitions", None) or []
                ),
                scoring_plans=list(
                    getattr(state, "scoring_plans", None) or []
                ),
                samples_per_group_by_modality=(
                    samples_per_group_by_modality(state)
                ),
            )
        else:
            report = None
        if report is not None and report.findings:
            warnings.append(
                f"Supplementary finding {index} PI review observation: "
                + "; ".join(finding.detail for finding in report.findings)
            )
        if not candidate_ids:
            warnings.append(
                f"Supplementary finding {index} has no successful execution "
                "receipt from this pass."
            )
        elif modality is None:
            warnings.append(
                f"Supplementary finding {index} was retained for PI review because "
                "its execution modality was not established."
            )
        records.append(record)
    return records, warnings


async def _execute_supplementary_pass(
    agent,
    *,
    clean_tasks,
    context_summary,
    direction,
    metadata,
    registered_data_files,
    var_name,
    cluster_key,
    round_number,
    max_iterations,
    step_timeout,
    produced,
    analysis_log,
    warnings,
    envelope,
) -> Dict[str, Any]:
    transcript: List[str] = []
    final_result: Optional[Dict[str, Any]] = None

    abort_envelope: Optional[Dict[str, Any]] = None

    async def _one_step() -> str:
        """One executor iteration: generate, run, and record the result.

        Returns a loop signal — "continue", "done" or "abort" — so the
        caller can time-box a whole step without this body knowing that it
        is being timed.
        """
        nonlocal final_result, abort_envelope
        prompt = agent._build_supplementary_executor_prompt(
            clean_tasks, context_summary, direction, metadata,
            "\n\n".join(transcript[-3:]),
            registered_data_files=registered_data_files,
        )
        parsed = agent._safe_parse_json(
            await agent.chat(prompt, temperature=0.2),
            default={},
        )
        if not isinstance(parsed, dict):
            transcript.append("Executor returned unparseable response; reply with valid JSON.")
            return "continue"
        if parsed.get("done"):
            final_result = parsed.get("result") or {}
            return "done"

        code = parsed.get("code", "")
        if not code or not isinstance(code, str):
            transcript.append("No code provided; provide `code` or set done=true.")
            return "continue"

        unsafe = agent._supplementary_code_is_safe(code, var_name)
        state = agent._current_analysis_state
        if unsafe:
            transcript.append(
                f"[rejected] {unsafe}. Revise the code to avoid file writes and "
                f"unreviewed figures; live-object mutation or rebinding remains "
                f"allowed when the approved task requires it.")
            return "continue"

        started = time.perf_counter()
        exec_result = await agent.sandbox.execute(code)
        execution_time = time.perf_counter() - started

        # Cap at source so the record, the LLM transcript and the analysis
        # log all share one truncated string.
        output = truncate_for_llm(getattr(exec_result, "output", "") or "")
        error = getattr(exec_result, "error", None)
        if error:
            error = compact_traceback(error, max_chars=3000)
        exec_figures = list(getattr(exec_result, "figures", None) or [])

        cohort = active_cohort_identity(state)
        plan = getattr(state, "current_plan", None)
        record = CodeExecution(
            code=code,
            output=output,
            error=error,
            # Restricted repair cells do not own archival visuals. Keeping
            # base64 here would make the report analysis log treat an
            # unreviewed notebook render as a result figure.
            figures=[],
            execution_time=execution_time,
            round_number=round_number,
            analysis_modalities=infer_execution_modalities(code),
            evidence_schema_version=2,
            plan_id=str(getattr(plan, "id", "") or "") or None,
            plan_step_ids=[],
            execution_purpose="repair",
            plan_binding_status="unverified",
            plan_binding_reason=(
                "Supplementary repair execution is retained for PI review."
            ),
            execution_status="failed" if error else "succeeded",
            lineage_id=getattr(state, "restore_branch_id", None) or "main",
            analysis_generation=int(
                (getattr(state, "sandbox_state", None) or {}).get(
                    "analysis_generation", 0
                ) or 0
            ),
            cohort_generation=int(
                cohort.get("cohort_generation", 0) or 0
            ),
            cohort_digest=str(cohort.get("cohort_digest") or "") or None,
        )
        produced.append(record)
        agent.code_history.append(record)
        agent.last_result = exec_result
        analysis_log.append(AnalysisStep(
            step_number=len(analysis_log) + 1, step_type="execute_code",
            code=code, code_output=output[-3000:],
            code_error=error if error else None,
            code_description=f"Supplementary execution {len(produced)}",
            code_step_type="technical", round_number=round_number,
        ))
        if exec_figures and not any(
            "incidental inline figure" in warning for warning in warnings
        ):
            warnings.append(
                "Discarded an incidental inline figure from the restricted "
                "supplementary pass; publication visuals must use the main "
                "figure/review pipeline."
            )

        # The execution id is shown so the model can cite the exact cell a
        # finding came from, which is what makes the findings auditable.
        transcript.append(
            f"[execution_id {record.id}]\n[code]\n{code[:1500]}\n"
            f"[stdout]\n{output[:6000]}\n[error]\n{error or 'none'}")

        return "continue"

    for _ in range(max_iterations):
        try:
            signal = await asyncio.wait_for(_one_step(), timeout=step_timeout)
        except asyncio.TimeoutError:
            # The step, not the pass, ran out of time. Interrupt the cell it
            # left running, tell the executor what happened, and let it pick
            # a cheaper route — abandoning the whole pass here would throw
            # away the steps that already succeeded.
            logger.warning(
                f"[{getattr(agent, 'session_id', '?')}] supplementary step "
                f"exceeded its {step_timeout}s budget")
            agent._interrupt_kernel_after_timeout()
            warnings.append(
                f"A supplementary step exceeded its {step_timeout}s budget and "
                f"was interrupted.")
            transcript.append(
                "[timeout] The previous step exceeded its time budget and the "
                "kernel was interrupted. Do not retry that computation as-is: "
                "use a cheaper approach, reduce its scope, or set done=true and "
                "report what is already available.")
            continue
        if signal == "done":
            break
        if signal == "abort":
            return abort_envelope or envelope(
                "failed", "Supplementary pass aborted.", ["aborted"])

    if final_result is None:
        # Ending on the iteration limit does not mean nothing was computed.
        # The Host records the cells that actually ran; it does not keep a
        # signature gate deciding whether the live object changed.
        orphan_warnings = ["max_iterations reached"]
        if produced:
            orphan_warnings.append(
                "Produced by execution_id(s): "
                + ", ".join(str(rec.id) for rec in produced[-5:])
            )
        return envelope(
            "partial",
            "Supplementary pass reached its iteration limit before reporting a result.",
            orphan_warnings,
        )

    status = str(final_result.get("status", "") or "").strip().lower()
    if status not in _VALID_STATUSES:
        status = "completed"
    result = envelope(status, str(final_result.get("summary", "") or "").strip()
                       or "Supplementary analysis completed.", [])
    raw_findings = [
        str(f).strip() for f in (final_result.get("findings") or []) if str(f).strip()
    ]
    finding_records, finding_warnings = _validated_supplementary_findings(
        raw_findings, produced, agent._current_analysis_state,
    )
    result["finding_records"] = finding_records
    result["findings"] = [record["text"] for record in finding_records]
    result["warnings"] += finding_warnings
    result["warnings"] += [
        str(w).strip() for w in (final_result.get("warnings") or []) if str(w).strip()
    ]

    # Without a successful receipt, retain the prose in audit history without
    # letting it support the current report.
    if not any(execution_supports_evidence(r) for r in produced):
        if result["findings"]:
            result["warnings"].append(
                "Supplementary findings have no successful execution receipt and "
                "remain available only in audit history.")
        if result["status"] == "completed":
            result["status"] = "partial"
    return result



async def run_supplementary_tasks(
    agent,
    tasks: List[str],
    context_summary: str,
    direction: str,
    max_iterations: int = 5,
    timeout_seconds: int = 5520,
    total_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute PI-proposed supplementary tasks on the LIVE analysis object.

    Restricted to code execution: no tool loop, no literature, no PI
    consultation. Publication deliverables are routed back to the main executor,
    because only that path can register, review and contract-check them. Any
    incidental inline render is discarded rather than entering report state.
    Never raises — every failure path returns a status.

    Two independent budgets, because one shared ceiling cannot serve both:
    `timeout_seconds` bounds ONE step (its code generation, its execution and
    result capture), so a heavy analysis — GRN inference, deconvolution
    training — gets its full allowance instead of inheriting whatever the
    earlier steps left over; `total_timeout_seconds` (default: the sum of the
    per-step budgets) only backstops a pass that never converges. A step that
    overruns interrupts the kernel and is reported back to the executor, which
    then picks a cheaper route or stops — it does not abandon the pass.

    Returns the raw dict for `sanitize_review_result`, plus `code_history` (the
    CodeExecution records produced, for evidence provenance) and `analysis_log`.
    """
    state = getattr(agent, "_current_analysis_state", None)
    registered_data_files = (
        agent._render_registered_data_files(state)
        if state is not None and hasattr(agent, "_render_registered_data_files")
        else ""
    )
    max_iterations = max(1, min(int(max_iterations), 8))
    step_timeout = max(1, int(timeout_seconds))
    total_timeout = int(total_timeout_seconds or 0)
    if total_timeout <= 0:
        total_timeout = step_timeout * max_iterations
    # A total below one step's budget would re-create the problem this split
    # exists to fix: the pass ceiling cutting off a single legitimate analysis.
    total_timeout = max(total_timeout, step_timeout)
    clean_tasks = [str(t).strip()[:600] for t in (tasks or []) if str(t).strip()]

    # Accumulated in the enclosing scope so the timeout and exception handlers
    # can still hand back what ran. A supplementary execution IS evidence the
    # caller cites, unlike an annotation-validation cell, so dropping it on a
    # timeout would orphan cells that are already in the notebook.
    produced: List[CodeExecution] = []
    analysis_log: List[AnalysisStep] = []
    warnings: List[str] = []
    active_object_selection: Dict[str, Any] = {}
    image_versions_before = agent._supplementary_image_versions(
        getattr(agent.sandbox, "working_dir", None)
    )
    quarantined_images: Set[str] = set()

    def _envelope(status: str, summary: str, extra_warnings: List[str]) -> Dict[str, Any]:
        newly_quarantined = agent._quarantine_supplementary_images(
            state, image_versions_before
        ) - quarantined_images
        if newly_quarantined:
            quarantined_images.update(newly_quarantined)
            warnings.append(
                "Withheld unreviewed image file(s) written during restricted "
                "supplementary execution: " + ", ".join(sorted(newly_quarantined))
            )
        # These are bounded observations of the cells this pass actually
        # returned.  They deliberately carry no scientific interpretation;
        # the PI uses the output/error context to decide whether a requested
        # gap was addressed.  The loop is capped at eight iterations, which
        # is below the existing review-list bound.
        evidence_records = [
            {
                "execution_id": str(record.id)[:100],
                "status": "failed" if record.error else "completed",
                "stdout_excerpt": (record.output or "")[-2000:],
                "error": (
                    compact_traceback(record.error, max_chars=1000)
                    if record.error else ""
                ),
            }
            for record in produced
        ]
        return {
            "status": status,
            "summary": summary,
            "findings": [],
            "finding_records": [],
            "figures": [],
            "warnings": warnings + list(extra_warnings),
            "code_history": list(produced),
            "analysis_log": list(analysis_log),
            "evidence_records": evidence_records,
            "active_object_selection": dict(active_object_selection),
        }

    if not clean_tasks:
        return _envelope("completed", "No supplementary tasks were requested.", [])

    preferred_var_name = "adata"
    introspect = agent.introspect_validation_metadata
    try:
        parameters = inspect.signature(introspect).parameters.values()
        supports_preferred = (
            "preferred_var_name" in inspect.signature(introspect).parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        )
    except (TypeError, ValueError):
        supports_preferred = True
    metadata = await (
        introspect(
            None,
            preferred_var_name=preferred_var_name,
            require_preferred=False,
        )
        if supports_preferred else introspect(None)
    )
    var_name = metadata.get("var")
    cluster_key = metadata.get("cluster_key")
    active_object_selection.update({
        "name": str(var_name or ""),
        "selection_source": str(metadata.get("selection_source") or "unresolved"),
        "canonical_verified": False,
        "candidate_names": [
            str(name)[:100] for name in (metadata.get("candidate_names") or [])[:20]
        ],
        "selection_error": str(metadata.get("selection_error") or "")[:500],
    })
    if not var_name:
        reason = active_object_selection["selection_error"] or "no active analysis object"
        return _envelope(
            "failed",
            "No unambiguous active analysis object is available; supplementary tasks cannot run.",
            [reason],
        )
    # Foundation/signature verification is retired.  A non-`current` plan
    # source is resolved and bound by the checkpoint selection service before
    # this pass; the Host never vetoes scientific mutation here.
    round_number = int(getattr(agent._current_analysis_state, "current_round", 0) or 0)

    try:
        return await asyncio.wait_for(
            _execute_supplementary_pass(
                agent,
                clean_tasks=clean_tasks,
                context_summary=context_summary,
                direction=direction,
                metadata=metadata,
                registered_data_files=registered_data_files,
                var_name=var_name,
                cluster_key=cluster_key,
                round_number=round_number,
                max_iterations=max_iterations,
                step_timeout=step_timeout,
                produced=produced,
                analysis_log=analysis_log,
                warnings=warnings,
                envelope=_envelope,
            ),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[{getattr(agent, 'session_id', '?')}] supplementary pass exceeded its "
            f"total {total_timeout}s budget")
        # Cancelling the coroutine only stops US waiting: a cell already
        # dispatched keeps running in the kernel and would hold it against the
        # rest of the workflow. The sandbox interrupts on its OWN timeout, not
        # on ours, so this ceiling has to do it itself.
        agent._interrupt_kernel_after_timeout()
        return _envelope("partial", "Supplementary executor timed out.", ["timeout"])
    except Exception as e:
        compact_error = compact_traceback(str(e), max_chars=3000)
        logger.error(
            "[%s] supplementary executor failed: %s",
            getattr(agent, "session_id", "?"),
            compact_error,
        )
        return _envelope(
            "failed",
            f"Supplementary executor error: {compact_error}",
            [compact_error],
        )
