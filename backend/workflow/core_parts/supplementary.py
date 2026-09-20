"""SingleCellAnalysisWorkflow supplementary method owners."""
from __future__ import annotations

import asyncio
from backend.constants import LlmFailureAction
from backend.agents.literature.research import (
    literature_provenance_disclosure,
    literature_technical_failure,
)
from backend.services.adata_checkpoints.selection import selection_signature
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow.helpers import WorkflowSupersededError, WorkflowTerminatedError

import uuid
from datetime import datetime


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def signal_supplementary_analysis(
    self,
    request_text: str,
    target: str,
    request_id: str | None = None,
):
    """Queue a supplementary analysis request at the current HITL point.

    ``request_id`` is the durable request identity allocated by the route
    before this signal.  Legacy callers may omit it and keep the old
    text-as-queue-key behaviour.
    """
    queue_key = str(request_id or request_text)
    if not any(key == queue_key for key, _queued_target in self._supplementary_queue):
        self._supplementary_queue.append((queue_key, target))
    if target == "annotation":
        self._annotation_event.set()
    elif target in ("direction", "round"):
        self._continue_event.set()


async def _persist_request_map(state, request_map) -> None:
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    sandbox_state["supplementary_requests_v1"] = dict(request_map)
    state.sandbox_state = sandbox_state
    await state.increment_version()
    from backend.services.session_store import async_save_session

    await async_save_session(str(state.id), state, force=True)


async def _mark_request_interrupted(
    workflow,
    state,
    request_id: str,
    message: str,
) -> None:
    """Persist cancellation without overwriting newer request/source fields."""
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    request_map = dict(sandbox_state.get("supplementary_requests_v1") or {})
    record = dict(request_map.get(str(request_id)) or {})
    if not record:
        return
    sandbox = getattr(getattr(workflow, "bioinformatics_agent", None), "sandbox", None)
    checkpoint_name = str(
        getattr(sandbox, "active_executor_checkpoint_name", "") or ""
    )
    if checkpoint_name:
        from backend.agents.bioinformatics.executor_checkpoint import (
            is_executor_checkpoint_name,
        )

        if is_executor_checkpoint_name(
            checkpoint_name, session_id=str(state.id),
        ):
            record["executor_checkpoint_name"] = checkpoint_name
    record["status"] = "interrupted"
    record["error_summary"] = str(message)[:500]
    record["updated_at"] = datetime.now().isoformat()
    request_map[str(request_id)] = record
    await asyncio.shield(_persist_request_map(state, request_map))


def _compact_terminal_requests(state, request_map) -> bool:
    """Bound full terminal request payloads without deleting correlation IDs."""
    terminal = [
        (request_id, record)
        for request_id, record in request_map.items()
        if isinstance(record, dict)
        and record.get("status") in {"settled", "failed"}
        and not record.get("compacted")
    ]
    terminal.sort(
        key=lambda item: (
            str((item[1] or {}).get("updated_at") or ""),
            str(item[0]),
        ),
        reverse=True,
    )
    changed = False
    preparation_by_id = {
        str(getattr(item, "preparation_id", "") or ""): item
        for item in (getattr(state, "adata_source_preparations", None) or [])
    }
    for request_id, record in terminal[100:]:
        checkpoint_name = str(record.get("executor_checkpoint_name") or "")
        if checkpoint_name:
            from backend.agents.bioinformatics.executor_checkpoint import (
                is_executor_checkpoint_name,
            )
            from backend.config import get_session_states

            if (
                is_executor_checkpoint_name(
                    checkpoint_name, session_id=str(state.id),
                )
                and (get_session_states(str(state.id)) / checkpoint_name).is_file()
            ):
                continue
        preparation = preparation_by_id.get(
            str(record.get("adata_source_preparation_id") or "")
        )
        if preparation is not None and str(getattr(preparation, "status", "")) in {
            "pending", "applied_warm", "pending_reconcile"
        }:
            continue
        source = record.get("adata_source") or {}
        request_map[request_id] = {
            "request_id": str(record.get("request_id") or request_id),
            "request_scope_id": str(record.get("request_scope_id") or ""),
            "target": str(record.get("target") or ""),
            "round_number": int(record.get("round_number") or 0),
            "status": str(record.get("status") or "failed"),
            "adata_source_mode": str(
                (source.get("mode") if isinstance(source, dict) else None)
                or "current"
            ),
            "source_selection_signature": record.get(
                "source_selection_signature"
            ),
            "adata_source_preparation_id": record.get(
                "adata_source_preparation_id"
            ),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "compacted": True,
        }
        changed = True
    return changed

def _merge_supplementary_findings(self, agent_key: str, new_findings: List[str]):
    """Merge supplementary findings into the current graph state for an agent.

    `_initialize_round` seeds every agent-result channel with an explicit
    `None`, and `_current_graph_state` is the bioinformatics node's INPUT dict
    (the round's real result is returned in a new dict), so this key normally
    holds `None` rather than being absent — which means `.get(key, {})` returns
    `None` and the default never applies. The old form guarded only the read of
    `findings` and then assigned into that `None`, raising
    `TypeError: 'NoneType' object does not support item assignment` for every
    supplementary target — annotation, direction and round alike — with the
    caller's try/except turning it into a generic "supplementary analysis
    failed" after the analysis had already been paid for.
    """
    if not self._current_graph_state or not new_findings:
        return
    existing = self._current_graph_state.get(agent_key)
    if not isinstance(existing, dict):
        existing = {}
    existing_findings = existing.get("findings") or []
    existing["findings"] = list(existing_findings) + [
        f"[Supplementary] {f}" for f in new_findings
    ]
    self._current_graph_state[agent_key] = existing

def _absorb_supplementary_executions(self, state: AnalysisState, supp_result: Dict[str, Any]):
    """Record a user-requested supplementary pass's executions on the session.

    Without this the pass runs, spends kernel time, and leaves nothing citable:
    its `code_history` and `analysis_log` were dropped, so no finding it
    produced could be traced to an execution — unlike the round path and unlike
    the automatic review pass, which both merge them. Dedup is by execution
    identity, matching those two.
    """
    try:
        existing_code_ids = {c.id for c in state.code_history}
        for execution in supp_result.get("code_history") or []:
            if getattr(execution, "id", None) is None or execution.id in existing_code_ids:
                continue
            state.code_history.append(execution)
            existing_code_ids.add(execution.id)

        existing_step_ids = {s.id for s in state.analysis_log}
        for step in supp_result.get("analysis_log") or []:
            if isinstance(step, dict):
                step = AnalysisStep(**step)
            if step.id in existing_step_ids:
                continue
            step.round_number = max(int(getattr(state, "current_round", 1) or 1), 1)
            state.analysis_log.append(step)
            existing_step_ids.add(step.id)
    except Exception as e:
        logger.warning(
            f"[{self.session_id}] could not absorb supplementary executions: {e}"
        )

async def _run_supplementary_analysis(
    self,
    state: AnalysisState,
    request_text: str,
    context: str,
    request_id: str | None = None,
    request_scope_id: str | None = None,
):
    """Execute user-requested supplementary analysis at a HITL point."""
    # Resolve a durable request record when the queue entry carries an id.
    _request_map = dict(
        (getattr(state, "sandbox_state", None) or {}).get(
            "supplementary_requests_v1"
        )
        or {}
    )
    _durable = _request_map.get(str(request_text)) if request_text else None
    if isinstance(_durable, dict):
        request_id = str(_durable.get("request_id") or request_text)
        request_scope_id = str(
            _durable.get("request_scope_id") or request_scope_id or ""
        )
        request_text = str(_durable.get("request_text") or request_text)
        _durable["status"] = "planning"
        _durable["updated_at"] = datetime.now().isoformat()
        _request_map[request_id] = _durable
    request_id = str(request_id or f"supp_{uuid.uuid4().hex}")
    request_scope_id = str(request_scope_id or f"supp_scope_{uuid.uuid4().hex}")
    if not isinstance(_durable, dict):
        now = datetime.now().isoformat()
        _durable = {
            "request_id": request_id,
            "request_text": str(request_text or "")[:2000],
            "target": "",
            "round_number": int(getattr(state, "current_round", 0) or 0),
            "request_scope_id": request_scope_id,
            "status": "planning",
            "adata_source": None,
            "source_selection_signature": None,
            "adata_source_preparation_id": None,
            "created_at": now,
            "updated_at": now,
            "error_summary": None,
        }
        _request_map[request_id] = _durable
    await _persist_request_map(state, _request_map)
    await self._send_message(
        "status",
        f"PI is planning supplementary analysis: {request_text[:200]}...",
        AgentType.PI,
    )

    _skip_llm_call = object()
    _intervention_node = (
        "bioinformatics_analysis" if context == "annotation" else "wait_for_user_input"
    )

    async def _run_llm_call(call, stage_label: str):
        """Retry only one failed supplementary PI/literature call."""
        while True:
            try:
                return await call()
            except WorkflowSupersededError:
                raise
            except asyncio.CancelledError:
                await _mark_request_interrupted(
                    self, state, request_id,
                    f"Supplementary {stage_label} was interrupted.",
                )
                raise
            except WorkflowTerminatedError:
                await _mark_request_interrupted(
                    self, state, request_id,
                    f"Supplementary {stage_label} was terminated.",
                )
                raise
            except UpstreamExhaustedError as error:
                try:
                    action = await self._wait_for_llm_intervention(
                        error, _intervention_node, state,
                    )
                except WorkflowSupersededError:
                    raise
                except asyncio.CancelledError:
                    await _mark_request_interrupted(
                        self, state, request_id,
                        f"Supplementary {stage_label} intervention was interrupted.",
                    )
                    raise
                if action == LlmFailureAction.RETRY:
                    continue
                if action == LlmFailureAction.TERMINATE:
                    await _mark_request_interrupted(
                        self, state, request_id,
                        f"Supplementary {stage_label} was terminated after LLM failure.",
                    )
                    raise WorkflowTerminatedError(
                        f"User terminated supplementary {stage_label} after LLM failure"
                    ) from error
                return _skip_llm_call

    # PI agent plans supplementary tasks
    supp_plan = {}
    _supp_technical_failure = False
    _supp_error_summary = None
    try:
        planned = await _run_llm_call(
            lambda: self.pi_agent.plan_supplementary_analysis(
                state, request_text, context
            ),
            "planning",
        )
        if planned is _skip_llm_call:
            _supp_technical_failure = True
            _supp_error_summary = "PI supplementary planning was skipped after LLM failure"
        else:
            supp_plan = planned or {}
    except asyncio.CancelledError:
        await _mark_request_interrupted(
            self, state, request_id, "Supplementary planning was interrupted."
        )
        raise
    except WorkflowSupersededError:
        raise
    except WorkflowTerminatedError:
        await _mark_request_interrupted(
            self, state, request_id, "Supplementary planning was terminated."
        )
        raise
    except Exception as exc:
        _supp_technical_failure = True
        _supp_error_summary = f"PI supplementary planning failed: {type(exc).__name__}"
        logger.exception("[%s] Supplementary PI planning failed", self.session_id)
        await self._send_message(
            "error", _supp_error_summary, AgentType.PI
        )

    # B/C: the durable request owns its source.  Omitted selection may inherit
    # the current plan's preparation only when the normalized selection is
    # identical; it must never be an implicit executor fallback.
    from ...models.schemas import AdataSourceSelection

    _current_source = getattr(getattr(state, "current_plan", None), "adata_source", None)
    _supp_source = supp_plan.get("adata_source")
    if _supp_source is None:
        _supp_source = (
            _current_source.model_copy(deep=True)
            if _current_source is not None
            else AdataSourceSelection()
        )
    elif isinstance(_supp_source, dict):
        _supp_source = AdataSourceSelection.model_validate(_supp_source)
    _supp_signature = selection_signature(_supp_source)
    _inherited_prep_id = None
    if (
        _current_source is not None
        and _current_source.model_dump(mode="json")
        == _supp_source.model_dump(mode="json")
    ):
        _inherited_prep_id = getattr(
            getattr(state, "current_plan", None),
            "adata_source_preparation_id",
            None,
        )
    _durable = dict(_request_map.get(request_id) or {})
    _durable.update({
        "adata_source": _supp_source.model_dump(mode="json"),
        "source_selection_signature": _supp_signature,
        "adata_source_preparation_id": _inherited_prep_id,
        "status": "running" if not _supp_technical_failure else "planning",
        "updated_at": datetime.now().isoformat(),
    })
    _request_map[request_id] = _durable
    await _persist_request_map(state, _request_map)

    # Execute bioinformatics tasks if planned
    bio_tasks = supp_plan.get("bioinformatics_tasks", [])
    supp_result = None
    if bio_tasks and self.bioinformatics_agent:
        await self._send_message("status", "Running supplementary bioinformatics analysis...", AgentType.BIOINFORMATICS)
        _sandbox = getattr(self.bioinformatics_agent, "sandbox", None)
        _had_checkpoint_owner = hasattr(
            _sandbox, "active_executor_checkpoint_name",
        )
        _parent_checkpoint_name = getattr(
            _sandbox, "active_executor_checkpoint_name", None,
        )
        try:
            combined_instruction = "Supplementary analysis tasks:\n" + "\n".join(f"- {t}" for t in bio_tasks)
            supp_result = await self.bioinformatics_agent.run_autonomous_analysis(
                state,
                combined_instruction,
                max_steps=30,
                figure_review_callback=self._figure_review_callback,
                adata_source=_supp_source,
                run_scope_id=request_scope_id,
                settlement_scope_id=request_scope_id,
                run_kind="supplementary",
                supplementary_request_id=request_id,
            )
            if supp_result:
                self._merge_supplementary_findings(
                    "bioinformatics_result", supp_result.get("findings", [])
                )
                self._absorb_supplementary_executions(state, supp_result)
        except asyncio.CancelledError:
            await _mark_request_interrupted(
                self, state, request_id,
                "Supplementary execution was interrupted.",
            )
            raise
        except WorkflowSupersededError:
            raise
        except WorkflowTerminatedError:
            await _mark_request_interrupted(
                self, state, request_id,
                "Supplementary bioinformatics was terminated.",
            )
            raise
        except Exception as e:
            _supp_technical_failure = True
            _supp_error_summary = (
                f"Supplementary bioinformatics failed: {type(e).__name__}"
            )
            logger.error(f"[{self.session_id}] Supplementary bioinformatics failed: {e}")
            await self._send_message("error", f"Supplementary bioinformatics analysis failed: {e}", AgentType.BIOINFORMATICS)
        finally:
            if _had_checkpoint_owner:
                _sandbox.active_executor_checkpoint_name = _parent_checkpoint_name
            else:
                try:
                    delattr(_sandbox, "active_executor_checkpoint_name")
                except AttributeError:
                    pass

    # Execute literature tasks if planned
    lit_tasks = supp_plan.get("literature_topics", [])
    literature_failures = []
    if lit_tasks and self.literature_agent:
        await self._send_message("status", "Running supplementary literature research...", AgentType.LITERATURE)
        try:
            for topic in lit_tasks[:3]:
                async def _research_topic():
                    return await self.literature_agent.process(
                        state,
                        topic,
                        topic=topic,
                        use_deep_research=True,
                        pi_plan_fn=self.pi_agent.plan_literature_search,
                        pi_review_fn=self.pi_agent.review_literature_search,
                        research_context=(
                            f"Supplementary request: {request_text}\n"
                            f"Decision context: {context}\n"
                            f"Request checkpoint: {request_scope_id}"
                        )[:12000],
                    )

                lit_result = await _run_llm_call(
                    _research_topic, "literature research",
                )
                if lit_result is _skip_llm_call:
                    _supp_technical_failure = True
                    _supp_error_summary = (
                        "Supplementary literature was skipped after LLM failure"
                    )
                    break
                if lit_result:
                    technical_failure = literature_technical_failure(lit_result)
                    if technical_failure:
                        literature_failures.append({"topic": topic, **technical_failure})
                        _supp_technical_failure = True
                        _supp_error_summary = (
                            "Supplementary literature returned a technical failure: "
                            + literature_provenance_disclosure(lit_result)
                        )[:500]
                    report_text = lit_result.get("report", "") or lit_result.get("synthesis", "")
                    if report_text:
                        self._merge_supplementary_findings(
                            "literature_result", [report_text[:500]]
                        )
                    if technical_failure:
                        self._merge_supplementary_findings(
                            "literature_result", [literature_provenance_disclosure(lit_result)]
                        )
                        graph_state = getattr(self, "_current_graph_state", None)
                        if graph_state is not None:
                            literature_result = graph_state.get("literature_result") or {}
                            literature_result["status"] = "partial"
                            literature_result.setdefault("technical_failures", []).append(
                                {"topic": topic, **technical_failure}
                            )
                            graph_state["literature_result"] = literature_result
        except asyncio.CancelledError:
            await _mark_request_interrupted(
                self, state, request_id,
                "Supplementary execution was interrupted.",
            )
            raise
        except WorkflowSupersededError:
            raise
        except WorkflowTerminatedError:
            await _mark_request_interrupted(
                self, state, request_id,
                "Supplementary literature research was terminated.",
            )
            raise
        except Exception as e:
            _supp_technical_failure = True
            _supp_error_summary = (
                f"Supplementary literature failed: {type(e).__name__}"
            )
            logger.error(f"[{self.session_id}] Supplementary literature failed: {e}")
            await self._send_message("error", f"Supplementary literature research failed: {e}", AgentType.LITERATURE)

    # Mandatory supplementary settlement capture, after absorbing executions
    # and before the updated report segment/state save.  A partially failed
    # supplement still captures the actual live canonical state.
    from ..adata_checkpoint_hooks import (
        capture_and_persist_boundary,
        clear_settlement_scope_callback,
    )

    latest_round = state.get_latest_round()
    linked_round = (
        latest_round
        if latest_round is not None
        and int(latest_round.round_number or 0) == int(state.current_round or 0)
        else None
    )
    _attempt_ids = (
        list(supp_result.get("settlement_attempt_execution_ids") or [])
        if isinstance(supp_result, dict)
        else []
    )
    _evidence_ids = []
    if isinstance(supp_result, dict):
        for execution in supp_result.get("code_history") or []:
            execution_id = str(getattr(execution, "id", "") or "")
            if (
                execution_id
                and getattr(execution, "execution_status", "failed") == "succeeded"
                and bool(getattr(execution, "evidence_eligible", True))
            ):
                _evidence_ids.append(execution_id)
    _source_context = (
        dict(supp_result.get("resolved_adata_source_context") or {})
        if isinstance(supp_result, dict)
        else {}
    )
    _clear_attempt_scope = clear_settlement_scope_callback(
        state,
        (
            supp_result.get("executor_settlement_scope_id")
            if isinstance(supp_result, dict)
            else request_scope_id
        )
        or request_scope_id,
    )

    def _settle_request(outcome) -> bool:
        changed = bool(_clear_attempt_scope(outcome))
        request_map = dict(
            (getattr(state, "sandbox_state", None) or {}).get(
                "supplementary_requests_v1"
            )
            or {}
        )
        record = dict(request_map.get(request_id) or {})
        if record:
            _analysis_status = (
                str(supp_result.get("status") or "").strip().lower()
                if isinstance(supp_result, dict)
                else ""
            )
            _analysis_failed = _supp_technical_failure or _analysis_status in {
                "failed", "error", "timeout", "adata_source_error",
            }
            record["status"] = "failed" if _analysis_failed else "settled"
            if literature_failures:
                record["literature_technical_failures"] = literature_failures
            record["checkpoint_outcome"] = (
                getattr(outcome.record, "status", None)
                or outcome.disclosure
                or "no_canonical_adata"
            )
            record["error_summary"] = (
                str(_supp_error_summary or outcome.error_summary or "")[:500]
                or None
            )
            record["updated_at"] = datetime.now().isoformat()
            if isinstance(supp_result, dict):
                record["adata_source"] = (
                    supp_result.get("adata_source") or record.get("adata_source")
                )
                record["executor_checkpoint_name"] = supp_result.get(
                    "executor_checkpoint_name"
                )
            request_map[request_id] = record
            _compact_terminal_requests(state, request_map)
            sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
            sandbox_state["supplementary_requests_v1"] = request_map
            state.sandbox_state = sandbox_state
            changed = True
        return changed

    capture_outcome = await capture_and_persist_boundary(
        self,
        state,
        boundary="supplementary_complete",
        event_scope=request_scope_id,
        round_number=int(state.current_round or 0),
        planning_stage=str(supp_plan.get("stage") or "full"),
        phase=str(getattr(state.analysis_phase, "value", state.analysis_phase) or ""),
        source_plan_id=getattr(getattr(state, "current_plan", None), "id", None),
        round_result=linked_round,
        direction=str(supp_plan.get("direction") or ""),
        research_label=request_text[:240],
        summary_at_capture=str(supp_plan.get("summary") or request_text)[:600],
        outcome_at_capture=(
            "partial" if _supp_technical_failure else
            str(supp_result.get("status") or "completed")
            if isinstance(supp_result, dict)
            else "partial"
        ),
        supplementary_request_id=request_id,
        executor_settlement_scope_id_at_capture=(
            supp_result.get("executor_settlement_scope_id")
            if isinstance(supp_result, dict)
            else request_scope_id
        ) or request_scope_id,
        source_preparation_ids_at_capture=tuple(
            _source_context.get("source_preparation_ids") or []
        ),
        active_source_preparation_id=_source_context.get(
            "active_source_preparation_id"
        ) or _source_context.get("preparation_id"),
        source_scope_id_at_capture=_source_context.get(
            "active_source_scope_id"
        ) or _source_context.get("scope_id"),
        execution_ids_at_capture=tuple(_evidence_ids),
        attempt_execution_ids_at_capture=tuple(_attempt_ids),
        before_state_save=_settle_request,
        executor_checkpoint_name=(
            supp_result.get("executor_checkpoint_name")
            if isinstance(supp_result, dict)
            else None
        ),
    )
    if not capture_outcome.available and capture_outcome.disclosure:
        await self._send_message(
            "status",
            "No canonical AnnData checkpoint was captured at this supplementary "
            f"boundary ({capture_outcome.disclosure}).",
            AgentType.PI,
        )

    # Generate new detailed report segment
    seg_index = sum(1 for s in state.detailed_report_segments if "supplementary" in s.phase)
    segment = await self._build_and_save_detailed_segment(
        state,
        f"supplementary_{context}_{seg_index}",
        f"Supplementary Analysis: {request_text[:60]}",
        graph_state=self._current_graph_state,
    )
    report_round = next((
        result for result in reversed(state.round_results)
        if result.round_number == state.current_round
        and result.lineage_id == (state.restore_branch_id or "main")
        and (context != "round" or result.phase != "integration")
    ), None)
    from ...models.schemas import RoundResult
    from ...services.report.report_outline import report_result_identity
    owner = report_round or RoundResult(
        round_number=state.current_round,
        phase=str(getattr(state.analysis_phase, "value", state.analysis_phase)),
        lineage_id=state.restore_branch_id or "main",
        direction=state.current_round_direction or "",
    )
    segment.supplement_owner_key = report_result_identity(owner)
    if report_round is not None:
        # Supplementary segments keep their own content and order. Associate
        # them explicitly rather than collecting every segment with this number.
        report_round.report_supplement_ids.append(segment.id)

    await self._send_message(
        "status",
        (
            "Supplementary analysis finished with a technical failure. Updating reports..."
            if _supp_technical_failure
            else "Supplementary analysis complete. Updating reports..."
        ),
        AgentType.PI,
    )

    # Save state
    if self.state_callback:
        try:
            await self.state_callback(state)
        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to save state after supplementary analysis")
