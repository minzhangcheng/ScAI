"""Pure response assembly for the execute-code tool family."""
from __future__ import annotations

from .executor_dispatch import ToolAction


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def build_execution_tool_result(code_execution, result) -> str:
    content = (
        f"Execution {'successful' if not result.error else 'failed'}.\n"
        f"Evidence execution ID: {code_execution.id}\n"
        f"Plan binding: {code_execution.plan_binding_status}"
        + (
            f" (plan {code_execution.plan_id}, steps {code_execution.plan_step_ids})\n"
            if code_execution.plan_binding_status in {"candidate", "confirmed", "bound"}
            else "\n"
        )
    )
    if code_execution.plan_binding_reason:
        content += f"Plan-link note: {code_execution.plan_binding_reason}\n"
    if code_execution.cohort_digest:
        content += (
            "Cohort identity: generation "
            f"{code_execution.cohort_generation}, "
            f"digest {code_execution.cohort_digest[:12]}...\n"
        )
    if result.output:
        content += f"Output:\n{result.output}\n"
    if result.error:
        content += f"Error: {result.error}\n"
        if "ModuleNotFoundError" in result.error:
            content += (
                "\nHint: A required package is not installed. "
                "Report the missing dependency, explain which analysis step is limited, "
                "and use an already-installed alternative only when the substitution is "
                "scientifically valid.\n"
            )
    if result.figures:
        content += f"Generated {len(result.figures)} figure(s).\n"
    return content


def reject_sample_design_code(context, tool_call, code, description, step_type) -> bool:
    """Record and reject code incompatible with the measured sample design."""
    design_error = code_sample_design_violation(code, context.state)
    if not design_error:
        return False
    logger.warning(
        "[Bioinformatics] Blocked code incompatible with sample design: %s",
        design_error,
    )
    context.messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": f"Execution blocked: {design_error}",
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="execute_code",
        agent_reasoning=context.agent_reasoning,
        code=code,
        code_error=design_error,
        code_description=description,
        code_step_type=step_type,
    ))
    return True


def execution_plan_binding(
    state, args,
) -> tuple[str, list[int], str, str | None]:
    """Normalize an agent nomination without blocking scientific execution.

    The nomination is only a candidate for the Supervisor's later semantic
    reconciliation.  Invalid or missing IDs are recorded as unverified instead
    of spending another autonomous-agent turn on a schema repair.
    """
    purpose = str(args.get("execution_purpose") or "").strip().lower()
    warnings: list[str] = []
    if purpose not in {"planned", "supplementary", "repair"}:
        if purpose:
            warnings.append(
                "execution_purpose was not recognized and was treated as supplementary"
            )
        purpose = "supplementary"
    raw_ids = args.get("plan_step_ids")
    if raw_ids is None:
        raw_ids = []
    elif not isinstance(raw_ids, list):
        warnings.append("plan_step_ids was not an array and was ignored")
        raw_ids = []
    step_ids: list[int] = []
    for value in raw_ids:
        if isinstance(value, bool):
            warnings.append("a non-integer plan_step_id was ignored")
            continue
        try:
            step_id = int(value)
        except (TypeError, ValueError):
            warnings.append("a non-integer plan_step_id was ignored")
            continue
        if step_id <= 0:
            warnings.append("a non-positive plan_step_id was ignored")
            continue
        if step_id not in step_ids:
            step_ids.append(step_id)

    plan = getattr(state, "current_plan", None)
    approved_ids = {
        int(step.id) for step in (getattr(plan, "steps", None) or [])
    }
    invalid = sorted(set(step_ids) - approved_ids)
    if invalid:
        warnings.append(
            f"unknown plan_step_ids {invalid} were ignored"
        )
        step_ids = [step_id for step_id in step_ids if step_id in approved_ids]
    has_plan = bool(str(getattr(plan, "id", "") or "").strip())
    if purpose in {"planned", "repair"}:
        if has_plan and step_ids:
            status = "candidate"
        else:
            status = "unverified"
            warnings.append(
                "planned work had no valid current-plan step nomination; "
                "Supervisor reconciliation will determine any semantic link"
            )
    else:
        status = "supplementary"
        if step_ids:
            warnings.append(
                "supplementary execution cannot claim plan credit; nominated "
                "step IDs remain audit context only"
            )
    return purpose, step_ids, status, "; ".join(dict.fromkeys(warnings)) or None


def execution_input_binding(state, args):
    """Normalize plan/profile candidates without refusing executable code."""
    purpose, step_ids, binding_status, warning = execution_plan_binding(
        state, args
    )
    raw_profile_ids = args.get("input_profile_ids")
    profile_ids = []
    if raw_profile_ids is None:
        raw_profile_ids = []
    elif not isinstance(raw_profile_ids, list):
        warning = warning or "input_profile_ids was not an array and was ignored"
        raw_profile_ids = []
    if isinstance(raw_profile_ids, list):
        proposed_ids = list(dict.fromkeys(
            str(value).strip() for value in raw_profile_ids if str(value).strip()
        ))
        known_ids = {item.id for item in state.bulk_input_profiles}
        profile_ids = [value for value in proposed_ids if value in known_ids]
        invalid_ids = sorted(set(proposed_ids) - known_ids)
        if invalid_ids:
            note = f"unknown input_profile_ids {invalid_ids} were ignored"
            warning = f"{warning}; {note}" if warning else note
    return purpose, step_ids, profile_ids, binding_status, warning


def build_code_execution_receipt(
    state, code, result, execution_time, purpose, step_ids, cohort, profile_ids,
    binding_status, binding_reason, persistent_outputs,
):
    """Build the schema-v2 receipt from host-observed execution facts."""
    return CodeExecution(
        code=code,
        output=result.output,
        error=result.error,
        figures=result.figures,
        execution_time=execution_time,
        round_number=state.current_round,
        analysis_modalities=infer_execution_modalities(code),
        evidence_schema_version=2,
        plan_id=str(getattr(state.current_plan, "id", "") or "").strip() or None,
        plan_step_ids=step_ids,
        execution_purpose=purpose,
        plan_binding_status=binding_status,
        plan_binding_reason=binding_reason,
        execution_status="failed" if result.error else "succeeded",
        lineage_id=getattr(state, "restore_branch_id", None) or "main",
        analysis_generation=int(
            (state.sandbox_state or {}).get("analysis_generation", 0) or 0
        ),
        cohort_generation=int(cohort.get("cohort_generation", 0) or 0),
        cohort_digest=str(cohort.get("cohort_digest") or "") or None,
        input_profile_ids=profile_ids,
        declared_persistent_outputs=[
            str(name).strip()
            for name in (persistent_outputs or [])
            if str(name).strip().isidentifier()
        ],
        realized_persistent_outputs=(
            [
                str(name).strip()
                for name in (persistent_outputs or [])
                if str(name).strip().isidentifier()
            ]
            if not result.error else []
        ),
    )


async def record_execution_result(
    context, code_exec, result, code, description, step_type, purpose, step_ids,
):
    """Publish one execution receipt, log entry, result, and exploratory figures."""
    agent = context.agent
    agent.code_history.append(code_exec)
    agent.last_result = result
    saved_figures = []
    if result.output:
        saved_figures = [
            Path(path).name for path in re.findall(
                r'\[Auto-saved\] (?:JPG|PNG): (.+\.(?:jpe?g|png))', result.output,
            )
        ]
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="execute_code",
        agent_reasoning=context.agent_reasoning,
        code=code,
        code_output=(result.output or "")[-3000:],
        code_error=compact_traceback(result.error, max_chars=3000) if result.error else None,
        code_description=description,
        code_step_type=step_type,
        plan_id=code_exec.plan_id,
        plan_step_ids=step_ids,
        execution_purpose=purpose,
        execution_id=code_exec.id,
        saved_figure_filenames=saved_figures,
    ))
    if result.output:
        await agent._notify("result", result.output[:2000])
    for figure in result.figures:
        await agent._notify("figure", figure)
        context.all_figures.append(figure)
        context.step_figures.append(figure)


async def handle_execute_code(context, tool_call) -> ToolAction:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    context.has_substantive_work = True
    code = args.get("code", "")
    description = args.get("description", "")
    step_type = args.get("step_type", "technical")
    persistent_outputs = args.get("persistent_outputs", [])
    (
        execution_purpose,
        plan_step_ids,
        input_profile_ids,
        plan_binding_status,
        binding_warning,
    ) = (
        execution_input_binding(state, args)
    )
    if reject_sample_design_code(context, tool_call, code, description, step_type):
        return ToolAction.CONTINUE
    is_safe, security_error = self._validate_code(code)
    if not is_safe:
        error_msg = f"Code validation failed: {security_error}"
        logger.warning(f"[Bioinformatics] {error_msg}")
        await self._notify("error", error_msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Execution blocked: {error_msg}. Please rewrite the code without using dangerous operations."
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="execute_code",
            agent_reasoning=agent_reasoning,
            code=code, code_error=error_msg,
            code_description=description, code_step_type=step_type,
        ))
        return ToolAction.CONTINUE
    if (
        state.analysis_phase == AnalysisPhase.EXPLORATION
        and not context.annotation_completed
        and _CELL_TYPE_ASSIGN_RE.search(code)
    ):
        guard_msg = (
            "Manual assignment to cell-type-like adata.obs columns "
            "(cell_type, celltype, cluster_identity, cell_identity, cell_annotation, annotation) "
            "is not allowed before cell type annotations have been confirmed through "
            "the annotate_clusters workflow. Please call annotate_clusters first."
        )
        logger.warning(f"[Bioinformatics] Blocked manual cell-type-like assignment: annotation not confirmed")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Execution blocked: {guard_msg}"
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="execute_code",
            agent_reasoning=agent_reasoning,
            code=code, code_error=guard_msg,
            code_description=description, code_step_type=step_type,
        ))
        return ToolAction.CONTINUE
    await self._notify("status", f"[{step_type.upper()}] {description}")
    await self._notify("code", code)
    instrumented_code, sensitive_call_count = (
        instrument_expression_contract_calls(code)
    )
    preinstall_expression_runtime = (
        has_direct_expression_contract_call(code)
    )
    guarded_code = (
        instrumented_code
        if (
            preinstall_expression_runtime
            or not sensitive_call_count
        )
        else (
            build_expression_runtime_code()
            + "\n"
            + instrumented_code
        )
    )
    figures_dir = Path(self.sandbox.working_dir) / "figures"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wrapped_code = _build_exec_autosave_wrapper(
        guarded_code, str(figures_dir), timestamp
    )
    try:
        cohort_before = await probe_canonical_cohort(self.sandbox)
    except Exception as cohort_error:
        logger.warning(
            "[Bioinformatics] Could not inspect pre-execution cohort: %s",
            compact_traceback(str(cohort_error), max_chars=1000),
        )
        cohort_before = None
    representation_will_change = mutates_expression_representation(code)
    preexisting_artifact_versions = (
        workspace_artifact_versions(self.sandbox.working_dir)
        if representation_will_change else {}
    )
    execution_started = time.perf_counter()
    user_code_execution_started = False
    invalidated_count = 0
    staleness_note = ""
    try:
        runtime_setup_result = None
        if preinstall_expression_runtime:
            runtime_setup_result = await self.sandbox.execute(
                build_expression_runtime_code(),
                save_to_history=False,
                skip_safety_check=True,
            )
        if (
            runtime_setup_result is not None
            and runtime_setup_result.error
        ):
            result = runtime_setup_result
        else:
            user_code_execution_started = True
            result = await self.sandbox.execute(
                wrapped_code,
                declared_outputs=list(persistent_outputs or []),
                track_globals=True,
                skip_safety_check=True,
            )
    finally:
        execution_time = time.perf_counter() - execution_started
        try:
            (
                invalidated_count,
                _stale_stems,
                staleness_note,
            ) = record_executed_representation_change(
                state,
                code,
                self.sandbox.working_dir,
                preexisting_artifact_versions,
                execution_started=user_code_execution_started,
            )
        except Exception as provenance_error:
            logger.warning(
                "[Bioinformatics] Could not record expression "
                "representation invalidation: %s",
                compact_traceback(
                    str(provenance_error), max_chars=1000
                ),
            )
        if invalidated_count:
            logger.warning(
                "[Bioinformatics] Invalidated %d stale artifacts "
                "after expression representation change",
                invalidated_count,
            )
    await self.sandbox.execute(
        "import anndata as _ad\n"
        "for _name, _obj in list(globals().items()):\n"
        "    if isinstance(_obj, _ad.AnnData):\n"
        "        _obj.obs_names_make_unique()\n"
        "        _obj.var_names_make_unique()\n"
        "del _ad\n",
        save_to_history=False, skip_safety_check=True
    )
    try:
        cohort_after = await probe_canonical_cohort(self.sandbox)
        cohort_identity = record_cohort_transition(
            state, cohort_before, cohort_after
        )
    except Exception as cohort_error:
        logger.warning(
            "[Bioinformatics] Could not record active cohort identity: %s",
            compact_traceback(str(cohort_error), max_chars=1000),
        )
        cohort_identity = active_cohort_identity(state)
    expression_contracts = parse_expression_contracts(result.output)
    if expression_contracts:
        record_expression_contract(state, expression_contracts)
    result.output = strip_expression_contract_markers(result.output)
    if result.output:
        result.output = truncate_for_llm(result.output)
    if result.error:
        result.error = compact_traceback(result.error, max_chars=3000)
    if staleness_note:
        if result.error:
            result.error = f"{result.error}\n{staleness_note}".strip()
        else:
            result.output = f"{result.output or ''}\n{staleness_note}".strip()
    code_exec = build_code_execution_receipt(
        state, code, result, execution_time, execution_purpose, plan_step_ids,
        cohort_identity, input_profile_ids, plan_binding_status,
        binding_warning, persistent_outputs,
    )
    await record_execution_result(
        context, code_exec, result, code, description, step_type,
        execution_purpose, plan_step_ids,
    )
    swallowed_type, swallowed_doc_hint = (
        await self._detect_and_lookup_swallowed_error(result, code)
    )
    if swallowed_type:
        code_exec.evidence_eligible = False
        code_exec.execution_status = "partial"
        code_exec.evidence_rejection_reason = (
            f"Swallowed technical error: {swallowed_type}"
        )
    if result.error:
        await self._notify("error", result.error)
        context.consecutive_errors += 1
        context.total_errors += 1
    else:
        context.consecutive_errors = 0
        self._force_skip_counter = 0
        self._force_skip_error_type = None
        if (self._current_analysis_state
                and self._current_analysis_state.species
                and not self._current_analysis_state.species_genes_validated
                and self._current_analysis_state.current_round == 1):
            validated = await self._auto_validate_species_genes()
            if validated:
                self._current_analysis_state.species_genes_validated = True
        if self._current_analysis_state:
            sandbox_state = self._current_analysis_state.sandbox_state or {}
            tracker = getattr(self.sandbox, "runtime_dependency_tracker", None)
            if tracker is not None:
                sandbox_state["runtime_dependencies_v2"] = sorted(
                    tracker.required_outputs
                )
                sandbox_state["runtime_dependency_tracker_v2"] = tracker.to_state_dict()
            else:
                # No tracker (should not happen with track_globals): fall
                # back to append-only explicit declarations.
                deps = sandbox_state.setdefault("runtime_dependencies_v2", [])
                for var_name in (persistent_outputs or []):
                    _var_name = str(var_name).strip()
                    if (
                        _var_name.isidentifier()
                        and _var_name not in deps
                    ):
                        deps.append(_var_name)
            sandbox_state["runtime_revision"] = getattr(
                self.sandbox, "runtime_revision", 0
            )
            self._current_analysis_state.sandbox_state = sandbox_state
            self.sandbox.sync_runtime_persistence_state(
                self._current_analysis_state
            )
    tool_result = build_execution_tool_result(code_exec, result)
    if swallowed_type:
        tool_result += _build_swallowed_error_feedback(
            swallowed_type, swallowed_doc_hint
        )
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result
    })
    if context.consecutive_errors >= context.max_consecutive_errors:
        if state.analysis_phase == AnalysisPhase.EXPLORATION and not context.annotation_completed:
            logger.warning(f"[Bioinformatics] {context.consecutive_errors} consecutive errors, redirecting to mandatory steps")
            messages.append({
                "role": "user",
                "content": (
                    "[System] Too many consecutive code execution errors. "
                    "However, mandatory steps are not yet complete. "
                    "Try a simpler approach: run sc.tl.rank_genes_groups(adata_hvg, 'leiden', method='wilcoxon') "
                    "then call annotate_clusters."
                )
            })
        else:
            logger.warning(f"[Bioinformatics] {context.consecutive_errors} consecutive code errors, last-chance attempt")
            messages.append({
                "role": "user",
                "content": (
                    "[System] You have had several consecutive code execution errors. "
                    "This is your last chance to fix the code. Take a completely different "
                    "approach: simplify the code, use a different library function, or "
                    "write a minimal version first. If this attempt also fails, you MUST "
                    "call report_findings with whatever results you have so far."
                )
            })
        context.consecutive_errors = 0  # Reset after circuit breaker triggers
    elif context.total_errors >= context.max_total_errors:
        logger.warning(f"[Bioinformatics] {context.total_errors} total errors reached hard limit, forcing report")
        messages.append({
            "role": "user",
            "content": (
                "[System] Too many code execution errors overall. "
                "Stop trying to fix the code. Report your findings so far "
                "using the report_findings tool and move on."
            )
        })
    return ToolAction.CONTINUE
