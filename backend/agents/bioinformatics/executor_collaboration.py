"""Shared response helpers for collaboration and lookup tools."""
from __future__ import annotations

from .executor_dispatch import ToolAction
from .shared import current_execution_identity_error
from ...services.figure_contract import (
    audit_figure_contract,
    collect_plan_figure_requirements,
)
from ...services.figure_revision_tasks import find_figure_revision_task


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def species_validation_response(state, assessment, action=None) -> str:
    if assessment.get("match", True):
        return f"Gene names are consistent with {state.species}. Proceed with analysis."
    if action:
        if action.get("action") == "change_species":
            return f"Species updated to {action.get('new_species', '')}. Proceed."
        if action.get("action") == "custom":
            return "User instruction received. Proceed with analysis."
        return f"Keeping species as {state.species}. Proceed with caution."
    return (
        f"Warning: gene names may not match {state.species}. "
        f"{assessment.get('reasoning', '')}. Proceeding anyway."
    )


def method_guidance(method_question: str, *, format_rule, search_rules) -> str:
    hits = search_rules(method_question, limit=3)
    if hits:
        return "\n\n".join(format_rule(rule) for rule in hits)
    return (
        "No rule in the corpus covers that question. Proceed with standard "
        "practice and state your reasoning in the findings."
    )


async def handle_consult_pi(context, tool_call) -> ToolAction:
    context.has_substantive_work = True
    args = tool_call.arguments
    question = args.get("question", "")
    pi_context = args.get("context", "")
    options = args.get("options", [])

    await context.agent._notify("status", f"Consulting PI: {question[:100]}...")
    await context.agent._notify(
        "pi_question",
        json.dumps({"question": question, "context": pi_context, "options": options}),
    )
    if context.agent.pi_callback:
        pi_response = await context.agent.pi_callback(question, pi_context)
        await context.agent._notify("pi_response", pi_response)
    else:
        pi_response = "PI guidance not available. Proceed with standard practices."

    if PLAN_REVISION_NEEDED_TAG in pi_response:
        clean_response = pi_response.replace(PLAN_REVISION_NEEDED_TAG, "").strip()
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                f"PI Response: {clean_response}\n\n"
                "[System: A plan revision has been triggered. Pause and wait for updated plan.]"
            ),
        })
        context.plan_revision_needed = True
        context.plan_revision_context = {
            "question": question,
            "context": pi_context,
        }
        return ToolAction.BREAK

    context.messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": f"PI Response: {pi_response}",
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="consult_pi",
        agent_reasoning=context.agent_reasoning,
        pi_question=question,
        pi_context=pi_context[:1000],
        pi_options=options or [],
        pi_response=(pi_response or "")[:2000],
    ))
    return ToolAction.CONTINUE


async def handle_annotate_clusters(context, tool_call) -> ToolAction:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    context.has_substantive_work = True
    cluster_markers = args.get("cluster_markers", {})
    tissue = args.get("tissue", "")
    organism = args.get("organism", "mouse")
    # Override with session species if available
    if self._current_analysis_state and self._current_analysis_state.species:
        llm_organism = organism
        organism = self._current_analysis_state.species
        if llm_organism != organism:
            logger.info(f"[Bioinformatics] organism overridden: {llm_organism} -> {organism}")
    cluster_key_hint = args.get("cluster_key") or None
    rank_key_hint = args.get("rank_key") or "rank_genes_groups"
    # Sanitize LLM-provided cluster_markers (handles malformed dict/string formats)
    cluster_markers = self._sanitize_cluster_markers(cluster_markers)
    n_clusters = len(cluster_markers)
    # Discover the active AnnData variable and clustering key in sandbox
    adata_var, cluster_key = await self._discover_adata_in_sandbox(cluster_key_hint)
    if adata_var is None:
        tool_result = (
            "Error: Could not find an AnnData object with a clustering column "
            "(leiden/louvain/cluster, or any column starting with 'leiden'/'louvain') "
            "in the sandbox. Before calling annotate_clusters, "
            "you must: (1) load data into an AnnData variable, (2) run clustering "
            "(e.g., sc.tl.leiden(adata)), and (3) run differential expression "
            "(sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')). "
            "If using non-default column names (e.g., 'leiden_broad'), pass "
            "cluster_key explicitly."
        )
        logger.warning("[Bioinformatics] annotate_clusters called but no AnnData with clustering found")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })
        context.annotation_retry_required = True
        context.annotation_retry_attempts += 1
        return ToolAction.CONTINUE
    # Auto-extract markers from kernel adata if LLM didn't provide them
    if n_clusters == 0 and self.sandbox:
        logger.info(f"[Bioinformatics] cluster_markers empty, auto-extracting from {adata_var}.uns[{rank_key_hint!r}]")
        # P2-11: Read padj cutoff from session-configurable deg_thresholds.
        _deg_thresholds = getattr(self._current_analysis_state, "deg_thresholds", None) or {}
        _padj_cutoff = float(_deg_thresholds.get("padj", 0.05))
        extract_code = (
            "import json as _json\n"
            "_markers = {}\n"
            "try:\n"
            "    import scanpy as sc\n"
            f"    for _cl in {adata_var}.obs['{cluster_key}'].cat.categories:\n"
            f"        _df = sc.get.rank_genes_groups_df({adata_var}, group=str(_cl), key={rank_key_hint!r})\n"
            f"        _df = _df[_df['pvals_adj'] < {_padj_cutoff}]\n"
            "        _markers[str(_cl)] = _df.head(15)['names'].tolist()\n"
            "except Exception as _e:\n"
            "    _markers = {'__error__': str(_e)}\n"
            "print(_json.dumps(_markers))\n"
        )
        try:
            extract_result = await self.sandbox.execute(extract_code)
            if extract_result.success and extract_result.output.strip():
                try:
                    extracted = json.loads(extract_result.output.strip())
                except json.JSONDecodeError as je:
                    logger.warning(f"[Bioinformatics] Auto-extract markers JSON parse failed: {je}")
                    extracted = {}
                if "__error__" not in extracted and len(extracted) > 0:
                    cluster_markers = extracted
                    n_clusters = len(cluster_markers)
                    logger.info(f"[Bioinformatics] Auto-extracted markers for {n_clusters} clusters from adata")
                else:
                    err_msg = extracted.get("__error__", "unknown")
                    logger.warning(f"[Bioinformatics] Auto-extract failed: {err_msg}")
        except Exception as e:
            logger.warning(f"[Bioinformatics] Auto-extract markers exception: {e}")
    if n_clusters == 0:
        tool_result = (
            f"Error: cluster_markers is empty and auto-extraction from "
            f"{adata_var}.uns[{rank_key_hint!r}] failed. You must first run "
            f"sc.tl.rank_genes_groups({adata_var}, '{cluster_key}', method='wilcoxon', "
            f"key={rank_key_hint!r}) and ensure it completes successfully before "
            f"calling annotate_clusters."
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })
        context.annotation_retry_required = True
        context.annotation_retry_attempts += 1
        return ToolAction.CONTINUE
    await self._notify(
        "status",
        f"Preparing marker-database and literature evidence for "
        f"{n_clusters} clusters ({organism} {tissue})...",
    )
    # marker DB 不可用（缺文件 / 陈旧 revision）时的 HITL 门控
    # （必须在算 priors 之前）
    database_prior_result = DatabasePriorResult(
        "skipped", {}, "marker database is unavailable")
    marker_db_skipped = False
    if not _get_marker_db().usable() and self.marker_db_intervention_callback:
        from backend.constants import LlmFailureAction as _LFA
        from backend.workflow.helpers import WorkflowTerminatedError
        action = await self.marker_db_intervention_callback()
        if action == _LFA.TERMINATE:
            raise WorkflowTerminatedError("Analysis terminated by user at marker DB gate")
        if action == _LFA.RETRY:
            _get_marker_db().reload()  # clear stale empty cache after fetch
        elif action == _LFA.SKIP:
            marker_db_skipped = True
    else:
        # Usable, but a source whose download failed leaves
        # the DB narrower than the revision intends and
        # fires no gate, so nothing would ask for it again.
        # Quiet, once per process, never blocking.
        from backend.services import marker_priors as _mp
        if await asyncio.to_thread(
                _mp.top_up_missing_tables, _get_marker_db()):
            _get_marker_db().reload()
    workspace_dir = getattr(self.sandbox, "working_dir", None)
    if marker_db_skipped:
        database_prior_result = DatabasePriorResult(
            "skipped", {}, "user chose literature-only annotation")
    elif not _get_marker_db().usable():
        database_prior_result = DatabasePriorResult(
            "skipped", {}, "marker database remained unavailable")
    elif workspace_dir is None:
        database_prior_result = DatabasePriorResult(
            "skipped", {}, "sandbox workspace is unavailable")
    else:
        database_prior_result = await self._compute_database_priors(
            adata_var, cluster_key, rank_key_hint, organism,
            workspace_dir, tissue=tissue,
            cluster_markers=cluster_markers)
    database_priors = database_prior_result.priors
    await self._notify(
        "status", database_prior_result.realtime_message(n_clusters))
    annotation_result: Dict[str, Any] = {}
    if self.annotate_clusters_callback:
        annotation_result = await self.annotate_clusters_callback(
            cluster_markers, tissue, organism,
            analysis_log=analysis_log,
            database_priors=database_priors,
        )
    else:
        annotation_result = {
            "status": "unavailable",
            "annotations": {},
            "error": "Cluster annotation service not available. Use consult_pi instead."
        }
    # If annotation service returned an error, inform the agent
    # and do NOT set context.annotation_completed (so report_findings gate blocks)
    if annotation_result.get("status") == "error":
        error_detail = annotation_result.get("error", "Unknown annotation error")
        # Check if there's raw_response for debugging
        raw_response = annotation_result.get("raw_response", "")
        tool_result = (
            f"[ANNOTATION FAILED] {error_detail}\n"
            f"Raw response preview: {raw_response[:500] if raw_response else 'N/A'}\n"
            "Do NOT manually assign cell types via execute_code. "
            "Cell type annotation MUST go through annotate_clusters for user review. "
            "You may retry annotate_clusters or consult_pi for guidance."
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="annotate_clusters",
            agent_reasoning=agent_reasoning,
            annotation_result_summary=tool_result,
            annotation_database_priors_summary=(
                database_prior_result.summary(len(cluster_markers))),
        ))
        return ToolAction.CONTINUE
    # Format result for the agent to use in code
    annotations = annotation_result.get("annotations", {})
    tissue_summary = annotation_result.get("tissue_summary", "")
    # Build a code-friendly response
    result_lines = []
    if annotation_result.get("already_confirmed"):
        if context.planning_stage == "stage1":
            result_lines.append(
                "[NOTE: These annotations were already confirmed by the user. "
                "No re-annotation needed. The platform will render the Stage 1 "
                "baseline figures after this pass; call report_findings now.]\n"
            )
        else:
            result_lines.append(
                "[NOTE: These annotations were already confirmed by the user. "
                "No re-annotation needed. Proceed to downstream analysis.]\n"
            )
    if tissue_summary:
        result_lines.append(f"Tissue context: {tissue_summary}\n")
    result_lines.append("Cluster annotations:")
    for cid, info in annotations.items():
        if isinstance(info, dict):
            cell_type = info.get("cell_type", "Unknown")
            confidence = info.get("confidence", "low")
            key_markers = info.get("key_markers", [])
            evidence = info.get("evidence", "")
            result_lines.append(
                f"  Cluster {cid}: {cell_type} (confidence: {confidence})"
            )
            if key_markers:
                result_lines.append(f"    Key markers: {', '.join(key_markers[:5])}")
            if evidence:
                result_lines.append(f"    Evidence: {evidence[:200]}")
    result_lines.append(f"\nTotal clusters annotated: {len(annotations)}")
    # Provide ready-to-use Python code snippet
    # Use json.dumps for safe serialization (prevents code injection via cell type names)
    cell_type_dict = {
        str(cid): (info.get("cell_type", "Unknown") if isinstance(info, dict) else "Unknown")
        for cid, info in annotations.items()
    }
    dict_str = json.dumps(cell_type_dict, ensure_ascii=False)
    result_lines.append(
        f"\nYou can now assign these annotations to {adata_var}.obs['cell_type'] using:\n"
        f"cluster_to_celltype = {dict_str}\n"
        f"{adata_var}.obs['cell_type'] = {adata_var}.obs['{cluster_key}'].astype(str).map(cluster_to_celltype).fillna('Unknown')"
    )
    tool_result = "\n".join(result_lines)
    # Auto-apply annotations to adata.obs['cell_type'] in sandbox
    # so the agent doesn't need to manually execute the mapping code
    if annotations and self.sandbox:
        cell_type_dict = {
            str(cid): (info.get("cell_type", "Unknown") if isinstance(info, dict) else "Unknown")
            for cid, info in annotations.items()
        }
        inject_code = (
            "import json as _json\n"
            f"_ct_map = _json.loads({json.dumps(json.dumps(cell_type_dict, ensure_ascii=False))})\n"
            f"{adata_var}.obs['cell_type'] = {adata_var}.obs['{cluster_key}'].astype(str).map(_ct_map).fillna('Unknown')\n"
            f"print(f'cell_type applied: {{{adata_var}.obs[\"cell_type\"].value_counts().to_dict()}}')\n"
        )
        try:
            inject_result = await self.sandbox.execute(inject_code)
            if inject_result.success:
                logger.info(f"[Bioinformatics] Auto-applied cell_type annotations to adata.obs")
            else:
                logger.warning(f"[Bioinformatics] Failed to auto-apply annotations: {inject_result.error}")
        except Exception as e:
            logger.warning(f"[Bioinformatics] Exception auto-applying annotations: {e}")
    await self._notify("annotation_result", json.dumps(
        annotation_result, ensure_ascii=False, default=str
    )[:2000])
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result
    })
    # Record annotate_clusters step in analysis log
    markers_summary = ", ".join(
        f"Cluster {k}: {', '.join(str(x) for x in v[:3]) if isinstance(v, list) else str(v)[:50]}"
        for k, v in list(cluster_markers.items())[:5]
    )
    if len(cluster_markers) > 5:
        markers_summary += f" ... ({len(cluster_markers)} clusters total)"
    analysis_log.append(AnalysisStep(
        step_number=step_count, step_type="annotate_clusters",
        agent_reasoning=agent_reasoning,
        cluster_markers_summary=markers_summary, annotation_tissue=tissue,
        annotation_organism=organism, annotation_result_summary=(tool_result or ""),
        annotation_database_priors_summary=database_prior_result.summary(
            len(cluster_markers)),
    ))
    context.annotation_completed = True
    context.annotation_retry_required = False
    context.annotation_retry_attempts = 0
    # The workflow renders and verifies baseline figures
    # immediately after this autonomous pass.
    if context.planning_stage == "stage1":
        messages.append({
            "role": "user",
            "content": (
                "STAGE 1 ANNOTATION COMPLETE. Cell annotation is done and confirmed by the user. "
                "The platform will now render and verify the cell-type, cluster, batch/sample, "
                "and marker baseline figures supported by the annotated object. "
                "Call report_findings now. "
                "Do NOT run differential expression, pathway enrichment, "
                "cell communication, trajectory analysis, or any other downstream analysis. "
                "These will be done in Stage 2 after PI reviews your Stage 1 results "
                "and creates a new analysis plan."
            )
        })
    return ToolAction.CONTINUE


async def handle_search_literature(context, tool_call) -> ToolAction:
    context.has_substantive_work = True
    topic = tool_call.arguments.get("topic", "")
    await context.agent._notify("status", f"Searching literature: {topic[:50]}...")
    if context.agent.literature_callback:
        lit_result = await context.agent.literature_callback(topic)
        await context.agent._notify("literature", lit_result[:1000])
    else:
        lit_result = "Literature search not available."
    context.messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": f"Literature findings: {lit_result}",
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="search_literature",
        agent_reasoning=context.agent_reasoning,
        literature_topic=topic,
        literature_result=(lit_result or "")[:2000],
    ))
    return ToolAction.CONTINUE


async def handle_validate_species(context, tool_call) -> ToolAction:
    gene_names = tool_call.arguments.get("gene_names", [])
    state = context.agent._current_analysis_state
    if not state or not state.species:
        tool_result = "No species set in session. Skipping validation."
    else:
        assessment = await context.agent._assess_gene_species_match(
            gene_names, state.species
        )
        action = None
        if not assessment.get("match", True) and context.agent.species_validation_callback:
            action = await context.agent.species_validation_callback(
                gene_names, state.species, assessment
            )
            if action.get("action") == "custom":
                context.agent.analysis_context["species_user_directive"] = action.get(
                    "user_input", ""
                )
        tool_result = species_validation_response(state, assessment, action)
        state.species_genes_validated = True
    context.messages.append({
        "role": "tool", "tool_call_id": tool_call.id, "content": tool_result,
    })
    return ToolAction.CONTINUE


async def handle_lookup_documentation(context, tool_call) -> ToolAction:
    args = tool_call.arguments
    function_path = args.get("function_path", "")
    question = args.get("question", "")
    method_question = args.get("method_question", "")
    method_answer = ""
    if method_question:
        try:
            from ...services.method_rules import format_rule, search_rules

            method_answer = method_guidance(
                method_question, format_rule=format_rule, search_rules=search_rules
            )
        except Exception as exc:
            logger.warning("[Bioinformatics] method rule lookup failed: %s", exc)

    if method_question and not function_path:
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": method_answer or "Method guidance unavailable.",
        })
        context.analysis_log.append(AnalysisStep(
            step_number=context.step_count,
            step_type="lookup_documentation",
            agent_reasoning=context.agent_reasoning,
            code_description=f"Method lookup: {method_question[:80]}",
        ))
        return ToolAction.CONTINUE

    if context.agent.technique_agent:
        await context.agent._notify("status", f"Looking up docs: {function_path}...")
        try:
            doc_result = await context.agent.technique_agent.lookup_docs(
                context.agent.sandbox, function_path, question=question
            )
        except Exception as exc:
            logger.warning("[Bioinformatics] lookup_documentation failed: %s", exc)
            doc_result = f"Documentation lookup failed: {exc}"
    else:
        doc_result = "Documentation lookup not available."
    if method_answer:
        doc_result = f"{doc_result}\n\n{method_answer}"
    context.messages.append({
        "role": "tool", "tool_call_id": tool_call.id, "content": doc_result,
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="lookup_documentation",
        agent_reasoning=context.agent_reasoning,
        code_description=f"Docs lookup: {function_path}",
    ))
    return ToolAction.CONTINUE


async def handle_publication_figure(context, tool_call) -> ToolAction:
    args = tool_call.arguments
    state = context.state
    code = args.get("code", "")
    figure_name = args.get(
        "figure_name", f"figure_{len(state.publication_figures) + 1}"
    )
    logical_figure_id = str(args.get("logical_figure_id") or figure_name).strip()
    description = args.get("description", "")
    figure_type = args.get("figure_type", "other")
    source_execution_id = str(args.get("evidence_execution_id") or "").strip()
    raw_plan_step_id = args.get("plan_step_id")
    malformed_plan_step = False
    try:
        plan_step_id = (
            int(raw_plan_step_id)
            if raw_plan_step_id is not None and str(raw_plan_step_id).strip()
            else None
        )
    except (TypeError, ValueError):
        plan_step_id = None
        malformed_plan_step = True
    approved_step_ids = {
        int(step.id) for step in (getattr(state.current_plan, "steps", None) or [])
    }
    invalid_plan_step = malformed_plan_step or (
        plan_step_id is not None and plan_step_id not in approved_step_ids
    )
    nomination_note = ""
    if invalid_plan_step:
        nomination_note = (
            f"Submitted plan_step_id {raw_plan_step_id!r} was not current and was "
            "ignored; Supervisor semantic review remained authoritative."
        )
        plan_step_id = None
    rendered = False
    current_figure = None
    safe_fig_name = "".join(
        char if char.isalnum() or char == "_" else "_" for char in figure_name
    )
    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    source_execution = next((
        execution for execution in context.agent.code_history
        if str(getattr(execution, "id", "") or "") == source_execution_id
    ), None)
    source_error = current_execution_identity_error(state, source_execution)

    binding_decision = "supplementary"
    binding_reason = nomination_note or None
    requirement_key = None
    purpose = "supplementary"
    plan = getattr(state, "current_plan", None)
    requirements = collect_plan_figure_requirements(plan)
    if requirements:
        callback = getattr(context.agent, "figure_plan_binding_callback", None)
        if callback is not None:
            try:
                semantic = await callback(state, {
                    "code": code,
                    "figure_name": figure_name,
                    "figure_type": figure_type,
                    "description": description,
                    "purpose": "result",
                    "plan_step_id": plan_step_id,
                })
            except Exception as exc:
                semantic = {
                    "decision": "supplementary",
                    "plan_requirement_key": None,
                    "reason": f"Supervisor semantic binding failed: {exc}",
                    "parse_failed": True,
                }
        else:
            semantic = {
                "decision": "supplementary",
                "plan_requirement_key": None,
                "reason": "Supervisor semantic binding callback is unavailable.",
                "parse_failed": True,
            }
        semantic_decision = str(semantic.get("decision") or "supplementary")
        semantic_reason = str(semantic.get("reason") or "")[:2000]
        proposed_key = str(
            semantic.get("plan_requirement_key") or ""
        ).strip() or None
        requirement_by_key = {item.key: item for item in requirements}
        missing_keys = {
            item.requirement.key
            for item in audit_figure_contract(
                state,
                plan=plan,
                planning_stage=str(getattr(plan, "stage", "full") or "full"),
                include_stage_baseline=False,
            ).missing
        }
        requirement = requirement_by_key.get(proposed_key)
        objective_binding_error = None
        if semantic_decision in {"accept", "rebind"}:
            if requirement is None:
                objective_binding_error = (
                    "Supervisor returned a plan requirement outside the current "
                    "materialized visual contract."
                )
            elif proposed_key not in missing_keys:
                objective_binding_error = (
                    "Supervisor selected a requirement that is already satisfied."
                )
            elif source_error is not None:
                objective_binding_error = source_error
        if (
            semantic_decision in {"accept", "rebind"}
            and requirement is not None
            and proposed_key in missing_keys
            and source_error is None
        ):
            binding_decision = semantic_decision
            binding_reason = semantic_reason or nomination_note or None
            plan_step_id = requirement.plan_step_id
            requirement_key = requirement.key
            purpose = "result"
        else:
            plan_step_id = None
            binding_decision = "supplementary"
            binding_reason = (
                objective_binding_error
                or semantic_reason
                or source_error
                or nomination_note
                or "No unambiguous current plan requirement was confirmed."
            )
    else:
        plan_step_id = None
        binding_reason = (
            source_error or nomination_note or "No current plan figure requirement exists."
        )

    revision_task = None
    if requirement_key is not None:
        revision_task = find_figure_revision_task(
            state,
            plan_id=str(getattr(plan, "id", "") or "") or None,
            plan_requirement_key=requirement_key,
        )
        if revision_task is not None:
            logical_figure_id = revision_task.logical_figure_id

    if revision_task is not None and revision_task.status == "exhausted_excluded":
        tool_result = (
            "Figure redraw budget already exhausted for this plan requirement. "
            "The unresolved figure remains excluded from reports; continue the "
            "analysis without another automatic redraw."
        )
    else:
        await context.agent._notify(
            "status", f"Generating publication figure: {figure_name}"
        )
        previous_figure_ids = {
            getattr(figure, "id", None)
            for figure in state.publication_figures
            if (getattr(figure, "lineage_id", None) or "main") == active_lineage
        }
        tool_result = await context.agent._handle_generate_publication_figure(
            code=code,
            figure_name=figure_name,
            description=description,
            figure_type=figure_type,
            state=state,
            review_callback=context.review_callback,
            plan_step_id=plan_step_id,
            logical_figure_id=logical_figure_id,
            purpose=purpose,
            source_execution_id=(source_execution_id if source_error is None else None),
            submitted_plan_step_id=(
                int(raw_plan_step_id)
                if raw_plan_step_id is not None and not malformed_plan_step
                else None
            ),
            plan_requirement_key=requirement_key,
            supervisor_binding_decision=binding_decision,
            supervisor_binding_reason=binding_reason,
        )
        current_figure = next((
            figure for figure in reversed(state.publication_figures)
            if getattr(figure, "id", None) not in previous_figure_ids
            and figure.name == safe_fig_name
            and (getattr(figure, "lineage_id", None) or "main") == active_lineage
        ), None)
        rendered = current_figure is not None
        if rendered:
            context.has_figure_work = True
            context.figures_rendered += 1
    if rendered and current_figure and current_figure.png_base64:
        context.step_figures.append(current_figure.png_base64)
    context.messages.append({
        "role": "tool", "tool_call_id": tool_call.id, "content": tool_result,
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="generate_publication_figure",
        agent_reasoning=context.agent_reasoning,
        pub_figure_name=figure_name,
        pub_figure_description=description,
        pub_figure_type=figure_type,
        pub_figure_plan_step_id=plan_step_id,
        pub_figure_code=code,
        pub_figure_result=(tool_result or "")[:2000],
    ))
    return ToolAction.CONTINUE
