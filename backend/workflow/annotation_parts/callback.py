"""Annotation literature, PI matching, and HITL callback phases."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

async def _generate_annotation_brief(self, cluster_markers, organism, tissue, state):
    """Ask PI for a focused annotation Deep Research brief (None on failure/empty)."""
    return await safe_pi_brief(
        lambda: self.pi_agent.generate_annotation_research_brief(
            cluster_markers=cluster_markers,
            organism=organism,
            tissue=tissue,
            state=state,
        ),
        self.session_id,
        "annotation",
    )

async def begin_annotation(
    self, cluster_markers, tissue, organism, analysis_log=None,
    database_priors=None,
):
    n_clusters = len(cluster_markers)
    if n_clusters == 0:
        logger.warning(f"[{self.session_id}] annotate_clusters called with empty cluster_markers")
        return {
            "status": "error",
            "annotations": {},
            "error": "cluster_markers is empty. Run rank_genes_groups and extract markers first."
        }

    if self._current_analysis_state is None:
        raise RuntimeError(
            "_current_analysis_state must be set before annotate_clusters; "
            "this fallback was previously a silent ValidationError because user_id is required."
        )
    state = self._current_analysis_state

    # Early exit: if annotations were already confirmed, return cached results
    if state.annotation_state and state.annotation_state.confirmed:
        logger.info(f"[{self.session_id}] Annotation already confirmed, returning cached annotations")
        cached_annotations = {}
        for ca in state.annotation_state.cluster_annotations:
            cached_annotations[ca.cluster_id] = {
                "cell_type": ca.user_override or ca.cell_type,
                "subtypes": ca.subtypes,
                "key_markers": ca.key_markers,
                "confidence": ca.confidence,
                "evidence": ca.evidence,
                "alternative_cell_types": ca.alternative_cell_types,
                "low_confidence_reason": ca.low_confidence_reason,
            }
        return {
            "annotations": cached_annotations,
            "tissue_summary": state.annotation_state.tissue_summary or "",
            "already_confirmed": True,
        }

    # Refresh literature config before annotation use
    self.literature_agent.refresh_runtime_config_if_needed()

    # --- PI-guided annotation ---
    lit_mode = getattr(self.literature_agent, 'literature_mode', 'pubmed_llm')
    pi_search_guided = False
    pi_query = None
    pi_evaluate = None

    # Step 0a: PI generates search query (only for pubmed_llm mode)
    if lit_mode == "pubmed_llm":
        try:
            pi_query = await self.pi_agent.generate_annotation_search_query(
                cluster_markers=cluster_markers,
                organism=organism,
                tissue=tissue,
                state=state,
            )
            logger.info(f"[{self.session_id}] PI generated annotation query: {pi_query[:100]}")

            async def pi_evaluate(articles, search_history):
                return await self.pi_agent.evaluate_annotation_search_results(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    articles=articles,
                    search_history=search_history,
                    state=state,
                )

            pi_search_guided = True
            await self._send_message(
                "status",
                f"PI-guided literature search for {n_clusters} clusters in {organism} {tissue}...",
                AgentType.PI
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI query generation failed, falling back to Literature Agent: {e}")
    else:
        logger.info(
            f"[{self.session_id}] Literature mode is '{lit_mode}', "
            f"PI will guide matching only (search handled by {lit_mode})"
        )

    if not pi_search_guided:
        await self._send_message(
            "status",
            f"Literature-based cluster annotation: {n_clusters} clusters in {organism} {tissue}...",
            AgentType.LITERATURE
        )

    # For deep-research modes, PI authors a focused annotation brief
    # (None → annotate_clusters falls back to its self-built focused topic).
    annotation_brief = None
    if lit_mode in ("gemini_deep_research", "deep_research"):
        annotation_brief = await self._generate_annotation_brief(
            cluster_markers, organism, tissue, state
        )

    # Step 1: Literature Agent executes search
    try:
        if pi_search_guided:
            result = await self.literature_agent.annotate_clusters(
                cluster_markers=cluster_markers,
                organism=organism,
                tissue=tissue,
                state=state,
                pi_initial_query=pi_query,
                pi_evaluate_fn=pi_evaluate,
                skip_matching=True,
            )
        elif lit_mode == "pubmed_llm":
            result = await self.literature_agent.annotate_clusters(
                cluster_markers=cluster_markers,
                organism=organism,
                tissue=tissue,
                state=state
            )
        else:
            result = await self.literature_agent.annotate_clusters(
                cluster_markers=cluster_markers,
                organism=organism,
                tissue=tissue,
                state=state,
                skip_matching=True,
                pi_research_brief=annotation_brief,
            )
    except Exception as e:
        error_msg = f"Cluster annotation failed: {str(e)}"
        logger.error(f"[{self.session_id}] {error_msg}")
        await self._send_message("error", error_msg, AgentType.LITERATURE)
        result = self._build_fallback_annotations(cluster_markers, e)

    return {
        "_continue": True, "state": state, "n_clusters": n_clusters,
        "result": result, "annotation_brief": annotation_brief,
    }

async def match_annotations(
    self, ctx, cluster_markers, tissue, organism, database_priors=None,
):
    state = ctx["state"]
    n_clusters = ctx["n_clusters"]
    result = ctx["result"]
    annotation_brief = ctx["annotation_brief"]
    # Step 2: PI Agent matches clusters to cell types
    if result.get("status") == "literature_only":
        literature_report = result.get("literature_report", "")
        await self._send_message(
            "status",
            f"PI is matching {n_clusters} clusters to cell types using literature...",
            AgentType.PI
        )
        try:
            matching_result = await self.pi_agent.annotate_clusters_from_literature(
                cluster_markers=cluster_markers,
                organism=organism,
                tissue=tissue,
                literature_report=literature_report,
                triage=result.get("triage", {}),
                focused_evidence=result.get("focused_evidence", {}),
                database_priors=database_priors or {},
            )
            result["status"] = "completed"
            result["annotations"] = matching_result.get("annotations", {})
            result["tissue_summary"] = matching_result.get("tissue_summary", "")
        except Exception as e:
            logger.warning(
                f"[{self.session_id}] PI matching failed, falling back to Literature Agent: {e}"
            )
            try:
                result = await self.literature_agent.annotate_clusters(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    state=state,
                    pi_research_brief=annotation_brief,
                )
            except Exception as e2:
                logger.error(f"[{self.session_id}] Literature fallback also failed: {e2}")
                result = self._build_fallback_annotations(cluster_markers, e2)

    annotations = result.get("annotations", {})
    n_annotated = sum(
        1 for v in annotations.values()
        if isinstance(v, dict) and v.get("cell_type", "Unknown") != "Unknown"
    )
    await self._send_message(
        "status",
        f"Cluster annotation complete: {n_annotated}/{n_clusters} clusters identified",
        AgentType.LITERATURE
    )

    ctx["annotations"] = annotations

async def review_and_confirm_annotations(
    self, ctx, cluster_markers, tissue, organism, analysis_log=None,
    database_priors=None,
):
    state = ctx["state"]
    result = ctx["result"]
    annotations = ctx["annotations"]
    # --- PI review + user confirmation ---
    if not (state.annotation_state and state.annotation_state.confirmed):
        for cid, info in annotations.items():
            if isinstance(info, dict) and cid in cluster_markers:
                existing = info.get("key_markers", [])
                if len(existing) < 10:
                    seen = set(existing)
                    supplemented = list(existing)
                    for gene in cluster_markers[cid]:
                        if gene not in seen:
                            supplemented.append(gene)
                            seen.add(gene)
                        if len(supplemented) >= 10:
                            break
                    info["key_markers"] = supplemented

        await self._send_message(
            "status",
            "PI is reviewing cluster annotations and assigning confidence scores...",
            AgentType.PI
        )
        try:
            pi_review = await self.pi_agent.review_cluster_annotations(
                state, annotations, tissue, organism, triage=result.get("triage", {})
            )
            for cid, review in pi_review.items():
                if cid in annotations and isinstance(review, dict):
                    annotations[cid].update(review)
            result["annotations"] = annotations
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI annotation review failed: {e}")

        # Annotation validation round: data-grounded checks + PI revision before user sees them.
        validation_meta = self._empty_validation_meta("not_run")
        try:
            annotations, validation_meta = await self._run_annotation_validation_round(
                state, annotations, cluster_markers, tissue, organism,
                triage=result.get("triage", {}), database_priors=database_priors,
                focused_evidence=result.get("focused_evidence", {}),
                literature_report=result.get("literature_report", ""),
            )
            result["annotations"] = annotations
        except Exception as e:
            logger.warning(f"[{self.session_id}] annotation validation round failed (non-fatal): {e}")
            validation_meta = self._empty_validation_meta(
                "failed", f"Annotation validation failed: {e}"
            )

        try:
            proportion_meta = await self.bioinformatics_agent.compute_cluster_group_proportions(state)
        except Exception as e:
            logger.warning(f"[{self.session_id}] cluster group proportion calculation failed (non-fatal): {e}")
            proportion_meta = {"group_labels": [], "proportions": {}, "column_used": None, "dimension": "group"}

        cluster_anns = self._build_cluster_annotations(annotations)
        self._apply_group_proportions(cluster_anns, proportion_meta)

        if state.annotation_state and state.annotation_state.confirmed:
            logger.info(f"[{self.session_id}] Annotation already confirmed, applying overrides directly")
            for ca in state.annotation_state.cluster_annotations:
                if ca.user_override and ca.cluster_id in annotations:
                    annotations[ca.cluster_id]["cell_type"] = ca.user_override
                    logger.info(
                        f"[{self.session_id}] User override (restored): cluster {ca.cluster_id} "
                        f"→ {ca.user_override}"
                    )
            result["annotations"] = annotations
            await self._send_message(
                "status",
                "Annotation confirmed (restored), continuing analysis...",
                AgentType.LITERATURE
            )
            return result

        original_log = state.analysis_log
        if analysis_log:
            state.analysis_log = analysis_log

        try:
            analysis_summary = self._build_annotation_context_report(state, self._current_graph_state)

            from ...services.report import generate_annotation_context_report_html
            try:
                html_path = generate_annotation_context_report_html(state, self._current_graph_state, cluster_anns)
                state.annotation_report_html_path = html_path
                report_html_url = f"/api/session/{self.session_id}/annotation-report/html"
            except Exception as e:
                logger.warning(f"[{self.session_id}] Annotation context report generation failed: {e}")
                report_html_url = None

            await self._build_and_save_detailed_segment(
                state, "pre_annotation",
                "Analysis Summary Before Cell Annotation",
                graph_state=self._current_graph_state,
            )
            detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if state.detailed_report_html_path else None
        finally:
            state.analysis_log = original_log

        ann_state = AnnotationConfirmationState(
            cluster_annotations=cluster_anns,
            tissue=tissue,
            organism=organism,
            tissue_summary=result.get("tissue_summary", ""),
            analysis_summary=analysis_summary,
            pending=True,
            proportion_group_labels=proportion_meta.get("group_labels", []),
            proportion_dimension=proportion_meta.get("dimension", "group"),
            **validation_meta,
        )
        state.annotation_state = ann_state


        # Step 3: Annotation confirmation loop
        original_status = state.status
        while True:
            self._annotation_event.clear()

            await self._send_message(
                "annotation_confirmation",
                self._build_annotation_confirmation_payload(
                    state, cluster_anns, tissue, organism,
                    result.get("tissue_summary", ""), analysis_summary,
                    report_html_url, detailed_report_url),
                AgentType.LITERATURE,
            )

            self._set_status(state, TaskStatus.WAITING_ANNOTATION)
            await self._send_state_sync(state, "annotation_confirmation")

            signaled = await wait_with_skip_time(
                self._annotation_event,
                state.round_wait_seconds,
                self.session_id,
                "annotation_confirmation",
            )
            if not signaled:
                # Auto-skip: accept PI annotations as-is, mark as confirmed
                if state.annotation_state:
                    state.annotation_state.confirmed = True
                    state.annotation_state.pending = False
                    state.annotation_record = build_annotation_record(state, auto_confirmed=True)
                state.status = original_status
                await self._send_state_sync(state, "annotation_auto_skipped")
                break

            if self._supplementary_queue and self._supplementary_queue[0][1] == "annotation":
                req, _target = self._supplementary_queue.popleft()
                state.annotation_supplementary_requests.append(req)
                state.status = TaskStatus.RUNNING
                await self._send_state_sync(state, "supplementary_running")

                await self._run_supplementary_analysis(state, req, "annotation")

                analysis_summary = self._build_annotation_context_report(state, self._current_graph_state)

                try:
                    pi_review = await self.pi_agent.review_cluster_annotations(
                        state, annotations, tissue, organism, triage=result.get("triage", {})
                    )
                    for cid, review in pi_review.items():
                        if cid in annotations and isinstance(review, dict):
                            annotations[cid].update(review)
                    result["annotations"] = annotations
                except Exception as e:
                    logger.warning(f"[{self.session_id}] PI re-review after supplementary failed: {e}")

                # Supplementary re-review changed labels -> re-validate (force past idempotency).
                try:
                    annotations, validation_meta = await self._run_annotation_validation_round(
                        state, annotations, cluster_markers, tissue, organism,
                        triage=result.get("triage", {}), database_priors=database_priors,
                        focused_evidence=result.get("focused_evidence", {}),
                        literature_report=result.get("literature_report", ""),
                        force=True,
                    )
                    result["annotations"] = annotations
                    cluster_anns = self._build_cluster_annotations(annotations)
                except Exception as e:
                    logger.warning(f"[{self.session_id}] post-supplementary re-validation failed: {e}")
                    validation_meta = self._empty_validation_meta(
                        "failed", f"Post-supplementary annotation validation failed: {e}"
                    )

                try:
                    proportion_meta = await self.bioinformatics_agent.compute_cluster_group_proportions(state)
                except Exception as e:
                    logger.warning(f"[{self.session_id}] post-supplementary proportion calculation failed (non-fatal): {e}")
                    proportion_meta = {"group_labels": [], "proportions": {}, "column_used": None, "dimension": "group"}
                self._apply_group_proportions(cluster_anns, proportion_meta)

                try:
                    html_path = generate_annotation_context_report_html(state, self._current_graph_state, cluster_anns)
                    state.annotation_report_html_path = html_path
                    report_html_url = f"/api/session/{self.session_id}/annotation-report/html"
                except Exception as e:
                    logger.warning(f"[{self.session_id}] Report re-generation failed: {e}")

                detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if state.detailed_report_html_path else None

                ann_state = AnnotationConfirmationState(
                    cluster_annotations=cluster_anns,
                    tissue=tissue,
                    organism=organism,
                    tissue_summary=result.get("tissue_summary", ""),
                    analysis_summary=analysis_summary,
                    pending=True,
                    proportion_group_labels=proportion_meta.get("group_labels", []),
                    proportion_dimension=proportion_meta.get("dimension", "group"),
                    **validation_meta,
                )
                state.annotation_state = ann_state
                continue
            else:
                logger.info(f"[{self.session_id}] Annotation confirmation received")
                if state.status == TaskStatus.PAUSED:
                    logger.info(f"[{self.session_id}] User paused during annotation wait, returning early")
                    return result
                state.status = original_status
                await self._send_state_sync(state, "annotation_resumed")
                break

        # Step 4: Apply user overrides
        confirmed = state.annotation_state
        if confirmed and confirmed.confirmed:
            for ca in confirmed.cluster_annotations:
                if ca.user_override and ca.cluster_id in annotations:
                    annotations[ca.cluster_id]["cell_type"] = ca.user_override
                    logger.info(
                        f"[{self.session_id}] User override: cluster {ca.cluster_id} "
                        f"→ {ca.user_override}"
                    )
            result["annotations"] = annotations

        await self._send_message(
            "status",
            "Annotation confirmed, continuing analysis...",
            AgentType.LITERATURE
        )

    return result


async def annotate_clusters_callback(
    self, cluster_markers, tissue, organism, analysis_log=None,
    database_priors=None,
):
    ctx = await begin_annotation(
        self, cluster_markers, tissue, organism, analysis_log, database_priors
    )
    if "_continue" not in ctx:
        return ctx
    await match_annotations(
        self, ctx, cluster_markers, tissue, organism, database_priors
    )
    return await review_and_confirm_annotations(
        self, ctx, cluster_markers, tissue, organism, analysis_log, database_priors
    )
