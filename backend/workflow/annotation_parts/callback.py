"""Annotation literature, PI matching, and HITL callback phases."""
from __future__ import annotations

import asyncio
import time

from ...utils.llm_client import UpstreamExhaustedError
from ...agents.literature.research import literature_technical_failure

#: What the model reads when a restored session resumes into this tool call.
#: The call it made has no result yet -- the user is looking at the dialog --
#: so the checkpoint pairs it with this, and a resumed executor sees an
#: instruction rather than a truncated transcript. Calling annotate_clusters
#: again is cheap and deterministic: begin_annotation returns the confirmed
#: annotations straight from the state once the user has answered.
_ANNOTATION_RESUME_NOTE = (
    "[ANNOTATION AWAITING USER REVIEW] The cluster annotations were produced "
    "and handed to the user for confirmation. If you are reading this after a "
    "session was restored from this point, call annotate_clusters again with "
    "the same arguments: it returns the confirmed annotations directly. Do not "
    "re-run clustering or differential expression."
)


async def _capture_annotation_snapshot(self, state) -> None:
    """The restore point for the cell annotation decision. Never raises.

    Part B spec 1: this is the earliest decision an analysis makes and every
    later label depends on it, which is why it carries the highest retention
    rank. Taken before the dialog opens and before an unattended run accepts
    the annotations on the user's behalf -- with ``round_wait_seconds = 0``
    the PI decides, and at this layer that is the same decision made by a
    different party.
    """
    from backend.constants import GATE_ANNOTATION_CONFIRMATION
    from backend.services.round_snapshots.writer import capture_decision_snapshot

    pending = getattr(
        getattr(self, "bioinformatics_agent", None), "pending_tool_checkpoint", None,
    )
    prepare = None
    if pending is not None:
        async def prepare() -> None:
            await pending.checkpoint_now(
                "pre_annotation_confirmation", result_text=_ANNOTATION_RESUME_NOTE,
            )

    await capture_decision_snapshot(
        self,
        state,
        gate=GATE_ANNOTATION_CONFIRMATION,
        workflow_state=getattr(self, "_current_graph_state", None),
        prepare=prepare,
    )


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
                "citations": [c.model_dump() for c in (ca.citations or [])],
                "citation_status": ca.citation_status,
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
    search_deadline = time.monotonic() + float(
        getattr(self.literature_agent, "search_wall_clock_seconds", 600) or 600
    )

    # Step 0a: PI generates search query (only for pubmed_llm mode)
    if lit_mode == "pubmed_llm":
        try:
            pi_query = await asyncio.wait_for(
                self.pi_agent.generate_annotation_search_query(
                    cluster_markers=cluster_markers,
                    organism=organism,
                    tissue=tissue,
                    state=state,
                ),
                timeout=max(0.001, search_deadline - time.monotonic()),
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
        except UpstreamExhaustedError:
            raise
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
                search_deadline=search_deadline,
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
                pi_plan_fn=self.pi_agent.plan_literature_search,
                pi_review_fn=self.pi_agent.review_literature_search,
            )
    except UpstreamExhaustedError:
        raise
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
            result["status"] = (
                "partial" if literature_technical_failure(result) else "completed"
            )
            result["matching_status"] = "completed"
            result["annotations"] = matching_result.get("annotations", {})
            result["tissue_summary"] = matching_result.get("tissue_summary", "")
        except UpstreamExhaustedError:
            raise
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
                    pi_plan_fn=self.pi_agent.plan_literature_search,
                    pi_review_fn=self.pi_agent.review_literature_search,
                )
            except UpstreamExhaustedError:
                raise
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

    ctx["result"] = result
    ctx["annotations"] = annotations

async def review_and_confirm_annotations(
    self, ctx, cluster_markers, tissue, organism, analysis_log=None,
    database_priors=None, annotation_source=None,
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
        except UpstreamExhaustedError:
            raise
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

            from ...services.report.stage_reviews import build_stage_review
            from ...agents.bioinformatics.annotation_preview import render_candidate_annotation_preview
            preview = await render_candidate_annotation_preview(
                getattr(self.bioinformatics_agent, "sandbox", None), annotation_source, cluster_anns,
            )
            report_html_url = await build_stage_review(
                state, self.pi_agent, "stage1", summary=analysis_summary,
                annotations=cluster_anns, graph_state=self._current_graph_state, validation=validation_meta,
                annotation_preview=preview,
            )

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

            # Part B hook B, after the pending state is published and before
            # the wait: a snapshot taken any later could not restore into the
            # dialog, and one taken earlier would not carry it.
            await _capture_annotation_snapshot(self, state)

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
                except UpstreamExhaustedError:
                    raise
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

                preview = await render_candidate_annotation_preview(
                    getattr(self.bioinformatics_agent, "sandbox", None), annotation_source, cluster_anns,
                )
                report_html_url = await build_stage_review(
                    state, self.pi_agent, "stage1", summary=analysis_summary,
                    annotations=cluster_anns, graph_state=self._current_graph_state, validation=validation_meta,
                    annotation_preview=preview,
                )

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
    database_priors=None, annotation_source=None,
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
        self, ctx, cluster_markers, tissue, organism, analysis_log, database_priors, annotation_source
    )
