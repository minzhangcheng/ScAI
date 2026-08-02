"""Cluster annotation helpers and HITL confirmation flow."""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models.schemas import (
    AgentType, AnalysisState, AnnotationConfirmationState,
    ClusterAnnotation, TaskStatus,
)
from ..services.interaction_record_capture import build_annotation_record
from .helpers import safe_pi_brief, wait_with_skip_time
from .annotation_validation import compute_validation_attempt_id, should_run_validation

if TYPE_CHECKING:
    from .state import WorkflowState

logger = logging.getLogger(__name__)


class AnnotationMixin:
    """Mixin providing cluster-annotation methods. Composed into SingleCellAnalysisWorkflow."""

    @staticmethod
    def _build_fallback_annotations(
        cluster_markers: Dict[str, Any], error: Exception
    ) -> Dict[str, Any]:
        """Build fallback annotations when annotation service fails."""
        fallback_annotations = {}
        for cid in cluster_markers:
            fallback_annotations[str(cid)] = {
                "cell_type": "Unknown",
                "confidence": "low",
                "key_markers": (
                    list(cluster_markers[cid])[:10]
                    if isinstance(cluster_markers[cid], list)
                    else []
                ),
                "evidence": "",
                "low_confidence_reason": f"Annotation service failed: {str(error)}",
            }
        return {"annotations": fallback_annotations, "tissue_summary": ""}

    def _build_annotation_context_report(self, state: AnalysisState, graph_state: Optional["WorkflowState"]) -> str:
        """Build a comprehensive markdown summary of current analysis context for cell annotation."""
        parts = ["### Analysis Context Report\n"]

        # 1. Experimental Design
        parts.append("#### Experimental Design")
        parts.append(state.build_experimental_design_summary())
        parts.append("")

        # 2. Biochemistry & Pathology findings (from graph state if available)
        if graph_state:
            biochem = graph_state.get("biochemistry_result")
            if biochem and biochem.get("findings"):
                parts.append("#### Biochemistry Findings")
                for f in biochem["findings"]:
                    parts.append(f"- {f}")
                parts.append("")

            patho = graph_state.get("pathology_result")
            if patho and patho.get("findings"):
                parts.append("#### Pathology Findings")
                for f in patho["findings"]:
                    parts.append(f"- {f}")
                parts.append("")

        # 3. Previous Rounds' Key Findings
        if state.round_results and len(state.round_results) > 0:
            parts.append("#### Previous Analysis Highlights")
            last_round = state.round_results[-1]
            if last_round.pi_summary:
                parts.append(last_round.pi_summary)
            parts.append("")

        # 4. Literature Searches (current round)
        lit_steps = [step for step in state.analysis_log if step.step_type == "search_literature"]
        if lit_steps:
            parts.append("#### Recent Literature Research")
            for step in lit_steps[-3:]:  # Last 3 searches
                topic = step.literature_topic or "Unknown topic"
                result = step.literature_result or "No findings recorded"
                parts.append(f"**Topic:** {topic}")
                parts.append(f"**Findings:** {result[:300]}...\n")

        return "\n".join(parts)

    @staticmethod
    def _build_cluster_annotations(annotations: Dict) -> List[ClusterAnnotation]:
        """Build ClusterAnnotation list from annotation dict."""
        return [
            ClusterAnnotation(
                cluster_id=cid,
                cell_type=info.get("cell_type", "Unknown"),
                subtypes=info.get("subtypes", []),
                key_markers=info.get("key_markers", []),
                confidence=info.get("confidence", "low"),
                evidence=info.get("evidence", ""),
                n_cells=max(0, int(info.get("n_cells", 0) or 0)),
                n_cells_known=(
                    bool(info.get("n_cells_known"))
                    if "n_cells_known" in info
                    else "n_cells" in info and info.get("n_cells") is not None
                ),
                n_cells_source=(
                    str(info.get("n_cells_source") or "unknown")
                    if "n_cells_source" in info
                    else (
                        "annotation_payload"
                        if "n_cells" in info and info.get("n_cells") is not None
                        else "unknown"
                    )
                ),
                group_proportions=info.get("group_proportions", {}) or {},
                alternative_cell_types=info.get("alternative_cell_types", []),
                low_confidence_reason=info.get("low_confidence_reason", ""),
                requires_reclustering=info.get("requires_reclustering", False),
                reclustering_reason=info.get("reclustering_reason", ""),
                suggested_reclustering_scope=info.get("suggested_reclustering_scope"),
            )
            for cid, info in annotations.items()
            if isinstance(info, dict)
        ]

    @staticmethod
    def _apply_group_proportions(
        cluster_anns: List[ClusterAnnotation],
        proportion_meta: Optional[Dict[str, Any]],
    ) -> None:
        proportions = (proportion_meta or {}).get("proportions") or {}
        counts = (proportion_meta or {}).get("counts") or {}
        for ca in cluster_anns:
            ca.group_proportions = dict(proportions.get(str(ca.cluster_id), {}) or {})
            count = counts.get(str(ca.cluster_id))
            if count is not None:
                ca.n_cells = max(0, int(count))
                ca.n_cells_known = True
                ca.n_cells_source = "cluster_group_proportions"

    @staticmethod
    def _annotations_from_cluster_annotations(cluster_annotations) -> Dict[str, Any]:
        """Rebuild an annotations dict from a saved ClusterAnnotation list (the
        inverse of _build_cluster_annotations), preserving validation revisions."""
        out: Dict[str, Any] = {}
        for ca in (cluster_annotations or []):
            out[ca.cluster_id] = {
                "cell_type": ca.user_override or ca.cell_type,
                "subtypes": list(ca.subtypes),
                "key_markers": list(ca.key_markers),
                "confidence": ca.confidence,
                "evidence": ca.evidence,
                "n_cells": ca.n_cells,
                "n_cells_known": ca.n_cells_known,
                "n_cells_source": ca.n_cells_source,
                "group_proportions": dict(ca.group_proportions or {}),
                "alternative_cell_types": list(ca.alternative_cell_types),
                "low_confidence_reason": ca.low_confidence_reason,
                "requires_reclustering": ca.requires_reclustering,
                "reclustering_reason": ca.reclustering_reason,
                "suggested_reclustering_scope": ca.suggested_reclustering_scope,
            }
        return out

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

    async def _annotate_clusters_callback(
        self,
        cluster_markers: Dict,
        tissue: str,
        organism: str,
        analysis_log: Optional[List] = None,
        database_priors: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Callback for bioinformatics agent to annotate all clusters via literature.

        When annotations have not yet been confirmed by the user, this callback additionally:
        1. Has PI Agent review annotations and assign confidence scores
        2. Blocks execution until the user confirms annotations via frontend
        """
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

                from ..services.report import generate_annotation_context_report_html
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

            await self._create_auto_snapshot(state, "pre_annotation")

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
                    safe_point_hook=getattr(self, "await_restore_safe_point", None),
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

    def _validation_enabled(self) -> bool:
        """Whether the annotation validation round is enabled (config flag; default True)."""
        try:
            from ..config import default_config
            cfg = getattr(self, "config", None) or default_config
            return bool(getattr(cfg, "enable_annotation_validation_round", True))
        except Exception:
            return True

    @staticmethod
    def _empty_validation_meta(status: str, summary: str = "", attempt_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "validation_attempt_id": attempt_id,
            "validation_status": status,
            "validation_plan": None,
            "validation_result": None,
            "validation_summary": summary or None,
            "validation_text_summary": None,
            "reclustering_recommendations": None,
        }

    @staticmethod
    def _build_validation_text_summary(val_result: Dict[str, Any]) -> str:
        """Compact, vision-free per-cluster verdict summary."""
        lines = [val_result.get("summary", "")]
        for cid, cr in (val_result.get("cluster_results") or {}).items():
            if isinstance(cr, dict):
                lines.append(f"- Cluster {cid}: {cr.get('support_level','?')}; "
                             f"action={cr.get('suggested_action','')}; qc={cr.get('qc_flags', [])}")
        return "\n".join(l for l in lines if l)

    @staticmethod
    def _build_annotation_confirmation_payload(
        state: AnalysisState, cluster_anns: List[ClusterAnnotation], tissue: str, organism: str,
        tissue_summary: str, analysis_summary: str,
        report_html_url: Optional[str], detailed_report_url: Optional[str],
    ) -> Dict[str, Any]:
        """Build the annotation_confirmation payload, including optional
        validation and proportion fields read from the pending annotation state."""
        vstate = state.annotation_state
        return {
            "annotations": [a.model_dump(mode="json") for a in cluster_anns],
            "tissue": tissue,
            "organism": organism,
            "tissue_summary": tissue_summary,
            "analysis_summary": analysis_summary,
            "report_html_url": report_html_url,
            "detailed_report_url": detailed_report_url,
            "proportion_group_labels": getattr(vstate, "proportion_group_labels", []),
            "proportion_dimension": getattr(vstate, "proportion_dimension", "group"),
            "validation_status": getattr(vstate, "validation_status", None),
            "validation_summary": getattr(vstate, "validation_summary", None),
            "validation_text_summary": getattr(vstate, "validation_text_summary", None),
            "validation_plan": getattr(vstate, "validation_plan", None),
            "validation_result": getattr(vstate, "validation_result", None),
            "reclustering_recommendations": getattr(vstate, "reclustering_recommendations", None),
        }

    def _validation_limits(self):
        """(max_iterations, timeout_seconds) from config; defaults 5 / 600.
        Clamped defensively (iter in [1,8], timeout >= 1) so a stray config can't
        make validation time out instantly or loop unboundedly."""
        try:
            from ..config import default_config
            cfg = getattr(self, "config", None) or default_config
            it = int(getattr(cfg, "annotation_validation_max_iterations", 5))
            to = int(getattr(cfg, "annotation_validation_timeout_seconds", 600))
        except Exception:
            return (5, 600)
        return (max(1, min(it, 8)), max(1, to))

    async def _run_annotation_validation_round(
        self,
        state: AnalysisState,
        annotations: Dict[str, Any],
        cluster_markers: Dict[str, Any],
        tissue: str,
        organism: str,
        triage: Optional[Dict[str, Any]] = None,
        database_priors: Optional[Dict[str, Any]] = None,
        focused_evidence: Optional[Dict[str, Any]] = None,
        literature_report: str = "",
        force: bool = False,
    ):
        """Run the gated annotation validation round.

        Returns (annotations, validation_meta). Never raises into the caller and
        never writes state.annotation_state (the caller folds validation_meta into
        the AnnotationConfirmationState it builds). `force=True` bypasses the
        idempotency reuse (used after supplementary analysis re-reviews labels)."""
        triage = triage or {}; database_priors = database_priors or {}; focused_evidence = focused_evidence or {}

        if not self._validation_enabled():
            return annotations, self._empty_validation_meta(
                "skipped", "Annotation validation is disabled by configuration.")

        # 1) Metadata introspection (also yields cluster_key for the attempt id).
        try:
            context = await self.bioinformatics_agent.introspect_validation_metadata(None)
        except Exception as e:
            logger.warning(f"[{self.session_id}] validation introspection failed: {e}")
            context = {}
        cluster_key = context.get("cluster_key") or ""

        # 2) Idempotency guard (reads PREVIOUS pending state, e.g. after snapshot restore).
        attempt_id = compute_validation_attempt_id(
            getattr(state, "current_round", 0), tissue, organism, cluster_key, cluster_markers)
        prev = state.annotation_state
        if (not force and prev is not None and getattr(prev, "validation_attempt_id", None) == attempt_id
                and getattr(prev, "validation_status", None) in ("completed", "failed", "skipped", "partial")):
            logger.info(f"[{self.session_id}] reusing validation result for attempt {attempt_id[:8]}")
            reused = self._annotations_from_cluster_annotations(getattr(prev, "cluster_annotations", None))
            return (reused or annotations), {
                "validation_attempt_id": attempt_id,
                "validation_status": prev.validation_status,
                "validation_plan": prev.validation_plan,
                "validation_result": prev.validation_result,
                "validation_summary": prev.validation_summary,
                "validation_text_summary": prev.validation_text_summary,
                "reclustering_recommendations": prev.reclustering_recommendations,
            }

        # 3) Trigger gate.
        if not should_run_validation(annotations, triage):
            return annotations, self._empty_validation_meta(
                "skipped", "All annotations high-confidence; no validation concerns detected.", attempt_id)
        if not context.get("var"):
            return annotations, self._empty_validation_meta(
                "skipped", "No active analysis object available for validation.", attempt_id)

        # 4) PI validation questions.
        await self._send_message("status", "PI is planning annotation validation checks...", AgentType.PI)
        try:
            plan = await self.pi_agent.generate_annotation_validation_questions(
                cluster_markers=cluster_markers, annotations=annotations, tissue=tissue, organism=organism,
                triage=triage, focused_evidence=focused_evidence, database_priors=database_priors,
                metadata_context=context) or {}
        except Exception as e:
            logger.warning(f"[{self.session_id}] validation planning failed: {e}")
            return annotations, self._empty_validation_meta(
                "failed", "Validation planning failed; using preliminary annotations.", attempt_id)

        if not plan.get("tasks"):
            meta = self._empty_validation_meta(
                "skipped", "PI determined no validation checks were necessary.", attempt_id)
            meta["validation_plan"] = plan
            return annotations, meta

        # 5) Restricted bioinformatics execution.
        await self._send_message("status", "Bioinformatics agent is validating preliminary annotations...",
                                 AgentType.BIOINFORMATICS)
        max_iter, timeout_s = self._validation_limits()
        try:
            val_result = await self.bioinformatics_agent.validate_annotations_with_plan(
                validation_plan=plan, annotations=annotations, cluster_markers=cluster_markers,
                tissue=tissue, organism=organism, context=context,
                max_iterations=max_iter, timeout_seconds=timeout_s)
        except Exception as e:
            logger.warning(f"[{self.session_id}] validation execution failed: {e}")
            val_result = {"status": "failed", "summary": str(e), "cluster_results": {},
                          "suggested_annotation_changes": {}, "artifacts": [], "warnings": [str(e)]}

        # 5b) Resolve artifact paths to session-rooted ABSOLUTE paths. The /api/file
        # route validates ownership by a 'sessions/<sid>/' path segment, so a bare
        # 'workspace/...' relative path would be rejected. The executor saves under
        # the session workspace; rebuild the absolute path here.
        try:
            from ..config import get_session_workspace
            from .annotation_validation import resolve_artifact_path
            ws = get_session_workspace(self.session_id)  # .../sessions/<sid>/workspace
            for art in (val_result.get("artifacts") or []):
                if isinstance(art, dict) and art.get("path"):
                    art["path"] = resolve_artifact_path(art["path"], str(ws))
        except Exception as e:
            logger.warning(f"[{self.session_id}] artifact path resolution failed: {e}")

        # 6) PI revision (PI sees even failed results). Build a short literature
        # context from focused evidence + the literature report so revision is
        # literature-grounded per spec, without dumping the full report.
        lit_bits = []
        if focused_evidence:
            import json as _jl
            lit_bits.append("Focused evidence: " + _jl.dumps(focused_evidence, ensure_ascii=False)[:1200])
        if literature_report:
            lit_bits.append("Literature report: " + literature_report[:1500])
        literature_context = "\n".join(lit_bits)

        await self._send_message("status", "PI is revising annotations using validation results...", AgentType.PI)
        try:
            revision = await self.pi_agent.revise_annotations_after_validation(
                annotations=annotations, validation_plan=plan, validation_result=val_result,
                literature_context=literature_context, tissue=tissue, organism=organism)
            revised = revision.get("annotations", annotations)
            summary_for_user = revision.get("validation_summary_for_user", "")
        except Exception as e:
            logger.warning(f"[{self.session_id}] validation revision failed: {e}")
            revised = annotations
            summary_for_user = "Validation revision failed; using preliminary annotations."

        # 7) Reclustering recommendations + write per-cluster flags into annotations dict.
        recs = []
        for cid, cr in (val_result.get("cluster_results") or {}).items():
            if isinstance(cr, dict) and cr.get("requires_reclustering"):
                recs.append({"cluster_id": cid, "requires_reclustering": True,
                             "reclustering_reason": cr.get("reclustering_reason", ""),
                             "suggested_reclustering_scope": cr.get("suggested_reclustering_scope")})
                if cid in revised and isinstance(revised[cid], dict):
                    revised[cid]["requires_reclustering"] = True
                    revised[cid]["reclustering_reason"] = cr.get("reclustering_reason", "")
                    revised[cid]["suggested_reclustering_scope"] = cr.get("suggested_reclustering_scope")

        meta = {
            "validation_attempt_id": attempt_id,
            "validation_status": val_result.get("status", "failed"),
            "validation_plan": plan,
            "validation_result": val_result,
            "validation_summary": summary_for_user or None,
            "validation_text_summary": self._build_validation_text_summary(val_result),
            "reclustering_recommendations": recs or None,
        }
        return revised, meta
