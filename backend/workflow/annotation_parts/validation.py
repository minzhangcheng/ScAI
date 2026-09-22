"""Annotation validation trigger, execution, and revision phases."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

def _validation_enabled(self) -> bool:
    """Whether the annotation validation round is enabled (config flag; default True)."""
    try:
        from ...config import default_config
        cfg = getattr(self, "config", None) or default_config
        return bool(getattr(cfg, "enable_annotation_validation_round", True))
    except Exception:
        return True

def _validation_limits(self):
    """(max_iterations, timeout_seconds) from config; defaults 5 / 600.
    Clamped defensively (iter in [1,8], timeout >= 600) so a stray config can't
    make validation time out instantly or loop unboundedly."""
    try:
        from ...config import default_config
        cfg = getattr(self, "config", None) or default_config
        it = int(getattr(cfg, "annotation_validation_max_iterations", 5))
        to = int(getattr(cfg, "annotation_validation_timeout_seconds", 600))
    except Exception:
        return (5, 600)
    from backend.utils.llm_client_parts.retry import llm_timeout_seconds
    return (max(1, min(it, 8)), llm_timeout_seconds(to))

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
    annotation_source: Optional[Dict[str, Any]] = None,
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
        context = (await self.bioinformatics_agent.resolve_analysis_source_metadata(annotation_source)
                   if annotation_source else await self.bioinformatics_agent.introspect_validation_metadata(None))
    except Exception as e:
        logger.warning(f"[{self.session_id}] validation introspection failed: {e}")
        context = {}
    cluster_key = context.get("cluster_key") or ""

    # 2) Idempotency guard (reads PREVIOUS pending state, e.g. after snapshot restore).
    attempt_id = compute_validation_attempt_id(
        getattr(state, "current_round", 0), tissue, organism, cluster_key, cluster_markers,
        source_identity={key: context[key] for key in ("partition_digest", "preparation_id") if context.get(key)})
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
        from ...config import get_session_workspace
        from ..annotation_validation import resolve_artifact_path
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
