"""Cell-type annotation review, validation questions, and literature-backed annotation.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
import logging


# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ..literature.shared import ANNOTATION_REVIEW_MAX_CHARS, normalize_candidate_celltype

from .shared import (  # noqa: F401
    _CONTROL_SIGNAL_RESULTS,
    _BRIEF_PREFERRED_TYPES,
    _BRIEF_QC_MARKERS,
    _DIRECTION_CANDIDATE_LIST_FIELDS,
    _normalize_supporting_data,
    _normalize_direction_candidate,
    _normalize_integration_scored_candidates,
    _bounded_brief_fallback,
    _format_database_priors_block,
    _build_default_bio_tasks,
)


# Articles shown to the PI when it judges whether the retrieved literature is
# sufficient. Earlier windows (15 articles, 200-char abstracts, always the first
# 15 retrieved) meant round 2+ evaluations never saw what the refined query found.
_EVAL_MAX_ARTICLES = 25
_EVAL_ABSTRACT_CHARS = 300


def _evidence_articles(fe: Dict[str, Any], limit: int) -> str:
    """Render this cluster's articles, identifier first.

    The PMID used to be omitted, so the matcher was shown titles and abstracts
    it could not cite: measured on one run, zero identifiers reached the prompt
    and none of the fifteen clusters carried a citation. A model cannot name
    what it was never given.
    """
    return "\n".join(
        f"- PMID {a.get('pmid', '?')}: {a.get('title', '')}: {(a.get('abstract') or '')[:300]}"
        for a in (fe.get("articles") or [])[:limit]
    )


def _shown_articles(fe: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """The article records _evidence_articles just rendered, same slice.

    Kept beside the renderer so the recorded pool cannot drift from what the
    prompt actually showed.
    """
    return [a for a in (fe.get("articles") or [])[:limit] if isinstance(a, dict)]


def _cluster_evidence_lines(cid: str, bucket: str, entry: Dict[str, Any],
                            fe: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Render one cluster's triage note and retrieved literature.

    Returns the prompt lines and the article records behind them, so the
    annotation can record what it was based on without re-deriving it.
    """
    lines: List[str] = []
    shown: List[Dict[str, Any]] = []
    if bucket == "context_dependent_state":
        lines.append(
            f"[Note: Cluster {cid} classified as context-dependent state — parent: "
            f"{entry.get('parent_celltype') or '?'}, driven by "
            f"{entry.get('context') or 'experimental context'}. "
            f"Annotate as a state variant of the parent, no literature deep-dive expected.]"
        )
    # A cluster in an unadjudicated bucket that the search never reached says
    # nothing; rendering it as "exhausted" would assert evidence scarcity that
    # was never measured.
    if not fe and bucket in ("covered", "context_dependent_state"):
        return lines, shown

    status = fe.get("status")
    candidate = normalize_candidate_celltype(
        entry.get("candidate_celltype") or entry.get("parent_celltype")
    ) or "marker-defined population"

    if status in ("strong", "limited"):
        tag = "" if status == "strong" else " (limited evidence)"
        lines.append(
            f"[Focused evidence for Cluster {cid} (candidate: {candidate}){tag}]\n"
            + _evidence_articles(fe, 5)
        )
        shown.extend(_shown_articles(fe, 5))
    elif status == "retrieved":
        lines.append(
            f"[Retrieved literature for Cluster {cid} (query subject: {candidate}) — "
            f"not relevance-adjudicated; judge for yourself whether each paper supports "
            f"this cluster's identity]\n" + _evidence_articles(fe, 3)
        )
        shown.extend(_shown_articles(fe, 3))
    elif status == "contradictory":
        if bucket == "suspected_contamination":
            lines.append(f"[Cluster {cid}: focused evidence supports suspected contamination "
                         f"or non-target origin]")
        else:
            lines.append(f"[Cluster {cid}: focused evidence contradicts candidate {candidate}; "
                         f"consider alternative cell identities]")
    else:
        reason = fe.get("exhaustion_reason")
        error = fe.get("error")
        if reason in ("search_failed", "judge_failed"):
            detail = f" Error: {error}" if error else ""
            lines.append(
                f"[Focused evidence unavailable for Cluster {cid} due to runtime failure "
                f"({reason}).{detail} Do not interpret as absence of literature.]"
            )
        elif reason == "no_pubmed_results":
            lines.append(
                f"[Focused search found no PubMed results for Cluster {cid} — this is evidence scarcity, "
                f"not a runtime failure.]"
            )
        elif reason == "empty_query":
            lines.append(
                f"[Focused search skipped for Cluster {cid} — no candidate term or marker query terms were available.]"
            )
        elif reason == "no_relevant_evidence":
            titles = "; ".join(a.get("title", "") for a in (fe.get("articles") or [])[:3])
            suffix = f" Retrieved articles were not relevant: {titles}" if titles else " Retrieved articles were not relevant."
            lines.append(f"[Focused evidence exhausted for Cluster {cid}.{suffix}]")
        else:
            lines.append(f"[Focused evidence exhausted for Cluster {cid} — no relevant literature found]")
    return lines, shown


def _render_cluster_evidence(
    cluster_ids, triage: Dict[str, Any], focused_evidence: Dict[str, Any]
) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    """Per-cluster evidence block, in cluster order, for the matching prompt.

    Every triaged cluster is rendered, including the covered ones: they used to
    carry no evidence of their own into the matcher, which then had only the
    pooled review to cite and frequently cited nothing.
    """
    triage = triage or {}
    focused_evidence = focused_evidence or {}

    by_cluster: Dict[str, Any] = {}
    for bucket in ("rare_unresolved", "suspected_contamination", "context_dependent_state"):
        for entry in triage.get(bucket) or []:
            if isinstance(entry, dict):
                by_cluster.setdefault(str(entry.get("cluster_id")), (bucket, entry))
    covered_names = triage.get("covered_celltypes") or {}
    for raw_cid in triage.get("covered") or []:
        cid = str(raw_cid)
        by_cluster.setdefault(
            cid, ("covered", {"cluster_id": cid, "candidate_celltype": covered_names.get(cid)})
        )

    ordered = [str(c) for c in (cluster_ids or [])]
    ordered += [cid for cid in by_cluster if cid not in set(ordered)]

    lines: List[str] = []
    pool: Dict[str, List[Dict[str, Any]]] = {}
    for cid in ordered:
        found = by_cluster.get(cid)
        if found is None:
            continue
        bucket, entry = found
        fe = focused_evidence.get(cid) or {}
        cluster_lines, shown = _cluster_evidence_lines(cid, bucket, entry, fe)
        lines.extend(cluster_lines)
        # An entry with an empty list still matters: it separates "the search
        # ran for this cluster and found nothing" from "this cluster was never
        # reached", which the citation status has to tell apart.
        pool[cid] = shown

    if not lines:
        return "", pool
    return (
        "\n\n=== PER-CLUSTER LITERATURE EVIDENCE ===\n"
        "This is the primary evidence for each cluster. The background review that follows is\n"
        "tissue-level context, not a substitute: where the two disagree about a cluster, the\n"
        "evidence here is the one tied to that cluster's own markers.\n"
        + "\n".join(lines) + "\n"
    ), pool


class AnnotationMixin:
    """Cell-type annotation review, validation questions, and literature-backed annotation."""

    async def review_cluster_annotations(
        self,
        state: AnalysisState,
        annotations: Dict[str, Any],
        tissue: str,
        organism: str,
        triage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        PI reviews Literature Agent's cluster annotations and assigns confidence scores.
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Reviewing cluster annotations for {organism} {tissue}")

        ann_lines = []
        for cid, info in annotations.items():
            if isinstance(info, dict):
                ct = info.get("cell_type", "Unknown")
                markers = ", ".join(info.get("key_markers", [])[:10])
                evidence = (info.get("evidence", ""))[:200]
                ann_lines.append(
                    f"- Cluster {cid}: {ct} | Markers: {markers} | Evidence: {evidence}"
                )
        ann_text = "\n".join(ann_lines) if ann_lines else "No annotations available"

        triage = triage or {}
        triage_lines = []
        for it in triage.get("suspected_contamination", []):
            triage_lines.append(f"- Cluster {it.get('cluster_id')}: suspected technical/contamination — "
                                 f"{it.get('reason','')} (mark low confidence; pass this reason through verbatim)")
        for it in triage.get("context_dependent_state", []):
            triage_lines.append(f"- Cluster {it.get('cluster_id')}: context-dependent state variant of "
                                 f"{it.get('parent_celltype') or '?'} (confidence low/medium)")
        triage_block = ("\n## Triage notes from search phase\n" + "\n".join(triage_lines)) if triage_lines else ""

        prompt = f"""You are reviewing cell type annotations generated by a literature search for {organism} {tissue}.

## Cluster Annotations from Literature Search
{ann_text}
{triage_block}

Biological judgement principles:
- Account for assay-specific capture biases (including differences between single-cell and
  single-nucleus data); do not infer a lineage from isolated assay-enriched genes.
- marker set coherence: require positive + negative markers + tissue plausibility + intra-cluster
  consistency; do not call a type on 1-2 markers alone.
- condition-skewed clusters are likely state variants, not new cell types.
- suspected technical/contamination clusters: keep low confidence and pass the provided reason through.
- a tissue-plausible population must NOT be down-graded merely for being rare.
- distinguish a stable lineage identity from reactive, transitional, or progenitor-like states using
  coherent positive and negative markers rather than a preselected cell-type rule.

For EACH cluster, evaluate:
1. Is the cell type annotation correct given the marker genes and tissue context?
2. Assign a confidence level:
   - "high": Markers strongly and specifically match the annotated cell type in this tissue
   - "medium": Markers partially match or could belong to a related cell type
   - "low": Markers are ambiguous, contradictory, or insufficient for confident annotation
3. For clusters with "medium" or "low" confidence, provide:
   - alternative_cell_types: other possible cell types these markers could indicate
   - low_confidence_reason: explanation of why confidence is not high

Reply in JSON format (include ALL clusters):
{{
    "<cluster_id>": {{
        "cell_type": "confirmed or corrected cell type",
        "confidence": "high/medium/low",
        "alternative_cell_types": ["Alternative Type 1", "Alternative Type 2"],
        "low_confidence_reason": "Reason for low/medium confidence (empty string if high)"
    }},
    ...
}}
"""

        response = await self.chat(prompt, temperature=0.2)
        result = self._safe_parse_json(response, default={})

        if not result or not isinstance(result, dict):
            return {}

        # 确定性后处理: 对 suspected_contamination 簇强制低置信 + reason 透传 + 候选入 alt
        for it in triage.get("suspected_contamination", []):
            cid = str(it.get("cluster_id"))
            if cid in result and isinstance(result[cid], dict):
                result[cid]["confidence"] = "low"
                reason = it.get("reason", "")
                if reason:
                    result[cid]["low_confidence_reason"] = reason
                cand = normalize_candidate_celltype(it.get("candidate_celltype"))
                if cand:
                    alts = result[cid].setdefault("alternative_cell_types", [])
                    if cand not in alts:
                        alts.append(cand)
        return result

    async def generate_annotation_validation_questions(
        self,
        cluster_markers: Dict[str, Any],
        annotations: Dict[str, Any],
        tissue: str,
        organism: str,
        triage: Optional[Dict[str, Any]] = None,
        focused_evidence: Optional[Dict[str, Any]] = None,
        database_priors: Optional[Dict[str, Any]] = None,
        metadata_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """PI produces biological validation QUESTIONS (not code). Returns {} on
        failure (caller treats as skip)."""
        self.refresh_runtime_config_if_needed()
        import json as _j
        triage = triage or {}; focused_evidence = focused_evidence or {}
        database_priors = database_priors or {}; metadata_context = metadata_context or {}

        ann_lines = []
        for cid, info in annotations.items():
            if isinstance(info, dict):
                ann_lines.append(
                    f"- Cluster {cid}: {info.get('cell_type','Unknown')} [conf={info.get('confidence','?')}] "
                    f"markers={', '.join(info.get('key_markers', [])[:8])} "
                    f"alt={info.get('alternative_cell_types', [])} reason={info.get('low_confidence_reason','')}")
        ann_text = "\n".join(ann_lines) or "No annotations."

        prompt = f"""You are the PI planning a focused annotation-VALIDATION round for {organism} {tissue}.
Raise biological validation QUESTIONS for clusters whose labels are uncertain, triage-flagged,
technical/QC-suspect, or possibly a state rather than a lineage. Do NOT write code; a bioinformatics
agent will translate your questions into concrete checks against real columns.

## Preliminary annotations
{ann_text}

## Triage from search phase
{_j.dumps(triage, ensure_ascii=False)[:1500]}

## Focused literature evidence
{_j.dumps(focused_evidence, ensure_ascii=False)[:1500]}

## Database priors (marker-based cell-type priors)
{_j.dumps(database_priors, ensure_ascii=False)[:1000]}

## Available analysis metadata (real columns you may reference)
{_j.dumps(metadata_context, ensure_ascii=False)[:1500]}

For each uncertain cluster, ask whether the candidate label is lineally supported, whether it might be
technical/mixed/doublet, whether a state label fits better, or whether exclusion-marker evidence is needed.
You MAY suggest tissue markers from your reasoning, but do not assume columns that aren't listed.

Reply with STRICT JSON only:
{{
  "overall_goal": "Validate preliminary cell annotations before user confirmation.",
  "scope": "annotation_validation_only",
  "tasks": [{{"task_id": "<id>", "cluster_ids": ["<id>"], "question": "<biological question>",
      "biological_concern": "<why>", "suggested_checks": ["<high-level check>"],
      "decision_rule": "<evidence -> label/confidence>"}}],
  "required_outputs": ["per_cluster_validation_summary", "suggested_annotation_changes", "generated_tables_or_figures"],
  "constraints": ["No DEG/pathway/communication/trajectory/velocity.",
                  "No clustering/filtering/normalization changes to the main object."]
}}
If every annotation is high-confidence and unflagged, return {{"scope": "annotation_validation_only", "tasks": []}}."""

        result = self._safe_parse_json(await self.chat(prompt, temperature=0.2), default={})
        if not result or "parse_error" in result:
            return {}
        return result if isinstance(result, dict) else {}

    async def revise_annotations_after_validation(
        self,
        annotations: Dict[str, Any],
        validation_plan: Dict[str, Any],
        validation_result: Dict[str, Any],
        literature_context: str,
        tissue: str,
        organism: str,
    ) -> Dict[str, Any]:
        """PI revises annotations using validation evidence. Returns
        {"annotations": <revised>, "validation_summary_for_user": <str>}. On parse
        failure returns originals unchanged with a generic summary."""
        self.refresh_runtime_config_if_needed()
        import json as _j, copy as _copy
        revised = _copy.deepcopy(annotations)

        ann_lines = [f"- Cluster {cid}: {info.get('cell_type','Unknown')} [conf={info.get('confidence','?')}]"
                     for cid, info in annotations.items() if isinstance(info, dict)]
        ann_text = "\n".join(ann_lines) or "No annotations."

        prompt = f"""You are the PI revising preliminary cell annotations for {organism} {tissue}
using data-grounded validation evidence. Update labels/confidence ONLY where evidence warrants.
Never remove a cluster: if evidence is poor, use a broad or technical/QC label with low confidence.

## Preliminary annotations
{ann_text}

## Validation plan (questions asked)
{_j.dumps(validation_plan, ensure_ascii=False)[:2000]}

## Validation evidence (restricted bioinformatics checks)
{_j.dumps(validation_result, ensure_ascii=False)[:5000]}

## Literature context
{(literature_context or '')[:1500]}

Per cluster you may update: cell_type, subtypes, confidence, evidence, alternative_cell_types, low_confidence_reason.
Reply with STRICT JSON only:
{{
  "revisions": {{"<id>": {{"cell_type": "...", "confidence": "high|medium|low", "evidence": "...",
      "subtypes": [], "alternative_cell_types": [], "low_confidence_reason": "..."}}}},
  "validation_summary_for_user": "<2-4 sentences: what changed and what remains uncertain>"
}}
Only include clusters you are actually changing in "revisions"."""

        parsed = self._safe_parse_json(await self.chat(prompt, temperature=0.2), default={})
        if not isinstance(parsed, dict):
            return {"annotations": revised,
                    "validation_summary_for_user": "Validation completed; no annotation changes were applied."}
        for cid, change in (parsed.get("revisions") or {}).items():
            if cid in revised and isinstance(revised[cid], dict) and isinstance(change, dict):
                new_type = str(change.get("cell_type") or "").strip()
                old_type = str(revised[cid].get("cell_type") or "").strip()
                if new_type and new_type != old_type:
                    # These articles were gathered for the previous label. Keep
                    # them: dropping them would make a re-examined cluster look
                    # less supported than one nobody looked at again. Mark them
                    # so nothing reads them as backing the label they now sit
                    # beside.
                    for citation in revised[cid].get("citations") or []:
                        if isinstance(citation, dict):
                            citation["stale"] = True
                for k, v in change.items():
                    if v not in (None, ""):
                        revised[cid][k] = v
        summary = parsed.get("validation_summary_for_user") or \
            "Validation completed; annotations reviewed against data evidence."
        return {"annotations": revised, "validation_summary_for_user": str(summary)}

    @staticmethod
    def _attach_annotation_citations(
        parsed: Dict[str, Any],
        evidence_pool: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Record, per cluster, the articles its annotation was based on.

        Two different facts are kept. The pool is what the prompt put in front
        of the matcher, so the record survives a matcher that names nothing --
        measured, that is most clusters. ``used_by_model`` is what the matcher
        said it used, and an identifier it names that was never shown for that
        cluster is dropped rather than stored as provenance it cannot have.
        """
        annotations = parsed.get("annotations")
        if not isinstance(annotations, dict):
            return parsed

        unshown_total = 0
        for cid, info in annotations.items():
            if not isinstance(info, dict):
                continue
            claimed = {
                str(pmid).strip()
                for pmid in (info.get("cited_pmids") or [])
                if str(pmid).strip()
            }
            info.pop("cited_pmids", None)

            shown = evidence_pool.get(str(cid))
            if shown is None:
                # No per-cluster evidence block reached the prompt for this
                # cluster. That is not the same as finding no literature.
                info["citations"] = []
                info["citation_status"] = "unavailable"
                continue

            by_pmid: Dict[str, Dict[str, Any]] = {}
            for article in shown:
                pmid = str(article.get("pmid") or "").strip()
                if pmid:
                    by_pmid.setdefault(pmid, article)
            unshown_total += len(claimed - set(by_pmid))

            citations = [
                {
                    "pmid": pmid,
                    "title": str(article.get("title") or "")[:300],
                    "year": (str(article.get("year") or "").strip() or None),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source": "focused_search",
                    "used_by_model": pmid in claimed,
                }
                for pmid, article in by_pmid.items()
            ]
            info["citations"] = citations
            if any(citation["used_by_model"] for citation in citations):
                info["citation_status"] = "grounded"
            elif citations:
                info["citation_status"] = "pool_only"
            else:
                info["citation_status"] = "no_evidence"

        if unshown_total:
            logger.warning(
                "[PI] Annotation matcher named %s PMID(s) absent from the evidence "
                "shown for their cluster; dropped from citations.", unshown_total
            )
        return parsed

    @staticmethod
    def _summarize_cluster_markers(
        cluster_markers: Dict[str, List[str]]
    ) -> Tuple[int, str]:
        """Summarize markers across ALL clusters → (n_clusters, markers_summary)."""
        n_clusters = len(cluster_markers)
        markers_summary = "\n".join(
            f"  Cluster {cid}: {', '.join(genes[:8])}"
            for cid, genes in cluster_markers.items()
        )
        return n_clusters, markers_summary

    async def generate_annotation_search_query(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        state: Optional[AnalysisState] = None,
    ) -> str:
        """Generate an optimized PubMed search query for cell type annotation.

        Uses PI's strong model to create high-quality queries with proper MeSH terms
        and Boolean operators, instead of delegating to Literature Agent's weak model.
        """
        logger.info(f"[PI] Generating annotation search query for {organism} {tissue}")

        n_clusters, markers_summary = self._summarize_cluster_markers(cluster_markers)

        # Experimental background from state
        background = ""
        if state and state.initial_instruction:
            background = f"\nExperimental background:\n{state.initial_instruction[:500]}"

        prompt = f"""You are a principal investigator planning a PubMed literature search to identify cell types
in a single-cell RNA sequencing dataset.

Organism: {organism}
Tissue: {tissue}
Number of clusters: {n_clusters}
{background}

Cluster marker genes (top genes per cluster):
{markers_summary}

Generate an optimized PubMed search query to find literature about cell types in {organism} {tissue}
that will help annotate these clusters. The query should retrieve articles about:
- Known cell type markers and signatures in this tissue
- Single-cell RNA sequencing studies of this tissue
- Cell type classification and identification

Rules for PubMed query:
1. Use MeSH terms where appropriate (e.g., "Single-Cell Analysis"[MeSH] plus the MeSH term that matches the dataset's tissue)
2. Use Boolean operators (AND, OR, NOT)
3. Anchor the query on three required concepts joined by AND: the organism, the tissue/organ,
   and single-cell or single-nucleus transcriptomics. Add one more AND concept for cell-type
   identification (atlas, cell type, cell identity, cell population) written as OR alternatives
4. Do NOT add generic words such as marker, markers, signature, classification or annotation
   as standalone OR alternatives — they match papers on any topic and bury the atlas-level
   studies this query exists to find. Combine such a word with the cell-type concept
   (e.g. "cell type markers") if you use it at all
5. Optionally add 2-3 marker genes that are highly specific to known cell types in this tissue,
   as OR alternatives inside the cell-type concept, never as their own AND requirement
6. Use field tags like [Title/Abstract], [MeSH] when helpful
7. Keep the query focused: a precise query returning a few dozen atlas-level studies is
   better than a broad one padded with tangential papers

Return ONLY the PubMed query string, nothing else."""

        response = await self.chat_without_history(prompt, temperature=0.2)
        # Clean up response
        query = response.strip().strip('"').strip("'").strip("`")
        query = re.sub(r'^```\w*\n?', '', query)
        query = re.sub(r'\n?```$', '', query)
        return query.strip()

    async def generate_annotation_research_brief(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        state: Optional[AnalysisState] = None,
    ) -> str:
        """Write a FOCUSED Deep Research brief for cell-type annotation.

        Produces prose instructions for the deep-research model, using the
        experimental background as reference only (no injection). Returns the
        stripped brief; an empty string signals the caller to fall back.
        """
        n_clusters, markers_summary = self._summarize_cluster_markers(cluster_markers)
        background = self._brief_background_block(state)

        prompt = f"""You are a principal investigator writing a Deep Research brief to identify cell types
in a single-cell RNA sequencing dataset.

Organism: {organism}
Tissue: {tissue}
Number of clusters: {n_clusters}

Cluster marker genes (top genes per cluster):
{markers_summary}{background}

This is a cell-type annotation task. Decide the search scope, then write a FOCUSED Deep Research
brief about marker genes and cell-type signatures for {organism} {tissue} that will help map these
clusters to known cell types. Do NOT expand into disease mechanisms, treatments, or
experimental-group comparisons unless a specific marker directly requires it.

{self._BRIEF_RULES}

Return ONLY the research brief text."""

        response = await self.chat_without_history(prompt, temperature=0.3)
        return (response or "").strip()

    @staticmethod
    def _annotation_evaluation_window(
        articles: List[Dict[str, str]],
        search_history: List[Dict[str, Any]],
        max_articles: int = _EVAL_MAX_ARTICLES,
    ) -> List[Dict[str, str]]:
        """Articles to show the evaluator: the latest round's retrievals first.

        `articles` is cumulative in retrieval order, so a fixed head window kept
        showing the first round's hits and hid whatever the refined query found.
        """
        latest_new = 0
        if search_history:
            try:
                latest_new = int(search_history[-1].get("new_results") or 0)
            except (TypeError, ValueError, AttributeError):
                latest_new = 0
        latest: List[Dict[str, str]] = []
        earlier = list(articles)
        if 0 < latest_new < len(articles):
            latest = list(articles[len(articles) - latest_new:])
            earlier = list(articles[: len(articles) - latest_new])
            latest = latest[: max(1, (max_articles * 2) // 3)]
        return (latest + earlier)[:max_articles]

    async def evaluate_annotation_search_results(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        articles: List[Dict[str, str]],
        search_history: List[Dict[str, Any]],
        state: Optional[AnalysisState] = None,
    ) -> Dict[str, Any]:
        """Evaluate PubMed search results with four-bucket triage.

        Uses PI's strong model to classify every cluster into:
        covered / rare_unresolved / suspected_contamination / context_dependent_state.
        """
        markers_summary = "\n".join(
            f"  Cluster {cid}: {', '.join(genes[:8])}"
            for cid, genes in cluster_markers.items()
        )

        shown = self._annotation_evaluation_window(articles, search_history)
        articles_summary = "\n".join(
            f"- [{a.get('year', 'N/A')}] {a['title']}\n"
            f"  Abstract: {(a.get('abstract') or 'N/A')[:_EVAL_ABSTRACT_CHARS]}"
            for a in shown
        )

        previous_queries = "\n".join(
            f"  Round {h['round']}: {h['query']}"
            for h in search_history
        )

        background = ""
        if state is not None and getattr(state, "initial_instruction", ""):
            background = f"\n## Experimental Context (use to judge condition-driven states)\n{state.initial_instruction[:500]}\n"
            try:
                background += state.build_experimental_design_summary()
            except Exception:
                pass

        prompt = f"""You are a principal investigator evaluating whether PubMed search results are sufficient
to annotate cell clusters in a {organism} {tissue} single-cell/single-nucleus RNA sequencing dataset.

## Cluster Marker Genes (need to identify cell types for ALL clusters)
{markers_summary}
{background}
## Previous Queries (DO NOT repeat these)
{previous_queries}

## Articles Found ({len(articles)} total, showing {len(shown)} with abstracts; articles retrieved by the most recent query are listed first)
{articles_summary}

Classify EVERY cluster into exactly one of four buckets, judging against the experimental
context (treatment groups / conditions) and tissue biology:
- covered: marker matches a known cell type already well supported by retrieved literature.
- rare_unresolved: a real, tissue-plausible but rare cell type with insufficient literature.
- suspected_contamination: markers biologically implausible for this tissue, OR technical-noise
  signature (batch effect, low-quality cluster, high mitochondrial/ribosomal fraction, doublet, ambient RNA).
- context_dependent_state: a KNOWN cell type's state variant driven by experimental context
  (e.g. condition-skewed cluster), NOT an independent new type — physiological reasoning suffices,
  no literature deep-dive. Do NOT use this bucket for tumor cells under study, developmental
  time-series target stages, or models where the baseline is equally treated.

Respond in JSON format:
{{
    "sufficient": true or false,
    "reasoning": "which clusters lack literature support",
    "refined_query": "a NEW PubMed query (MUST differ from all previous queries)",
    "missing_aspects": ["cell types / markers not yet covered"],
    "covered": ["<cluster_id>", ...],
    "covered_celltypes": {{"<cluster_id>": "the cell type the retrieved literature supports for this covered cluster, or null"}},
    "rare_unresolved": [{{"cluster_id": "..", "candidate_celltype": "Cell Ontology term or null",
        "candidate_celltype_confidence": "high|medium|low", "synonyms": ["CL official aliases, may be empty"],
        "top_markers": ["2-3 markers"], "triage_confidence": "high|medium|low",
        "evidence_need": "why the main loop cannot cover this cluster + what evidence is needed"}}],
    "suspected_contamination": [{{"cluster_id": "..", "candidate_celltype": "term or null",
        "top_markers": ["markers"], "triage_confidence": "high|medium|low",
        "reason": "why implausible / which technical-noise signature",
        "evidence_need": "what to verify"}}],
    "context_dependent_state": [{{"cluster_id": "..", "parent_celltype": "known type",
        "context": "the experimental condition driving the state", "top_markers": ["markers"],
        "triage_confidence": "high|medium|low", "state_evidence_note": "optional light background, NOT a deep-dive"}}]
}}

IMPORTANT:
- Classify ALL clusters; do not omit any cluster from the four buckets.
- refined_query MUST be valid PubMed (AND/OR), and MUST NOT duplicate any previous query.
- candidate_celltype MUST be a normalized Cell Ontology term; if you cannot, set it to null — do NOT invent.
- covered_celltypes SHOULD name the supported cell type for every covered cluster (null when you
  cannot). It is used to retrieve literature specific to that cluster, so a name here is what
  gives a covered cluster any evidence of its own.
- synonyms may be empty; do NOT fabricate synonyms when unsure.
- Every non-covered entry MUST include evidence_need."""

        response = await self.chat_without_history(prompt, temperature=0.3)
        result = self._safe_parse_json(response, default={})
        if not result or "sufficient" not in result:
            result = {"sufficient": True}
        for key in ("covered", "rare_unresolved", "suspected_contamination", "context_dependent_state"):
            result.setdefault(key, [])

        # --- Deterministic post-processing: normalize cluster IDs and enforce one bucket per cluster ---
        all_ids = {str(c) for c in cluster_markers.keys()}
        buckets = ("rare_unresolved", "suspected_contamination", "context_dependent_state")

        result["covered"] = [str(c) for c in result.get("covered", []) if str(c) in all_ids]
        covered_ids = set(result["covered"])

        cleaned_by_bucket = {bucket: [] for bucket in buckets}
        assigned_noncovered = set()

        # Higher priority buckets go first. Context states should not also be rare/contamination;
        # suspected contamination should not also be rare unresolved.
        priority_order = ("context_dependent_state", "suspected_contamination", "rare_unresolved")
        for bucket in priority_order:
            for item in result.get(bucket, []):
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("cluster_id"))
                if cid not in all_ids:
                    continue
                if cid in covered_ids:
                    continue
                if cid in assigned_noncovered:
                    continue
                cleaned_by_bucket[bucket].append(item)
                assigned_noncovered.add(cid)

        for bucket in buckets:
            result[bucket] = cleaned_by_bucket[bucket]

        # Models emit the literal strings "null"/"None" for the JSON null the
        # prompt asks for; downstream they became PubMed terms and prompt text.
        for bucket in ("rare_unresolved", "suspected_contamination"):
            for item in result[bucket]:
                item["candidate_celltype"] = normalize_candidate_celltype(
                    item.get("candidate_celltype")
                )
                if isinstance(item.get("synonyms"), list):
                    item["synonyms"] = [
                        s for s in item["synonyms"]
                        if normalize_candidate_celltype(s) is not None
                    ]
        for item in result["context_dependent_state"]:
            item["parent_celltype"] = normalize_candidate_celltype(item.get("parent_celltype"))

        raw_covered_names = result.get("covered_celltypes")
        covered_names: Dict[str, str] = {}
        if isinstance(raw_covered_names, dict):
            for raw_cid, raw_name in raw_covered_names.items():
                cid = str(raw_cid)
                if cid not in covered_ids:
                    continue
                name = normalize_candidate_celltype(raw_name)
                if name:
                    covered_names[cid] = name
        result["covered_celltypes"] = covered_names

        triaged = covered_ids | assigned_noncovered
        result["untriaged_cluster_ids"] = sorted(all_ids - triaged)
        result["triage_complete"] = bool(all_ids) and not (all_ids - triaged)
        result["triage_low_confidence"] = bool(all_ids) and (len(assigned_noncovered) / len(all_ids) > 0.40)
        return result

    async def annotate_clusters_from_literature(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        literature_report: str,
        triage: Optional[Dict[str, Any]] = None,
        focused_evidence: Optional[Dict[str, Any]] = None,
        database_priors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Annotate cell clusters by matching marker genes to literature-derived signatures.
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Annotating {len(cluster_markers)} clusters from literature for {organism} {tissue}")

        tissue_part = f" {tissue}" if tissue else ""
        markers_text = "\n".join(
            f"Cluster {cid}: {', '.join(genes[:15])}"
            for cid, genes in cluster_markers.items()
        )

        triage = triage or {}
        focused_evidence = focused_evidence or {}
        hints_block, evidence_pool = _render_cluster_evidence(
            list(cluster_markers.keys()), triage, focused_evidence
        )
        priors_block = _format_database_priors_block(database_priors)

        prompt = f"""Annotate each cell cluster in {organism}{tissue_part} by matching its marker \
genes to known cell type signatures, using the evidence below.
{hints_block}{priors_block}
=== BACKGROUND LITERATURE REVIEW (tissue-level context; secondary to the per-cluster evidence above) ===
{literature_report[:ANNOTATION_REVIEW_MAX_CHARS]}

=== CLUSTER MARKER GENES (from sc.tl.rank_genes_groups) ===
{markers_text}

For each cluster, provide:
1. The most likely cell type identity
2. Possible subtypes if applicable
3. Key marker genes that support this annotation (from the cluster's markers)
4. Confidence level (high/medium/low) based on how well markers match literature
5. Brief evidence/reasoning citing the literature
6. cited_pmids: the PMIDs from THIS cluster's own evidence block that you actually
   relied on. Use only identifiers printed in that block, never one from memory,
   and return an empty list when none of them supported the call

IMPORTANT:
- Be as specific as the markers actually support, and no more: a subtype label is justified
  only when its defining markers are present in this cluster's list, and in sparse 3'
  data the distinguishing transcripts are often too sparse to make that call. A label more
  specific than the evidence is unsupported, and everything downstream inherits it
- If a cluster has ambiguous markers, list the top 2 candidates and set confidence accordingly
- Consider the tissue context ({tissue}) when making assignments
- Not every cluster is a cell type. Use "Doublet/multiplet (suspected)", "Low-quality/dying
  (suspected)", "Ambient/background (suspected)", "Mixed/unresolved", "Transitional/intermediate
  state" or "Unknown" when they fit, rather than forcing a lineage label; for a suspected
  technical label say in `evidence` which per-cell QC would confirm or refute it

Return a JSON object with this EXACT structure:
{{
    "annotations": {{
        "<cluster_id>": {{
            "cell_type": "Primary cell type name",
            "subtypes": ["possible subtype 1", "possible subtype 2"],
            "key_markers": ["gene1", "gene2", "gene3"],
            "confidence": "high|medium|low",
            "evidence": "Brief reasoning with literature support",
            "cited_pmids": ["12345678"]
        }}
    }},
    "tissue_summary": "Brief summary of expected cell types in this tissue"
}}
"""

        response = await self.chat_without_history(prompt, temperature=0.2)
        parsed = self._safe_parse_json(response, default={})

        if parsed and "annotations" in parsed:
            return self._attach_annotation_citations(parsed, evidence_pool)

        # Retry once with a more explicit prompt if first attempt fails
        retry_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no additional text or markdown formatting."
        response_retry = await self.chat_without_history(retry_prompt, temperature=0.2)
        parsed_retry = self._safe_parse_json(response_retry, default={})

        if parsed_retry and "annotations" in parsed_retry:
            return self._attach_annotation_citations(parsed_retry, evidence_pool)

        # Both attempts failed
        logger.warning(f"[PI] Failed to parse annotation matching JSON after 2 attempts. Response preview: {response[:500]}")
        return {
            "status": "error",
            "error": "Failed to parse structured annotations after 2 attempts",
            "raw_response": response[:2000],
            "annotations": {},
            "tissue_summary": ""
        }

    def _format_cluster_annotation_summary(self, state) -> str:
        if not (state.annotation_state and state.annotation_state.cluster_annotations):
            return "No annotation data"
        lines = [
            f"- Cluster {ca.cluster_id}: {ca.cell_type} "
            f"[confidence: {ca.confidence}] markers: {', '.join(ca.key_markers[:10])}"
            for ca in state.annotation_state.cluster_annotations
        ]
        return "\n".join(lines)
