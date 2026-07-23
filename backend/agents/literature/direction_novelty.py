"""Novelty/prior-art judging of research-direction candidates.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
import time


logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState, DirectionSearchQuery, PubMedEvidenceItem

from .shared import (  # noqa: F401
    _FALLBACK_EXCEPTIONS,
    _MAX_MAIN_RETRIES,
    _RETRY_DELAY_BASE,
    plausibility_terms_for_tissue,
    _gene_case_variants,
    _sanitize_synonyms,
    status_from_judge,
    build_focused_query,
)


class DirectionNoveltyMixin:
    """Novelty/prior-art judging of research-direction candidates."""

    def _aggregate_qualification(self, triage_metadata_list: List[Dict[str, Any]]) -> str:
        priority = {
            "matched_species_tissue_available": 5,
            "no_matched_species_tissue_evidence": 4,
            "cross_tissue_only": 3,
            "cross_species_only": 2,
            "": -1,
        }
        best = ""
        best_prio = -1
        for m in triage_metadata_list:
            q = str(m.get("species_tissue_qualification", ""))
            prio = priority.get(q, -1)
            if prio > best_prio:
                best_prio = prio
                best = q
        return best

    def _article_matches_candidate(self, article: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        genes = [str(g).lower() for g in candidate.get("key_genes", []) if str(g).strip()]
        pathways = [str(p).lower() for p in candidate.get("key_pathways", []) if str(p).strip()]
        cells = [str(c).lower() for c in candidate.get("key_cell_types", []) if str(c).strip()]
        title = (article.get("title") or "").lower()
        abstract = (article.get("abstract") or "").lower()
        text = f"{title} {abstract}"
        return any(needle in text for needle in genes + pathways + cells)

    def _build_empty_direction_result(self, direction_id: str, direction_name: str) -> Dict[str, Any]:
        return {
            "direction_id": direction_id,
            "direction": direction_name,
            "status": "partial",
            "search_tier": "standard",
            "literature_novelty": "unknown",
            "confidence": "low",
            "negative_validation_confidence": "none",
            "negative_validation_axes": {},
            "direct_evidence_count": 0,
            "analogous_evidence_count": 0,
            "contradictory_evidence_count": 0,
            "methodological_precedent_count": 0,
            "direct_evidence": [],
            "analogous_evidence": [],
            "contradictory_evidence": [],
            "methodological_precedent": [],
            "queries": [],
            "species_tissue_qualification": "",
            "gap_assessment": "",
            "summary": "",
            "limitations": [],
            "timeout_seconds": None,
            "timed_out": False,
            "fallback_tier": "",
            "scheduler_limitation": "",
            "novelty_status": "unknown",
            "support_level": "unknown",
            "contradiction_level": "unknown",
            "prior_art_level": "unknown",
            "similar_published_directions": [],
            "knowledge_gap_summary": "",
        }

    def _build_gap_assessment(
        self,
        direct: List[PubMedEvidenceItem],
        analogous: List[PubMedEvidenceItem],
        contradictory: List[PubMedEvidenceItem],
        axes: Dict[str, bool],
        limitations: List[str],
    ) -> str:
        parts = []
        if not direct:
            parts.append("No direct prior evidence.")
        else:
            parts.append(f"Direct evidence: {len(direct)} articles.")
        if not analogous:
            parts.append("No analogous mechanism evidence.")
        else:
            parts.append(f"Analogous evidence: {len(analogous)} articles.")
        if contradictory:
            parts.append(f"Contradictory evidence found: {len(contradictory)} articles.")
        covered = [k for k, v in axes.items() if v]
        if covered:
            parts.append(f"Validation axes covered: {', '.join(covered)}.")
        if limitations:
            parts.append(f"Limitations: {'; '.join(limitations[:3])}.")
        return " ".join(parts) if parts else "No gap assessment available."

    def _build_triage_summary(
        self,
        triage_metadata_list: List[Dict[str, Any]],
        direct: List[PubMedEvidenceItem],
        analogous: List[PubMedEvidenceItem],
        contradictory: List[PubMedEvidenceItem],
    ) -> str:
        parts = []
        if direct:
            parts.append(f"Found {len(direct)} direct evidence articles.")
        if analogous:
            parts.append(f"Found {len(analogous)} analogous evidence articles.")
        if contradictory:
            parts.append(f"Found {len(contradictory)} contradictory evidence articles.")
        for m in triage_metadata_list:
            s = m.get("summary", "")
            if s:
                parts.append(s)
        return " ".join(parts) if parts else "No literature evidence found for this direction."

    def _contradiction_level(self, contradictory_count: int) -> str:
        if contradictory_count >= 2:
            return "strong"
        if contradictory_count == 1:
            return "moderate"
        return "none"

    async def _evaluate_minimum_attempt_candidate(
        self,
        candidate: Dict[str, Any],
        cached_articles: List[Dict[str, Any]],
        query_timeout_seconds: float,
        direction_timeout_seconds: float,
        scheduler_limitation: str,
    ) -> Dict[str, Any]:
        """Run the minimum evidence attempt for one candidate.

        _run_query_batches returns True when any query batch timed out or when
        the direction deadline expired; False means all attempted batches ended
        without timeout.
        """
        query_timeout_seconds = min(query_timeout_seconds, self.MIN_ATTEMPT_QUERY_TIMEOUT)
        direction_timeout_seconds = min(direction_timeout_seconds, self.MIN_ATTEMPT_DIRECTION_TIMEOUT)
        direction_id = candidate.get("direction_id", "")
        direction_name = candidate.get("direction", "")
        result = self._build_empty_direction_result(direction_id, direction_name)
        queries = self._minimum_attempt_queries(candidate)
        if not queries:
            result["limitations"].append("No valid minimum-attempt query could be built")
            result["scheduler_limitation"] = "no_minimum_query"
            result["novelty_status"] = "unknown_due_to_failure"
            return result
        evidence: List[PubMedEvidenceItem] = []
        executed_categories: set = set()
        triage_metadata_list: List[Dict[str, Any]] = []
        limitations: List[str] = []
        matching_cache = [a for a in cached_articles if self._article_matches_candidate(a, candidate)]
        timed_out = await self._run_query_batches(
            queries,
            candidate,
            matching_cache,
            query_timeout_seconds,
            all_evidence=evidence,
            executed_categories=executed_categories,
            triage_metadata_list=triage_metadata_list,
            all_limitations=limitations,
            deadline=time.monotonic() + direction_timeout_seconds,
        )
        if timed_out and "Minimum-attempt query timed out" not in limitations:
            limitations.append("Minimum-attempt query timed out")
        return self._finalize_seeded_direction_result(
            result,
            evidence,
            queries,
            limitations,
            fallback_tier="minimum_attempt",
            scheduler_limitation=scheduler_limitation,
        )

    async def _evaluate_single_candidate(
        self,
        candidate: Dict[str, Any],
        cached_articles: List[Dict[str, Any]],
        query_timeout_seconds: float,
        deadline: Optional[float] = None,
        direction_timeout_seconds: Optional[float] = None,
        tier3_query_timeout_seconds: float = 45.0,
    ) -> Dict[str, Any]:
        direction_id = candidate.get("direction_id", "")
        direction_name = candidate.get("direction", "")
        result = self._build_empty_direction_result(direction_id, direction_name)
        terms = self._candidate_terms(candidate)
        all_evidence: List[PubMedEvidenceItem] = []
        executed_categories: set = set()
        all_limitations: List[str] = []
        timed_out_flag = False
        triage_metadata_list: List[Dict[str, Any]] = []
        queries: List[DirectionSearchQuery] = []

        def mark_direction_timeout() -> None:
            nonlocal timed_out_flag
            timed_out_flag = True
            if "Direction-level timeout exceeded" not in all_limitations:
                all_limitations.append("Direction-level timeout exceeded")

        def direction_deadline_expired() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        # Cached articles matching this candidate are invariant across all batches.
        matching_cache = [a for a in cached_articles if self._article_matches_candidate(a, candidate)]

        if direction_deadline_expired():
            mark_direction_timeout()

        # Phase 1: Standard queries
        standard_queries = self._direction_queries(candidate, deep=False)
        queries.extend(standard_queries)

        if not timed_out_flag:
            if await self._run_query_batches(
                standard_queries, candidate, matching_cache, query_timeout_seconds,
                all_evidence=all_evidence, executed_categories=executed_categories,
                triage_metadata_list=triage_metadata_list, all_limitations=all_limitations,
                deadline=deadline,
            ):
                timed_out_flag = True

        # Check escalation conditions
        direct_ev, analogous_ev, contradictory_ev, methodological_ev = self._partition_evidence(all_evidence)

        has_methods = bool(terms["methods"])
        needs_deep = (
            len(direct_ev) < 2
            or (len(analogous_ev) < 1 and len(direct_ev) < 1)
            or any(m.get("contradictory_ambiguous", False) for m in triage_metadata_list)
            or any(m.get("off_topic", False) for m in triage_metadata_list)
            or any(q.degraded for q in standard_queries)
            or (has_methods and len(methodological_ev) == 0)
        )

        search_tier = "standard"
        if needs_deep and not timed_out_flag:
            deep_queries = self._direction_queries(candidate, deep=True)
            queries.extend(deep_queries)
            search_tier = "deep"
            if await self._run_query_batches(
                deep_queries, candidate, matching_cache, query_timeout_seconds,
                all_evidence=all_evidence, executed_categories=executed_categories,
                triage_metadata_list=triage_metadata_list, all_limitations=all_limitations,
                deadline=deadline,
            ):
                timed_out_flag = True

        # Tier-3 fallback: if deep search timed out with zero evidence, try broad queries
        fallback_tier = search_tier
        if timed_out_flag and not all_evidence:
            terms = self._candidate_terms(candidate)
            tier3_queries = [
                DirectionSearchQuery(
                    category="tier3_support",
                    query=self._build_tier3_broad_query(terms, challenge=False),
                    degraded=True,
                    notes="Tier 3 broad support fallback",
                ),
                DirectionSearchQuery(
                    category="tier3_challenge",
                    query=self._build_tier3_broad_query(terms, challenge=True),
                    degraded=True,
                    notes="Tier 3 broad challenge fallback",
                ),
            ]
            tier3_queries = [q for q in tier3_queries if q.query]
            if tier3_queries:
                queries.extend(tier3_queries)
                fallback_tier = "tier3_broad"
                await self._run_query_batches(
                    tier3_queries,
                    candidate,
                    matching_cache,
                    min(tier3_query_timeout_seconds, query_timeout_seconds),
                    all_evidence=all_evidence,
                    executed_categories=executed_categories,
                    triage_metadata_list=triage_metadata_list,
                    all_limitations=all_limitations,
                    deadline=time.monotonic() + tier3_query_timeout_seconds,
                )

        # Deduplicate by PMID so the same article is not counted across multiple query categories.
        all_evidence = self._dedup_evidence(all_evidence)

        # Re-count after deep search
        direct_ev, analogous_ev, contradictory_ev, methodological_ev = self._partition_evidence(all_evidence)

        # Compute axes — an axis counts as covered only when the candidate actually
        # had the relevant terms AND a non-degraded query for that axis executed.
        has_gene = bool(terms["genes"])
        has_cell = bool(terms["cells"])
        has_tissue = bool(terms["tissue"][0])
        has_pathway = bool(terms["pathways"])
        axes = {
            "gene_or_model": has_gene and (
                "direct_prior_work" in executed_categories or "gene_model_relaxed" in executed_categories),
            "tissue_or_cell_type": (has_cell or has_tissue) and (
                "direct_prior_work" in executed_categories or "cell_type_focused" in executed_categories),
            "pathway_or_mechanism": has_pathway and (
                "analogous_mechanism" in executed_categories or "pathway_focused" in executed_categories),
            "analogous_mechanism": (has_pathway or has_gene) and (
                "analogous_mechanism" in executed_categories or "cross_species_or_tissue" in executed_categories),
        }

        all_axes_covered = all(axes.values())
        no_surviving_evidence = len(direct_ev) == 0 and len(analogous_ev) == 0
        neg_conf: str = "none"
        if all_axes_covered and no_surviving_evidence:
            neg_conf = "high"
        elif all_axes_covered:
            neg_conf = "medium"
        elif any(axes.values()):
            neg_conf = "low"

        # Aggregate species_tissue_qualification
        best_qualification = self._aggregate_qualification(triage_metadata_list)

        # Novelty assessment (spec §231: absence-based high novelty requires strong negative validation).
        novelty = "unknown"
        if len(direct_ev) >= 2:
            novelty = "low"
        elif len(direct_ev) >= 1:
            novelty = "medium"
        elif len(analogous_ev) >= 1:
            # Related prior work exists -> moderate novelty, never "high".
            novelty = "medium"
        elif len(direct_ev) == 0 and len(analogous_ev) == 0:
            # No direct/analogous evidence: only claim high novelty when the negative
            # search was strong enough to mean "genuinely unstudied".
            novelty = "high" if neg_conf in {"high", "medium"} else "unknown"

        confidence = "high"
        if not all_evidence:
            confidence = "low" if not timed_out_flag else "failed"
        elif len(direct_ev) < 2:
            confidence = "medium"

        # Failed/timed-out searches cannot assert novelty (spec error-handling rule 3).
        if timed_out_flag or confidence == "failed":
            novelty = "unknown"

        gap = self._build_gap_assessment(direct_ev, analogous_ev, contradictory_ev, axes, all_limitations)
        summary = self._build_triage_summary(triage_metadata_list, direct_ev, analogous_ev, contradictory_ev)

        novelty_status = self._resolve_novelty_status(
            novelty=novelty,
            timed_out=timed_out_flag,
            confidence=confidence,
            direct_count=len(direct_ev),
            analogous_count=len(analogous_ev),
            neg_conf=neg_conf,
        )

        result.update({
            "status": "completed" if not timed_out_flag else "partial",
            "search_tier": search_tier,
            "literature_novelty": novelty,
            "confidence": confidence,
            "negative_validation_confidence": neg_conf,
            "negative_validation_axes": axes,
            "direct_evidence_count": len(direct_ev),
            "analogous_evidence_count": len(analogous_ev),
            "contradictory_evidence_count": len(contradictory_ev),
            "methodological_precedent_count": len(methodological_ev),
            "direct_evidence": [e.model_dump() for e in direct_ev],
            "analogous_evidence": [e.model_dump() for e in analogous_ev],
            "contradictory_evidence": [e.model_dump() for e in contradictory_ev],
            "methodological_precedent": [e.model_dump() for e in methodological_ev],
            "queries": [q.model_dump() for q in queries],
            "species_tissue_qualification": best_qualification,
            "gap_assessment": gap,
            "summary": summary,
            "limitations": all_limitations,
            "timeout_seconds": direction_timeout_seconds if timed_out_flag else None,
            "timed_out": timed_out_flag,
            "fallback_tier": fallback_tier,
            "novelty_status": novelty_status,
            "support_level": self._support_level(len(direct_ev), len(analogous_ev)),
            "contradiction_level": self._contradiction_level(len(contradictory_ev)),
            "prior_art_level": self._prior_art_level(len(direct_ev), len(analogous_ev)),
            "similar_published_directions": [e.model_dump() for e in direct_ev[:5]],
            "knowledge_gap_summary": gap,
        })

        return result

    def _finalize_seeded_direction_result(
        self,
        result: Dict[str, Any],
        evidence: List[PubMedEvidenceItem],
        queries: List[DirectionSearchQuery],
        limitations: List[str],
        *,
        fallback_tier: str,
        scheduler_limitation: str = "",
    ) -> Dict[str, Any]:
        deduped = self._dedup_evidence(evidence)
        direct_ev, analogous_ev, contradictory_ev, methodological_ev = self._partition_evidence(deduped)
        timed_out = any("timeout" in str(lim).lower() for lim in limitations)
        confidence = "failed" if timed_out and not deduped else ("medium" if deduped else "low")
        novelty = "unknown" if timed_out else ("medium" if (direct_ev or analogous_ev) else "unknown")
        result.update({
            "status": "partial",
            "literature_novelty": novelty,
            "confidence": confidence,
            "direct_evidence_count": len(direct_ev),
            "analogous_evidence_count": len(analogous_ev),
            "contradictory_evidence_count": len(contradictory_ev),
            "methodological_precedent_count": len(methodological_ev),
            "direct_evidence": [e.model_dump() for e in direct_ev],
            "analogous_evidence": [e.model_dump() for e in analogous_ev],
            "contradictory_evidence": [e.model_dump() for e in contradictory_ev],
            "methodological_precedent": [e.model_dump() for e in methodological_ev],
            "queries": [q.model_dump() for q in queries],
            "limitations": result.get("limitations", []) + limitations,
            "timed_out": timed_out,
            "fallback_tier": fallback_tier,
            "scheduler_limitation": scheduler_limitation,
            "support_level": self._support_level(len(direct_ev), len(analogous_ev)),
            "contradiction_level": self._contradiction_level(len(contradictory_ev)),
            "prior_art_level": self._prior_art_level(len(direct_ev), len(analogous_ev)),
            "novelty_status": self._resolve_novelty_status(
                novelty=novelty,
                timed_out=timed_out,
                confidence=confidence,
                direct_count=len(direct_ev),
                analogous_count=len(analogous_ev),
                neg_conf="none",
            ),
            "similar_published_directions": [e.model_dump() for e in direct_ev[:5]],
            "knowledge_gap_summary": "Minimum-attempt evidence only; full evidence evaluation was capped or unavailable.",
        })
        return result

    def _focused_result_score(self, result: Dict[str, Any]) -> tuple:
        """Rank focused-search outcomes; on exhausted ties, surface runtime failures."""
        status = result.get("status", "exhausted")
        return (
            self._STATUS_RANK.get(status, 0),
            self._EXHAUSTION_REASON_RANK.get(result.get("exhaustion_reason"), 0),
        )

    async def _judge_focused_relevance(self, item, tissue, articles, bucket) -> dict:
        """轻量 LLM 裁决,沿 identity/marker/tissue/method 维度评相关性。

        携带 candidate + markers + bucket 语境: 对 suspected_contamination,candidate 可能为 None,
        由 markers 指向判定 contradictory(marker 主要指向其它系统/细胞类型)。
        """
        candidate = item.get("candidate_celltype") or "the marker-defined population"
        markers = ", ".join(item.get("top_markers", [])[:3])
        purpose = ("verify rare cell identity" if bucket == "rare_unresolved"
                   else "test whether these markers plausibly arise in this tissue or indicate contamination")
        listing = "\n".join(
            f"- {a.get('title','')}: {a.get('abstract','')[:200]}" for a in articles[:10]
        ) or "(no articles)"
        prompt = f"""Judge how many PubMed articles support identifying "{candidate}" (markers: {markers})
in {tissue} tissue. Purpose: {purpose}. Score along cell-type identity, marker support, tissue
relevance, method relevance.

Articles:
{listing}

Respond JSON: {{"strong": <count strongly relevant>, "moderate": <count moderately relevant>,
"contradictory": <true if markers mainly point to a DIFFERENT system/celltype than the candidate/tissue>}}"""
        resp = await self.chat_without_history(prompt, temperature=0.1)
        judge = self._safe_parse_json(resp, default={}) or {}
        for k in ("strong", "moderate"):
            judge.setdefault(k, 0)
        judge.setdefault("contradictory", False)
        return {"status": status_from_judge(judge), "judge": judge}

    @staticmethod
    def _partition_evidence(
        items: List["PubMedEvidenceItem"],
    ) -> Tuple[List["PubMedEvidenceItem"], ...]:
        """Split evidence into (direct, analogous, contradictory, methodological)."""
        return (
            [e for e in items if e.supports == "direct"],
            [e for e in items if e.supports == "analogous"],
            [e for e in items if e.supports == "contradictory"],
            [e for e in items if e.supports == "methodological"],
        )

    def _prior_art_level(self, direct_count: int, analogous_count: int) -> str:
        if direct_count >= 2:
            return "high"
        if direct_count == 1 or analogous_count >= 2:
            return "medium"
        if analogous_count == 1:
            return "low"
        return "low"

    def _resolve_novelty_status(
        self,
        *,
        novelty: str,
        timed_out: bool,
        confidence: str,
        direct_count: int,
        analogous_count: int,
        neg_conf: str,
    ) -> str:
        if timed_out:
            return "unknown_due_to_timeout"
        if confidence == "failed":
            return "unknown_due_to_failure"
        if direct_count > 0 or analogous_count > 0:
            return "known_area"
        if novelty == "high" and neg_conf in {"high", "medium"}:
            return "genuine_novelty"
        return "unknown"

    def _support_level(self, direct_count: int, analogous_count: int) -> str:
        if direct_count >= 2:
            return "strong"
        if direct_count >= 1 or analogous_count >= 2:
            return "moderate"
        if analogous_count >= 1:
            return "weak"
        return "none"

    async def _triage_articles_with_llm(
        self,
        articles: List[Dict[str, Any]],
        candidate: Dict[str, Any],
        query: DirectionSearchQuery,
    ) -> Tuple[List[PubMedEvidenceItem], Dict[str, Any]]:
        if not articles:
            return [], {"off_topic": False, "contradictory_ambiguous": False,
                         "species_tissue_qualification": "", "summary": "No articles to triage."}

        terms = self._candidate_terms(candidate)
        articles_text = "\n".join(
            f"[PMID: {a.get('pmid', '?')}] {a.get('title', '')} - {a.get('abstract', '')[:400]}"
            for a in articles[:30]
        )

        prompt = f"""You are a biomedical literature triage system. Assign each article to the most appropriate bucket.

Research direction: {candidate.get('direction', '')}
Key genes: {', '.join(terms['genes'])}
Key cell types: {', '.join(terms['cells'])}
Key pathways: {', '.join(terms['pathways'])}
Organism: {', '.join(terms['organism'])}
Tissue: {', '.join(terms['tissue'])}
Query category: {query.category}

Buckets:
- direct: Directly studies the same gene(s), cell type(s), pathway(s), and experimental context
- analogous: Studies a related but distinct mechanism, cell type, or context that could inform this direction
- contradictory: Presents evidence that contradicts or challenges the proposed biology
- methodological: Describes a method or technique relevant to the proposed approach
- irrelevant: Not relevant to this research direction

Articles:
{articles_text}

Respond ONLY with valid JSON (no markdown, no code fences):
{{
  "items": [
    {{"pmid": "...", "bucket": "direct|analogous|contradictory|methodological|irrelevant", "relevance": "high|medium|low", "why_it_matters": "brief explanation"}}
  ],
  "off_topic": true or false,
  "contradictory_ambiguous": true or false,
  "species_tissue_qualification": "matched_species_tissue_available" | "no_matched_species_tissue_evidence" | "cross_tissue_only" | "cross_species_only" | "",
  "summary": "One-sentence summary of this batch's evidence."
}}"""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.1)
        parsed = self._safe_parse_json(response, default={})
        items_raw = parsed.get("items", [])
        if not isinstance(items_raw, list):
            items_raw = []

        evidence_items: List[PubMedEvidenceItem] = []
        for it in items_raw:
            pmid = str(it.get("pmid", ""))
            bucket = str(it.get("bucket", "irrelevant"))
            # Only keep non-irrelevant items
            if bucket == "irrelevant":
                continue
            supports = bucket if bucket in ("direct", "analogous", "contradictory", "methodological") else "analogous"
            relevance = str(it.get("relevance", "low"))
            if relevance not in ("high", "medium", "low"):
                relevance = "low"
            article = next((a for a in articles if str(a.get("pmid", "")) == pmid), None)
            evidence_items.append(PubMedEvidenceItem(
                pmid=pmid,
                title=article.get("title", "") if article else "",
                year=article.get("year", "") if article else None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                relevance=relevance,
                supports=supports,
                source="session_cache" if (article and article.get("source") == "session_cache") else "new_search",
                species_context=", ".join(terms["organism"]),
                tissue_context=", ".join(terms["tissue"]),
                why_it_matters=str(it.get("why_it_matters", "")),
            ))

        metadata = {
            "off_topic": bool(parsed.get("off_topic", False)),
            "contradictory_ambiguous": bool(parsed.get("contradictory_ambiguous", False)),
            "species_tissue_qualification": str(parsed.get("species_tissue_qualification", "")),
            "summary": str(parsed.get("summary", "")),
        }
        return evidence_items, metadata

    async def evaluate_direction_novelty_candidates(
        self,
        candidates: List[Dict[str, Any]],
        state: Optional[AnalysisState] = None,
        query_timeout_seconds: float = 30.0,
        direction_timeout_seconds: float = 180.0,
        max_full_evidence_candidates: int = 4,
        tier3_query_timeout_seconds: float = 45.0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        cached_articles = self._extract_session_cached_pubmed_articles(state)

        for index, candidate in enumerate(candidates):
            direction_id = candidate.get("direction_id", "")
            direction_name = candidate.get("direction", "")
            if index >= max_full_evidence_candidates:
                results.append(await self._evaluate_minimum_attempt_candidate(
                    candidate,
                    cached_articles,
                    query_timeout_seconds=query_timeout_seconds,
                    direction_timeout_seconds=direction_timeout_seconds,
                    scheduler_limitation="full_evaluation_skipped_by_candidate_cap",
                ))
                continue

            deadline = time.monotonic() + direction_timeout_seconds
            try:
                dir_result = await self._evaluate_single_candidate(
                    candidate,
                    cached_articles,
                    query_timeout_seconds,
                    deadline=deadline,
                    direction_timeout_seconds=direction_timeout_seconds,
                    tier3_query_timeout_seconds=tier3_query_timeout_seconds,
                )
                if not dir_result.get("queries"):
                    dir_result = await self._evaluate_minimum_attempt_candidate(
                        candidate,
                        cached_articles,
                        query_timeout_seconds=query_timeout_seconds,
                        direction_timeout_seconds=direction_timeout_seconds,
                        scheduler_limitation="full_evaluation_returned_no_queries",
                    )
                results.append(dir_result)
            except Exception as exc:
                logger.warning(f"[Literature] Direction evaluation failed for {direction_id}: {exc}")
                fallback = await self._evaluate_minimum_attempt_candidate(
                    candidate,
                    cached_articles,
                    query_timeout_seconds=query_timeout_seconds,
                    direction_timeout_seconds=direction_timeout_seconds,
                    scheduler_limitation="full_evaluation_failed_before_minimum_attempt",
                )
                fallback["status"] = "failed" if not fallback.get("queries") else "partial"
                fallback["limitations"].append(str(exc)[:200])
                if not fallback.get("timed_out"):
                    fallback["novelty_status"] = "unknown_due_to_failure"
                results.append(fallback)

        return results
