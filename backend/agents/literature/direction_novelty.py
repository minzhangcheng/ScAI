"""Novelty/prior-art judging of research-direction candidates.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import asyncio
import copy
import hashlib
import json
import logging
import time
from ...utils.llm_client import UpstreamExhaustedError
from ...utils.llm_client_parts.retry import llm_timeout_seconds


logger = logging.getLogger(__name__)

# Retrieval budgets for the novelty gate. Model calls have an independent
# deployment-configured minimum (600 seconds by default); their elapsed time
# does not consume these windows.
NOVELTY_QUERY_TIMEOUT_SECONDS = 90.0
PI_DIRECTION_LLM_TIMEOUT_SECONDS = 600.0
NOVELTY_DIRECTION_TIMEOUT_SECONDS = 420.0
NOVELTY_TIER3_QUERY_TIMEOUT_SECONDS = 90.0

# Candidates share no state, so the gate's wall clock does not have to be the
# sum of their budgets. Bounded so a wide candidate list cannot open an
# unlimited number of concurrent gateway calls.
NOVELTY_CANDIDATE_CONCURRENCY = 4
PI_DIRECTION_SEARCH_POLICY_VERSION = "direction_search_v1"
PI_DIRECTION_MAX_INITIAL_TASKS = 6
PI_DIRECTION_MAX_FOLLOWUP_TASKS = 4
PI_DIRECTION_REVIEW_RESERVE_SECONDS = 90.0
from ...models.schemas import (
    AnalysisState,
    DirectionSearchQuery,
    PubMedEvidenceItem,
    DirectionSearchTask,
    DirectionSearchExecution,
    DirectionArticleInventory,
    DirectionLiteratureExtraction,
    DirectionPIReview,
    DirectionLiteratureInProgress,
)
from .query_builders import (
    DirectionSearchCompilationError,
    compile_direction_search_plan,
    compile_direction_search_task,
)

from .shared import (  # noqa: F401
    normalize_candidate_celltype,
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
            "key_evidence": "",
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
        triaged_pmids: Set[str] = {
            str(a.get("pmid") or "") for a in matching_cache if a.get("pmid")
        }
        deadline = time.monotonic() + direction_timeout_seconds

        def exclude_llm_time(elapsed):
            nonlocal deadline
            if deadline is not None:
                deadline += elapsed

        cache_ev, cache_meta, cache_timed_out, cache_limits = (
            await self._triage_cached_articles_once(
                candidate, matching_cache, query_timeout_seconds, deadline,
                pause_deadlines_fn=exclude_llm_time,
            )
        )
        evidence.extend(cache_ev)
        if cache_meta:
            triage_metadata_list.append(cache_meta)
        limitations.extend(cache_limits)
        timed_out = await self._run_query_batches(
            queries,
            candidate,
            [],
            query_timeout_seconds,
            all_evidence=evidence,
            executed_categories=executed_categories,
            triage_metadata_list=triage_metadata_list,
            all_limitations=limitations,
            deadline=deadline,
            triaged_pmids=triaged_pmids,
            pause_deadlines_fn=exclude_llm_time,
        )
        timed_out = timed_out or cache_timed_out
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

    async def _triage_cached_articles_once(
        self,
        candidate: Dict[str, Any],
        matching_cache: List[Dict[str, Any]],
        query_timeout_seconds: float,
        deadline: Optional[float],
        pause_deadlines_fn: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[PubMedEvidenceItem], Dict[str, Any], bool, List[str]]:
        """Triage this candidate's cached articles once, not once per query.

        ``matching_cache`` is computed per candidate and is identical for every
        one of its queries, but the per-query path re-triaged it each time --
        five to seven identical LLM calls per candidate, each able to consume a
        whole step budget. That, not PubMed, is what exhausted the direction
        budget before any searched evidence could be triaged.
        """
        evidence: List[PubMedEvidenceItem] = []
        meta: Dict[str, Any] = {}
        limitations: List[str] = []
        if not matching_cache:
            return evidence, meta, False, limitations
        cache_query = DirectionSearchQuery(
            category="session_cache",
            query="",
            degraded=True,
            notes="Session-cached articles, triaged once per candidate",
        )
        try:
            evidence, meta = await self._await_literature_llm(
                self._triage_articles_with_llm(matching_cache, candidate, cache_query),
                timeout_seconds=query_timeout_seconds,
                on_elapsed=pause_deadlines_fn,
            )
        except UpstreamExhaustedError:
            raise
        except Exception as exc:
            return evidence, meta, False, [f"Cache triage failed: {str(exc)[:100]}"]
        return evidence, meta, False, limitations

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

        def exclude_llm_time(elapsed):
            nonlocal deadline
            if deadline is not None:
                deadline += elapsed

        def direction_deadline_expired() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        # Cached articles matching this candidate are invariant across all batches,
        # so they are triaged once here and the per-query path is handed an empty
        # cache. Passing them down would re-run this call for every query.
        matching_cache = [a for a in cached_articles if self._article_matches_candidate(a, candidate)]
        # Everything already put in front of triage for THIS candidate. The
        # cached articles are triaged first, so they seed it.
        triaged_pmids: Set[str] = {
            str(a.get("pmid") or "") for a in matching_cache if a.get("pmid")
        }

        if direction_deadline_expired():
            mark_direction_timeout()

        if not timed_out_flag:
            cache_ev, cache_meta, cache_timed_out, cache_limits = (
                await self._triage_cached_articles_once(
                    candidate, matching_cache, query_timeout_seconds, deadline,
                pause_deadlines_fn=exclude_llm_time,
                )
            )
            all_evidence.extend(cache_ev)
            if cache_meta:
                triage_metadata_list.append(cache_meta)
            all_limitations.extend(cache_limits)
            if cache_timed_out:
                timed_out_flag = True

        # Phase 1: Standard queries
        standard_queries = self._direction_queries(candidate, deep=False)
        queries.extend(standard_queries)

        if not timed_out_flag:
            if await self._run_query_batches(
                standard_queries, candidate, [], query_timeout_seconds,
                all_evidence=all_evidence, executed_categories=executed_categories,
                triage_metadata_list=triage_metadata_list, all_limitations=all_limitations,
                deadline=deadline, triaged_pmids=triaged_pmids,
                pause_deadlines_fn=exclude_llm_time,
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
                deep_queries, candidate, [], query_timeout_seconds,
                all_evidence=all_evidence, executed_categories=executed_categories,
                triage_metadata_list=triage_metadata_list, all_limitations=all_limitations,
                deadline=deadline, triaged_pmids=triaged_pmids,
                pause_deadlines_fn=exclude_llm_time,
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
                    [],
                    min(tier3_query_timeout_seconds, query_timeout_seconds),
                    all_evidence=all_evidence,
                    executed_categories=executed_categories,
                    triage_metadata_list=triage_metadata_list,
                    all_limitations=all_limitations,
                    deadline=time.monotonic() + tier3_query_timeout_seconds,
                    triaged_pmids=triaged_pmids,
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
            "key_evidence": self._build_key_evidence(
                direct_ev, contradictory_ev, analogous_ev
            ),
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
            "key_evidence": self._build_key_evidence(
                direct_ev, contradictory_ev, analogous_ev
            ),
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
        candidate = (
            normalize_candidate_celltype(item.get("candidate_celltype"))
            or "the marker-defined population"
        )
        markers = ", ".join(item.get("top_markers", [])[:3])
        purpose = ("verify rare cell identity" if bucket == "rare_unresolved"
                   else "test whether these markers plausibly arise in this tissue or indicate contamination")
        listing = "\n".join(
            f"- {a.get('title','')}: {(a.get('abstract') or '')[:400]}" for a in articles[:10]
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
    def _build_key_evidence(
        direct_ev: List["PubMedEvidenceItem"],
        contradictory_ev: List["PubMedEvidenceItem"],
        analogous_ev: Optional[List["PubMedEvidenceItem"]] = None,
        *,
        per_bucket: int = 3,
        title_chars: int = 90,
    ) -> str:
        """Name the articles behind the counts, for whoever reads this record.

        The PI scoring step is handed counts and a prose summary and no article
        identity at all, so it can neither check a "direct evidence: 4" claim nor
        cite one, and a mis-bucketed article is indistinguishable from a real
        hit. Bounded on purpose: this shares a per-record slot with the rest of
        the evidence block.

        Analogous articles are named only when there is no direct evidence.
        Rendering direct and contradictory alone left this field EMPTY for
        exactly the adjacent_work_only directions -- measured, three of eight
        across two runs, each carrying 57 to 77 analogous articles -- which are
        the ones where the scorer most needs something it can look at. When
        direct evidence exists it is the more useful thing to show, and adding
        the adjacent pile beside it only dilutes the slot.
        """
        def render(label: str, items: List["PubMedEvidenceItem"]) -> str:
            if not items:
                return ""
            shown = []
            for item in items[:per_bucket]:
                pmid = getattr(item, "pmid", "") or "?"
                year = getattr(item, "year", "") or ""
                title = (getattr(item, "title", "") or "").strip()[:title_chars]
                stamp = f" ({year})" if year else ""
                shown.append(f"PMID {pmid}{stamp} {title}".strip())
            hidden = len(items) - len(shown)
            suffix = f" (+{hidden} more)" if hidden > 0 else ""
            return f"{label}: " + "; ".join(shown) + suffix

        rendered = [
            render("direct", direct_ev),
            render("contradictory", contradictory_ev),
        ]
        if not direct_ev:
            rendered.append(render("adjacent (not direct)", analogous_ev or []))
        return " | ".join(part for part in rendered if part)

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
        if direct_count > 0:
            return "known_area"
        if analogous_count > 0:
            # Related work is not prior work on this question. Collapsing the two
            # made "known_area" the verdict for every candidate that retrieved
            # anything at all -- and since triage assigns most articles to the
            # analogous bucket, that was every candidate, whether or not one
            # paper addressed the question. Say which of the two was found.
            return "adjacent_work_only"
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
        # Retrieval is keyword-based, so a query built from gene symbols returns
        # the same symbol in any context: a melatonin-receptor MT2 paper came
        # back as direct prior art for a metallothionein direction. The bucket
        # definition already asked for the experimental context, but the prompt
        # never said what it was. Supplying it costs no extra search.
        study_context = str(getattr(self, "_direction_study_context", "") or "").strip()
        study_line = (
            f"Study under investigation: {study_context}\n"
            if study_context else
            "Study under investigation: not supplied; judge the context from the "
            "research direction above.\n"
        )
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
{study_line}
Buckets:
- direct: Studies this question in this study's experimental context, meaning the same perturbation, model or condition as the study named above. The same gene, pathway or cell type under a different perturbation, disease or intervention is NOT direct
- analogous: Studies a related but distinct mechanism, cell type, or context that could inform this direction, including the same genes under a different perturbation
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

    # ------------------------------------------------------------------
    # PI-directed direction search (the ordinary topic path is above/below
    # this section and intentionally does not call any of these methods).

    @staticmethod
    def _direction_gate_id(
        candidates: List[Dict[str, Any]],
        state: Optional[AnalysisState],
        phase: str,
    ) -> str:
        ids = [str(c.get("direction_id") or "") for c in candidates]
        question_payload = [
            {
                key: candidate.get(key)
                for key in (
                    "direction_id",
                    "direction",
                    "core_observation",
                    "main_hypothesis",
                    "alternative_hypotheses",
                    "key_genes",
                    "key_cell_types",
                    "key_pathways",
                    "organism",
                    "tissue",
                    "experimental_context",
                )
            }
            for candidate in candidates
        ]
        question_hash = hashlib.sha256(
            json.dumps(
                question_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode()
        ).hexdigest()[:20]
        round_number = int(getattr(state, "current_round", 0) or 0) if state is not None else 0
        cycle = int(getattr(state, "integration_cycle", 0) or 0) if state is not None else 0
        return (
            f"{phase}:{round_number}:{cycle}:" + ",".join(ids) +
            f":{question_hash}:{PI_DIRECTION_SEARCH_POLICY_VERSION}"
        )

    @staticmethod
    def _checkpoint_store(state: Optional[AnalysisState]) -> Optional[Dict[str, Any]]:
        if state is None:
            return None
        store = getattr(state, "direction_literature_in_progress", None)
        if store is None:
            store = {}
            try:
                setattr(state, "direction_literature_in_progress", store)
            except Exception:
                return None
        return store

    @classmethod
    def _load_direction_checkpoint(
        cls,
        state: Optional[AnalysisState],
        gate_id: str,
        candidates: List[Dict[str, Any]],
        phase: str,
    ) -> DirectionLiteratureInProgress:
        store = cls._checkpoint_store(state)
        existing = store.get(gate_id) if isinstance(store, dict) else None
        if existing:
            try:
                checkpoint = DirectionLiteratureInProgress.model_validate(existing)
                expected = [str(c.get("direction_id") or "") for c in candidates]
                if checkpoint.candidate_ids == expected:
                    return checkpoint
            except Exception:
                logger.warning("[Literature] Ignoring malformed direction-search checkpoint %s", gate_id)
        return DirectionLiteratureInProgress(
            gate_id=gate_id,
            policy_version=PI_DIRECTION_SEARCH_POLICY_VERSION,
            phase=phase,
            round_number=int(getattr(state, "current_round", 0) or 0) if state is not None else 0,
            integration_cycle=(
                int(getattr(state, "integration_cycle", 0) or 0) if state is not None else None
            ),
            candidate_ids=[str(c.get("direction_id") or "") for c in candidates],
        )

    @classmethod
    def _save_direction_checkpoint(
        cls,
        state: Optional[AnalysisState],
        checkpoint: DirectionLiteratureInProgress,
    ) -> None:
        store = cls._checkpoint_store(state)
        if isinstance(store, dict):
            # Keep a JSON-shaped value in AnalysisState so normal state
            # persistence and old readers can carry it without custom hooks.
            store[checkpoint.gate_id] = checkpoint.model_dump(mode="json")

    @classmethod
    def _add_direction_limitation(
        cls,
        checkpoint: DirectionLiteratureInProgress,
        direction_id: str,
        limitation: str,
    ) -> None:
        if not limitation:
            return
        values = checkpoint.limitations_by_candidate.setdefault(direction_id, [])
        if limitation not in values:
            values.append(limitation)

    @staticmethod
    def _direction_plan_hash(tasks: List[DirectionSearchTask]) -> str:
        payload = [task.model_dump(mode="json") for task in tasks]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    @staticmethod
    def _direction_callback_result(result: Any, key: str) -> Any:
        if isinstance(result, dict):
            return result.get(key, result.get("tasks" if key == "plan" else "reviews", result))
        return result

    async def _direction_llm_call_with_intervention(
        self,
        callback: Callable[..., Any],
        args: tuple,
        kwargs: Dict[str, Any],
        *,
        state: Optional[AnalysisState],
        phase: str,
        intervention_fn: Optional[Callable[..., Any]],
        timeout_seconds: Optional[float] = None,
        pause_deadlines_fn: Optional[Callable[[float], None]] = None,
        failure_fn: Optional[Callable[[UpstreamExhaustedError], None]] = None,
        agent_name: str = "pi",
    ) -> Any:
        """Give LLM calls their own budget and preserve the workflow HITL path.

        Retrieval deadlines never shorten a model request. Time spent awaiting
        the model or a human is excluded from the remaining retrieval budget.
        """
        current_timeout = llm_timeout_seconds(timeout_seconds)
        while True:
            try:
                started = time.monotonic()
                try:
                    result = callback(*args, **kwargs)
                    if hasattr(result, "__await__"):
                        result = await asyncio.wait_for(result, timeout=current_timeout)
                    return result
                except asyncio.TimeoutError as exc:
                    raise UpstreamExhaustedError(
                        f"Direction literature {agent_name} call timed out after {current_timeout:g}s",
                        agent_name=agent_name, failure_kind="timeout",
                    ) from exc
                finally:
                    if pause_deadlines_fn is not None:
                        pause_deadlines_fn(max(0.0, time.monotonic() - started))
            except UpstreamExhaustedError as error:
                if failure_fn is not None:
                    failure_fn(error)
                if intervention_fn is None:
                    raise
                intervention_node = (
                    "pi_synthesis" if phase == "exploration"
                    else "integration_synthesis" if phase == "integration"
                    else phase
                )
                paused_at = time.monotonic()
                action = await intervention_fn(error, intervention_node, state)
                if pause_deadlines_fn is not None:
                    pause_deadlines_fn(max(0.0, time.monotonic() - paused_at))
                action_name = str(getattr(action, "value", action)).lower()
                if "retry" in action_name:
                    continue
                if "skip" in action_name:
                    return None
                if "terminate" in action_name:
                    from ...workflow.helpers import WorkflowTerminatedError
                    raise WorkflowTerminatedError(
                        f"User terminated direction literature search after {phase} LLM failure"
                    ) from error
                raise

    @staticmethod
    def _direction_article_from_raw(
        article: Dict[str, Any],
        *,
        source: str,
        task_id: str,
        query: str,
    ) -> DirectionArticleInventory:
        pmid = str(article.get("pmid") or article.get("pubmed_id") or "").strip()
        title = str(article.get("title") or "")
        return DirectionArticleInventory(
            pmid=pmid,
            title=title,
            url=str(article.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")),
            source="session_cache" if source == "session_cache" else "new_search",
            task_ids=[task_id],
            query_links=[query] if query else [],
            abstract=str(article.get("abstract") or ""),
            raw_source_cache_ref=(str(article.get("raw_source_cache_ref")) if article.get("raw_source_cache_ref") else None),
            authors=[str(a) for a in (article.get("authors") or [])] if isinstance(article.get("authors"), list) else [],
            journal=str(article.get("journal") or ""),
            year=(str(article.get("year")) if article.get("year") else None),
        )

    @staticmethod
    def _merge_direction_article(
        checkpoint: DirectionLiteratureInProgress,
        article: DirectionArticleInventory,
        task_id: str,
        query: str,
    ) -> None:
        existing = next((item for item in checkpoint.article_inventory if item.pmid == article.pmid), None)
        if existing is None:
            checkpoint.article_inventory.append(article)
            return
        if task_id not in existing.task_ids:
            existing.task_ids.append(task_id)
        if query and query not in existing.query_links:
            existing.query_links.append(query)
        if not existing.abstract and article.abstract:
            existing.abstract = article.abstract
        if existing.source == "new_search" and article.source == "session_cache":
            # Keep the first retrieval provenance if a live search later finds
            # the same PMID; this does not discard the task/query links.
            pass

    @staticmethod
    def _normalised_passage(value: str) -> str:
        return " ".join(str(value or "").split()).casefold()

    async def _extract_direction_search_articles(
        self,
        candidate: Dict[str, Any],
        task: DirectionSearchTask,
        articles: List[Dict[str, Any]],
    ) -> Tuple[List[DirectionLiteratureExtraction], List[str]]:
        """Extract source-linked observations while retaining raw inventory."""
        if not articles:
            return [], []
        article_payload = [
            {
                "pmid": str(a.get("pmid") or a.get("pubmed_id") or ""),
                "title": str(a.get("title") or ""),
                "abstract": str(a.get("abstract") or "")[:5000],
            }
            for a in articles
        ]
        prompt = f"""Extract bounded observations from the following PubMed articles for one PI-approved evidence question.
Direction: {candidate.get('direction', '')}
Evidence question: {task.evidence_question}
Mechanism link: {task.mechanism_link}
Articles (do not invent absent context):
{json.dumps(article_payload, ensure_ascii=False)}

Return JSON only: {{"items":[{{"pmid":"...","study_context":{{"species":"unknown","tissue":"unknown","cell_type":"unknown","perturbation":"unknown","design":"unknown","outcomes":"unknown"}},"finding":"...","source_passage":"a short exact passage from the abstract or empty if none","addressed_link":"...","applicability_limits":["..."],"provisional_classification":"direct|linked|background|conflicting|irrelevant|uncertain|unknown"}}]}}.
Copy source_passage as one contiguous substring, preserving internal text and punctuation.
Do not paraphrase, replace names with aliases, remove parentheticals, or add a terminal period.
Keep every PMID you can assess. Literature labels are provisional and never delete an article."""
        response = await self.chat_without_history(prompt, temperature=0.1)
        parsed = self._safe_parse_json(response, default={})
        if not isinstance(parsed, dict) or parsed.get("parse_error"):
            return [], [
                "Literature extraction response was not valid JSON; raw inventory retained."
            ]
        if "items" not in parsed:
            return [], [
                "Literature extraction response omitted the items field; raw inventory retained."
            ]
        raw_items = parsed.get("items")
        if not isinstance(raw_items, list):
            return [], ["Literature extraction returned no readable item list; raw inventory retained."]
        if not raw_items:
            return [], [
                "Literature extraction returned no items for retained articles; raw inventory retained."
            ]
        by_pmid = {
            str(a.get("pmid") or a.get("pubmed_id") or ""): a for a in articles
        }
        output: List[DirectionLiteratureExtraction] = []
        limitations: List[str] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                limitations.append("Literature extraction contained a non-object item.")
                continue
            pmid = str(raw.get("pmid") or "").strip()
            article = by_pmid.get(pmid)
            if article is None:
                limitations.append(f"Extraction referenced PMID {pmid or '<empty>'} outside the retained inventory.")
                continue
            passage = str(raw.get("source_passage") or "")
            abstract = str(article.get("abstract") or "")
            passage_status = "verified" if passage else "not_provided"
            passage_limitation = None
            if passage and self._normalised_passage(passage) not in self._normalised_passage(abstract):
                passage_status = "unverified"
                passage_limitation = f"Extraction passage for PMID {pmid} was not found in the retained abstract."
                limitations.append(passage_limitation)
            try:
                extraction = DirectionLiteratureExtraction.model_validate({
                    **raw,
                    "pmid": pmid,
                    "task_ids": [task.task_id],
                    "source_passage": passage,
                    "source_passage_status": passage_status,
                    "limitation": passage_limitation or raw.get("limitation"),
                })
            except Exception as exc:
                limitations.append(f"Invalid extraction for PMID {pmid}: {str(exc)[:140]}")
                continue
            output.append(extraction)
        if not output and raw_items:
            limitations.append("No valid source-linked extraction survived validation; raw inventory retained.")
        return output, list(dict.fromkeys(limitations))

    @staticmethod
    def _direction_prior_records(
        state: AnalysisState,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bound prior searches to the same stored candidate identity.

        Candidate IDs are reused by later gates, so an ID is relevant only when
        the persisted candidate direction also matches the current direction.
        """
        current = {
            (
                str(candidate.get("direction_id") or "").strip(),
                " ".join(str(candidate.get("direction") or "").split()).casefold(),
            )
            for candidate in candidates
            if str(candidate.get("direction_id") or "").strip()
            and str(candidate.get("direction") or "").strip()
        }
        prior: List[Dict[str, Any]] = []
        for record in reversed(list(getattr(state, "direction_selection_records", None) or [])):
            record_candidates = list(getattr(record, "candidates", None) or [])
            matching_ids = {
                str(candidate.get("direction_id") or "").strip()
                for candidate in record_candidates
                if isinstance(candidate, dict)
                and (
                    str(candidate.get("direction_id") or "").strip(),
                    " ".join(str(candidate.get("direction") or "").split()).casefold(),
                ) in current
            }
            if not matching_ids:
                continue
            evidence_items: List[Dict[str, Any]] = []
            for evidence in list(getattr(record, "literature_evidence", None) or []):
                if not isinstance(evidence, dict) or str(evidence.get("direction_id") or "") not in matching_ids:
                    continue
                evidence_items.append({
                    "direction_id": evidence.get("direction_id"),
                    "direction": evidence.get("direction"),
                    "policy_version": evidence.get("policy_version"),
                    "plan_status": evidence.get("plan_status"),
                    "assessment_status": evidence.get("assessment_status"),
                    "queries": list(evidence.get("queries") or [])[:12],
                    "task_executions": list(evidence.get("task_executions") or [])[:12],
                    "unexecuted_task_ids": list(evidence.get("unexecuted_task_ids") or [])[:12],
                    "assessment_limitations": list(evidence.get("assessment_limitations") or [])[:12],
                    "limitations": list(evidence.get("limitations") or [])[:12],
                    "summary": evidence.get("summary"),
                })
                if len(evidence_items) >= 24:
                    break
            if evidence_items:
                prior.append({
                    "record_id": str(getattr(record, "record_id", "") or ""),
                    "phase": str(getattr(record, "phase", "") or ""),
                    "round_number": int(getattr(record, "round_number", 0) or 0),
                    "integration_cycle": getattr(record, "integration_cycle", None),
                    "literature_evidence": evidence_items,
                })
            if len(prior) >= 8:
                break
        return prior

    @staticmethod
    def _direction_packet(
        candidates: List[Dict[str, Any]],
        checkpoint: DirectionLiteratureInProgress,
        *,
        candidate_deadlines: Dict[str, float],
        requested_detail_pmids: Optional[Set[str]] = None,
        included_direction_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        requested_detail_pmids = requested_detail_pmids or set()
        included_direction_ids = included_direction_ids or set(checkpoint.candidate_ids)
        by_id = {str(c.get("direction_id") or ""): c for c in candidates}
        packet_candidates: List[Dict[str, Any]] = []
        for direction_id in checkpoint.candidate_ids:
            if direction_id not in included_direction_ids:
                continue
            executions = [e for e in checkpoint.task_executions if e.direction_id == direction_id]
            task_ids = {e.task_id for e in executions}
            articles = [a for a in checkpoint.article_inventory if task_ids.intersection(a.task_ids)]
            extractions = [x for x in checkpoint.extractions if task_ids.intersection(x.task_ids)]
            # The packet is compact by construction, while the complete
            # inventory remains in the checkpoint and is available for named
            # detail requests.  IDs are always disclosed when detail is omitted.
            extraction_limit = 80
            packet_candidates.append({
                "direction_id": direction_id,
                "direction": by_id.get(direction_id, {}).get("direction", ""),
                "planned_tasks": [
                    task.model_dump(mode="json")
                    for task in checkpoint.tasks
                    if task.direction_id == direction_id
                ],
                "tasks": [e.model_dump(mode="json") for e in executions],
                "article_ids": [a.pmid for a in articles],
                "extractions": [x.model_dump(mode="json") for x in extractions[:extraction_limit]],
                "omitted_article_detail_count": len(articles),
                "omitted_extraction_detail_count": max(0, len(extractions) - extraction_limit),
                "remaining_budget_seconds": max(0.0, candidate_deadlines.get(direction_id, 0.0) - time.monotonic()),
                "technical_limitations": [
                    *checkpoint.limitations_by_candidate.get(direction_id, []),
                    *([checkpoint.planning_errors[direction_id]]
                      if direction_id in checkpoint.planning_errors else []),
                ],
                "unexecuted_task_ids": [
                    task_id for task_id in checkpoint.unexecuted_task_ids
                    if any(t.task_id == task_id and t.direction_id == direction_id for t in checkpoint.tasks)
                ],
            })
        included_task_ids = {
            task.task_id
            for task in checkpoint.tasks
            if task.direction_id in included_direction_ids
        }
        return {
            "policy_version": checkpoint.policy_version,
            "plan_id": checkpoint.plan_id,
            "plan_hash": checkpoint.plan_hash,
            "candidates": packet_candidates,
            # The first PI review receives every identity and retained link, but
            # no duplicate abstract bodies, except sources whose returned
            # quotation did not verify and needs direct PI inspection.
            "article_index": [
                {
                    "pmid": article.pmid,
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "task_ids": list(article.task_ids),
                    "query_links": list(article.query_links),
                }
                for article in checkpoint.article_inventory
                if included_task_ids.intersection(article.task_ids)
            ],
            "requested_article_details": [
                article.model_dump(mode="json")
                for article in checkpoint.article_inventory
                if (article.pmid in requested_detail_pmids or any(
                    extraction.pmid == article.pmid
                    and extraction.source_passage_status == "unverified"
                    and included_task_ids.intersection(extraction.task_ids)
                    for extraction in checkpoint.extractions
                )) and included_task_ids.intersection(article.task_ids)
            ],
            "remaining_unexecuted_task_ids": list(checkpoint.unexecuted_task_ids),
        }

    @staticmethod
    def _direction_review_map(
        result: Any,
        *,
        checkpoint: DirectionLiteratureInProgress,
        expected_candidate_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[DirectionPIReview], Dict[str, str]]:
        raw_reviews = result.get("reviews", []) if isinstance(result, dict) else result
        errors = result.get("review_errors", {}) if isinstance(result, dict) else {}
        if not isinstance(raw_reviews, list):
            raw_reviews = []
        reviews: List[DirectionPIReview] = []
        candidate_ids = expected_candidate_ids or set(checkpoint.candidate_ids)
        known_task_ids = {task.task_id for task in checkpoint.tasks}
        known_pmids_by_candidate: Dict[str, Set[str]] = {
            direction_id: set() for direction_id in checkpoint.candidate_ids
        }
        task_direction = {task.task_id: task.direction_id for task in checkpoint.tasks}
        for article in checkpoint.article_inventory:
            for task_id in article.task_ids:
                direction_id = task_direction.get(task_id)
                if direction_id is not None:
                    known_pmids_by_candidate.setdefault(direction_id, set()).add(article.pmid)
        seen_directions: Set[str] = set()
        seen_new_task_ids: Set[str] = set()
        for raw in raw_reviews:
            try:
                review = raw if isinstance(raw, DirectionPIReview) else DirectionPIReview.model_validate(raw)
                if review.direction_id not in candidate_ids:
                    raise ValueError(f"unknown candidate id {review.direction_id!r}")
                if review.direction_id in seen_directions:
                    raise ValueError(f"duplicate review for candidate {review.direction_id!r}")
                invalid_pmids = (
                    set(review.requested_detail_pmids) | set(review.cited_article_ids)
                ) - known_pmids_by_candidate.get(review.direction_id, set())
                if invalid_pmids:
                    raise ValueError(
                        f"article ids are not linked to candidate {review.direction_id!r}: "
                        f"{sorted(invalid_pmids)}"
                    )
                for task in review.follow_up_tasks:
                    if task.direction_id != review.direction_id:
                        raise ValueError(
                            f"follow-up task {task.task_id!r} is linked to candidate "
                            f"{task.direction_id!r}, expected {review.direction_id!r}"
                        )
                    if task.task_id in known_task_ids or task.task_id in seen_new_task_ids:
                        raise ValueError(f"follow-up task id {task.task_id!r} is not globally unique")
                    seen_new_task_ids.add(task.task_id)
                seen_directions.add(review.direction_id)
                reviews.append(review)
            except Exception as exc:
                if isinstance(raw, dict):
                    errors[str(raw.get("direction_id") or "")] = str(exc)[:240]
        for direction_id in candidate_ids - seen_directions:
            errors.setdefault(direction_id, "PI review unavailable for candidate")
        return reviews, dict(errors) if isinstance(errors, dict) else {}

    async def _execute_direction_task(
        self,
        checkpoint: DirectionLiteratureInProgress,
        task: DirectionSearchTask,
        compiled_query: str,
        candidate: Dict[str, Any],
        state: Optional[AnalysisState],
        *,
        deadline: float,
        batch: str,
        query_timeout_seconds: float,
        plan_hash: str,
        pause_deadlines_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        existing = next(
            (e for e in checkpoint.task_executions if e.task_id == task.task_id and e.batch == batch),
            None,
        )
        if existing is not None and existing.status in {
            "completed", "zero_result", "failed", "timeout", "unexecuted"
        }:
            return
        resume_extraction = existing is not None and existing.status == "partial"
        execution = existing or DirectionSearchExecution(
            task_id=task.task_id,
            direction_id=task.direction_id,
            compiled_query=compiled_query,
            status="running",
            batch="followup" if batch == "followup" else "initial",
        )
        if existing is None:
            checkpoint.task_executions.append(execution)
        else:
            execution.compiled_query = compiled_query
            execution.status = "running"
        started = time.monotonic()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                execution.status = "unexecuted"
                execution.budget_limitation = (
                    "candidate retrieval/extraction budget exhausted; PI review time reserved"
                )
                if task.task_id not in checkpoint.unexecuted_task_ids:
                    checkpoint.unexecuted_task_ids.append(task.task_id)
                return
            timeout = min(query_timeout_seconds, remaining)
            cache_key = (
                f"direction_search:{PI_DIRECTION_SEARCH_POLICY_VERSION}:{plan_hash}:"
                f"{task.direction_id}:{task.task_id}:{hashlib.sha256(compiled_query.encode()).hexdigest()}"
            )
            cache = getattr(state, "literature_search_cache", None) if state is not None else None
            cached_entry = cache.get(cache_key) if isinstance(cache, dict) else None
            articles: List[Dict[str, Any]]
            source = "new_search"
            retained_articles = [
                article.model_dump(mode="json")
                for article in checkpoint.article_inventory
                if task.task_id in article.task_ids
            ]
            resume_extraction = resume_extraction and bool(retained_articles)
            if resume_extraction:
                articles = retained_articles
                source = "session_cache" if all(
                    article.get("source") == "session_cache" for article in articles
                ) else "new_search"
            elif (
                isinstance(cached_entry, dict)
                and cached_entry.get("policy_version") == PI_DIRECTION_SEARCH_POLICY_VERSION
                and cached_entry.get("plan_hash") == plan_hash
                and cached_entry.get("candidate_question") == task.evidence_question
                and cached_entry.get("compiled_query") == compiled_query
                and isinstance(cached_entry.get("result"), list)
            ):
                articles = copy.deepcopy(cached_entry["result"])
                source = "session_cache"
            else:
                articles = await asyncio.wait_for(
                    self._pubmed_search(compiled_query, max_results=20),
                    timeout=timeout,
                )
                articles = list(articles or [])
                if isinstance(cache, dict):
                    cache[cache_key] = {
                        "policy_version": PI_DIRECTION_SEARCH_POLICY_VERSION,
                        "plan_hash": plan_hash,
                        "candidate_question": task.evidence_question,
                        "compiled_query": compiled_query,
                        "result": copy.deepcopy(articles),
                    }
            if not resume_extraction:
                execution.returned_count = len(articles)
                execution.total_hit_count = None
            seen_pmids: set[str] = set()
            raw_pmids = [
                str(a.get("pmid") or a.get("pubmed_id") or "")
                for a in articles
                if str(a.get("pmid") or a.get("pubmed_id") or "")
            ]
            if not resume_extraction:
                execution.duplicate_count = max(0, len(raw_pmids) - len(set(raw_pmids)))
            for article in articles:
                inventory = self._direction_article_from_raw(
                    article, source=source, task_id=task.task_id, query=compiled_query
                )
                if not inventory.pmid or inventory.pmid in seen_pmids:
                    continue
                seen_pmids.add(inventory.pmid)
                self._merge_direction_article(checkpoint, inventory, task.task_id, compiled_query)
            if not resume_extraction:
                execution.retrieved_pmids = raw_pmids[:20]
            execution.status = "zero_result" if not articles else "partial"
            if articles:
                extractions, limits = await self._direction_llm_call_with_intervention(
                    self._extract_direction_search_articles,
                    (candidate, task, articles), {},
                    state=state, phase=checkpoint.phase,
                    intervention_fn=None,
                    pause_deadlines_fn=pause_deadlines_fn,
                    agent_name="literature",
                )
                checkpoint.extractions = [
                    item for item in checkpoint.extractions
                    if task.task_id not in item.task_ids
                ]
                checkpoint.extractions.extend(extractions)
                for limitation in limits:
                    self._add_direction_limitation(
                        checkpoint, task.direction_id, limitation
                    )
                execution.status = "partial" if limits else "completed"
                if not limits and task.task_id in checkpoint.unexecuted_task_ids:
                    checkpoint.unexecuted_task_ids.remove(task.task_id)
        except asyncio.TimeoutError:
            execution.status = "timeout"
            execution.budget_limitation = "task retrieval or extraction timed out within shared candidate deadline"
            if task.task_id not in checkpoint.unexecuted_task_ids:
                checkpoint.unexecuted_task_ids.append(task.task_id)
            self._add_direction_limitation(
                checkpoint,
                task.direction_id,
                f"Task {task.task_id} timed out; any retrieved raw inventory was retained.",
            )
        except asyncio.CancelledError:
            raise
        except UpstreamExhaustedError:
            raise
        except Exception as exc:
            execution.status = "failed"
            execution.error = str(exc)[:400]
            self._add_direction_limitation(
                checkpoint,
                task.direction_id,
                f"Task {task.task_id} failed technically: {str(exc)[:180]}"
            )
        finally:
            execution.elapsed_seconds = max(0.0, time.monotonic() - started)
            checkpoint.spent_seconds += execution.elapsed_seconds
            self._save_direction_checkpoint(state, checkpoint)

    async def evaluate_direction_literature_search(
        self,
        candidates: List[Dict[str, Any]],
        state: Optional[AnalysisState] = None,
        *,
        pi_plan_fn: Optional[Callable[..., Any]] = None,
        pi_review_fn: Optional[Callable[..., Any]] = None,
        intervention_fn: Optional[Callable[..., Any]] = None,
        phase: str = "exploration",
        query_timeout_seconds: float = NOVELTY_QUERY_TIMEOUT_SECONDS,
        direction_timeout_seconds: float = NOVELTY_DIRECTION_TIMEOUT_SECONDS,
        max_full_evidence_candidates: int = 8,
        concurrency: int = NOVELTY_CANDIDATE_CONCURRENCY,
    ) -> List[Dict[str, Any]]:
        """Execute one bounded PI-plan/retrieve/extract/review direction gate."""
        if not candidates:
            return []
        candidate_ids = [str(c.get("direction_id") or "").strip() for c in candidates]
        identity_errors: Dict[str, str] = {}
        seen_ids: set[str] = set()
        for index, direction_id in enumerate(candidate_ids):
            if not direction_id:
                identity_errors[f"__index_{index}"] = "candidate direction_id is empty"
            elif direction_id in seen_ids:
                identity_errors[direction_id] = "candidate direction_id is duplicated"
            seen_ids.add(direction_id)

        gate_id = self._direction_gate_id(candidates, state, phase)
        checkpoint = self._load_direction_checkpoint(state, gate_id, candidates, phase)
        reviews = {review.direction_id: review for review in checkpoint.pi_reviews}
        # Older checkpoints have no explicit task batches. Recover them from
        # accepted PI requests and receipts, never from list position alone.
        requested_followups = {
            task.task_id
            for review in checkpoint.pi_reviews
            for task in review.follow_up_tasks
        }
        first_followups: Dict[str, Set[str]] = {}
        retained_task_ids = {task.task_id for task in checkpoint.tasks}
        for review in checkpoint.pi_reviews:
            if review.action == "search_more" and any(
                task.task_id in retained_task_ids for task in review.follow_up_tasks
            ):
                first_followups.setdefault(
                    review.direction_id, {task.task_id for task in review.follow_up_tasks}
                )
        executed_followups = {
            execution.task_id for execution in checkpoint.task_executions
            if execution.batch == "followup"
        }
        for task in checkpoint.tasks:
            if task.task_id in checkpoint.task_batches:
                continue
            if task.task_id in executed_followups or (
                checkpoint.followup_used_by_candidate.get(task.direction_id)
                and task.task_id in first_followups.get(task.direction_id, set())
                and task.task_id not in checkpoint.unexecuted_task_ids
            ):
                checkpoint.task_batches[task.task_id] = "followup"
            elif task.task_id in requested_followups:
                checkpoint.task_batches[task.task_id] = "forbidden"
            else:
                checkpoint.task_batches[task.task_id] = "initial"
        # Candidate windows are established once for this gate.  A workflow
        # Retry resumes against these wall-clock deadlines; it never receives a
        # fresh 420/150 second allowance merely because the caller re-entered
        # the coordinator.
        now_epoch = time.time()
        for index, direction_id in enumerate(candidate_ids):
            if not direction_id or direction_id in checkpoint.candidate_deadline_epoch:
                continue
            window = (
                direction_timeout_seconds
                if index < max_full_evidence_candidates
                else min(direction_timeout_seconds, getattr(self, "MIN_ATTEMPT_DIRECTION_TIMEOUT", 150.0))
            )
            checkpoint.candidate_deadline_epoch[direction_id] = now_epoch + window
        self._save_direction_checkpoint(state, checkpoint)
        monotonic_now = time.monotonic()
        candidate_deadlines = {
            direction_id: monotonic_now + max(0.0, deadline_epoch - time.time())
            for direction_id, deadline_epoch in checkpoint.candidate_deadline_epoch.items()
            if direction_id in candidate_ids
        }
        gate_started = monotonic_now
        review_reserves = {
            direction_id: min(
                PI_DIRECTION_REVIEW_RESERVE_SECONDS,
                (
                    direction_timeout_seconds
                    if index < max_full_evidence_candidates
                    else min(direction_timeout_seconds, getattr(self, "MIN_ATTEMPT_DIRECTION_TIMEOUT", 150.0))
                ) / 2.0,
            )
            for index, direction_id in enumerate(candidate_ids)
        }

        def work_deadline(direction_id: str) -> float:
            return candidate_deadlines.get(direction_id, gate_started) - review_reserves[direction_id]

        def shift_deadlines(paused_seconds: float) -> None:
            if paused_seconds <= 0:
                return
            for direction_id in candidate_deadlines:
                candidate_deadlines[direction_id] += paused_seconds
                if direction_id in checkpoint.candidate_deadline_epoch:
                    checkpoint.candidate_deadline_epoch[direction_id] += paused_seconds
            self._save_direction_checkpoint(state, checkpoint)

        def shift_candidate_deadline(direction_id: str, excluded_seconds: float) -> None:
            if excluded_seconds <= 0 or direction_id not in candidate_deadlines:
                return
            candidate_deadlines[direction_id] += excluded_seconds
            if direction_id in checkpoint.candidate_deadline_epoch:
                checkpoint.candidate_deadline_epoch[direction_id] += excluded_seconds

        async def intervene_after_settled(error: UpstreamExhaustedError) -> str:
            if intervention_fn is None:
                raise error
            intervention_node = (
                "pi_synthesis" if phase == "exploration"
                else "integration_synthesis" if phase == "integration"
                else phase
            )
            paused_at = time.monotonic()
            action = await intervention_fn(error, intervention_node, state)
            shift_deadlines(max(0.0, time.monotonic() - paused_at))
            action_name = str(getattr(action, "value", action)).lower()
            if "terminate" in action_name:
                from ...workflow.helpers import WorkflowTerminatedError
                raise WorkflowTerminatedError(
                    f"User terminated direction literature search after {phase} LLM failure"
                ) from error
            return action_name

        # Plan once per stable gate. A retry resumes from the persisted plan and
        # never silently replaces it with a template query.
        planning_skipped = False
        if not checkpoint.tasks:
            plan_result: Any = None
            if pi_plan_fn is not None:
                plan_started = time.monotonic()

                def record_planning_failure(error: UpstreamExhaustedError) -> None:
                    checkpoint.plan_status = "failed"
                    for direction_id in candidate_ids:
                        checkpoint.planning_errors[direction_id] = str(error)[:300]
                    self._save_direction_checkpoint(state, checkpoint)

                try:
                    plan_result = await self._direction_llm_call_with_intervention(
                        pi_plan_fn,
                        (state, candidates),
                        {
                            "prior_records": self._direction_prior_records(state, candidates),
                            "phase": phase,
                        },
                        state=state,
                        phase=phase,
                        intervention_fn=intervention_fn,
                        timeout_seconds=PI_DIRECTION_LLM_TIMEOUT_SECONDS,
                        pause_deadlines_fn=shift_deadlines,
                        failure_fn=record_planning_failure,
                    )
                except (asyncio.CancelledError, UpstreamExhaustedError):
                    raise
                except Exception as exc:
                    if exc.__class__.__name__ in {
                        "WorkflowSupersededError", "WorkflowTerminatedError"
                    }:
                        raise
                    checkpoint.plan_status = "failed"
                    for direction_id in candidate_ids:
                        checkpoint.planning_errors[direction_id] = (
                            f"PI direction-search planning failed technically: {str(exc)[:220]}"
                        )
                    checkpoint.spent_seconds += max(
                        0.0, time.monotonic() - plan_started
                    )
                    plan_result = None
                else:
                    checkpoint.spent_seconds += max(0.0, time.monotonic() - plan_started)
            if plan_result is None:
                planning_skipped = True
                if checkpoint.plan_status != "failed":
                    checkpoint.plan_status = "unavailable"
                for direction_id in candidate_ids:
                    checkpoint.planning_errors.setdefault(
                        direction_id, "PI direction-search plan was unavailable"
                    )
            else:
                checkpoint.planning_errors.clear()
                raw_tasks = plan_result.get("tasks", []) if isinstance(plan_result, dict) else plan_result
                if not isinstance(raw_tasks, list):
                    raw_tasks = []
                valid_tasks: List[DirectionSearchTask] = []
                planning_errors = plan_result.get("planning_errors", {}) if isinstance(plan_result, dict) else {}
                if not isinstance(planning_errors, dict):
                    planning_errors = {}
                for raw_task in raw_tasks:
                    try:
                        task = raw_task if isinstance(raw_task, DirectionSearchTask) else DirectionSearchTask.model_validate(raw_task)
                        if task.direction_id not in set(candidate_ids):
                            raise DirectionSearchCompilationError(
                                f"unknown candidate id {task.direction_id!r}"
                            )
                        valid_tasks.append(task)
                    except Exception as exc:
                        direction_id = str(raw_task.get("direction_id") or "") if isinstance(raw_task, dict) else ""
                        planning_errors[direction_id] = str(exc)[:240]
                # Unique tasks and candidate identity are host technical facts.
                unique: Dict[str, DirectionSearchTask] = {}
                for task in valid_tasks:
                    if task.task_id not in unique:
                        unique[task.task_id] = task
                    else:
                        planning_errors[task.direction_id] = f"duplicate task id {task.task_id!r}"
                checkpoint.tasks = list(unique.values())
                checkpoint.task_batches.update({task.task_id: "initial" for task in checkpoint.tasks})
                for direction_id in candidate_ids:
                    if direction_id and not any(
                        task.direction_id == direction_id for task in checkpoint.tasks
                    ):
                        planning_errors.setdefault(
                            direction_id, "PI returned no valid search task"
                        )
                checkpoint.plan_id = str(plan_result.get("plan_id") or "") if isinstance(plan_result, dict) else ""
                checkpoint.plan_hash = self._direction_plan_hash(checkpoint.tasks)
                checkpoint.plan_status = "completed" if checkpoint.tasks and not planning_errors else "partial"
                checkpoint.planning_errors.update({
                    str(direction_id): str(error)[:300]
                    for direction_id, error in planning_errors.items()
                    if direction_id
                })
            self._save_direction_checkpoint(state, checkpoint)

        # If the checkpoint was created by an earlier retry, restore its plan
        # hash before compiling; no query plan is inferred from old fields.
        if checkpoint.tasks and not checkpoint.plan_hash:
            checkpoint.plan_hash = self._direction_plan_hash(checkpoint.tasks)

        task_by_candidate: Dict[str, List[DirectionSearchTask]] = {direction_id: [] for direction_id in candidate_ids}
        for task in checkpoint.tasks:
            if checkpoint.task_batches.get(task.task_id) == "initial":
                task_by_candidate.setdefault(task.direction_id, []).append(task)
        compiled: Dict[str, str] = {}
        compile_errors: Dict[str, str] = {}
        try:
            # Validate the whole initial plan once so a malformed task never
            # dispatches a partially degraded query.
            initial_tasks = [task for tasks in task_by_candidate.values() for task in tasks]
            compiled = compile_direction_search_plan(initial_tasks, candidate_ids=set(candidate_ids))
        except Exception:
            # Preserve valid tasks and identify only the task-level failures.
            compiled = {}
            for task in initial_tasks:
                try:
                    compiled[task.task_id] = compile_direction_search_task(task, candidate_ids=set(candidate_ids))
                except Exception as exc:
                    compile_errors[task.task_id] = str(exc)[:300]
            checkpoint.compile_errors.update(compile_errors)

        for direction_id, tasks_for_candidate in task_by_candidate.items():
            for task in tasks_for_candidate[PI_DIRECTION_MAX_INITIAL_TASKS:]:
                if task.task_id not in checkpoint.unexecuted_task_ids:
                    checkpoint.unexecuted_task_ids.append(task.task_id)
            for task in tasks_for_candidate[:PI_DIRECTION_MAX_INITIAL_TASKS]:
                if task.task_id in compile_errors and task.task_id not in checkpoint.unexecuted_task_ids:
                    checkpoint.unexecuted_task_ids.append(task.task_id)

        by_id = {str(c.get("direction_id") or ""): c for c in candidates}
        semaphore = asyncio.Semaphore(max(1, concurrency))
        candidate_ready_at: Dict[str, float] = {}
        candidate_queued_at: Dict[str, float] = {}

        async def run_candidate(index: int, direction_id: str) -> None:
            if direction_id in reviews:
                candidate_ready_at[direction_id] = time.monotonic()
                return
            async with semaphore:
                queued_at = candidate_queued_at.get(direction_id, time.monotonic())
                shift_candidate_deadline(
                    direction_id, max(0.0, time.monotonic() - queued_at)
                )
                try:
                    candidate = by_id.get(direction_id, {})
                    task_limit = PI_DIRECTION_MAX_INITIAL_TASKS if index < max_full_evidence_candidates else 1
                    tasks_for_candidate = task_by_candidate.get(direction_id, [])
                    for deferred_task in tasks_for_candidate[task_limit:]:
                        if deferred_task.task_id not in checkpoint.unexecuted_task_ids:
                            checkpoint.unexecuted_task_ids.append(deferred_task.task_id)
                        self._add_direction_limitation(
                            checkpoint,
                            direction_id,
                            f"Task {deferred_task.task_id} was left unexecuted by the candidate scheduling ceiling.",
                        )
                    for task in tasks_for_candidate[:task_limit]:
                        if task.task_id in compile_errors:
                            self._add_direction_limitation(
                                checkpoint,
                                direction_id,
                                f"Task {task.task_id} was not executed: {compile_errors[task.task_id]}"
                            )
                            continue
                        await self._execute_direction_task(
                            checkpoint,
                            task,
                            compiled[task.task_id],
                            candidate,
                            state,
                            deadline=work_deadline(direction_id),
                            batch="initial",
                            query_timeout_seconds=(
                                query_timeout_seconds
                                if index < max_full_evidence_candidates
                                else min(
                                    query_timeout_seconds,
                                    getattr(self, "MIN_ATTEMPT_QUERY_TIMEOUT", 60.0),
                                )
                            ),
                            plan_hash=checkpoint.plan_hash,
                            pause_deadlines_fn=lambda elapsed, direction_id=direction_id: shift_candidate_deadline(
                                direction_id, elapsed
                            ),
                        )
                finally:
                    candidate_ready_at[direction_id] = time.monotonic()

        retrieval_skipped = planning_skipped
        while True:
            queued_now = time.monotonic()
            candidate_queued_at = {
                direction_id: queued_now for direction_id in candidate_ids if direction_id
            }
            tasks = [
                asyncio.create_task(run_candidate(index, direction_id))
                for index, direction_id in enumerate(candidate_ids)
                if direction_id
            ]
            try:
                await asyncio.gather(*tasks)
                break
            except UpstreamExhaustedError as error:
                for running_task in tasks:
                    if not running_task.done():
                        running_task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._save_direction_checkpoint(state, checkpoint)
                action_name = await intervene_after_settled(error)
                if "retry" in action_name:
                    continue
                if "skip" in action_name:
                    retrieval_skipped = True
                    for execution in checkpoint.task_executions:
                        if execution.status in {"running", "partial"}:
                            execution.status = "partial"
                            if execution.task_id not in checkpoint.unexecuted_task_ids:
                                checkpoint.unexecuted_task_ids.append(execution.task_id)
                            self._add_direction_limitation(
                                checkpoint,
                                execution.direction_id,
                                "Direction literature extraction was skipped after an upstream failure; retained raw evidence remains incomplete.",
                            )
                    break
                raise
            except BaseException:
                for running_task in tasks:
                    if not running_task.done():
                        running_task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._save_direction_checkpoint(state, checkpoint)
                raise

        # A candidate's allowance pays for its own planning/retrieval/extraction
        # work, not time spent waiting for sibling candidates to settle before a
        # batched PI review.
        review_ready_at = time.monotonic()
        for direction_id, ready_at in candidate_ready_at.items():
            shift_candidate_deadline(
                direction_id, max(0.0, review_ready_at - ready_at)
            )

        self._save_direction_checkpoint(state, checkpoint)
        review_errors: Dict[str, str] = {}
        full_candidate_ids = set(candidate_ids[:max_full_evidence_candidates])

        def review_cohorts(direction_ids: Set[str]) -> List[Set[str]]:
            # Short minimum-attempt windows must not shorten the PI call for
            # full-evidence candidates. Keep each scheduling class batched.
            overflow = direction_ids - full_candidate_ids
            full = direction_ids & full_candidate_ids
            return [cohort for cohort in (overflow, full) if cohort]

        active_review_ids = {
            direction_id
            for direction_id in candidate_ids
            if direction_id not in reviews
        }
        if pi_review_fn is not None and not retrieval_skipped and active_review_ids:
            pending_cohorts = review_cohorts(active_review_ids)
            for cohort_index, cohort_ids in enumerate(pending_cohorts):
                cohort_started = time.monotonic()
                review_packet = self._direction_packet(
                    candidates,
                    checkpoint,
                    candidate_deadlines=candidate_deadlines,
                    included_direction_ids=cohort_ids,
                )
                try:
                    review_result = await self._direction_llm_call_with_intervention(
                        pi_review_fn,
                        (state, review_packet),
                        {"phase": phase},
                        state=state,
                        phase=phase,
                        intervention_fn=intervention_fn,
                        timeout_seconds=PI_DIRECTION_LLM_TIMEOUT_SECONDS,
                        pause_deadlines_fn=shift_deadlines,
                    )
                except (asyncio.CancelledError, UpstreamExhaustedError):
                    raise
                except Exception as exc:
                    if exc.__class__.__name__ in {
                        "WorkflowSupersededError", "WorkflowTerminatedError"
                    }:
                        raise
                    review_result = None
                    for direction_id in cohort_ids:
                        review_errors[direction_id] = (
                            f"PI review failed technically: {str(exc)[:220]}"
                        )
                finally:
                    cohort_elapsed = max(0.0, time.monotonic() - cohort_started)
                    checkpoint.spent_seconds += cohort_elapsed

                if review_result is not None:
                    parsed_reviews, parsed_review_errors = self._direction_review_map(
                        review_result,
                        checkpoint=checkpoint,
                        expected_candidate_ids=cohort_ids,
                    )
                    review_errors.update(parsed_review_errors)
                    for review in parsed_reviews:
                        if review.direction_id in by_id:
                            reviews[review.direction_id] = review
                            checkpoint.pi_reviews.append(review)
                    self._save_direction_checkpoint(state, checkpoint)
        elif not retrieval_skipped:
            review_errors.update({
                direction_id: "PI review callback unavailable"
                for direction_id in active_review_ids
            })
        if retrieval_skipped:
            review_errors.update({
                direction_id: "PI review was skipped after an upstream failure"
                for direction_id in candidate_ids
            })

        provided_detail_pmids_by_candidate = {
            direction_id: set(pmids)
            for direction_id, pmids in checkpoint.provided_detail_pmids_by_candidate.items()
        }

        async def resolve_requested_details() -> None:
            """Re-present only named raw sources inside the existing deadline."""
            if pi_review_fn is None or retrieval_skipped:
                return
            while True:
                pending_by_candidate = {
                    review.direction_id: (
                        set(review.requested_detail_pmids)
                        - provided_detail_pmids_by_candidate.get(review.direction_id, set())
                    )
                    for review in reviews.values()
                    if set(review.requested_detail_pmids)
                    - provided_detail_pmids_by_candidate.get(review.direction_id, set())
                }
                if not pending_by_candidate:
                    return
                detail_direction_ids = {
                    direction_id for direction_id in pending_by_candidate
                }
                detail_cohorts = review_cohorts(detail_direction_ids)
                for cohort_index, cohort_ids in enumerate(detail_cohorts):
                    requested_detail_pmids = set().union(*(
                        pending_by_candidate[direction_id]
                        for direction_id in cohort_ids
                    ))
                    for direction_id in cohort_ids:
                        provided_detail_pmids_by_candidate.setdefault(
                            direction_id, set()
                        ).update(pending_by_candidate[direction_id])
                    detail_packet = self._direction_packet(
                        candidates,
                        checkpoint,
                        candidate_deadlines=candidate_deadlines,
                        requested_detail_pmids=requested_detail_pmids,
                        included_direction_ids=cohort_ids,
                    )
                    detail_started = time.monotonic()
                    try:
                        detail_result = await self._direction_llm_call_with_intervention(
                            pi_review_fn,
                            (state, detail_packet),
                            {"phase": phase},
                            state=state,
                            phase=phase,
                            intervention_fn=intervention_fn,
                            timeout_seconds=PI_DIRECTION_LLM_TIMEOUT_SECONDS,
                            pause_deadlines_fn=shift_deadlines,
                        )
                    except (asyncio.CancelledError, UpstreamExhaustedError):
                        raise
                    except Exception as exc:
                        if exc.__class__.__name__ in {
                            "WorkflowSupersededError", "WorkflowTerminatedError"
                        }:
                            raise
                        detail_result = None
                        for direction_id in cohort_ids:
                            review_errors[direction_id] = (
                                f"PI source-detail review failed technically: {str(exc)[:220]}"
                            )
                    finally:
                        detail_elapsed = max(0.0, time.monotonic() - detail_started)
                        checkpoint.spent_seconds += detail_elapsed

                    if detail_result is None:
                        continue
                    detail_reviews, detail_errors = self._direction_review_map(
                        detail_result,
                        checkpoint=checkpoint,
                        expected_candidate_ids=cohort_ids,
                    )
                    review_errors.update(detail_errors)
                    for detail_review in detail_reviews:
                        reviews[detail_review.direction_id] = detail_review
                        checkpoint.pi_reviews.append(detail_review)
                        review_errors.pop(detail_review.direction_id, None)
                        checkpoint.provided_detail_pmids_by_candidate[detail_review.direction_id] = sorted(
                            provided_detail_pmids_by_candidate[detail_review.direction_id]
                        )
                    self._save_direction_checkpoint(state, checkpoint)

        # A PI may inspect retained abstracts by PMID before its action is
        # accepted. Each round must name a previously unseen PMID and receives
        # a separate model-call budget without extending retrieval work.
        await resolve_requested_details()

        # One and only one follow-up batch per candidate. Review requests after
        # that cap are retained as unexecuted tasks and disclosed in limitations.
        followup_requests = [
            review for review in reviews.values()
            if review.action == "search_more" and review.follow_up_tasks
            and (
                not checkpoint.followup_used_by_candidate.get(review.direction_id)
                or any(
                    checkpoint.task_batches.get(task.task_id) == "followup"
                    for task in review.follow_up_tasks
                )
            )
        ]
        followup_wait_started = time.monotonic()
        followup_ready_at: Dict[str, float] = {}
        for review in followup_requests:
            direction_id = review.direction_id
            shift_candidate_deadline(
                direction_id, max(0.0, time.monotonic() - followup_wait_started)
            )
            candidate_index = candidate_ids.index(direction_id)
            if candidate_index >= max_full_evidence_candidates:
                for task in review.follow_up_tasks:
                    if task not in checkpoint.tasks:
                        checkpoint.tasks.append(task)
                    checkpoint.task_batches[task.task_id] = "forbidden"
                    if task.task_id not in checkpoint.unexecuted_task_ids:
                        checkpoint.unexecuted_task_ids.append(task.task_id)
                self._add_direction_limitation(
                    checkpoint,
                    direction_id,
                    "Follow-up search was left unexecuted by the candidate scheduling ceiling.",
                )
                followup_ready_at[direction_id] = time.monotonic()
                continue
            checkpoint.followup_used_by_candidate[direction_id] = True
            followups = review.follow_up_tasks[:PI_DIRECTION_MAX_FOLLOWUP_TASKS]
            for task in review.follow_up_tasks[PI_DIRECTION_MAX_FOLLOWUP_TASKS:]:
                if not any(existing.task_id == task.task_id for existing in checkpoint.tasks):
                    checkpoint.tasks.append(task)
                checkpoint.task_batches[task.task_id] = "forbidden"
                if task.task_id not in checkpoint.unexecuted_task_ids:
                    checkpoint.unexecuted_task_ids.append(task.task_id)
            for task in followups:
                if not any(existing.task_id == task.task_id for existing in checkpoint.tasks):
                    checkpoint.tasks.append(task)
                checkpoint.task_batches.setdefault(task.task_id, "followup")
            self._save_direction_checkpoint(state, checkpoint)
            for task in followups:
                if checkpoint.task_batches.get(task.task_id) != "followup":
                    continue
                try:
                    query = compile_direction_search_task(task, candidate_ids=set(candidate_ids))
                except Exception as exc:
                    checkpoint.unexecuted_task_ids.append(task.task_id)
                    self._add_direction_limitation(
                        checkpoint,
                        direction_id,
                        f"Follow-up task {task.task_id} was not executed: {str(exc)[:180]}"
                    )
                    continue
                if time.monotonic() >= candidate_deadlines.get(direction_id, 0.0):
                    checkpoint.unexecuted_task_ids.append(task.task_id)
                    self._add_direction_limitation(
                        checkpoint,
                        direction_id,
                        f"Follow-up task {task.task_id} was left unexecuted after the shared deadline."
                    )
                    continue
                while True:
                    try:
                        await self._execute_direction_task(
                            checkpoint,
                            task,
                            query,
                            by_id[direction_id],
                            state,
                            deadline=work_deadline(direction_id),
                            batch="followup",
                            query_timeout_seconds=query_timeout_seconds,
                            plan_hash=checkpoint.plan_hash,
                            pause_deadlines_fn=lambda elapsed, direction_id=direction_id: shift_candidate_deadline(
                                direction_id, elapsed
                            ),
                        )
                        break
                    except UpstreamExhaustedError as error:
                        action_name = await intervene_after_settled(error)
                        if "retry" in action_name:
                            continue
                        if "skip" in action_name:
                            execution = next(
                                (
                                    item for item in checkpoint.task_executions
                                    if item.task_id == task.task_id and item.batch == "followup"
                                ),
                                None,
                            )
                            if execution is not None:
                                execution.status = "partial"
                            if task.task_id not in checkpoint.unexecuted_task_ids:
                                checkpoint.unexecuted_task_ids.append(task.task_id)
                            self._add_direction_limitation(
                                checkpoint,
                                direction_id,
                                f"Follow-up task {task.task_id} was skipped after an upstream failure; retained raw evidence remains incomplete.",
                            )
                            break
                        raise
            followup_ready_at[direction_id] = time.monotonic()

        if followup_requests:
            final_ready_at = time.monotonic()
            for direction_id, ready_at in followup_ready_at.items():
                shift_candidate_deadline(
                    direction_id, max(0.0, final_ready_at - ready_at)
                )
            self._save_direction_checkpoint(state, checkpoint)
            final_review_ids = {
                review.direction_id
                for review in followup_requests
            }
            if pi_review_fn is not None and final_review_ids:
                final_cohorts = review_cohorts(final_review_ids)
                final_reviews: List[DirectionPIReview] = []
                for cohort_index, cohort_ids in enumerate(final_cohorts):
                    final_packet = self._direction_packet(
                        candidates,
                        checkpoint,
                        candidate_deadlines=candidate_deadlines,
                        included_direction_ids=cohort_ids,
                    )
                    final_started = time.monotonic()
                    try:
                        final_result = await self._direction_llm_call_with_intervention(
                            pi_review_fn,
                            (state, final_packet),
                            {"phase": phase},
                            state=state,
                            phase=phase,
                            intervention_fn=intervention_fn,
                            timeout_seconds=PI_DIRECTION_LLM_TIMEOUT_SECONDS,
                            pause_deadlines_fn=shift_deadlines,
                        )
                    except (asyncio.CancelledError, UpstreamExhaustedError):
                        raise
                    except Exception as exc:
                        if exc.__class__.__name__ in {
                            "WorkflowSupersededError", "WorkflowTerminatedError"
                        }:
                            raise
                        final_result = None
                        for direction_id in cohort_ids:
                            review_errors[direction_id] = (
                                f"Final PI review failed technically: {str(exc)[:220]}"
                            )
                    finally:
                        final_elapsed = max(0.0, time.monotonic() - final_started)
                        checkpoint.spent_seconds += final_elapsed

                    if final_result is None:
                        continue
                    cohort_reviews, final_errors = self._direction_review_map(
                        final_result,
                        checkpoint=checkpoint,
                        expected_candidate_ids=cohort_ids,
                    )
                    final_reviews.extend(cohort_reviews)
                    review_errors.update(final_errors)
                    for review in cohort_reviews:
                        reviews[review.direction_id] = review
                        checkpoint.pi_reviews.append(review)
                        review_errors.pop(review.direction_id, None)
                        for task in review.follow_up_tasks:
                            checkpoint.task_batches[task.task_id] = "forbidden"
                    self._save_direction_checkpoint(state, checkpoint)
                for review in final_reviews:
                    if review.action == "search_more" and review.follow_up_tasks:
                        for task in review.follow_up_tasks:
                            if not any(
                                existing.task_id == task.task_id
                                for existing in checkpoint.tasks
                            ):
                                checkpoint.tasks.append(task)
                            checkpoint.task_batches[task.task_id] = "forbidden"
                            if task.task_id not in checkpoint.unexecuted_task_ids:
                                checkpoint.unexecuted_task_ids.append(task.task_id)
                        self._add_direction_limitation(
                            checkpoint,
                            review.direction_id,
                            "PI requested additional search after the one-follow-up ceiling; work remains unexecuted."
                        )
                await resolve_requested_details()
                for review in reviews.values():
                    if review.action != "search_more" or not review.follow_up_tasks:
                        continue
                    for task in review.follow_up_tasks:
                        if not any(
                            existing.task_id == task.task_id
                            for existing in checkpoint.tasks
                        ):
                            checkpoint.tasks.append(task)
                        checkpoint.task_batches[task.task_id] = "forbidden"
                        if task.task_id not in checkpoint.unexecuted_task_ids:
                            checkpoint.unexecuted_task_ids.append(task.task_id)
                    self._add_direction_limitation(
                        checkpoint,
                        review.direction_id,
                        "PI requested additional search after the one-follow-up ceiling; work remains unexecuted.",
                    )
        # Also retain rejected requests restored between final review and its
        # task bookkeeping. They must never become a new initial batch.
        for review in reviews.values():
            for task in review.follow_up_tasks:
                if checkpoint.task_batches.get(task.task_id) != "forbidden":
                    continue
                if not any(existing.task_id == task.task_id for existing in checkpoint.tasks):
                    checkpoint.tasks.append(task)
                if task.task_id not in checkpoint.unexecuted_task_ids:
                    checkpoint.unexecuted_task_ids.append(task.task_id)
        self._save_direction_checkpoint(state, checkpoint)

        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            direction_id = str(candidate.get("direction_id") or "")
            direction_name = str(candidate.get("direction") or "")
            task_ids = {task.task_id for task in checkpoint.tasks if task.direction_id == direction_id}
            executions = [e for e in checkpoint.task_executions if e.direction_id == direction_id]
            articles = [a for a in checkpoint.article_inventory if task_ids.intersection(a.task_ids)]
            extractions = [x for x in checkpoint.extractions if task_ids.intersection(x.task_ids)]
            review = reviews.get(direction_id)
            final_review = (
                review
                if review is not None
                and review.action == "finish"
                and direction_id not in review_errors
                and not (
                    set(review.requested_detail_pmids)
                    - set(checkpoint.provided_detail_pmids_by_candidate.get(direction_id, []))
                )
                else None
            )
            limitations = list(dict.fromkeys([
                *checkpoint.limitations_by_candidate.get(direction_id, []),
                *([checkpoint.planning_errors[direction_id]] if direction_id in checkpoint.planning_errors else []),
                *([review_errors[direction_id]] if direction_id in review_errors else []),
            ]))
            if direction_id in identity_errors:
                limitations.append(identity_errors[direction_id])
            result = self._build_empty_direction_result(direction_id, direction_name)
            result.update({
                "status": "completed" if final_review and not limitations and not any(
                    e.status in {"failed", "timeout", "unexecuted"} for e in executions
                ) else "partial",
                "policy_version": checkpoint.policy_version,
                "plan_id": checkpoint.plan_id,
                "plan_hash": checkpoint.plan_hash,
                "plan_status": checkpoint.plan_status,
                "tasks": [task.model_dump(mode="json") for task in checkpoint.tasks if task.direction_id == direction_id],
                "task_executions": [e.model_dump(mode="json") for e in executions],
                "article_inventory": [a.model_dump(mode="json") for a in articles],
                "extractions": [x.model_dump(mode="json") for x in extractions],
                "pi_reviews": [
                    item.model_dump(mode="json")
                    for item in checkpoint.pi_reviews
                    if item.direction_id == direction_id
                ],
                "assessment_status": final_review.assessment_status if final_review else "unavailable",
                "assessment_uncertainty": list(final_review.uncertainty) if final_review else [],
                "assessment_limitations": limitations,
                "unexecuted_task_ids": [
                    task_id for task_id in checkpoint.unexecuted_task_ids
                    if any(task.task_id == task_id and task.direction_id == direction_id for task in checkpoint.tasks)
                ],
                "queries": [
                    {
                        "category": "pi_directed",
                        "query": execution.compiled_query,
                        "result_count": execution.returned_count,
                        "duplicate_count": execution.duplicate_count,
                        "degraded": execution.status in {"failed", "timeout", "unexecuted"},
                        "notes": execution.error or execution.budget_limitation or "",
                    }
                    for execution in executions
                ],
                "direct_evidence_count": sum(1 for item in extractions if item.provisional_classification == "direct"),
                "analogous_evidence_count": sum(1 for item in extractions if item.provisional_classification == "linked"),
                "contradictory_evidence_count": sum(1 for item in extractions if item.provisional_classification == "conflicting"),
                "methodological_precedent_count": 0,
                "summary": final_review.rationale if final_review else "PI assessment unavailable; retrieved evidence is retained for review.",
                "limitations": limitations,
                "timeout_seconds": max(0.0, direction_timeout_seconds - max(0.0, candidate_deadlines.get(direction_id, gate_started) - gate_started)) if any(e.status == "timeout" for e in executions) else None,
                "timed_out": any(e.status == "timeout" for e in executions),
                "novelty_status": "unknown",
                "literature_novelty": "unknown",
                "support_level": "unknown",
                "contradiction_level": "unknown",
                "prior_art_level": "unknown",
                "negative_validation_confidence": "none",
                "negative_validation_axes": {},
                "key_evidence": "; ".join(
                    f"PMID {item.pmid}: {item.finding}" for item in extractions[:3] if item.finding
                ),
                "gap_assessment": "; ".join(limitations[:3]),
            })
            if final_review is not None:
                classifications = final_review.final_scientific_classifications or {}
                allowed_classifications = {
                    "literature_novelty": {"high", "medium", "low", "unknown"},
                    "prior_art_level": {"high", "medium", "low", "unknown"},
                    "support_level": {"strong", "moderate", "weak", "none", "unknown"},
                    "contradiction_level": {"strong", "moderate", "weak", "none", "unknown"},
                }
                for field, allowed_values in allowed_classifications.items():
                    value = classifications.get(field)
                    if value in allowed_values:
                        result[field] = value
                if classifications.get("novelty_status") in {
                    "genuine_novelty", "known_area", "adjacent_work_only", "unknown_due_to_failure", "unknown_due_to_timeout", "unknown"
                }:
                    result["novelty_status"] = classifications["novelty_status"]
                result["similar_published_directions"] = [
                    {"pmid": pmid} for pmid in final_review.cited_article_ids[:5]
                ]
            results.append(result)
        checkpoint.status = "completed" if not planning_skipped and all(
            not r.get("unexecuted_task_ids") for r in results
        ) else "partial"
        self._save_direction_checkpoint(state, checkpoint)
        return results

    # Name used by workflow call sites and direct owner tests.
    evaluate_direction_candidates_with_pi = evaluate_direction_literature_search

    def direction_evidence_from_checkpoint(
        self,
        candidates: List[Dict[str, Any]],
        state: Optional[AnalysisState] = None,
        *,
        phase: str = "exploration",
    ) -> List[Dict[str, Any]]:
        """Project retained technical evidence after an unexpected caller failure."""
        gate_id = self._direction_gate_id(candidates, state, phase)
        checkpoint = self._load_direction_checkpoint(state, gate_id, candidates, phase)
        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            direction_id = str(candidate.get("direction_id") or "")
            task_ids = {task.task_id for task in checkpoint.tasks if task.direction_id == direction_id}
            executions = [e for e in checkpoint.task_executions if e.direction_id == direction_id]
            articles = [a for a in checkpoint.article_inventory if task_ids.intersection(a.task_ids)]
            extractions = [x for x in checkpoint.extractions if task_ids.intersection(x.task_ids)]
            limitations = list(dict.fromkeys([
                *checkpoint.limitations_by_candidate.get(direction_id, []),
                *(
                    [checkpoint.planning_errors[direction_id]]
                    if direction_id in checkpoint.planning_errors else []
                ),
                "Direction literature caller failed after checkpoint evidence was retained.",
            ]))
            result = self._build_empty_direction_result(direction_id, str(candidate.get("direction") or ""))
            result.update({
                "status": "partial",
                "policy_version": checkpoint.policy_version,
                "plan_id": checkpoint.plan_id,
                "plan_hash": checkpoint.plan_hash,
                "plan_status": checkpoint.plan_status,
                "tasks": [t.model_dump(mode="json") for t in checkpoint.tasks if t.direction_id == direction_id],
                "task_executions": [e.model_dump(mode="json") for e in executions],
                "article_inventory": [a.model_dump(mode="json") for a in articles],
                "extractions": [x.model_dump(mode="json") for x in extractions],
                "pi_reviews": [],
                "assessment_status": "unavailable",
                "assessment_uncertainty": [],
                "assessment_limitations": limitations,
                "unexecuted_task_ids": [
                    task_id for task_id in checkpoint.unexecuted_task_ids
                    if any(t.task_id == task_id and t.direction_id == direction_id for t in checkpoint.tasks)
                ],
                "limitations": limitations,
            })
            results.append(result)
        return results

    async def evaluate_direction_novelty_candidates(
        self,
        candidates: List[Dict[str, Any]],
        state: Optional[AnalysisState] = None,
        query_timeout_seconds: float = NOVELTY_QUERY_TIMEOUT_SECONDS,
        direction_timeout_seconds: float = NOVELTY_DIRECTION_TIMEOUT_SECONDS,
        max_full_evidence_candidates: int = 8,
        tier3_query_timeout_seconds: float = NOVELTY_TIER3_QUERY_TIMEOUT_SECONDS,
        concurrency: int = NOVELTY_CANDIDATE_CONCURRENCY,
    ) -> List[Dict[str, Any]]:
        """Judge each candidate's prior art. One record per candidate, in order.

        Candidates are independent, so they are evaluated concurrently under a
        bounded semaphore: serially, the gate's wall clock was the sum of every
        candidate's direction budget, which is what made a generous budget
        unaffordable and a stingy one useless.
        """
        cached_articles = self._extract_session_cached_pubmed_articles(state)
        # One study per session and per gate run, so this is set once here and
        # read by the triage step rather than threaded through four call frames.
        # The research objective is used and the candidate's own
        # experimental_context is not: one observed run filled that field with
        # "2 group(s): Con (3 sample(s)), DKO (3 sample(s))" for every candidate,
        # which names no perturbation and would mislead the bucket decision.
        self._direction_study_context = (
            str(getattr(state, "initial_instruction", "") or "").strip()[:400]
            if state is not None else ""
        )
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def evaluate_one(index: int, candidate: Dict[str, Any]) -> Dict[str, Any]:
            direction_id = candidate.get("direction_id", "")
            async with semaphore:
                if index >= max_full_evidence_candidates:
                    return await self._evaluate_minimum_attempt_candidate(
                        candidate,
                        cached_articles,
                        query_timeout_seconds=query_timeout_seconds,
                        direction_timeout_seconds=direction_timeout_seconds,
                        scheduler_limitation="full_evaluation_skipped_by_candidate_cap",
                    )

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
                    return dir_result
                except UpstreamExhaustedError:
                    raise
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
                    return fallback

        # gather preserves input order, which the candidate cap above depends on.
        tasks = [
            asyncio.create_task(evaluate_one(index, candidate))
            for index, candidate in enumerate(candidates)
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # A failed batch must settle before the workflow offers Retry;
            # gather alone leaves sibling LLM requests running after an error.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
