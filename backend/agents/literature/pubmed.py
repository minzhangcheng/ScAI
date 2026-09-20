"""PubMed search execution, XML parsing, and batched query rounds.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Callable, Dict, Any, List, Optional, Set, Tuple
import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import time
import urllib.parse
import weakref
import xml.etree.ElementTree as ET

import httpx


logger = logging.getLogger(__name__)

PUBMED_QUERY_POLICY_VERSION = 3

# NCBI E-utilities request policy.
#
# ``tool`` and ``email`` identify the caller so NCBI can make contact before it
# blocks anyone; neither raises a quota. The ceiling is per IP address: three
# requests per second without an API key, ten with one, and anything over it
# comes back as HTTP 429.
#
# The previous throttle was a 0.4 s sleep sitting *between* one call's esearch
# and its efetch. That paces a single sequential caller and nothing else. The
# direction-novelty gate evaluates candidates concurrently, so several tasks
# issued requests within the same second; NCBI rejected whole batches and the
# only trace was a missing evidence axis on one direction. Every request now
# passes through one process-wide gate, so concurrency cannot exceed the rate,
# and a rejection is retried instead of costing a batch.
NCBI_TOOL_NAME = "ScAI"
NCBI_RATE_LIMIT_PER_SECOND = 3.0
NCBI_RATE_LIMIT_PER_SECOND_WITH_KEY = 10.0
NCBI_RATE_LIMIT_RETRIES = 3
NCBI_RATE_LIMIT_BACKOFF_SECONDS = 1.0
# Two audited session queries prepared GET URLs of 7,679 and 33,218 characters;
# both were retried unchanged and degraded to LLM-only knowledge. The threshold
# is deliberately below both measurements and is retunable if NCBI's limits
# change.
NCBI_MAX_GET_URI_CHARS = 4000


class _EntrezThrottle:
    """A shared request gate: one slot per interval, per event loop.

    Keyed by loop so a test that builds its own loop is not paced by another
    one's timestamps, and weakly so a finished loop does not stay referenced.
    """

    def __init__(self) -> None:
        self._states: "weakref.WeakKeyDictionary[Any, List[Any]]" = (
            weakref.WeakKeyDictionary()
        )

    async def acquire(self, rate_per_second: float) -> None:
        loop = asyncio.get_running_loop()
        state = self._states.get(loop)
        if state is None:
            state = [asyncio.Lock(), 0.0]
            self._states[loop] = state
        interval = 1.0 / max(rate_per_second, 0.1)
        async with state[0]:
            now = time.monotonic()
            wait_for = state[1] - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            state[1] = max(now, state[1]) + interval


_entrez_throttle = _EntrezThrottle()


def _normalized_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _search_cache_key(
    *,
    topic: str,
    query: str,
    organism: str,
    tissue: str,
    pi_guided: bool,
    query_policy_version: int = PUBMED_QUERY_POLICY_VERSION,
    research_context: str = "",
    planned_queries: Optional[List[str]] = None,
) -> str:
    payload = {
        "topic": _normalized_search_text(topic),
        "query": _normalized_search_text(query),
        "organism": _normalized_search_text(organism),
        "tissue": _normalized_search_text(tissue),
        "pi_guided": bool(pi_guided),
        "query_policy_version": int(query_policy_version),
        "research_context": _normalized_search_text(research_context),
        "planned_queries": [
            _normalized_search_text(value) for value in (planned_queries or [])
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
from ...models.schemas import (
    AnalysisState,
    LiteratureReport,
    DirectionSearchQuery,
    PubMedEvidenceItem,
)
from ...utils.llm_client import UpstreamExhaustedError

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


def _search_continuation(evaluation, query, label, round_num):
    """Decide whether to keep searching, and why we stopped if not.

    Returns ``(stop_reason, next_query)``; ``stop_reason`` is None to continue.
    "Stop searching" and "the evidence supports a conclusion" are two judgements
    that used to share one boolean, so a broken evaluator and an
    exhausted-but-thin search both came out as "sufficient".
    """
    if evaluation.get("evaluation_failed"):
        logger.warning(
            f"[Literature] {label} stopped after {round_num} rounds: "
            "search evaluation unavailable; evidence left unassessed"
        )
        return "evaluation_failed", query
    if evaluation.get("search_exhausted"):
        logger.info(
            f"[Literature] {label} search exhausted after {round_num} rounds "
            f"(evidence: {evaluation.get('evidence_strength', 'unassessed')})"
        )
        return "search_exhausted", query
    refined_query = str(evaluation.get("refined_query") or query).strip()
    if _normalized_search_text(refined_query) == _normalized_search_text(query):
        return "query_stagnation", query
    return None, refined_query


def _evaluation_meta(evaluation) -> dict:
    """Project the search evaluation onto the metadata the report reads.

    Without these, "we stopped because the evaluator broke" and "we stopped
    because the search was exhausted and the evidence is strong" were the same
    record downstream.
    """
    if not isinstance(evaluation, dict):
        return {
            "search_exhausted": False,
            "evidence_strength": "unassessed",
            "evaluation_failed": False,
            "evaluation_reasoning": "",
            "missing_aspects": [],
        }
    return {
        "search_exhausted": bool(evaluation.get("search_exhausted")),
        "evidence_strength": str(evaluation.get("evidence_strength") or "unassessed"),
        "evaluation_failed": bool(evaluation.get("evaluation_failed")),
        "evaluation_reasoning": str(evaluation.get("reasoning") or "")[:600],
        "missing_aspects": list(evaluation.get("missing_aspects") or [])[:10],
    }


class PubMedMixin:
    """PubMed search execution, XML parsing, and batched query rounds."""

    async def _execute_query_batch(
        self,
        query: DirectionSearchQuery,
        candidate: Dict[str, Any],
        matching_cache: List[Dict[str, Any]],
        query_timeout_seconds: float,
        deadline: Optional[float] = None,
        triaged_pmids: Optional[Set[str]] = None,
    ) -> Tuple[List[PubMedEvidenceItem], Dict[str, Any], bool, List[str]]:
        evidence: List[PubMedEvidenceItem] = []
        limitations: List[str] = []
        timed_out = False
        meta: Dict[str, Any] = {}

        def remaining_timeout() -> Optional[float]:
            if deadline is None:
                return query_timeout_seconds
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            return min(query_timeout_seconds, remaining)

        def mark_direction_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            if "Direction-level timeout exceeded" not in limitations:
                limitations.append("Direction-level timeout exceeded")

        # Cached articles matching this candidate are computed once per candidate
        # by the caller and passed in.
        if matching_cache:
            try:
                step_timeout = remaining_timeout()
                if step_timeout is None:
                    mark_direction_timeout()
                    return evidence, meta, timed_out, limitations
                cache_ev, cache_meta = await asyncio.wait_for(
                    self._triage_articles_with_llm(matching_cache, candidate, query),
                    timeout=step_timeout,
                )
                evidence.extend(cache_ev)
                meta = cache_meta
            except asyncio.TimeoutError:
                if deadline is not None and time.monotonic() >= deadline:
                    mark_direction_timeout()
                else:
                    limitations.append(f"Cache triage timeout for query: {query.category}")
                    timed_out = True

        # PubMed search
        try:
            step_timeout = remaining_timeout()
            if step_timeout is None:
                mark_direction_timeout()
                return evidence, meta, timed_out, limitations
            articles = await asyncio.wait_for(
                self._pubmed_search(query.query, max_results=20),
                timeout=step_timeout,
            )
            query.result_count = len(articles)
        except asyncio.TimeoutError:
            if deadline is not None and time.monotonic() >= deadline:
                mark_direction_timeout()
            else:
                limitations.append(f"PubMed search timeout for query: {query.category}")
                timed_out = True
            return evidence, meta, timed_out, limitations
        except Exception as exc:
            limitations.append(f"PubMed search failed for {query.category}: {str(exc)[:100]}")
            return evidence, meta, timed_out, limitations

        if not articles:
            return evidence, meta, timed_out, limitations

        # A direction's query categories overlap heavily: measured on two real
        # runs, a fifth of every article sent to triage had already been triaged
        # for the same direction under a different category. The verdict was
        # paid for and then discarded, because _dedup_evidence keeps the first
        # one per PMID. Drop the repeats before the LLM call, not after.
        if triaged_pmids is not None:
            fresh = [
                a for a in articles
                if str(a.get("pmid") or "") not in triaged_pmids
            ]
            skipped = len(articles) - len(fresh)
            query.duplicate_count = skipped
            if skipped:
                logger.debug(
                    "[Literature] %s: %s of %s articles already triaged for this "
                    "direction; not re-sent", query.category, skipped, len(articles)
                )
            triaged_pmids.update(
                str(a.get("pmid") or "") for a in fresh if a.get("pmid")
            )
            articles = fresh
            if not articles:
                return evidence, meta, timed_out, limitations

        # LLM triage
        try:
            step_timeout = remaining_timeout()
            if step_timeout is None:
                mark_direction_timeout()
                return evidence, meta, timed_out, limitations
            llm_ev, llm_meta = await asyncio.wait_for(
                self._triage_articles_with_llm(articles, candidate, query),
                timeout=step_timeout,
            )
            evidence.extend(llm_ev)
            meta = llm_meta
            query.relevant_count = len(llm_ev)
        except asyncio.TimeoutError:
            if deadline is not None and time.monotonic() >= deadline:
                mark_direction_timeout()
            else:
                limitations.append(f"LLM triage timeout for query: {query.category}")
                timed_out = True

        return evidence, meta, timed_out, limitations

    def _extract_session_cached_pubmed_articles(self, state: Optional[AnalysisState]) -> List[Dict[str, Any]]:
        if not state:
            return []
        articles: List[Dict[str, Any]] = []
        seen_pmids: set = set()
        for report in state.literature_reports:
            for ref in report.references:
                url = ref.get("url", "")
                if "pubmed" not in url.lower():
                    continue
                pmid = ""
                for part in url.rstrip("/").split("/"):
                    if part.isdigit():
                        pmid = part
                        break
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                articles.append({
                    "pmid": pmid,
                    "title": ref.get("title", ""),
                    "abstract": "",
                    "year": "",
                    "authors": "",
                    "journal": "",
                    "source": "session_cache",
                })
        return articles

    async def _evaluate_pubmed_results(
        self,
        topic: str,
        articles: List[Dict[str, str]],
        search_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM evaluates PubMed results and suggests refined queries."""
        # Include truncated abstracts for better relevance judgment.
        # Sort by year (newest first) so recent rounds' articles are not dropped,
        # then sample up to 30 (model context is ample).
        def _year_key(a: Dict[str, str]) -> int:
            try:
                return int(str(a.get("year", "")).strip()[:4])
            except (ValueError, TypeError):
                return 0
        sorted_articles = sorted(articles, key=_year_key, reverse=True)
        sampled = sorted_articles[:30]
        articles_summary = "\n".join(
            f"- [{a.get('year', 'N/A')}] {a['title']}\n  Abstract: {a.get('abstract', 'N/A')}"
            for a in sampled
        )

        # List all previous queries explicitly to prevent duplication
        previous_queries = "\n".join(
            f"  Round {h['round']}: {h['query']}"
            for h in search_history
        )

        prompt = f"""Evaluate whether the following PubMed search results are sufficient for a
comprehensive literature review on the topic:

Topic: {topic}

Previous queries (DO NOT repeat these):
{previous_queries}

Articles found ({len(articles)} total, showing {len(sampled)} most recent with abstracts):
{articles_summary}

Evaluation criteria:
- Are there articles directly relevant to the topic, as opposed to merely adjacent to it?
- Are the aspects this specific topic turns on covered? Derive them from the topic itself
  rather than from a fixed list; a descriptive, methodological, developmental or comparative
  topic has no therapeutic angle to cover, and demanding one produces padding
- For each CONTESTABLE claim, was contradicting, qualifying or non-replicating work looked
  for as deliberately as supporting work? Weigh what you find by study design, directness and
  how well the model applies here — not by how many papers sit on each side, and not by giving
  the two equal space regardless of quality. A well-established descriptive fact needs no
  manufactured counterpoint; searching for one only imports irrelevant literature
- Do the retrieved studies differ in design or model, so the conclusion does not rest on a
  single group or a single system?

Note on counts: the number of articles is not the criterion. A genuinely novel or narrow topic
may have very few relevant papers, and reporting that honestly ("sparse literature; N directly
relevant articles") is a correct and informative outcome. Do NOT keep broadening the query until
some arbitrary number of hits accumulates — that yields tangentially related papers that will
later be cited as if they supported the claim.

Respond in JSON format. Note that the two judgements below are separate: a search can be
exhausted while the evidence remains thin, and that combination is a useful, honest result.
{{
    "search_exhausted": true or false,
    "evidence_strength": "strong | moderate | sparse | contradictory | insufficient",
    "reasoning": "why the search is or is not exhausted, and how strong the evidence is",
    "refined_query": "a NEW PubMed search query targeting missing aspects (MUST differ from all previous queries)",
    "missing_aspects": ["aspects not yet covered"]
}}

IMPORTANT:
- The refined_query MUST be a valid PubMed search query using AND/OR operators
- The refined_query MUST NOT duplicate any previous query listed above
- If the articles are mostly irrelevant, the query may be wrong — try different
  terminology or a different angle before broadening. If broadening is genuinely the
  right move, keep the query on-topic; do not relax it until unrelated work matches.
  If the literature is simply sparse, say so: set search_exhausted=true with
  evidence_strength="sparse". Exhausting the search and finding little is an honest
  result; manufacturing hits to fill it is not"""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.3)
        result = self._safe_parse_json(response)
        if not result or "search_exhausted" not in result:
            # Stopping here is right -- an unreadable evaluator should not spin the
            # loop -- but reporting it as "sufficient" turns a tooling failure into
            # a claim about the evidence. Stop, and say why.
            logger.warning(
                "[Literature] Sufficiency evaluation unreadable; stopping the search "
                "and recording the evaluation as unavailable"
            )
            return {
                "search_exhausted": False,
                "evidence_strength": "unassessed",
                "evaluation_failed": True,
                "reasoning": (
                    "The search evaluation could not be read, so the search loop stopped "
                    "early. This says nothing about whether the retrieved literature supports "
                    "any conclusion; treat the evidence as unassessed."
                ),
            }
        return result

    async def _focused_rare_search(
        self,
        items,
        organism: str,
        tissue: str,
        bucket: str = "rare_unresolved",
        deadline: Optional[float] = None,
        retain_fn: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> dict:
        """两级 focused 检索,按簇返回 {articles,status,tier}。

        rare_unresolved: tier1(加 tissue) → 仅当 exhausted/contradictory 时升级 tier2(去 tissue)
        → 仍为空时升级 tier3(去 markers,保留 tissue)。
        suspected_contamination: 顺序跑 tier1=discovery(无 tissue) 再 tier2=plausibility(加 tissue),
        (受 NCBI 限速约束，未并发；如需并发须配 Semaphore 后再改)
                                 取相关性 rank 更高者(spec §5.2: 两条都查)。
        空 query 直接记为 exhausted,不发请求。
        """
        results: dict = {}

        async def _run(item, tier):
            def exhausted(reason, error=None, articles=None):
                return {
                    "articles": list(articles or []),
                    "retained_articles": list(articles or []),
                    "status": "exhausted",
                    "tier": tier,
                    "exhaustion_reason": reason,
                    "error": error,
                }

            q = build_focused_query(item, organism, tissue, tier=tier, bucket=bucket)
            if not q:
                return exhausted("empty_query")
            def remaining() -> Optional[float]:
                if deadline is None:
                    return None
                value = deadline - time.monotonic()
                return value if value > 0 else None

            try:
                timeout = remaining()
                if deadline is not None and timeout is None:
                    return exhausted("wall_clock_budget")
                search = self._pubmed_search(q, max_results=10)
                arts = await (
                    asyncio.wait_for(search, timeout=timeout)
                    if timeout is not None else search
                )
            except asyncio.TimeoutError:
                return exhausted("wall_clock_budget")
            except Exception as e:
                logger.warning(
                    "[Literature] Focused PubMed search failed for cluster %s tier %s: %s",
                    item.get("cluster_id"),
                    tier,
                    e,
                )
                return exhausted("search_failed", str(e))

            if not arts:
                return exhausted("no_pubmed_results")
            if retain_fn is not None:
                retain_fn(arts)

            timeout = remaining()
            if deadline is not None and timeout is None:
                return exhausted("wall_clock_budget", articles=arts)
            try:
                if timeout is not None:
                    await asyncio.wait_for(asyncio.sleep(0.4), timeout=timeout)
                else:
                    await asyncio.sleep(0.4)
            except asyncio.TimeoutError:
                return exhausted("wall_clock_budget", articles=arts)

            try:
                timeout = remaining()
                if deadline is not None and timeout is None:
                    return exhausted("wall_clock_budget", articles=arts)
                judge = self._judge_focused_relevance(item, tissue, arts, bucket)
                judged = await (
                    asyncio.wait_for(judge, timeout=timeout)
                    if timeout is not None else judge
                )
            except asyncio.TimeoutError:
                return exhausted("wall_clock_budget", articles=arts)
            except UpstreamExhaustedError:
                raise
            except Exception as e:
                logger.warning(
                    "[Literature] Focused relevance judge failed for cluster %s tier %s: %s",
                    item.get("cluster_id"),
                    tier,
                    e,
                )
                return exhausted("judge_failed", str(e), articles=arts)

            status = judged.get("status", "exhausted") if isinstance(judged, dict) else "exhausted"
            return {
                "articles": arts,
                "retained_articles": arts,
                "status": status,
                "tier": tier,
                "exhaustion_reason": "no_relevant_evidence" if status == "exhausted" else None,
                "error": None,
            }

        for item in items or []:
            cid = item.get("cluster_id")
            if cid is None:
                continue
            if bucket == "suspected_contamination":
                r1 = await _run(item, 1)   # discovery
                r2 = await _run(item, 2)   # plausibility
                best = r2 if self._focused_result_score(r2) > self._focused_result_score(r1) else r1
                retained = list(r1.get("retained_articles") or []) + list(
                    r2.get("retained_articles") or []
                )
            else:
                best = await _run(item, 1)
                retained = list(best.get("retained_articles") or [])
                if best["status"] in ("exhausted", "contradictory"):
                    alt = await _run(item, 2)
                    retained.extend(alt.get("retained_articles") or [])
                    if self._focused_result_score(alt) > self._focused_result_score(best):
                        best = alt
                # Tier 2 only drops tissue, which recovered one cluster of the
                # eleven that came back empty. What actually starves the search
                # is the mandatory marker block, so a cluster still without
                # evidence gets one more attempt without it.
                if best["status"] in ("exhausted", "contradictory"):
                    alt = await _run(item, 3)
                    retained.extend(alt.get("retained_articles") or [])
                    if self._focused_result_score(alt) > self._focused_result_score(best):
                        best = alt
            best = dict(best)
            best["retained_articles"] = self._dedup_by_pmid(
                retained, lambda article: article.get("pmid"), drop_empty=True,
            )
            results[cid] = best
        return results

    async def _multi_round_pubmed(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        pubmed_max_results: Optional[int] = None,
        current_round_findings: Optional[List[str]] = None,
        initial_query: Optional[str] = None,
        evaluate_fn: Optional[Callable] = None,
        organism: str = "",
        tissue: str = "",
        seed_queries: Optional[List[str]] = None,
        cluster_markers: Optional[Dict[str, List[str]]] = None,
        research_context: str = "",
        planned_queries: Optional[List[str]] = None,
        deadline_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Multi-round PubMed search with LLM-guided query refinement.

        When ``initial_query`` and ``evaluate_fn`` are provided, the search
        uses PI Agent-guided query generation and evaluation instead of
        Literature Agent's own LLM-based methods.

        ``seed_queries`` are deterministic supplementary queries run once in the
        first round; ``cluster_markers`` lets the per-cluster evidence pass build
        a query for clusters whose triage entry carries no markers of its own.
        """
        pi_guided = initial_query is not None and evaluate_fn is not None
        label = "PI-guided PubMed" if pi_guided else "PubMed"
        active_query_policy_version = PUBMED_QUERY_POLICY_VERSION

        all_articles: List[Dict[str, str]] = []
        seen_pmids: Set[str] = set()
        search_history: List[Dict[str, Any]] = []
        query_attempts: List[Dict[str, Any]] = []
        latest_triage: Dict[str, Any] = {}
        # Separate from latest_triage, which is the annotation triage payload:
        # only the PI-guided branch ever wrote that, so the plain PubMed branch
        # produced metadata contradicting its own stop_reason.
        latest_evaluation: Dict[str, Any] = {}
        plateau_streak = 0
        evaluation_count = 0
        post_triage_rounds = 0
        technical_stop_error = ""
        post_triage_limit = int(
            getattr(self, "post_triage_refinement_rounds", 2) or 0
        )
        stop_reason = "max_search_rounds"
        search_started = time.monotonic()
        search_wall_clock_seconds = float(
            getattr(self, "search_wall_clock_seconds", 600) or 600
        )
        max_articles_per_topic = int(
            getattr(self, "max_articles_per_topic", 200) or 200
        )
        max_sufficiency_evaluations = int(
            getattr(self, "max_sufficiency_evaluations", 8) or 8
        )
        plateau_rounds = int(getattr(self, "plateau_rounds", 4) or 4)
        plateau_min_new_articles = int(
            getattr(self, "plateau_min_new_articles", 1) or 0
        )
        deadline = deadline_override or (search_started + search_wall_clock_seconds)

        # Generate initial PubMed query from the topic (or use provided one)
        if pi_guided:
            query = initial_query
        else:
            query = await self._generate_pubmed_query(topic, state, current_round_findings)

        cache_key = _search_cache_key(
            topic=topic,
            query=query,
            organism=organism,
            tissue=tissue,
            pi_guided=pi_guided,
            query_policy_version=active_query_policy_version,
            research_context=research_context,
            planned_queries=planned_queries or [query, *(seed_queries or [])],
        )
        normalized_cache_query = _normalized_search_text(query)
        source_report: Optional[LiteratureReport] = None
        checkpoint: Dict[str, Any] = {
            "status": "in_progress",
            "topic": topic,
            "research_context": str(research_context or "")[:12000],
            "planned_queries": list(planned_queries or [query, *(seed_queries or [])])[:3],
            "query_attempts": [],
            "search_history": [],
            "articles": [],
        }

        def source_references() -> List[Dict[str, str]]:
            return [
                {
                    "pmid": str(article.get("pmid") or ""),
                    "title": str(article.get("title") or ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid')}/",
                    "abstract": str(article.get("abstract") or ""),
                    "year": str(article.get("year") or ""),
                    "authors": str(article.get("authors") or ""),
                    "journal": str(article.get("journal") or article.get("source") or ""),
                }
                for article in all_articles
            ]

        def checkpoint_sources(extra_articles: Optional[List[Dict[str, Any]]] = None) -> None:
            nonlocal source_report, all_articles, stop_reason
            for article in extra_articles or []:
                pmid = str(article.get("pmid") or "")
                if pmid and pmid not in seen_pmids:
                    all_articles.append(article)
                    seen_pmids.add(pmid)
            if len(all_articles) > max_articles_per_topic:
                all_articles = all_articles[:max_articles_per_topic]
                stop_reason = "article_budget"
            checkpoint.update({
                "query_attempts": copy.deepcopy(query_attempts[-30:]),
                "search_history": copy.deepcopy(search_history[-20:]),
                "articles": copy.deepcopy(all_articles[:max_articles_per_topic]),
            })
            if all_articles:
                refs = source_references()
                if source_report is None:
                    source_report = LiteratureReport(
                        topic=topic,
                        report=(
                            "PubMed sources were retrieved. Synthesis is pending or unavailable; "
                            "the retained source inventory remains available."
                        ),
                        references=refs,
                    )
                    reports = getattr(self, "reports", None)
                    if reports is None:
                        reports = []
                        self.reports = reports
                    reports.append(source_report)
                    if state is not None:
                        state.literature_reports.append(source_report)
                else:
                    source_report.references = refs
            if state is not None:
                cache = getattr(state, "literature_search_cache", None)
                if cache is None:
                    cache = {}
                    setattr(state, "literature_search_cache", cache)
                entry = dict(cache.get(cache_key) or {})
                entry.update({
                    "normalized_topic": _normalized_search_text(topic),
                    "normalized_query": normalized_cache_query,
                    "checkpoint": copy.deepcopy(checkpoint),
                })
                entry.pop("result", None)
                cache[cache_key] = entry
                while len(cache) > 100:
                    cache.pop(next(iter(cache)), None)

        def cache_result(result: Dict[str, Any]) -> None:
            if state is None:
                return
            cache = getattr(state, "literature_search_cache", None)
            if cache is None:
                cache = {}
                setattr(state, "literature_search_cache", cache)
            search_meta = result.get("search_meta") or {}
            technically_complete = (
                result.get("status") == "completed"
                and not result.get("source_processing_error")
                and not search_meta.get("evaluation_failed")
            )
            checkpoint_payload = {
                **copy.deepcopy(checkpoint),
                "status": "complete" if technically_complete else "partial",
                "query_attempts": copy.deepcopy(query_attempts[-30:]),
                "search_history": copy.deepcopy(search_history[-20:]),
                "articles": copy.deepcopy(all_articles[:max_articles_per_topic]),
            }
            if not technically_complete:
                checkpoint_payload.update({
                    "stop_reason": search_meta.get("stop_reason"),
                    "source_processing_error": result.get("source_processing_error"),
                })
            entry = {
                "normalized_topic": _normalized_search_text(topic),
                "normalized_query": normalized_cache_query,
                "checkpoint": checkpoint_payload,
            }
            if technically_complete:
                entry["result"] = copy.deepcopy(result)
            cache[cache_key] = entry
            while len(cache) > 100:
                oldest_key = next(iter(cache))
                cache.pop(oldest_key, None)

        if state is not None:
            cache = getattr(state, "literature_search_cache", None) or {}
            cached = cache.get(cache_key)
            cached_result = cached.get("result") if isinstance(cached, dict) else None
            if isinstance(cached_result, dict):
                reused = copy.deepcopy(cached_result)
                meta = dict(reused.get("search_meta") or {})
                meta.update({"stop_reason": "cache_reuse", "cache_hit": True})
                reused["search_meta"] = meta
                return reused

        for round_num in range(1, self.max_search_rounds + 1):
            if time.monotonic() >= deadline:
                stop_reason = "wall_clock_budget"
                technical_stop_error = (
                    "PubMed search stopped because the wall-clock budget was exhausted "
                    "before the next query completed."
                )
                break
            logger.info(
                f"[Literature] {label} round {round_num}/{self.max_search_rounds}: "
                f"{query[:100]}"
            )

            # Search PubMed (first round uses validation + degradation)
            round_attempt_start = len(query_attempts)
            try:
                if round_num == 1:
                    query, articles = await asyncio.wait_for(
                        self._validate_and_degrade_query(
                            query, topic,
                            max_results=pubmed_max_results or 20,
                            state=state,
                            current_round_findings=current_round_findings,
                            enable_semantic_revision=not pi_guided,
                            attempt_log=query_attempts,
                        ),
                        timeout=max(0.001, deadline - time.monotonic()),
                    )
                else:
                    articles = await asyncio.wait_for(
                        self._pubmed_search(query, max_results=pubmed_max_results or 20),
                        timeout=max(0.001, deadline - time.monotonic()),
                    )
            except asyncio.TimeoutError:
                stop_reason = "wall_clock_budget"
                technical_stop_error = (
                    "PubMed search stopped because a query exceeded the remaining "
                    "wall-clock budget."
                )
                if round_num > 1:
                    query_attempts.append({
                        "kind": "refinement", "query": query,
                        "result_count": None, "error": "wall_clock_budget",
                    })
                break
            except UpstreamExhaustedError:
                raise
            except Exception as e:
                logger.warning(f"[Literature] PubMed search failed in round {round_num}: {e}")
                if round_num == 1 and not all_articles:
                    raise  # Re-raise to trigger fallback
                stop_reason = "pubmed_error"
                technical_stop_error = (
                    f"PubMed search failed technically: {type(e).__name__}: {e}"
                )[:1000]
                if round_num > 1:
                    query_attempts.append({
                        "kind": "refinement", "query": query,
                        "result_count": None, "error": str(e)[:500],
                    })
                break

            if round_num > 1:
                query_attempts.append({
                    "kind": "refinement",
                    "query": query,
                    "result_count": len(articles),
                })

            primary_result_count = len(articles)

            # Deduplicate by PMID
            new_articles = []
            for a in articles:
                if a["pmid"] not in seen_pmids:
                    new_articles.append(a)
                    seen_pmids.add(a["pmid"])
            primary_new_results = len(new_articles)
            all_articles.extend(new_articles)
            checkpoint_sources()

            if round_num == 1 and seed_queries:
                seed_articles = await self._run_seed_queries(
                    seed_queries,
                    max_results=int(getattr(self, "seed_query_articles", 20) or 20),
                    seen_pmids=seen_pmids,
                    query_attempts=query_attempts,
                    deadline=deadline,
                )
                if seed_articles:
                    # Seeds head the pool: the evaluator and the synthesis
                    # prompt both read from the front of it.
                    all_articles = seed_articles + all_articles
                    new_articles = seed_articles + new_articles
                    checkpoint_sources()

            if len(all_articles) >= max_articles_per_topic:
                all_articles = all_articles[:max_articles_per_topic]
                stop_reason = "article_budget"

            plateau_streak = (
                plateau_streak + 1
                if len(new_articles) <= plateau_min_new_articles else 0
            )

            search_history.append({
                "round": round_num,
                "query": query,
                "new_results": primary_new_results,
                "primary_result_count": primary_result_count,
                "supplementary_new_results": max(
                    0, len(new_articles) - primary_new_results,
                ),
                "total_results": len(all_articles),
                "query_attempts": copy.deepcopy(
                    query_attempts[round_attempt_start:][-10:]
                ),
            })
            checkpoint_sources()

            logger.info(
                f"[Literature] {label} round {round_num}: "
                f"{len(new_articles)} new articles, {len(all_articles)} total"
            )

            if round_num == 1 and not all_articles and not pi_guided:
                stop_reason = "no_pubmed_results"
                logger.info(f"[Literature] {label} stop: no PubMed results")
                break

            # Evaluate if more searching needed
            if stop_reason == "article_budget" and not pi_guided:
                logger.info(f"[Literature] {label} stop: article budget")
                break
            if plateau_streak >= plateau_rounds and (all_articles or not pi_guided):
                stop_reason = "marginal_yield_plateau"
                logger.info(f"[Literature] {label} stop: marginal-yield plateau")
                break

            if pi_guided or (round_num < self.max_search_rounds and all_articles):
                if evaluation_count >= max_sufficiency_evaluations:
                    stop_reason = "sufficiency_evaluation_budget"
                    break
                if pi_guided:
                    evaluation_count += 1
                    try:
                        evaluation = await asyncio.wait_for(
                            evaluate_fn(
                                copy.deepcopy(all_articles),
                                copy.deepcopy(search_history),
                            ),
                            timeout=max(0.001, deadline - time.monotonic()),
                        )
                    except asyncio.TimeoutError:
                        stop_reason = "wall_clock_budget"
                        technical_stop_error = (
                            "PI literature search review exceeded the remaining "
                            "wall-clock budget."
                        )
                        break
                    except UpstreamExhaustedError:
                        raise
                    except Exception as e:
                        logger.warning(f"[Literature] PI evaluation failed in round {round_num}: {e}, stopping")
                        evaluation = {
                            "search_exhausted": False,
                            "evidence_strength": "unassessed",
                            "evaluation_failed": True,
                            "reasoning": f"Search evaluation raised: {e}",
                        }
                    latest_triage = evaluation
                    latest_evaluation = evaluation
                    # 确定性语义终止: 全部簇已归桶 且 triage 非低置信 且 评估认为文献充分。
                    # 评估明确说 insufficient 并给出 refined_query 时,再允许有限几轮定向检索;
                    # 否则针对未解决簇的 refined_query 永远不会执行。
                    if evaluation.get("triage_complete") and not evaluation.get("triage_low_confidence"):
                        if evaluation.get("sufficient", True) or post_triage_rounds >= post_triage_limit:
                            logger.info(f"[Literature] {label} semantic termination: all clusters triaged")
                            stop_reason = "triage_complete"
                            break
                        post_triage_rounds += 1
                        logger.info(
                            f"[Literature] {label} triage complete but evaluator reports "
                            f"insufficient coverage; refinement round "
                            f"{post_triage_rounds}/{post_triage_limit}"
                        )
                else:
                    evaluation_count += 1
                    try:
                        evaluation = await asyncio.wait_for(
                            self._evaluate_pubmed_results(
                                topic,
                                copy.deepcopy(all_articles),
                                copy.deepcopy(search_history),
                            ),
                            timeout=max(0.001, deadline - time.monotonic()),
                        )
                    except asyncio.TimeoutError:
                        stop_reason = "wall_clock_budget"
                        technical_stop_error = (
                            "Literature search evaluation exceeded the remaining "
                            "wall-clock budget."
                        )
                        break

                latest_evaluation = evaluation
                decision, next_query = _search_continuation(
                    evaluation, query, label, round_num
                )
                if decision is not None:
                    stop_reason = (
                        "no_pubmed_results"
                        if not all_articles and decision == "search_exhausted"
                        else decision
                    )
                    break
                if stop_reason == "article_budget":
                    break
                query = next_query

            # Request pacing now lives in _entrez_get, which gates every
            # E-utilities call process-wide rather than only the ones this
            # sequential loop happens to make.

        if not all_articles:
            technical_partial = stop_reason in {"wall_clock_budget", "pubmed_error"}
            if technical_partial and not search_history:
                report_text = (
                    "PubMed search stopped technically before any query completed, and "
                    "no source inventory was obtained."
                )
            elif technical_partial:
                report_text = (
                    "Completed PubMed query attempts did not retrieve sources before the "
                    "search stopped technically. This does not establish that relevant "
                    "literature does not exist."
                )
            else:
                report_text = (
                    "No PubMed articles were found for the executed queries on this topic. "
                    "This records evidence scarcity for this search and does not establish "
                    "that relevant literature does not exist."
                )
            if latest_evaluation.get("reasoning"):
                report_text += (
                    " PI review: " + str(latest_evaluation["reasoning"])[:1000]
                )
            report = LiteratureReport(topic=topic, report=report_text, references=[])
            self.reports.append(report)
            if state is not None:
                state.literature_reports.append(report)
            result = {
                "status": "completed",
                "report": report_text,
                "references": [],
                "findings": [report_text],
                "search_rounds": len(search_history),
            }
            result["search_meta"] = {
                "stop_reason": stop_reason,
                "cache_hit": False,
                "rounds": len(search_history),
                "article_count": 0,
                "seed_queries": sum(
                    1 for attempt in query_attempts if attempt.get("kind") == "seed"
                ),
                "cluster_evidence": 0,
                "sufficiency_evaluations": evaluation_count,
                "elapsed_seconds": round(time.monotonic() - search_started, 3),
                "fallback": "none",
                "query_policy_version": active_query_policy_version,
                "query_attempts": copy.deepcopy(query_attempts),
                **_evaluation_meta(latest_evaluation or {
                    "search_exhausted": stop_reason == "no_pubmed_results",
                    "evidence_strength": "unassessed",
                    "reasoning": (
                        report_text if stop_reason == "no_pubmed_results"
                        else f"Search stopped technically: {stop_reason}."
                    ),
                }),
            }
            if result["search_meta"].get("evaluation_failed"):
                result.update({
                    "status": "partial",
                    "source_processing_error": (
                        "PI literature search review failed technically; the zero-result "
                        "search remains unassessed."
                    ),
                    "source_provenance": "pubmed_search_unassessed",
                    "retained_source_report_ids": [report.id],
                })
            elif technical_partial:
                result.update({
                    "status": "partial",
                    "source_processing_error": (
                        technical_stop_error
                        or f"PubMed search stopped technically: {stop_reason}."
                    ),
                    "source_provenance": "pubmed_search_partial",
                    "retained_source_report_ids": [],
                })
            cache_result(result)
            return result

        # --- Per-cluster evidence: run after main loop, before synthesis ---
        focused_evidence: Dict[str, Any] = {}
        focused_articles_for_synthesis: List[Dict[str, Any]] = []
        annotation_triage = bool(
            latest_triage
            and any(
                key in latest_triage
                for key in (
                    "covered", "rare_unresolved", "suspected_contamination",
                    "context_dependent_state",
                )
            )
        )
        if pi_guided and annotation_triage:
            focused_evidence = await self._collect_cluster_evidence(
                latest_triage, organism, tissue,
                cluster_markers=cluster_markers, deadline=deadline,
                retain_fn=checkpoint_sources,
            )
            for cid, data in focused_evidence.items():
                adjudicated = data.get("status") in ("strong", "limited")
                for article in data.get("retained_articles", data.get("articles", [])):
                    pmid = article.get("pmid")
                    if not pmid:
                        continue
                    if pmid not in seen_pmids:
                        all_articles.append(article)
                        seen_pmids.add(pmid)
                    # Only adjudicated evidence is promoted ahead of the main
                    # pool for synthesis; unadjudicated references would crowd
                    # the synthesis budget with unweighed papers.
                    if adjudicated and article in data.get("articles", []):
                        focused_articles_for_synthesis.append(article)
            checkpoint_sources()

        if len(all_articles) > max_articles_per_topic:
            all_articles = all_articles[:max_articles_per_topic]
            stop_reason = "article_budget"

        # Prioritize focused evidence for synthesis because _synthesize_pubmed_results
        # reads only the first `synthesis_max_articles` (annotation searches read more).
        synthesis_articles: List[Dict[str, Any]] = self._dedup_by_pmid(
            focused_articles_for_synthesis + all_articles, lambda a: a.get("pmid"),
            drop_empty=True,
        )[:max_articles_per_topic]

        synth_kwargs: Dict[str, Any] = {"evaluation": latest_evaluation or None}
        if pi_guided:
            synthesize = self._synthesize_pubmed_results
            try:
                synth_parameters = inspect.signature(synthesize).parameters
            except (TypeError, ValueError):
                synth_parameters = {}
            if "max_articles" in synth_parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in synth_parameters.values()
            ):
                synth_kwargs["max_articles"] = int(
                    getattr(self, "annotation_synthesis_max_articles", 40) or 40
                )
        references = source_references()
        try:
            report_text = await self._synthesize_pubmed_results(
                topic,
                synthesis_articles,
                **synth_kwargs,
            )
        except UpstreamExhaustedError:
            raise
        except Exception as exc:
            failure_text = (
                "PubMed sources were retrieved and retained, but synthesis failed "
                f"technically ({type(exc).__name__}). No knowledge-only text replaced "
                "the retrieved source inventory."
            )
            if source_report is not None:
                source_report.report = failure_text
                source_report.references = references
            final = {
                "status": "partial",
                "report": failure_text,
                "references": references,
                "findings": [failure_text],
                "report_id": source_report.id if source_report is not None else None,
                "search_rounds": len(search_history),
                "source_processing_error": f"{type(exc).__name__}: {exc}"[:1000],
                "source_provenance": "pubmed_retrieved_partial",
                "retained_source_report_ids": (
                    [source_report.id] if source_report is not None else []
                ),
            }
        else:
            finalize = self._finalize_result
            try:
                finalize_parameters = inspect.signature(finalize).parameters
            except (TypeError, ValueError):
                finalize_parameters = {}
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in finalize_parameters.values()
            )
            if "existing_report" in finalize_parameters or accepts_kwargs:
                finalize_kwargs = {"existing_report": source_report}
                if (
                    "partial_source_provenance" in finalize_parameters
                    or accepts_kwargs
                ):
                    finalize_kwargs["partial_source_provenance"] = (
                        "pubmed_retrieved_partial"
                    )
                final = await finalize(
                    topic, report_text, references, state, len(search_history),
                    **finalize_kwargs,
                )
            else:
                final = await finalize(
                    topic, report_text, references, state, len(search_history)
                )
        final["focused_evidence"] = focused_evidence
        final["triage"] = latest_triage
        final["search_meta"] = {
            "stop_reason": stop_reason,
            "cache_hit": False,
            "rounds": len(search_history),
            "article_count": len(all_articles),
            "seed_queries": sum(
                1 for attempt in query_attempts if attempt.get("kind") == "seed"
            ),
            "cluster_evidence": len(focused_evidence),
            "sufficiency_evaluations": evaluation_count,
            "elapsed_seconds": round(time.monotonic() - search_started, 3),
            "query_policy_version": active_query_policy_version,
            "query_attempts": copy.deepcopy(query_attempts),
            **_evaluation_meta(latest_evaluation),
        }
        if final["search_meta"].get("evaluation_failed"):
            final.update({
                "status": "partial",
                "source_processing_error": (
                    "PI literature search review failed technically; retrieved sources "
                    "were retained but scientific coverage remains unassessed."
                ),
                "source_provenance": "pubmed_retrieved_partial",
                "retained_source_report_ids": (
                    [source_report.id] if source_report is not None else []
                ),
            })
        if stop_reason in {"wall_clock_budget", "pubmed_error"}:
            stop_error = (
                technical_stop_error
                or f"PubMed search stopped technically: {stop_reason}."
            )
            existing_error = str(final.get("source_processing_error") or "")
            final.update({
                "status": "partial",
                "source_processing_error": (
                    f"{existing_error}; {stop_error}" if existing_error else stop_error
                )[:1000],
                "source_provenance": "pubmed_retrieved_partial",
                "retained_source_report_ids": (
                    [source_report.id] if source_report is not None else []
                ),
            })
        cache_result(final)
        return final

    async def _run_seed_queries(
        self,
        seed_queries: List[str],
        *,
        max_results: int,
        seen_pmids: Set[str],
        query_attempts: List[Dict[str, Any]],
        deadline: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """Run deterministic seed queries once and return their new articles.

        ``max_results`` is both the per-seed retmax and the budget for the list:
        seeds are tried in order and the rest are skipped once the budget is
        filled, so a caller can pass a precise query followed by a broader
        fallback and pay for the second one only when the first comes up short.

        A seed is supplementary evidence, never the search's success condition:
        its failure is recorded in the attempt log and skipped, so the primary
        query's own outcome still decides whether the search falls back.
        """
        collected: List[Dict[str, str]] = []
        for raw_seed in seed_queries or []:
            seed = str(raw_seed or "").strip()
            if not seed:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                query_attempts.append({
                    "kind": "seed", "query": seed,
                    "result_count": None, "error": "wall_clock_budget",
                })
                break
            try:
                search = self._pubmed_search(seed, max_results=max_results)
                if deadline is None:
                    articles = await search
                else:
                    articles = await asyncio.wait_for(
                        search, timeout=max(0.001, deadline - time.monotonic())
                    )
            except Exception as exc:
                logger.warning(f"[Literature] Seed query failed: {str(exc)[:200]}")
                query_attempts.append({
                    "kind": "seed", "query": seed,
                    "result_count": None, "error": str(exc)[:500],
                })
                continue
            query_attempts.append({
                "kind": "seed", "query": seed, "result_count": len(articles),
            })
            for article in articles:
                pmid = article.get("pmid")
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                collected.append(article)
            if len(collected) >= max_results:
                break
            await asyncio.sleep(0.4)
        return collected

    def _reference_search_items(
        self,
        triage: Dict[str, Any],
        cluster_markers: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """One reference-query item per context-state and covered cluster.

        The contested buckets keep the adjudicated path and are absent here.
        """
        markers = cluster_markers or {}
        max_items = int(getattr(self, "max_cluster_reference_queries", 40) or 40)
        items: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for entry in triage.get("context_dependent_state") or []:
            cid = str(entry.get("cluster_id") or "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            items.append({
                "cluster_id": cid,
                "candidate_celltype": entry.get("parent_celltype"),
                "synonyms": [],
                "top_markers": list(entry.get("top_markers") or markers.get(cid) or [])[:3],
                "bucket": "context_dependent_state",
            })

        covered_names = triage.get("covered_celltypes") or {}
        for raw_cid in triage.get("covered") or []:
            cid = str(raw_cid or "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            items.append({
                "cluster_id": cid,
                "candidate_celltype": covered_names.get(cid),
                "synonyms": [],
                "top_markers": list(markers.get(cid) or [])[:3],
                "bucket": "covered",
            })
        return items[:max_items]

    async def _cluster_reference_search(
        self,
        items: List[Dict[str, Any]],
        organism: str,
        tissue: str,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One tier-1 PubMed query per cluster, with no relevance judge.

        These clusters are not contested, so an LLM adjudication per cluster buys
        little; what the matcher lacked was any literature tied to the cluster at
        all. Results come back as ``retrieved`` and are shown to the matcher as
        unadjudicated candidate references.
        """
        results: Dict[str, Any] = {}
        max_articles = int(getattr(self, "cluster_reference_articles", 5) or 5)

        def exhausted(reason, error=None, bucket=None):
            return {
                "articles": [], "status": "exhausted", "tier": 1,
                "exhaustion_reason": reason, "error": error, "bucket": bucket,
            }

        for item in items or []:
            cid = str(item.get("cluster_id") or "")
            if not cid:
                continue
            bucket = item.get("bucket")
            query = build_focused_query(
                item, organism, tissue, tier=1, bucket="rare_unresolved"
            )
            if not query:
                results[cid] = exhausted("empty_query", bucket=bucket)
                continue
            if deadline is not None and time.monotonic() >= deadline:
                results[cid] = exhausted("wall_clock_budget", bucket=bucket)
                continue
            try:
                search = self._pubmed_search(query, max_results=max_articles)
                if deadline is None:
                    articles = await search
                else:
                    articles = await asyncio.wait_for(
                        search, timeout=max(0.001, deadline - time.monotonic())
                    )
            except asyncio.TimeoutError:
                results[cid] = exhausted("wall_clock_budget", bucket=bucket)
                continue
            except Exception as exc:
                logger.warning(
                    "[Literature] Reference PubMed search failed for cluster %s: %s",
                    cid, exc,
                )
                results[cid] = exhausted("search_failed", str(exc)[:500], bucket=bucket)
                continue
            if not articles:
                results[cid] = exhausted("no_pubmed_results", bucket=bucket)
                continue
            results[cid] = {
                "articles": list(articles)[:max_articles],
                "status": "retrieved",
                "tier": 1,
                "exhaustion_reason": None,
                "error": None,
                "bucket": bucket,
            }
            await asyncio.sleep(0.4)
        return results

    async def _collect_cluster_evidence(
        self,
        triage: Dict[str, Any],
        organism: str,
        tissue: str,
        cluster_markers: Optional[Dict[str, List[str]]] = None,
        deadline: Optional[float] = None,
        retain_fn: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Retrieve literature for every triaged cluster, in two regimes.

        ``rare_unresolved`` and ``suspected_contamination`` are the contested
        calls and keep the adjudicated search: tiered queries plus an LLM
        relevance judge returning strong/limited/contradictory. ``covered`` and
        ``context_dependent_state`` clusters used to receive nothing and had to
        be annotated from the pooled review alone, so each now gets one cheap
        reference query whose result is labelled unadjudicated.
        """
        evidence: Dict[str, Any] = {}

        for bucket in ("rare_unresolved", "suspected_contamination"):
            items = triage.get(bucket) or []
            if not items:
                continue
            focused_search = self._focused_rare_search
            try:
                focused_parameters = inspect.signature(focused_search).parameters
            except (TypeError, ValueError):
                focused_parameters = {}
            focused_kwargs: Dict[str, Any] = {"bucket": bucket}
            if (
                "deadline" in focused_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in focused_parameters.values()
                )
            ):
                focused_kwargs["deadline"] = deadline
            if (
                "retain_fn" in focused_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in focused_parameters.values()
                )
            ):
                focused_kwargs["retain_fn"] = retain_fn
            found = await focused_search(items, organism, tissue, **focused_kwargs)
            for cid, data in (found or {}).items():
                entry = dict(data or {})
                entry.setdefault("bucket", bucket)
                evidence[str(cid)] = entry

        reference_items = self._reference_search_items(triage, cluster_markers)
        if reference_items:
            found = await self._cluster_reference_search(
                reference_items, organism, tissue, deadline=deadline
            )
            for cid, data in (found or {}).items():
                if str(cid) in evidence:
                    continue
                evidence[str(cid)] = dict(data or {})
        return evidence

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, str]]:
        """Parse PubMed XML response into article metadata."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ValueError("PubMed efetch returned malformed XML") from e
        if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
            raise ValueError("PubMed efetch returned an unexpected or error response")

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                pmid = article_elem.findtext(".//PMID", "")
                if not pmid.strip():
                    raise ValueError("PubMed efetch article is missing its PMID")
                title_elem = article_elem.find(".//ArticleTitle")
                title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""

                # Concatenate abstract sections (use itertext to capture sub-element text like <i>, <sub>)
                abstract_parts = article_elem.findall(".//AbstractText")
                abstract = " ".join(
                    ''.join(p.itertext()).strip()
                    for p in abstract_parts
                    if ''.join(p.itertext()).strip()
                )

                year = article_elem.findtext(".//PubDate/Year", "")
                if not year:
                    # Try MedlineDate as fallback
                    medline_date = article_elem.findtext(".//PubDate/MedlineDate", "")
                    if medline_date:
                        year = medline_date[:4]

                # Extract authors
                authors = []
                for author_elem in article_elem.findall(".//Author"):
                    last = author_elem.findtext("LastName", "")
                    first = author_elem.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {first}".strip())

                # Extract journal
                journal = article_elem.findtext(".//Journal/Title", "")

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": ", ".join(authors[:5]),
                    "journal": journal,
                })
            except (AttributeError, TypeError):
                continue

        return articles

    def _entrez_api_key(self) -> str:
        """An optional NCBI API key. Absent is the normal, supported case.

        A key is never required to use E-utilities; it only lifts the per-IP
        ceiling from three requests per second to ten. It is read from the
        environment rather than the literature config so that raising the
        ceiling stays an operational choice and needs no config migration.
        """
        return str(
            getattr(self, "pubmed_api_key", "") or os.environ.get("NCBI_API_KEY", "")
        ).strip()

    def _entrez_identity_params(self) -> Dict[str, str]:
        """The caller identification NCBI asks every client to send."""
        params: Dict[str, str] = {"tool": NCBI_TOOL_NAME}
        if self.pubmed_email:
            params["email"] = self.pubmed_email
        api_key = self._entrez_api_key()
        if api_key:
            params["api_key"] = api_key
        return params

    async def _entrez_get(
        self,
        url: str,
        params: Dict[str, str],
        client_kwargs: Dict[str, Any],
    ) -> httpx.Response:
        """One throttled E-utilities request, retried when NCBI rejects the rate."""
        rate = (
            NCBI_RATE_LIMIT_PER_SECOND_WITH_KEY
            if self._entrez_api_key()
            else NCBI_RATE_LIMIT_PER_SECOND
        )
        encoded_length = len(url) + 1 + len(urllib.parse.urlencode(params))
        use_post = encoded_length >= NCBI_MAX_GET_URI_CHARS
        if use_post:
            logger.warning(
                "[Literature] NCBI E-utilities URI for %s is %d characters; using POST",
                url,
                encoded_length,
            )
        last_error: Optional[Exception] = None
        for attempt in range(NCBI_RATE_LIMIT_RETRIES + 1):
            await _entrez_throttle.acquire(rate)
            async with httpx.AsyncClient(**client_kwargs) as client:
                if use_post:
                    resp = await client.post(url, data=params)
                else:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 414:
                        # The 414->POST switch is an in-attempt transport
                        # correction, not a new 429 retry iteration. It still
                        # spends a second request, so take another throttle slot.
                        await _entrez_throttle.acquire(rate)
                        logger.warning(
                            "[Literature] NCBI E-utilities GET for %s was %d "
                            "characters and returned 414; retrying as POST",
                            url,
                            encoded_length,
                        )
                        use_post = True
                        resp = await client.post(url, data=params)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            last_error = httpx.HTTPStatusError(
                "NCBI E-utilities returned 429 Too Many Requests",
                request=resp.request,
                response=resp,
            )
            if attempt < NCBI_RATE_LIMIT_RETRIES:
                delay = NCBI_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "[Literature] NCBI rate limit hit (attempt %s/%s); retrying in %.1fs",
                    attempt + 1,
                    NCBI_RATE_LIMIT_RETRIES + 1,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _pubmed_search(
        self,
        query: str,
        max_results: int = 20
    ) -> List[Dict[str, str]]:
        """Search PubMed using NCBI E-utilities and return article metadata."""
        client_kwargs: Dict[str, Any] = {"timeout": 30.0}
        if self.proxy_url:
            client_kwargs["proxy"] = self.proxy_url

        # Step 1: esearch to get PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params: Dict[str, str] = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
            **self._entrez_identity_params(),
        }

        resp = await self._entrez_get(search_url, search_params, client_kwargs)
        search_data = resp.json()
        if not isinstance(search_data, dict):
            raise ValueError("PubMed esearch returned an invalid response object")
        search_result = search_data.get("esearchresult")
        if (
            search_data.get("error") or search_data.get("ERROR")
            or not isinstance(search_result, dict)
            or search_result.get("error") or search_result.get("ERROR")
        ):
            raise ValueError("PubMed esearch returned an error or missing result")
        pmids = search_result.get("idlist")
        if not isinstance(pmids, list) or any(
            not isinstance(pmid, str) or not pmid.isdigit() for pmid in pmids
        ):
            raise ValueError("PubMed esearch returned an invalid or missing PMID list")
        if not pmids:
            return []

        # Step 2: efetch to get article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params: Dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            **self._entrez_identity_params(),
        }

        resp = await self._entrez_get(fetch_url, fetch_params, client_kwargs)
        xml_text = resp.text

        # Step 3: Parse XML
        articles = self._parse_pubmed_xml(xml_text)
        if not articles:
            raise ValueError("PubMed efetch returned no articles for retrieved PMIDs")
        return articles

    async def _quick_search(
        self,
        topic: str,
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """
        Quick literature search using LLM knowledge
        (Final fallback when all search methods fail)
        """
        prompt = f"""
Based on your knowledge, please provide a literature review on the following topic:

Topic: {topic}

Please include:
1. Related major findings and mechanisms
2. Important references (if you know specific papers)
3. Current research consensus
4. Unresolved questions

Note: If you are not certain about specific citation information, please clearly state that this is a summary based on general knowledge.
"""

        if self.fallback_model:
            if self.fallback_llm_client:
                client = self.fallback_llm_client
                logger.info(
                    f"[Literature] Quick search using fallback model: {self.fallback_model} "
                    f"(via fallback client -> {client.base_url})"
                )
            else:
                client = self.llm_client
                logger.warning(
                    f"[Literature] Quick search using fallback model: {self.fallback_model} "
                    f"but fallback_llm_client is None, using default client -> {client.base_url}"
                )
            resp = await self._budgeted_text_call(
                client.chat,
                model=self.fallback_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                prefer_stream=True,
            )
            response = resp.content or ""
        else:
            response = await self.chat_without_history(prompt, temperature=0.3)

        response = (
            "Knowledge-only review; no live source retrieval supports this text.\n\n"
            + response
        )
        report = LiteratureReport(
            topic=topic,
            report=response,
            references=[]
        )

        self.reports.append(report)
        if state is not None:
            state.literature_reports.append(report)

        findings = await self._extract_findings(response)

        return {
            "status": "completed",
            "report": response,
            "references": [],
            "findings": findings,
            "note": "Review based on LLM knowledge, not real-time literature search",
            "source_provenance": "knowledge_only",
            "report_id": report.id
        }

    async def _run_query_batches(
        self, queries, candidate, matching_cache, timeout, *,
        all_evidence, executed_categories, triage_metadata_list, all_limitations,
        deadline: Optional[float] = None,
        triaged_pmids: Optional[Set[str]] = None,
    ) -> bool:
        """Run a list of query batches, accumulating into the passed-in lists.
        Returns True if any batch timed out or the direction deadline expired."""
        timed_out = False

        def direction_deadline_expired() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        def mark_direction_timeout() -> None:
            if "Direction-level timeout exceeded" not in all_limitations:
                all_limitations.append("Direction-level timeout exceeded")

        for q in queries:
            if direction_deadline_expired():
                mark_direction_timeout()
                return True
            evidence, meta, q_timedout, limitations = await self._execute_query_batch(
                q, candidate, matching_cache, timeout, deadline=deadline,
                triaged_pmids=triaged_pmids,
            )
            all_evidence.extend(evidence)
            if not q.degraded:
                executed_categories.add(q.category)
            if meta:
                triage_metadata_list.append(meta)
            if q_timedout:
                timed_out = True
            all_limitations.extend(limitations)
            if direction_deadline_expired():
                mark_direction_timeout()
                timed_out = True
                break
        return timed_out

    async def _synthesize_pubmed_results(
        self,
        topic: str,
        articles: List[Dict[str, str]],
        evaluation: Optional[Dict[str, Any]] = None,
        max_articles: Optional[int] = None,
    ) -> str:
        """Synthesize PubMed articles into a comprehensive literature review."""
        # Build article text with abstracts; only the first `max_articles`
        # (default `synthesis_max_articles`, 20) are read.
        article_cap = int(
            max_articles or getattr(self, "synthesis_max_articles", 20) or 20
        )
        articles_text = "\n\n".join(
            f"[PMID: {a['pmid']}] {a['title']} ({a.get('year', 'N/A')})\n"
            f"Authors: {a.get('authors', 'N/A')}\n"
            f"Journal: {a.get('journal', 'N/A')}\n"
            f"Abstract: {a['abstract'][:600]}"
            for a in articles[:article_cap]
        )

        evaluation_context = ""
        if isinstance(evaluation, dict):
            assessment = {
                "search_exhausted": bool(evaluation.get("search_exhausted")),
                "evidence_strength": str(
                    evaluation.get("evidence_strength") or "unassessed"
                ),
                "reasoning": str(evaluation.get("reasoning") or "")[:2000],
            }
            evaluation_context = f"""
Advisory assessment from the Literature Agent's search review:
{json.dumps(assessment, ensure_ascii=False)}

Use this assessment as context for scientific judgment, not as a host acceptance
decision. Keep the retrieved articles visible, but distinguish direct evidence
from adjacent or background work. If no retrieved paper is directly relevant,
state that plainly and do not cite adjacent papers as support for the target claim.
"""

        prompt = f"""Based on the following PubMed articles, write a comprehensive literature review.

Please include:
1. Overview of the research field and key findings
2. Molecular mechanisms described in the literature
3. Areas of consensus and controversy
4. Current research trends and recent advances
5. Cite specific PMIDs that support each point (format: PMID: XXXXX)
6. Gaps in current knowledge and future directions

Write a well-structured review with clear sections.

Topic: {topic}

Articles ({len(articles)} total):
{articles_text}
{evaluation_context}"""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.4)
        return response
