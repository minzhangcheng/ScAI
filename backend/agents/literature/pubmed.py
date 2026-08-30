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
import re
import time
import xml.etree.ElementTree as ET

import httpx


logger = logging.getLogger(__name__)


def _normalized_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _search_cache_key(
    *, topic: str, query: str, organism: str, tissue: str, pi_guided: bool
) -> str:
    payload = {
        "topic": _normalized_search_text(topic),
        "query": _normalized_search_text(query),
        "organism": _normalized_search_text(organism),
        "tissue": _normalized_search_text(tissue),
        "pi_guided": bool(pi_guided),
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
    ) -> dict:
        """两级 focused 检索,按簇返回 {articles,status,tier}。

        rare_unresolved: tier1(加 tissue) → 仅当 exhausted/contradictory 时升级 tier2(去 tissue)。
        suspected_contamination: 顺序跑 tier1=discovery(无 tissue) 再 tier2=plausibility(加 tissue),
        (受 NCBI 限速约束，未并发；如需并发须配 Semaphore 后再改)
                                 取相关性 rank 更高者(spec §5.2: 两条都查)。
        空 query 直接记为 exhausted,不发请求。
        """
        results: dict = {}

        async def _run(item, tier):
            def exhausted(reason, error=None):
                return {
                    "articles": [],
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

            timeout = remaining()
            if deadline is not None and timeout is None:
                return exhausted("wall_clock_budget")
            try:
                if timeout is not None:
                    await asyncio.wait_for(asyncio.sleep(0.4), timeout=timeout)
                else:
                    await asyncio.sleep(0.4)
            except asyncio.TimeoutError:
                return exhausted("wall_clock_budget")

            try:
                timeout = remaining()
                if deadline is not None and timeout is None:
                    return exhausted("wall_clock_budget")
                judge = self._judge_focused_relevance(item, tissue, arts, bucket)
                judged = await (
                    asyncio.wait_for(judge, timeout=timeout)
                    if timeout is not None else judge
                )
            except asyncio.TimeoutError:
                return exhausted("wall_clock_budget")
            except Exception as e:
                logger.warning(
                    "[Literature] Focused relevance judge failed for cluster %s tier %s: %s",
                    item.get("cluster_id"),
                    tier,
                    e,
                )
                return exhausted("judge_failed", str(e))

            status = judged.get("status", "exhausted") if isinstance(judged, dict) else "exhausted"
            return {
                "articles": arts,
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
            else:
                best = await _run(item, 1)
                if best["status"] in ("exhausted", "contradictory"):
                    alt = await _run(item, 2)
                    if self._focused_result_score(alt) > self._focused_result_score(best):
                        best = alt
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
    ) -> Dict[str, Any]:
        """Multi-round PubMed search with LLM-guided query refinement.

        When ``initial_query`` and ``evaluate_fn`` are provided, the search
        uses PI Agent-guided query generation and evaluation instead of
        Literature Agent's own LLM-based methods.
        """
        pi_guided = initial_query is not None and evaluate_fn is not None
        label = "PI-guided PubMed" if pi_guided else "PubMed"

        all_articles: List[Dict[str, str]] = []
        seen_pmids: Set[str] = set()
        search_history: List[Dict[str, Any]] = []
        latest_triage: Dict[str, Any] = {}
        # Separate from latest_triage, which is the annotation triage payload:
        # only the PI-guided branch ever wrote that, so the plain PubMed branch
        # produced metadata contradicting its own stop_reason.
        latest_evaluation: Dict[str, Any] = {}
        plateau_streak = 0
        evaluation_count = 0
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
        deadline = search_started + search_wall_clock_seconds

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
        )
        normalized_cache_query = _normalized_search_text(query)

        def cache_result(result: Dict[str, Any]) -> None:
            if state is None:
                return
            cache = getattr(state, "literature_search_cache", None)
            if cache is None:
                cache = {}
                setattr(state, "literature_search_cache", cache)
            cache[cache_key] = {
                "normalized_topic": _normalized_search_text(topic),
                "normalized_query": normalized_cache_query,
                "result": copy.deepcopy(result),
            }
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
                break
            logger.info(
                f"[Literature] {label} round {round_num}/{self.max_search_rounds}: "
                f"{query[:100]}"
            )

            # Search PubMed (first round uses validation + degradation)
            try:
                if round_num == 1:
                    query, articles = await asyncio.wait_for(
                        self._validate_and_degrade_query(
                            query, topic,
                            max_results=pubmed_max_results or 20
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
                break
            except Exception as e:
                logger.warning(f"[Literature] PubMed search failed in round {round_num}: {e}")
                if round_num == 1 and not all_articles:
                    raise  # Re-raise to trigger fallback
                stop_reason = "pubmed_error"
                break

            # Deduplicate by PMID
            new_articles = []
            for a in articles:
                if a["pmid"] not in seen_pmids:
                    new_articles.append(a)
                    seen_pmids.add(a["pmid"])
            all_articles.extend(new_articles)

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
                "new_results": len(new_articles),
                "total_results": len(all_articles),
            })

            logger.info(
                f"[Literature] {label} round {round_num}: "
                f"{len(new_articles)} new articles, {len(all_articles)} total"
            )

            # Evaluate if more searching needed
            if stop_reason == "article_budget":
                logger.info(f"[Literature] {label} stop: article budget")
                break
            if plateau_streak >= plateau_rounds:
                stop_reason = "marginal_yield_plateau"
                logger.info(f"[Literature] {label} stop: marginal-yield plateau")
                break

            if round_num < self.max_search_rounds and all_articles:
                if evaluation_count >= max_sufficiency_evaluations:
                    stop_reason = "sufficiency_evaluation_budget"
                    break
                if pi_guided:
                    evaluation_count += 1
                    try:
                        evaluation = await asyncio.wait_for(
                            evaluate_fn(all_articles, search_history),
                            timeout=max(0.001, deadline - time.monotonic()),
                        )
                    except asyncio.TimeoutError:
                        stop_reason = "wall_clock_budget"
                        break
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
                    # 确定性语义终止: 全部簇已归桶 且 triage 非低置信
                    if evaluation.get("triage_complete") and not evaluation.get("triage_low_confidence"):
                        logger.info(f"[Literature] {label} semantic termination: all clusters triaged")
                        stop_reason = "triage_complete"
                        break
                else:
                    evaluation_count += 1
                    try:
                        evaluation = await asyncio.wait_for(
                            self._evaluate_pubmed_results(
                                topic, all_articles, search_history
                            ),
                            timeout=max(0.001, deadline - time.monotonic()),
                        )
                    except asyncio.TimeoutError:
                        stop_reason = "wall_clock_budget"
                        break

                latest_evaluation = evaluation
                decision, next_query = _search_continuation(
                    evaluation, query, label, round_num
                )
                if decision is not None:
                    stop_reason = decision
                    break
                query = next_query

            # Rate limiting for NCBI E-utilities (3 req/s without API key)
            await asyncio.sleep(0.4)

        if not all_articles:
            logger.warning(f"[Literature] No PubMed articles found ({label}), falling back to quick search")
            fallback = await self._quick_search(topic, state)
            fallback["search_meta"] = {
                "stop_reason": stop_reason,
                "cache_hit": False,
                "rounds": len(search_history),
                "article_count": 0,
                "sufficiency_evaluations": evaluation_count,
                "elapsed_seconds": round(time.monotonic() - search_started, 3),
                "fallback": "llm_knowledge",
                **_evaluation_meta(latest_evaluation),
            }
            cache_result(fallback)
            return fallback

        # --- Focused search: run after main loop, before synthesis ---
        focused_evidence: Dict[str, Any] = {}
        focused_articles_for_synthesis: List[Dict[str, Any]] = []
        if pi_guided and latest_triage:
            for bucket in ("rare_unresolved", "suspected_contamination"):
                items = latest_triage.get(bucket, [])
                if not items:
                    continue
                focused_search = self._focused_rare_search
                try:
                    focused_parameters = inspect.signature(focused_search).parameters
                except (TypeError, ValueError):
                    focused_parameters = {}
                focused_kwargs = {"bucket": bucket}
                if (
                    "deadline" in focused_parameters
                    or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in focused_parameters.values()
                    )
                ):
                    focused_kwargs["deadline"] = deadline
                fe = await focused_search(
                    items, organism, tissue, **focused_kwargs
                )
                focused_evidence.update(fe)
                for cid, data in fe.items():
                    if data.get("status") in ("strong", "limited"):
                        for article in data.get("articles", []):
                            pmid = article.get("pmid")
                            if not pmid:
                                continue
                            if pmid not in seen_pmids:
                                all_articles.append(article)
                                seen_pmids.add(pmid)
                            focused_articles_for_synthesis.append(article)

        if len(all_articles) > max_articles_per_topic:
            all_articles = all_articles[:max_articles_per_topic]
            stop_reason = "article_budget"

        # Prioritize focused evidence for synthesis because _synthesize_pubmed_results
        # intentionally truncates to the first 20 articles.
        synthesis_articles: List[Dict[str, Any]] = self._dedup_by_pmid(
            focused_articles_for_synthesis + all_articles, lambda a: a.get("pmid"),
            drop_empty=True,
        )[:max_articles_per_topic]

        report_text = await self._synthesize_pubmed_results(topic, synthesis_articles)

        references = [
            {"title": a["title"], "url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"}
            for a in all_articles
        ]

        final = await self._finalize_result(topic, report_text, references, state, len(search_history))
        final["focused_evidence"] = focused_evidence
        final["triage"] = latest_triage
        final["search_meta"] = {
            "stop_reason": stop_reason,
            "cache_hit": False,
            "rounds": len(search_history),
            "article_count": len(all_articles),
            "sufficiency_evaluations": evaluation_count,
            "elapsed_seconds": round(time.monotonic() - search_started, 3),
            **_evaluation_meta(latest_evaluation),
        }
        cache_result(final)
        return final

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, str]]:
        """Parse PubMed XML response into article metadata."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"[Literature] PubMed XML parse error: {e}")
            return []

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                pmid = article_elem.findtext(".//PMID", "")
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

    async def _pubmed_search(
        self,
        query: str,
        max_results: int = 20
    ) -> List[Dict[str, str]]:
        """Search PubMed using NCBI E-utilities and return article metadata."""
        # Step 1: esearch to get PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params: Dict[str, str] = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if self.pubmed_email:
            search_params["email"] = self.pubmed_email

        client_kwargs: Dict[str, Any] = {"timeout": 30.0}
        if self.proxy_url:
            client_kwargs["proxy"] = self.proxy_url

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(search_url, params=search_params)
            resp.raise_for_status()
            search_data = resp.json()

        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        # Step 2: efetch to get article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params: Dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if self.pubmed_email:
            fetch_params["email"] = self.pubmed_email

        await asyncio.sleep(0.4)  # Rate limiting

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(fetch_url, params=fetch_params)
            resp.raise_for_status()
            xml_text = resp.text

        # Step 3: Parse XML
        return self._parse_pubmed_xml(xml_text)

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
            )
            response = resp.content or ""
        else:
            response = await self.chat_without_history(prompt, temperature=0.3)

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
            "report_id": report.id
        }

    async def _run_query_batches(
        self, queries, candidate, matching_cache, timeout, *,
        all_evidence, executed_categories, triage_metadata_list, all_limitations,
        deadline: Optional[float] = None,
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
                q, candidate, matching_cache, timeout, deadline=deadline
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
        articles: List[Dict[str, str]]
    ) -> str:
        """Synthesize PubMed articles into a comprehensive literature review."""
        # Build article text with abstracts (limit to first 20 articles)
        articles_text = "\n\n".join(
            f"[PMID: {a['pmid']}] {a['title']} ({a.get('year', 'N/A')})\n"
            f"Authors: {a.get('authors', 'N/A')}\n"
            f"Journal: {a.get('journal', 'N/A')}\n"
            f"Abstract: {a['abstract'][:600]}"
            for a in articles[:20]
        )

        prompt = f"""Based on the following PubMed articles, write a comprehensive literature review
on the topic: {topic}

Articles ({len(articles)} total):
{articles_text}

Please include:
1. Overview of the research field and key findings
2. Molecular mechanisms described in the literature
3. Areas of consensus and controversy
4. Current research trends and recent advances
5. Cite specific PMIDs that support each point (format: PMID: XXXXX)
6. Gaps in current knowledge and future directions

Write a well-structured review with clear sections."""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.4)
        return response
