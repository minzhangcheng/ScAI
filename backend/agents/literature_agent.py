"""
Literature Agent - Multi-mode literature research with fallback chain

Supports two research modes:
1. Gemini Deep Research (Google Interactions API) — single call per topic;
   Deep Research handles multi-round search internally on Google's servers.
2. PubMed + LLM (NCBI E-utilities search + LLM synthesis) — multi-round
   iterative search with LLM-guided query refinement.

Fallback chain: Gemini DR → PubMed + LLM → quick search (LLM knowledge only).
When in DR mode, gap coverage is handled at the workflow level by falling back
to PubMed+LLM for insufficiently covered topics.
"""
from typing import Callable, Dict, Any, List, Optional, Set, Tuple
import json
import re
import asyncio
import logging
import xml.etree.ElementTree as ET

import httpx
from openai import APIConnectionError as OpenAIConnectionError

from .base import BaseAgent
from ..utils.settings_provider import AgentSettingsProvider

logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, LiteratureReport
)
from ..utils.llm_client import OpenAICompatibleClient, GoogleDeepResearchClient, RETRYABLE_EXCEPTIONS
from openai import RateLimitError

_FALLBACK_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (RateLimitError,)

# 主模型重试配置（用于 _chat_with_fallback）
_MAX_MAIN_RETRIES = 2
_RETRY_DELAY_BASE = 2.0  # 秒，与 llm_client 保持一致


class LiteratureAgent(BaseAgent):
    """
    Literature Agent - Multi-mode literature research

    Supports:
    - Gemini Deep Research (Google Interactions API)
    - OpenRouter Deep Research (models with built-in web search)
    - PubMed + LLM (NCBI E-utilities + LLM synthesis)

    Fallback chain:
    - Gemini Deep Research → PubMed + LLM → quick_search (LLM knowledge only)
    """

    # Citation-format footer for the deep_research-LLM prompt. This text is a
    # load-bearing contract with _extract_references_from_text's regex — keep it
    # verbatim. PI-authored briefs never include citation-format directives.
    _DEEP_RESEARCH_CITATION_FOOTER = (
        "IMPORTANT - Citation format:\n"
        "For each cited paper, provide a reference list at the end in this exact format:\n"
        "[N] Author et al. (Year). Title. Journal. URL_or_DOI\n"
        "Example: [1] Smith et al. (2023). Role of TGF-beta in liver fibrosis. "
        "Nature. https://doi.org/10.1038/xxx"
    )

    _FINDINGS_EXTRACT_MAX_CHARS = 100000

    def __init__(
        self,
        settings_provider: 'AgentSettingsProvider',
        # Literature-specific parameters
        literature_mode: str = "deep_research",
        deep_research_model: str = "",
        deep_research_llm_base_url: str = "",
        deep_research_api_key: str = "",
        google_api_key: str = "",
        pubmed_email: str = "",
        max_search_rounds: int = 5,
        proxy_url: Optional[str] = None,
    ):
        super().__init__(
            agent_type=AgentType.LITERATURE,
            settings_provider=settings_provider,
        )
        self.reports: List[LiteratureReport] = []

        # Mode configuration
        self.literature_mode = literature_mode
        self.deep_research_model = deep_research_model
        self.pubmed_email = pubmed_email
        self.max_search_rounds = max(1, min(20, max_search_rounds))
        self.proxy_url = proxy_url

        # Separate LLM client for deep research mode (optional)
        self._deep_research_llm_client: Optional[OpenAICompatibleClient] = None
        if deep_research_llm_base_url and deep_research_api_key:
            self._deep_research_llm_client = OpenAICompatibleClient(
                api_key=deep_research_api_key,
                base_url=deep_research_llm_base_url,
                proxy_url=proxy_url,
            )

        # Create GoogleDeepResearchClient only when needed (Gemini mode)
        self._deep_research: Optional[GoogleDeepResearchClient] = None
        if google_api_key:
            self._deep_research = GoogleDeepResearchClient(
                google_api_key, proxy_url=proxy_url
            )

        # Cache for annotate_clusters results to avoid redundant Deep Research calls
        self._annotation_cache: Dict[str, Any] = {}
        self._ANNOTATION_CACHE_MAX = 100

    @property
    def is_deep_research_available(self) -> bool:
        """Check if deep research mode is available and properly configured."""
        if self.literature_mode not in ("gemini_deep_research", "deep_research"):
            return False
        return self._deep_research is not None or bool(self.deep_research_model)

    @property
    def fallback_model(self) -> str:
        return self._provider.get_model("pi")

    @property
    def fallback_llm_client(self) -> OpenAICompatibleClient:
        return self._provider.get_llm_client("pi")

    def resolve_model_label(self, search_mode: str) -> str:
        """Map a ``search_mode`` string (from _do_research) to a display label
        for the final report's Literature row.

        - gemini_deep_research -> "Google Deep Research"
        - deep_research        -> the configured DR model (or lit model if unset)
        - everything else (pubmed_llm, quick_search, "(fallback from ...)") ->
          the literature synthesis model (matches the legacy report value)
        """
        mode = (search_mode or "").strip()
        if mode == "gemini_deep_research":
            return "Google Deep Research"
        if mode == "deep_research":
            return self.deep_research_model or self.model
        return self.model

    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        mode_override: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process literature research instruction"""
        effective_mode = mode_override if mode_override is not None else self.literature_mode
        if effective_mode == "disabled":
            logger.info("[Literature] Literature agent is disabled, skipping")
            return {"report": "", "status": "disabled"}

        topic = kwargs.get("topic", instruction)
        use_deep_research = kwargs.get("use_deep_research", True)
        current_round_findings = kwargs.get("current_round_findings", None)
        research_brief = kwargs.get("research_brief")

        logger.info(
            f"[Literature] Processing topic: {topic[:100]}... "
            f"(mode={effective_mode}, deep={use_deep_research})"
        )

        if use_deep_research:
            return await self._do_research(topic, state, current_round_findings=current_round_findings, mode_override=mode_override, research_brief=research_brief)
        else:
            return await self._quick_search(topic, state)

    # ===================================================================
    # Core routing with fallback chain
    # ===================================================================

    async def _do_research(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        pubmed_max_results: Optional[int] = None,
        current_round_findings: Optional[List[str]] = None,
        mode_override: Optional[str] = None,
        research_brief: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route to configured research mode with fallback chain.

        Fallback: deep_research/gemini → pubmed → quick_search

        Args:
            topic: Research topic
            state: Optional analysis state for context
            pubmed_max_results: Override max articles per PubMed search round (default 20)
            current_round_findings: Findings from current round's other agents
            mode_override: Override self.literature_mode for this call
        """
        mode = mode_override if mode_override is not None else self.literature_mode
        actual_mode = mode  # Track which mode actually produced results

        try:
            if mode == "deep_research":
                result = await self._deep_research_via_chat(topic, state, research_brief=research_brief)
            elif mode == "gemini_deep_research":
                result = await self._multi_round_gemini(topic, state, research_brief=research_brief)
            elif mode == "pubmed_llm":
                result = await self._multi_round_pubmed(
                    topic, state, pubmed_max_results=pubmed_max_results,
                    current_round_findings=current_round_findings
                )
            else:
                logger.warning(f"[Literature] Unknown mode: {mode}, using quick search")
                actual_mode = "quick_search"
                result = await self._quick_search(topic, state)
            result["search_mode"] = actual_mode
            return result
        except Exception as e:
            logger.warning(f"[Literature] Mode {mode} failed: {e}")

            # Fallback: deep_research or gemini → try pubmed
            if mode in ("gemini_deep_research", "deep_research"):
                logger.info("[Literature] Falling back to PubMed + LLM mode")
                try:
                    result = await self._multi_round_pubmed(
                        topic, state, pubmed_max_results=pubmed_max_results,
                        current_round_findings=current_round_findings
                    )
                    result["search_mode"] = f"pubmed_llm (fallback from {mode})"
                    return result
                except Exception as e2:
                    logger.warning(f"[Literature] PubMed fallback also failed: {e2}")

            # Final fallback: quick search (LLM knowledge only)
            logger.info("[Literature] Falling back to quick search (LLM knowledge only)")
            result = await self._quick_search(topic, state)
            result["search_mode"] = f"quick_search (fallback from {mode})"
            return result

    # ===================================================================
    # Mode 1: Gemini Deep Research (single call — DR handles rounds internally)
    # ===================================================================

    async def _multi_round_gemini(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        research_brief: Optional[str] = None
    ) -> Dict[str, Any]:
        """Single Gemini Deep Research call via Google Interactions API.

        Deep Research handles multi-round search/synthesis internally on Google's
        servers, so we only need one call per topic. Gap coverage (if any) is
        handled at the workflow level by falling back to PubMed+LLM.
        """
        if not self._deep_research:
            raise ValueError("Google API Key not configured for Gemini Deep Research")

        query = research_brief if research_brief else topic
        logger.info(f"[Literature] Gemini Deep Research: {query[:200]}")

        result = await self._deep_research.research(query)
        report_text = result.get("report", "")
        sources = result.get("sources", [])

        references = [
            {"title": s.get("title", ""), "url": s.get("uri", "")}
            for s in sources
        ]

        return await self._finalize_result(topic, report_text, references, state, 1)

    # ===================================================================
    # Mode 2: Deep Research via OpenRouter (single call)
    # ===================================================================

    async def _deep_research_via_chat(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        research_brief: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deep research via OpenRouter chat API (OpenAI o4, Perplexity sonar, etc.).

        Makes a single long-running chat call with high max_tokens and timeout.
        The deep research model handles web search and synthesis internally.
        """
        if not self.deep_research_model:
            raise ValueError("deep_research_model not configured for deep_research mode")

        if research_brief:
            body = research_brief
        else:
            body = (
                "Conduct a literature review on the following topic.\n\n"
                f"Research topic: {topic}"
            )

        prompt = f"{body}\n\n{self._DEEP_RESEARCH_CITATION_FOOTER}"

        logger.info(
            f"[Literature] Deep Research via {self.deep_research_model}: "
            f"{topic[:200]}"
        )

        self._set_llm_context()

        # Use separate deep research LLM client if configured
        dr_client = self._deep_research_llm_client or self.llm_client

        # The LLM client disables retries for timeout>300s (to avoid re-running
        # expensive long requests). Retry transient *connection* errors here.
        _max_conn_retries = 2
        _last_err: Optional[Exception] = None
        for _attempt in range(_max_conn_retries + 1):
            try:
                response = await dr_client.chat(
                    model=self.deep_research_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=None,
                    timeout=1800.0,
                )
                break
            except (OpenAIConnectionError, httpx.ConnectError, httpx.ConnectTimeout, OSError) as e:
                _last_err = e
                if _attempt < _max_conn_retries:
                    _delay = 5 * (_attempt + 1)
                    logger.warning(
                        f"[Literature] Deep research connection error "
                        f"(attempt {_attempt + 1}/{_max_conn_retries + 1}): "
                        f"{type(e).__name__}: {str(e)[:150]}. Retrying in {_delay}s..."
                    )
                    await asyncio.sleep(_delay)
                else:
                    raise
        else:
            raise _last_err  # type: ignore[misc]

        report_text = response.content or ""
        if not report_text:
            raise ValueError("Deep research returned empty response")

        logger.info(
            f"[Literature] Deep Research completed: "
            f"{len(report_text)} chars"
        )

        references = self._select_references(response, report_text)
        return await self._finalize_result(topic, report_text, references, state, 1)

    @staticmethod
    def _select_references(response, report_text: str):
        """Prefer structured citations from the Responses API; fall back to
        regex extraction over the report text."""
        citations = getattr(response, "citations", None)
        if citations:
            return citations
        return LiteratureAgent._extract_references_from_text(report_text)

    @staticmethod
    def _extract_references_from_text(text: str) -> List[Dict[str, str]]:
        """Extract references from deep research output text.

        Tries structured [N] Title... URL format first, falls back to plain URL extraction.
        """
        refs: List[Dict[str, str]] = []
        seen: set = set()

        # Try structured format: [N] Title... URL
        for m in re.finditer(
            r'\[(\d+)\]\s*(.{5,300}?)\s*(https?://[^\s\)\]]+)', text
        ):
            url = m.group(3).rstrip('.,;')
            if url not in seen:
                seen.add(url)
                title = m.group(2).strip(' .-,')
                refs.append({"title": title, "url": url})

        # Fallback to plain URL extraction if structured refs are insufficient
        if len(refs) < 3:
            for m in re.finditer(r'https?://[^\s\)\]]+', text):
                url = m.group(0).rstrip('.,;')
                if url not in seen:
                    seen.add(url)
                    refs.append({"title": "", "url": url})

        return refs

    # ===================================================================
    # Mode 3: PubMed + LLM (multi-round)
    # ===================================================================

    async def _validate_and_degrade_query(
        self,
        query: str,
        topic: str,
        max_results: int = 20
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Try the query; if PubMed returns 0 results, progressively simplify.

        Degradation strategy:
        1. Original LLM-generated query
        2. Strip field tags and MeSH qualifiers, keep core terms
        3. Simple keyword search from topic + species + tissue

        Returns (final_query, articles) tuple.
        """
        attempt = 0
        max_attempts = 3
        while attempt < max_attempts:
            try:
                articles = await self._pubmed_search(query, max_results=max_results)
                if articles:
                    if attempt > 0:
                        logger.info(f"[Literature] Query succeeded after {attempt} degradation(s): {query[:80]}")
                    return query, articles
            except Exception as e:
                logger.warning(f"[Literature] PubMed search error (attempt {attempt + 1}): {e}")

            # Degrade the query
            if attempt == 0:
                # Strategy 1: Strip field tags and quotes
                degraded = re.sub(r'\[(?:MeSH|Title/Abstract|Title|Author|Journal)\]', '', query)
                degraded = re.sub(r'"([^"]*)"', r'\1', degraded)
                degraded = degraded.strip()
                if degraded and degraded != query:
                    query = degraded
                    logger.info(f"[Literature] Degrading query (strip tags): {query[:80]}")
                    await asyncio.sleep(0.4)
                    attempt += 1
                    continue
                # Strategy 1 had no effect — skip to Strategy 2 immediately
                attempt = 1

            # Strategy 2: Fall back to simple keyword search from topic
            topic_words = topic.split()[:5]
            query = " ".join(topic_words) if topic_words else topic
            logger.info(f"[Literature] Degrading query (simple keywords): {query[:80]}")
            await asyncio.sleep(0.4)
            # Final search attempt with simplified query, then exit
            try:
                articles = await self._pubmed_search(query, max_results=max_results)
                if articles:
                    logger.info(f"[Literature] Query succeeded with simple keywords: {query[:80]}")
                    return query, articles
            except Exception as e:
                logger.warning(f"[Literature] PubMed search error (simple keywords): {e}")
            break

        return query, []

    async def _multi_round_pubmed(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        pubmed_max_results: Optional[int] = None,
        current_round_findings: Optional[List[str]] = None,
        initial_query: Optional[str] = None,
        evaluate_fn: Optional[Callable] = None,
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

        # Generate initial PubMed query from the topic (or use provided one)
        if pi_guided:
            query = initial_query
        else:
            query = await self._generate_pubmed_query(topic, state, current_round_findings)

        for round_num in range(1, self.max_search_rounds + 1):
            logger.info(
                f"[Literature] {label} round {round_num}/{self.max_search_rounds}: "
                f"{query[:100]}"
            )

            # Search PubMed (first round uses validation + degradation)
            try:
                if round_num == 1:
                    query, articles = await self._validate_and_degrade_query(
                        query, topic,
                        max_results=pubmed_max_results or 20
                    )
                else:
                    articles = await self._pubmed_search(query, max_results=pubmed_max_results or 20)
            except Exception as e:
                logger.warning(f"[Literature] PubMed search failed in round {round_num}: {e}")
                if round_num == 1 and not all_articles:
                    raise  # Re-raise to trigger fallback
                break

            # Deduplicate by PMID
            new_articles = []
            for a in articles:
                if a["pmid"] not in seen_pmids:
                    new_articles.append(a)
                    seen_pmids.add(a["pmid"])
            all_articles.extend(new_articles)

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
            if round_num < self.max_search_rounds and all_articles:
                if pi_guided:
                    try:
                        evaluation = await evaluate_fn(all_articles, search_history)
                    except Exception as e:
                        logger.warning(f"[Literature] PI evaluation failed in round {round_num}: {e}, stopping search")
                        evaluation = {"sufficient": True}
                else:
                    evaluation = await self._evaluate_pubmed_results(
                        topic, all_articles, search_history
                    )
                if evaluation.get("sufficient", True):
                    logger.info(f"[Literature] {label} sufficient after {round_num} rounds")
                    break
                query = evaluation.get("refined_query", query)

            # Rate limiting for NCBI E-utilities (3 req/s without API key)
            await asyncio.sleep(0.4)

        if not all_articles:
            logger.warning(f"[Literature] No PubMed articles found ({label}), falling back to quick search")
            return await self._quick_search(topic, state)

        # Synthesize all collected articles
        report_text = await self._synthesize_pubmed_results(topic, all_articles)

        references = [
            {"title": a["title"], "url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"}
            for a in all_articles
        ]

        return await self._finalize_result(topic, report_text, references, state, len(search_history))

    # ===================================================================
    # PubMed E-utilities integration
    # ===================================================================

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

    async def _generate_pubmed_query(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        current_round_findings: Optional[List[str]] = None
    ) -> str:
        """Use LLM to convert a research topic into an optimized PubMed search query.

        Passes raw experimental context to the LLM and lets it identify species,
        tissue, gene names, and disease models rather than using hardcoded regex.
        """
        ctx = self._build_research_context(topic, state, current_round_findings)

        # Build optional context sections
        extra_sections = []
        if ctx["treatment_groups"]:
            extra_sections.append(f"Treatment groups: {', '.join(ctx['treatment_groups'])}")
        if ctx["biochemistry_indicators"]:
            extra_sections.append(f"Biochemistry indicators: {', '.join(ctx['biochemistry_indicators'][:8])}")
        if ctx["previous_findings"]:
            extra_sections.append(f"Previous analysis findings:\n{ctx['previous_findings'][:500]}")
        if ctx["current_round_findings"]:
            extra_sections.append(f"Current round findings:\n{ctx['current_round_findings'][:300]}")
        extra_block = "\n\n".join(extra_sections)

        prompt = f"""Convert the following research topic into an optimized PubMed search query.

Research topic: {topic}

Experimental background:
{ctx["initial_instruction"][:500] if ctx["initial_instruction"] else "Not available."}

{extra_block}

Rules for PubMed query:
1. Use MeSH terms where appropriate (e.g., "Mice, Knockout"[MeSH])
2. Use Boolean operators (AND, OR, NOT)
3. Keep the query focused but not overly restrictive — aim for 20-200 results
4. Identify species, tissue/organ, and disease model from the background and include as filters
5. If specific gene names appear in the findings, pick the 2-3 most relevant ones
6. Use field tags like [Title/Abstract], [MeSH] when helpful
7. Prefer broader terms over highly specific ones to avoid zero-result queries

Return ONLY the PubMed query string, nothing else."""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.2)
        # Clean up the response - remove quotes, markdown, etc.
        query = response.strip().strip('"').strip("'").strip("`")
        # Remove markdown code block markers
        query = re.sub(r'^```\w*\n?', '', query)
        query = re.sub(r'\n?```$', '', query)
        return query.strip()

    # ===================================================================
    # LLM evaluation and synthesis helpers
    # ===================================================================

    async def _chat_with_fallback(self, prompt: str, temperature: float = 0.3) -> Tuple[str, bool]:
        """Returns (response, used_fallback)."""
        # 主模型重试 N 次
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_MAIN_RETRIES):
            try:
                return await self.chat_without_history(prompt, temperature=temperature), False
            except _FALLBACK_EXCEPTIONS as e:
                last_err = e
                if isinstance(e, RateLimitError):
                    # RateLimitError 已由内层 retry_with_backoff 做了充分退避，外层继续重试无意义
                    logger.warning(
                        f"[Literature] LLM call hit RateLimitError on attempt {attempt + 1}/{_MAX_MAIN_RETRIES}, skipping to fallback"
                    )
                    break  # 跳出主模型重试循环，走 fallback
                if attempt < _MAX_MAIN_RETRIES - 1:
                    delay = _RETRY_DELAY_BASE * (attempt + 1)
                    logger.warning(
                        f"[Literature] LLM call failed (attempt {attempt + 1}/{_MAX_MAIN_RETRIES}): "
                        f"{type(e).__name__}: {str(e)[:100]}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # 最后一次失败（且非 RateLimitError），记录日志
                    logger.warning(
                        f"[Literature] LLM call failed final attempt ({attempt + 1}/{_MAX_MAIN_RETRIES}): "
                        f"{type(e).__name__}: {str(e)[:100]}"
                    )

        # 所有主模型重试都失败，尝试 fallback model
        if self.fallback_llm_client and self.fallback_model:
            logger.warning(
                f"[Literature] Main model failed after {_MAX_MAIN_RETRIES} attempts, "
                f"trying fallback model: {self.fallback_model}"
            )
            try:
                resp = await self.fallback_llm_client.chat(
                    model=self.fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                if not resp.content:
                    logger.warning("[Literature] Fallback model returned empty content")
                    raise last_err or ValueError("Fallback model returned empty content")
                return resp.content, True
            except Exception as fallback_err:
                # fallback 也失败时，记录 fallback 的真实错误
                logger.error(
                    f"[Literature] Fallback model also failed: "
                    f"{type(fallback_err).__name__}: {str(fallback_err)[:150]}"
                )
                raise fallback_err from last_err

        # 无 fallback 时抛出主模型的最后错误
        raise last_err or ValueError("LLM call failed")

    async def _evaluate_search_results(
        self,
        topic: str,
        search_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM evaluates if search results are sufficient (for Gemini/OpenRouter modes)."""
        history_summary = json.dumps(search_history, ensure_ascii=False, indent=2)

        prompt = f"""Evaluate whether the search results so far are sufficient for a comprehensive
literature review on the topic:

Topic: {topic}

Search history:
{history_summary[:8000]}

Respond in JSON format:
{{
    "sufficient": true or false,
    "reasoning": "why sufficient or not",
    "refined_query": "suggested refined search query if not sufficient",
    "missing_aspects": ["aspects not yet covered"]
}}"""

        response = await self.chat_without_history(prompt, temperature=0.3)
        result = self._safe_parse_json(response)
        if not result or "sufficient" not in result:
            return {"sufficient": True}
        return result

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
- Are there articles directly relevant to the topic?
- Are key aspects covered (mechanism, phenotype, therapeutic implications)?
- Are there at least 5 highly relevant articles?

Respond in JSON format:
{{
    "sufficient": true or false,
    "reasoning": "why sufficient or not",
    "refined_query": "a NEW PubMed search query targeting missing aspects (MUST differ from all previous queries)",
    "missing_aspects": ["aspects not yet covered"]
}}

IMPORTANT:
- The refined_query MUST be a valid PubMed search query using AND/OR operators
- The refined_query MUST NOT duplicate any previous query listed above
- If the articles are mostly irrelevant, try a simpler, broader query"""

        response, _ = await self._chat_with_fallback(prompt, temperature=0.3)
        result = self._safe_parse_json(response)
        if not result or "sufficient" not in result:
            return {"sufficient": True}
        return result

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

    async def _synthesize_multiple_reports(
        self,
        topic: str,
        reports: List[str]
    ) -> str:
        """Combine multiple round reports into a single comprehensive review."""
        reports_text = "\n\n---\n\n".join(
            f"Report {i + 1}:\n{r[:6000]}" for i, r in enumerate(reports)
        )

        prompt = f"""Synthesize the following multiple research reports into a single comprehensive
literature review on the topic: {topic}

{reports_text}

Create a unified, well-structured review that:
1. Combines insights from all reports without redundancy
2. Maintains all specific citations and references
3. Provides a coherent narrative
4. Highlights the most important findings"""

        return await self.chat_without_history(prompt, temperature=0.4)

    # ===================================================================
    # Result finalization helper
    # ===================================================================

    async def _finalize_result(
        self,
        topic: str,
        report_text: str,
        references: List[Dict[str, str]],
        state: Optional[AnalysisState],
        search_rounds: int
    ) -> Dict[str, Any]:
        """Create and store a LiteratureReport, extract findings, return standardized result."""
        report = LiteratureReport(
            topic=topic,
            report=report_text,
            references=references,
        )
        self.reports.append(report)
        if state is not None:
            state.literature_reports.append(report)

        findings = await self._extract_findings(report_text)

        return {
            "status": "completed",
            "report": report_text,
            "references": references,
            "findings": findings,
            "report_id": report.id,
            "search_rounds": search_rounds,
        }

    # ===================================================================
    # Quick search fallback (LLM knowledge only)
    # ===================================================================

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
            resp = await client.chat(
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

    # ===================================================================
    # Shared helpers
    # ===================================================================

    def _build_research_context(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        current_round_findings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Collect raw research context from AnalysisState for LLM-based query generation.

        Instead of using regex/hardcoded lists to extract species, tissue, genes, etc.,
        this method collects raw text material and lets the downstream LLM identify
        relevant biological entities when generating PubMed queries.
        """
        context: Dict[str, Any] = {
            "topic": topic,
            "initial_instruction": "",
            "biochemistry_indicators": [],
            "treatment_groups": [],
            "previous_findings": "",
            "current_round_findings": "",
        }

        if not state:
            return context

        context["initial_instruction"] = state.initial_instruction or ""

        if state.treatment_groups:
            context["treatment_groups"] = [g.name for g in state.treatment_groups]

        if state.biochemistry_indicators:
            unique_indicators = list(set(bi.indicator for bi in state.biochemistry_indicators))
            context["biochemistry_indicators"] = unique_indicators[:10]

        # Collect truncated findings from previous rounds (let LLM identify genes/pathways)
        finding_parts = []
        for rr in state.round_results:
            for f in rr.bioinformatics_findings[:3]:
                finding_parts.append(str(f)[:150])
            for f in rr.pathology_findings[:2]:
                finding_parts.append(str(f)[:150])

        # Include previous literature reports so round 2+ searches avoid redundancy
        for lr in state.literature_reports:
            finding_parts.append(f"[Literature: {lr.topic[:80]}] {lr.report[:200]}")

        if finding_parts:
            context["previous_findings"] = "; ".join(finding_parts)

        if current_round_findings:
            context["current_round_findings"] = "; ".join(current_round_findings[:5])

        return context

    async def _extract_findings(self, report: str) -> List[str]:
        """Extract key findings from a literature report."""
        prompt = f"""
Extract key findings from the following literature review (in list form):

{report[:self._FINDINGS_EXTRACT_MAX_CHARS]}

Please list 3-5 most important findings, one per line.
"""

        response = await self.chat_without_history(prompt, temperature=0.3)

        findings = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                line = re.sub(r'^[-*•\d.]\s*', '', line)
                if line:
                    findings.append(line)

        return findings[:5]

    # ===================================================================
    # Specialized utility methods (used by bioinformatics agent)
    # ===================================================================

    async def get_cell_markers(
        self,
        cell_type: str,
        organism: str = "mouse",
        tissue: str = "",
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """Get marker genes for a specific cell type via literature search."""
        tissue_context = f" {tissue}" if tissue else ""
        topic = f"{cell_type} marker genes {organism}{tissue_context} single cell RNA sequencing"

        result = await self._do_research(topic, state)

        # Extract marker genes from report
        prompt = f"""
Extract marker genes for {cell_type} from the following literature review:

{result['report'][:8000]}

Please return in JSON format:
{{
    "cell_type": "{cell_type}",
    "positive_markers": ["gene1", "gene2", ...],
    "negative_markers": ["gene1", ...],
    "confidence": "high/medium/low"
}}
"""

        response = await self.chat_without_history(prompt, temperature=0.2)

        try:
            parsed = self._safe_parse_json(response)
            if parsed and "positive_markers" in parsed:
                return parsed
            else:
                logger.warning(f"[Literature] No JSON found in get_cell_markers for {cell_type}")
        except Exception as e:
            logger.warning(f"[Literature] Failed to parse get_cell_markers for {cell_type}: {e}")

        return {
            "cell_type": cell_type,
            "positive_markers": [],
            "negative_markers": [],
            "confidence": "low"
        }

    async def annotate_clusters(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str = "mouse",
        tissue: str = "",
        state: Optional[AnalysisState] = None,
        pi_initial_query: Optional[str] = None,
        pi_evaluate_fn: Optional[Callable] = None,
        skip_matching: bool = False,
        pi_research_brief: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Two-phase cluster annotation: one literature search + LLM matching.

        Phase 1: Single _do_research() call with tissue/organism-focused query
        Phase 2: LLM matches each cluster's markers against retrieved literature

        When PI-guided parameters are provided, Phase 1 uses PI Agent's query and
        evaluation logic instead of Literature Agent's own LLM. When skip_matching
        is True, Phase 2 is skipped (PI Agent handles matching externally).

        Args:
            cluster_markers: Dict mapping cluster_id (str) to list of top marker genes
                            e.g. {"0": ["Cd79a", "Cd79b", "Ms4a1"], "1": ["Cd3d", "Cd3e"]}
            organism: Organism name (e.g., "mouse", "human")
            tissue: Tissue/organ context (e.g., "kidney", "liver")
            state: Optional analysis state for context
            pi_initial_query: Pre-generated PubMed query from PI Agent (strong model)
            pi_evaluate_fn: Async callable for PI Agent to evaluate search sufficiency
            skip_matching: If True, skip Phase 2 (LLM matching) — PI handles it externally

        Returns:
            Dict with annotations per cluster, literature_report, references, status
        """
        if self.literature_mode == "disabled":
            logger.info("[Literature] Literature agent is disabled, skipping annotate_clusters")
            return {"annotations": {}, "literature_report": "", "references": [], "status": "disabled"}

        # --- Cache check: avoid redundant Deep Research calls ---
        all_genes_for_key: set = set()
        for genes in cluster_markers.values():
            all_genes_for_key.update(genes[:10])
        import hashlib
        genes_hash = hashlib.md5(",".join(sorted(all_genes_for_key)).encode()).hexdigest()[:16]
        cache_key = f"{organism}:{tissue}:{genes_hash}"
        if cache_key in self._annotation_cache:
            logger.info(f"[Literature] annotate_clusters cache HIT: {cache_key}")
            return self._annotation_cache[cache_key]

        n_clusters = len(cluster_markers)

        if n_clusters == 0:
            logger.warning("[Literature] annotate_clusters called with empty cluster_markers, skipping")
            return {
                "status": "error",
                "annotations": {},
                "tissue_summary": "",
                "literature_report": "",
                "references": [],
                "n_clusters": 0,
                "organism": organism,
                "error": "No clusters provided for annotation."
            }

        # ---- Phase 1: One literature search ----
        # Sample top genes across clusters for the search query
        all_genes: set = set()
        for genes in cluster_markers.values():
            all_genes.update(genes[:10])  # Top 10 per cluster
        gene_sample = ", ".join(list(all_genes)[:20])  # Cap at 20 genes

        tissue_part = f" {tissue}" if tissue else ""
        topic = (
            f"Cell types in {organism}{tissue_part} identified by single-cell RNA sequencing. "
            f"Key marker genes include: {gene_sample}. "
            f"Identify known cell type markers and signatures for{tissue_part} tissue."
        )

        logger.info(
            f"[Literature] annotate_clusters Phase 1: {n_clusters} clusters, "
            f"organism={organism}, tissue={tissue}, pi_guided={pi_initial_query is not None}"
        )

        # Use PI-guided PubMed search if PI query and evaluator are provided
        if pi_initial_query and pi_evaluate_fn and self.literature_mode == "pubmed_llm":
            # PI Agent provides the search strategy; Literature Agent executes
            research_result = await self._multi_round_pubmed(
                topic, state, pubmed_max_results=50,
                initial_query=pi_initial_query, evaluate_fn=pi_evaluate_fn,
            )
        else:
            # Default: Literature Agent handles everything (original behavior)
            # Also used for deep_research/gemini modes where PI query guidance doesn't apply
            research_result = await self._do_research(
                topic, state, pubmed_max_results=50, research_brief=pi_research_brief
            )

        literature_report = research_result.get("report", "")
        references = research_result.get("references", [])

        # ---- Phase 2: LLM matching ----
        # When skip_matching is True, PI Agent handles matching externally
        if skip_matching:
            logger.info(f"[Literature] annotate_clusters Phase 2 skipped (PI Agent will handle matching)")
            result = {
                "status": "literature_only",
                "annotations": {},
                "tissue_summary": "",
                "literature_report": literature_report,
                "references": references,
                "n_clusters": n_clusters,
                "organism": organism,
                "tissue": tissue
            }
            # Do NOT cache partial results — cache will be populated after PI matching
            return result

        markers_text = "\n".join(
            f"Cluster {cid}: {', '.join(genes[:15])}"
            for cid, genes in cluster_markers.items()
        )

        matching_prompt = f"""Based on the following literature review about cell types in {organism}{tissue_part}, \
annotate each cell cluster by matching its marker genes to known cell type signatures.

=== LITERATURE REVIEW ===
{literature_report[:12000]}

=== CLUSTER MARKER GENES (from sc.tl.rank_genes_groups) ===
{markers_text}

For each cluster, provide:
1. The most likely cell type identity
2. Possible subtypes if applicable
3. Key marker genes that support this annotation (from the cluster's markers)
4. Confidence level (high/medium/low) based on how well markers match literature
5. Brief evidence/reasoning citing the literature

IMPORTANT:
- Be specific about cell types (e.g., "CD8+ cytotoxic T cells" not just "T cells")
- If a cluster has ambiguous markers, list the top 2 candidates
- Consider the tissue context ({tissue}) when making assignments
- Use "Unknown" only if markers truly don't match any known cell type

Return a JSON object with this EXACT structure:
{{
    "annotations": {{
        "<cluster_id>": {{
            "cell_type": "Primary cell type name",
            "subtypes": ["possible subtype 1", "possible subtype 2"],
            "key_markers": ["gene1", "gene2", "gene3"],
            "confidence": "high|medium|low",
            "evidence": "Brief reasoning with literature support"
        }}
    }},
    "tissue_summary": "Brief summary of expected cell types in this tissue"
}}
"""

        logger.info(f"[Literature] annotate_clusters Phase 2: LLM matching {n_clusters} clusters")
        response = await self.chat_without_history(matching_prompt, temperature=0.2)

        # Parse the JSON response
        parsed = self._safe_parse_json(response)

        annotations = {}
        tissue_summary = ""
        if parsed and "annotations" in parsed:
            annotations = parsed["annotations"]
            tissue_summary = parsed.get("tissue_summary", "")
        else:
            logger.warning("[Literature] Failed to parse annotations JSON, returning raw response")
            annotations = {
                "error": "Failed to parse structured annotations",
                "raw_response": response[:2000]
            }

        logger.info(
            f"[Literature] annotate_clusters complete: "
            f"{len(annotations)} clusters annotated"
        )

        result = {
            "status": "completed",
            "annotations": annotations,
            "tissue_summary": tissue_summary,
            "literature_report": literature_report,
            "references": references,
            "n_clusters": n_clusters,
            "organism": organism,
            "tissue": tissue
        }
        # Evict oldest entries if cache exceeds limit
        if len(self._annotation_cache) >= self._ANNOTATION_CACHE_MAX:
            oldest_key = next(iter(self._annotation_cache))
            del self._annotation_cache[oldest_key]
        self._annotation_cache[cache_key] = result
        return result

    async def validate_pathway(
        self,
        pathway_name: str,
        genes: List[str],
        context: str = "",
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """Validate pathway enrichment with literature evidence."""
        topic = f"{pathway_name} pathway {' '.join(genes[:10])} disease mechanism"

        result = await self._do_research(topic, state)

        prompt = f"""
Based on the literature review, validate the biological significance of the following pathway enrichment:

Pathway name: {pathway_name}
Enriched genes: {', '.join(genes[:20])}
Research context: {context}

Literature information:
{result['report'][:8000]}

Please evaluate:
1. Known role of this pathway in disease/physiology
2. Relevance of enriched genes to the pathway
3. Biological significance of this finding

Reply in JSON format:
{{
    "pathway": "{pathway_name}",
    "known_role": "Known role of the pathway",
    "gene_relevance": "Gene relevance assessment",
    "biological_significance": "Biological significance",
    "confidence": "high/medium/low",
    "key_references": ["Key references"]
}}
"""

        response = await self.chat_without_history(prompt, temperature=0.3)

        try:
            parsed = self._safe_parse_json(response)
            if parsed and "pathway" in parsed:
                return parsed
            else:
                logger.warning(f"[Literature] No JSON found in validate_pathway for {pathway_name}")
        except Exception as e:
            logger.warning(f"[Literature] Failed to parse validate_pathway for {pathway_name}: {e}")

        return {
            "pathway": pathway_name,
            "confidence": "low"
        }

    async def search_mechanism(
        self,
        phenotype: str,
        genes: Optional[List[str]] = None,
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """Search for molecular mechanisms underlying a phenotype."""
        query_parts = [phenotype, "molecular mechanism"]
        if genes:
            query_parts.extend(genes[:5])

        topic = " ".join(query_parts)
        result = await self._do_research(topic, state)

        return {
            "phenotype": phenotype,
            "report": result.get("report", ""),
            "references": result.get("references", []),
            "findings": result.get("findings", [])
        }
