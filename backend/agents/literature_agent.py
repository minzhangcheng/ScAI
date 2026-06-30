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
import time
import xml.etree.ElementTree as ET

import httpx
from openai import APIConnectionError as OpenAIConnectionError

from .base import BaseAgent
from ..utils.settings_provider import AgentSettingsProvider

logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, LiteratureReport,
    DirectionSearchQuery, PubMedEvidenceItem, DirectionLiteratureEvidence,
    LiteratureMechanismSignal,
)
from ..utils.llm_client import OpenAICompatibleClient, GoogleDeepResearchClient, RETRYABLE_EXCEPTIONS, UpstreamExhaustedError
from openai import RateLimitError

_FALLBACK_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (RateLimitError,)

# 主模型重试配置（用于 _chat_with_fallback）
_MAX_MAIN_RETRIES = 2
_RETRY_DELAY_BASE = 2.0  # 秒，与 llm_client 保持一致

# ===================================================================
# Focused query pure function helpers (Batch 1)
# ===================================================================

# 污染 plausibility 检索的保守组织结构/非实质成分术语(通用,适用于所有组织)。
# 未在此处列出的组织特定词(如肝 portal tract)由 plausibility_terms_for_tissue() 按 tissue 补充。
_GENERIC_CONTAMINATION_PLAUSIBILITY_TERMS: list = [
    '"nerve fiber"[tiab]',
    'innervation[tiab]',
    '"peripheral nerve"[tiab]',
    'Schwann[tiab]',
    'perivascular[tiab]',
]

# 组织特定的 plausibility terms,仅当 tissue 名称含对应 key 时加入。
_TISSUE_CONTAMINATION_PLAUSIBILITY_TERMS: Dict[str, list] = {
    "liver": ['"portal tract"[tiab]', '"portal triad"[tiab]', '"Glisson capsule"[tiab]'],
    "hepatic": ['"portal tract"[tiab]', '"portal triad"[tiab]', '"Glisson capsule"[tiab]'],
}

# 旧常量名保留为别名,避免直接从该文件 import 它的外部代码报错。
_DEFAULT_CONTAMINATION_PLAUSIBILITY_TERMS: list = _GENERIC_CONTAMINATION_PLAUSIBILITY_TERMS


def plausibility_terms_for_tissue(tissue: str) -> list:
    """Return generic + tissue-specific terms for suspected-contamination plausibility queries.

    Liver-specific anatomy terms (portal tract, portal triad, Glisson capsule) only appear
    when the tissue string contains "liver" or "hepatic". Unknown/default tissues receive
    only generic nerve/perivascular terms.
    """
    normalized = (tissue or "").strip().lower()
    terms = list(_GENERIC_CONTAMINATION_PLAUSIBILITY_TERMS)
    for key, extra_terms in _TISSUE_CONTAMINATION_PLAUSIBILITY_TERMS.items():
        if key in normalized:
            terms.extend(extra_terms)
            break
    return terms


def _gene_case_variants(gene: str) -> list:
    """跨物种 ortholog 符号双拼: 小鼠式(Pkhd1) + 人类式(PKHD1)。已全大写则只返回一个。"""
    gene = (gene or "").strip()
    if not gene:
        return []
    upper = gene.upper()
    return [gene] if gene == upper else [gene, upper]


def _sanitize_synonyms(synonyms) -> list:
    """去重、去空、长度<=60、移除破坏 PubMed query 的危险字符。"""
    seen, out = set(), []
    for s in synonyms or []:
        s = re.sub(r'["\[\]()]', "", str(s)).strip()
        if not s or len(s) > 60 or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def status_from_judge(judge: dict) -> str:
    """judge 计数 → 四态。优先 contradictory,其次 strong/limited/exhausted。"""
    strong = int(judge.get("strong", 0))
    moderate = int(judge.get("moderate", 0))
    if judge.get("contradictory"):
        return "contradictory"
    if strong >= 2:
        return "strong"
    if strong >= 1 and moderate >= 1:
        return "limited"
    return "exhausted"


def build_focused_query(item: dict, organism: str, tissue: str, tier: int, bucket: str,
                        plausibility_terms: list = None) -> str:
    """构造 focused PubMed query;无可用检索词时返回空串(调用方应跳过)。

    rare_unresolved: 候选名+synonyms+markers 双拼;tier1 加 tissue,tier2 去 tissue。
    suspected_contamination: 仅 markers 双拼(不用可疑候选名);tier1=discovery(无 tissue),
                             tier2=plausibility(加 tissue + 可选 plausibility_terms)。
    """
    species = (organism or "").strip()
    or_terms = []
    for m in item.get("top_markers", [])[:3]:
        for v in _gene_case_variants(m):
            or_terms.append(f'"{v}"[tiab]')

    if bucket == "rare_unresolved":
        cand = item.get("candidate_celltype")
        if cand:
            or_terms.insert(0, f'"{cand}"[tiab]')
        for syn in _sanitize_synonyms(item.get("synonyms")):
            or_terms.append(f'"{syn}"[tiab]')

    if bucket == "suspected_contamination" and not or_terms:
        return ""

    if bucket == "suspected_contamination" and tier == 2:
        terms = plausibility_terms if plausibility_terms is not None else plausibility_terms_for_tissue(tissue)
        for t in terms:
            or_terms.append(t)

    if not or_terms:
        return ""

    parts = []
    if species:
        parts.append(species)
    parts.append("(" + " OR ".join(or_terms) + ")")

    add_tissue = (bucket == "rare_unresolved" and tier == 1) or \
                 (bucket == "suspected_contamination" and tier == 2)
    if add_tissue and tissue:
        parts.append(tissue)
    return " AND ".join(parts)


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

    # Upper bounds on the per-query / per-direction budget for the cheap
    # minimum-attempt fallback path, so a slow fallback can't consume the full
    # budget meant for the primary evaluation.
    MIN_ATTEMPT_QUERY_TIMEOUT = 10.0
    MIN_ATTEMPT_DIRECTION_TIMEOUT = 15.0

    # Citation guidance for the deep_research-LLM prompt. Deliberately format-agnostic:
    # _extract_references_from_text accepts BOTH end-of-text "[N] Title URL" lists and
    # inline "[text](URL)" markdown links, so the footer no longer mandates one rigid
    # format. PI-authored briefs never include citation-format directives.
    _DEEP_RESEARCH_CITATION_FOOTER = (
        "IMPORTANT - Citations:\n"
        "Support each key claim with a clickable source. Cite either inline as a markdown "
        "link [text](URL), or as a numbered reference list at the end in the form "
        "[N] Title. URL_or_DOI (or use both). Always include a title and a DOI/URL where available."
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
        max_search_rounds: int = 10,
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
        self.deep_research_llm_base_url = deep_research_llm_base_url
        self.deep_research_api_key = deep_research_api_key
        self.pubmed_email = pubmed_email
        self.max_search_rounds = max(1, min(50, max_search_rounds))
        self.proxy_url = proxy_url

        # Separate LLM client for deep research mode (optional)
        self._deep_research_llm_client: Optional[OpenAICompatibleClient] = None
        if self.deep_research_llm_base_url and self.deep_research_api_key:
            self._deep_research_llm_client = self._make_deep_research_client()

        # Create GoogleDeepResearchClient only when needed (Gemini mode)
        self._google_api_key = google_api_key
        self._deep_research: Optional[GoogleDeepResearchClient] = None
        if google_api_key:
            self._deep_research = self._make_google_client()

        # Cache for annotate_clusters results to avoid redundant Deep Research calls
        self._annotation_cache: Dict[str, Any] = {}
        self._ANNOTATION_CACHE_MAX = 100
        self._client_close_tasks: Set[asyncio.Task] = set()

    @property
    def is_deep_research_available(self) -> bool:
        """Check if deep research mode is available and properly configured."""
        if self.literature_mode not in ("gemini_deep_research", "deep_research"):
            return False
        return self._deep_research is not None or bool(self.deep_research_model)

    def _make_deep_research_client(self) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient(
            api_key=self.deep_research_api_key,
            base_url=self.deep_research_llm_base_url,
            proxy_url=self.proxy_url,
            session_id=self.session_id,
            agent_name=self.agent_type.value,
        )
        apply_retry_callback = getattr(self._provider, "apply_retry_callback", None)
        if apply_retry_callback:
            apply_retry_callback(client)
        return client

    def _make_google_client(self) -> GoogleDeepResearchClient:
        return GoogleDeepResearchClient(
            self._google_api_key,
            proxy_url=self.proxy_url,
        )

    def _deep_research_client_signature(self) -> Tuple[str, str]:
        return (
            self.deep_research_llm_base_url or "",
            self.deep_research_api_key or "",
        )

    def _close_stale_client(self, client) -> None:
        if client is None or not hasattr(client, "close"):
            return

        async def _close():
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"[Literature] Failed to close stale runtime client: {e}")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_close())
        else:
            task = loop.create_task(_close())
            self._client_close_tasks.add(task)
            task.add_done_callback(self._client_close_tasks.discard)

    async def close_runtime_clients(self) -> None:
        """Close literature-owned clients that are not owned by AgentSettingsProvider."""
        clients = [self._deep_research_llm_client, self._deep_research]
        self._deep_research_llm_client = None
        self._deep_research = None
        if self._client_close_tasks:
            pending_tasks = list(self._client_close_tasks)
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            self._client_close_tasks.clear()
        for client in clients:
            if client is None or not hasattr(client, "close"):
                continue
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"[Literature] Failed to close runtime client: {e}")

    def _sync_literature_config_from_provider(self) -> None:
        lit_config = self._provider.get_literature_config() or {}

        new_mode = lit_config.get("mode", self.literature_mode)
        new_model = lit_config.get("deep_research_model", self.deep_research_model)
        new_base_url = lit_config.get("deep_research_llm_base_url", self.deep_research_llm_base_url)
        new_api_key = lit_config.get("deep_research_api_key", self.deep_research_api_key)
        new_google_api_key = lit_config.get("google_api_key", self._google_api_key)
        new_pubmed_email = lit_config.get("pubmed_email", self.pubmed_email)
        new_max_rounds = lit_config.get("max_search_rounds", self.max_search_rounds)

        old_client_signature = self._deep_research_client_signature()

        self.literature_mode = new_mode
        self.deep_research_model = new_model
        self.deep_research_llm_base_url = new_base_url
        self.deep_research_api_key = new_api_key
        self.pubmed_email = new_pubmed_email
        self.max_search_rounds = max(1, min(50, int(new_max_rounds or 10)))

        new_client_signature = self._deep_research_client_signature()
        if new_client_signature != old_client_signature:
            old_client = self._deep_research_llm_client
            self._deep_research_llm_client = None
            self._close_stale_client(old_client)
            if self.deep_research_llm_base_url and self.deep_research_api_key:
                self._deep_research_llm_client = self._make_deep_research_client()

        if new_google_api_key != self._google_api_key:
            old_client = self._deep_research
            self._deep_research = None
            self._close_stale_client(old_client)
            self._google_api_key = new_google_api_key
            if new_google_api_key:
                self._deep_research = self._make_google_client()

    def on_runtime_config_changed(
        self,
        previous_revision: Optional[int],
        current_revision: int,
    ) -> None:
        self._sync_literature_config_from_provider()
        self._annotation_cache.clear()

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
        self.refresh_runtime_config_if_needed()
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
        except UpstreamExhaustedError:
            # The LLM upstream is genuinely exhausted (all retries failed).
            # Do NOT silently degrade to quick_search — every fallback mode
            # also needs the same LLM, so degrading just yields a low-quality
            # result while hiding the failure. Propagate so the workflow's
            # LLM-failure HITL gate lets the user choose retry/skip/stop.
            raise
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
                except UpstreamExhaustedError:
                    raise
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
        self.refresh_runtime_config_if_needed()
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
                    tools=[{"type": "web_search_preview"}],
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

        Collects two structured forms — end-of-text "[N] Title URL" entries and
        inline markdown "[text](URL)" links — then falls back to bare URLs only when
        the structured forms yield fewer than 3 references. pmc text-fragment suffixes
        ("#:~:text=...") are stripped before dedup.
        """
        refs: List[Dict[str, str]] = []
        seen: set = set()

        def _norm(url: str) -> str:
            url = url.rstrip('.,;')
            return url.split('#:~:text=', 1)[0]

        def _add(title: str, url: str) -> None:
            url = _norm(url)
            if url and url not in seen:
                seen.add(url)
                refs.append({"title": title.strip(' .-,'), "url": url})

        # 1. End-of-text numbered list: [N] Title... URL
        for m in re.finditer(r'\[(\d+)\]\s*(.{5,300}?)\s*(https?://[^\s\)\]]+)', text):
            _add(m.group(2), m.group(3))

        # 2. Inline markdown links: [text](URL)
        #    (?<!!) excludes image markdown ![alt](url) — image URLs aren't references
        for m in re.finditer(r'(?<!!)\[([^\]]{1,200}?)\]\((https?://[^\s\)]+)\)', text):
            _add(m.group(1), m.group(2))

        # 3. Bare URL fallback — only when structured extraction is insufficient
        if len(refs) < 3:
            for m in re.finditer(r'https?://[^\s\)\]]+', text):
                _add("", m.group(0))

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
        zero_streak = 0

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

            zero_streak = zero_streak + 1 if not new_articles else 0

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
            if zero_streak >= 3:
                logger.info(f"[Literature] {label} stop: 3 consecutive zero-new rounds")
                break

            if round_num < self.max_search_rounds and all_articles:
                if pi_guided:
                    try:
                        evaluation = await evaluate_fn(all_articles, search_history)
                    except Exception as e:
                        logger.warning(f"[Literature] PI evaluation failed in round {round_num}: {e}, stopping")
                        evaluation = {"sufficient": True}
                    latest_triage = evaluation
                    # 确定性语义终止: 全部簇已归桶 且 triage 非低置信
                    if evaluation.get("triage_complete") and not evaluation.get("triage_low_confidence"):
                        logger.info(f"[Literature] {label} semantic termination: all clusters triaged")
                        break
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

        # --- Focused search: run after main loop, before synthesis ---
        focused_evidence: Dict[str, Any] = {}
        focused_articles_for_synthesis: List[Dict[str, Any]] = []
        if pi_guided and latest_triage:
            for bucket in ("rare_unresolved", "suspected_contamination"):
                items = latest_triage.get(bucket, [])
                if not items:
                    continue
                fe = await self._focused_rare_search(items, organism, tissue, bucket=bucket)
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

        # Prioritize focused evidence for synthesis because _synthesize_pubmed_results
        # intentionally truncates to the first 20 articles.
        synthesis_articles: List[Dict[str, Any]] = self._dedup_by_pmid(
            focused_articles_for_synthesis + all_articles, lambda a: a.get("pmid"),
            drop_empty=True)

        report_text = await self._synthesize_pubmed_results(topic, synthesis_articles)

        references = [
            {"title": a["title"], "url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"}
            for a in all_articles
        ]

        final = await self._finalize_result(topic, report_text, references, state, len(search_history))
        final["focused_evidence"] = focused_evidence
        final["triage"] = latest_triage
        return final

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

    async def extract_mechanism_signals(
        self,
        report_text: str,
        references: List[Dict[str, str]],
        observed_context: str,
    ) -> Optional[List["LiteratureMechanismSignal"]]:
        """Generic, mechanism-agnostic extraction of biological mechanisms in the
        retrieved literature that co-occur with this study's observed data.
        Returns None on no content or any failure (non-critical path)."""
        from ..models.schemas import LiteratureMechanismSignal

        if not (report_text and report_text.strip()) and not references:
            return None

        ref_lines = []
        for r in references[:40]:
            title = r.get("title", "") if isinstance(r, dict) else ""
            pmid = r.get("pmid", "") if isinstance(r, dict) else ""
            url = r.get("url", "") if isinstance(r, dict) else ""
            ref_lines.append(f"- {title} (PMID: {pmid or 'N/A'}; {url})")
        refs_block = "\n".join(ref_lines) if ref_lines else "None"

        prompt = f"""From the retrieved literature below, extract biological mechanisms that
co-occur with the experimental observations in this study's data. This is a generic
extraction — do NOT limit yourself to any particular mechanism, pathway, or gene.

## This Study's Observed Data
{observed_context or "None"}

## Retrieved Literature Report
{(report_text or "")[:24000]}

## References
{refs_block}

For EACH mechanism that the literature associates with this study's observations, provide:
- The mechanism name AS STATED in the literature (do not invent)
- Supporting PMIDs when available; use an empty list if none
- Whether the literature describes experimental, correlational, review, or hypothetical evidence
- Relevance to this study's specific experimental context (1-2 sentences)
- Which observed data points this mechanism connects to
- Caveats/limitations the literature itself notes

Return JSON only:
{{
    "mechanism_signals": [{{
        "mechanism": "...",
        "supporting_pmids": ["..."],
        "evidence_type": "experimental|correlational|review|hypothesis|unknown",
        "relevance_to_current_study": "...",
        "observed_data_links": ["..."],
        "limitations": ["..."]
    }}]
}}
If no mechanisms clearly co-occur with the observed data, return {{"mechanism_signals": []}}.
"""
        try:
            # Use chat_without_history (same pattern as _extract_findings): this is an
            # independent structured extraction that must not pollute or depend on the
            # literature agent's message history.
            response = await self.chat_without_history(prompt, temperature=0.3)
        except Exception as e:
            logger.warning(f"[Literature] mechanism extraction LLM call failed: {e}")
            return None

        data = self._safe_parse_json(response, default=None)
        if not isinstance(data, dict):
            return None
        out = []
        for entry in (data.get("mechanism_signals") or []):
            try:
                out.append(LiteratureMechanismSignal(**entry))
            except Exception:
                continue
        return out

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
        self.refresh_runtime_config_if_needed()
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
        self.refresh_runtime_config_if_needed()
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
        gene_sample = ", ".join(list(all_genes)[:40])  # Cap at 40 genes (fallback topic only)

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
                organism=organism, tissue=tissue,
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
                "tissue": tissue,
                "focused_evidence": research_result.get("focused_evidence", {}),
                "triage": research_result.get("triage", {}),
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

    _STATUS_RANK = {"strong": 3, "limited": 2, "contradictory": 1, "exhausted": 0}
    _EXHAUSTION_REASON_RANK = {
        "search_failed": 3,
        "judge_failed": 3,
        "no_relevant_evidence": 2,
        "no_pubmed_results": 1,
        "empty_query": 0,
        None: 0,
    }

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

    async def _focused_rare_search(self, items, organism: str, tissue: str,
                                   bucket: str = "rare_unresolved") -> dict:
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
            try:
                arts = await self._pubmed_search(q, max_results=10)
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

            await asyncio.sleep(0.4)

            try:
                judged = await self._judge_focused_relevance(item, tissue, arts, bucket)
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

    async def validate_pathway(
        self,
        pathway_name: str,
        genes: List[str],
        context: str = "",
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """Validate pathway enrichment with literature evidence."""
        self.refresh_runtime_config_if_needed()
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
        self.refresh_runtime_config_if_needed()
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

    # ===================================================================
    # Direction novelty literature triage (Task 2)
    # ===================================================================

    def _candidate_terms(self, candidate: Dict[str, Any]) -> Dict[str, List[str]]:
        return {
            "genes": [str(x).strip() for x in candidate.get("key_genes", []) if str(x).strip()],
            "cells": [str(x).strip() for x in candidate.get("key_cell_types", []) if str(x).strip()],
            "pathways": [str(x).strip() for x in candidate.get("key_pathways", []) if str(x).strip()],
            "methods": [str(x).strip() for x in candidate.get("suggested_methods", []) if str(x).strip()],
            "organism": [str(candidate.get("organism") or "").strip()],
            "tissue": [str(candidate.get("tissue") or "").strip()],
            "context": [str(candidate.get("experimental_context") or "").strip()],
        }

    def _format_term_list(self, terms: List[str], field: str = "tiab") -> str:
        quoted = [f'"{t}"[{field}]' for t in terms if t]
        if not quoted:
            return ""
        if len(quoted) == 1:
            return quoted[0]
        return "(" + " OR ".join(quoted) + ")"

    def _build_direct_prior_work(self, terms: Dict[str, List[str]], deep: bool) -> Optional[str]:
        # Prefer genes, then fall back to cells, then pathways.
        core = (
            self._format_term_list(terms["genes"])
            or self._format_term_list(terms["cells"])
            or self._format_term_list(terms["pathways"])
        )
        if not core:
            return None
        parts = [core]

        org = terms["organism"][0]
        tis = terms["tissue"][0]
        if org:
            parts.append(f'"{org}"[all]')
        if tis and not deep:
            parts.append(f'"{tis}"[tiab]')

        return " AND ".join(parts)

    def _build_analogous_mechanism(self, terms: Dict[str, List[str]]) -> Optional[str]:
        parts = []
        pathway_part = self._format_term_list(terms["pathways"])
        if pathway_part:
            parts.append(pathway_part)
        if not pathway_part:
            gene_part = self._format_term_list(terms["genes"])
            if gene_part:
                parts.append(gene_part)
        if not parts:
            return None

        org = terms["organism"][0]
        if org:
            parts.append(f'"{org}"[all]')
        tis = terms["tissue"][0]
        if tis:
            parts.append(f'"{tis}"[tiab]')

        return " AND ".join(parts)

    def _build_contradictory_expectedness(self, terms: Dict[str, List[str]]) -> Optional[str]:
        gene_part = self._format_term_list(terms["genes"])
        if not gene_part:
            pathway_part = self._format_term_list(terms["pathways"])
            if pathway_part:
                gene_part = pathway_part
        if not gene_part:
            return None

        return f"({gene_part}) AND (adverse[tiab] OR negative[tiab] OR inhibition[tiab] OR antagonist[tiab] OR detrimental[tiab])"

    def _build_active_challenge_query(self, terms: Dict[str, List[str]]) -> Optional[str]:
        core = (
            self._format_term_list(terms["genes"])
            or self._format_term_list(terms["pathways"])
            or self._format_term_list(terms["cells"])
        )
        if not core:
            return None
        challenge_terms = (
            "contradict*[tiab] OR negative[tiab] OR absence[tiab] OR "
            "downregulat*[tiab] OR alternative[tiab] OR contrary[tiab] OR "
            "not associated[tiab]"
        )
        return f"({core}) AND ({challenge_terms})"

    def _build_prior_art_query(self, terms: Dict[str, List[str]]) -> Optional[str]:
        core_parts = []
        gene_part = self._format_term_list(terms["genes"])
        pathway_part = self._format_term_list(terms["pathways"])
        cell_part = self._format_term_list(terms["cells"])
        if gene_part:
            core_parts.append(gene_part)
        if pathway_part:
            core_parts.append(pathway_part)
        if cell_part:
            core_parts.append(cell_part)
        if not core_parts:
            return None
        parts = [" AND ".join(core_parts[:2])]
        org = terms["organism"][0]
        tis = terms["tissue"][0]
        if org:
            parts.append(f'"{org}"[all]')
        if tis:
            parts.append(f'"{tis}"[tiab]')
        return " AND ".join(parts)

    def _build_tier3_broad_query(self, terms: Dict[str, List[str]], challenge: bool) -> str:
        tokens: List[str] = []
        for key in ("genes", "pathways", "cells"):
            for term in terms.get(key, []):
                clean = str(term).strip()
                if clean and clean not in tokens:
                    tokens.append(clean)
        if not tokens:
            context = terms.get("context", [""])[0]
            tokens = [w for w in str(context).split()[:6] if w]
        query = " ".join(tokens[:8])
        if challenge:
            query = f"{query} alternative mechanism negative evidence contradiction contrary findings"
        return query.strip()

    def _build_methodological_precedent(self, terms: Dict[str, List[str]]) -> Optional[str]:
        method_part = self._format_term_list(terms["methods"])
        if not method_part:
            return None

        parts = [method_part]
        org = terms["organism"][0]
        if org:
            parts.append(f'"{org}"[all]')
        tis = terms["tissue"][0]
        if tis:
            parts.append(f'"{tis}"[tiab]')

        return " AND ".join(parts)

    def _build_gene_model_relaxed(self, terms: Dict[str, List[str]]) -> Optional[str]:
        gene_part = self._format_term_list(terms["genes"])
        if not gene_part:
            return None
        return gene_part

    def _build_cell_type_focused(self, terms: Dict[str, List[str]]) -> Optional[str]:
        cell_part = self._format_term_list(terms["cells"])
        if not cell_part:
            gene_part = self._format_term_list(terms["genes"])
            if not gene_part:
                return None
            cell_part = gene_part
        parts = [cell_part]
        tis = terms["tissue"][0]
        if tis:
            parts.append(f'"{tis}"[tiab]')
        return " AND ".join(parts)

    def _build_pathway_focused(self, terms: Dict[str, List[str]]) -> Optional[str]:
        pathway_part = self._format_term_list(terms["pathways"])
        gene_part = self._format_term_list(terms["genes"])
        if not pathway_part and not gene_part:
            return None
        parts = []
        if pathway_part:
            parts.append(pathway_part)
        if gene_part:
            parts.append(gene_part)
        return " AND ".join(parts) if len(parts) > 1 else parts[0]

    def _build_cross_species_or_tissue(self, terms: Dict[str, List[str]]) -> Optional[str]:
        # Core mechanism term: prefer genes, then pathways, then cell types.
        core = (self._format_term_list(terms["genes"])
                or self._format_term_list(terms["pathways"])
                or self._format_term_list(terms["cells"]))
        if not core:
            return None
        parts = [core]
        org = terms["organism"][0]
        if org:
            # Cross-species: same mechanism studied in other organisms.
            # Target tissue is intentionally omitted so cross-tissue evidence is not excluded.
            parts.append(f'NOT "{org}"[all]')
        return " AND ".join(parts) if len(parts) > 1 else parts[0]

    def _build_degraded_fallback(self, terms: Dict[str, List[str]]) -> str:
        parts = []
        org = terms["organism"][0]
        tis = terms["tissue"][0]
        if org:
            parts.append(f'"{org}"[all]')
        if tis:
            parts.append(f'"{tis}"[tiab]')
        combined = []
        for key in ("pathways", "cells", "methods", "genes"):
            for t in terms.get(key, []):
                if t:
                    combined.append(f'"{t}"[tiab]')
        if combined:
            parts.append("(" + " OR ".join(combined) + ")")
        if not parts:
            return ""
        return " AND ".join(parts)

    def _meaningful_term_count(self, terms: Dict[str, List[str]]) -> int:
        count = 0
        for key in ("genes", "cells", "pathways", "methods"):
            if any(t for t in terms.get(key, []) if t):
                count += 1
        return count

    def _direction_queries(self, candidate: Dict[str, Any], deep: bool = False) -> List[DirectionSearchQuery]:
        terms = self._candidate_terms(candidate)
        queries: List[DirectionSearchQuery] = []
        meaningful = self._meaningful_term_count(terms)

        standard_builders = [
            # Keep direct_prior_work because negative-validation axes rely on this category name.
            ("direct_prior_work", lambda t: self._build_direct_prior_work(t, deep=False)),
            ("analogous_mechanism", self._build_analogous_mechanism),
            ("challenge_contradiction", self._build_active_challenge_query),
            ("prior_art", self._build_prior_art_query),
            ("contradictory_expectedness", self._build_contradictory_expectedness),
            ("methodological_precedent", self._build_methodological_precedent),
        ]
        deep_builders = [
            ("gene_model_relaxed", self._build_gene_model_relaxed),
            ("cell_type_focused", self._build_cell_type_focused),
            ("pathway_focused", self._build_pathway_focused),
            ("cross_species_or_tissue", self._build_cross_species_or_tissue),
        ]

        builders = deep_builders if deep else standard_builders
        # The degraded fallback depends only on `terms`, so every builder would
        # produce the same query string. Compute it once and emit a single
        # degraded query to avoid running the same PubMed search N times.
        degraded_fallback = self._build_degraded_fallback(terms) if meaningful < 2 else None
        emitted_degraded = False
        for category, builder in builders:
            query_str = builder(terms)
            if not query_str:
                continue
            if meaningful < 2 and degraded_fallback:
                if not emitted_degraded:
                    queries.append(DirectionSearchQuery(
                        category=category,
                        query=degraded_fallback,
                        degraded=True,
                        notes=f"Degraded: fewer than 2 meaningful axes (had {meaningful})"
                    ))
                    emitted_degraded = True
                continue
            queries.append(DirectionSearchQuery(category=category, query=query_str))

        return queries

    @staticmethod
    def _dedup_by_pmid(items, pmid_of, drop_empty=False):
        """Order-preserving dedup by PMID. `pmid_of` extracts the pmid from an item
        (attr access for evidence objects, dict access for raw articles).
        `drop_empty=True` discards items without a pmid; otherwise they are kept
        un-deduplicated."""
        seen: set = set()
        out = []
        for it in items:
            key = str(pmid_of(it) or "")
            if not key:
                if not drop_empty:
                    out.append(it)
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    def _dedup_evidence(self, items: List["PubMedEvidenceItem"]) -> List["PubMedEvidenceItem"]:
        return self._dedup_by_pmid(items, lambda e: getattr(e, "pmid", ""))

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

    def _article_matches_candidate(self, article: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        genes = [str(g).lower() for g in candidate.get("key_genes", []) if str(g).strip()]
        pathways = [str(p).lower() for p in candidate.get("key_pathways", []) if str(p).strip()]
        cells = [str(c).lower() for c in candidate.get("key_cell_types", []) if str(c).strip()]
        title = (article.get("title") or "").lower()
        abstract = (article.get("abstract") or "").lower()
        text = f"{title} {abstract}"
        return any(needle in text for needle in genes + pathways + cells)

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

    def _minimum_attempt_queries(self, candidate: Dict[str, Any]) -> List[DirectionSearchQuery]:
        terms = self._candidate_terms(candidate)
        support = self._build_direct_prior_work(terms, deep=False)
        challenge = self._build_active_challenge_query(terms)
        queries: List[DirectionSearchQuery] = []
        if support:
            queries.append(DirectionSearchQuery(category="direct_prior_work", query=support))
        if challenge:
            queries.append(DirectionSearchQuery(category="challenge_contradiction", query=challenge))
        # Keep both support and challenge so the minimum-attempt path can still
        # surface contradictory evidence (parity with the tier3 fallback below,
        # which also runs both); the shared direction deadline bounds total cost.
        if queries:
            return queries[:2]
        broad_support = self._build_tier3_broad_query(terms, challenge=False)
        broad_challenge = self._build_tier3_broad_query(terms, challenge=True)
        fallback = []
        if broad_support:
            fallback.append(DirectionSearchQuery(category="tier3_support", query=broad_support, degraded=True))
        if broad_challenge:
            fallback.append(DirectionSearchQuery(category="tier3_challenge", query=broad_challenge, degraded=True))
        if fallback:
            return fallback
        return []

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

    def _support_level(self, direct_count: int, analogous_count: int) -> str:
        if direct_count >= 2:
            return "strong"
        if direct_count >= 1 or analogous_count >= 2:
            return "moderate"
        if analogous_count >= 1:
            return "weak"
        return "none"

    def _contradiction_level(self, contradictory_count: int) -> str:
        if contradictory_count >= 2:
            return "strong"
        if contradictory_count == 1:
            return "moderate"
        return "none"

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
