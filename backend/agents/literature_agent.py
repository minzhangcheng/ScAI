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
from ..utils.formatting import as_list

logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, LiteratureReport,
    DirectionSearchQuery, PubMedEvidenceItem, DirectionLiteratureEvidence,
    LiteratureMechanismSignal,
)
from ..utils.llm_client import OpenAICompatibleClient, GoogleDeepResearchClient, RETRYABLE_EXCEPTIONS, UpstreamExhaustedError
from openai import RateLimitError

# Re-exported so `backend.agents.literature_agent.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .literature.shared import (  # noqa: F401
    _FALLBACK_EXCEPTIONS,
    _MAX_MAIN_RETRIES,
    _RETRY_DELAY_BASE,
    plausibility_terms_for_tissue,
    _gene_case_variants,
    _sanitize_synonyms,
    status_from_judge,
    build_focused_query,
)
from .literature import (  # noqa: F401
    ClientsMixin,
    QueryBuildersMixin,
    PubMedMixin,
    DirectionNoveltyMixin,
    SynthesisMixin,
    MarkersMixin,
    ResearchMixin,
)
class LiteratureAgent(
    ClientsMixin,
    QueryBuildersMixin,
    PubMedMixin,
    DirectionNoveltyMixin,
    SynthesisMixin,
    MarkersMixin,
    ResearchMixin,
    BaseAgent,
):
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
    # budget meant for the primary evaluation. They still have to be large
    # enough for one triage call to finish: at 10s/15s the fallback timed out
    # before any model could answer, which turned "cheap" into "guaranteed
    # unknown".
    MIN_ATTEMPT_QUERY_TIMEOUT = 60.0
    MIN_ATTEMPT_DIRECTION_TIMEOUT = 150.0

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
        literature_mode: str = "pubmed_llm",
        deep_research_model: str = "o4-mini-deep-research",
        deep_research_llm_base_url: str = "",
        deep_research_api_key: str = "",
        google_api_key: str = "",
        pubmed_email: str = "",
        max_search_rounds: int = 20,
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
        self._client_close_task_ids: Set[int] = set()
        self._retired_runtime_clients: Dict[int, Any] = {}
        self._runtime_client_use_counts: Dict[int, int] = {}
        self._runtime_client_cleanup_lock = asyncio.Lock()
        self._runtime_clients_closing = False
        self._runtime_clients_closed = False












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


    # ===================================================================
    # Mode 1: Gemini Deep Research (single call — DR handles rounds internally)
    # ===================================================================


    # ===================================================================
    # Mode 2: Deep Research via OpenRouter (single call)
    # ===================================================================




    # ===================================================================
    # Mode 3: PubMed + LLM (multi-round)
    # ===================================================================



    # ===================================================================
    # PubMed E-utilities integration
    # ===================================================================




    # ===================================================================
    # LLM evaluation and synthesis helpers
    # ===================================================================






    # ===================================================================
    # Result finalization helper
    # ===================================================================



    # ===================================================================
    # Quick search fallback (LLM knowledge only)
    # ===================================================================


    # ===================================================================
    # Shared helpers
    # ===================================================================



    # ===================================================================
    # Specialized utility methods (used by bioinformatics agent)
    # ===================================================================



    _STATUS_RANK = {"strong": 3, "limited": 2, "contradictory": 1, "exhausted": 0}
    _EXHAUSTION_REASON_RANK = {
        "search_failed": 3,
        "judge_failed": 3,
        "no_relevant_evidence": 2,
        "no_pubmed_results": 1,
        "empty_query": 0,
        None: 0,
    }






    # ===================================================================
    # Direction novelty literature triage (Task 2)
    # ===================================================================




































