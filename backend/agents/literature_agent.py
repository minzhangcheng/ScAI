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
from typing import Dict, Any, List, Optional, Set, Tuple
import json
import re
import asyncio
import logging
import xml.etree.ElementTree as ET

import httpx

from .base import BaseAgent

logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, LiteratureReport
)
from ..utils.llm_client import OpenAICompatibleClient, GoogleDeepResearchClient


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

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        config: Optional['AgentConfig'] = None,
        # Literature-specific parameters
        literature_mode: str = "deep_research",
        deep_research_model: str = "",
        google_api_key: str = "",
        pubmed_email: str = "",
        max_search_rounds: int = 5,
        proxy_url: Optional[str] = None,
    ):
        from .base import AgentConfig
        super().__init__(
            agent_type=AgentType.LITERATURE,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config or AgentConfig()
        )
        self.reports: List[LiteratureReport] = []

        # Mode configuration
        self.literature_mode = literature_mode
        self.deep_research_model = deep_research_model
        self.pubmed_email = pubmed_email
        self.max_search_rounds = max(1, min(20, max_search_rounds))
        self.proxy_url = proxy_url

        # Create GoogleDeepResearchClient only when needed (Gemini mode)
        self._deep_research: Optional[GoogleDeepResearchClient] = None
        if google_api_key:
            self._deep_research = GoogleDeepResearchClient(
                google_api_key, proxy_url=proxy_url
            )

    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process literature research instruction"""
        topic = kwargs.get("topic", instruction)
        use_deep_research = kwargs.get("use_deep_research", True)
        current_round_findings = kwargs.get("current_round_findings", None)

        logger.info(
            f"[Literature] Processing topic: {topic[:100]}... "
            f"(mode={self.literature_mode}, deep={use_deep_research})"
        )

        if use_deep_research:
            return await self._do_research(topic, state, current_round_findings=current_round_findings)
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
        current_round_findings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Route to configured research mode with fallback chain.

        Fallback: deep_research/gemini → pubmed → quick_search

        Args:
            topic: Research topic
            state: Optional analysis state for context
            pubmed_max_results: Override max articles per PubMed search round (default 20)
            current_round_findings: Findings from current round's other agents
        """
        mode = self.literature_mode

        try:
            if mode == "deep_research":
                return await self._deep_research_via_chat(topic, state, current_round_findings=current_round_findings)
            elif mode == "gemini_deep_research":
                return await self._multi_round_gemini(topic, state)
            elif mode == "pubmed_llm":
                return await self._multi_round_pubmed(
                    topic, state, pubmed_max_results=pubmed_max_results,
                    current_round_findings=current_round_findings
                )
            else:
                logger.warning(f"[Literature] Unknown mode: {mode}, using quick search")
                return await self._quick_search(topic, state)
        except Exception as e:
            logger.warning(f"[Literature] Mode {mode} failed: {e}")

            # Fallback: deep_research or gemini → try pubmed
            if mode in ("gemini_deep_research", "deep_research"):
                logger.info("[Literature] Falling back to PubMed + LLM mode")
                try:
                    return await self._multi_round_pubmed(
                        topic, state, pubmed_max_results=pubmed_max_results,
                        current_round_findings=current_round_findings
                    )
                except Exception as e2:
                    logger.warning(f"[Literature] PubMed fallback also failed: {e2}")

            # Final fallback: quick search (LLM knowledge only)
            logger.info("[Literature] Falling back to quick search (LLM knowledge only)")
            return await self._quick_search(topic, state)

    # ===================================================================
    # Mode 1: Gemini Deep Research (single call — DR handles rounds internally)
    # ===================================================================

    async def _multi_round_gemini(
        self,
        topic: str,
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """Single Gemini Deep Research call via Google Interactions API.

        Deep Research handles multi-round search/synthesis internally on Google's
        servers, so we only need one call per topic. Gap coverage (if any) is
        handled at the workflow level by falling back to PubMed+LLM.
        """
        if not self._deep_research:
            raise ValueError("Google API Key not configured for Gemini Deep Research")

        query = self._build_research_query(topic, state)
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
        current_round_findings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Deep research via OpenRouter chat API (OpenAI o4, Perplexity sonar, etc.).

        Makes a single long-running chat call with high max_tokens and timeout.
        The deep research model handles web search and synthesis internally.
        """
        if not self.deep_research_model:
            raise ValueError("deep_research_model not configured for deep_research mode")

        ctx = self._build_research_context(topic, state, current_round_findings=current_round_findings)

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

        prompt = f"""Conduct a comprehensive biomedical literature review on the following topic.

Research topic: {topic}

Experimental background:
{ctx["initial_instruction"][:500] if ctx["initial_instruction"] else "Not available."}

{extra_block}

Please provide:
1. A comprehensive review covering key findings, molecular mechanisms, and clinical relevance
2. Current research consensus and major unresolved questions
3. Implications for the experimental context described above

IMPORTANT - Citation format:
For each cited paper, provide a reference list at the end in this exact format:
[N] Author et al. (Year). Title. Journal. URL_or_DOI
Example: [1] Smith et al. (2023). Role of TGF-beta in liver fibrosis. Nature. https://doi.org/10.1038/xxx"""

        logger.info(
            f"[Literature] Deep Research via {self.deep_research_model}: "
            f"{topic[:200]}"
        )

        response = await self.llm_client.chat(
            model=self.deep_research_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=262144,
            timeout=1800.0,
        )

        report_text = response.content or ""
        if not report_text:
            raise ValueError("Deep research returned empty response")

        logger.info(
            f"[Literature] Deep Research completed: "
            f"{len(report_text)} chars"
        )

        references = self._extract_references_from_text(report_text)
        return await self._finalize_result(topic, report_text, references, state, 1)

    def _extract_references_from_text(self, text: str) -> List[Dict[str, str]]:
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
        current_round_findings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Multi-round PubMed search with LLM-guided query refinement."""
        all_articles: List[Dict[str, str]] = []
        seen_pmids: Set[str] = set()
        search_history: List[Dict[str, Any]] = []

        # Generate initial PubMed query from the topic
        query = await self._generate_pubmed_query(topic, state, current_round_findings)

        for round_num in range(1, self.max_search_rounds + 1):
            logger.info(
                f"[Literature] PubMed round {round_num}/{self.max_search_rounds}: "
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
                f"[Literature] PubMed round {round_num}: "
                f"{len(new_articles)} new articles, {len(all_articles)} total"
            )

            # Evaluate if more searching needed
            if round_num < self.max_search_rounds and all_articles:
                evaluation = await self._evaluate_pubmed_results(
                    topic, all_articles, search_history
                )
                if evaluation.get("sufficient", True):
                    logger.info(f"[Literature] PubMed sufficient after {round_num} rounds")
                    break
                query = evaluation.get("refined_query", query)

            # Rate limiting for NCBI E-utilities (3 req/s without API key)
            await asyncio.sleep(0.4)

        if not all_articles:
            logger.warning("[Literature] No PubMed articles found, falling back to quick search")
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
                title = article_elem.findtext(".//ArticleTitle", "")

                # Concatenate abstract sections
                abstract_parts = article_elem.findall(".//AbstractText")
                abstract = " ".join(
                    p.text for p in abstract_parts if p.text
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

        response = await self.chat(prompt, temperature=0.2)
        # Clean up the response - remove quotes, markdown, etc.
        query = response.strip().strip('"').strip("'").strip("`")
        # Remove markdown code block markers
        query = re.sub(r'^```\w*\n?', '', query)
        query = re.sub(r'\n?```$', '', query)
        return query.strip()

    # ===================================================================
    # LLM evaluation and synthesis helpers
    # ===================================================================

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

        response = await self.chat(prompt, temperature=0.3)
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
        # Include truncated abstracts for better relevance judgment
        articles_summary = "\n".join(
            f"- [{a.get('year', 'N/A')}] {a['title']}\n  Abstract: {a.get('abstract', 'N/A')[:200]}"
            for a in articles[:15]
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

Articles found ({len(articles)} total, showing first 15 with abstracts):
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

        response = await self.chat(prompt, temperature=0.3)
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

        return await self.chat(prompt, temperature=0.4)

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

        return await self.chat(prompt, temperature=0.4)

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

        response = await self.chat(prompt, temperature=0.3)

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
                finding_parts.append(f[:150])
            for f in rr.pathology_findings[:2]:
                finding_parts.append(f[:150])
        if finding_parts:
            context["previous_findings"] = "; ".join(finding_parts)

        if current_round_findings:
            context["current_round_findings"] = "; ".join(current_round_findings[:5])

        return context

    def _build_research_query(
        self,
        topic: str,
        state: Optional[AnalysisState] = None
    ) -> str:
        """Build a plain-text research query string (backward compatible)."""
        parts = [topic]
        if state and state.initial_instruction:
            parts.append(state.initial_instruction[:200])
        return " ".join(parts)

    async def _extract_findings(self, report: str) -> List[str]:
        """Extract key findings from a literature report."""
        prompt = f"""
Extract key findings from the following literature review (in list form):

{report[:10000]}

Please list 3-5 most important findings, one per line.
"""

        response = await self.chat(prompt, temperature=0.3)

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

        response = await self.chat(prompt, temperature=0.2)

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
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """
        Two-phase cluster annotation: one literature search + LLM matching.

        Phase 1: Single _do_research() call with tissue/organism-focused query
        Phase 2: LLM matches each cluster's markers against retrieved literature

        Args:
            cluster_markers: Dict mapping cluster_id (str) to list of top marker genes
                            e.g. {"0": ["Cd79a", "Cd79b", "Ms4a1"], "1": ["Cd3d", "Cd3e"]}
            organism: Organism name (e.g., "mouse", "human")
            tissue: Tissue/organ context (e.g., "kidney", "liver")
            state: Optional analysis state for context

        Returns:
            Dict with annotations per cluster, literature_report, references, status
        """
        n_clusters = len(cluster_markers)

        # ---- Phase 1: One literature search ----
        # Sample top genes across clusters for the search query
        all_genes: set = set()
        for genes in cluster_markers.values():
            all_genes.update(genes[:5])  # Top 5 per cluster
        gene_sample = ", ".join(list(all_genes)[:20])  # Cap at 20 genes

        tissue_part = f" {tissue}" if tissue else ""
        topic = (
            f"Cell types in {organism}{tissue_part} identified by single-cell RNA sequencing. "
            f"Key marker genes include: {gene_sample}. "
            f"Identify known cell type markers and signatures for{tissue_part} tissue."
        )

        logger.info(
            f"[Literature] annotate_clusters Phase 1: {n_clusters} clusters, "
            f"organism={organism}, tissue={tissue}"
        )

        # Use higher pubmed_max_results (50) for broader tissue coverage
        research_result = await self._do_research(
            topic, state, pubmed_max_results=50
        )

        literature_report = research_result.get("report", "")
        references = research_result.get("references", [])

        # ---- Phase 2: LLM matching ----
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
        response = await self.chat(matching_prompt, temperature=0.2)

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

        return {
            "status": "completed",
            "annotations": annotations,
            "tissue_summary": tissue_summary,
            "literature_report": literature_report,
            "references": references,
            "n_clusters": n_clusters,
            "organism": organism,
            "tissue": tissue
        }

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

        response = await self.chat(prompt, temperature=0.3)

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
