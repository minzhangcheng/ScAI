"""Finding extraction, reference selection, and report synthesis.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Dict, Any, List, Optional
import re
import logging


logger = logging.getLogger(__name__)
from ...models.schemas import (
    AnalysisState,
    LiteratureReport,
    PubMedEvidenceItem,
    LiteratureMechanismSignal,
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


class SynthesisMixin:
    """Finding extraction, reference selection, and report synthesis."""

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

    @staticmethod
    def _select_references(response, report_text: str):
        """Prefer structured citations from the Responses API; fall back to
        regex extraction over the report text."""
        citations = getattr(response, "citations", None)
        if citations:
            return citations
        return SynthesisMixin._extract_references_from_text(report_text)

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

    async def extract_mechanism_signals(
        self,
        report_text: str,
        references: List[Dict[str, str]],
        observed_context: str,
    ) -> Optional[List["LiteratureMechanismSignal"]]:
        """Generic, mechanism-agnostic extraction of biological mechanisms in the
        retrieved literature that co-occur with this study's observed data.
        Returns None on no content or any failure (non-critical path)."""
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
