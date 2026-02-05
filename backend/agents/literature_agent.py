"""
Literature Agent - Uses Google Deep Research for literature support
"""
from typing import Dict, Any, List, Optional
import json
import re
import asyncio
import logging

from .base import BaseAgent

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, LiteratureReport
)
from ..utils.llm_client import OpenAICompatibleClient, GoogleDeepResearchClient


class LiteratureAgent(BaseAgent):
    """
    Literature Agent
    
    Uses Google Gemini Deep Research for comprehensive literature review.
    
    Responsibilities:
    - Search and synthesize biomedical literature
    - Provide evidence for cell type annotations
    - Support pathway analysis interpretation
    - Identify relevant molecular mechanisms
    """
    
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        deep_research_client: GoogleDeepResearchClient,
        model: str,
        system_prompt: str,
        config: Optional['AgentConfig'] = None
    ):
        from .base import AgentConfig
        super().__init__(
            agent_type=AgentType.LITERATURE,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config or AgentConfig()
        )
        self.deep_research = deep_research_client
        self.reports: List[LiteratureReport] = []
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process literature research instruction"""

        topic = kwargs.get("topic", instruction)
        use_deep_research = kwargs.get("use_deep_research", True)

        logger.info(f"[Literature] Processing topic: {topic[:100]}... (deep_research={use_deep_research})")

        if use_deep_research:
            return await self._deep_research(topic, state)
        else:
            return await self._quick_search(topic, state)
    
    async def _deep_research(
        self,
        topic: str,
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """
        Perform deep research using Google Deep Research Agent

        Args:
            topic: Research topic
            state: Optional analysis state for context. If None, no state updates occur.
        """
        try:
            # Build research query
            query = self._build_research_query(topic, state)

            # Start research
            result = await self.deep_research.research(query)

            # Process and store report
            report = LiteratureReport(
                topic=topic,
                report=result.get("report", ""),
                references=[{
                    "title": src.get("title", ""),
                    "url": src.get("uri", "")
                } for src in result.get("sources", [])]
            )

            self.reports.append(report)
            # Only append to state if provided
            if state is not None:
                state.literature_reports.append(report)

            # Extract key findings
            findings = await self._extract_findings(report.report)

            return {
                "status": "completed",
                "report": report.report,
                "references": report.references,
                "findings": findings,
                "report_id": report.id
            }

        except Exception as e:
            # Fallback to quick search if deep research fails
            return await self._quick_search(topic, state)
    
    async def _quick_search(
        self,
        topic: str,
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """
        Quick literature search using LLM knowledge
        (Fallback when Deep Research is unavailable)

        Args:
            topic: Research topic
            state: Optional analysis state for context. If None, no state updates occur.
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
        # Only append to state if provided
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
    
    def _build_research_query(
        self,
        topic: str,
        state: Optional[AnalysisState] = None
    ) -> str:
        """Build an optimized research query

        Args:
            topic: Research topic
            state: Optional analysis state for context enrichment
        """
        # Add context from analysis
        context_parts = [topic]

        # Add relevant keywords based on initial instruction (if state provided)
        initial_instruction = state.initial_instruction if state else ""
        if initial_instruction:
            if "knockout" in initial_instruction.lower() or "KO" in initial_instruction:
                context_parts.append("knockout mouse model phenotype")

        if "single cell" in topic.lower() or "scRNA" in topic.lower() or "snRNA" in topic.lower():
            context_parts.append("single cell RNA sequencing marker genes")

        if "pathway" in topic.lower():
            context_parts.append("signaling pathway mechanism")

        return " ".join(context_parts)
    
    async def _extract_findings(self, report: str) -> List[str]:
        """Extract key findings from a literature report"""
        
        prompt = f"""
Extract key findings from the following literature review (in list form):

{report[:3000]}

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
    
    async def get_cell_markers(
        self,
        cell_type: str,
        organism: str = "mouse",
        tissue: str = "",
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """
        Get marker genes for a specific cell type

        Used by bioinformatics agent for cell type annotation
        Args:
            cell_type: The cell type to look up markers for
            organism: The organism (default: mouse)
            tissue: The tissue context (will be inferred from project if not provided)
            state: Optional analysis state for context enrichment
        """
        tissue_context = f" {tissue}" if tissue else ""
        topic = f"{cell_type} marker genes {organism}{tissue_context} single cell RNA sequencing"

        result = await self._deep_research(topic, state)
        
        # Extract marker genes from report
        prompt = f"""
Extract marker genes for {cell_type} from the following literature review:

{result['report'][:2000]}

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
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning(f"[Literature] No JSON found in get_cell_markers response for {cell_type}")
        except json.JSONDecodeError as e:
            logger.warning(f"[Literature] Failed to parse JSON in get_cell_markers for {cell_type}: {e}")

        return {
            "cell_type": cell_type,
            "positive_markers": [],
            "negative_markers": [],
            "confidence": "low"
        }
    
    async def validate_pathway(
        self,
        pathway_name: str,
        genes: List[str],
        context: str = "",
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """
        Validate pathway enrichment with literature evidence

        Args:
            pathway_name: Name of the pathway to validate
            genes: List of enriched genes
            context: Research context description
            state: Optional analysis state for context enrichment
        """
        topic = f"{pathway_name} pathway {' '.join(genes[:10])} disease mechanism"

        result = await self._deep_research(topic, state)
        
        prompt = f"""
Based on the literature review, validate the biological significance of the following pathway enrichment:

Pathway name: {pathway_name}
Enriched genes: {', '.join(genes[:20])}
Research context: {context}

Literature information:
{result['report'][:2000]}

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
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning(f"[Literature] No JSON found in validate_pathway response for {pathway_name}")
        except json.JSONDecodeError as e:
            logger.warning(f"[Literature] Failed to parse JSON in validate_pathway for {pathway_name}: {e}")

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
        """
        Search for molecular mechanisms underlying a phenotype

        Args:
            phenotype: Phenotype to search mechanisms for
            genes: Optional list of relevant genes
            state: Optional analysis state for context enrichment
        """
        query_parts = [phenotype, "molecular mechanism"]
        if genes:
            query_parts.extend(genes[:5])

        topic = " ".join(query_parts)
        result = await self._deep_research(topic, state)
        
        return {
            "phenotype": phenotype,
            "report": result.get("report", ""),
            "references": result.get("references", []),
            "findings": result.get("findings", [])
        }
