"""Cell-type marker lookup, cluster annotation, and pathway validation.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Callable, Dict, Any, List, Optional
import logging


logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState

from .shared import (  # noqa: F401
    ANNOTATION_REVIEW_MAX_CHARS,
    build_atlas_query,
    _FALLBACK_EXCEPTIONS,
    _MAX_MAIN_RETRIES,
    _RETRY_DELAY_BASE,
    plausibility_terms_for_tissue,
    _gene_case_variants,
    _sanitize_synonyms,
    status_from_judge,
    build_focused_query,
)


class MarkersMixin:
    """Cell-type marker lookup, cluster annotation, and pathway validation."""

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
            cluster_markers: Dict mapping cluster_id (str) to its observed top marker genes
            organism: Organism name from the research context or data metadata
            tissue: Tissue/organ context from the research context or data metadata
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
            # Tier 1 is the precise, title-anchored atlas query; tier 2 only runs
            # if tier 1 came up short (see _run_seed_queries' budget rule).
            atlas_seeds: List[str] = []
            for tier in (1, 2):
                seed = build_atlas_query(organism, tissue, tier=tier)
                if seed and seed not in atlas_seeds:
                    atlas_seeds.append(seed)
            research_result = await self._multi_round_pubmed(
                topic, state, pubmed_max_results=50,
                initial_query=pi_initial_query, evaluate_fn=pi_evaluate_fn,
                organism=organism, tissue=tissue,
                seed_queries=atlas_seeds,
                cluster_markers=cluster_markers,
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
{literature_report[:ANNOTATION_REVIEW_MAX_CHARS]}

=== CLUSTER MARKER GENES (from sc.tl.rank_genes_groups) ===
{markers_text}

For each cluster, provide:
1. The most likely cell type identity
2. Possible subtypes if applicable
3. Key marker genes that support this annotation (from the cluster's markers)
4. Confidence level (high/medium/low) based on how well markers match literature
5. Brief evidence/reasoning citing the literature

IMPORTANT:
- Be as specific as the markers actually support, and no more. "CD8+ cytotoxic T cells" is
  better than "T cells" ONLY when the distinguishing markers are present in this cluster's list;
  in 3' single-cell data CD4/CD8 transcripts are often too sparse to make that call. Assign the
  most specific label the evidence carries, then stop — an over-specific label is not a more
  informative answer, it is a wrong one, and everything downstream inherits it
- If a cluster has ambiguous markers, list the top 2 candidates and set confidence accordingly
- Consider the tissue context ({tissue}) when making assignments
- Not every cluster is a cell type. Use these when they fit, rather than forcing a lineage label:
    * "Doublet/multiplet (suspected)" — the marker list mixes the defining markers of two
      unrelated lineages
    * "Low-quality/dying (suspected)" — the markers are dominated by mitochondrial or ribosomal
      genes with no lineage-specific signature
    * "Ambient/background (suspected)" — the top markers are the highly expressed genes of
      another, abundant population without that population's full signature
    * "Mixed/unresolved" — several distinct signatures that sub-clustering would be needed to separate
    * "Transitional/intermediate state" — markers of two related types along one lineage
    * "Unknown" — a real, coherent population whose identity the markers do not establish
  You are given ONLY each cluster's ranked marker genes — no mitochondrial fraction, no detected-
  gene counts, no doublet scores, no per-cell values. Judge from the marker list alone. The three
  "(suspected)" labels are therefore hypotheses the marker list is CONSISTENT with, not calls:
  say so in `evidence`, name the per-cell QC that would confirm or refute it, and never state one
  as established
  Forcing one of these into a specific cell type propagates a population that does not exist into
  every downstream comparison, and it cannot be detected later

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

    async def validate_pathway(
        self,
        pathway_name: str,
        genes: List[str],
        context: str = "",
        state: Optional[AnalysisState] = None
    ) -> Dict[str, Any]:
        """Validate pathway enrichment with literature evidence."""
        self.refresh_runtime_config_if_needed()
        topic = (
            f"{pathway_name} pathway {' '.join(genes[:10])} "
            f"{context[:120] + ' ' if context else ''}"
            "function, regulation, and evidence for and against involvement"
        )

        result = await self._do_research(topic, state)

        prompt = f"""
Based on the literature review, ASSESS the following pathway enrichment. Assess means decide
whether the literature supports, contradicts, or is silent on its relevance here — not confirm it.
An enrichment result that the literature does not support, or that is driven by a handful of
broadly expressed genes, is a legitimate and useful assessment:

Pathway name: {pathway_name}
Enriched genes: {', '.join(genes[:20])}
Research context: {context}

Literature information:
{result['report'][:8000]}

Please evaluate:
1. Known role of this pathway, in whatever context the literature actually establishes — do not
   assume a disease context unless the research context above states one
2. Relevance of the enriched genes to the pathway, and whether they are specific to it or broadly
   expressed genes that drive many enrichment results
3. What the literature supports, what it contradicts, and what it does not address here

Reply in JSON format:
{{
    "pathway": "{pathway_name}",
    "known_role": "Known role of the pathway",
    "gene_relevance": "Gene relevance assessment",
    "biological_significance": "What this supports here, at the claim level the evidence carries",
    "contradicting_or_absent_evidence": "Literature that argues against this, or a statement that the literature does not address it",
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
