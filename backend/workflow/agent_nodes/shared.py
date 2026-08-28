"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from ...utils.llm_client import UpstreamExhaustedError
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...agents.bioinformatics_agent import count_saved_figures
from ..helpers import _get_task_text, _has_stage1_keyword, _normalize_tasks, cache_hit, cache_invalidate, cache_store, decide_bioinfo_post_execution, prepend_species, safe_pi_brief, STAGE1_KEYWORDS
from ...utils.blob_store import BLOB_REF_PREFIX
from ..retry import with_retry

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict


def _reset_pi_instructions_for_stage2(pi_instructions: Dict[str, Any]) -> Dict[str, Any]:
    cleared = dict(pi_instructions)
    cleared["bioinformatics_tasks"] = []
    cleared["pathology_tasks"] = []
    cleared["biochemistry_tasks"] = []
    cleared["literature_topics"] = []
    cleared["overview"] = ""
    return cleared


def _stable_code_hash(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


_STAGE1_INSTRUCTION = """
## STAGE 1 - EXPLORATION PHASE (QC + Clustering + Annotation ONLY)

You are in STAGE 1 of the exploration phase. Execute ONLY the following analyses:

MANDATORY STEPS:
1. Load the input data (h5ad file)
2. Basic QC: filter cells by min_genes, filter genes by min_cells, compute QC metrics
3. Doublet detection assessment (scrublet or equivalent)
4. Batch effect assessment: check if multiple samples/batches exist, evaluate need for correction
5. Normalization (scanpy.pp.normalize_total), log1p
6. Highly variable gene selection (scanpy.pp.highly_variable_genes)
7. PCA (scanpy.pp.pca)
8. UMAP (scanpy.pp.neighbors + scanpy.tl.umap)
9. Clustering (scanpy.tl.leiden with appropriate resolution)
10. Cell type annotation: find marker genes per cluster (scanpy.tl.rank_genes_groups), then call annotate_clusters tool

QUALITY-CONTROL FIGURE: the platform renders one for this stage itself, from what the
kernel holds — the distribution of each per-observation quality metric this run computed,
split by the batch/sample key where the data has one, and the variance profile of the
dimensionality reduction. Do NOT reproduce it. Add a figure of your own only if a quality
decision you made cannot be read off those distributions (an unusual filter, a batch
effect you corrected, a doublet call you overrode) — and if you do, call
generate_publication_figure with figure_type="qc".

STAGE 1 BASELINE FIGURES (REQUIRED): These panels are platform-owned. After annotation is accepted, Stage 1 must
leave a visual baseline that makes the annotation auditable before downstream analysis
begins. Preserve the required UMAP, labels, cluster assignments and marker results on the
live AnnData object; the platform deterministically renders the available panels below
after this pass. Do not spend autonomous publication-figure calls duplicating them:
- `stage1_cell_type_umap`: Cell-type UMAP colored by the accepted `cell_type` labels
  (always, when those labels and UMAP coordinates exist)
- `stage1_batch_umap`: Batch/sample UMAP when a multi-level batch, sample, library, or
  donor field exists
- `stage1_cluster_umap`: Cluster UMAP when Leiden/Louvain labels exist
- `stage1_marker_dotplot`: Marker dotplot when `rank_genes_groups` results exist
Missing optional fields are a reason to omit only the corresponding panel, not the
remaining baseline figures. These baseline figures are Stage 1 evidence, not downstream
result figures; do not suppress or replace them with condition-level analyses.

WHAT NOT TO DO (these will be done in Stage 2):
- Do NOT run condition-level differential expression (DEG)
- Do NOT run GO/KEGG pathway enrichment
- Do NOT run cell-cell communication analysis (liana)
- Do NOT generate downstream result figures for condition-level DEG, enrichment,
  communication, trajectory, or other Stage 2 analyses. The quality-control and Stage 1
  baseline figures above are explicitly required and are not covered by this restriction
- Do NOT run trajectory/velocity analysis

IMPORTANT: After completing steps 1-10, call report_findings with a summary of:
- Number of cells/genes after QC
- Batch effects detected (if any) and correction applied
- Number of clusters found
- Key marker genes per cluster
- Annotation results (cell types identified)

These Stage 1 findings will be used by PI to plan Stage 2 downstream analysis.
"""


_MECHANISM_CAPTURE_INSTRUCTION = f"""

After completing downstream analysis, identify up to 8 data-driven mechanism programs when
the results support them (do NOT invent weak programs just to reach a count). A data-driven
mechanism program is a recurring, cell-type-localized or cross-validated pattern supported by
observed genes, pathways, module scores, enrichment terms, bulk/scRNA overlap, or cell-state
changes.

{MECHANISM_EVIDENCE_POLICY}

For each program include: the cell type/compartment, concrete triggering observations,
supporting genes/pathways/scores/tables, whether bulk RNA-seq supports it if bulk data exists,
alternative explanations, missing evidence, and a cautious interpretation.

Call register_mechanism_programs BEFORE report_findings. If no supportable program exists, call
register_mechanism_programs with an empty programs list and explain why in no_program_reason.
"""


def _build_exploration_instruction(
    has_bulk: bool, has_scrna: bool, scrna_replicated: bool = True,
    bulk_replicated: bool = True,
) -> str:
    condition_analysis = (
        "Execute the PI-approved bulk objectives; PI owns the supported claim "
        "level and Bioinformatics owns the method and implementation details"
        if not has_scrna else
        "Perform sample-level pseudobulk condition DEG for supported cell type groups"
        if scrna_replicated else
        "For single-cell condition comparisons, report descriptive effect sizes/ranks only; "
        "do not report cell-level or inferential pseudobulk p/FDR"
    )
    enrichment_analysis = (
        "Perform only the downstream analyses selected in the approved PI plan"
        if not has_scrna else
        "Run GO/KEGG pathway enrichment on supported replicated DEG results"
        if scrna_replicated else
        "Use replicated bulk DEG for inferential enrichment when bulk exists; treat "
        "single-cell pathway localization as exploratory"
        if bulk_replicated else
        "Keep pathway comparisons exploratory because neither sequencing arm has "
        "verified condition replication"
    )
    result_figure_analysis = (
        "Register result figures and tables that the approved bulk analysis produces"
        if not has_scrna else
        "For each supported single-cell pseudobulk DEG and enrichment result, "
        "generate the corresponding result figure"
        if scrna_replicated else
        "Generate inferential DEG/enrichment figures only for a replicated bulk "
        "arm; single-cell figures must show descriptive localization or effect sizes"
        if bulk_replicated else
        "Generate descriptive result figures only; do not create volcano or "
        "significance-ranked enrichment figures for unreplicated contrasts"
    )
    analysis_scope = (
        "Use the host input profile as facts, not as a method selector"
        if not has_scrna else
        "Keep analyses simple and high-throughput (sample-level DEG + GO/KEGG)"
        if scrna_replicated else
        "Keep inference in replicated bulk DEG + GO/KEGG and use single-cell "
        "data only for descriptive localization"
        if bulk_replicated else
        "Keep all condition comparisons descriptive and exploratory"
    )
    instr = f"""
## EXPLORATION PHASE INSTRUCTIONS
This is Round 1 — comprehensive, unbiased exploration. You MUST:
1. Complete the FULL standard pipeline (QC → normalization → clustering → annotation)
2. After annotation: merge biologically similar clusters (e.g., subtypes of the same cell type
   differing only in activation/maturation state) into unified cell type groups before downstream comparison
3. {condition_analysis}
4. {enrichment_analysis}
5. {analysis_scope}
6. Do NOT skip any cell type — but DO merge related subtypes/states of the same cell type
7. Complete cell communication analysis (liana)
8. {result_figure_analysis}
"""
    if has_bulk:
        instr += """
10. **Bulk scientific ownership**:
    - Treat the Host BulkInputProfile as observed input facts only.
    - The PI decides the scientific question and supportable claim level, including
      whether differential analysis, concordance, or deconvolution is relevant.
    - Bioinformatics selects and justifies the method and implementation details.
    - Record the method decision and register publication tables produced by the work.
    - No specific bulk analysis or package is mandatory unless it appears in the
      approved PI plan; missing semantic evidence is reported as unverified, not rejected.
"""
    instr += _MECHANISM_CAPTURE_INSTRUCTION
    return instr


def _build_innovation_instruction(
    direction: str, has_bulk: bool, bulk_replicated: bool = True,
) -> str:
    direction_ctx = direction or "PI-directed"
    instr = f"""
## INNOVATION PHASE INSTRUCTIONS
This is a targeted, hypothesis-driven round.
Research direction: {direction_ctx}

ANALYSIS REQUIREMENTS:
1. Execute ALL tasks assigned by PI below — do not skip any
2. For each task, generate at least one publication-quality figure
3. Use advanced analysis methods as specified (GSEA, gene scoring, trajectory, TF activity, etc.)
4. After completing PI's tasks, if you identify additional relevant analyses, execute them
5. When performing DEG, always include volcano plots and heatmaps of top genes
6. When performing pathway analysis, include bar plots of top enriched terms
7. For any scoring analysis (gene signatures, TF activity), show UMAP feature plots AND violin/box plots by condition
8. Report detailed statistical results (p-values, effect sizes, gene counts) — not just summaries
9. Before computing any score, consult PI about scoring method and gene list source
10. After computing any score, call register_score to document methodology for the report
"""
    if has_bulk:
        instr += """
11. **Bulk scientific ownership**:
    Bulk data remains available, but the platform does not prescribe direction
    validation, differential analysis, concordance, deconvolution, or a package.
    Follow the PI-approved question and claim level; Bioinformatics chooses and
    records scientifically compatible methods and implementation details.
"""

    instr += """
You may skip irrelevant cell populations and focus on those relevant to the hypothesis.
Do NOT re-run completed steps from Round 1 (QC, clustering, cell annotation, etc.).
"""
    return instr


def _build_stage2_instruction(
    has_bulk: bool, has_scrna: bool, scrna_replicated: bool = True,
    bulk_replicated: bool = True,
) -> str:
    """Stage 2 downstream-analysis instruction (DEG/GO/cellchat/trajectory),
    plus method-neutral bulk ownership guidance when bulk data is present. Pure
    function so it is unit-testable."""
    condition_step = (
        "Execute the PI-approved bulk objectives using methods selected and "
        "justified by Bioinformatics"
        if not has_scrna else
        "Sample-level pseudobulk condition differential expression per supported cell type"
        if scrna_replicated else
        "Descriptive single-cell condition effect sizes/ranks per supported cell type; "
        "no cell-level or inferential pseudobulk p/FDR"
    )
    instr = f"""
## STAGE 2 - DOWNSTREAM ANALYSIS (builds on Stage 1 results)

Stage 1 (QC, clustering, cell type annotation) is COMPLETE. The annotated data
(adata with a 'cell_type' column, saved as adata_annotated.h5ad) is already in
the kernel / working directory.

Do NOT re-run QC, clustering, or cell type annotation. Load the Stage 1 results
and proceed to downstream analysis ONLY:
1. {condition_step}
2. GO/KEGG pathway enrichment on DEG results (organism-appropriate libraries)
3. Cell-cell communication analysis (liana) where biologically relevant
4. Trajectory/pseudotime analysis if applicable
5. For each result, generate a publication-quality figure
"""
    if has_bulk:
        instr += """
6. **Bulk scientific ownership**: use Host profiles only as input facts. The PI
   chooses the question and claim level; Bioinformatics chooses and records the
   method and implementation details. Do not add differential analysis,
   concordance, deconvolution, or another bulk task unless the approved plan calls
   for it. Register publication tables that should enter the report.
"""
    instr += _MECHANISM_CAPTURE_INSTRUCTION
    return instr


def build_phase_instruction(planning_stage: str, analysis_phase: str,
                            has_bulk: bool, has_scrna: bool, direction: str,
                            scrna_replicated: bool = True,
                            bulk_replicated: bool = True) -> str:
    """Build the bioinformatics phase instruction. Pure function so it is unit-testable.

    Branch order matters: stage1 and stage2 are checked BEFORE the
    EXPLORATION/INNOVATION phase branches, because during round-1 stage2 the
    analysis_phase is still EXPLORATION (it flips to INNOVATION only in
    pi_synthesis).
    """
    if planning_stage == "stage1":
        if not has_scrna or scrna_replicated:
            return _STAGE1_INSTRUCTION
        return _STAGE1_INSTRUCTION + """

## DESIGN OVERRIDE — UNREPLICATED SINGLE-CELL CONDITIONS
Each single-cell condition has fewer than two independent biological units.
Assess the uncorrected sample/condition structure, but do NOT use Harmony,
BBKNN, ComBat, scVI batch correction, or another integration method to remove
sample/library differences that are confounded with condition. Concatenation
with explicit sample and condition labels is allowed. This override takes
precedence over any generic request for batch correction.
"""
    if planning_stage == "stage2":
        return _build_stage2_instruction(
            has_bulk, has_scrna, scrna_replicated=scrna_replicated,
            bulk_replicated=bulk_replicated,
        )
    if analysis_phase == "exploration":
        return _build_exploration_instruction(
            has_bulk, has_scrna, scrna_replicated=scrna_replicated,
            bulk_replicated=bulk_replicated,
        )
    return _build_innovation_instruction(
        direction, has_bulk, bulk_replicated=bulk_replicated,
    )
