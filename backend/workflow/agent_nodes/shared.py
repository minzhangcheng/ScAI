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
    deferred_topics = (
        pi_instructions.get("literature_topics")
        or pi_instructions.get("deferred_literature_topics")
        or []
    )
    deferred_overview = (
        pi_instructions.get("overview")
        or pi_instructions.get("deferred_literature_overview")
        or ""
    )
    cleared["bioinformatics_tasks"] = []
    cleared["pathology_tasks"] = []
    cleared["biochemistry_tasks"] = []
    cleared["literature_topics"] = []
    cleared["overview"] = ""
    # Literature runs only after Stage 2. Preserve its round-level intent under
    # dedicated keys without leaking Stage 1 execution tasks into the approved
    # Stage 2 bioinformatics plan.
    cleared["deferred_literature_topics"] = deferred_topics
    cleared["deferred_literature_overview"] = deferred_overview
    return cleared


def _stable_code_hash(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


_STAGE1_INSTRUCTION = """
## STAGE 1 - EXPLORATION PHASE (QC + Clustering + Annotation ONLY)

You are in STAGE 1 of the exploration phase, on single-cell resolution data.
Execute ONLY the following analyses:

REQUIRED STEPS (the sequence is fixed; every threshold, resolution and method choice
inside it is yours to make from the data. If a step does not apply to this dataset,
say which one and why rather than silently skipping it):
1. Load the input data (h5ad file)
2. Basic QC: filter cells by min_genes, filter genes by min_cells, compute QC metrics.
   Choose the cutoffs from the observed distributions, not from a remembered default.
   Verify the input scale before anything else: is adata.X raw integer counts? Integrality
   of the values is a clue, not proof — also read adata.layers, adata.raw, any processing
   note in adata.uns, and the description the data was uploaded with. Never copy an
   already-normalised .X into adata.layers['counts']; if no raw counts exist anywhere, say
   so, because it constrains every count-based method later.
   Compute, where they apply to this dataset, the observations that make the distributions
   readable: per-sample MAD-based outlier flags, ribosomal and haemoglobin fractions
   alongside the mitochondrial one, and cell-cycle scoring
   (sc.tl.score_genes_cell_cycle). Evaluate whether each one applies here rather than
   running the whole list by default. A mitochondrial cutoff appropriate to whole-cell data
   does not transfer to single-nucleus data.
   No threshold removes cells automatically: none of these observations is a filter on its
   own. You choose and justify each cutoff from the observed distributions and report what
   was removed and why
3. Doublet detection: assess from the design whether and how to run it, and record what you
   decided. For per-library captures, run it per library by default
   (sc.pp.scrublet(adata, batch_key=...)); scvi-tools SOLO is available as a second opinion
   where the first call is ambiguous. Pooled donors, hashed libraries, or data that arrived
   already doublet-filtered each need their own justification for what you run. A capture
   library is not the same thing as a biological sample — check which of the two the column
   you pass as batch_key actually holds
4. Batch effect assessment: check if multiple samples/batches exist, evaluate need for
   correction. Besides the embedding coloured by the batch/sample key, compute a
   per-cluster sample-mixing statistic — a sample-by-cluster crosstab, or an entropy of the
   sample labels within each cluster — so the PI has a numeric basis instead of an
   impression of a plot. Better mixing is not by itself evidence that an integration is more
   correct: an integration can also mix samples by removing the biology that separated them
5. Normalization (scanpy.pp.normalize_total), log1p. Keep the raw integer counts in
   adata.layers['counts'] before you normalize — downstream count-based models need them
6. Highly variable gene selection (scanpy.pp.highly_variable_genes). For multi-sample data,
   flavor="seurat_v3" with batch_key= is one option worth evaluating; it expects raw counts,
   so it needs layer="counts" — by this point .X is already normalised
7. PCA (scanpy.pp.pca)
8. UMAP (scanpy.pp.neighbors + scanpy.tl.umap)
9. Clustering (scanpy.tl.leiden). Justify the resolution you picked; if the cluster count
   is unstable across nearby resolutions, report that rather than presenting one as settled.
   scanpy's default backend is still leidenalg; flavor="igraph", n_iterations=2,
   directed=False is the current equivalent where it applies. Report the resolution sweep
   and the value you chose together — the chosen value alone is not auditable
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
alternative explanations, missing evidence, and an interpretation stating what the evidence supports
at the strongest class of evidence the available modalities reach.

Call register_mechanism_programs BEFORE report_findings. If no supportable program exists, call
register_mechanism_programs with an empty programs list and explain why in no_program_reason.
"""


def _build_exploration_instruction(
    has_bulk: bool, has_scrna: bool, scrna_replicated: bool = True,
    bulk_replicated: bool = True,
) -> str:
    # Retain the replication keywords for caller compatibility. Measured
    # counts are presented separately as facts; they must not let the host
    # rewrite the PI-approved scientific plan.
    del has_scrna, scrna_replicated, bulk_replicated
    condition_analysis = (
        "Execute the condition analyses and claim level selected in the PI-approved plan; "
        "consult the PI if the measured design facts make a planned method ambiguous"
    )
    enrichment_analysis = "Perform only the downstream analyses selected in the approved PI plan"
    result_figure_analysis = "Generate the result figures requested by the PI-approved plan"
    analysis_scope = (
        "Treat measured sample-design metadata as facts for PI review, not as a "
        "host-selected method or claim level"
    )
    instr = f"""
## EXPLORATION PHASE INSTRUCTIONS
This is Round 1 — comprehensive, unbiased exploration. You MUST:
1. Complete the FULL standard single-cell pipeline (QC → normalization → clustering → annotation)
2. After annotation, decide DELIBERATELY at what resolution to compare, and say which you chose:
   - Merging subtypes that differ only in activation/maturation state into one cell type gives
     a coarser, less fragmented comparison. Note that it does NOT add statistical power for a
     between-condition claim: power there comes from independent biological units (samples,
     donors, animals), and cells within one sample are not independent replicates of it.
     More cells make an estimate for THAT sample more precise; they do not make a difference
     between conditions more significant.
   - But that state IS the signal in many designs (naive vs activated, quiescent vs cycling,
     resting vs responding). Merging first destroys exactly what the study may be asking about,
     and it contradicts the mechanism programs you are asked to register below, which take
     cell-state changes as evidence.
   Default to reporting BOTH resolutions when the merge is not obviously safe: the merged
   comparison for statistical stability, and the unmerged one to show whether a state shifted.
   State the merge rule you applied so the PI can overrule it.
3. {condition_analysis}
4. {enrichment_analysis}
5. {analysis_scope}
6. Do NOT silently drop a cell type. Every population is either analysed or explicitly
   reported as not analysable, with the reason (too few cells, ambiguous identity, doublet-like)
7. Cell-cell communication analysis (liana) when the data has single-cell resolution and the
   design supports the comparison; if it does not apply, say so instead of running it anyway
8. {result_figure_analysis}
"""
    if has_bulk:
        instr += """
9. **Bulk scientific ownership**:
    - Treat the Host BulkInputProfile as observed input facts only.
    - The PI decides the scientific question and supportable claim level, including
      whether differential analysis, concordance, or deconvolution is relevant.
    - Bioinformatics selects and justifies the method and implementation details.
    - Record the method decision and register publication tables produced by the work.
    - No specific bulk analysis or package is mandatory unless it appears in the
      approved PI plan; missing execution linkage is reported to the PI, not rejected.
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
2. Generate a publication-quality figure for each result that a figure communicates better
   than text. A task whose outcome is a single number, a null result, or a statement about
   feasibility does not need one — say so rather than producing a figure to satisfy a count
3. Use the method each task actually calls for. GSEA, gene scoring, trajectory and TF activity
   are available, but they are options, not a checklist to work through
4. After completing PI's tasks, if you identify an additional analysis worth running, propose it
   to the PI with what prompted it — do not run unplanned analyses and report them as findings.
   Analyses chosen after seeing the data are exploratory by construction; that has to be on the
   record, and register_analysis_plan is how
5. When performing DEG, include a volcano plot (or equivalent effect-size vs significance view).
   A heatmap of the top DE genes across the same samples that selected them is circular — the block structure is guaranteed by the selection and is not evidence. Include one only as an illustration, label it as descriptive, and never cite it as support for the difference it was selected to show
6. When performing pathway analysis, a bar plot of top enriched terms is usually the
   clearest view — use it unless something else reads better
7. For a scoring analysis, show the score's distribution across the groups being compared.
   A UMAP feature plot is the right companion ONLY where single-cell coordinates exist; for
   data without them, the distribution plot alone is the result
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
    del has_scrna, scrna_replicated, bulk_replicated
    condition_step = (
        "Execute the condition analyses, including differential expression when "
        "selected in the PI-approved plan, at the claim level chosen by the PI; "
        "consult the PI if measured design facts require a scientific choice"
    )
    stage1_state = """Stage 1 (QC, clustering, cell type annotation) is COMPLETE. The annotated data
(adata with a 'cell_type' column, saved as adata_annotated.h5ad) is already in
the kernel / working directory.

Do NOT re-run QC, clustering, or cell type annotation. Load the Stage 1 results
and proceed to downstream analysis ONLY:"""
    downstream_menu = """2. The downstream analyses the PI-approved plan calls for. Where the plan leaves the
   method open, these are available and appropriate to single-cell data — pick from
   them by what the question needs, not by working down the list:
   - pathway enrichment over the differential results (organism-appropriate libraries)
   - pseudobulk differential expression, where the plan asks for a between-condition
     comparison whose units are samples: aggregate with
     dc.get_pseudobulk(adata, sample_col=..., groups_col=..., layer="counts", mode="sum",
     min_cells=..., min_counts=...) (decoupler 1.8), filter with
     dc.filter_by_expr(..., group=...), then fit pydeseq2 with a design formula that names
     the covariates this design actually needs. For a magnitude claim, test against a
     non-zero threshold with lfc_null and alt_hypothesis rather than filtering on the fold
     change afterwards; apply lfc_shrink(coeff=...) only when that coefficient is the
     contrast being reported. State the minimum cells per sample you required and which
     samples it dropped.
   - cell-cell communication (liana), when the design supports comparing interactions.
     Across conditions there are two separate routes and they must not be chained:
     (a) per-sample scoring with
     li.mt.rank_aggregate.by_sample(adata, groupby=..., sample_key=..., resource_name=...),
     followed by a sample-level comparison or a multi-sample decomposition chosen from the
     design; or (b) per-cell-type sample-level pseudobulk DE first, then li.multi.df_to_lr
     to map those DE statistics onto ligand-receptor pairs. On route (b) the interaction_*
     columns are means of the ligand and receptor statistics, not a calibrated communication
     p-value: they rank and describe interactions, they do not test them.
   - trajectory / pseudotime, when a continuum is the question and a root can be justified
   Say which you did not run and why; an analysis that does not fit the question is
   correctly left undone, not silently skipped.
3. Generate a publication-quality figure for each result a figure communicates better
   than text; say which results you left as text and why"""

    instr = f"""
## STAGE 2 - DOWNSTREAM ANALYSIS (builds on Stage 1 results)

{stage1_state}
1. {condition_step}
{downstream_menu}
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
        return _STAGE1_INSTRUCTION
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
