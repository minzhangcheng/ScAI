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
import re
import logging
from typing import Optional


logger = logging.getLogger(__name__)
from ...utils.llm_client import RETRYABLE_EXCEPTIONS
from openai import RateLimitError

_FALLBACK_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (RateLimitError,)

# 主模型重试配置（用于 _chat_with_fallback）
_MAX_MAIN_RETRIES = 2
_RETRY_DELAY_BASE = 2.0  # 秒，与 llm_client 保持一致

# How much of the synthesized literature review the cluster-matching prompts may
# read. The synthesis step regularly writes 30k+ characters; cutting it at 12k
# dropped whole cell-type sections before the matcher ever saw them.
ANNOTATION_REVIEW_MAX_CHARS = 40000

# Strings a model emits in place of the JSON null the triage prompt asks for.
# Used verbatim they became PubMed terms ("null"[tiab] matches every
# knockout-mouse paper) and prompt text ('identifying "null"').
_NULL_CANDIDATE_TOKENS = frozenset({
    "", "null", "none", "nil", "n/a", "na", "unknown", "not applicable",
})


def normalize_candidate_celltype(value) -> Optional[str]:
    """Return the candidate cell-type term, or None for absent/null-like values."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().strip("\"'`").strip()
    if text.lower() in _NULL_CANDIDATE_TOKENS:
        return None
    return text

# ===================================================================
# Focused query pure function helpers (Batch 1)
# ===================================================================

def plausibility_terms_for_tissue(tissue: str) -> list:
    """Return no static plausibility vocabulary.

    Tissue plausibility terms must come from the current annotation evidence or an
    explicit caller-provided list, never from a built-in tissue keyword table.  The
    function remains as a compatibility hook for callers that imported it directly.
    """
    del tissue
    return []


# Vocabulary for the deterministic atlas seed query. Deliberately generic: it names
# study designs and cell-inventory wording only, never a tissue, disease or organism,
# so one template serves any dataset.
_ATLAS_DESIGN_TERMS = (
    '"single-cell RNA sequencing"[tiab]',
    '"single-nucleus RNA sequencing"[tiab]',
    '"scRNA-seq"[tiab]',
    '"snRNA-seq"[tiab]',
    '"single-cell transcriptomics"[tiab]',
    '"Single-Cell Analysis"[MeSH Terms]',
)
# Cell-inventory wording. Deliberately narrow: the broad forms ("cell type",
# "heterogeneity", "landscape" anywhere in the abstract) match papers on any
# subject and reproduce the very dilution this seed exists to avoid. Measured
# against live PubMed on 2026-09-08, the broad form returned 418 hits with 10 of
# the top 20 on topic; this one returned 41 with 15 of 20 on topic.
_ATLAS_SCOPE_TERMS = (
    '"atlas"[tiab]',
    '"cell atlas"[tiab]',
    '"census"[tiab]',
    '"taxonomy"[tiab]',
    '"compendium"[tiab]',
    '"reference map"[tiab]',
    '"cell type"[ti]',
    '"cell types"[ti]',
    '"cellular landscape"[tiab]',
    '"transcriptomic landscape"[tiab]',
)


def build_atlas_query(organism: str, tissue: str, tier: int = 1) -> str:
    """A deterministic seed query for cell-inventory studies of this organism and tissue.

    The PI writes the primary annotation query, and a model asked for breadth
    reliably returns one whose OR list of generic words ("marker", "signature",
    "classification") matches papers on any topic and outranks the atlas-level
    studies annotation actually needs. This template is the floor under that: it
    names only the dataset's own organism and tissue, and its remaining two
    concepts are study design and cell-inventory wording.

    The organism is passed bare, with no quoting or field tag, so PubMed's own
    automatic term mapping supplies the species synonyms. A field-tagged phrase
    does not get that expansion: `"Mus musculus"[all]` returned 3 hits against
    live PubMed where the bare term returned 418, because most mouse papers say
    "mouse" or "mice" and are indexed under the English MeSH heading. Reading the
    synonyms off PubMed is also what keeps this domain-neutral — the alternative
    is a built-in table of organism names.

    Tier 1 requires the tissue in the title, which is what makes the result set
    small and on-topic. Tier 2 relaxes that to title/abstract or MeSH, for a
    tissue whose literature does not habitually name it in the title.
    """
    species = (organism or "").strip()
    organ = (tissue or "").strip()
    if not species and not organ:
        return ""
    parts = []
    if species:
        parts.append(species)
    if organ:
        parts.append(
            f'"{organ}"[ti]' if tier == 1
            else f'("{organ}"[tiab] OR "{organ}"[MeSH Terms])'
        )
    parts.append("(" + " OR ".join(_ATLAS_DESIGN_TERMS) + ")")
    parts.append("(" + " OR ".join(_ATLAS_SCOPE_TERMS) + ")")
    return " AND ".join(parts)


def _gene_case_variants(gene: str) -> list:
    """Return the observed symbol plus its uppercase form when they differ."""
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
        if normalize_candidate_celltype(s) is None:
            continue
        seen.add(s)
        out.append(s)
    return out


def status_from_judge(judge: dict) -> str:
    """judge 计数 → 四态。strong/limited 优先于 contradictory,其后 exhausted。

    A contradictory flag used to erase the strong hits it sat next to: a query
    whose gene symbols also matched off-topic papers made the judge report
    "mainly points elsewhere" while still counting two directly supporting
    articles, and the matcher was then told the evidence contradicted a
    correct candidate. Counted support is the more specific signal.
    """
    strong = int(judge.get("strong", 0) or 0)
    moderate = int(judge.get("moderate", 0) or 0)
    if strong >= 2:
        return "strong"
    if strong >= 1 and moderate >= 1:
        return "limited"
    if judge.get("contradictory"):
        return "contradictory"
    return "exhausted"


def build_focused_query(item: dict, organism: str, tissue: str, tier: int, bucket: str,
                        plausibility_terms: list = None) -> str:
    """构造 focused PubMed query;无可用检索词时返回空串(调用方应跳过)。

    rare_unresolved: (候选名 OR synonyms) AND (markers 双拼);tier1 加 tissue,tier2 去 tissue。
        候选名与基因符号是两个 AND 组,不再放进同一个 OR 组:一个恰好也是肿瘤抗原的
        基因符号曾单独拉回整批无关论文。无候选名时只用 markers。
    suspected_contamination: 仅 markers 双拼(不用可疑候选名);tier1=discovery(无 tissue),
                             tier2=plausibility(加 tissue + 可选 plausibility_terms)。
    """
    species = (organism or "").strip()
    marker_terms = []
    for m in item.get("top_markers", [])[:3]:
        for v in _gene_case_variants(m):
            marker_terms.append(f'"{v}"[tiab]')

    if bucket == "suspected_contamination":
        if not marker_terms:
            return ""
        or_terms = list(marker_terms)
        if tier == 2:
            terms = plausibility_terms if plausibility_terms is not None else plausibility_terms_for_tissue(tissue)
            or_terms.extend(terms)
        parts = [species] if species else []
        parts.append("(" + " OR ".join(or_terms) + ")")
        if tier == 2 and tissue:
            parts.append(tissue)
        return " AND ".join(parts)

    candidate_terms = []
    cand = normalize_candidate_celltype(item.get("candidate_celltype"))
    if cand:
        candidate_terms.append(f'"{cand}"[tiab]')
    for syn in _sanitize_synonyms(item.get("synonyms")):
        candidate_terms.append(f'"{syn}"[tiab]')

    if not candidate_terms and not marker_terms:
        return ""

    parts = [species] if species else []
    if candidate_terms:
        parts.append("(" + " OR ".join(candidate_terms) + ")")
    if marker_terms:
        parts.append("(" + " OR ".join(marker_terms) + ")")
    if tier == 1 and tissue:
        parts.append(tissue)
    return " AND ".join(parts)


