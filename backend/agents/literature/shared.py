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


logger = logging.getLogger(__name__)
from ...utils.llm_client import RETRYABLE_EXCEPTIONS
from openai import RateLimitError

_FALLBACK_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (RateLimitError,)

# 主模型重试配置（用于 _chat_with_fallback）
_MAX_MAIN_RETRIES = 2
_RETRY_DELAY_BASE = 2.0  # 秒，与 llm_client 保持一致

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


