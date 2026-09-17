"""Deterministic visual baselines for innovation rounds (round 2 and later).

Stage 1 has ``stage1_baseline_figures``: the platform probes the live kernel and
renders the representation itself, so the round cannot settle without a visual
floor. Innovation rounds had no equivalent. A round could execute a dozen method
families and register almost no figure, because every panel there depends on the
executor choosing to draw one, and nothing downstream can turn a computed result
into an artifact after the fact.

This module gives those rounds the same floor, built from the same three parts:

* a probe that reports what the kernel currently holds and mutates nothing;
* independent panel specs, each suppressed only by its own missing input; and
* fail-soft rendering, because a missing optional visualization must never turn
  a completed analysis into a failed workflow.

The panels mirror the visual conventions the innovation instruction already
states: a differential result gets an effect-size-vs-significance view, a term
enrichment gets a ranked bar chart, and a score gets its distribution across the
groups being compared.

Keep this module domain-neutral. Candidate fields describe analysis structure,
not a tissue, disease, organism, or research direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "InnovationBaselineFigureSpec",
    "build_innovation_baseline_probe_code",
    "parse_innovation_baseline_probe_output",
    "build_innovation_baseline_figure_specs",
    "should_render_innovation_baselines",
    "ensure_innovation_baseline_figures",
]

_PROBE_MARKER = "__SCAI_INNOVATION_BASELINE_PROBE__"

_MAX_SCORE_KEYS = 6
_MAX_GROUP_LEVELS = 12
_MIN_GROUP_LEVELS = 2
_MAX_DE_TABLES = 2
_MAX_ENRICHMENT_TABLES = 2
_MIN_DE_ROWS = 50
_MIN_ENRICHMENT_ROWS = 3
_MAX_ENRICHMENT_ROWS = 5000
_ENRICHMENT_TERMS_SHOWN = 15
_MAX_SCATTER_POINTS = 120_000

# Experimental-design fields, not biology: what the round compares across.
_GROUP_KEYS = (
    "condition",
    "group",
    "treatment",
    "genotype",
    "arm",
    "cohort",
    "sample_group",
    "condition_group",
    "status",
)
_SCORE_HINT = "score"
_LFC_HINTS = (
    "log2foldchange",
    "log2fc",
    "logfc",
    "avg_log2fc",
    "logfoldchange",
    "log2_fold_change",
    "coef",
)
_PVAL_HINTS = (
    "padj",
    "p_val_adj",
    "adjusted p-value",
    "adj_pval",
    "qvalue",
    "fdr",
    "pvalue",
    "pval",
    "p_val",
    "p-value",
)
_TERM_HINTS = (
    "term",
    "pathway",
    "description",
    "go_term",
    "native",
    "name",
)


@dataclass(frozen=True)
class InnovationBaselineFigureSpec:
    """One deterministic figure request ready for the publication wrapper."""

    name: str
    description: str
    figure_type: str
    code: str


def _valid_output_names(output_names: Optional[Iterable[Any]]) -> List[str]:
    """Keep only safe, ordered kernel identifiers from the explicit allowlist."""
    result: List[str] = []
    for value in output_names or []:
        name = str(value or "").strip()
        if name and name.isidentifier() and name not in result:
            result.append(name)
    return result


def build_innovation_baseline_probe_code(
    output_names: Optional[Iterable[Any]] = None,
) -> str:
    """Return kernel code that reports what this round can still visualize.

    Discovery happens in the live kernel because the analysis agent names its
    own objects. ``output_names`` is an explicit receipt-derived allowlist; the
    probe reports metadata only and never draws or mutates. A declared realized
    output is a successful output-name declaration, not host-measured proof that
    the execution materially changed the object.
    """

    allowed_names = _valid_output_names(output_names)

    return (
        "import json as _ib_json\n"
        "_ib_out = {}\n"
        "try:\n"
        "    import pandas as _ib_pd\n"
        f"    _ib_output_names = {allowed_names!r}\n"
        f"    _ib_group_hints = {list(_GROUP_KEYS)!r}\n"
        f"    _ib_lfc_hints = {list(_LFC_HINTS)!r}\n"
        f"    _ib_pval_hints = {list(_PVAL_HINTS)!r}\n"
        f"    _ib_term_hints = {list(_TERM_HINTS)!r}\n"
        f"    _ib_score_hint = {_SCORE_HINT!r}\n"
        f"    _ib_max_scores = {_MAX_SCORE_KEYS!r}\n"
        f"    _ib_max_levels = {_MAX_GROUP_LEVELS!r}\n"
        f"    _ib_min_levels = {_MIN_GROUP_LEVELS!r}\n"
        f"    _ib_min_de_rows = {_MIN_DE_ROWS!r}\n"
        f"    _ib_min_enr_rows = {_MIN_ENRICHMENT_ROWS!r}\n"
        f"    _ib_max_enr_rows = {_MAX_ENRICHMENT_ROWS!r}\n"
        "    def _ib_match(_columns, _hints):\n"
        "        _lower = {str(_c).strip().lower(): str(_c) for _c in _columns}\n"
        "        for _hint in _hints:\n"
        "            if _hint in _lower:\n"
        "                return _lower[_hint]\n"
        "        for _hint in _hints:\n"
        "            for _low, _orig in _lower.items():\n"
        "                if _hint in _low:\n"
        "                    return _orig\n"
        "        return None\n"
        "    def _ib_levels(_series, _cap):\n"
        "        _vals = _series[_series.notna()].astype(str)\n"
        "        if _vals.empty:\n"
        "            return []\n"
        "        _uniq = sorted(str(_v) for _v in _ib_pd.unique(_vals))\n"
        "        return _uniq if len(_uniq) <= _cap else []\n"
        "    def _ib_pick_group(_obs):\n"
        "        _lower = {str(_c).strip().lower(): str(_c) for _c in _obs.columns}\n"
        "        for _hint in _ib_group_hints:\n"
        "            _key = _lower.get(_hint)\n"
        "            if not _key:\n"
        "                continue\n"
        "            _levels = _ib_levels(_obs[_key], _ib_max_levels)\n"
        "            if len(_levels) >= _ib_min_levels:\n"
        "                return _key, _levels\n"
        "        for _hint in _ib_group_hints:\n"
        "            for _low, _orig in _lower.items():\n"
        "                if _hint not in _low:\n"
        "                    continue\n"
        "                _levels = _ib_levels(_obs[_orig], _ib_max_levels)\n"
        "                if len(_levels) >= _ib_min_levels:\n"
        "                    return _orig, _levels\n"
        "        return None, []\n"
        # Every "adjusted" hint below is an exact column name somewhere, and
        # none of them is a substring of scanpy's ``pvals_adj`` — while the
        # generic ``pval`` IS a substring of its ``pvals``. Matching by hint
        # order alone therefore hands the raw p-value to a panel that draws a
        # 0.05 line. Ask which candidate says it is adjusted instead.
        "    def _ib_pick_pval(_columns):\n"
        "        _cands = [str(_c) for _c in _columns "
        "if _ib_match([_c], _ib_pval_hints)]\n"
        "        for _c in _cands:\n"
        "            _low = str(_c).strip().lower()\n"
        "            if 'adj' in _low or 'fdr' in _low or _low.startswith('q'):\n"
        "                return _c\n"
        "        return _cands[0] if _cands else None\n"
        # obs columns are appended as the analysis proceeds, so the TAIL is
        # this round's work and the head is round 1's QC. Truncating from the
        # front dropped the score the panel exists to show and kept
        # doublet/cell-cycle scores in its place.
        "    def _ib_score_columns(_obs):\n"
        "        _found = []\n"
        "        for _col in _obs.columns:\n"
        "            if _ib_score_hint not in str(_col).strip().lower():\n"
        "                continue\n"
        "            if not _ib_pd.api.types.is_numeric_dtype(_obs[_col]):\n"
        "                continue\n"
        "            if not bool(_obs[_col].notna().any()):\n"
        "                continue\n"
        "            _found.append(str(_col))\n"
        "        return _found[-_ib_max_scores:]\n"
        # The table panels need only pandas. Sharing one try with the AnnData
        # scan meant an import or an obs read there suppressed the volcano and
        # the enrichment chart too, against this module's own rule that a
        # panel is suppressed only by its own missing input.
        "    _ib_best = None\n"
        "    try:\n"
        "        import anndata as _ib_ad\n"
        "        for _ib_name in _ib_output_names:\n"
        "            _ib_obj = globals().get(_ib_name)\n"
        "            if not isinstance(_ib_obj, _ib_ad.AnnData):\n"
        "                continue\n"
        "            _ib_group_key, _ib_group_levels = _ib_pick_group(_ib_obj.obs)\n"
        "            _ib_scores = _ib_score_columns(_ib_obj.obs)\n"
        # Score COUNT before cell count. A round that scores a subset leaves
        # the full object carrying only the inherited scores; ranking by n_obs
        # then picked the object without the round's own result on it.
        "            _ib_rank = (\n"
        "                int(bool(_ib_scores and _ib_group_key)),\n"
        "                len(_ib_scores),\n"
        "                int(_ib_obj.n_obs),\n"
        "                _ib_name,\n"
        "            )\n"
        "            if _ib_best is None or _ib_rank > _ib_best[0]:\n"
        "                _ib_best = (\n"
        "                    _ib_rank, _ib_name, int(_ib_obj.n_obs),\n"
        "                    _ib_group_key, _ib_group_levels, _ib_scores,\n"
        "                )\n"
        "    except Exception:\n"
        "        _ib_best = None\n"
        "    _ib_de = []\n"
        "    _ib_enr = []\n"
        "    for _ib_name in _ib_output_names:\n"
        "        _ib_obj = globals().get(_ib_name)\n"
        "        if not isinstance(_ib_obj, _ib_pd.DataFrame):\n"
        "            continue\n"
        "        _ib_rows = int(len(_ib_obj))\n"
        "        _ib_cols = list(_ib_obj.columns)\n"
        "        _ib_pval = _ib_pick_pval(_ib_cols)\n"
        "        if not _ib_pval:\n"
        "            continue\n"
        # An effect-size column makes the table a differential result whatever
        # its length, so a table too short to be a landscape yields NO panel.
        # Falling through to the enrichment branch published gene symbols as
        # enriched terms: "name" substring-matches scanpy's "names" column.
        "        _ib_lfc = _ib_match(_ib_cols, _ib_lfc_hints)\n"
        "        if _ib_lfc:\n"
        "            if _ib_rows >= _ib_min_de_rows and len(_ib_de) < 4:\n"
        "                _ib_de.append({\n"
        "                    'var': _ib_name, 'lfc': _ib_lfc, 'pval': _ib_pval,\n"
        "                    'gene': _ib_match(_ib_cols, ['gene', 'symbol', 'feature', 'names']),\n"
        "                    'n_rows': _ib_rows,\n"
        "                })\n"
        "            continue\n"
        "        _ib_term = _ib_match(_ib_cols, _ib_term_hints)\n"
        "        if _ib_term and _ib_min_enr_rows <= _ib_rows <= _ib_max_enr_rows:\n"
        "            if len(_ib_enr) < 4:\n"
        "                _ib_enr.append({\n"
        "                    'var': _ib_name, 'term': _ib_term,\n"
        "                    'pval': _ib_pval, 'n_rows': _ib_rows,\n"
        "                })\n"
        "    if _ib_best is not None:\n"
        "        (_ib_rank, _ib_var, _ib_n_obs, _ib_group_key,\n"
        "         _ib_group_levels, _ib_scores) = _ib_best\n"
        "    else:\n"
        "        _ib_var, _ib_n_obs, _ib_group_key = '', 0, None\n"
        "        _ib_group_levels, _ib_scores = [], []\n"
        "    _ib_out = {\n"
        "        'var': _ib_var,\n"
        "        'n_obs': _ib_n_obs,\n"
        "        'group_key': _ib_group_key,\n"
        "        'group_levels': _ib_group_levels,\n"
        "        'score_keys': _ib_scores,\n"
        "        'de_tables': _ib_de,\n"
        "        'enrichment_tables': _ib_enr,\n"
        "    }\n"
        "except Exception as _ib_err:\n"
        "    _ib_out = {'error': repr(_ib_err)}\n"
        f"print({_PROBE_MARKER!r} + _ib_json.dumps(_ib_out))\n"
    )


def _bounded_strings(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _clean_table(entry: Any, *, required: tuple) -> Optional[Dict[str, Any]]:
    """Keep a probe table entry only when every column it names is usable."""
    if not isinstance(entry, Mapping):
        return None
    var_name = str(entry.get("var") or "")
    if not var_name.isidentifier():
        return None
    cleaned: Dict[str, Any] = {"var": var_name, "n_rows": 0}
    for field in required:
        column = str(entry.get(field) or "")
        if not column:
            return None
        cleaned[field] = column
    try:
        cleaned["n_rows"] = int(entry.get("n_rows") or 0)
    except (TypeError, ValueError):
        return None
    gene = str(entry.get("gene") or "")
    if gene:
        cleaned["gene"] = gene
    return cleaned


def parse_innovation_baseline_probe_output(
    output: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Parse the last probe marker, returning ``None`` for an unusable probe."""

    if not output:
        return None
    for line in reversed(str(output).splitlines()):
        line = line.strip()
        if not line.startswith(_PROBE_MARKER):
            continue
        try:
            payload = json.loads(line[len(_PROBE_MARKER):])
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("error"):
            return None

        var_name = str(payload.get("var") or "")
        de_tables = [
            table for table in (
                _clean_table(item, required=("lfc", "pval"))
                for item in (payload.get("de_tables") or [])
            ) if table
        ][:_MAX_DE_TABLES]
        enrichment_tables = [
            table for table in (
                _clean_table(item, required=("term", "pval"))
                for item in (payload.get("enrichment_tables") or [])
            ) if table
        ][:_MAX_ENRICHMENT_TABLES]

        score_keys = _bounded_strings(payload.get("score_keys"), _MAX_SCORE_KEYS)
        group_key = str(payload.get("group_key") or "") or None
        group_levels = _bounded_strings(
            payload.get("group_levels"), _MAX_GROUP_LEVELS
        )
        # An AnnData-backed panel needs a usable object name; the table panels
        # carry their own and stay available even when no AnnData was found.
        if not var_name.isidentifier():
            var_name = ""
            score_keys = []
        try:
            n_obs = int(payload.get("n_obs") or 0)
        except (TypeError, ValueError):
            return None

        normalized = {
            "var": var_name,
            "n_obs": n_obs,
            "group_key": group_key,
            "group_levels": group_levels,
            "score_keys": score_keys,
            "de_tables": de_tables,
            "enrichment_tables": enrichment_tables,
        }
        if not any((
            score_keys and group_key and len(group_levels) >= _MIN_GROUP_LEVELS,
            de_tables,
            enrichment_tables,
        )):
            return None
        return normalized
    return None


def _build_score_distribution_code(
    probe: Mapping[str, Any], scores: List[str], group_key: str, levels: List[str]
) -> str:
    var_name = str(probe["var"])
    n_obs = int(probe.get("n_obs") or 0)
    return (
        "import numpy as _ib_np\n"
        "import pandas as _ib_pd\n"
        "import matplotlib.pyplot as _ib_plt\n"
        f"_ib_obs = {var_name}.obs\n"
        f"_ib_scores = {scores!r}\n"
        f"_ib_group = {group_key!r}\n"
        f"_ib_levels = {levels!r}\n"
        "_ib_labels = _ib_obs[_ib_group].astype(str).to_numpy()\n"
        "_ib_n = len(_ib_scores)\n"
        "_ib_fig, _ib_axes = _ib_plt.subplots(\n"
        "    1, _ib_n, figsize=(max(4.2, 3.6 * _ib_n), 4.4),\n"
        "    layout='constrained', squeeze=False,\n"
        ")\n"
        "for _ib_i, _ib_key in enumerate(_ib_scores):\n"
        "    _ib_ax = _ib_axes[0][_ib_i]\n"
        "    _ib_vals = _ib_pd.to_numeric(\n"
        "        _ib_obs[_ib_key], errors='coerce'\n"
        "    ).to_numpy(dtype=float)\n"
        "    _ib_data = []\n"
        "    _ib_ticks = []\n"
        "    for _ib_level in _ib_levels:\n"
        "        _ib_sel = _ib_vals[\n"
        "            (_ib_labels == _ib_level) & _ib_np.isfinite(_ib_vals)\n"
        "        ]\n"
        "        if _ib_sel.size == 0:\n"
        "            continue\n"
        "        _ib_data.append(_ib_sel)\n"
        "        _ib_ticks.append(f'{_ib_level}\\n(n={_ib_sel.size:,})')\n"
        "    if not _ib_data:\n"
        "        _ib_ax.set_axis_off()\n"
        "        continue\n"
        "    _ib_ax.violinplot(_ib_data, showmedians=True, showextrema=False)\n"
        "    _ib_ax.set_xticks(range(1, len(_ib_ticks) + 1))\n"
        "    _ib_ax.set_xticklabels(_ib_ticks, fontsize=8)\n"
        "    _ib_ax.set_title(_ib_key, fontsize=10)\n"
        "    _ib_ax.set_ylabel('score')\n"
        "    _ib_ax.spines['top'].set_visible(False)\n"
        "    _ib_ax.spines['right'].set_visible(False)\n"
        "_ib_fig.suptitle(\n"
        f"    'Platform result baseline — score distribution by ' + _ib_group +\n"
        f"    ' ({var_name}, {n_obs:,} observations)', fontsize=11,\n"
        ")\n"
    )


def _build_volcano_code(table: Mapping[str, Any]) -> str:
    var_name = str(table["var"])
    lfc = str(table["lfc"])
    pval = str(table["pval"])
    return (
        "import numpy as _ib_np\n"
        "import pandas as _ib_pd\n"
        "import matplotlib.pyplot as _ib_plt\n"
        f"_ib_df = {var_name}\n"
        f"_ib_x = _ib_pd.to_numeric(_ib_df[{lfc!r}], errors='coerce').to_numpy(dtype=float)\n"
        f"_ib_p = _ib_pd.to_numeric(_ib_df[{pval!r}], errors='coerce').to_numpy(dtype=float)\n"
        "_ib_ok = _ib_np.isfinite(_ib_x) & _ib_np.isfinite(_ib_p) & (_ib_p > 0)\n"
        "_ib_x = _ib_x[_ib_ok]\n"
        "_ib_y = -_ib_np.log10(_ib_p[_ib_ok])\n"
        # No significance threshold is drawn: the host must not choose a
        # scientific cutoff, and the p-value column picked above may be raw
        # rather than multiple-testing-adjusted, so a fixed line would not
        # even be a consistent error rate. Every finite row is plotted as one
        # neutral series, and the legend reports the row count and the
        # column names used instead of a classification.
        "_ib_total = int(_ib_x.size)\n"
        f"_ib_cap = {_MAX_SCATTER_POINTS!r}\n"
        "_ib_thinned = _ib_total > _ib_cap\n"
        "if _ib_thinned:\n"
        "    _ib_keep = _ib_np.linspace(0, _ib_total - 1, _ib_cap, dtype=int)\n"
        "    _ib_x, _ib_y = _ib_x[_ib_keep], _ib_y[_ib_keep]\n"
        "_ib_fig, _ib_ax = _ib_plt.subplots(figsize=(6.6, 5.2), layout='constrained')\n"
        "if _ib_x.size:\n"
        "    _ib_ax.scatter(\n"
        "        _ib_x, _ib_y, s=5, alpha=0.45, linewidths=0,\n"
        "        color='#4878a8', rasterized=_ib_x.size > 50000,\n"
        f"        label=f'n = {{_ib_total:,}} rows; x: {lfc}, y: -log10({pval})',\n"
        "    )\n"
        "    _ib_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),\n"
        "                  frameon=False, fontsize=8, markerscale=2.0)\n"
        "    if _ib_thinned:\n"
        "        _ib_ax.text(\n"
        "            0.99, 0.01,\n"
        "            f'counts over all {_ib_total:,} rows; points thinned for rendering',\n"
        "            transform=_ib_ax.transAxes, ha='right', va='bottom',\n"
        "            fontsize=7, color='#6b7280',\n"
        "        )\n"
        "else:\n"
        "    _ib_ax.set_axis_off()\n"
        f"_ib_ax.set_xlabel({lfc!r})\n"
        f"_ib_ax.set_ylabel('-log10(' + {pval!r} + ')')\n"
        f"_ib_ax.set_title('Platform result baseline — {var_name}', fontsize=11)\n"
    )


def _build_enrichment_code(table: Mapping[str, Any]) -> str:
    var_name = str(table["var"])
    term = str(table["term"])
    pval = str(table["pval"])
    return (
        "import numpy as _ib_np\n"
        "import pandas as _ib_pd\n"
        "import matplotlib.pyplot as _ib_plt\n"
        f"_ib_df = {var_name}\n"
        f"_ib_p = _ib_pd.to_numeric(_ib_df[{pval!r}], errors='coerce')\n"
        f"_ib_terms = _ib_df[{term!r}].astype(str)\n"
        "_ib_ok = _ib_p.notna() & (_ib_p > 0)\n"
        "_ib_score = -_ib_np.log10(_ib_p[_ib_ok].to_numpy(dtype=float))\n"
        "_ib_label = _ib_terms[_ib_ok].to_numpy()\n"
        f"_ib_top = {_ENRICHMENT_TERMS_SHOWN!r}\n"
        "_ib_order = _ib_np.argsort(_ib_score)[::-1][:_ib_top][::-1]\n"
        "_ib_height = max(3.2, 0.34 * max(1, len(_ib_order)) + 1.4)\n"
        "_ib_fig, _ib_ax = _ib_plt.subplots(\n"
        "    figsize=(8.4, _ib_height), layout='constrained'\n"
        ")\n"
        "if len(_ib_order):\n"
        "    _ib_names = [\n"
        "        _ib_t if len(_ib_t) <= 58 else _ib_t[:55] + '...'\n"
        "        for _ib_t in _ib_label[_ib_order]\n"
        "    ]\n"
        "    _ib_ax.barh(range(len(_ib_order)), _ib_score[_ib_order],\n"
        "                color='#4878a8')\n"
        "    _ib_ax.set_yticks(range(len(_ib_order)))\n"
        "    _ib_ax.set_yticklabels(_ib_names, fontsize=8)\n"
        "    _ib_ax.axvline(1.30103, color='#546e7a', lw=0.8, ls='--')\n"
        "    _ib_ax.spines['top'].set_visible(False)\n"
        "    _ib_ax.spines['right'].set_visible(False)\n"
        "else:\n"
        "    _ib_ax.set_axis_off()\n"
        f"_ib_ax.set_xlabel('-log10(' + {pval!r} + ')')\n"
        "_ib_ax.set_title(\n"
        f"    'Platform result baseline — top terms in {var_name}', fontsize=11\n"
        ")\n"
    )


def build_innovation_baseline_figure_specs(
    probe: Mapping[str, Any], *, round_number: int = 0
) -> List[InnovationBaselineFigureSpec]:
    """Build every panel the probe supports, in report-priority order.

    Panels are independent: a probe that found only a differential table still
    yields the volcano. Names carry the round because an innovation round's
    result is not a refresh of the previous round's — unlike Stage 1, where a
    stable name deliberately upserts the same baseline.
    """

    specs: List[InnovationBaselineFigureSpec] = []
    prefix = f"r{int(round_number or 0)}_platform"

    scores = _bounded_strings(probe.get("score_keys"), _MAX_SCORE_KEYS)
    group_key = probe.get("group_key")
    levels = _bounded_strings(probe.get("group_levels"), _MAX_GROUP_LEVELS)
    var_name = str(probe.get("var") or "")
    if (
        scores and group_key and len(levels) >= _MIN_GROUP_LEVELS
        and var_name.isidentifier()
    ):
        specs.append(InnovationBaselineFigureSpec(
            name=f"{prefix}_score_distribution",
            description=(
                "Platform-generated distribution of every registered score "
                f"across {group_key}; the round's quantitative baseline."
            ),
            figure_type="violin",
            code=_build_score_distribution_code(
                probe, scores, str(group_key), levels
            ),
        ))

    for table in (probe.get("de_tables") or [])[:_MAX_DE_TABLES]:
        cleaned = _clean_table(table, required=("lfc", "pval"))
        if not cleaned:
            continue
        specs.append(InnovationBaselineFigureSpec(
            name=f"{prefix}_volcano_{cleaned['var']}",
            description=(
                "Platform-generated effect-size versus p-value view of "
                f"{cleaned['var']}; the differential result's baseline view."
            ),
            figure_type="volcano",
            code=_build_volcano_code(cleaned),
        ))

    for table in (probe.get("enrichment_tables") or [])[:_MAX_ENRICHMENT_TABLES]:
        cleaned = _clean_table(table, required=("term", "pval"))
        if not cleaned:
            continue
        specs.append(InnovationBaselineFigureSpec(
            name=f"{prefix}_enrichment_{cleaned['var']}",
            description=(
                "Platform-generated ranked view of the most significant terms "
                f"in {cleaned['var']}."
            ),
            figure_type="barplot",
            code=_build_enrichment_code(cleaned),
        ))

    return specs


def _safe_baseline_output_name(state: Any, requested: str) -> str:
    """Avoid overwriting an analysis-owned figure that used a baseline name.

    Mirrors the Stage 1 rule: the store upserts by ``name``, and the platform
    must not take ownership of a reviewed result merely because an agent chose
    the same name first.
    """
    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    figures = [
        figure
        for figure in (getattr(state, "publication_figures", None) or [])
        if (getattr(figure, "lineage_id", None) or "main") == active_lineage
    ]

    def available_or_platform_owned(name: str) -> bool:
        matches = [fig for fig in figures if str(getattr(fig, "name", "")) == name]
        if not matches:
            return True
        return all(
            bool(getattr(fig, "diagnostic", False))
            or str(getattr(fig, "purpose", "") or "").lower()
            in {"baseline", "diagnostic"}
            for fig in matches
        )

    if available_or_platform_owned(requested):
        return requested
    base = f"{requested}_platform_baseline"
    candidate = base
    suffix = 2
    while not available_or_platform_owned(candidate):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def should_render_innovation_baselines(current_round: Any) -> bool:
    """True for the rounds this floor covers.

    Round 1 already has one: ``stage1_baseline_figures`` renders the
    representation after Stage 1 and again after Stage 2, so adding these
    panels there would duplicate it. Integration rounds never reach the
    executor, which leaves the round number as the whole condition. Pure so the
    boundary is testable without constructing the workflow node.
    """
    try:
        return int(current_round or 0) >= 2
    except (TypeError, ValueError):
        return False


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _active_receipt_identity(state: Any) -> Optional[Dict[str, Any]]:
    """Return the current technical identity, or no identity to query."""
    plan_id = str(
        getattr(getattr(state, "current_plan", None), "id", "") or ""
    ).strip()
    if not plan_id:
        return None
    try:
        return {
            "round_number": int(getattr(state, "current_round", 0) or 0),
            "plan_id": plan_id,
            "lineage_id": getattr(state, "restore_branch_id", None) or "main",
            "adata_source_preparation_id": getattr(
                state.current_plan, "adata_source_preparation_id", None
            ),
        }
    except (TypeError, ValueError):
        return None


def _receipt_matches_current(receipt: Any, active: Mapping[str, Any]) -> bool:
    """Match lifecycle and identity fields without reading execution prose."""
    if (
        _entry_value(receipt, "evidence_schema_version") != 2
        or _entry_value(receipt, "execution_status") != "succeeded"
        or _entry_value(receipt, "error")
    ):
        return False
    try:
        return (
            int(_entry_value(receipt, "round_number", 0) or 0)
            == active["round_number"]
            and str(_entry_value(receipt, "plan_id", "") or "")
            == active["plan_id"]
            and (_entry_value(receipt, "lineage_id", None) or "main")
            == active["lineage_id"]
            and (
                not active["adata_source_preparation_id"]
                or _entry_value(receipt, "adata_source_preparation_id")
                == active["adata_source_preparation_id"]
            )
        )
    except (TypeError, ValueError):
        return False


def _current_realized_output_names(state: Any) -> List[str]:
    """Select declared outputs from newest matching successful receipts.

    The names bound here are declarations recorded by successful executions;
    they do not prove that an object was newly created or materially changed.
    """
    active = _active_receipt_identity(state)
    if active is None:
        return []
    names: List[str] = []
    history = list(getattr(state, "code_history", None) or [])
    for receipt in reversed(history):
        if not _receipt_matches_current(receipt, active):
            continue
        outputs = _entry_value(receipt, "realized_persistent_outputs", None) or []
        if isinstance(outputs, (str, bytes)):
            outputs = [outputs]
        for value in outputs:
            name = str(value or "").strip()
            if name and name.isidentifier() and name not in names:
                names.append(name)
    return names


async def ensure_innovation_baseline_figures(
    agent: Any, state: Any
) -> Dict[str, str]:
    """Render every innovation baseline the kernel supports.

    Intended workflow call::

        await ensure_innovation_baseline_figures(self.bioinformatics_agent, state)

    Call once per innovation round, after the executor has settled and before
    the round's figure audit. Never raises: a probe failure, an unusable probe,
    or a single rendering failure each degrade to fewer panels, never to a
    failed round.
    """

    sandbox = getattr(agent, "sandbox", None)
    if sandbox is None:
        return {}
    output_names = _current_realized_output_names(state)
    if not output_names:
        logger.info(
            "[Bioinformatics] No current receipt-declared outputs for "
            "innovation baseline"
        )
        return {}
    try:
        probe_result = await sandbox.execute(
            build_innovation_baseline_probe_code(output_names),
            records_history=False,
            skip_safety_check=True,
            mutates_namespace=False,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.warning("[Bioinformatics] Innovation baseline probe failed: %r", err)
        return {}
    if getattr(probe_result, "error", None):
        logger.info(
            "[Bioinformatics] Innovation baseline probe returned an error; skipping"
        )
        return {}

    probe = parse_innovation_baseline_probe_output(
        getattr(probe_result, "output", None)
    )
    if not probe:
        logger.info("[Bioinformatics] No innovation baseline is currently plottable")
        return {}

    round_number = int(getattr(state, "current_round", 0) or 0)
    specs = build_innovation_baseline_figure_specs(
        probe, round_number=round_number
    )
    if not specs:
        return {}
    logger.info(
        "[Bioinformatics] Rendering %d innovation baseline figure(s) for round %d",
        len(specs), round_number,
    )
    rendered: Dict[str, str] = {}
    for spec in specs:
        output_name = _safe_baseline_output_name(state, spec.name)
        try:
            result = await agent._handle_generate_publication_figure(
                code=spec.code,
                figure_name=output_name,
                description=spec.description,
                figure_type=spec.figure_type,
                state=state,
                review_callback=None,
                diagnostic=True,
                purpose="baseline",
            )
        except Exception as err:  # pragma: no cover - defensive
            logger.warning(
                "[Bioinformatics] Innovation baseline %s failed: %r",
                output_name, err,
            )
            continue
        rendered[output_name] = str(result or "")
    return rendered
