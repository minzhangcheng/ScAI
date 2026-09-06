"""Deterministic quality-control diagnostic panel for the exploration round.

Measured over seven real multi-round sessions: the agent ran a full QC pass every
single time and plotted it zero times. The original Stage-1 instruction listed ten
mandatory steps but deferred result figures wholesale; Stage 2 then forbade
re-running Round-1 work; and the plan audit could not notice, because
``METHOD_SIGNATURES`` holds tool families and no plotting token, so a step that
promised a plot was ``verifiable=False`` and by design never counted as a gap. The
instruction now distinguishes downstream result figures from required Stage-1
baselines, but prompt wording still is not an artifact guarantee. This module is
the deterministic quality-control half of that guarantee; the annotation,
batch/sample, cluster, and marker half lives in
``stage1_baseline_figures.py``.

Asking the model for the figure is the other half of this fix and is not enough on
its own. The same class of prompt-only figure rule has already failed here once:
the prompt that said "each figure appears exactly once" produced reports embedding
one figure three times, which is why ``services/report/figure_placement.py``
enforces placement in code. This module is the corresponding enforcement -- when
the exploration round ends with no figure of its own, the platform draws the panel
itself from what the kernel actually holds.

**Keep this generic.** The column names below are *scanpy API* names -- what
``sc.pp.calculate_qc_metrics`` and ``sc.pp.scrublet`` write into ``.obs`` -- not
organism, tissue or disease vocabulary. That is the same distinction
``METHOD_SIGNATURES`` draws when it admits tool tokens and rejects prose. A run
whose pipeline produced none of them falls back to plain numeric ``.obs`` columns,
and a run with nothing plottable produces nothing at all rather than an empty
canvas.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DIAGNOSTIC_FIGURE_NAME",
    "DIAGNOSTIC_FIGURE_TYPE",
    "DIAGNOSTIC_DESCRIPTION",
    "MAX_DIAGNOSTIC_METRICS",
    "build_probe_code",
    "parse_probe_output",
    "build_diagnostic_panel_code",
]

DIAGNOSTIC_FIGURE_NAME = "exploration_qc_diagnostics"

# ``qc`` is a first-class figure_type rather than ``other`` so the panel can be
# recognised without matching on its filename. Filenames describe the plot, not
# the stage, which is the ambiguity ``figure_placement.py`` exists to remove.
DIAGNOSTIC_FIGURE_TYPE = "qc"

DIAGNOSTIC_DESCRIPTION = (
    "Platform-generated quality-control diagnostics for the exploration round: "
    "distributions of the per-observation quality metrics this run computed, "
    "split by the run's own batch key where one exists, and the variance profile "
    "of its principal components. Rendered by the platform, not by the analysis "
    "agent, and not reviewed as a result figure."
)

# Six distributions plus one variance profile fits a 3x3 grid at a readable font
# size. Past six panels the axis labels start colliding, which the layout linter
# correctly reports as a blocking defect, and a blocked figure helps nobody.
MAX_DIAGNOSTIC_METRICS = 6

# Ordered by how universally a QC pass produces them. All four are names written
# by a library call, not chosen by the analyst.
_PRIORITY_METRICS = (
    "n_genes_by_counts",
    "total_counts",
    "doublet_score",
    "n_cells_by_counts",
)

# Experimental-design vocabulary, deliberately not biological vocabulary: these
# name how the samples were collected, which is what a QC panel needs to split on.
_GROUP_KEY_HINTS = (
    "batch",
    "library",
    "sample",
    "donor",
    "condition",
    "treatment",
    "group",
)

# Above this many levels a split stops being readable and starts being a rug.
_MAX_GROUP_LEVELS = 8


def build_probe_code() -> str:
    """Return kernel code that reports what a diagnostic panel could plot.

    Runs before any drawing. The publication-figure wrapper saves in a ``finally``
    block, so a plotting cell that decides mid-way that it has nothing to draw
    still writes a file -- an empty canvas is worse than no figure. Deciding in a
    separate, cheap probe keeps "nothing to plot" a silent no-op.

    Discovery is done in the kernel rather than from a remembered variable name:
    nothing constrains what the analysis called its object, and a stale hint picks
    the wrong one silently.
    """
    return (
        "import json as _scai_json\n"
        "_scai_out = {}\n"
        "try:\n"
        "    import anndata as _scai_ad\n"
        "    import numpy as _scai_np\n"
        "    import pandas as _scai_pd\n"
        f"    _scai_priority = {list(_PRIORITY_METRICS)!r}\n"
        f"    _scai_hints = {list(_GROUP_KEY_HINTS)!r}\n"
        f"    _scai_max_levels = {_MAX_GROUP_LEVELS!r}\n"
        f"    _scai_cap = {MAX_DIAGNOSTIC_METRICS!r}\n"
        "    _scai_best = None\n"
        "    for _scai_name, _scai_obj in list(globals().items()):\n"
        "        if _scai_name.startswith('_') or not isinstance(_scai_obj, _scai_ad.AnnData):\n"
        "            continue\n"
        "        _scai_cols = [\n"
        "            _c for _c in _scai_obj.obs.columns\n"
        "            if _scai_pd.api.types.is_numeric_dtype(_scai_obj.obs[_c])\n"
        "            and not _scai_pd.api.types.is_bool_dtype(_scai_obj.obs[_c])\n"
        "            and _scai_obj.obs[_c].nunique(dropna=True) > 1\n"
        "        ]\n"
        "        _scai_named = [_c for _c in _scai_cols if _c in _scai_priority\n"
        "                       or _c.startswith('pct_counts_') or _c.endswith('doublet_score')]\n"
        # Rank by how many named quality metrics the object carries, then by size.
        # Preferring the LARGER object on a tie is deliberate: for a quality-control
        # panel the least-filtered copy is the informative one, because it still shows
        # the observations the thresholds removed. Which object was chosen is never
        # implicit -- the panel's title carries the variable name and both dimensions.
        "        _scai_cand = (len(_scai_named), int(_scai_obj.n_obs), _scai_name)\n"
        "        if _scai_best is None or _scai_cand > _scai_best[0]:\n"
        "            _scai_best = (_scai_cand, _scai_name, _scai_obj, _scai_cols, _scai_named)\n"
        "    if _scai_best is not None and _scai_best[3]:\n"
        "        _scai_var, _scai_obj, _scai_cols, _scai_named = _scai_best[1:]\n"
        "        _scai_rest = [_c for _c in _scai_cols if _c not in _scai_named]\n"
        "        _scai_ordered = (\n"
        "            [_c for _c in _scai_priority if _c in _scai_named]\n"
        "            + sorted(_c for _c in _scai_named if _c.startswith('pct_counts_'))\n"
        "            + [_c for _c in _scai_named\n"
        "               if _c not in _scai_priority and not _c.startswith('pct_counts_')\n"
        "               and not _c.startswith('log1p_')]\n"
        "            + sorted(_c for _c in _scai_rest if not _c.startswith('log1p_'))\n"
        # scanpy writes log1p_<metric> beside every metric it computes, so an
        # unordered pass fills half the panel with the logarithm of a column
        # already shown next to it. Kept reachable, ranked last.
        "            + sorted(_c for _c in _scai_named + _scai_rest\n"
        "                     if _c.startswith('log1p_'))\n"
        "        )\n"
        "        _scai_seen = set()\n"
        "        _scai_metrics = []\n"
        "        for _c in _scai_ordered:\n"
        "            if _c not in _scai_seen:\n"
        "                _scai_seen.add(_c)\n"
        "                _scai_metrics.append(_c)\n"
        "        _scai_metrics = _scai_metrics[:_scai_cap]\n"
        "        _scai_group = None\n"
        "        _scai_levels = []\n"
        "        for _scai_hint in _scai_hints:\n"
        "            for _c in _scai_obj.obs.columns:\n"
        "                if _scai_hint not in str(_c).lower():\n"
        "                    continue\n"
        "                _scai_vals = _scai_obj.obs[_c]\n"
        "                if _scai_pd.api.types.is_numeric_dtype(_scai_vals) and not _scai_pd.api.types.is_bool_dtype(_scai_vals):\n"
        "                    continue\n"
        # dropna first: an object column's NaN stringifies to a literal 'nan' level,
        # which would be drawn as its own series and would push a genuine 8-level
        # design over the cap, suppressing the split entirely.
        "                _scai_u = [str(_v) for _v in\n"
        "                           _scai_pd.unique(_scai_vals.dropna().astype(str))]\n"
        "                if 2 <= len(_scai_u) <= _scai_max_levels:\n"
        "                    _scai_group, _scai_levels = str(_c), sorted(_scai_u)\n"
        "                    break\n"
        "            if _scai_group:\n"
        "                break\n"
        "        _scai_var_ratio = []\n"
        "        try:\n"
        "            _scai_var_ratio = [\n"
        "                float(_v) for _v in list(_scai_obj.uns['pca']['variance_ratio'])[:50]\n"
        "            ]\n"
        "        except Exception:\n"
        "            _scai_var_ratio = []\n"
        "        _scai_out = {\n"
        "            'var': _scai_var,\n"
        "            'n_obs': int(_scai_obj.n_obs),\n"
        "            'n_vars': int(_scai_obj.n_vars),\n"
        "            'metrics': _scai_metrics,\n"
        "            'group_key': _scai_group,\n"
        "            'group_levels': _scai_levels,\n"
        "            'variance_ratio': _scai_var_ratio,\n"
        "        }\n"
        "except Exception as _scai_err:\n"
        "    _scai_out = {'error': repr(_scai_err)}\n"
        "print('__SCAI_QC_PROBE__' + _scai_json.dumps(_scai_out))\n"
    )


_PROBE_MARKER = "__SCAI_QC_PROBE__"


def parse_probe_output(output: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the probe payload, or None when there is nothing worth plotting.

    Tolerant by design: a probe that emitted nothing, emitted noise, or found no
    numeric column is a reason to skip quietly, never a reason to fail the round.
    """
    if not output:
        return None
    for line in reversed(str(output).splitlines()):
        line = line.strip()
        if not line.startswith(_PROBE_MARKER):
            continue
        try:
            payload = json.loads(line[len(_PROBE_MARKER):])
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("error"):
            return None
        metrics = [str(m) for m in (payload.get("metrics") or [])]
        var = str(payload.get("var") or "")
        # ``var`` is the one probe value the panel code interpolates unquoted -- it
        # names a binding rather than carrying a value. Requiring an identifier here
        # is what keeps that safe, since the code-safety validator upstream returns
        # clean on a SyntaxError and would not catch a malformed name.
        if not metrics or not var.isidentifier():
            return None
        payload["var"] = var
        payload["metrics"] = metrics[:MAX_DIAGNOSTIC_METRICS]
        return payload
    return None


def build_diagnostic_panel_code(probe: Dict[str, Any]) -> str:
    """Return the plotting code for the panel the probe found.

    Every value the probe discovered is inlined as a literal rather than
    rediscovered, so the panel describes exactly the object that was measured.
    The code leaves one open figure for the publication wrapper to save; it never
    calls ``savefig`` itself.
    """
    metrics: List[str] = [str(m) for m in probe.get("metrics") or []][:MAX_DIAGNOSTIC_METRICS]
    var_name = str(probe.get("var") or "")
    group_key = probe.get("group_key")
    group_levels = [str(g) for g in (probe.get("group_levels") or [])]
    variance_ratio = [float(v) for v in (probe.get("variance_ratio") or [])]
    n_obs = int(probe.get("n_obs") or 0)
    n_vars = int(probe.get("n_vars") or 0)

    return (
        "import numpy as _qc_np\n"
        "import pandas as _qc_pd\n"
        "import matplotlib.pyplot as _qc_plt\n"
        f"_qc_obj = {var_name}\n"
        f"_qc_metrics = {metrics!r}\n"
        f"_qc_group_key = {group_key!r}\n"
        f"_qc_levels = {group_levels!r}\n"
        f"_qc_variance = {variance_ratio!r}\n"
        "_qc_panels = list(_qc_metrics) + (['__variance__'] if _qc_variance else [])\n"
        "_qc_n = len(_qc_panels)\n"
        "_qc_ncols = 1 if _qc_n == 1 else (2 if _qc_n <= 4 else 3)\n"
        "_qc_nrows = int(_qc_np.ceil(_qc_n / _qc_ncols))\n"
        "_qc_fig, _qc_axes = _qc_plt.subplots(\n"
        "    _qc_nrows, _qc_ncols,\n"
        "    figsize=(4.6 * _qc_ncols, 3.6 * _qc_nrows),\n"
        "    layout='constrained',\n"
        ")\n"
        "_qc_axes = _qc_np.atleast_1d(_qc_axes).ravel()\n"
        "_qc_colors = _qc_plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#4c72b0'])\n"
        "_qc_legend_done = False\n"
        "for _qc_i, _qc_panel in enumerate(_qc_panels):\n"
        "    _qc_ax = _qc_axes[_qc_i]\n"
        "    if _qc_panel == '__variance__':\n"
        "        _qc_y = _qc_np.asarray(_qc_variance, dtype=float)\n"
        "        _qc_ax.plot(_qc_np.arange(1, len(_qc_y) + 1), _qc_y, marker='o', ms=3, lw=1.2,\n"
        "                    color=_qc_colors[0])\n"
        "        _qc_ax.set_xlabel('principal component')\n"
        "        _qc_ax.set_ylabel('variance ratio')\n"
        "        _qc_ax.set_title('PCA variance profile')\n"
        "        continue\n"
        "    _qc_series = _qc_pd.to_numeric(_qc_obj.obs[_qc_panel], errors='coerce')\n"
        "    _qc_finite = _qc_series[_qc_np.isfinite(_qc_series)]\n"
        "    if _qc_finite.empty:\n"
        "        _qc_ax.set_axis_off()\n"
        "        continue\n"
        "    _qc_lo, _qc_hi = _qc_np.percentile(_qc_finite, [0.5, 99.5])\n"
        "    if not _qc_np.isfinite(_qc_lo) or not _qc_np.isfinite(_qc_hi) or _qc_hi <= _qc_lo:\n"
        "        _qc_lo, _qc_hi = float(_qc_finite.min()), float(_qc_finite.max())\n"
        "    if _qc_hi <= _qc_lo:\n"
        "        _qc_hi = _qc_lo + 1.0\n"
        "    _qc_bins = _qc_np.linspace(_qc_lo, _qc_hi, 60)\n"
        "    if _qc_group_key and _qc_levels:\n"
        "        _qc_labels = _qc_obj.obs[_qc_group_key].astype(str)\n"
        "        # Observations with no group label are drawn in the ungrouped panel\n"
        "        # rather than as a phantom level; the probe already excluded them.\n"
        "        for _qc_j, _qc_level in enumerate(_qc_levels):\n"
        "            _qc_vals = _qc_series[(_qc_labels == _qc_level).values]\n"
        "            _qc_vals = _qc_vals[_qc_np.isfinite(_qc_vals)]\n"
        "            if _qc_vals.empty:\n"
        "                continue\n"
        "            _qc_ax.hist(_qc_vals, bins=_qc_bins, histtype='step', density=True, lw=1.4,\n"
        "                        color=_qc_colors[_qc_j % len(_qc_colors)], label=_qc_level)\n"
        "    else:\n"
        "        _qc_ax.hist(_qc_finite, bins=_qc_bins, density=True, color=_qc_colors[0], alpha=0.85)\n"
        # No per-axes title: the x label already names the metric, and repeating
        # it doubles the text the layout linter has to fit for no added meaning.
        "    _qc_ax.set_xlabel(_qc_panel)\n"
        "    _qc_ax.set_ylabel('density')\n"
        "    if _qc_group_key and _qc_levels and not _qc_legend_done:\n"
        "        _qc_handles, _qc_lbls = _qc_ax.get_legend_handles_labels()\n"
        "        if _qc_handles:\n"
        "            _qc_ax.legend(title=_qc_group_key, frameon=False, fontsize=9)\n"
        "            _qc_legend_done = True\n"
        "for _qc_k in range(len(_qc_panels), len(_qc_axes)):\n"
        "    _qc_axes[_qc_k].set_axis_off()\n"
        f"_qc_fig.suptitle('Quality-control diagnostics — {var_name} "
        f"({n_obs:,} obs x {n_vars:,} vars)')\n"
    )


async def ensure_exploration_diagnostic_figure(agent: Any, state: Any) -> Optional[str]:
    """Render the exploration round's quality-control panel, if one is possible.

    Deliberately unconditional for the exploration round rather than a fallback
    that only fires when the round produced nothing: quality control is the one
    chapter whose evidence every later claim rests on, and "the agent happened to
    draw something else" is not a reason to leave it unillustrated. Re-running is
    harmless -- the figure is upserted by name.

    This can never fail the round. Every failure path returns ``None`` and logs,
    mirroring the layout linter's contract: a diagnostic that cannot be produced
    is missing information, not an error in the analysis it describes.
    """
    sandbox = getattr(agent, "sandbox", None)
    if sandbox is None:
        return None
    try:
        probe_result = await sandbox.execute(
            build_probe_code(), save_to_history=False, skip_safety_check=True, mutates_namespace=False
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.warning("[Bioinformatics] QC diagnostic probe failed: %r", err)
        return None
    if getattr(probe_result, "error", None):
        logger.info(
            "[Bioinformatics] QC diagnostic probe returned an error, skipping panel"
        )
        return None

    probe = parse_probe_output(getattr(probe_result, "output", None))
    if not probe:
        logger.info(
            "[Bioinformatics] No per-observation numeric metrics found, "
            "skipping the QC diagnostic panel"
        )
        return None

    logger.info(
        "[Bioinformatics] Rendering QC diagnostics from %s: %d metric(s), group=%r",
        probe.get("var"), len(probe.get("metrics") or []), probe.get("group_key"),
    )
    try:
        return await agent._handle_generate_publication_figure(
            code=build_diagnostic_panel_code(probe),
            figure_name=DIAGNOSTIC_FIGURE_NAME,
            description=DIAGNOSTIC_DESCRIPTION,
            figure_type=DIAGNOSTIC_FIGURE_TYPE,
            state=state,
            # No reviewer, by construction. Handing a platform-rendered
            # diagnostic to the figure-review loop would spend revision budget
            # asking a model to improve code it cannot edit, and the PI's own
            # brief-gallery rubric is instructed to reject exactly this kind of
            # panel -- so its verdict would say nothing about legibility. The
            # renderer's layout lint still runs inside the wrapper and still
            # gates the figure.
            review_callback=None,
            diagnostic=True,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.warning("[Bioinformatics] QC diagnostic panel failed: %r", err)
        return None
