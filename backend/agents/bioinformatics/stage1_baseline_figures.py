"""Deterministic visual baselines for Stage 1 single-cell exploration.

Stage 1 establishes the representation every downstream result builds on, but a
prompt that asks an agent to draw that representation is not an artifact
guarantee. This module probes the live kernel and renders the baseline directly:

* accepted cell-type labels on UMAP;
* batch/sample labels on UMAP, when a multi-level design field exists;
* Leiden/Louvain clusters on UMAP, when present; and
* a marker dotplot from stored ``rank_genes_groups`` results, when present.

Every panel is independent. Missing UMAP coordinates suppress the UMAP panels but
not a possible marker panel; a missing batch key suppresses only the batch panel.
The service is fail-soft because a missing optional visualization must not turn a
completed analysis into a failed workflow.

Keep this module domain-neutral. Candidate fields describe analysis structure,
not a tissue, disease, organism, or research direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ANNOTATION_UMAP_NAME",
    "BATCH_UMAP_NAME",
    "CLUSTER_UMAP_NAME",
    "MARKER_DOTPLOT_NAME",
    "Stage1BaselineFigureSpec",
    "build_stage1_baseline_probe_code",
    "parse_stage1_baseline_probe_output",
    "build_stage1_baseline_figure_specs",
    "ensure_stage1_baseline_figures",
]


ANNOTATION_UMAP_NAME = "stage1_cell_type_umap"
BATCH_UMAP_NAME = "stage1_batch_umap"
CLUSTER_UMAP_NAME = "stage1_cluster_umap"
MARKER_DOTPLOT_NAME = "stage1_marker_dotplot"

_PROBE_MARKER = "__SCAI_STAGE1_BASELINE_PROBE__"
_MAX_UMAP_LEVELS = 60
_MAX_BATCH_LEVELS = 60
_MAX_MARKER_GROUPS = 40
_MAX_MARKER_GENES = 18
_MARKERS_PER_RANK_GROUP = 3
_MAX_UMAP_POINTS = 120_000

_ANNOTATION_KEYS = (
    "cell_type",
    "celltype",
    "cell_type_annotation",
    "annotation",
)
_BATCH_KEYS = (
    "batch",
    "batch_id",
    "sample",
    "sample_id",
    "library",
    "library_id",
    "donor",
    "donor_id",
)
_CLUSTER_KEYS = ("leiden", "louvain")


@dataclass(frozen=True)
class Stage1BaselineFigureSpec:
    """One deterministic figure request ready for the publication wrapper."""

    name: str
    description: str
    figure_type: str
    code: str


def build_stage1_baseline_probe_code(managed_names=None) -> str:
    """Return kernel code that discovers the best Stage 1 AnnData object.

    Discovery happens in the live kernel because the analysis agent is free to
    name its AnnData object. The probe returns metadata only; it never draws or
    mutates the object.  Exact Host-managed anchor names are skipped so an alias
    cannot win the candidate scan.
    """

    managed = sorted({str(name) for name in (managed_names or []) if str(name)})
    return (
        "import json as _s1_json\n"
        "_s1_out = {}\n"
        "try:\n"
        "    import anndata as _s1_ad\n"
        "    import numpy as _s1_np\n"
        "    import pandas as _s1_pd\n"
        f"    _s1_managed = set({managed!r})\n"
        "    _s1_seen = set()\n"
        f"    _s1_annotation_hints = {list(_ANNOTATION_KEYS)!r}\n"
        f"    _s1_batch_hints = {list(_BATCH_KEYS)!r}\n"
        f"    _s1_cluster_hints = {list(_CLUSTER_KEYS)!r}\n"
        f"    _s1_max_umap_levels = {_MAX_UMAP_LEVELS!r}\n"
        f"    _s1_max_batch_levels = {_MAX_BATCH_LEVELS!r}\n"
        f"    _s1_max_marker_groups = {_MAX_MARKER_GROUPS!r}\n"
        f"    _s1_max_marker_genes = {_MAX_MARKER_GENES!r}\n"
        f"    _s1_markers_per_group = {_MARKERS_PER_RANK_GROUP!r}\n"
        "    def _s1_levels(_series, _cap):\n"
        "        _vals = _series.dropna().astype(str)\n"
        "        if _vals.empty:\n"
        "            return []\n"
        "        _levels = sorted(str(_v) for _v in _s1_pd.unique(_vals))\n"
        "        return _levels if len(_levels) <= _cap else []\n"
        "    def _s1_pick_exact(_obj, _hints, _cap, _minimum=1):\n"
        "        _by_lower = {str(_c).lower(): str(_c) for _c in _obj.obs.columns}\n"
        "        for _hint in _hints:\n"
        "            _key = _by_lower.get(_hint)\n"
        "            if not _key:\n"
        "                continue\n"
        "            _levels = _s1_levels(_obj.obs[_key], _cap)\n"
        "            if len(_levels) >= _minimum:\n"
        "                return _key, _levels\n"
        "        return None, []\n"
        "    def _s1_pick_batch(_obj):\n"
        "        _key, _levels = _s1_pick_exact(\n"
        "            _obj, _s1_batch_hints, _s1_max_batch_levels, 2\n"
        "        )\n"
        "        if _key:\n"
        "            return _key, _levels\n"
        "        for _col in _obj.obs.columns:\n"
        "            _low = str(_col).lower()\n"
        "            if not any(_hint in _low for _hint in _s1_batch_hints):\n"
        "                continue\n"
        "            _levels = _s1_levels(_obj.obs[_col], _s1_max_batch_levels)\n"
        "            if len(_levels) >= 2:\n"
        "                return str(_col), _levels\n"
        "        return None, []\n"
        "    def _s1_pick_cluster(_obj):\n"
        "        _key, _levels = _s1_pick_exact(\n"
        "            _obj, _s1_cluster_hints, _s1_max_umap_levels, 2\n"
        "        )\n"
        "        if _key:\n"
        "            return _key, _levels\n"
        "        for _col in _obj.obs.columns:\n"
        "            _low = str(_col).lower()\n"
        "            if not any(_low.startswith(_hint + '_') for _hint in _s1_cluster_hints):\n"
        "                continue\n"
        "            _levels = _s1_levels(_obj.obs[_col], _s1_max_umap_levels)\n"
        "            if len(_levels) >= 2:\n"
        "                return str(_col), _levels\n"
        "        return None, []\n"
        "    def _s1_marker_genes(_obj):\n"
        "        _rank = _obj.uns.get('rank_genes_groups')\n"
        "        if not isinstance(_rank, dict) or 'names' not in _rank:\n"
        "            return []\n"
        "        _names = _rank.get('names')\n"
        "        _columns = []\n"
        "        if isinstance(_names, _s1_pd.DataFrame):\n"
        "            _columns = [list(_names[_c]) for _c in list(_names.columns)[:_s1_max_marker_groups]]\n"
        "        elif isinstance(_names, dict):\n"
        "            _columns = [list(_names[_c]) for _c in list(_names)[:_s1_max_marker_groups]]\n"
        "        elif getattr(getattr(_names, 'dtype', None), 'names', None):\n"
        "            _columns = [list(_names[_c]) for _c in list(_names.dtype.names)[:_s1_max_marker_groups]]\n"
        "        elif getattr(_names, 'ndim', 0) == 2:\n"
        "            _columns = [list(_names[:, _i]) for _i in range(min(_names.shape[1], _s1_max_marker_groups))]\n"
        "        _available = {str(_g) for _g in _obj.var_names}\n"
        "        if getattr(_obj, 'raw', None) is not None:\n"
        "            _available.update(str(_g) for _g in _obj.raw.var_names)\n"
        "        _selected = []\n"
        "        for _column in _columns:\n"
        "            _taken = 0\n"
        "            for _gene in _column:\n"
        "                _gene = str(_gene)\n"
        "                if not _gene or _gene == 'nan' or _gene not in _available:\n"
        "                    continue\n"
        "                if _gene not in _selected:\n"
        "                    _selected.append(_gene)\n"
        "                    _taken += 1\n"
        "                if _taken >= _s1_markers_per_group or len(_selected) >= _s1_max_marker_genes:\n"
        "                    break\n"
        "            if len(_selected) >= _s1_max_marker_genes:\n"
        "                break\n"
        "        return _selected\n"
        "    _s1_best = None\n"
        "    for _s1_name, _s1_obj in list(globals().items()):\n"
        "        if _s1_name.startswith('_') or not isinstance(_s1_obj, _s1_ad.AnnData):\n"
        "            continue\n"
        "        if _s1_name in _s1_managed or id(_s1_obj) in _s1_seen:\n"
        "            continue\n"
        "        _s1_seen.add(id(_s1_obj))\n"
        "        _s1_umap_key = None\n"
        "        for _candidate in ('X_umap', 'umap'):\n"
        "            _coords = _s1_obj.obsm.get(_candidate)\n"
        "            if _coords is not None and getattr(_coords, 'shape', (0, 0))[0] == _s1_obj.n_obs and getattr(_coords, 'shape', (0, 0))[1] >= 2:\n"
        "                _s1_umap_key = _candidate\n"
        "                break\n"
        "        _s1_annotation_key, _s1_annotation_levels = _s1_pick_exact(\n"
        "            _s1_obj, _s1_annotation_hints, _s1_max_umap_levels, 1\n"
        "        )\n"
        "        _s1_batch_key, _s1_batch_levels = _s1_pick_batch(_s1_obj)\n"
        "        _s1_cluster_key, _s1_cluster_levels = _s1_pick_cluster(_s1_obj)\n"
        "        _s1_markers = _s1_marker_genes(_s1_obj)\n"
        "        _s1_score = (\n"
        "            int(bool(_s1_umap_key and _s1_annotation_key)),\n"
        "            int(bool(_s1_annotation_key)),\n"
        "            int(bool(_s1_umap_key)),\n"
        "            int(bool(_s1_markers)),\n"
        "            int(bool(_s1_cluster_key)),\n"
        "            int(_s1_obj.n_obs),\n"
        "            _s1_name,\n"
        "        )\n"
        "        if _s1_best is None or _s1_score > _s1_best[0]:\n"
        "            _s1_best = (\n"
        "                _s1_score, _s1_name, _s1_obj, _s1_umap_key,\n"
        "                _s1_annotation_key, _s1_annotation_levels,\n"
        "                _s1_batch_key, _s1_batch_levels,\n"
        "                _s1_cluster_key, _s1_cluster_levels, _s1_markers,\n"
        "            )\n"
        "    if _s1_best is not None:\n"
        "        (_s1_score, _s1_var, _s1_obj, _s1_umap_key,\n"
        "         _s1_annotation_key, _s1_annotation_levels,\n"
        "         _s1_batch_key, _s1_batch_levels,\n"
        "         _s1_cluster_key, _s1_cluster_levels, _s1_markers) = _s1_best\n"
        "        _s1_marker_group_key = _s1_annotation_key or _s1_cluster_key\n"
        "        _s1_marker_group_levels = (\n"
        "            _s1_annotation_levels if _s1_annotation_key else _s1_cluster_levels\n"
        "        )\n"
        "        if len(_s1_marker_group_levels) > _s1_max_marker_groups:\n"
        "            _s1_marker_group_levels = []\n"
        "        _s1_out = {\n"
        "            'var': _s1_var,\n"
        "            'n_obs': int(_s1_obj.n_obs),\n"
        "            'n_vars': int(_s1_obj.n_vars),\n"
        "            'umap_key': _s1_umap_key,\n"
        "            'annotation_key': _s1_annotation_key,\n"
        "            'annotation_levels': _s1_annotation_levels,\n"
        "            'batch_key': _s1_batch_key,\n"
        "            'batch_levels': _s1_batch_levels,\n"
        "            'cluster_key': _s1_cluster_key,\n"
        "            'cluster_levels': _s1_cluster_levels,\n"
        "            'marker_group_key': _s1_marker_group_key,\n"
        "            'marker_group_levels': _s1_marker_group_levels,\n"
        "            'marker_genes': _s1_markers,\n"
        "        }\n"
        "except Exception as _s1_err:\n"
        "    _s1_out = {'error': repr(_s1_err)}\n"
        f"print({_PROBE_MARKER!r} + _s1_json.dumps(_s1_out))\n"
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


def parse_stage1_baseline_probe_output(output: Optional[str]) -> Optional[Dict[str, Any]]:
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
        if not var_name.isidentifier():
            return None

        try:
            normalized: Dict[str, Any] = {
                "var": var_name,
                "n_obs": int(payload.get("n_obs") or 0),
                "n_vars": int(payload.get("n_vars") or 0),
                "umap_key": str(payload.get("umap_key") or "") or None,
                "annotation_key": str(payload.get("annotation_key") or "") or None,
                "annotation_levels": _bounded_strings(
                    payload.get("annotation_levels"), _MAX_UMAP_LEVELS
                ),
                "batch_key": str(payload.get("batch_key") or "") or None,
                "batch_levels": _bounded_strings(
                    payload.get("batch_levels"), _MAX_BATCH_LEVELS
                ),
                "cluster_key": str(payload.get("cluster_key") or "") or None,
                "cluster_levels": _bounded_strings(
                    payload.get("cluster_levels"), _MAX_UMAP_LEVELS
                ),
                "marker_group_key": str(payload.get("marker_group_key") or "") or None,
                "marker_group_levels": _bounded_strings(
                    payload.get("marker_group_levels"), _MAX_MARKER_GROUPS
                ),
                "marker_genes": _bounded_strings(
                    payload.get("marker_genes"), _MAX_MARKER_GENES
                ),
            }
        except (TypeError, ValueError):
            return None
        if not any((
            normalized["annotation_key"] and normalized["annotation_levels"]
            and normalized["umap_key"],
            normalized["batch_key"] and normalized["batch_levels"]
            and normalized["umap_key"],
            normalized["cluster_key"] and normalized["cluster_levels"]
            and normalized["umap_key"],
            normalized["marker_group_key"] and normalized["marker_group_levels"]
            and normalized["marker_genes"],
        )):
            return None
        return normalized
    return None


def _build_categorical_umap_code(
    probe: Mapping[str, Any], *, key: str, levels: List[str], title: str
) -> str:
    var_name = str(probe["var"])
    umap_key = str(probe["umap_key"])
    n_obs = int(probe.get("n_obs") or 0)
    n_vars = int(probe.get("n_vars") or 0)
    return (
        "import numpy as _s1_np\n"
        "import matplotlib.pyplot as _s1_plt\n"
        f"_s1_obj = {var_name}\n"
        f"_s1_coords = _s1_np.asarray(_s1_obj.obsm[{umap_key!r}])[:, :2]\n"
        f"_s1_key = {key!r}\n"
        f"_s1_levels = {levels!r}\n"
        "_s1_series = _s1_obj.obs[_s1_key]\n"
        "_s1_labels = _s1_series.astype(str).to_numpy()\n"
        "_s1_valid = _s1_series.notna().to_numpy() & _s1_np.isfinite(_s1_coords).all(axis=1)\n"
        f"_s1_cap = {_MAX_UMAP_POINTS!r}\n"
        "_s1_indices = _s1_np.flatnonzero(_s1_valid)\n"
        "if len(_s1_indices) > _s1_cap:\n"
        "    _s1_parts = []\n"
        "    _s1_per_level = max(200, _s1_cap // max(1, len(_s1_levels)))\n"
        "    for _s1_level in _s1_levels:\n"
        "        _s1_idx = _s1_np.flatnonzero(_s1_valid & (_s1_labels == _s1_level))\n"
        "        if len(_s1_idx) > _s1_per_level:\n"
        "            _s1_idx = _s1_idx[_s1_np.linspace(0, len(_s1_idx) - 1, _s1_per_level, dtype=int)]\n"
        "        _s1_parts.append(_s1_idx)\n"
        "    _s1_indices = _s1_np.unique(_s1_np.concatenate(_s1_parts)) if _s1_parts else _s1_indices[:_s1_cap]\n"
        "    if len(_s1_indices) > _s1_cap:\n"
        "        _s1_indices = _s1_indices[_s1_np.linspace(0, len(_s1_indices) - 1, _s1_cap, dtype=int)]\n"
        "_s1_ncols = max(1, int(_s1_np.ceil(len(_s1_levels) / 18)))\n"
        "_s1_height = max(6.0, min(12.0, 4.5 + 0.22 * min(len(_s1_levels), 34)))\n"
        "_s1_fig, _s1_ax = _s1_plt.subplots(\n"
        "    figsize=(8.0 + 1.5 * _s1_ncols, _s1_height), layout='constrained'\n"
        ")\n"
        "_s1_colors = _s1_np.vstack([\n"
        "    _s1_plt.get_cmap(_s1_name)(_s1_np.linspace(0.0, 1.0, 20))\n"
        "    for _s1_name in ('tab20', 'tab20b', 'tab20c')\n"
        "])\n"
        "_s1_size = max(2.0, min(10.0, 1500.0 / max(1.0, len(_s1_indices)) ** 0.5))\n"
        "for _s1_i, _s1_level in enumerate(_s1_levels):\n"
        "    _s1_idx = _s1_indices[_s1_labels[_s1_indices] == _s1_level]\n"
        "    if len(_s1_idx) == 0:\n"
        "        continue\n"
        "    _s1_total = int(_s1_np.sum(_s1_valid & (_s1_labels == _s1_level)))\n"
        "    _s1_ax.scatter(\n"
        "        _s1_coords[_s1_idx, 0], _s1_coords[_s1_idx, 1],\n"
        "        s=_s1_size, alpha=0.72, linewidths=0, rasterized=len(_s1_indices) > 50000,\n"
        "        color=_s1_colors[_s1_i % len(_s1_colors)], label=f'{_s1_level} (n={_s1_total:,})',\n"
        "    )\n"
        "_s1_ax.set_xlabel('UMAP 1')\n"
        "_s1_ax.set_ylabel('UMAP 2')\n"
        f"_s1_ax.set_title({title!r} + f' — {{_s1_key}}')\n"
        "_s1_ax.set_xticks([])\n"
        "_s1_ax.set_yticks([])\n"
        "_s1_ax.legend(\n"
        "    title=_s1_key, bbox_to_anchor=(1.02, 1.0), loc='upper left',\n"
        "    frameon=False, markerscale=2.2, fontsize=8, ncol=_s1_ncols,\n"
        ")\n"
        f"_s1_fig.suptitle('Stage 1 baseline — {var_name} ({n_obs:,} cells x {n_vars:,} features)', fontsize=11)\n"
    )


def _build_marker_dotplot_code(probe: Mapping[str, Any]) -> str:
    var_name = str(probe["var"])
    group_key = str(probe["marker_group_key"])
    groups = list(probe["marker_group_levels"])
    genes = list(probe["marker_genes"])
    return (
        "import numpy as _s1_np\n"
        "import matplotlib.pyplot as _s1_plt\n"
        "try:\n"
        "    from scipy import sparse as _s1_sparse\n"
        "except Exception:\n"
        "    _s1_sparse = None\n"
        f"_s1_obj = {var_name}\n"
        f"_s1_group_key = {group_key!r}\n"
        f"_s1_groups = {groups!r}\n"
        f"_s1_genes = {genes!r}\n"
        "_s1_source = _s1_obj\n"
        "if getattr(_s1_obj, 'raw', None) is not None:\n"
        "    _s1_raw_names = {str(_g) for _g in _s1_obj.raw.var_names}\n"
        "    if all(_g in _s1_raw_names for _g in _s1_genes):\n"
        "        _s1_source = _s1_obj.raw\n"
        "_s1_index = {}\n"
        "for _s1_i, _s1_gene in enumerate(_s1_source.var_names):\n"
        "    _s1_index.setdefault(str(_s1_gene), _s1_i)\n"
        "_s1_genes = [_g for _g in _s1_genes if _g in _s1_index]\n"
        "_s1_cols = [_s1_index[_g] for _g in _s1_genes]\n"
        "_s1_x = _s1_source.X[:, _s1_cols]\n"
        "_s1_labels = _s1_obj.obs[_s1_group_key].astype(str).to_numpy()\n"
        "_s1_means = _s1_np.zeros((len(_s1_groups), len(_s1_genes)), dtype=float)\n"
        "_s1_fraction = _s1_np.zeros_like(_s1_means)\n"
        "for _s1_i, _s1_group in enumerate(_s1_groups):\n"
        "    _s1_mask = _s1_labels == _s1_group\n"
        "    if not _s1_mask.any():\n"
        "        continue\n"
        "    _s1_chunk = _s1_x[_s1_mask]\n"
        "    if _s1_sparse is not None and _s1_sparse.issparse(_s1_chunk):\n"
        "        _s1_means[_s1_i] = _s1_np.asarray(_s1_chunk.mean(axis=0)).ravel()\n"
        "        _s1_fraction[_s1_i] = _s1_np.asarray((_s1_chunk > 0).mean(axis=0)).ravel()\n"
        "    else:\n"
        "        _s1_chunk = _s1_np.asarray(_s1_chunk, dtype=float)\n"
        "        _s1_means[_s1_i] = _s1_np.nanmean(_s1_chunk, axis=0)\n"
        "        _s1_fraction[_s1_i] = _s1_np.nanmean(_s1_chunk > 0, axis=0)\n"
        "_s1_center = _s1_np.nanmean(_s1_means, axis=0, keepdims=True)\n"
        "_s1_scale = _s1_np.nanstd(_s1_means, axis=0, keepdims=True)\n"
        "_s1_scale[~_s1_np.isfinite(_s1_scale) | (_s1_scale == 0)] = 1.0\n"
        "_s1_color = _s1_np.clip((_s1_means - _s1_center) / _s1_scale, -2.5, 2.5)\n"
        "_s1_xpos, _s1_ypos = _s1_np.meshgrid(\n"
        "    _s1_np.arange(len(_s1_genes)), _s1_np.arange(len(_s1_groups))\n"
        ")\n"
        "_s1_width = max(8.0, min(15.0, 4.5 + 0.48 * len(_s1_genes)))\n"
        "_s1_height = max(5.0, min(14.0, 2.5 + 0.32 * len(_s1_groups)))\n"
        "_s1_fig, _s1_ax = _s1_plt.subplots(\n"
        "    figsize=(_s1_width, _s1_height), layout='constrained'\n"
        ")\n"
        "_s1_points = _s1_ax.scatter(\n"
        "    _s1_xpos.ravel(), _s1_ypos.ravel(),\n"
        "    s=18.0 + 240.0 * _s1_fraction.ravel(), c=_s1_color.ravel(),\n"
        "    cmap='RdBu_r', vmin=-2.5, vmax=2.5, linewidths=0.25, edgecolors='#444444',\n"
        ")\n"
        "_s1_ax.set_xticks(_s1_np.arange(len(_s1_genes)), _s1_genes, rotation=45, ha='right')\n"
        "_s1_ax.set_yticks(_s1_np.arange(len(_s1_groups)), _s1_groups)\n"
        "_s1_ax.set_xlabel('Stored top-ranked marker genes')\n"
        "_s1_ax.set_ylabel(_s1_group_key)\n"
        "_s1_ax.set_title('Stage 1 marker baseline (size: fraction detected; color: group-standardized mean)')\n"
        "_s1_ax.grid(False)\n"
        "_s1_bar = _s1_fig.colorbar(_s1_points, ax=_s1_ax, shrink=0.72, pad=0.02)\n"
        "_s1_bar.set_label('Mean expression z-score across groups')\n"
    )


def build_stage1_baseline_figure_specs(
    probe: Mapping[str, Any],
) -> List[Stage1BaselineFigureSpec]:
    """Build every baseline supported by the probe, in report-priority order."""

    specs: List[Stage1BaselineFigureSpec] = []
    umap_key = probe.get("umap_key")

    annotation_key = probe.get("annotation_key")
    annotation_levels = list(probe.get("annotation_levels") or [])
    if umap_key and annotation_key and annotation_levels:
        specs.append(Stage1BaselineFigureSpec(
            name=ANNOTATION_UMAP_NAME,
            description=(
                f"Platform-generated Stage 1 UMAP colored by accepted "
                f"{annotation_key} labels; the primary annotation baseline."
            ),
            figure_type="umap",
            code=_build_categorical_umap_code(
                probe, key=str(annotation_key), levels=annotation_levels,
                title="Accepted cell-type annotation",
            ),
        ))

    batch_key = probe.get("batch_key")
    batch_levels = list(probe.get("batch_levels") or [])
    if umap_key and batch_key and len(batch_levels) >= 2:
        specs.append(Stage1BaselineFigureSpec(
            name=BATCH_UMAP_NAME,
            description=(
                f"Platform-generated Stage 1 UMAP colored by {batch_key} to show "
                "sample or batch mixing in the retained representation."
            ),
            figure_type="umap",
            code=_build_categorical_umap_code(
                probe, key=str(batch_key), levels=batch_levels,
                title="Batch/sample structure",
            ),
        ))

    cluster_key = probe.get("cluster_key")
    cluster_levels = list(probe.get("cluster_levels") or [])
    if umap_key and cluster_key and len(cluster_levels) >= 2:
        specs.append(Stage1BaselineFigureSpec(
            name=CLUSTER_UMAP_NAME,
            description=(
                f"Platform-generated Stage 1 UMAP colored by {cluster_key} clusters "
                "before downstream condition-level analyses."
            ),
            figure_type="umap",
            code=_build_categorical_umap_code(
                probe, key=str(cluster_key), levels=cluster_levels,
                title="Unsupervised cluster baseline",
            ),
        ))

    marker_key = probe.get("marker_group_key")
    marker_levels = list(probe.get("marker_group_levels") or [])
    marker_genes = list(probe.get("marker_genes") or [])
    if marker_key and marker_levels and marker_genes:
        specs.append(Stage1BaselineFigureSpec(
            name=MARKER_DOTPLOT_NAME,
            description=(
                "Platform-generated Stage 1 marker dotplot using genes retained "
                "from the stored rank_genes_groups result, summarized by "
                f"{marker_key}. Point size is the detected fraction and color is "
                "mean expression standardized across displayed groups."
            ),
            figure_type="dotplot",
            code=_build_marker_dotplot_code(probe),
        ))

    return specs


def _safe_baseline_output_name(state: Any, requested: str) -> str:
    """Avoid overwriting an analysis-owned figure that used a baseline name.

    The publication-figure store upserts by ``name``. Stable platform names are
    useful for refreshing Stage 1 baselines, but must not give the platform
    ownership of an unrelated reviewed result merely because an agent happened
    to choose the same name first.
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


async def ensure_stage1_baseline_figures(
    agent: Any, state: Any
) -> Dict[str, str]:
    """Render all available Stage 1 baselines and return handler results by name.

    Intended workflow call::

        await ensure_stage1_baseline_figures(self.bioinformatics_agent, analysis_state)

    Call after each Stage 1/Stage 2 exploration pass. Stable names make the
    second call an upsert, which refreshes any Stage 1 baseline invalidated by a
    Stage 2 representation change. The function never raises for a probe or
    individual rendering failure.
    """

    sandbox = getattr(agent, "sandbox", None)
    if sandbox is None:
        return {}
    try:
        _managed_getter = getattr(sandbox, "host_managed_global_names", None)
        _managed = _managed_getter() if callable(_managed_getter) else set()
        probe_result = await sandbox.execute(
            build_stage1_baseline_probe_code(_managed),
            skip_safety_check=True,
            mutates_namespace=False,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.warning("[Bioinformatics] Stage 1 baseline probe failed: %r", err)
        return {}
    if getattr(probe_result, "error", None):
        logger.info(
            "[Bioinformatics] Stage 1 baseline probe returned an error; skipping"
        )
        return {}

    probe = parse_stage1_baseline_probe_output(
        getattr(probe_result, "output", None)
    )
    if not probe:
        logger.info(
            "[Bioinformatics] No annotated Stage 1 baseline is currently plottable"
        )
        return {}

    specs = build_stage1_baseline_figure_specs(probe)
    logger.info(
        "[Bioinformatics] Rendering %d Stage 1 baseline figure(s) from %s",
        len(specs), probe.get("var"),
    )
    rendered: Dict[str, str] = {}
    for spec in specs:
        output_name = _safe_baseline_output_name(state, spec.name)
        prior_ids = {figure.id for figure in getattr(state, "publication_figures", [])}
        lineage = getattr(state, "restore_branch_id", None) or "main"
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
                "[Bioinformatics] Stage 1 baseline %s failed: %r", output_name, err
            )
            continue
        for figure in getattr(state, "publication_figures", []):
            # Each render attempt receives a fresh ID; tag only this call's records.
            if (figure.id not in prior_ids and figure.name == output_name
                    and (figure.lineage_id or "main") == lineage):
                figure.report_role = "foundation_annotation"
        rendered[output_name] = str(result or "")
    return rendered
