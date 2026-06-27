"""Unit tests for validation metadata introspection code in bioinformatics_agent.

Mirrors the test_discover_adata.py pattern: exec the generated sandbox code
against real AnnData objects to verify the correct object is selected.
"""
import json
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _run_introspect_snippet(globals_dict: dict, hint: Optional[str]) -> dict:
    hint_repr = repr(hint)
    snippet = (
        "import json as _json, os as _os, anndata as _ad\n"
        "import pandas as _pd\n"
        f"_hint = {hint_repr}\n"
        "_qc = ('n_genes','n_counts','total_counts','pct_counts','percent','mito','ribo','hb',"
        "'hemoglobin','doublet','scrublet','ambient','cycle','phase','s_score','g2m')\n"
        "_cond = ('condition','treatment','genotype','disease','status')\n"
        "_samp = ('sample','donor','patient','subject')\n"
        "_batch = ('batch','lane','run','library')\n"
        "_out = {}\n"
        "for _name, _o in list(globals().items()):\n"
        "    if not isinstance(_o, _ad.AnnData) or _name.startswith('_'): continue\n"
        "    _cols = [str(_c) for _c in _o.obs.columns]; _low = {_c: _c.lower() for _c in _cols}\n"
        "    _cand = [_c for _c in _cols if _c.startswith('leiden') or _c.startswith('louvain') or _c=='cluster']\n"
        "    _key = _hint if (_hint and _hint in _cols) else None\n"
        "    if _key is None:\n"
        "        for _k in ['leiden', 'louvain', 'cluster']:\n"
        "            if _k in _cols: _key = _k; break\n"
        "    if _key is None:\n"
        "        _key = _cand[0] if _cand else None\n"
        "    if _key is None: continue\n"
        "    _summ = {}\n"
        "    for _c in _cols:\n"
        "        try:\n"
        "            _s = _o.obs[_c]\n"
        "            if str(_s.dtype) in ('category','object') or _s.nunique() <= 30:\n"
        "                _vc = _s.value_counts().head(5)\n"
        "                _summ[_c] = {'n_categories': int(_s.nunique()), 'top': [str(_i) for _i in _vc.index]}\n"
        "        except Exception: pass\n"
        "    try: _writable = _os.access('workspace', _os.W_OK)\n"
        "    except Exception: _writable = False\n"
        "    _out = {'var': _name, 'cluster_key': _key, 'candidate_cluster_keys': _cand,\n"
        "            'obs_columns': _cols,\n"
        "            'qc_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _qc)],\n"
        "            'condition_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _cond)],\n"
        "            'sample_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _samp)],\n"
        "            'batch_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _batch)],\n"
        "            'obs_summaries': _summ,\n"
        "            'layers': [str(_l) for _l in list(_o.layers.keys())],\n"
        "            'n_obs': int(_o.n_obs), 'n_vars': int(_o.n_vars),\n"
        "            'has_raw': _o.raw is not None, 'output_dir_writable': bool(_writable)}\n"
        "    break\n"
        "print(_json.dumps(_out))\n"
    )
    ns = dict(globals_dict)
    exec(snippet.replace("print(_json.dumps(_out))", "_result = _out"), ns)
    return ns.get("_result", {})


def _mk_adata(obs_cols: dict, n_obs: int = 6, n_vars: int = 3) -> ad.AnnData:
    X = np.zeros((n_obs, n_vars), dtype=np.float32)
    obs = pd.DataFrame({k: pd.Categorical(v) for k, v in obs_cols.items()})
    return ad.AnnData(X=X, obs=obs)


def test_introspect_skips_raw_picks_clustered():
    adata_con = _mk_adata({"sample": ["Con"] * 6})
    adata = _mk_adata({"sample": ["Con", "DKO"] * 3, "leiden": ["0", "1"] * 3})
    result = _run_introspect_snippet({"adata_con": adata_con, "adata": adata}, hint=None)
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden"


def test_introspect_returns_empty_when_no_clustering():
    adata_con = _mk_adata({"sample": ["A", "B", "C", "A", "B", "C"]})
    result = _run_introspect_snippet({"adata_con": adata_con}, hint=None)
    assert result == {}


def test_introspect_single_clustered():
    adata = _mk_adata({"leiden": ["0", "1"] * 3})
    result = _run_introspect_snippet({"adata": adata}, hint=None)
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden"


def test_introspect_hint_overrides_default():
    adata = _mk_adata({"leiden": ["0", "1"] * 3, "leiden_fine": ["a", "b"] * 3})
    result = _run_introspect_snippet({"adata": adata}, hint="leiden_fine")
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden_fine"


def test_introspect_prefix_fallback():
    adata = _mk_adata({"leiden_broad": ["0", "1"] * 3})
    result = _run_introspect_snippet({"adata": adata}, hint=None)
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden_broad"


def test_introspect_ignores_private_vars():
    adata = _mk_adata({"leiden": ["0", "1"] * 3})
    result = _run_introspect_snippet({"_hidden": adata}, hint=None)
    assert result == {}


def test_introspect_prefers_exact_leiden_over_prefix():
    adata = _mk_adata({
        "sample": ["Con", "DKO"] * 3,
        "leiden_res0.3": [str(i % 3) for i in range(6)],
        "leiden_res0.5": [str(i % 4) for i in range(6)],
        "leiden": [str(i % 2) for i in range(6)],
    })
    result = _run_introspect_snippet({"adata": adata}, hint=None)
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden"
    assert "leiden_res0.3" in result.get("candidate_cluster_keys", [])


def test_introspect_prefix_fallback_when_no_exact_leiden():
    adata = _mk_adata({
        "leiden_res0.3": [str(i % 3) for i in range(6)],
        "leiden_res0.5": [str(i % 4) for i in range(6)],
    })
    result = _run_introspect_snippet({"adata": adata}, hint=None)
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden_res0.3"


def test_introspect_skips_raw_picks_clustered_with_multi_leiden():
    adata_con = _mk_adata({"sample": ["Con"] * 6})
    adata = _mk_adata({
        "sample": ["Con", "DKO"] * 3,
        "leiden_res0.3": [str(i % 3) for i in range(6)],
        "leiden_res0.5": [str(i % 4) for i in range(6)],
        "leiden": [str(i % 2) for i in range(6)],
        "cell_type": ["A", "B"] * 3,
    })
    result = _run_introspect_snippet({"adata_con": adata_con, "adata": adata}, hint=None)
    assert result.get("var") == "adata"
    assert result.get("cluster_key") == "leiden"
