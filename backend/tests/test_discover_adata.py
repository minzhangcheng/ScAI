"""Unit tests for AnnData discovery logic in bioinformatics_agent."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _run_discover_snippet(globals_dict: dict, hint: str | None) -> dict:
    """Simulate the sandbox discovery by exec-ing the same code body."""
    hint_repr = repr(hint)
    snippet = (
        "import json as _json, anndata as _ad\n"
        f"_hint = {hint_repr}\n"
        "_result = {}\n"
        "for _name, _obj in list(globals().items()):\n"
        "    if not isinstance(_obj, _ad.AnnData) or _name.startswith('_'):\n"
        "        continue\n"
        "    _cols = list(_obj.obs.columns)\n"
        "    _key = None\n"
        "    if _hint and _hint in _cols:\n"
        "        _key = _hint\n"
        "    if _key is None:\n"
        "        for _k in ['leiden', 'louvain', 'cluster']:\n"
        "            if _k in _cols:\n"
        "                _key = _k; break\n"
        "    if _key is None:\n"
        "        for _c in _cols:\n"
        "            if _c.startswith('leiden') or _c.startswith('louvain'):\n"
        "                _key = _c; break\n"
        "    if _key is not None:\n"
        "        _result = {'var': _name, 'cluster_key': _key}\n"
        "        break\n"
        "_out = _json.dumps(_result)\n"
    )
    ns = dict(globals_dict)
    exec(snippet, ns)
    return json.loads(ns["_out"])


def _mk_adata(col_name: str) -> ad.AnnData:
    X = np.zeros((6, 3), dtype=np.float32)
    obs = pd.DataFrame({col_name: pd.Categorical(["0", "1", "0", "1", "0", "1"])})
    return ad.AnnData(X=X, obs=obs)


def test_discover_matches_default_leiden():
    adata = _mk_adata("leiden")
    result = _run_discover_snippet({"adata": adata}, hint=None)
    assert result == {"var": "adata", "cluster_key": "leiden"}


def test_discover_prefix_fallback_leiden_broad():
    adata = _mk_adata("leiden_broad")
    result = _run_discover_snippet({"adata": adata}, hint=None)
    assert result == {"var": "adata", "cluster_key": "leiden_broad"}


def test_discover_hint_wins_over_default():
    X = np.zeros((4, 2), dtype=np.float32)
    obs = pd.DataFrame({
        "leiden": pd.Categorical(["0", "1", "0", "1"]),
        "leiden_fine": pd.Categorical(["a", "b", "a", "b"]),
    })
    adata = ad.AnnData(X=X, obs=obs)
    result = _run_discover_snippet({"adata": adata}, hint="leiden_fine")
    assert result == {"var": "adata", "cluster_key": "leiden_fine"}


def test_discover_returns_empty_when_no_clustering():
    X = np.zeros((3, 2), dtype=np.float32)
    obs = pd.DataFrame({"sample": ["A", "B", "C"]})
    adata = ad.AnnData(X=X, obs=obs)
    result = _run_discover_snippet({"adata": adata}, hint=None)
    assert result == {}


def test_discover_ignores_private_vars():
    adata = _mk_adata("leiden")
    result = _run_discover_snippet({"_adata_hidden": adata}, hint=None)
    assert result == {}