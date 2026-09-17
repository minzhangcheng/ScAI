# Kernel-side inspection of the live AnnData object and cluster/group proportions.
# Moved verbatim out of backend/agents/bioinformatics_agent.py; composed into BioinformaticsAgent.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import asyncio
import functools
import json
import logging
import base64
import io
import uuid
import re
import ast
import textwrap
import time
from pathlib import Path
from datetime import datetime

import aiofiles

from ..base import BaseAgent, AgentConfig
from ...utils.settings_provider import AgentSettingsProvider
from ...utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult, _TPL_NOOP_INSTALL,
)
from ...utils.checkpoint import CheckpointManager
import inspect
import json as _json
from backend.services import ora_kernel
from backend.services.marker_priors import MarkerDB, summarize_database_priors

from ...utils.formatting import format_revision_history, truncate_for_llm
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)


from .shared import (  # noqa: F401
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
    Tuple,
    asyncio,
    functools,
    json,
    logging,
    base64,
    io,
    uuid,
    re,
    ast,
    textwrap,
    time,
    Path,
    datetime,
    aiofiles,
    BaseAgent,
    AgentConfig,
    AgentSettingsProvider,
    validate_code_safety,
    DANGEROUS_PATTERNS,
    extract_code_from_markdown,
    LocalJupyterSandbox,
    CodeExecutionResult,
    _TPL_NOOP_INSTALL,
    CheckpointManager,
    inspect,
    _json,
    ora_kernel,
    MarkerDB,
    summarize_database_priors,
    _ORA_KERNEL_SRC,
    format_revision_history,
    truncate_for_llm,
    MECHANISM_EVIDENCE_POLICY,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _OVERCLAIM_WORDS,
    _NEGATION_WORDS,
    _MAX_PROGRAMS,
    _MAX_GENES,
    _MAX_OBS_LEN,
    _DEFAULT_MISSING,
    _REQUIRED_FIELDS,
    _ensure_list,
    _overclaim_unnegated,
    validate_mechanism_program_payload,
    TOOLS,
    KNOWN_TOOL_NAMES,
    build_gate_skip_note,
    count_saved_figures,
    AgentType,
    AnalysisPhase,
    AnalysisState,
    MessageRole,
    CodeExecution,
    DataDrivenMechanismProgram,
    PublicationFigure,
    FigureReviewStatus,
    AnalysisStep,
    PlanStage,
    DATA_TYPE_SCRNASEQ,
    DATA_TYPE_BULK_RNASEQ,
    OpenAICompatibleClient,
    ToolCall,
    UpstreamExhaustedError,
    EMPTY_CONTENT_PLACEHOLDER,
    tool_calls_are_paired,
    append_missing_tool_results,
    PLAN_REVISION_NEEDED_TAG,
    strip_ansi,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_invalidated_file_stems,
    record_representation_change,
    unchanged_artifact_stems,
    workspace_artifact_versions,
    SupervisorEvalResult,
    _DYNAMIC_CONTEXT_PREFIX,
    _APPROVED_PLAN_PREFIX,
    _CELL_TYPE_ASSIGN_RE,
    _FAILED_EVIDENCE_OUTPUT_RE,
    build_approved_plan_context,
    inject_revised_plan_context,
    execution_supports_evidence,
    successful_evidence_executions,
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
    _build_swallowed_error_feedback,
    _MARKER_DB,
    _get_marker_db,
    marker_priors_write,
)


class IntrospectionMixin:
    """Kernel-side inspection of the live AnnData object and cluster/group proportions."""

    def _sanitize_cluster_markers(self, cluster_markers) -> dict:
        """Sanitize and validate LLM-provided cluster_markers format.

        Handles known malformed patterns:
        1. {"str": "<JSON string>"} — unwrap and parse the inner JSON
        2. String type — try to parse as JSON
        3. Values that are not list[str] — mark as invalid, return {} to trigger auto-extraction

        Returns a cleaned dict of cluster_id -> list of gene names.
        """
        # Step 1: Handle non-dict inputs
        if not isinstance(cluster_markers, dict):
            if isinstance(cluster_markers, str):
                try:
                    parsed = json.loads(cluster_markers)
                    if isinstance(parsed, dict):
                        cluster_markers = parsed
                    else:
                        return {}
                except (json.JSONDecodeError, TypeError):
                    return {}
            else:
                return {}

        # Step 2: Detect and unwrap {"str": "<JSON>"} pattern from LLM JSON serialization bug
        if len(cluster_markers) == 1:
            only_key = next(iter(cluster_markers))
            only_val = cluster_markers[only_key]
            if isinstance(only_val, str) and only_val.strip().startswith("{"):
                try:
                    parsed = json.loads(only_val)
                    if isinstance(parsed, dict) and len(parsed) > 1:
                        logger.warning(
                            f"[Bioinformatics] Fixed malformed cluster_markers: "
                            f"unwrapped JSON string from key '{only_key}', "
                            f"recovered {len(parsed)} clusters"
                        )
                        cluster_markers = parsed
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"[Bioinformatics] Failed to parse suspected JSON string "
                        f"in cluster_markers['{only_key}']: {e}, "
                        f"value preview: {str(only_val)[:200]}"
                    )

        # Step 3: Validate each value is list[str] (gene name list)
        valid = {}
        for k, v in cluster_markers.items():
            if isinstance(v, list) and len(v) > 0 and all(isinstance(g, str) for g in v):
                valid[k] = v

        if len(valid) == 0 and len(cluster_markers) > 0:
            logger.warning(
                f"[Bioinformatics] cluster_markers has {len(cluster_markers)} entries "
                f"but none contain valid gene lists, forcing auto-extraction"
            )
            return {}

        if len(valid) < len(cluster_markers):
            logger.warning(
                f"[Bioinformatics] Dropped {len(cluster_markers) - len(valid)} "
                f"invalid entries from cluster_markers"
            )

        return valid

    async def _discover_adata_in_sandbox(
        self,
        cluster_key_hint: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Discover the active AnnData variable name and clustering key in the sandbox.

        Resolution order for cluster key:
          1. cluster_key_hint (from LLM tool args), if the column exists.
          2. Default names 'leiden', 'louvain', 'cluster'.
          3. First column whose name starts with 'leiden' or 'louvain'.

        Returns:
            Tuple of (variable_name, clustering_key), or (None, None) on failure.
        """
        if not self.sandbox:
            return (None, None)

        hint_repr = repr(cluster_key_hint)
        _managed_getter = getattr(self.sandbox, "host_managed_global_names", None)
        _managed = set(_managed_getter()) if callable(_managed_getter) else set()
        managed_repr = repr(sorted(_managed))
        discover_code = (
            "import json as _json, anndata as _ad\n"
            f"_hint = {hint_repr}\n"
            f"_managed = set({managed_repr})\n"
            "_seen_ids = set()\n"
            "_result = {}\n"
            "for _name, _obj in list(globals().items()):\n"
            "    if not isinstance(_obj, _ad.AnnData) or _name.startswith('_'):\n"
            "        continue\n"
            "    if _name in _managed or id(_obj) in _seen_ids:\n"
            "        continue\n"
            "    _seen_ids.add(id(_obj))\n"
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
            "print(_json.dumps(_result))\n"
        )
        try:
            result = await self.sandbox.execute(discover_code, records_history=False, skip_safety_check=True, mutates_namespace=False)
            if result.success and result.output and result.output.strip():
                parsed = json.loads(result.output.strip())
                var_name = parsed.get("var")
                cluster_key = parsed.get("cluster_key")
                if var_name and cluster_key:
                    logger.info(
                        f"[Bioinformatics] Discovered AnnData var={var_name}, "
                        f"cluster_key={cluster_key} (hint={cluster_key_hint!r})"
                    )
                    return (var_name, cluster_key)
        except Exception as e:
            logger.warning(f"[Bioinformatics] AnnData discovery failed: {e}")

        return (None, None)

    async def introspect_validation_metadata(
        self,
        cluster_key_hint: Optional[str],
        preferred_var_name: Optional[str] = None,
        *,
        require_preferred: bool = False,
    ) -> Dict[str, Any]:
        """Inspect one active AnnData without depending on globals insertion order.

        Existing callers that do not name a preferred variable retain the legacy
        first-available behavior. Supplementary execution supplies the durable
        resolved source name (or the legacy ``adata`` convention) and can require
        an exact match for an explicitly selected working binding.
        """
        if not self.sandbox:
            return {}
        hint_repr = repr(cluster_key_hint)
        preferred_repr = repr(preferred_var_name)
        _managed_getter = getattr(self.sandbox, "host_managed_global_names", None)
        _managed = set(_managed_getter()) if callable(_managed_getter) else set()
        managed_repr = repr(sorted(_managed))
        code = (
            "import json as _json, os as _os, anndata as _ad\n"
            f"_managed = set({managed_repr})\n"
            "import pandas as _pd\n"
            f"_hint = {hint_repr}\n"
            f"_preferred = {preferred_repr}\n"
            f"_require_preferred = {bool(require_preferred)!r}\n"
            "_qc = ('n_genes','n_counts','total_counts','pct_counts','percent','mito','ribo','hb',"
            "'hemoglobin','doublet','scrublet','ambient','cycle','phase','s_score','g2m')\n"
            "_cond = ('condition','treatment','genotype','disease','status','group','arm','cohort')\n"
            "_samp = ('sample','donor','patient','subject')\n"
            "_batch = ('batch','lane','run','library')\n"
            "def _cluster_key(_o):\n"
            "    _cols = [str(_c) for _c in _o.obs.columns]\n"
            "    _cand = [_c for _c in _cols if _c.startswith('leiden') or _c.startswith('louvain') or _c=='cluster']\n"
            "    _key = _hint if (_hint and _hint in _cols) else None\n"
            "    if _key is None:\n"
            "        for _k in ['leiden', 'louvain', 'cluster']:\n"
            "            if _k in _cols: _key = _k; break\n"
            "    if _key is None: _key = _cand[0] if _cand else None\n"
            "    return _key, _cand\n"
            "_candidates = {}\n"
            "_seen_ids = set()\n"
            "for _name, _o in list(globals().items()):\n"
            "    if not isinstance(_o, _ad.AnnData) or _name.startswith('_'): continue\n"
            "    if _name in _managed or id(_o) in _seen_ids: continue\n"
            "    _seen_ids.add(id(_o))\n"
            "    _key, _cand = _cluster_key(_o)\n"
            "    if _key is not None: _candidates[_name] = (_key, _cand)\n"
            "_selected = None; _selection_source = None; _selection_error = None\n"
            "if _preferred and _preferred in _candidates:\n"
            "    _selected = _preferred; _selection_source = 'preferred'\n"
            "elif _preferred and _require_preferred:\n"
            "    _selection_error = 'preferred AnnData is unavailable or has no clustering key'\n"
            "elif _preferred and len(_candidates) == 1:\n"
            "    _selected = next(iter(_candidates)); _selection_source = 'sole_candidate'\n"
            "elif _preferred and len(_candidates) > 1:\n"
            "    _selection_error = 'multiple AnnData candidates and no authoritative preferred object'\n"
            "elif _candidates:\n"
            "    _selected = next(iter(_candidates)); _selection_source = 'first_available'\n"
            "_out = {'candidate_names': list(_candidates), 'selection_source': _selection_source, 'selection_error': _selection_error}\n"
            "if _selected is not None:\n"
            "    _name = _selected; _o = globals()[_name]\n"
            "    _cols = [str(_c) for _c in _o.obs.columns]; _low = {_c: _c.lower() for _c in _cols}\n"
            "    _key, _cand = _candidates[_name]\n"
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
            "    _out.update({'var': _name, 'cluster_key': _key, 'candidate_cluster_keys': _cand,\n"
            "            'obs_columns': _cols,\n"
            "            'qc_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _qc)],\n"
            "            'condition_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _cond)],\n"
            "            'sample_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _samp)],\n"
            "            'batch_like': [_c for _c in _cols if any(_p in _low[_c] for _p in _batch)],\n"
            "            'obs_summaries': _summ,\n"
            "            'layers': [str(_l) for _l in list(_o.layers.keys())],\n"
            "            'n_obs': int(_o.n_obs), 'n_vars': int(_o.n_vars),\n"
            "            'has_raw': _o.raw is not None, 'output_dir_writable': bool(_writable)})\n"
            "print(_json.dumps(_out))\n"
        )
        try:
            result = await self.sandbox.execute(code, records_history=False, skip_safety_check=True, mutates_namespace=False)
            out = (getattr(result, "output", "") or "").strip()
            line = next((ln for ln in out.splitlines()[::-1] if ln.strip().startswith("{")), None)
            parsed = json.loads(line) if line else {}
            if not isinstance(parsed, dict):
                return {}
            return parsed if parsed.get("var") or parsed.get("selection_error") else {}
        except Exception as e:
            logger.warning(f"[{getattr(self, 'session_id', '?')}] validation metadata introspection failed: {e}")
            return {}

    @staticmethod
    def _normalize_group_label(value: Any) -> str:
        return "".join(str(value).lower().split())

    @classmethod
    def _matching_treatment_labels(cls, obs_labels: List[str], treatment_groups: Optional[List[Any]]) -> List[str]:
        labels = [str(label) for label in obs_labels]
        present = {cls._normalize_group_label(label): label for label in labels}
        matched: List[str] = []
        for group in sorted(treatment_groups or [], key=lambda g: getattr(g, "order", 0)):
            group_name = getattr(group, "name", "")
            key = cls._normalize_group_label(group_name)
            if key in present and present[key] not in matched:
                matched.append(present[key])
        return matched

    @classmethod
    def _ordered_proportion_labels(
        cls,
        obs_labels: List[str],
        treatment_groups: Optional[List[Any]],
    ) -> List[str]:
        labels = [str(label) for label in obs_labels]
        matched = cls._matching_treatment_labels(labels, treatment_groups)

        if matched:
            remaining = sorted(label for label in labels if label not in matched)
            return matched + remaining
        return sorted(labels)

    @classmethod
    def _select_proportion_grouping(
        cls,
        meta: Dict[str, Any],
        treatment_groups: Optional[List[Any]],
    ) -> Dict[str, Any]:
        obs_summaries = meta.get("obs_summaries") or {}

        best = None
        best_count = 0
        for col in meta.get("condition_like") or []:
            summary = obs_summaries.get(col) or {}
            labels = [str(label) for label in summary.get("top") or []]
            n_categories = int(summary.get("n_categories") or len(labels))
            if n_categories <= 1 or not labels:
                continue
            matched = cls._matching_treatment_labels(labels, treatment_groups)
            # Require the column to actually be the experimental grouping: every
            # visible category must map to a treatment group and the total category
            # count must be compatible with the number of groups. A partial match
            # (e.g. ["case", "Control"]) is disease/status metadata, not the
            # user-defined groups, so it falls back to overall per the spec.
            fully_mapped = (
                len(matched) == len(labels)
                and n_categories <= len(treatment_groups or [])
            )
            if fully_mapped and len(matched) > best_count:
                best_count = len(matched)
                best = {
                    "column": col,
                    "dimension": "group",
                    "labels": cls._ordered_proportion_labels(labels, treatment_groups),
                }

        if best is not None:
            return best

        return {
            "column": None,
            "dimension": "overall",
            "labels": ["整体"],
        }

    @staticmethod
    def _empty_cluster_group_proportions() -> Dict[str, Any]:
        return {
            "group_labels": [],
            "proportions": {},
            "counts": {},
            "column_used": None,
            "dimension": "group",
        }

    async def compute_cluster_group_proportions(
        self,
        state: AnalysisState,
        cluster_key_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.sandbox:
            return self._empty_cluster_group_proportions()

        try:
            meta = await self.introspect_validation_metadata(cluster_key_hint)
            var_name = meta.get("var")
            cluster_key = meta.get("cluster_key")
            if not var_name or not cluster_key:
                return self._empty_cluster_group_proportions()

            grouping = self._select_proportion_grouping(meta, getattr(state, "treatment_groups", []))
            group_col = grouping["column"]
            dimension = grouping["dimension"]
            ordered_labels = grouping["labels"]

            code = (
                "import json as _json\n"
                "import pandas as _pd\n"
                f"_a = {var_name}\n"
                f"_cluster_key = {cluster_key!r}\n"
                f"_group_col = {group_col!r}\n"
                f"_dimension = {dimension!r}\n"
                f"_ordered_labels = {ordered_labels!r}\n"
                "_obs = _a.obs.copy()\n"
                "_clusters = _obs[_cluster_key].astype(str)\n"
                "_counts = {str(_cid): int(_count) for _cid, _count in _clusters.value_counts().items()}\n"
                "if _group_col is not None and _group_col in _obs.columns:\n"
                "    _groups = _obs[_group_col].astype(str)\n"
                "    _tab = _pd.crosstab(_clusters, _groups, normalize='columns') * 100\n"
                "    _tab = _tab.round(1)\n"
                "    _present = [str(_c) for _c in _tab.columns]\n"
                "    _labels = [_l for _l in map(str, _ordered_labels) if _l in _present]\n"
                "    _labels += sorted(_c for _c in _present if _c not in _labels)\n"
                "    _props = {}\n"
                "    for _cid, _row in _tab.iterrows():\n"
                "        _props[str(_cid)] = {str(_label): float(_row.get(_label, 0.0)) for _label in _labels}\n"
                "    _out = {'group_labels': _labels, 'proportions': _props, 'counts': _counts, 'column_used': _group_col, 'dimension': _dimension}\n"
                "else:\n"
                "    _vc = _clusters.value_counts(normalize=True).sort_index() * 100\n"
                "    _props = {str(_cid): {'整体': float(round(_pct, 1))} for _cid, _pct in _vc.items()}\n"
                "    _out = {'group_labels': ['整体'], 'proportions': _props, 'counts': _counts, 'column_used': None, 'dimension': 'overall'}\n"
                "print(_json.dumps(_out, ensure_ascii=False))\n"
            )

            result = await self.sandbox.execute(code, records_history=False, skip_safety_check=True, mutates_namespace=False)
            out = (getattr(result, "output", "") or "").strip()
            line = next((ln for ln in out.splitlines()[::-1] if ln.strip().startswith("{")), None)
            parsed = json.loads(line) if line else {}
            if not isinstance(parsed, dict):
                return self._empty_cluster_group_proportions()
            return {
                "group_labels": list(parsed.get("group_labels") or []),
                "proportions": dict(parsed.get("proportions") or {}),
                "counts": dict(parsed.get("counts") or {}),
                "column_used": parsed.get("column_used"),
                "dimension": parsed.get("dimension") or "group",
            }
        except Exception as e:
            logger.warning(f"[{getattr(self, 'session_id', '?')}] cluster group proportion calculation failed: {e}")
            return self._empty_cluster_group_proportions()

    async def capture_main_object_signature(self, var_name: str, cluster_key: Optional[str]) -> Dict[str, Any]:
        """Capture a CONTENT fingerprint of the MAIN AnnData for tamper detection.

        Keys match annotation_validation.MAIN_OBJECT_SIGNATURE_KEYS. x_sum/x_nnz
        catch in-place X edits; obs_hash/var_hash catch edits to existing columns
        that leave shape/id unchanged. Returns {} on failure.
        """
        if not self.sandbox or not var_name:
            return {}
        ck_repr = repr(cluster_key)
        code = (
            "import json as _json, numpy as _np, pandas as _pd, hashlib as _hashlib\n"
            "from scipy import sparse as _sp\n"
            f"_a = {var_name}\n"
            f"_ck = {ck_repr}\n"
            "_X = _a.X\n"
            "_xsum = float(_X.sum())\n"
            "_xnnz = int(_X.nnz) if _sp.issparse(_X) else int(_np.count_nonzero(_X))\n"
            "_counts = (_a.obs[_ck].value_counts().to_dict() if (_ck and _ck in _a.obs.columns) else {})\n"
            "try: _obsh = str(int(_pd.util.hash_pandas_object(_a.obs, index=True).sum()))\n"
            "except Exception: _obsh = 'na'\n"
            "try: _obsih = str(int(_pd.util.hash_pandas_object(_a.obs.index.to_series(), index=False).sum()))\n"
            "except Exception: _obsih = 'na'\n"
            "try: _varh = str(int(_pd.util.hash_pandas_object(_a.var.index.to_series(), index=False).sum()))\n"
            "except Exception: _varh = 'na'\n"
            "def _series_fp(_s):\n"
            "    try: return _hashlib.sha256(_pd.util.hash_pandas_object(_s, index=True).values.tobytes()).hexdigest()\n"
            "    except Exception: return _hashlib.sha256(repr(_s).encode()).hexdigest()\n"
            "def _array_fp(_v):\n"
            "    _arr = _np.asarray(_v)\n"
            "    if _arr.dtype.hasobject: return _hashlib.sha256(repr(_v).encode()).hexdigest()\n"
            "    _arr = _np.ascontiguousarray(_arr)\n"
            "    _h = _hashlib.sha256()\n"
            "    _h.update(_arr.dtype.str.encode())\n"
            "    _h.update(repr(tuple(_arr.shape)).encode())\n"
            "    _h.update(memoryview(_arr).cast('B'))\n"
            "    return _h.hexdigest()\n"
            "def _value_fp(_v):\n"
            "    try:\n"
            "        _shape = list(getattr(_v, 'shape', ()))\n"
            "        _nnz = int(_v.nnz) if _sp.issparse(_v) else (int(_np.count_nonzero(_v)) if _shape else None)\n"
            "        if _sp.issparse(_v):\n"
            "            _parts = [_array_fp(_v.data)]\n"
            "            for _attr in ('indices', 'indptr', 'row', 'col'):\n"
            "                if hasattr(_v, _attr): _parts.append(_attr + ':' + _array_fp(getattr(_v, _attr)))\n"
            "            _content = _hashlib.sha256('|'.join(_parts).encode()).hexdigest()\n"
            "        elif _shape:\n"
            "            _content = _array_fp(_v)\n"
            "        else:\n"
            "            _content = _hashlib.sha256(repr(_v).encode()).hexdigest()\n"
            "        return {'shape': _shape, 'dtype': str(getattr(_v, 'dtype', '')), 'nnz': _nnz, 'content_sha256': _content}\n"
            "    except Exception: return {'repr': _hashlib.sha256(repr(_v).encode()).hexdigest()}\n"
            "_obsfp = {str(_c): _series_fp(_a.obs[_c]) for _c in _a.obs.columns}\n"
            "_varfp = {str(_c): _series_fp(_a.var[_c]) for _c in _a.var.columns}\n"
            "_rawfp = None if _a.raw is None else {'n_obs': int(_a.raw.n_obs), 'n_vars': int(_a.raw.n_vars), 'x': _value_fp(_a.raw.X), 'var_hash': _series_fp(_a.raw.var.index.to_series())}\n"
            "_lfp = {}\n"
            "for _ln in list(_a.layers.keys()):\n"
            "    _L = _a.layers[_ln]\n"
            "    _lfp[str(_ln)] = _value_fp(_L)\n"
            "_sig = {'id': id(_a), 'n_obs': int(_a.n_obs), 'n_vars': int(_a.n_vars),\n"
            "        'obs_columns': [str(_c) for _c in _a.obs.columns],\n"
            "        'cluster_counts': {str(_k): int(_v) for _k, _v in _counts.items()},\n"
            "        'layers': [str(_l) for _l in list(_a.layers.keys())],\n"
            "        'x_id': id(_X), 'x_sum': _xsum, 'x_nnz': _xnnz, 'x_fingerprint': _value_fp(_X),\n"
            "        'obs_hash': _obsh, 'obs_index_hash': _obsih, 'var_hash': _varh, 'layers_fingerprint': _lfp,\n"
            "        'obs_fingerprints': _obsfp, 'var_fingerprints': _varfp,\n"
            "        'obsm_fingerprints': {str(_k): _value_fp(_v) for _k, _v in _a.obsm.items()},\n"
            "        'varm_fingerprints': {str(_k): _value_fp(_v) for _k, _v in _a.varm.items()},\n"
            "        'obsp_fingerprints': {str(_k): _value_fp(_v) for _k, _v in _a.obsp.items()},\n"
            "        'uns_fingerprints': {str(_k): _value_fp(_v) for _k, _v in _a.uns.items()},\n"
            "        'raw_fingerprint': _rawfp}\n"
            "print(_json.dumps(_sig))\n"
        )
        try:
            result = await self.sandbox.execute(code, records_history=False, skip_safety_check=True, mutates_namespace=False)
            out = (getattr(result, "output", "") or "").strip()
            line = next((ln for ln in out.splitlines()[::-1] if ln.strip().startswith("{")), None)
            parsed = json.loads(line) if line else {}
            return parsed if isinstance(parsed, dict) and "id" in parsed else {}
        except Exception as e:
            logger.warning(f"[{getattr(self, 'session_id', '?')}] signature capture failed: {e}")
            return {}
