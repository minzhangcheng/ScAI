"""Trusted kernel operations for AnnData checkpoint capture and registry identity.

This mixin performs no scientific judgement.  It serializes a candidate,
reports metadata-only registry identity, and lets the Host decide how to persist
and describe the result.  Callers that already hold ``_exec_lock`` use the
``*_locked`` methods and must never call public ``execute()`` recursively.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .shared import CodeExecutionResult


_SERIALIZE_MARKER = "__SCAI_ADATA_CANDIDATE__:"
_REGISTRY_MARKER = "__SCAI_ADATA_REGISTRY__:"
_REGISTER_MARKER = "__SCAI_ADATA_REGISTRY_UPDATED__:"
_RECONCILE_MARKER = "__SCAI_ADATA_REGISTRY_RECONCILED__:"


def _marker_payload(output: str, marker: str) -> Optional[dict]:
    for line in reversed(str(output or "").splitlines()):
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line[len(marker):])
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None
    return None


class AdataCheckpointMixin:
    """Kernel-side capture, registry and anchor operations."""

    def _get_adata_host_managed_names(self) -> set[str]:
        """Return the Host-managed exact-name backing set.

        The backing field has a distinct name from this accessor so instance
        state can never shadow the method (the original R0.1 regression).
        """
        names = getattr(self, "_adata_host_managed_name_set", None)
        if not isinstance(names, set):
            legacy = self.__dict__.get("_adata_host_managed_names")
            names = set(legacy) if isinstance(legacy, set) else set()
            self._adata_host_managed_name_set = names
        return names

    def remember_host_managed_names(self, names: Iterable[str]) -> None:
        managed = self._get_adata_host_managed_names()
        for name in names:
            text = str(name or "").strip()
            if text.isidentifier():
                managed.add(text)

    def forget_host_managed_names(self, names: Iterable[str]) -> None:
        managed = self._get_adata_host_managed_names()
        for name in names:
            managed.discard(str(name or "").strip())

    def host_managed_global_names(self) -> set[str]:
        return set(self._get_adata_host_managed_names())

    def _checkpoint_candidate_path(self) -> Path:
        root = Path(getattr(self, "_session_tmp"))
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root / f"adata-checkpoint-{uuid.uuid4().hex}.h5ad"

    async def serialize_adata_candidate_locked(
        self, candidate_path: Path,
    ) -> Dict[str, Any]:
        """Serialize live canonical ``adata`` without mutating it."""
        candidate = repr(str(candidate_path))
        marker = repr(_SERIALIZE_MARKER)
        code = f"""
import json as _scai_json
import os as _scai_os
_scai_result = {{"ok": False, "reason": "unknown"}}
try:
    _scai_obj = globals().get("adata")
    if _scai_obj is None or not hasattr(_scai_obj, "write_h5ad"):
        _scai_result = {{"ok": False, "reason": "no_canonical_adata"}}
    elif getattr(_scai_obj, "is_view", False):
        _scai_result = {{"ok": False, "reason": "unsupported_view"}}
    elif getattr(_scai_obj, "isbacked", False):
        _scai_result = {{"ok": False, "reason": "unsupported_backed"}}
    else:
        _scai_obj.write_h5ad(
            {candidate},
            convert_strings_to_categoricals=False,
        )
        _scai_result = {{
            "ok": True,
            "reason": "",
            "n_obs": int(_scai_obj.n_obs),
            "n_vars": int(_scai_obj.n_vars),
            "size_bytes": int(_scai_os.path.getsize({candidate})),
        }}
except Exception as _scai_exc:
    _scai_result = {{"ok": False, "reason": type(_scai_exc).__name__}}
print({marker} + _scai_json.dumps(_scai_result, sort_keys=True))
"""
        result: CodeExecutionResult = await self._execute_unlocked(
            code,
            records_history=False,
            skip_safety_check=True,
        )
        if result.error:
            return {"ok": False, "reason": "kernel_execution_failed"}
        payload = _marker_payload(result.output or "", _SERIALIZE_MARKER)
        if payload is None:
            return {"ok": False, "reason": "missing_capture_marker"}
        return payload

    async def query_adata_registry_locked(
        self,
        *,
        registry_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return metadata-only registry facts; never returns AnnData objects."""
        registry_repr = repr(str(registry_name or ""))
        marker = repr(_REGISTRY_MARKER)
        code = f"""
import json as _scai_json
_scai_name = {registry_repr}
_scai_registry = globals().get(_scai_name) if _scai_name else None
_scai_out = {{
    "registry_name": _scai_name,
    "registry_present": isinstance(_scai_registry, dict),
    "registry_version": (
        int(_scai_registry.get("version", 0))
        if isinstance(_scai_registry, dict) else 0
    ),
    # Canonical presence is a live namespace fact, not a registry fact.  The
    # first Stage-1 capture intentionally has no registry yet.
    "canonical": {{
        "anchor_name": None,
        "binding_identity_status": "unknown",
        "adata_lineage_id": None,
        "ancestor_checkpoint_id": None,
        "same_as_anchor": False,
        "adata_present": globals().get("adata") is not None,
    }},
    "preparations": {{}},
    "bindings": {{}},
}}
if isinstance(_scai_registry, dict):
    _scai_out["preparations"] = _scai_registry.get("preparations") or {{}}
    _scai_out["bindings"] = _scai_registry.get("bindings") or {{}}
    _scai_canonical = _scai_registry.get("canonical") or {{}}
    _scai_anchor = _scai_canonical.get("anchor_name")
    _scai_out["canonical"] = {{
        "anchor_name": _scai_anchor,
        "binding_identity_status": _scai_canonical.get(
            "binding_identity_status", "unknown",
        ),
        "adata_lineage_id": _scai_canonical.get("adata_lineage_id"),
        "ancestor_checkpoint_id": _scai_canonical.get("ancestor_checkpoint_id"),
        "same_as_anchor": bool(
            _scai_anchor
            and globals().get(_scai_anchor) is globals().get("adata")
        ),
        "adata_present": globals().get("adata") is not None,
    }}
print({marker} + _scai_json.dumps(_scai_out, sort_keys=True))
"""
        result = await self._execute_unlocked(
            code,
            records_history=False,
            skip_safety_check=True,
        )
        if result.error:
            return {
                "registry_name": str(registry_name or ""),
                "registry_present": False,
                "error": "kernel_execution_failed",
            }
        payload = _marker_payload(result.output or "", _REGISTRY_MARKER)
        if payload is None:
            return {
                "registry_name": str(registry_name or ""),
                "registry_present": False,
                "error": "missing_registry_marker",
            }
        return payload

    async def register_canonical_anchor_locked(
        self,
        *,
        registry_name: str,
        anchor_name: str,
        adata_lineage_id: str,
        ancestor_checkpoint_id: Optional[str],
    ) -> Dict[str, Any]:
        """Install a metadata-only registry and a top-level canonical anchor."""
        registry_repr = repr(str(registry_name))
        anchor_repr = repr(str(anchor_name))
        lineage_repr = repr(str(adata_lineage_id))
        ancestor_repr = repr(ancestor_checkpoint_id)
        marker = repr(_REGISTER_MARKER)
        code = f"""
import json as _scai_json
_scai_registry_name = {registry_repr}
_scai_anchor_name = {anchor_repr}
_scai_lineage = {lineage_repr}
_scai_ancestor = {ancestor_repr}
_scai_out = {{"ok": False, "reason": "unknown"}}
try:
    _scai_adata = globals().get("adata")
    if _scai_adata is None:
        _scai_out = {{"ok": False, "reason": "no_canonical_adata"}}
    else:
        _scai_existing = globals().get(_scai_anchor_name)
        if _scai_existing is not None and _scai_existing is not _scai_adata:
            _scai_out = {{"ok": False, "reason": "anchor_collision"}}
        else:
            _scai_registry = globals().get(_scai_registry_name)
            if _scai_registry is None:
                _scai_registry = {{
                    "version": 1,
                    "canonical": {{}},
                    "preparations": {{}},
                    "bindings": {{}},
                }}
                globals()[_scai_registry_name] = _scai_registry
            if (
                not isinstance(_scai_registry, dict)
                or int(_scai_registry.get("version", 0)) != 1
            ):
                _scai_out = {{"ok": False, "reason": "registry_unavailable"}}
            else:
                globals()[_scai_anchor_name] = _scai_adata
                _scai_registry["canonical"] = {{
                    "anchor_name": _scai_anchor_name,
                    "binding_identity_status": "exact",
                    "adata_lineage_id": _scai_lineage,
                    "ancestor_checkpoint_id": _scai_ancestor,
                }}
                _scai_out = {{
                    "ok": True,
                    "registry_name": _scai_registry_name,
                    "anchor_name": _scai_anchor_name,
                }}
except Exception as _scai_exc:
    _scai_out = {{"ok": False, "reason": type(_scai_exc).__name__}}
print({marker} + _scai_json.dumps(_scai_out, sort_keys=True))
"""
        result = await self._execute_unlocked(
            code,
            records_history=False,
            skip_safety_check=True,
        )
        if result.error:
            return {"ok": False, "reason": "kernel_execution_failed"}
        payload = _marker_payload(result.output or "", _REGISTER_MARKER)
        if payload is None:
            return {"ok": False, "reason": "missing_registration_marker"}
        if payload.get("ok"):
            self.remember_host_managed_names([registry_name, anchor_name])
        return payload


    async def apply_adata_source_locked(
        self,
        *,
        mode: str,
        registry_name: str,
        preparation_id: str,
        application_token: str,
        scope_id: str,
        owner_kind: str,
        owner_id: str,
        selection_signature: str,
        session_lineage_id: str,
        checkpoint_path: Optional[str] = None,
        expected_n_obs: Optional[int] = None,
        expected_n_vars: Optional[int] = None,
        adata_lineage_id: Optional[str] = None,
        ancestor_checkpoint_id: Optional[str] = None,
        binding_id: Optional[str] = None,
        binding_name: Optional[str] = None,
        anchor_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply one Host-resolved source to the live namespace.

        The caller holds ``_exec_lock`` and owns all identifiers.  The kernel
        validates H5AD readability and recorded shape before any user-visible
        binding changes.
        """
        marker = repr(_REGISTER_MARKER)
        mode_repr = repr(str(mode))
        registry_repr = repr(str(registry_name))
        preparation_repr = repr(str(preparation_id))
        token_repr = repr(str(application_token))
        scope_repr = repr(str(scope_id))
        owner_kind_repr = repr(str(owner_kind))
        owner_id_repr = repr(str(owner_id))
        selection_repr = repr(str(selection_signature))
        lineage_repr = repr(str(session_lineage_id))
        checkpoint_repr = repr(checkpoint_path)
        expected_n_obs_repr = repr(expected_n_obs)
        expected_n_vars_repr = repr(expected_n_vars)
        adata_lineage_repr = repr(adata_lineage_id)
        ancestor_repr = repr(ancestor_checkpoint_id)
        binding_id_repr = repr(binding_id)
        binding_name_repr = repr(binding_name)
        anchor_repr = repr(anchor_name)
        memory_limit = (getattr(self, "runtime_budgets", None) or {}).get(
            "max_rss_bytes"
        )
        memory_limit_repr = repr(
            int(memory_limit) if memory_limit and int(memory_limit) > 0 else None
        )
        code = f"""
import json as _scai_json
import copy as _scai_copy
_scai_mode = {mode_repr}
_scai_registry_name = {registry_repr}
_scai_preparation_id = {preparation_repr}
_scai_token = {token_repr}
_scai_scope = {scope_repr}
_scai_owner_kind = {owner_kind_repr}
_scai_owner_id = {owner_id_repr}
_scai_selection = {selection_repr}
_scai_session_lineage = {lineage_repr}
_scai_checkpoint = {checkpoint_repr}
_scai_expected_n_obs = {expected_n_obs_repr}
_scai_expected_n_vars = {expected_n_vars_repr}
_scai_adata_lineage = {adata_lineage_repr}
_scai_ancestor = {ancestor_repr}
_scai_binding_id = {binding_id_repr}
_scai_binding_name = {binding_name_repr}
_scai_anchor_name = {anchor_repr}
_scai_memory_limit = {memory_limit_repr}
_scai_out = {{"ok": False, "reason": "unknown"}}
_scai_missing = object()
_scai_targets = {{
    _name for _name in (
        _scai_registry_name, "adata", _scai_binding_name, _scai_anchor_name,
    ) if isinstance(_name, str) and _name
}}
_scai_previous = {{
    _name: globals().get(_name, _scai_missing) for _name in _scai_targets
}}
_scai_registry_snapshot = (
    _scai_copy.deepcopy(_scai_previous.get(_scai_registry_name))
    if isinstance(_scai_previous.get(_scai_registry_name), dict)
    else None
)

def _scai_rollback():
    for _name, _old in _scai_previous.items():
        if _old is _scai_missing:
            globals().pop(_name, None)
        elif (
            _name == _scai_registry_name
            and isinstance(_old, dict)
            and isinstance(_scai_registry_snapshot, dict)
        ):
            _old.clear()
            _old.update(_scai_registry_snapshot)
            globals()[_name] = _old
        else:
            globals()[_name] = _old
try:
    import anndata as _scai_ad

    def _scai_registry():
        _reg = globals().get(_scai_registry_name)
        if _reg is None:
            _reg = {{
                "version": 1,
                "canonical": {{}},
                "preparations": {{}},
                "bindings": {{}},
            }}
            globals()[_scai_registry_name] = _reg
        if not isinstance(_reg, dict) or int(_reg.get("version", 0)) != 1:
            raise RuntimeError("registry_unavailable")
        return _reg

    def _scai_store_preparation(_reg, _binding_id=None, _binding_name=None,
                                _anchor_name=None, _adata_lineage=None,
                                _ancestor=None):
        _reg.setdefault("preparations", {{}})[_scai_preparation_id] = {{
            "token": _scai_token,
            "scope_id": _scai_scope,
            "owner_kind": _scai_owner_kind,
            "owner_id": _scai_owner_id,
            "selection_signature": _scai_selection,
            "session_lineage_id": _scai_session_lineage,
            "binding_id": _binding_id,
            "binding_name": _binding_name,
            "anchor_name": _anchor_name,
            "adata_lineage_id": _adata_lineage,
            "ancestor_checkpoint_id": _ancestor,
            "binding_identity_status": "exact",
        }}

    if _scai_mode in ("checkpoint_copy", "checkpoint_branch"):
        if not _scai_checkpoint:
            _scai_out = {{"ok": False, "reason": "checkpoint_path_missing"}}
        else:
            _scai_estimated_materialized_box = [0]
            _scai_estimated_materialized = 0
            _scai_current_rss = 0
            try:
                import h5py as _scai_h5py
                import os as _scai_os
                with _scai_h5py.File(_scai_checkpoint, "r") as _scai_h5:
                    def _scai_count_dataset(_name, _obj):
                        if isinstance(_obj, _scai_h5py.Dataset):
                            _scai_estimated_materialized_box[0] += int(
                                _obj.size * max(int(_obj.dtype.itemsize), 1)
                            )
                    _scai_h5.visititems(_scai_count_dataset)
                _scai_file_bytes = int(_scai_os.path.getsize(_scai_checkpoint))
                _scai_estimated_materialized = max(
                    _scai_file_bytes,
                    int(_scai_estimated_materialized_box[0] * 1.25),
                )
                with open("/proc/self/statm", "r", encoding="utf-8") as _scai_statm:
                    _scai_current_rss = (
                        int(_scai_statm.read().split()[1])
                        * int(_scai_os.sysconf("SC_PAGE_SIZE"))
                    )
            except Exception:
                _scai_estimated_materialized = 0
                _scai_current_rss = 0
            if (
                _scai_memory_limit is not None
                and (
                    not _scai_estimated_materialized
                    or not _scai_current_rss
                    or _scai_current_rss + _scai_estimated_materialized
                    > int(_scai_memory_limit)
                )
            ):
                _scai_out = {{
                    "ok": False,
                    "reason": "insufficient_memory_headroom",
                    "current_rss_bytes": int(_scai_current_rss),
                    "estimated_materialized_bytes": int(
                        _scai_estimated_materialized
                    ),
                    "memory_limit_bytes": int(_scai_memory_limit),
                }}
                _scai_obj = None
            else:
                _scai_obj = _scai_ad.read_h5ad(_scai_checkpoint)
            if _scai_obj is None:
                pass
            elif getattr(_scai_obj, "isbacked", False):
                _scai_out = {{"ok": False, "reason": "backed_object_not_allowed"}}
            elif (
                _scai_expected_n_obs is not None
                and int(_scai_obj.n_obs) != int(_scai_expected_n_obs)
            ) or (
                _scai_expected_n_vars is not None
                and int(_scai_obj.n_vars) != int(_scai_expected_n_vars)
            ):
                _scai_out = {{"ok": False, "reason": "shape_mismatch"}}
            elif _scai_mode == "checkpoint_copy":
                if not _scai_binding_name or not _scai_anchor_name:
                    _scai_out = {{"ok": False, "reason": "binding_name_missing"}}
                elif (
                    _scai_binding_name in globals()
                    and globals()[_scai_binding_name] is not _scai_obj
                ):
                    _scai_out = {{"ok": False, "reason": "binding_name_collision"}}
                elif (
                    _scai_anchor_name in globals()
                    and globals()[_scai_anchor_name] is not _scai_obj
                ):
                    _scai_out = {{"ok": False, "reason": "anchor_collision"}}
                else:
                    _reg = _scai_registry()
                    globals()[_scai_binding_name] = _scai_obj
                    globals()[_scai_anchor_name] = _scai_obj
                    _reg.setdefault("bindings", {{}})[_scai_binding_id] = {{
                        "anchor_name": _scai_anchor_name,
                        "name": _scai_binding_name,
                        "session_lineage_id": _scai_session_lineage,
                        "adata_lineage_id": _scai_adata_lineage,
                        "ancestor_checkpoint_id": _scai_ancestor,
                        "status": "registered",
                    }}
                    _scai_store_preparation(
                        _reg, _scai_binding_id, _scai_binding_name,
                        _scai_anchor_name, _scai_adata_lineage, _scai_ancestor,
                    )
                    _scai_out = {{
                        "ok": True,
                        "mode": _scai_mode,
                        "binding_name": _scai_binding_name,
                        "anchor_name": _scai_anchor_name,
                        "adata_lineage_id": _scai_adata_lineage,
                        "n_obs": int(_scai_obj.n_obs),
                        "n_vars": int(_scai_obj.n_vars),
                    }}
            else:  # checkpoint_branch
                if not _scai_anchor_name:
                    _scai_out = {{"ok": False, "reason": "anchor_name_missing"}}
                elif (
                    _scai_anchor_name in globals()
                    and globals()[_scai_anchor_name] is not _scai_obj
                ):
                    _scai_out = {{"ok": False, "reason": "anchor_collision"}}
                else:
                    _reg = _scai_registry()
                    globals()["adata"] = _scai_obj
                    globals()[_scai_anchor_name] = _scai_obj
                    _reg["canonical"] = {{
                        "anchor_name": _scai_anchor_name,
                        "binding_identity_status": "exact",
                        "adata_lineage_id": _scai_adata_lineage,
                        "ancestor_checkpoint_id": _scai_ancestor,
                    }}
                    _scai_store_preparation(
                        _reg, None, None, _scai_anchor_name,
                        _scai_adata_lineage, _scai_ancestor,
                    )
                    _scai_out = {{
                        "ok": True,
                        "mode": _scai_mode,
                        "anchor_name": _scai_anchor_name,
                        "adata_lineage_id": _scai_adata_lineage,
                        "n_obs": int(_scai_obj.n_obs),
                        "n_vars": int(_scai_obj.n_vars),
                    }}
    elif _scai_mode in ("live_binding", "promote_live"):
        _reg = _scai_registry()
        _binding = (_reg.get("bindings") or {{}}).get(_scai_binding_id) or {{}}
        _binding_anchor = _binding.get("anchor_name")
        _binding_name = _binding.get("name")
        _anchor_obj = globals().get(_binding_anchor) if _binding_anchor else None
        _named_obj = globals().get(_binding_name) if _binding_name else None
        if _anchor_obj is None or _anchor_obj is not _named_obj:
            _scai_out = {{"ok": False, "reason": "binding_stale"}}
        else:
            _adata_lineage = _binding.get("adata_lineage_id")
            _ancestor = _binding.get("ancestor_checkpoint_id")
            if _scai_mode == "promote_live":
                if not _scai_anchor_name:
                    _scai_out = {{"ok": False, "reason": "anchor_name_missing"}}
                elif (
                    _scai_anchor_name in globals()
                    and globals()[_scai_anchor_name] is not _anchor_obj
                ):
                    _scai_out = {{"ok": False, "reason": "anchor_collision"}}
                else:
                    globals()["adata"] = _anchor_obj
                    globals()[_scai_anchor_name] = _anchor_obj
                    _reg["canonical"] = {{
                        "anchor_name": _scai_anchor_name,
                        "binding_identity_status": "exact",
                        "adata_lineage_id": _adata_lineage,
                        "ancestor_checkpoint_id": _ancestor,
                    }}
                    _scai_store_preparation(
                        _reg, _scai_binding_id, _binding_name,
                        _scai_anchor_name, _adata_lineage, _ancestor,
                    )
                    _scai_out = {{
                        "ok": True,
                        "mode": _scai_mode,
                        "binding_name": _binding_name,
                        "anchor_name": _scai_anchor_name,
                        "adata_lineage_id": _adata_lineage,
                        "n_obs": int(_anchor_obj.n_obs),
                        "n_vars": int(_anchor_obj.n_vars),
                    }}
            else:
                _scai_store_preparation(
                    _reg, _scai_binding_id, _binding_name,
                    _binding_anchor, _adata_lineage, _ancestor,
                )
                _scai_out = {{
                    "ok": True,
                    "mode": _scai_mode,
                    "binding_name": _binding_name,
                    "anchor_name": _binding_anchor,
                    "adata_lineage_id": _adata_lineage,
                    "n_obs": int(_anchor_obj.n_obs),
                    "n_vars": int(_anchor_obj.n_vars),
                }}
    else:
        _scai_out = {{"ok": False, "reason": "unsupported_mode"}}
    if not _scai_out.get("ok"):
        _scai_rollback()
except Exception as _scai_exc:
    _scai_rollback()
    _scai_out = {{"ok": False, "reason": type(_scai_exc).__name__}}
print({marker} + _scai_json.dumps(_scai_out, sort_keys=True))
"""
        result = await self._execute_unlocked(
            code,
            records_history=False,
            skip_safety_check=True,
        )
        if result.error:
            return {"ok": False, "reason": "kernel_execution_failed"}
        payload = _marker_payload(result.output or "", _REGISTER_MARKER)
        if payload is None:
            return {"ok": False, "reason": "missing_registration_marker"}
        if payload.get("ok"):
            names = [str(registry_name)]
            for candidate in (anchor_name, binding_name):
                if candidate:
                    names.append(str(candidate))
            self.remember_host_managed_names(names)
        return payload


    async def reconcile_adata_registry_after_execution(
        self,
        *,
        registry_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify canonical/preparation/binding identity after a cell.

        This lock-owning wrapper is safe to call after ``sandbox.execute()`` has
        returned.  It never rejects the cell, reloads data or judges scientific
        mutations; it only repairs registry/status facts and releases obsolete
        anchors whose final live role is gone.
        """
        lock = getattr(self, "_exec_lock", None)
        if lock is None:
            return await self._reconcile_adata_registry_locked(
                registry_name=registry_name
            )
        async with lock:
            return await self._reconcile_adata_registry_locked(
                registry_name=registry_name
            )

    async def rebase_adata_registry_lineage_locked(
        self,
        *,
        registry_name: str,
        preparation_lineages: Dict[str, Dict[str, str]],
        binding_lineages: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """Atomically acknowledge verified child ownership in the registry."""
        registry_repr = repr(str(registry_name))
        preparation_repr = repr(json.dumps(preparation_lineages, sort_keys=True))
        binding_repr = repr(json.dumps(binding_lineages, sort_keys=True))
        marker = repr(_REGISTER_MARKER)
        code = f"""
import copy as _scai_copy
import json as _scai_json
_scai_registry_name = {registry_repr}
_scai_preparation_lineages = _scai_json.loads({preparation_repr})
_scai_binding_lineages = _scai_json.loads({binding_repr})
_scai_registry = globals().get(_scai_registry_name)
_scai_out = {{"ok": False, "reason": "registry_unavailable"}}
if isinstance(_scai_registry, dict):
    _scai_before = _scai_copy.deepcopy(_scai_registry)
    try:
        _scai_preparations = _scai_registry.get("preparations") or {{}}
        _scai_bindings = _scai_registry.get("bindings") or {{}}
        for _scai_id, _scai_transition in _scai_preparation_lineages.items():
            _scai_entry = _scai_preparations.get(_scai_id)
            if not isinstance(_scai_entry, dict) or str(
                _scai_entry.get("session_lineage_id") or "main"
            ) not in {{
                str(_scai_transition.get("from") or "main"),
                str(_scai_transition.get("to") or "main"),
            }}:
                raise RuntimeError("preparation_lineage_mismatch")
        for _scai_id, _scai_transition in _scai_binding_lineages.items():
            _scai_entry = _scai_bindings.get(_scai_id)
            if not isinstance(_scai_entry, dict) or str(
                _scai_entry.get("session_lineage_id") or "main"
            ) not in {{
                str(_scai_transition.get("from") or "main"),
                str(_scai_transition.get("to") or "main"),
            }}:
                raise RuntimeError("binding_lineage_mismatch")
        for _scai_id, _scai_transition in _scai_preparation_lineages.items():
            _scai_preparations[_scai_id]["session_lineage_id"] = str(
                _scai_transition.get("to") or "main"
            )
        for _scai_id, _scai_transition in _scai_binding_lineages.items():
            _scai_bindings[_scai_id]["session_lineage_id"] = str(
                _scai_transition.get("to") or "main"
            )
        _scai_out = {{"ok": True}}
    except Exception as _scai_exc:
        _scai_registry.clear()
        _scai_registry.update(_scai_before)
        _scai_out = {{"ok": False, "reason": str(_scai_exc)}}
print({marker} + _scai_json.dumps(_scai_out, sort_keys=True))
"""
        result = await self._execute_unlocked(
            code,
            records_history=False,
            skip_safety_check=True,
        )
        if result.error:
            return {"ok": False, "reason": "kernel_execution_failed"}
        payload = _marker_payload(result.output or "", _REGISTER_MARKER)
        return payload or {"ok": False, "reason": "missing_registration_marker"}

    async def _reconcile_adata_registry_locked(
        self,
        *,
        registry_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        registry_repr = repr(str(registry_name or ""))
        marker = repr(_RECONCILE_MARKER)
        code = f"""
import json as _scai_json
_scai_reg_name = {registry_repr}
_scai_out = {{"ok": False, "changed": False, "reason": "unknown"}}
try:
    _scai_reg = globals().get(_scai_reg_name) if _scai_reg_name else None
    if not isinstance(_scai_reg, dict):
        _scai_out = {{"ok": True, "changed": False, "reason": "registry_absent"}}
    else:
        _scai_canonical = _scai_reg.setdefault("canonical", {{}})
        _scai_anchor = _scai_canonical.get("anchor_name")
        _scai_adata = globals().get("adata")
        _scai_named = globals().get(_scai_anchor) if _scai_anchor else None
        if _scai_adata is None:
            _scai_status = "missing"
        elif _scai_anchor and _scai_named is _scai_adata:
            _scai_status = "exact"
        else:
            _scai_status = "evolved"
        _scai_changed = False
        if _scai_canonical.get("binding_identity_status") != _scai_status:
            _scai_changed = True
        _scai_canonical["binding_identity_status"] = _scai_status
        _scai_prep_statuses = {{}}
        _scai_binding_statuses = {{}}
        _scai_released = []
        _scai_release_candidates = set()
        if _scai_status != "exact" and _scai_anchor:
            _scai_release_candidates.add(_scai_anchor)
            _scai_canonical["anchor_name"] = None

        # Reconcile every preparation independently.  A derived working
        # binding may be rebound or deleted while canonical `adata` remains
        # exact, so canonical status cannot gate this scan.
        for _scai_pid, _scai_prep in list((_scai_reg.get("preparations") or {{}}).items()):
            _scai_prep_anchor = _scai_prep.get("anchor_name")
            _scai_prep_binding = _scai_prep.get("binding_name")
            _scai_prep_anchor_obj = (
                globals().get(_scai_prep_anchor) if _scai_prep_anchor else None
            )
            if _scai_prep_binding:
                _scai_prep_live = globals().get(_scai_prep_binding)
            else:
                _scai_prep_live = globals().get("adata")
            if _scai_prep_live is None:
                _scai_prep_status = "missing"
            elif _scai_prep_anchor and _scai_prep_anchor_obj is _scai_prep_live:
                _scai_prep_status = "exact"
            else:
                _scai_prep_status = "evolved"
            if _scai_prep.get("binding_identity_status") != _scai_prep_status:
                _scai_changed = True
            _scai_prep["binding_identity_status"] = _scai_prep_status
            _scai_prep_statuses[_scai_pid] = _scai_prep_status
            if _scai_prep_status != "exact" and _scai_prep_anchor:
                _scai_release_candidates.add(_scai_prep_anchor)
                _scai_prep["anchor_name"] = None

        for _scai_bid, _scai_binding in list((_scai_reg.get("bindings") or {{}}).items()):
            _scai_binding_anchor = _scai_binding.get("anchor_name")
            _scai_binding_name = _scai_binding.get("name")
            _scai_binding_anchor_obj = (
                globals().get(_scai_binding_anchor) if _scai_binding_anchor else None
            )
            _scai_binding_obj = (
                globals().get(_scai_binding_name) if _scai_binding_name else None
            )
            _scai_binding_status = (
                "registered"
                if _scai_binding_anchor_obj is not None
                and _scai_binding_anchor_obj is _scai_binding_obj
                else "stale"
            )
            if _scai_binding.get("status") != _scai_binding_status:
                _scai_changed = True
            _scai_binding["status"] = _scai_binding_status
            _scai_binding_statuses[_scai_bid] = _scai_binding_status
            if _scai_binding_status != "registered" and _scai_binding_anchor:
                _scai_release_candidates.add(_scai_binding_anchor)
                _scai_binding["anchor_name"] = None

        # Release only anchors no retained role still references.
        for _scai_candidate in sorted(_scai_release_candidates):
            _scai_referenced = (
                (_scai_reg.get("canonical") or {{}}).get("anchor_name")
                == _scai_candidate
            )
            if not _scai_referenced:
                for _scai_ref in list((_scai_reg.get("preparations") or {{}}).values()) + list((_scai_reg.get("bindings") or {{}}).values()):
                    if _scai_ref.get("anchor_name") == _scai_candidate:
                        _scai_referenced = True
                        break
            if not _scai_referenced and _scai_candidate in globals():
                del globals()[_scai_candidate]
                _scai_released.append(_scai_candidate)
                _scai_changed = True
        _scai_out = {{
            "ok": True,
            "changed": _scai_changed,
            "canonical_status": _scai_status,
            "preparation_statuses": _scai_prep_statuses,
            "binding_statuses": _scai_binding_statuses,
            "released_anchors": _scai_released,
        }}
except Exception as _scai_exc:
    _scai_out = {{"ok": False, "changed": False, "reason": type(_scai_exc).__name__}}
print({marker} + _scai_json.dumps(_scai_out, sort_keys=True))
"""
        result = await self._execute_unlocked(
            code,
            records_history=False,
            skip_safety_check=True,
        )
        if result.error:
            return {"ok": False, "changed": False, "reason": "kernel_execution_failed"}
        payload = _marker_payload(result.output or "", _RECONCILE_MARKER)
        if payload is None:
            return {"ok": False, "changed": False, "reason": "missing_reconcile_marker"}
        for anchor in payload.get("released_anchors") or []:
            self.forget_host_managed_names([anchor])
        return payload
