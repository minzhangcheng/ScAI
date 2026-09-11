from types import SimpleNamespace

from backend.services.foundation_integrity import (
    accept_foundation_extensions,
    classify_foundation_signature,
    commit_foundation_bindings,
    compare_foundation_signature,
    foundation_write_guard,
    registered_derived_names,
    seal_foundation,
)


def _signature(**updates):
    value = {
        "id": 101, "n_obs": 10, "n_vars": 5,
        "obs_index_hash": "cells", "var_hash": "genes",
        "x_sum": 12.0, "x_nnz": 8,
        "x_fingerprint": {"shape": [10, 5], "sum": 12.0, "nnz": 8},
        "obs_fingerprints": {"cell_type": "types"},
        "var_fingerprints": {},
        "layers_fingerprint": {"counts": {"sum": 20.0, "nnz": 8}},
        "obsm_fingerprints": {"X_umap": {"shape": [10, 2], "sum": 1.0}},
        "varm_fingerprints": {}, "obsp_fingerprints": {},
        "uns_fingerprints": {"neighbors": {"repr": "n"}},
        "raw_fingerprint": None,
    }
    value.update(updates)
    return value


def _state():
    state = SimpleNamespace(sandbox_state={}, current_round=1)
    assert seal_foundation(state, _signature()) is None
    return state


def test_guard_blocks_rebinding_in_assignment_and_loop_target():
    state = _state()
    assert "rebinding" in foundation_write_guard("adata = sc.read_h5ad('other.h5ad')", state)
    assert "loop target" in foundation_write_guard(
        "for label, adata in [('subset', subset)]:\n    print(label)", state
    )


def test_guard_blocks_foundation_mutation_but_allows_new_result_key():
    state = _state()
    assert "normalize_total" in foundation_write_guard("sc.pp.normalize_total(adata)", state)
    assert foundation_write_guard("adata.obs['new_score'] = scores", state) is None
    assert foundation_write_guard("adata.obs['cell_type'] = labels", state)


def test_successful_cross_cell_copy_is_registered_as_derived():
    state = _state()
    code = "work = adata[adata.obs['cell_type'] == 'T'].copy()"
    assert foundation_write_guard(code, state) is None
    commit_foundation_bindings(state, code)
    assert registered_derived_names(state) == {"work"}
    assert foundation_write_guard("sc.pp.log1p(work)\nwork.X = work.X * 2", state) is None
    commit_foundation_bindings(state, "work = unknown_object")
    assert registered_derived_names(state) == set()


def test_guard_respects_local_function_parameter_named_adata():
    state = _state()
    code = (
        "def prepare(adata):\n"
        "    sc.pp.log1p(adata)\n"
        "work = adata.copy()\n"
        "prepare(work)"
    )

    assert foundation_write_guard(code, state) is None
    commit_foundation_bindings(state, code)
    assert registered_derived_names(state) == {"work"}
    assert foundation_write_guard(
        "def mutate_global():\n    sc.pp.log1p(adata)", state
    )


def test_postflight_allows_append_only_extension_then_protects_it():
    state = _state()
    extended = _signature(obs_fingerprints={"cell_type": "types", "new_score": "score-v1"})
    assert compare_foundation_signature(state, extended) == []
    accept_foundation_extensions(state, extended)
    changed = _signature(obs_fingerprints={"cell_type": "types", "new_score": "score-v2"})
    assert compare_foundation_signature(state, changed) == ["obs_fingerprints.new_score"]


def test_postflight_uses_content_identity_across_kernel_restore():
    state = _state()
    assert compare_foundation_signature(state, _signature(id=202)) == []
    assert compare_foundation_signature(state, _signature(x_sum=13.0)) == ["x_sum"]


def test_unavailable_postflight_is_not_persisted_as_confirmed_drift():
    state = _state()

    assert classify_foundation_signature(state, {}) == ("unavailable", [])
    assert "integrity_error" not in state.sandbox_state["stage1_foundation_v1"]
    assert foundation_write_guard("print(adata.shape)", state) is None
