import asyncio
from hashlib import sha256
import os
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from backend.models.schemas import (
    AnalysisState, AnalysisStep, PublicationFigure, FigureReviewStatus,
    PublicationTable,
)
from backend.utils.expression_contract import (
    build_expression_runtime_code,
    build_expression_preflight_code,
    describe_invalidated_artifacts,
    extract_sensitive_expression_inputs,
    instrument_expression_contract_calls,
    mutates_expression_representation,
    prepares_expression_before_sensitive_call,
    record_executed_representation_change,
    record_expression_contract,
    record_representation_change,
    record_round_start,
    round_start_ns,
    unchanged_artifact_stems,
    workspace_artifact_versions,
)


def test_generated_expression_contract_source_is_byte_stable():
    specs = [{"name": "adata", "layer": None}]
    assert sha256(build_expression_preflight_code(specs).encode()).hexdigest() == (
        "67ffbbe6833c9983858e4398020174402476bc7b811f3d03af7d9d7ee2fb7f7f"
    )
    assert sha256(build_expression_runtime_code().encode()).hexdigest() == (
        "2a3d965880094a6fdf265d9ba2b3fc6ffc98e127274c34e7f5ca1bcf051c994f"
    )


def test_extracts_scanpy_and_nmf_expression_inputs():
    code = """
X = adata.X
sc.tl.rank_genes_groups(adata, groupby='group')
model = NMF(n_components=3)
W = model.fit_transform(X)
"""
    specs = extract_sensitive_expression_inputs(code)
    assert {spec["name"] for spec in specs} == {"adata"}


def test_detects_expression_representation_mutation():
    assert mutates_expression_representation("sc.pp.normalize_total(work)\nsc.pp.log1p(work)")
    assert mutates_expression_representation("adata.X = corrected")
    assert not mutates_expression_representation("sc.tl.umap(adata)")
    assert prepares_expression_before_sensitive_call(
        "sc.pp.normalize_total(adata)\nsc.pp.log1p(adata)\n"
        "sc.tl.rank_genes_groups(adata, groupby='g')"
    )
    assert not prepares_expression_before_sensitive_call(
        "sc.tl.rank_genes_groups(adata, groupby='g')\nsc.pp.log1p(adata)"
    )


def test_runtime_preflight_rejects_raw_counts_and_accepts_log_values():
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])
    with pytest.raises(ValueError, match="raw integer counts"):
        exec(guard, {"adata": SimpleNamespace(X=np.array([[0, 1], [2, 5]]), shape=(2, 2))})

    namespace = {
        "adata": SimpleNamespace(
            X=np.array([[0.0, np.log1p(3.0)], [np.log1p(8.0), 0.0]]),
            shape=(2, 2),
        )
    }
    exec(guard, namespace)


def _execute_instrumented(code, namespace):
    instrumented, count = instrument_expression_contract_calls(code)
    assert count > 0
    exec(build_expression_runtime_code(), namespace)
    exec(instrumented, namespace)
    return instrumented, count


@pytest.mark.parametrize("lookup", [
    "dir", "names", "builtins.dir", "getattr(builtins, 'dir')", "lookups['dir']",
])
def test_namespace_lookup_reuses_existing_adata(lookup):
    adata = SimpleNamespace(shape=(3, 2))
    namespace = {"adata": adata}
    _execute_instrumented(
        "import builtins\n"
        "names = dir\n"
        "lookups = {'dir': dir}\n"
        f"found = 'adata' in {lookup}()\n"
        "if not found:\n"
        "    raise RuntimeError('Would reload raw data and discard the existing atlas')\n"
        "result = adata.shape\n",
        namespace,
    )
    assert namespace["found"] is True
    assert namespace["adata"] is adata
    assert namespace["result"] == (3, 2)


def test_namespace_builtins_preserve_function_scope_and_missing_names():
    namespace = {"global_marker": object()}
    _execute_instrumented(
        "def inspect_scope(adata):\n"
        "    local_marker = 7\n"
        "    names = dir\n"
        "    local_values = locals\n"
        "    return (\n"
        "        'adata' in names(),\n"
        "        local_values()['adata'] is adata,\n"
        "        vars()['local_marker'],\n"
        "        globals()['global_marker'] is global_marker,\n"
        "    )\n"
        "result = inspect_scope(object())\n"
        "missing_at_module_scope = 'adata' not in dir()\n",
        namespace,
    )
    assert namespace["result"] == (True, True, 7, True)
    assert namespace["missing_at_module_scope"] is True


def test_zero_argument_super_retains_the_method_frame():
    namespace = {}
    _execute_instrumented(
        "class Parent:\n"
        "    def value(self):\n"
        "        return 10\n"
        "class Child(Parent):\n"
        "    def value(self):\n"
        "        parent = super()\n"
        "        return parent.value() + 1\n"
        "result = Child().value()\n",
        namespace,
    )
    assert namespace["result"] == 11


def test_builtin_name_rebound_to_sensitive_callable_still_validates_input():
    calls = []
    raw = SimpleNamespace(X=np.array([[0, 4]]), shape=(1, 2), layers={})
    namespace = {
        "adata": raw,
        "sc": SimpleNamespace(tl=SimpleNamespace(score_genes=lambda obj: calls.append(obj))),
    }
    with pytest.raises(ValueError, match="raw integer counts"):
        _execute_instrumented("dir = sc.tl.score_genes\ndir(adata)", namespace)
    assert calls == []


def test_instrumentation_does_not_wrap_resolved_calls_again():
    code = "result = max(value for value in values)\nnames = dir()"
    once, count = instrument_expression_contract_calls(code)
    twice, second_count = instrument_expression_contract_calls(once)
    assert count == 2
    assert second_count == 0
    assert twice == once


@pytest.mark.parametrize("preinstall_runtime", [False, True])
def test_shared_execute_wrapper_reuses_data_and_saves_plot(tmp_path, preinstall_runtime):
    from backend.agents.bioinformatics.kernel_templates import _build_exec_autosave_wrapper
    import matplotlib.pyplot as plt

    adata = SimpleNamespace(shape=(3, 2))
    namespace = {
        "adata": adata,
        "_original_plt_close": plt.close,
        "_original_plt_show": plt.show,
    }
    source, _ = instrument_expression_contract_calls(
        "if 'adata' not in dir():\n"
        "    raise RuntimeError('Existing atlas incorrectly reported missing')\n"
        "result = adata.shape\n"
        "plt.figure()\n"
        "plt.plot([0, 1], [0, 1])\n"
    )
    if preinstall_runtime:
        exec(build_expression_runtime_code(), namespace)
    else:
        source = build_expression_runtime_code() + "\n" + source
    wrapper = _build_exec_autosave_wrapper(source, str(tmp_path), "namespace_regression")
    try:
        exec(wrapper, namespace)
        assert namespace["adata"] is adata
        assert namespace["result"] == (3, 2)
        assert list(tmp_path.glob("*.jpg"))
        assert list(tmp_path.glob("*.pdf"))
    finally:
        plt.close("all")


def test_runtime_contract_checks_new_same_cell_copy_not_stale_global():
    calls = []

    class Source:
        def __init__(self):
            self.copy_count = 0

        def copy(self):
            self.copy_count += 1
            return SimpleNamespace(
                X=np.array([[0, 1], [2, 5]]), shape=(2, 2), layers={}
            )

    source = Source()
    stale = SimpleNamespace(
        X=np.array([[0.0, np.log1p(2.0)]]), shape=(1, 2), layers={}
    )
    namespace = {
        "source": source,
        "work": stale,
        "sc": SimpleNamespace(
            tl=SimpleNamespace(rank_genes_groups=lambda obj, **kwargs: calls.append(obj))
        ),
    }

    with pytest.raises(ValueError, match="raw integer counts"):
        _execute_instrumented(
            "work = source.copy()\nsc.tl.rank_genes_groups(work, groupby='g')",
            namespace,
        )

    assert source.copy_count == 1
    assert calls == []
    assert namespace["work"] is not stale


def test_runtime_contract_forwards_actual_object_and_preserves_side_effects():
    calls = []
    counters = {"object": 0, "layer": 0}
    matrix = np.array([[0.0, np.log1p(3.0)], [np.log1p(8.0), 0.0]])
    obj = SimpleNamespace(X=matrix, shape=matrix.shape, layers={"log": matrix})

    def next_object():
        counters["object"] += 1
        return obj

    def next_layer():
        counters["layer"] += 1
        return "log"

    def score_genes(adata=None, *, gene_list, layer=None):
        calls.append((adata, tuple(gene_list), layer))

    namespace = {
        "items": [obj],
        "next_object": next_object,
        "next_layer": next_layer,
        "score": score_genes,
        "sc": SimpleNamespace(tl=SimpleNamespace(score_genes=score_genes)),
    }
    code = (
        "alias = sc.tl.score_genes\n"
        "alias(adata=next_object(), gene_list=['A'], layer=next_layer())\n"
        "sc.tl.score_genes(items[0], gene_list=['B'])"
    )
    _, count = _execute_instrumented(code, namespace)

    # Every call expression passes through the runtime dispatcher so aliases
    # cannot evade classification.  The count is therefore the number of
    # wrappers, not a static guess at how many calls are sensitive.
    assert count == 4
    assert counters == {"object": 1, "layer": 1}
    assert calls == [(obj, ("A",), "log"), (obj, ("B",), None)]


def test_runtime_contract_instruments_nmf_and_evaluates_matrix_once():
    seen = []
    counter = {"matrix": 0}
    matrix = np.array([[0.0, np.log1p(2.0)]])

    def make_matrix():
        counter["matrix"] += 1
        return matrix

    class NMF:
        def fit_transform(self, value):
            seen.append(value)
            return value

    namespace = {"NMF": NMF, "make_matrix": make_matrix}
    _, count = _execute_instrumented(
        "model = NMF()\nresult = model.fit_transform(make_matrix())", namespace
    )

    assert count == 3
    assert counter["matrix"] == 1
    assert seen == [matrix]


def test_runtime_contract_handles_nested_sensitive_calls_without_overlapping_edits():
    calls = []
    matrix = np.array([[0.0, np.log1p(3.0)]])
    obj = SimpleNamespace(X=matrix, shape=matrix.shape, layers={})

    def score_genes(value, *, gene_list):
        calls.append((value, tuple(gene_list)))
        return value

    namespace = {
        "adata": obj,
        "sc": SimpleNamespace(tl=SimpleNamespace(score_genes=score_genes)),
    }
    instrumented, count = _execute_instrumented(
        "result = sc.tl.score_genes("
        "sc.tl.score_genes(adata, gene_list=['inner']), "
        "gene_list=['outer'])",
        namespace,
    )

    assert count == 2
    assert instrumented.count("_scai_expression_contract_target(") == 2
    assert calls == [(obj, ("inner",)), (obj, ("outer",))]
    assert namespace["result"] is obj


def test_runtime_contract_handles_nested_nmf_and_non_ascii_byte_offsets():
    counters = {"construct": 0, "fit": 0}
    matrix = np.array([[0.0, np.log1p(2.0)]])

    class NMF:
        def __init__(self):
            counters["construct"] += 1

        def fit_transform(self, value):
            counters["fit"] += 1
            return value

    namespace = {"NMF": NMF, "X": matrix}
    instrumented, count = _execute_instrumented(
        "label = 'café prefix'; "
        "result = NMF().fit_transform(NMF().fit_transform(X))",
        namespace,
    )

    assert count == 2
    assert "label = 'café prefix'" in instrumented
    assert counters == {"construct": 2, "fit": 2}
    assert namespace["result"] is matrix


def test_runtime_contract_preserves_unparenthesized_generator_expressions():
    instrumented, count = instrument_expression_contract_calls(
        "result = max(value for value in values)"
    )
    assert count == 1

    namespace = {"values": [1, 3, 2]}
    exec(build_expression_runtime_code(), namespace)
    exec(instrumented, namespace)

    assert namespace["result"] == 3


def test_runtime_contract_preserves_calls_inside_f_strings():
    matrix = np.array([[0.0, np.log1p(2.0)]])
    obj = SimpleNamespace(X=matrix, shape=matrix.shape, layers={})

    for code in (
        "result = f'{sc.tl.score_genes(adata, gene_list=items)}'",
        'result = f"{sc.tl.score_genes(adata, gene_list=items)}"',
    ):
        calls = []

        def score_genes(value, *, gene_list):
            calls.append((value, tuple(gene_list)))
            return "validated"

        namespace = {
            "adata": obj,
            "items": ["A"],
            "sc": SimpleNamespace(
                tl=SimpleNamespace(score_genes=score_genes)
            ),
        }
        instrumented, count = instrument_expression_contract_calls(code)
        assert count == 1
        exec(build_expression_runtime_code(), namespace)
        exec(instrumented, namespace)

        assert namespace["result"] == "validated"
        assert calls == [(obj, ("A",))]


def test_runtime_contract_still_guards_sensitive_call_inside_generator():
    raw = SimpleNamespace(
        X=np.array([[0.0, 4.0]]), shape=(1, 2), layers={}
    )
    calls = []

    def score_genes(value, *, gene_list):
        calls.append((value, tuple(gene_list)))
        return [1, 2]

    namespace = {
        "adata": raw,
        "items": ["A"],
        "sc": SimpleNamespace(
            tl=SimpleNamespace(score_genes=score_genes)
        ),
    }
    instrumented, count = instrument_expression_contract_calls(
        "result = max(value for value in sc.tl.score_genes("
        "adata, gene_list=items))"
    )
    assert count == 2
    exec(build_expression_runtime_code(), namespace)

    with pytest.raises(ValueError, match="raw integer counts"):
        exec(instrumented, namespace)
    assert calls == []


def test_runtime_contract_guards_sensitive_call_inside_method_chain():
    calls = []
    raw = np.array([[0.0, 4.0]])

    class NMF:
        def fit_transform(self, value):
            calls.append(value)
            return value

    namespace = {"NMF": NMF, "raw": raw}
    instrumented, count = instrument_expression_contract_calls(
        "result = NMF().fit_transform(raw).sum()"
    )

    assert count == 2
    assert instrumented.count("_scai_expression_contract_target(") == 2
    exec(build_expression_runtime_code(), namespace)
    with pytest.raises(ValueError, match="raw integer counts"):
        exec(instrumented, namespace)
    assert calls == []


@pytest.mark.parametrize(
    "code",
    [
        "first = sc.tl.score_genes\nsecond = first\nsecond(adata=raw, gene_list=['A'])",
        "getattr(sc.tl, 'score_genes')(adata=raw, gene_list=['A'])",
        "funcs = {'score': sc.tl.score_genes}\nfuncs['score'](adata=raw, gene_list=['A'])",
        "sg, other = (sc.tl.score_genes, None)\nsg(adata=raw, gene_list=['A'])",
        "sg = lambda **kw: sc.tl.score_genes(**kw)\nsg(adata=raw, gene_list=['A'])",
        "def sg(**kw):\n    return sc.tl.score_genes(**kw)\nsg(adata=raw, gene_list=['A'])",
    ],
)
def test_runtime_contract_alias_forms_validate_the_runtime_callable(code):
    calls = []
    raw = SimpleNamespace(
        X=np.array([[0.0, 5.0]]), shape=(1, 2), layers={}
    )

    def replacement(value=None, *, adata=None, **_kwargs):
        calls.append(value if value is not None else adata)

    namespace = {
        "raw": raw,
        "sc": SimpleNamespace(tl=SimpleNamespace(score_genes=replacement)),
    }
    with pytest.raises(ValueError, match="raw integer counts"):
        _execute_instrumented(code, namespace)
    assert calls == []
    assert namespace["sc"].tl.score_genes is replacement


@pytest.mark.parametrize(
    "code",
    [
        "model = NMF()\nmodel.fit_transform(raw)",
        "Factor = NMF\nFactor2 = Factor\nmodel = Factor2()\nmodel.fit_transform(raw)",
        "model = NMF()\nother = model\nother.fit_transform(raw)",
        "Factor = getattr(sk, 'NMF')\nmodel = Factor()\nmodel.fit_transform(raw)",
    ],
)
def test_runtime_contract_nmf_alias_forms_validate_actual_matrix(code):
    calls = []
    raw = np.array([[0.0, 3.0]])

    class NMF:
        def fit_transform(self, value):
            calls.append(value)
            return value

    namespace = {"raw": raw, "NMF": NMF, "sk": SimpleNamespace(NMF=NMF)}
    with pytest.raises(ValueError, match="raw integer counts"):
        _execute_instrumented(code, namespace)
    assert calls == []


def test_runtime_contract_does_not_block_unrelated_same_named_callable():
    calls = []
    raw = np.array([[0.0, 7.0]])

    def score_genes(value):
        calls.append(value)
        return "ordinary helper result"

    namespace = {"raw": raw, "score_genes": score_genes}
    _execute_instrumented("result = score_genes(raw)", namespace)

    assert calls == [raw]
    assert namespace["result"] == "ordinary helper result"


def test_runtime_contract_does_not_block_same_name_on_ordinary_namespace():
    calls = []
    raw = np.array([[0.0, 7.0]])

    def score_genes(value):
        calls.append(value)
        return "ordinary namespace result"

    namespace = {
        "raw": raw,
        "sc": SimpleNamespace(utils=SimpleNamespace(score_genes=score_genes)),
    }
    _execute_instrumented("result = sc.utils.score_genes(raw)", namespace)

    assert calls == [raw]
    assert namespace["result"] == "ordinary namespace result"


def test_runtime_contract_validates_each_sensitive_calls_actual_input():
    calls = []
    valid = SimpleNamespace(
        X=np.array([[0.0, np.log1p(4.0)]]), shape=(1, 2), layers={}
    )
    raw = SimpleNamespace(
        X=np.array([[0.0, 4.0]]), shape=(1, 2), layers={}
    )

    def score_genes(value, **_kwargs):
        calls.append(value)

    namespace = {
        "valid": valid,
        "raw": raw,
        "sc": SimpleNamespace(tl=SimpleNamespace(score_genes=score_genes)),
    }
    with pytest.raises(ValueError, match="raw integer counts"):
        _execute_instrumented(
            "first = sc.tl.score_genes\n"
            "first(valid, gene_list=['A'])\n"
            "second = first\n"
            "second(raw, gene_list=['B'])",
            namespace,
        )

    assert calls == [valid]


@pytest.mark.parametrize("failure", [RuntimeError("boom"), asyncio.CancelledError()])
def test_runtime_contract_does_not_pollute_callable_after_error_or_cancel(failure):
    valid = SimpleNamespace(
        X=np.array([[0.0, np.log1p(4.0)]]), shape=(1, 2), layers={}
    )
    should_fail = True
    calls = []

    def monkeypatched_score(value, **_kwargs):
        nonlocal should_fail
        calls.append(value)
        if should_fail:
            should_fail = False
            raise failure
        return "recovered"

    api = SimpleNamespace(score_genes=monkeypatched_score)
    namespace = {"valid": valid, "sc": SimpleNamespace(tl=api)}
    instrumented, count = instrument_expression_contract_calls(
        "alias = sc.tl.score_genes\nresult = alias(valid, gene_list=['A'])"
    )
    assert count > 0
    exec(build_expression_runtime_code(), namespace)

    with pytest.raises(type(failure)):
        exec(instrumented, namespace)
    assert api.score_genes is monkeypatched_score

    exec(instrumented, namespace)
    assert api.score_genes is monkeypatched_score
    assert namespace["result"] == "recovered"
    assert calls == [valid, valid]


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([["secret-value"]], dtype=object),
        np.array([["secret-value"]], dtype="U20"),
        np.array([[True]], dtype=bool),
        np.array([[1 + 2j]], dtype=complex),
        np.array([["2026-01-01"]], dtype="datetime64[D]"),
    ],
)
def test_runtime_contract_rejects_unsupported_dtype_without_values(matrix):
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])
    with pytest.raises(ValueError, match="unsupported dtype") as exc_info:
        exec(
            guard,
            {"adata": SimpleNamespace(X=matrix, shape=matrix.shape, layers={})},
        )
    assert "secret-value" not in str(exc_info.value)


def test_runtime_contract_supports_sparse_and_numeric_pandas_inputs():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pd = pytest.importorskip("pandas")
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])

    sparse = scipy_sparse.csr_matrix([[0.0, np.log1p(4.0)]])
    exec(guard, {"adata": SimpleNamespace(X=sparse, shape=sparse.shape, layers={})})

    frame = pd.DataFrame({"a": [0.0, np.log1p(3.0)], "b": [np.nan, 0.0]})
    exec(guard, {"adata": frame})

    bad_frame = pd.DataFrame({"a": ["secret-value"]})
    with pytest.raises(ValueError, match="unsupported dtype") as exc_info:
        exec(guard, {"adata": bad_frame})
    assert "secret-value" not in str(exc_info.value)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 2), dtype=np.int64),
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
        np.array([[0, 2]], dtype=np.int16),
    ],
)
def test_runtime_contract_rejects_every_nonnegative_integer_dtype(matrix):
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])
    with pytest.raises(ValueError, match="raw integer counts"):
        exec(
            guard,
            {"adata": SimpleNamespace(X=matrix, shape=matrix.shape, layers={})},
        )


def test_runtime_contract_rejects_sparse_and_nullable_integer_inputs():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pd = pytest.importorskip("pandas")
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])

    sparse = scipy_sparse.csr_matrix(np.array([[0, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="raw integer counts"):
        exec(
            guard,
            {"adata": SimpleNamespace(X=sparse, shape=sparse.shape, layers={})},
        )

    for sparse_float in (
        scipy_sparse.csr_matrix(np.zeros((2, 2), dtype=float)),
        scipy_sparse.csr_matrix(np.array([[0.0, 1.0]], dtype=float)),
    ):
        with pytest.raises(ValueError, match="raw integer counts"):
            exec(
                guard,
                {
                    "adata": SimpleNamespace(
                        X=sparse_float, shape=sparse_float.shape, layers={}
                    )
                },
            )

    nullable = pd.DataFrame({"a": pd.array([0, 1], dtype="Int64")})
    with pytest.raises(ValueError, match="raw integer counts"):
        exec(guard, {"adata": nullable})


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 2), dtype=float),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 2.0], [3.0, 0.0]]),
        np.full((2, 2), np.nan),
        np.full((2, 2), np.inf),
    ],
)
def test_runtime_contract_fails_closed_without_continuous_evidence(matrix):
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])
    with pytest.raises(ValueError, match="raw integer counts"):
        exec(
            guard,
            {"adata": SimpleNamespace(X=matrix, shape=matrix.shape, layers={})},
        )


def test_runtime_contract_rejects_mixed_raw_and_continuous_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "raw": pd.array([0, 4], dtype="Int64"),
        "normalized": pd.array([0.0, np.log1p(4.0)], dtype="Float64"),
    })
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])

    with pytest.raises(ValueError, match="raw integer counts"):
        exec(guard, {"adata": frame})


def test_runtime_contract_does_not_treat_one_exact_integer_as_raw_component():
    values = np.full(10001, 0.5, dtype=float)
    old_sample = set(np.linspace(0, values.size - 1, 10000, dtype=int))
    missed_index = next(index for index in range(values.size) if index not in old_sample)
    values[missed_index] = 4.0
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])

    exec(
        guard,
        {
            "adata": SimpleNamespace(
                X=values.reshape(1, -1),
                shape=(1, values.size),
                layers={},
            )
        },
    )


def test_runtime_contract_accepts_incidental_integer_in_every_matrix_kind():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pd = pytest.importorskip("pandas")
    runtime = {}
    exec(build_expression_runtime_code(), runtime)
    chunk_size = runtime["_scai_expression_chunk_size"]
    values = np.full(chunk_size * 2 + 17, 0.5, dtype=float)
    values[-1] = 4.0
    matrices = (
        values.reshape(1, -1),
        scipy_sparse.csr_matrix(values.reshape(1, -1)),
        pd.DataFrame({"normalized": values}),
    )

    for matrix in matrices:
        contract = runtime["_scai_validate_expression_contract"](
            matrix, "chunked input", emit=False
        )
        assert contract["scale"] == "normalized_continuous"


def test_runtime_contract_accepts_log_matrix_with_incidental_exact_integer():
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])
    matrix = np.array([[0.0, 1.0, np.log1p(3.0)]], dtype=float)

    exec(guard, {"adata": SimpleNamespace(X=matrix, shape=matrix.shape, layers={})})


def test_runtime_contract_bounds_dense_sparse_and_pandas_temporary_vectors(
    monkeypatch,
):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pd = pytest.importorskip("pandas")
    runtime = {}
    exec(build_expression_runtime_code(), runtime)
    chunk_size = runtime["_scai_expression_chunk_size"]
    real_numpy = runtime["_scai_expression_np"]
    observed_sizes = {
        name: [] for name in ("isfinite", "isclose", "rint", "count_nonzero")
    }

    class BoundedOperationNumpy:
        def __getattr__(self, name):
            original = getattr(real_numpy, name)
            if name not in observed_sizes:
                return original

            def tracked(*args, **kwargs):
                for argument in args:
                    try:
                        observed_sizes[name].append(int(real_numpy.asarray(argument).size))
                    except (TypeError, ValueError):
                        pass
                return original(*args, **kwargs)

            return tracked

    runtime["_scai_expression_np"] = BoundedOperationNumpy()
    pandas_slice_sizes = []
    original_to_numpy = pd.Series.to_numpy

    def tracked_to_numpy(series, *args, **kwargs):
        pandas_slice_sizes.append(len(series))
        return original_to_numpy(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "to_numpy", tracked_to_numpy)
    size = chunk_size * 2 + 18
    values = np.full(size, np.log1p(2.0), dtype=float)
    noncontiguous_dense = values.reshape(-1, 2).T
    matrices = (
        noncontiguous_dense,
        scipy_sparse.csr_matrix(values.reshape(1, -1)),
        pd.DataFrame({"normalized": values}),
    )

    for matrix in matrices:
        contract = runtime["_scai_validate_expression_contract"](
            matrix, "bounded input", emit=False
        )
        assert contract["scale"] == "normalized_continuous"

    assert all(observed_sizes.values())
    assert max(max(sizes) for sizes in observed_sizes.values()) <= chunk_size
    assert pandas_slice_sizes
    assert max(pandas_slice_sizes) <= chunk_size


def test_runtime_contract_accepts_nullable_continuous_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "normalized": pd.array([0.0, np.log1p(4.0), pd.NA], dtype="Float64"),
    })
    guard = build_expression_preflight_code([{"name": "adata", "layer": None}])
    exec(guard, {"adata": frame})


def test_runtime_contract_marker_and_persisted_record_exclude_raw_extrema():
    matrix = np.array([[0.0, np.log1p(7.0)]])
    obj = SimpleNamespace(X=matrix, shape=matrix.shape, layers={})
    namespace = {
        "adata": obj,
        "sc": SimpleNamespace(
            tl=SimpleNamespace(score_genes=lambda value, **kwargs: None)
        ),
    }
    stream = StringIO()
    with redirect_stdout(stream):
        _execute_instrumented(
            "sc.tl.score_genes(adata, gene_list=['A'])", namespace
        )

    marker = stream.getvalue()
    assert "__SCAI_EXPRESSION_CONTRACT__" in marker
    assert '"min"' not in marker
    assert '"max"' not in marker

    state = AnalysisState(id="safe-contract", user_id="u", current_round=1)
    record_expression_contract(
        state,
        [{
            "name": "score_genes",
            "layer": None,
            "shape": [1, 2],
            "scale": "normalized_continuous",
            "dtype_class": "floating",
            "min": "secret-minimum",
            "max": "secret-maximum",
        }],
    )
    persisted = state.sandbox_state["expression_contract_v1"]["inputs"][0]
    assert persisted == {
        "name": "score_genes",
        "layer": None,
        "shape": [1, 2],
        "scale": "normalized_continuous",
        "dtype_class": "floating",
    }


def test_representation_change_retains_artifacts_with_warning():
    state = AnalysisState(id="expr", user_id="u", current_round=2)
    state.publication_figures = [
        PublicationFigure(
            name="old", description="", figure_type="umap",
            png_path="old.jpg", pdf_path="old.pdf",
            review_status=FigureReviewStatus.APPROVED, round_number=2,
        ),
        PublicationFigure(
            name="prior_round", description="", figure_type="umap",
            png_path="prior.jpg", pdf_path="prior.pdf",
            review_status=FigureReviewStatus.APPROVED, round_number=1,
        ),
    ]
    count = record_representation_change(state, "sc.pp.log1p(adata)")
    assert count == 0
    assert state.publication_figures[0].is_current
    assert state.publication_figures[0].invalidated_at is None
    assert "remains included" in state.publication_figures[0].description
    assert state.publication_figures[1].is_current
    assert state.sandbox_state["analysis_generation"] == 1


def test_representation_change_warning_tracks_round_and_generation():
    state = AnalysisState(id="expr-generation-aware", user_id="u", current_round=3)
    state.sandbox_state["analysis_generation"] = 4

    def artifact(name, *, round_number, analysis_generation, lineage_id="main"):
        return PublicationFigure(
            name=name,
            description=name,
            figure_type="umap",
            png_path=f"{name}.png",
            pdf_path=f"{name}.pdf",
            round_number=round_number,
            analysis_generation=analysis_generation,
            lineage_id=lineage_id,
        )

    retained_old = artifact(
        "retained-old", round_number=2, analysis_generation=2
    )
    new_generation = artifact(
        "new-generation", round_number=3, analysis_generation=5
    )
    future_round = artifact(
        "future-round", round_number=4, analysis_generation=2
    )
    other_lineage = artifact(
        "other-lineage", round_number=2, analysis_generation=2,
        lineage_id="branch-2",
    )
    retained_table = PublicationTable(
        name="retained-table",
        description="retained-table",
        csv_path="retained-table.csv",
        round_number=1,
        analysis_generation=2,
    )
    state.publication_figures = [
        retained_old, new_generation, future_round, other_lineage,
    ]
    state.publication_tables = [retained_table]

    assert record_representation_change(state, "sc.pp.log1p(adata)") == 0

    assert "remains included" in retained_old.description
    assert retained_old.expression_provenance["representation_change_warning"]
    assert retained_old.is_current is True
    assert "remains included" in retained_table.description
    assert retained_table.expression_provenance["representation_change_warning"]
    assert retained_table.is_current is True

    # Artifacts already at the new generation, beyond the current round, or
    # from another lineage are retained but untouched.
    assert "remains included" not in new_generation.description
    assert "representation_change_warning" not in new_generation.expression_provenance
    for untouched in (future_round, other_lineage):
        assert "remains included" not in untouched.description
        assert not untouched.expression_provenance
        assert untouched.is_current is True


def _state_with_current_expression_artifact(tmp_path, name):
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    (figures_dir / f"{name}.jpg").write_bytes(b"old figure")
    state = AnalysisState(id=f"expr-{name}", user_id="u", current_round=2)
    state.publication_figures = [
        PublicationFigure(
            name=name,
            description="",
            figure_type="umap",
            png_path=f"{name}.jpg",
            pdf_path=f"{name}.pdf",
            review_status=FigureReviewStatus.APPROVED,
            round_number=2,
        )
    ]
    return state, workspace_artifact_versions(tmp_path)


def test_failed_sensitive_call_after_mutation_retains_artifacts(tmp_path):
    state, before = _state_with_current_expression_artifact(
        tmp_path, "before_contract_failure"
    )
    matrix = np.array([[0.0, np.log1p(7.0)]])
    adata = SimpleNamespace(X=matrix.copy(), shape=matrix.shape, layers={})
    score_calls = []

    def scale(obj):
        obj.X = obj.X - 1.0

    namespace = {
        "adata": adata,
        "sc": SimpleNamespace(
            pp=SimpleNamespace(scale=scale),
            tl=SimpleNamespace(
                score_genes=lambda value, **kwargs: score_calls.append(value)
            ),
        ),
    }
    code = (
        "sc.pp.scale(adata)\n"
        "sc.tl.score_genes(adata, gene_list=['A'])"
    )
    with pytest.raises(ValueError, match="centered/scaled"):
        _execute_instrumented(code, namespace)

    count, stems, note = record_executed_representation_change(
        state,
        code,
        tmp_path,
        before,
        execution_started=True,
    )

    assert score_calls == []
    assert count == 0
    assert stems == set()
    assert "remain eligible" in note
    assert state.publication_figures[0].is_current
    assert "remains included" in state.publication_figures[0].description
    assert state.sandbox_state["analysis_generation"] == 1


def test_failed_statement_after_mutation_retains_artifacts(tmp_path):
    state, before = _state_with_current_expression_artifact(
        tmp_path, "before_runtime_failure"
    )
    matrix = np.array([[0.0, 1.0]])
    adata = SimpleNamespace(X=matrix.copy(), shape=matrix.shape, layers={})

    def log1p(obj):
        obj.X = np.log1p(obj.X)

    namespace = {
        "adata": adata,
        "sc": SimpleNamespace(pp=SimpleNamespace(log1p=log1p)),
    }
    code = "sc.pp.log1p(adata)\nraise RuntimeError('later failure')"
    with pytest.raises(RuntimeError, match="later failure"):
        exec(code, namespace)

    count, stems, _note = record_executed_representation_change(
        state,
        code,
        tmp_path,
        before,
        execution_started=True,
    )

    assert count == 0
    assert stems == set()
    assert state.publication_figures[0].is_current


def test_setup_failure_before_user_cell_does_not_invalidate_artifacts(tmp_path):
    state, before = _state_with_current_expression_artifact(
        tmp_path, "before_setup_failure"
    )
    count, stems, note = record_executed_representation_change(
        state,
        "sc.pp.log1p(adata)",
        tmp_path,
        before,
        execution_started=False,
    )

    assert (count, stems, note) == (0, set(), "")
    assert state.publication_figures[0].is_current
    assert "analysis_generation" not in state.sandbox_state


def test_matplotlib_table_scaling_is_not_an_expression_mutation():
    assert not mutates_expression_representation(
        "table = ax.table(cellText=rows)\n"
        "alias = table\n"
        "table.scale(1.0, 2.0)\n"
        "alias.scale(0.5, 0.5)"
    )


def test_unknown_scale_receiver_remains_fail_closed():
    for code in (
        "obj.scale(1.0, 1.0)",
        "table = ax.table(cellText=rows)\n"
        "table = make_table()\n"
        "table.scale(1.0, 1.0)",
        "sc.pp.scale(adata)",
    ):
        assert mutates_expression_representation(code)


def test_matplotlib_table_scaling_keeps_existing_figure_current(tmp_path):
    state, before = _state_with_current_expression_artifact(
        tmp_path, "table_geometry"
    )

    count, stems, note = record_executed_representation_change(
        state,
        "table = ax.table(cellText=rows)\n"
        "alias = table\n"
        "alias.scale(1.0, 2.0)",
        tmp_path,
        before,
        execution_started=True,
    )

    assert (count, stems, note) == (0, set(), "")
    assert state.publication_figures[0].is_current
    assert "analysis_generation" not in state.sandbox_state


def test_local_copy_mutation_is_not_a_representation_change():
    # The dominant small-model pattern: normalize a throwaway copy every step.
    # The canonical adata is untouched, so prior figures stay valid.
    assert not mutates_expression_representation(
        "work = adata.copy()\n"
        "sc.pp.normalize_total(work, target_sum=None)\n"
        "sc.pp.log1p(work)"
    )
    # Same line, semicolon-separated.
    assert not mutates_expression_representation(
        "adata_norm = adata.copy(); sc.pp.log1p(adata_norm)"
    )
    # Copy guarded by an existence check (verbatim small-model idiom).
    assert not mutates_expression_representation(
        "if 'adata_norm' not in locals():\n"
        "    adata_norm = adata.copy()\n"
        "    sc.pp.normalize_total(adata_norm)\n"
        "    sc.pp.log1p(adata_norm)"
    )
    # Subset of a local copy stays local; scaling it is still shielded.
    assert not mutates_expression_representation(
        "work = adata.copy()\n"
        "sub = work[work.obs['ct'] == 'Hep'].copy()\n"
        "sc.pp.scale(sub)"
    )
    # A freshly (re)loaded object is a distinct matrix, not the shared one.
    assert not mutates_expression_representation(
        "adata2 = sc.read_h5ad('x.h5ad')\n"
        "sc.pp.log1p(adata2)"
    )
    # Replacing .X on a local copy is shielded too.
    assert not mutates_expression_representation(
        "work = adata.copy()\n"
        "work.X = work.X.multiply(2)"
    )


def test_canonical_mutation_still_detected_alongside_local_copies():
    # A copy is normalized AND the canonical adata is normalized -> still counts.
    assert mutates_expression_representation(
        "work = adata.copy()\n"
        "sc.pp.log1p(work)\n"
        "sc.pp.log1p(adata)"
    )
    # Rebinding the canonical name to a non-fresh source does not shield it.
    assert mutates_expression_representation(
        "adata = load_shared()\n"
        "sc.pp.normalize_total(adata)"
    )


def test_cross_cell_derived_ledger_shields_only_registered_copy():
    assert not mutates_expression_representation(
        "sc.pp.log1p(work)", derived_names={"work"}
    )
    assert mutates_expression_representation("sc.pp.log1p(other)", derived_names={"work"})


# --- Numeric-library calls are not expression mutations -----------------------
# Taken verbatim from real session code. `np.log1p` shares a name with
# `sc.pp.log1p` and nothing else; matched on the bare name, its BinOp argument
# resolved to no target root and took the "unknown source, assume canonical"
# branch, so every one of these reported a canonical mutation.

@pytest.mark.parametrize("code", [
    "cpm = np.log1p(counts / counts.sum() * 1e6)",
    "logcpm = np.log1p(bulk.div(bulk.sum(0), axis=1) * 1e6)",
    "lc = np.log1p(obs.total_counts)",
    "z = scipy.stats.zscore(m)\nscaled = sklearn.preprocessing.scale(z)",
    "vals = numpy.log1p(numpy.asarray(x))",
])
def test_numeric_library_calls_are_not_representation_mutations(code):
    assert not mutates_expression_representation(code)


@pytest.mark.parametrize("code", [
    "sc.pp.log1p(adata)",
    "sc.pp.normalize_total(adata, target_sum=1e4)",
    "log1p(adata)",
    "pp.scale(adata, max_value=10)",
])
def test_scanpy_and_bare_mutations_still_detected(code):
    assert mutates_expression_representation(code)


# --- The file-stem blacklist is bounded to the round that is mutating ---------

def _stamp(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


def test_unchanged_stems_are_bounded_to_the_current_round(tmp_path):
    _stamp(tmp_path / "figures" / "from_round_1.png", 1000.0)
    _stamp(tmp_path / "figures" / "from_round_3.png", 3000.0)
    before = workspace_artifact_versions(tmp_path)

    # Unscoped, one mutating cell in round 3 blacklists round 1's figure too.
    assert unchanged_artifact_stems(tmp_path, before) == {
        "from_round_1", "from_round_3",
    }
    # Scoped to a round that started at t=2000, only that round's own output is
    # affected -- matching the rule invalidate_current_artifacts already applies.
    assert unchanged_artifact_stems(
        tmp_path, before, since_ns=int(2000.0 * 1_000_000_000)
    ) == {"from_round_3"}


def test_round_start_ns_comes_from_the_stamp_taken_when_the_round_opened(tmp_path):
    """The running round is never in analysis_log, so a log lookup finds nothing.

    The bioinformatics agent keeps its steps in a local list and they are merged
    into state.analysis_log -- and stamped with round_number -- only after the
    round's execution has finished. Deriving the start from the log therefore
    always fell through to time.time_ns(), a bound nothing predates: the stem
    blacklist came back empty on every main pass, so a figure drawn from raw
    counts and invalidated by a log1p in the same round shipped as a current
    result while the agent was told N artifacts had been withheld.
    """
    before_round = time.time_ns()
    state = AnalysisState(initial_instruction="x", user_id="u")
    state.current_round = 4  # no analysis_log entry for round 4 -- there never is

    assert record_round_start(state) >= before_round
    since = round_start_ns(state)
    assert before_round <= since <= time.time_ns()

    # A figure drawn after the round opened and not rewritten by the mutating
    # cell is stale, and must be named as such. Stamped in ns: seconds-as-float
    # loses sub-microsecond precision, which is the whole margin here.
    drawn = tmp_path / "figures" / "drawn_this_round.png"
    drawn.parent.mkdir(parents=True, exist_ok=True)
    drawn.write_bytes(b"x")
    os.utime(drawn, ns=(since + 1, since + 1))
    before = workspace_artifact_versions(tmp_path)
    assert unchanged_artifact_stems(tmp_path, before, since_ns=since) == {
        "drawn_this_round"
    }


def test_round_start_stamp_survives_a_second_entry_into_the_same_round():
    """Stage 1 and stage 2 re-enter the same node with the same round number.

    A stage-2 normalization does invalidate what stage 1 drew, so re-stamping on
    the second entry would hide exactly the case the mechanism exists for.
    """
    state = AnalysisState(initial_instruction="x", user_id="u")
    state.current_round = 1
    first = record_round_start(state)
    time.sleep(0.001)
    assert record_round_start(state) == first
    state.current_round = 2
    assert record_round_start(state) > first


def test_round_start_ns_falls_back_to_the_previous_round_boundary():
    """A session that predates the stamp still gets round scoping, not silence."""
    state = AnalysisState(initial_instruction="x", user_id="u")
    state.current_round = 3
    last_of_round_2 = datetime(2026, 1, 1, 11, 0, 0)
    state.analysis_log = [
        AnalysisStep(step_number=1, step_type="execute_code", round_number=1,
                     timestamp=datetime(2026, 1, 1, 9, 0, 0)),
        AnalysisStep(step_number=2, step_type="execute_code", round_number=2,
                     timestamp=last_of_round_2),
    ]
    assert round_start_ns(state) == int(last_of_round_2.timestamp() * 1_000_000_000)


def test_round_start_ns_with_nothing_to_bound_on_includes_everything(tmp_path):
    """Fail open, not silent: round 1 of a session owns every file on disk.

    Returning "now" here was the silent-failure default -- nothing predates it,
    so the caller withheld nothing while still reporting that it had.
    """
    _stamp(tmp_path / "figures" / "existing.png", 1000.0)
    before = workspace_artifact_versions(tmp_path)
    state = AnalysisState(initial_instruction="x", user_id="u")
    state.current_round = 1  # nothing logged, nothing stamped

    assert round_start_ns(state) == 0
    assert unchanged_artifact_stems(
        tmp_path, before, since_ns=round_start_ns(state)
    ) == {"existing"}


def test_round_start_ns_uses_the_earliest_step_of_the_current_round():
    state = AnalysisState(initial_instruction="x", user_id="u")
    state.current_round = 2
    early = datetime(2026, 1, 1, 10, 0, 0)
    state.analysis_log = [
        AnalysisStep(step_number=1, step_type="execute_code", round_number=1,
                     timestamp=datetime(2026, 1, 1, 9, 0, 0)),
        AnalysisStep(step_number=2, step_type="execute_code", round_number=2,
                     timestamp=early),
        AnalysisStep(step_number=3, step_type="execute_code", round_number=2,
                     timestamp=datetime(2026, 1, 1, 11, 0, 0)),
    ]
    assert round_start_ns(state) == int(early.timestamp() * 1_000_000_000)


def test_the_bioinformatics_node_opens_the_round_before_it_runs():
    """Without this call the stamp never exists and the whole mechanism is dead.

    Source-level because the alternative is standing up a whole workflow object:
    what matters is only that the round is opened *before* the autonomous loop,
    since everything that writes an artifact runs inside that loop.
    """
    import inspect
    import re

    import backend.workflow.agents as agents_mod

    from backend.workflow.agent_nodes import bioinformatics_node

    body = inspect.getsource(bioinformatics_node)
    opened = body.find("record_round_start(analysis_state)")
    runs = body.find("run_autonomous_analysis(")
    assert opened != -1, "the bioinformatics node must open the round"
    assert runs != -1 and opened < runs, (
        "the round must be stamped before the loop that produces the artifacts"
    )


# --- The producer is told what it invalidated --------------------------------

def test_invalidation_is_reported_back_to_the_agent():
    note = describe_invalidated_artifacts({"umap_cell_types", "volcano_a"}, 2)
    assert "umap_cell_types" in note and "volcano_a" in note
    assert "Regenerate" in note


def test_no_invalidation_produces_no_note():
    assert describe_invalidated_artifacts(set(), 0) == ""
