from types import SimpleNamespace

import numpy as np
import pytest

from backend.models.schemas import (
    AnalysisState, PublicationFigure, PublicationTable,
)
from backend.utils.expression_contract import (
    build_expression_runtime_code,
    build_expression_preflight_code,
    instrument_expression_contract_calls,
    record_representation_change,
)


def _execute_instrumented(code, namespace):
    instrumented, count = instrument_expression_contract_calls(code)
    assert count > 0
    exec(build_expression_runtime_code(), namespace)
    exec(instrumented, namespace)
    return instrumented, count


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


# --- Numeric-library calls are not expression mutations -----------------------
# Taken verbatim from real session code. `np.log1p` shares a name with
# `sc.pp.log1p` and nothing else; matched on the bare name, its BinOp argument
# resolved to no target root and took the "unknown source, assume canonical"
# branch, so every one of these reported a canonical mutation.


# --- The file-stem blacklist is bounded to the round that is mutating ---------


# --- The producer is told what it invalidated --------------------------------
