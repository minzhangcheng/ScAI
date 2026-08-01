import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from backend.models.schemas import (
    AnalysisState, AnalysisStep, PublicationFigure, FigureReviewStatus,
)
from backend.utils.expression_contract import (
    build_expression_preflight_code,
    describe_invalidated_artifacts,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    prepares_expression_before_sensitive_call,
    record_representation_change,
    round_start_ns,
    unchanged_artifact_stems,
    workspace_artifact_versions,
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


def test_representation_change_invalidates_same_round_artifacts_only():
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
    assert count == 1
    assert not state.publication_figures[0].is_current
    assert state.publication_figures[0].invalidated_at is not None
    assert state.publication_figures[1].is_current
    assert state.sandbox_state["analysis_generation"] == 1


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


def test_round_start_ns_without_a_logged_step_blacklists_nothing(tmp_path):
    _stamp(tmp_path / "figures" / "existing.png", 1000.0)
    before = workspace_artifact_versions(tmp_path)
    state = AnalysisState(initial_instruction="x", user_id="u")
    state.current_round = 4  # no analysis_log entry for round 4 yet

    since = round_start_ns(state)
    assert since > int(1000.0 * 1_000_000_000)
    assert unchanged_artifact_stems(tmp_path, before, since_ns=since) == set()


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


# --- The producer is told what it invalidated --------------------------------

def test_invalidation_is_reported_back_to_the_agent():
    note = describe_invalidated_artifacts({"umap_cell_types", "volcano_a"}, 2)
    assert "umap_cell_types" in note and "volcano_a" in note
    assert "Regenerate" in note


def test_no_invalidation_produces_no_note():
    assert describe_invalidated_artifacts(set(), 0) == ""
