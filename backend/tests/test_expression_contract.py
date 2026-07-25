from types import SimpleNamespace

import numpy as np
import pytest

from backend.models.schemas import AnalysisState, PublicationFigure, FigureReviewStatus
from backend.utils.expression_contract import (
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    prepares_expression_before_sensitive_call,
    record_representation_change,
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
