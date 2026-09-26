"""Deterministic Stage 1 annotation and representation baselines."""

import contextlib
import io

import pytest


from backend.agents.bioinformatics.stage1_baseline_figures import (
    build_stage1_baseline_figure_specs,
    build_stage1_baseline_probe_code,
    parse_stage1_baseline_probe_output,
)


_MARKER = "__SCAI_STAGE1_BASELINE_PROBE__"


def _make_adata():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(11)
    n_obs = 360
    n_vars = 12
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    obs["cell_type"] = pd.Categorical(
        np.repeat(["type_a", "type_b", "type_c"], n_obs // 3)
    )
    obs["sample_id"] = pd.Categorical(
        np.tile(np.repeat(["sample_1", "sample_2"], 30), 6)
    )
    obs["leiden"] = pd.Categorical(
        np.repeat(["0", "1", "2", "3"], n_obs // 4)
    )
    expression = rng.gamma(1.5, 1.0, size=(n_obs, n_vars)).astype("float32")
    adata = ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
    )
    adata.obsm["X_umap"] = rng.normal(size=(n_obs, 2)).astype("float32")
    adata.uns["rank_genes_groups"] = {
        "names": np.rec.fromarrays(
            [
                np.array(["gene_0", "gene_1", "gene_2"]),
                np.array(["gene_3", "gene_4", "gene_5"]),
                np.array(["gene_6", "gene_7", "gene_8"]),
                np.array(["gene_9", "gene_10", "gene_11"]),
            ],
            names=["0", "1", "2", "3"],
        )
    }
    return adata


def _exec_probe(namespace):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(
            compile(build_stage1_baseline_probe_code(), "<stage1-probe>", "exec"),
            namespace,
            namespace,
        )
    return parse_stage1_baseline_probe_output(buffer.getvalue())


def test_all_real_panels_render_and_pass_the_blocking_layout_gate():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from backend.agents.bioinformatics.kernel_templates import LAYOUT_LINT_SRC
    from backend.services.figure_layout import blocking_violations

    namespace = {"adata_annotated": _make_adata()}
    probe = _exec_probe(namespace)
    specs = build_stage1_baseline_figure_specs(probe)
    assert len(specs) == 4

    exec(LAYOUT_LINT_SRC, namespace, namespace)
    for spec in specs:
        plt.close("all")
        exec(compile(spec.code, f"<{spec.name}>", "exec"), namespace, namespace)
        assert len(plt.get_fignums()) == 1
        figure = plt.figure(plt.get_fignums()[0])
        violations = namespace["_scai_lint_figure_layout"](figure)
        assert blocking_violations(violations) == [], (
            f"{spec.name} must pass the deterministic layout gate: {violations}"
        )
    plt.close("all")
