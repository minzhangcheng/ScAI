"""Deterministic Stage 1 annotation and representation baselines."""

import asyncio
import contextlib
import io
import json

import pytest

from backend.models.schemas import FigureReviewStatus, PublicationFigure

from backend.agents.bioinformatics.stage1_baseline_figures import (
    ANNOTATION_UMAP_NAME,
    BATCH_UMAP_NAME,
    CLUSTER_UMAP_NAME,
    MARKER_DOTPLOT_NAME,
    build_stage1_baseline_figure_specs,
    build_stage1_baseline_probe_code,
    ensure_stage1_baseline_figures,
    parse_stage1_baseline_probe_output,
)


_MARKER = "__SCAI_STAGE1_BASELINE_PROBE__"


def _payload(**over):
    payload = {
        "var": "adata_annotated",
        "n_obs": 720,
        "n_vars": 18000,
        "umap_key": "X_umap",
        "annotation_key": "cell_type",
        "annotation_levels": ["type_a", "type_b", "type_c"],
        "batch_key": "sample_id",
        "batch_levels": ["sample_1", "sample_2"],
        "cluster_key": "leiden",
        "cluster_levels": ["0", "1", "2"],
        "marker_group_key": "cell_type",
        "marker_group_levels": ["type_a", "type_b", "type_c"],
        "marker_genes": ["gene_0", "gene_1", "gene_2", "gene_3"],
    }
    payload.update(over)
    return payload


def _probe_line(**over):
    return _MARKER + json.dumps(_payload(**over))


def test_probe_and_all_generated_panels_compile():
    compile(build_stage1_baseline_probe_code(), "<stage1-probe>", "exec")
    probe = parse_stage1_baseline_probe_output(_probe_line())
    specs = build_stage1_baseline_figure_specs(probe)
    assert [spec.name for spec in specs] == [
        ANNOTATION_UMAP_NAME,
        BATCH_UMAP_NAME,
        CLUSTER_UMAP_NAME,
        MARKER_DOTPLOT_NAME,
    ]
    for spec in specs:
        compile(spec.code, f"<{spec.name}>", "exec")
        assert "savefig" not in spec.code
        assert "plt.show" not in spec.code


@pytest.mark.parametrize("output", [
    None,
    "",
    "kernel chatter",
    _MARKER + "{bad json",
    _MARKER + '{"error": "ImportError"}',
    _MARKER + '{"var": "adata; import os", "umap_key": "X_umap", '
              '"annotation_key": "cell_type", "annotation_levels": ["a"]}',
    _MARKER + '{"var": "adata", "n_obs": "not-an-int", '
              '"umap_key": "X_umap", "annotation_key": "cell_type", '
              '"annotation_levels": ["a"]}',
    _MARKER + '{"var": "adata"}',
])
def test_unusable_probe_is_a_silent_skip(output):
    assert parse_stage1_baseline_probe_output(output) is None


def test_last_probe_marker_wins_among_kernel_noise():
    first = _probe_line(var="stale")
    last = _probe_line(var="current")
    parsed = parse_stage1_baseline_probe_output(
        f"warning\n{first}\nmore output\n{last}"
    )
    assert parsed["var"] == "current"


def test_missing_optional_fields_remove_only_their_panels():
    probe = parse_stage1_baseline_probe_output(_probe_line(
        batch_key=None,
        batch_levels=[],
        cluster_key=None,
        cluster_levels=[],
        marker_group_key=None,
        marker_group_levels=[],
        marker_genes=[],
    ))
    assert [spec.name for spec in build_stage1_baseline_figure_specs(probe)] == [
        ANNOTATION_UMAP_NAME
    ]


def test_marker_panel_does_not_require_umap_coordinates():
    probe = parse_stage1_baseline_probe_output(_probe_line(
        umap_key=None,
        annotation_key=None,
        annotation_levels=[],
        batch_key=None,
        batch_levels=[],
        cluster_key=None,
        cluster_levels=[],
    ))
    assert [spec.name for spec in build_stage1_baseline_figure_specs(probe)] == [
        MARKER_DOTPLOT_NAME
    ]


def test_all_generated_panels_pass_the_sandbox_code_validator():
    from backend.utils.sandbox import validate_code_safety

    probe = parse_stage1_baseline_probe_output(_probe_line())
    for spec in build_stage1_baseline_figure_specs(probe):
        assert validate_code_safety(spec.code) == (True, "")


class _Result:
    def __init__(self, output="", error=None):
        self.output = output
        self.error = error


class _Sandbox:
    def __init__(self, result=None, raises=False):
        self.result = result
        self.raises = raises
        self.calls = []

    async def execute(self, code, **kwargs):
        self.calls.append((code, kwargs))
        if self.raises:
            raise RuntimeError("kernel unavailable")
        return self.result


class _Agent:
    def __init__(self, sandbox, fail_name=None):
        self.sandbox = sandbox
        self.fail_name = fail_name
        self.figure_calls = []

    async def _handle_generate_publication_figure(self, **kwargs):
        self.figure_calls.append(kwargs)
        if kwargs["figure_name"] == self.fail_name:
            raise RuntimeError("one renderer failed")
        return f"rendered:{kwargs['figure_name']}"


def _run(agent):
    return asyncio.run(ensure_stage1_baseline_figures(agent, object()))


def test_service_registers_each_available_panel_as_platform_rendered():
    agent = _Agent(_Sandbox(_Result(output=_probe_line())))
    rendered = _run(agent)
    assert list(rendered) == [
        ANNOTATION_UMAP_NAME,
        BATCH_UMAP_NAME,
        CLUSTER_UMAP_NAME,
        MARKER_DOTPLOT_NAME,
    ]
    assert len(agent.figure_calls) == 4
    for call in agent.figure_calls:
        assert call["diagnostic"] is True
        assert call["review_callback"] is None
        assert call["state"] is not None


def test_one_failed_panel_does_not_suppress_the_remaining_baselines():
    agent = _Agent(
        _Sandbox(_Result(output=_probe_line())), fail_name=BATCH_UMAP_NAME
    )
    rendered = _run(agent)
    assert BATCH_UMAP_NAME not in rendered
    assert ANNOTATION_UMAP_NAME in rendered
    assert CLUSTER_UMAP_NAME in rendered
    assert MARKER_DOTPLOT_NAME in rendered


def test_baseline_name_cannot_overwrite_analysis_owned_figure():
    state = type("State", (), {})()
    state.publication_figures = [PublicationFigure(
        name=ANNOTATION_UMAP_NAME,
        description="Reviewed analysis-owned result",
        figure_type="umap",
        png_path="/tmp/result.jpg",
        pdf_path="/tmp/result.pdf",
        review_status=FigureReviewStatus.APPROVED,
        diagnostic=False,
        purpose="result",
    )]
    agent = _Agent(_Sandbox(_Result(output=_probe_line(
        batch_key=None,
        batch_levels=[],
        cluster_key=None,
        cluster_levels=[],
        marker_group_key=None,
        marker_group_levels=[],
        marker_genes=[],
    ))))

    rendered = asyncio.run(ensure_stage1_baseline_figures(agent, state))

    safe_name = f"{ANNOTATION_UMAP_NAME}_platform_baseline"
    assert list(rendered) == [safe_name]
    assert agent.figure_calls[0]["figure_name"] == safe_name
    assert state.publication_figures[0].name == ANNOTATION_UMAP_NAME


def test_existing_platform_baseline_keeps_the_stable_name():
    state = type("State", (), {})()
    state.publication_figures = [PublicationFigure(
        name=ANNOTATION_UMAP_NAME,
        description="Platform baseline",
        figure_type="umap",
        png_path="/tmp/baseline.jpg",
        pdf_path="/tmp/baseline.pdf",
        diagnostic=True,
        purpose="baseline",
    )]
    agent = _Agent(_Sandbox(_Result(output=_probe_line(
        batch_key=None,
        batch_levels=[],
        cluster_key=None,
        cluster_levels=[],
        marker_group_key=None,
        marker_group_levels=[],
        marker_genes=[],
    ))))

    rendered = asyncio.run(ensure_stage1_baseline_figures(agent, state))

    assert list(rendered) == [ANNOTATION_UMAP_NAME]
    assert agent.figure_calls[0]["figure_name"] == ANNOTATION_UMAP_NAME


@pytest.mark.parametrize("agent", [
    _Agent(None),
    _Agent(_Sandbox(raises=True)),
    _Agent(_Sandbox(_Result(error="probe failed"))),
    _Agent(_Sandbox(_Result(output=_MARKER + "{}"))),
])
def test_service_probe_failures_are_silent(agent):
    assert _run(agent) == {}
    assert agent.figure_calls == []


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


def test_real_probe_prefers_the_annotated_object_and_discovers_all_baselines():
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    annotated = _make_adata()
    larger_but_bare = ad.AnnData(
        X=np.zeros((annotated.n_obs + 100, 2), dtype="float32")
    )
    larger_but_bare.obsm["X_umap"] = np.zeros(
        (larger_but_bare.n_obs, 2), dtype="float32"
    )
    parsed = _exec_probe({
        "larger_but_bare": larger_but_bare,
        "adata_annotated": annotated,
    })
    assert parsed["var"] == "adata_annotated"
    assert parsed["annotation_key"] == "cell_type"
    assert parsed["batch_key"] == "sample_id"
    assert parsed["cluster_key"] == "leiden"
    assert len(parsed["marker_genes"]) == 12


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
