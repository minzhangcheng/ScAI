"""The exploration round's quality-control panel.

Across seven real multi-round sessions the agent ran a full QC pass every time and
plotted it zero times, so the report's quality-control chapter was fluent prose over
numbers nobody could check. These tests pin the two halves of the fix that are pure
functions: what the kernel probe reports, and what the panel code does with it.

The kernel code itself is exercised against a real AnnData where the dependency is
available, and skipped where it is not -- this suite must stay runnable without the
scientific stack installed.
"""
import asyncio
import json
import textwrap

import pytest

from backend.agents.bioinformatics.exploration_diagnostics import (
    DIAGNOSTIC_FIGURE_NAME,
    DIAGNOSTIC_FIGURE_TYPE,
    MAX_DIAGNOSTIC_METRICS,
    build_diagnostic_panel_code,
    build_probe_code,
    ensure_exploration_diagnostic_figure,
    parse_probe_output,
)



def _probe(**over):
    payload = {
        "var": "adata_work",
        "n_obs": 900,
        "n_vars": 23422,
        "metrics": ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        "group_key": "library_id",
        "group_levels": ["lib_A", "lib_B"],
        "variance_ratio": [0.3, 0.2, 0.1],
    }
    payload.update(over)
    return payload


# --- generated code is syntactically valid -----------------------------------


def test_probe_code_compiles():
    compile(build_probe_code(), "<probe>", "exec")


def test_panel_code_compiles_for_every_shape():
    for over in (
        {},
        {"group_key": None, "group_levels": []},
        {"variance_ratio": []},
        {"metrics": ["only_one"], "group_key": None, "group_levels": [], "variance_ratio": []},
    ):
        compile(build_diagnostic_panel_code(_probe(**over)), "<panel>", "exec")


def test_panel_code_never_saves_the_figure_itself():
    """The publication wrapper owns saving; a second savefig would fight it."""
    code = build_diagnostic_panel_code(_probe())
    assert "savefig" not in code
    assert "plt.show" not in code


def test_panel_code_inlines_exactly_what_the_probe_measured():
    code = build_diagnostic_panel_code(_probe())
    assert "_qc_obj = adata_work" in code
    assert "'library_id'" in code
    assert "n_genes_by_counts" in code


def test_panel_code_honours_the_metric_cap():
    many = [f"metric_{i}" for i in range(20)]
    code = build_diagnostic_panel_code(_probe(metrics=many))
    listed = [m for m in many if f"'{m}'" in code]
    assert len(listed) == MAX_DIAGNOSTIC_METRICS


def test_the_panel_is_a_qc_figure_named_for_its_stage_not_its_plot_type():
    assert DIAGNOSTIC_FIGURE_TYPE == "qc"
    assert "qc" in DIAGNOSTIC_FIGURE_NAME


# --- probe output parsing is tolerant ----------------------------------------


@pytest.mark.parametrize("output", [
    None,
    "",
    "unrelated kernel chatter",
    "__SCAI_QC_PROBE__{not json",
    '__SCAI_QC_PROBE__{"error": "ImportError(...)"}',
    '__SCAI_QC_PROBE__{"var": "a", "metrics": []}',
    '__SCAI_QC_PROBE__{"metrics": ["x"]}',
    "__SCAI_QC_PROBE__[1, 2, 3]",
])
def test_unusable_probe_output_is_a_silent_skip(output):
    assert parse_probe_output(output) is None


@pytest.mark.parametrize("var", ["", "not an identifier", "1adata", "adata; import os", 42])
def test_a_var_that_is_not_an_identifier_is_rejected(var):
    """``var`` names a binding and is the one probe value inlined unquoted."""
    assert parse_probe_output("__SCAI_QC_PROBE__" + json.dumps(_probe(var=var))) is None


def test_probe_output_is_read_from_the_marker_line_among_noise():
    noisy = "\n".join([
        "some warning from a library",
        "__SCAI_QC_PROBE__" + json.dumps(_probe()),
    ])
    parsed = parse_probe_output(noisy)
    assert parsed["var"] == "adata_work"
    assert parsed["group_key"] == "library_id"


def test_the_last_marker_line_wins():
    first = "__SCAI_QC_PROBE__" + json.dumps(_probe(var="stale"))
    last = "__SCAI_QC_PROBE__" + json.dumps(_probe(var="fresh"))
    assert parse_probe_output(f"{first}\n{last}")["var"] == "fresh"


def test_parsing_caps_metrics_even_if_the_kernel_did_not():
    payload = _probe(metrics=[f"m{i}" for i in range(30)])
    parsed = parse_probe_output("__SCAI_QC_PROBE__" + json.dumps(payload))
    assert len(parsed["metrics"]) == MAX_DIAGNOSTIC_METRICS


# --- the orchestration never fails the round ---------------------------------


class _Result:
    def __init__(self, output="", error=None):
        self.output, self.error = output, error


class _Sandbox:
    def __init__(self, result=None, raises=False):
        self._result, self._raises = result, raises
        self.calls = 0

    async def execute(self, code, **kwargs):
        self.calls += 1
        if self._raises:
            raise RuntimeError("kernel died")
        return self._result


class _Agent:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.figure_calls = []

    async def _handle_generate_publication_figure(self, **kwargs):
        self.figure_calls.append(kwargs)
        return "Figure saved."


def _run(agent):
    return asyncio.run(ensure_exploration_diagnostic_figure(agent, object()))


def test_no_sandbox_is_a_skip():
    assert _run(_Agent(None)) is None


def test_a_probe_that_raises_is_a_skip():
    agent = _Agent(_Sandbox(raises=True))
    assert _run(agent) is None
    assert agent.figure_calls == []


def test_a_probe_that_errors_is_a_skip():
    agent = _Agent(_Sandbox(_Result(error="NameError")))
    assert _run(agent) is None
    assert agent.figure_calls == []


def test_nothing_plottable_is_a_skip():
    agent = _Agent(_Sandbox(_Result(output="__SCAI_QC_PROBE__{}")))
    assert _run(agent) is None
    assert agent.figure_calls == []


def test_a_usable_probe_registers_an_unreviewed_diagnostic_figure():
    payload = "__SCAI_QC_PROBE__" + json.dumps(_probe())
    agent = _Agent(_Sandbox(_Result(output=payload)))
    assert _run(agent) == "Figure saved."
    call = agent.figure_calls[0]
    assert call["figure_name"] == DIAGNOSTIC_FIGURE_NAME
    assert call["figure_type"] == DIAGNOSTIC_FIGURE_TYPE
    assert call["diagnostic"] is True
    # No reviewer: the review loop would spend revision budget asking a model to
    # improve code it cannot edit, and the PI rubric rejects this kind of panel.
    assert call["review_callback"] is None


def test_a_failing_figure_handler_does_not_propagate():
    payload = "__SCAI_QC_PROBE__" + json.dumps(_probe())
    agent = _Agent(_Sandbox(_Result(output=payload)))

    async def boom(**kwargs):
        raise RuntimeError("render failed")

    agent._handle_generate_publication_figure = boom
    assert _run(agent) is None


# --- against a real AnnData, where the stack is available --------------------


def _make_adata():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(7)
    n = 300
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["n_genes_by_counts"] = rng.integers(200, 5000, n)
    obs["total_counts"] = rng.integers(500, 40000, n).astype(float)
    obs["pct_counts_mt"] = rng.gamma(1.0, 2.0, n)
    obs["doublet_score"] = rng.beta(2, 20, n)
    obs["constant"] = 1.0
    obs["is_flagged"] = rng.random(n) > 0.5
    obs["library_id"] = pd.Categorical(rng.choice(["lib_A", "lib_B"], n))
    a = ad.AnnData(X=rng.random((n, 8)).astype("float32"), obs=obs)
    a.uns["pca"] = {"variance_ratio": np.sort(rng.random(10))[::-1]}
    return a


def _exec_probe(namespace):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(build_probe_code(), "<probe>", "exec"), namespace, namespace)
    return parse_probe_output(buf.getvalue())


def test_probe_finds_the_object_and_drops_unplottable_columns():
    adata = _make_adata()
    parsed = _exec_probe({"adata_work": adata})
    assert parsed["var"] == "adata_work"
    assert parsed["group_key"] == "library_id"
    assert parsed["group_levels"] == ["lib_A", "lib_B"]
    # A constant column and a boolean flag are not distributions.
    assert "constant" not in parsed["metrics"]
    assert "is_flagged" not in parsed["metrics"]
    assert "n_genes_by_counts" in parsed["metrics"]
    assert parsed["variance_ratio"]


def test_a_missing_group_label_does_not_become_its_own_level():
    """Stringifying NaN would draw a phantom series and could push a genuine
    eight-level design past the cap, suppressing the split entirely."""
    np = pytest.importorskip("numpy")
    adata = _make_adata()
    labels = adata.obs["library_id"].astype(object).copy()
    labels.iloc[:20] = np.nan
    adata.obs["library_id"] = labels
    parsed = _exec_probe({"adata": adata})
    assert parsed["group_key"] == "library_id"
    assert parsed["group_levels"] == ["lib_A", "lib_B"]


def test_probe_ignores_non_anndata_globals():
    assert _exec_probe({"x": 1, "s": "text"}) is None


def test_probe_prefers_the_object_carrying_quality_metrics():
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    bare = ad.AnnData(X=np.zeros((10, 2), dtype="float32"))
    parsed = _exec_probe({"subset_no_metrics": bare, "adata_full": _make_adata()})
    assert parsed["var"] == "adata_full"


def test_the_panel_renders_one_figure_and_passes_the_layout_gate():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backend.agents.bioinformatics.kernel_templates import LAYOUT_LINT_SRC
    from backend.services.figure_layout import blocking_violations

    namespace = {"adata_work": _make_adata()}
    parsed = _exec_probe(namespace)
    plt.close("all")
    exec(compile(build_diagnostic_panel_code(parsed), "<panel>", "exec"),
         namespace, namespace)
    assert len(plt.get_fignums()) == 1, "the wrapper saves exactly one figure"

    fig = plt.figure(plt.get_fignums()[0])
    exec(LAYOUT_LINT_SRC, namespace, namespace)
    violations = namespace["_scai_lint_figure_layout"](fig)
    assert blocking_violations(violations) == [], (
        "the panel must not be held back by the layout gate it has to pass"
    )
    plt.close("all")


def test_stage1_instruction_asks_for_the_panel_and_scopes_the_deferral():
    from backend.workflow.agent_nodes.shared import _STAGE1_INSTRUCTION

    text = textwrap.dedent(_STAGE1_INSTRUCTION)
    assert 'figure_type="qc"' in text
    # The blanket deferral is what suppressed the figure for seven sessions.
    assert "Do NOT generate publication figures (save that for Stage 2)" not in text
    assert "RESULTS" in text
