"""Deterministic result baselines for innovation rounds.

Stage 1 gets its visual floor from ``stage1_baseline_figures``; rounds 2+ had
none, so a round that computed a great deal and drew nothing settled that way.
These tests pin the probe contract and the spec builder that give those rounds
the same platform-owned floor.
"""

import json

import pytest

from backend.agents.bioinformatics.innovation_baseline_figures import (
    build_innovation_baseline_figure_specs,
    build_innovation_baseline_probe_code,
    ensure_innovation_baseline_figures,
    parse_innovation_baseline_probe_output,
    should_render_innovation_baselines,
)


def _probe(**overrides):
    base = {
        "var": "adata",
        "n_obs": 18100,
        "group_key": "condition",
        "group_levels": ["CON", "DKO"],
        "score_keys": [],
        "de_tables": [],
        "enrichment_tables": [],
    }
    base.update(overrides)
    return base


def test_score_distribution_spec_is_built_from_obs_scores():
    specs = build_innovation_baseline_figure_specs(
        _probe(score_keys=["stress_score", "redox_score"]), round_number=3
    )

    assert [spec.figure_type for spec in specs] == ["violin"]
    spec = specs[0]
    assert spec.name == "r3_platform_score_distribution"
    assert "stress_score" in spec.code and "redox_score" in spec.code
    assert "condition" in spec.code


def test_volcano_spec_is_built_from_a_differential_table():
    specs = build_innovation_baseline_figure_specs(
        _probe(de_tables=[{
            "var": "deg_results",
            "lfc": "log2FoldChange",
            "pval": "padj",
            "gene": "gene",
            "n_rows": 21000,
        }]),
        round_number=3,
    )

    assert [spec.figure_type for spec in specs] == ["volcano"]
    spec = specs[0]
    assert spec.name == "r3_platform_volcano_deg_results"
    assert "deg_results" in spec.code
    assert "log2FoldChange" in spec.code and "padj" in spec.code


def test_enrichment_spec_is_built_from_a_term_table():
    specs = build_innovation_baseline_figure_specs(
        _probe(enrichment_tables=[{
            "var": "go_bp",
            "term": "Term",
            "pval": "Adjusted P-value",
            "n_rows": 180,
        }]),
        round_number=5,
    )

    assert [spec.figure_type for spec in specs] == ["barplot"]
    spec = specs[0]
    assert spec.name == "r5_platform_enrichment_go_bp"
    assert "go_bp" in spec.code and "Adjusted P-value" in spec.code


def test_panels_are_independent_and_missing_inputs_suppress_only_their_own():
    """A probe that finds only one kind of result still yields that one panel."""
    assert build_innovation_baseline_figure_specs(_probe(), round_number=2) == []

    scores_only = build_innovation_baseline_figure_specs(
        _probe(score_keys=["stress_score"]), round_number=2
    )
    assert [spec.figure_type for spec in scores_only] == ["violin"]

    everything = build_innovation_baseline_figure_specs(
        _probe(
            score_keys=["stress_score"],
            de_tables=[{
                "var": "deg", "lfc": "lfc", "pval": "padj",
                "gene": "gene", "n_rows": 100,
            }],
            enrichment_tables=[{
                "var": "go", "term": "Term", "pval": "padj", "n_rows": 50,
            }],
        ),
        round_number=2,
    )
    assert [spec.figure_type for spec in everything] == [
        "violin", "volcano", "barplot"
    ]


def test_score_distribution_needs_a_grouping_key():
    """A score with nothing to compare it across is not a deterministic panel."""
    specs = build_innovation_baseline_figure_specs(
        _probe(score_keys=["stress_score"], group_key=None, group_levels=[]),
        round_number=3,
    )
    assert specs == []


def test_probe_output_parses_and_rejects_an_unusable_payload():
    marker = "__SCAI_INNOVATION_BASELINE_PROBE__"
    payload = {
        "var": "adata",
        "n_obs": 100,
        "group_key": "condition",
        "group_levels": ["CON", "DKO"],
        "score_keys": ["stress_score"],
        "de_tables": [],
        "enrichment_tables": [],
    }
    parsed = parse_innovation_baseline_probe_output(
        "noise\n" + marker + json.dumps(payload)
    )
    assert parsed is not None
    assert parsed["var"] == "adata"
    assert parsed["score_keys"] == ["stress_score"]

    assert parse_innovation_baseline_probe_output(None) is None
    assert parse_innovation_baseline_probe_output("nothing here") is None
    assert parse_innovation_baseline_probe_output(
        marker + json.dumps({"error": "boom"})
    ) is None
    assert parse_innovation_baseline_probe_output(
        marker + json.dumps({"var": "not an identifier", "score_keys": ["s"]})
    ) is None
    # Nothing plottable at all.
    assert parse_innovation_baseline_probe_output(
        marker + json.dumps({
            "var": "adata", "group_key": "condition",
            "group_levels": ["CON"], "score_keys": [],
            "de_tables": [], "enrichment_tables": [],
        })
    ) is None


def test_probe_code_is_syntactically_valid_and_prints_the_marker():
    code = build_innovation_baseline_probe_code()
    compile(code, "<probe>", "exec")
    assert "__SCAI_INNOVATION_BASELINE_PROBE__" in code
    # The probe must never mutate the analysis state it inspects.
    for forbidden in ("savefig", "plt.", "del ", "drop("):
        assert forbidden not in code


def _make_adata():
    """An innovation-round object: registered scores plus a comparison field."""
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(11)
    n_obs = 400
    obs = pd.DataFrame(
        {
            "condition": np.repeat(["CON", "DKO"], n_obs // 2),
            "stress_score": rng.normal(size=n_obs),
            "redox_score": rng.normal(loc=0.4, size=n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    return ad.AnnData(
        X=rng.gamma(1.5, 1.0, size=(n_obs, 20)).astype("float32"),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
    )


def _make_de_table(rows=300):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "gene": [f"gene_{i}" for i in range(rows)],
        "log2FoldChange": rng.normal(scale=1.6, size=rows),
        "padj": rng.uniform(1e-12, 1.0, size=rows),
    })


def _make_enrichment_table(rows=40):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "Term": [f"pathway term number {i}" for i in range(rows)],
        "Adjusted P-value": rng.uniform(1e-9, 0.4, size=rows),
        "Combined Score": rng.uniform(1, 90, size=rows),
    })


def _exec_probe(namespace):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(
            compile(
                build_innovation_baseline_probe_code(), "<innovation-probe>", "exec"
            ),
            namespace,
            namespace,
        )
    return parse_innovation_baseline_probe_output(buffer.getvalue())


def test_real_probe_discovers_scores_differential_and_enrichment_results():
    pytest.importorskip("anndata")
    parsed = _exec_probe({
        "adata": _make_adata(),
        "deg_results": _make_de_table(),
        "go_bp": _make_enrichment_table(),
    })

    assert parsed is not None
    assert parsed["var"] == "adata"
    assert parsed["group_key"] == "condition"
    assert parsed["group_levels"] == ["CON", "DKO"]
    assert parsed["score_keys"] == ["stress_score", "redox_score"]
    assert [t["var"] for t in parsed["de_tables"]] == ["deg_results"]
    assert parsed["de_tables"][0]["lfc"] == "log2FoldChange"
    assert [t["var"] for t in parsed["enrichment_tables"]] == ["go_bp"]
    assert parsed["enrichment_tables"][0]["term"] == "Term"


def test_real_probe_does_not_mistake_a_small_table_for_a_volcano():
    """A handful of rows is a summary table, not an effect-size landscape."""
    pytest.importorskip("anndata")
    parsed = _exec_probe({
        "adata": _make_adata(),
        "tiny_de": _make_de_table(rows=5),
    })
    assert parsed is not None
    assert parsed["de_tables"] == []


def test_all_generated_panels_pass_the_sandbox_code_validator():
    from backend.utils.sandbox import validate_code_safety

    pytest.importorskip("anndata")
    probe = _exec_probe({
        "adata": _make_adata(),
        "deg_results": _make_de_table(),
        "go_bp": _make_enrichment_table(),
    })
    specs = build_innovation_baseline_figure_specs(probe, round_number=3)
    assert len(specs) == 3
    for spec in specs:
        assert validate_code_safety(spec.code) == (True, "")


def test_all_real_panels_render_and_pass_the_blocking_layout_gate():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from backend.agents.bioinformatics.kernel_templates import LAYOUT_LINT_SRC
    from backend.services.figure_layout import blocking_violations

    namespace = {
        "adata": _make_adata(),
        "deg_results": _make_de_table(),
        "go_bp": _make_enrichment_table(),
    }
    probe = _exec_probe(namespace)
    specs = build_innovation_baseline_figure_specs(probe, round_number=3)
    assert len(specs) == 3

    exec(LAYOUT_LINT_SRC, namespace, namespace)
    for spec in specs:
        plt.close("all")
        exec(compile(spec.code, f"<{spec.name}>", "exec"), namespace, namespace)
        assert len(plt.get_fignums()) == 1, spec.name
        figure = plt.figure(plt.get_fignums()[0])
        violations = namespace["_scai_lint_figure_layout"](figure)
        assert blocking_violations(violations) == [], (
            f"{spec.name} must pass the deterministic layout gate: {violations}"
        )
    plt.close("all")


class _Result:
    def __init__(self, output="", error=None):
        self.output = output
        self.error = error


class _Sandbox:
    def __init__(self, result=None, raises=False):
        self.result = result
        self.raises = raises

    async def execute(self, code, **kwargs):
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


class _State:
    def __init__(self, current_round=3, publication_figures=()):
        self.current_round = current_round
        self.publication_figures = list(publication_figures)
        self.restore_branch_id = None


def _probe_line():
    return "__SCAI_INNOVATION_BASELINE_PROBE__" + json.dumps({
        "var": "adata",
        "n_obs": 400,
        "group_key": "condition",
        "group_levels": ["CON", "DKO"],
        "score_keys": ["stress_score"],
        "de_tables": [{
            "var": "deg_results", "lfc": "log2FoldChange",
            "pval": "padj", "gene": "gene", "n_rows": 300,
        }],
        "enrichment_tables": [],
    })


def test_service_registers_each_panel_as_platform_rendered():
    import asyncio

    agent = _Agent(_Sandbox(_Result(output=_probe_line())))
    rendered = asyncio.run(
        ensure_innovation_baseline_figures(agent, _State(current_round=3))
    )

    assert sorted(rendered) == [
        "r3_platform_score_distribution", "r3_platform_volcano_deg_results"
    ]
    for call in agent.figure_calls:
        assert call["diagnostic"] is True
        assert call["purpose"] == "baseline"
        assert call["review_callback"] is None


def test_one_failed_panel_does_not_suppress_the_others():
    import asyncio

    agent = _Agent(
        _Sandbox(_Result(output=_probe_line())),
        fail_name="r3_platform_score_distribution",
    )
    rendered = asyncio.run(
        ensure_innovation_baseline_figures(agent, _State(current_round=3))
    )
    assert list(rendered) == ["r3_platform_volcano_deg_results"]


@pytest.mark.parametrize("sandbox", [
    _Sandbox(_Result(output="nothing useful")),
    _Sandbox(_Result(output="", error="kernel exploded")),
    _Sandbox(raises=True),
])
def test_probe_failures_are_a_silent_skip(sandbox):
    import asyncio

    agent = _Agent(sandbox)
    assert asyncio.run(
        ensure_innovation_baseline_figures(agent, _State())
    ) == {}
    assert agent.figure_calls == []


def test_platform_name_cannot_overwrite_an_analysis_owned_figure():
    import asyncio
    from types import SimpleNamespace

    owned = SimpleNamespace(
        name="r3_platform_volcano_deg_results",
        lineage_id="main",
        diagnostic=False,
        purpose="result",
    )
    agent = _Agent(_Sandbox(_Result(output=_probe_line())))
    rendered = asyncio.run(ensure_innovation_baseline_figures(
        agent, _State(current_round=3, publication_figures=[owned])
    ))

    assert "r3_platform_volcano_deg_results" not in rendered
    assert (
        "r3_platform_volcano_deg_results_platform_baseline" in rendered
    )


def test_round_one_keeps_its_own_baselines_and_rounds_two_plus_get_this_floor():
    """Round 1 is already covered by the Stage 1/Stage 2 baselines.

    Rendering both there would duplicate the representation panels, so this
    floor starts where that one stops. Integration rounds never reach the
    executor, so the round number is the whole condition.
    """
    assert should_render_innovation_baselines(0) is False
    assert should_render_innovation_baselines(1) is False
    assert should_render_innovation_baselines(2) is True
    assert should_render_innovation_baselines(9) is True
    assert should_render_innovation_baselines(None) is False


def test_workflow_node_actually_calls_the_service():
    """The floor only exists if the round's node invokes it.

    A unit-tested service that nothing calls is the failure mode this guard
    exists for; the node function itself is too heavily wired to construct in a
    unit test, so its call site is asserted on the source.
    """
    import ast
    import inspect

    from backend.workflow.agent_nodes import bioinformatics_node

    source = inspect.getsource(bioinformatics_node)
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "ensure_innovation_baseline_figures" in called
    assert "should_render_innovation_baselines" in called
