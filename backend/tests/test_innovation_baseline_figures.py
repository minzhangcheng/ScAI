"""Deterministic result baselines for innovation rounds.

Stage 1 gets its visual floor from ``stage1_baseline_figures``; rounds 2+ had
none, so a round that computed a great deal and drew nothing settled that way.
These tests pin the probe contract and the spec builder that give those rounds
the same platform-owned floor.
"""

import json
from types import SimpleNamespace

import pytest

from backend.models.schemas import CodeExecution

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
    code = build_innovation_baseline_probe_code(["adata"])
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


def _exec_probe(namespace, output_names):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(
            compile(
                build_innovation_baseline_probe_code(output_names),
                "<innovation-probe>",
                "exec",
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
    }, ["adata", "deg_results", "go_bp"])

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
    }, ["adata", "tiny_de"])
    assert parsed is not None
    assert parsed["de_tables"] == []


def test_real_probe_uses_only_the_explicit_output_allowlist():
    pytest.importorskip("anndata")
    parsed = _exec_probe(
        {
            "old_adata": _make_adata(),
            "old_de": _make_de_table(),
            "current_adata": _make_adata(),
            "current_de": _make_de_table(),
        },
        ["current_adata", "current_de"],
    )

    assert parsed is not None
    assert parsed["var"] == "current_adata"
    assert [table["var"] for table in parsed["de_tables"]] == ["current_de"]


def test_empty_output_allowlist_does_not_scan_result_shaped_globals():
    pytest.importorskip("anndata")
    parsed = _exec_probe(
        {"old_adata": _make_adata(), "old_de": _make_de_table()}, []
    )
    assert parsed is None


def test_all_generated_panels_pass_the_sandbox_code_validator():
    from backend.utils.sandbox import validate_code_safety

    pytest.importorskip("anndata")
    probe = _exec_probe({
        "adata": _make_adata(),
        "deg_results": _make_de_table(),
        "go_bp": _make_enrichment_table(),
    }, ["adata", "deg_results", "go_bp"])
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
    probe = _exec_probe(namespace, ["adata", "deg_results", "go_bp"])
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
        self.codes = []

    async def execute(self, code, **kwargs):
        self.codes.append(code)
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


def _receipt(
    *,
    outputs=("adata", "deg_results"),
    round_number=3,
    plan_id="plan-1",
    lineage_id="main",
    analysis_generation=4,
    cohort_generation=2,
    cohort_digest="d" * 64,
    **overrides,
):
    values = {
        "id": "receipt-current",
        "code": "current_analysis()",
        "round_number": round_number,
        "evidence_schema_version": 2,
        "plan_id": plan_id,
        "execution_status": "succeeded",
        "lineage_id": lineage_id,
        "analysis_generation": analysis_generation,
        "cohort_generation": cohort_generation,
        "cohort_digest": cohort_digest,
        "realized_persistent_outputs": list(outputs),
    }
    values.update(overrides)
    return CodeExecution(**values)


class _State:
    def __init__(self, current_round=3, publication_figures=(), code_history=None):
        self.current_round = current_round
        self.publication_figures = list(publication_figures)
        self.restore_branch_id = "main"
        self.current_plan = SimpleNamespace(id="plan-1")
        self.sandbox_state = {
            "analysis_generation": 4,
            "cohort_identity_v1": {
                "cohort_generation": 2,
                "cohort_digest": "d" * 64,
            },
        }
        self.code_history = list(
            code_history if code_history is not None else [_receipt()]
        )


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
    assert "'adata'" in agent.sandbox.codes[0]
    assert "'deg_results'" in agent.sandbox.codes[0]
    for call in agent.figure_calls:
        assert call["diagnostic"] is True
        assert call["purpose"] == "baseline"
        assert call["review_callback"] is None


def test_service_probe_allowlist_comes_from_current_receipt_outputs():
    import asyncio

    sandbox = _Sandbox(_Result(output=_probe_line()))
    state = _State(
        code_history=[_receipt(outputs=("current_adata", "current_de"))]
    )
    agent = _Agent(sandbox)
    asyncio.run(ensure_innovation_baseline_figures(agent, state))

    assert "'current_adata'" in sandbox.codes[0]
    assert "'current_de'" in sandbox.codes[0]
    assert "old_adata" not in sandbox.codes[0]
    assert "globals().items" not in sandbox.codes[0]


@pytest.mark.parametrize("receipt", [
    _receipt(round_number=2),
    _receipt(plan_id="plan-old"),
    _receipt(lineage_id="branch-old"),
    _receipt(analysis_generation=3),
    _receipt(cohort_generation=1),
    _receipt(cohort_digest="e" * 64),
    _receipt(execution_status="failed"),
    _receipt(error="execution failed"),
    _receipt(evidence_schema_version=1),
])
def test_non_current_or_failed_receipts_produce_no_baseline(receipt):
    import asyncio

    sandbox = _Sandbox(_Result(output=_probe_line()))
    agent = _Agent(sandbox)
    assert asyncio.run(
        ensure_innovation_baseline_figures(
            agent, _State(code_history=[receipt])
        )
    ) == {}
    assert sandbox.codes == []
    assert agent.figure_calls == []


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


# ---------------------------------------------------------------------------
# The vocabulary these panels are actually pointed at. The fixtures above are
# DESeq2- and Enrichr-shaped; this platform's own executor is told to build
# differential frames with `sc.get.rank_genes_groups_df`, whose columns are
# names / scores / logfoldchanges / pvals / pvals_adj -- and every defect below
# was invisible precisely because no fixture used them.
# ---------------------------------------------------------------------------


def _make_scanpy_de_table(rows=2000):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "names": [f"gene_{i}" for i in range(rows)],
        "scores": rng.normal(size=rows),
        "logfoldchanges": rng.normal(scale=1.5, size=rows),
        "pvals": rng.uniform(1e-12, 1.0, size=rows),
        "pvals_adj": rng.uniform(1e-9, 1.0, size=rows),
    })


def test_the_volcano_uses_the_adjusted_p_value_when_the_table_has_one():
    """None of the "adjusted" hints is a substring of scanpy's `pvals_adj`, while
    the generic `pval` IS a substring of its `pvals`. Matching by hint order alone
    handed the raw p-value to a panel that draws a 0.05 line and calls it
    significance."""
    parsed = _exec_probe({"marker_de": _make_scanpy_de_table()}, ["marker_de"])

    assert parsed is not None
    assert parsed["de_tables"][0]["pval"] == "pvals_adj"


def test_a_short_differential_table_produces_no_panel_at_all():
    """A table too short to be an effect-size landscape is not one -- and it is
    not a term enrichment either. `_TERM_HINTS` contains "name", which
    substring-matches scanpy's `names`, so falling through published gene symbols
    as a ranked bar chart of enriched terms. The executor is told to take per
    cluster top-N frames, so short differential tables are routine."""
    parsed = _exec_probe({"top20": _make_scanpy_de_table(rows=20)}, ["top20"])

    assert parsed is None, "a 20-row differential frame supports no baseline panel"


def test_table_panels_survive_a_kernel_with_no_anndata():
    """The volcano and the enrichment chart need only pandas. Sharing one try
    with the AnnData scan meant an import failure there suppressed them too,
    against this module's rule that a panel is suppressed only by its own
    missing input."""
    import sys

    pytest.importorskip("pandas")
    namespace = {"marker_de": _make_scanpy_de_table(), "go_bp": _make_enrichment_table()}

    saved = sys.modules.get("anndata", "absent")
    sys.modules["anndata"] = None
    try:
        parsed = _exec_probe(namespace, ["marker_de", "go_bp"])
    finally:
        if saved == "absent":
            sys.modules.pop("anndata", None)
        else:
            sys.modules["anndata"] = saved

    assert parsed is not None
    assert [t["var"] for t in parsed["de_tables"]] == ["marker_de"]
    assert [t["var"] for t in parsed["enrichment_tables"]] == ["go_bp"]
    assert parsed["var"] == "" and parsed["score_keys"] == []


def test_the_volcano_legend_counts_the_table_not_the_subsample():
    """The host must not choose a significance threshold, and the p-value
    column the platform picks may be raw rather than adjusted, so the volcano
    draws every row as one neutral series and the legend reports the full
    table's row count and the column names used -- never a classification
    derived from a hard-coded cutoff."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pytest.importorskip("numpy")

    table = _make_scanpy_de_table(rows=200_000)
    total = len(table)

    namespace = {"marker_de": table}
    parsed = _exec_probe(namespace, ["marker_de"])
    volcano = next(
        spec for spec in build_innovation_baseline_figure_specs(parsed, round_number=2)
        if spec.figure_type == "volcano"
    )

    assert "1.30103" not in volcano.code
    assert ">= 1.0" not in volcano.code
    assert "axvline" not in volcano.code
    assert "axhline" not in volcano.code

    plt.close("all")
    exec(compile(volcano.code, "<volcano>", "exec"), namespace, namespace)
    labels = [text.get_text() for text in namespace["_ib_ax"].get_legend().get_texts()]
    plt.close("all")

    assert any(f"{total:,}" in label for label in labels), labels
    assert any("logfoldchanges" in label for label in labels), labels
    assert any("pvals_adj" in label for label in labels), labels


def _scored_adata(score_names, n_obs=200):
    """An object whose obs carries the given score columns, in the given order."""
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(17)
    obs = pd.DataFrame(
        {"condition": np.repeat(["CON", "DKO"], n_obs // 2)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    for name in score_names:
        obs[name] = rng.normal(size=n_obs)
    return ad.AnnData(
        X=rng.gamma(1.5, 1.0, size=(n_obs, 8)).astype("float32"),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(8)]),
    )


def test_the_score_panel_shows_this_rounds_scores_not_round_ones_qc():
    """obs columns are appended as the analysis proceeds, so the tail is this
    round's work. Truncating from the front kept round 1's doublet and cell-cycle
    scores and dropped the signature the panel exists to show."""
    adata = _scored_adata([
        "doublet_score", "S_score", "G2M_score", "phase_score",
        "stress_score", "redox_score", "copper_signature_score",
    ])

    parsed = _exec_probe({"adata": adata}, ["adata"])

    assert parsed is not None
    assert "copper_signature_score" in parsed["score_keys"]
    assert "doublet_score" not in parsed["score_keys"]


def test_the_object_carrying_the_rounds_score_wins_over_the_bigger_one():
    """A round that scores a subset leaves the full object with only the
    inherited scores. Ranking by cell count picked the object the round's own
    result is not on."""
    full = _scored_adata(["doublet_score"], n_obs=400)
    subset = _scored_adata(["doublet_score", "copper_signature_score"], n_obs=120)

    parsed = _exec_probe(
        {"adata": full, "adata_subset": subset}, ["adata", "adata_subset"]
    )

    assert parsed is not None
    assert parsed["var"] == "adata_subset"
    assert "copper_signature_score" in parsed["score_keys"]
