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
)


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


# The vocabulary these panels are actually pointed at. The fixtures above are
# DESeq2- and Enrichr-shaped; this platform's own executor is told to build
# differential frames with `sc.get.rank_genes_groups_df`, whose columns are
# names / scores / logfoldchanges / pvals / pvals_adj -- and every defect below
# was invisible precisely because no fixture used them.
