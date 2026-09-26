"""Renderer-level layout linting and the report gate that reads its verdict.

The defect class these cover is the one an LLM vision reviewer names but cannot
localise: a legend anchored outside its axes landing on the neighbouring panel,
a panel letter hanging into the tick labels, tick labels crushed together. The
linter measures them from the renderer, so the assertions here are about
geometry, not about what any model said.
"""
import ast

import pytest

from backend.models.schemas import FigureReviewStatus, PublicationFigure
from backend.services.figure_layout import (
    LAYOUT_LINT_SENTINEL,
    LAYOUT_LINT_SRC,
    blocking_violations,
    lint_ran,
    parse_layout_violations,
)
from backend.services.report.artifacts import (
    eligible_artifacts,
    has_blocking_layout_defects,
    is_report_eligible,
)


# Sentinel parsing (no matplotlib needed)


@pytest.mark.parametrize(
    "prior_poison",
    ["builtins-print", "json-dumps", "plt-gcf", "sys-stdout"],
)
def test_publication_wrapper_uses_kernel_bootstrap_baselines_after_prior_pollution(
    tmp_path, capsys, monkeypatch, prior_poison
):
    import builtins
    import json as json_module
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure as mpl_figure
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from types import SimpleNamespace

    from backend.agents.bioinformatics.kernel_templates import (
        _build_publication_figure_wrapper,
    )

    wrapper = _build_publication_figure_wrapper(
        """
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 3))
fig.text(0.5, 0.5, 'OVERLAP', fontsize=12)
fig.text(0.5, 0.5, 'OVERLAP', fontsize=12)
""",
        str(tmp_path),
        str(tmp_path / "prior.jpg"),
        str(tmp_path / "prior.pdf"),
    )
    original_print = builtins.print
    original_dumps = json_module.dumps
    original_stdout = sys.stdout
    original_gcf = plt.gcf
    original_get_fignums = plt.get_fignums
    original_savefig = mpl_figure.Figure.savefig
    runtime_trust = SimpleNamespace(
        builtins_map=dict(vars(builtins)),
        print_fn=original_print,
        json_dumps=original_dumps,
        stdout=original_stdout,
        get_parent_header=lambda: None,
        set_stdout_parent=lambda _parent: None,
        plt_close=plt.close,
        plt_show=plt.show,
        plt_get_fignums=original_get_fignums,
        plt_gcf=original_gcf,
        figure_savefig=original_savefig,
        runtime_savefig=original_savefig,
    )
    monkeypatch.setitem(
        sys.modules,
        "scai_kernel_trust",
        SimpleNamespace(get_runtime_trust=lambda: runtime_trust),
    )
    namespace = {
        "_original_plt_close": plt.close,
        "_original_plt_show": plt.show,
        "_capturing_show": plt.show,
    }
    try:
        if prior_poison == "builtins-print":
            builtins.print = lambda *args, **kwargs: None
        elif prior_poison == "json-dumps":
            json_module.dumps = lambda *args, **kwargs: "[]"
        elif prior_poison == "plt-gcf":
            decoy = mpl_figure.Figure(figsize=(4, 3))
            FigureCanvasAgg(decoy)
            plt.gcf = lambda: decoy
        else:
            class _PriorFakeStdout:
                def write(self, value):
                    original_stdout.write(LAYOUT_LINT_SENTINEL + "[]\n")

                def flush(self):
                    original_stdout.flush()

            sys.stdout = _PriorFakeStdout()
        with matplotlib.rc_context():
            exec(
                compile(wrapper, "<publication-wrapper>", "exec"),
                namespace,
                namespace,
            )
    finally:
        builtins.print = original_print
        json_module.dumps = original_dumps
        sys.stdout = original_stdout
        plt.gcf = original_gcf
        plt.get_fignums = original_get_fignums
        mpl_figure.Figure.savefig = original_savefig

    output = capsys.readouterr().out
    violations = parse_layout_violations(output)
    assert lint_ran(output) is True
    assert blocking_violations(violations)
    assert output.rstrip().splitlines()[-1].startswith(LAYOUT_LINT_SENTINEL)
    assert (tmp_path / "prior.jpg").is_file()
    assert (tmp_path / "prior.pdf").is_file()


# Report eligibility

def _figure(name, status=FigureReviewStatus.APPROVED, **kwargs):
    return PublicationFigure(
        name=name, description="", figure_type="other",
        png_path=f"{name}.jpg", pdf_path=f"{name}.pdf",
        render_sha256="1" * 64, pdf_sha256="2" * 64,
        review_status=status, **kwargs,
    )


def test_blocking_layout_defect_keeps_a_figure_out_of_reports():
    """accepted_with_reservations is reachable both by a reviewer signing off
    and by the revision loop giving up, and says nothing about legibility."""
    figure = _figure(
        "covered", FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
    )
    assert has_blocking_layout_defects(figure) is True
    assert is_report_eligible(figure) is False
    assert eligible_artifacts([figure]) == []


def test_advisory_layout_defect_does_not_block():
    figure = _figure(
        "small_text", layout_violations=[{"kind": "tiny_text", "severity": "advise"}],
    )
    assert is_report_eligible(figure) is True


# The measurement itself

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def lint():
    namespace = {}
    exec(compile(LAYOUT_LINT_SRC, "<layout_lint>", "exec"), namespace, namespace)
    return namespace["_scai_lint_figure_layout"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_annotation_escaping_the_axes_reaches_the_suptitle(lint):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    fig.suptitle("FXR/bile-acid suppression and NRF2 adaptation in DKO hepatocytes")
    ax = fig.add_subplot(111)
    ax.boxplot([np.random.rand(20), np.random.rand(20) + 0.5])
    ax.text(0.5, 1.12, "p<1e-4", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, clip_on=False)

    assert "text_over_text" in {v["kind"] for v in lint(fig)}


# Wiring into the sandbox wrapper

def test_publication_wrapper_runs_the_linter_before_saving():
    from backend.agents.bioinformatics.kernel_templates import (
        _build_publication_figure_wrapper,
    )

    source = _build_publication_figure_wrapper(
        "pass", "/w/figures", "/w/figures/a.jpg", "/w/figures/a.pdf"
    )
    ast.parse(source)
    # Match the trusted call, not the linter source embedded as a string.
    call = "_scai_layout_v = _scai_trusted_lint(_scai_fig)"
    assert call in source
    assert "_scai_execute_publication_source()" in source
    # ...and before the save, or a blocking defect would be measured on a file
    # that has already been written and announced
    assert source.index(call) < source.index("Saved JPG")
    # matplotlib's own layout warnings are captured rather than left to the
    # kernel's global warnings.filterwarnings('ignore')
    assert "catch_warnings" in source
