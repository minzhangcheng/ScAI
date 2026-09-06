"""Tests for renderer-owned figure content and publication side effects."""

from __future__ import annotations

import ast
import json
import linecache
import os
from pathlib import Path

import pytest

from backend.agents.bioinformatics.figures import _publication_output_error
from backend.agents.bioinformatics.kernel_templates import (
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
)
from backend.services.figure_layout import LAYOUT_LINT_SENTINEL
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes
from backend.utils.error_parsing import strip_ansi


def _census_line(artists: int, axes: int = 1, texts=()) -> str:
    from backend.services.figure_content import FIGURE_CONTENT_SENTINEL

    return FIGURE_CONTENT_SENTINEL + json.dumps(
        {"data_artists": artists, "axes": axes, "texts": list(texts)}
    )


def _run_wrapper(tmp_path: Path, code: str, capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_close = plt.close
    original_show = plt.show
    original_plt_savefig = plt.savefig
    original_figure_savefig = plt.Figure.savefig
    wrapper = _build_publication_figure_wrapper(
        code,
        str(tmp_path),
        str(tmp_path / "canonical.jpg"),
        str(tmp_path / "canonical.pdf"),
    )
    namespace = {
        "_original_plt_close": original_close,
        "_original_plt_show": original_show,
        "_original_plt_savefig": original_plt_savefig,
        "_capturing_show": original_show,
        "_combined_savefig": original_figure_savefig,
    }
    with matplotlib.rc_context():
        exec(compile(wrapper, "<publication-wrapper>", "exec"), namespace, namespace)
    return capsys.readouterr().out, namespace, original_close, original_show, original_plt_savefig, original_figure_savefig


def test_sandbox_source_is_valid_python():
    from backend.services.figure_content import FIGURE_CONTENT_SRC

    ast.parse(FIGURE_CONTENT_SRC)


def test_unmeasured_content_is_not_a_verdict():
    from backend.services.figure_content import (
        FIGURE_CONTENT_ERROR_SENTINEL,
        census_ran,
        empty_render_reason,
        parse_figure_content,
    )

    outputs = (
        None,
        "",
        "Saved JPG: a.jpg",
        FIGURE_CONTENT_ERROR_SENTINEL + " renderer failed",
        "__SCAI_FIGURE_CONTENT__",
        "__SCAI_FIGURE_CONTENT__not-json",
        "__SCAI_FIGURE_CONTENT__" + json.dumps({"data_artists": "many"}),
        "__SCAI_FIGURE_CONTENT__"
        + json.dumps({"data_artists": True, "axes": 1, "texts": []}),
        "__SCAI_FIGURE_CONTENT__"
        + json.dumps({"data_artists": -1, "axes": 1, "texts": []}),
        "__SCAI_FIGURE_CONTENT__"
        + json.dumps({"data_artists": 0, "axes": False, "texts": []}),
        "__SCAI_FIGURE_CONTENT__"
        + json.dumps({"data_artists": 0, "axes": 1, "texts": ["ok", 3]}),
    )
    for output in outputs:
        assert parse_figure_content(output) is None
        assert census_ran(output) is False
        assert empty_render_reason(output) is None


def test_last_content_event_wins():
    from backend.services.figure_content import empty_render_reason, parse_figure_content

    output = _census_line(0) + "\n" + _census_line(3)
    assert parse_figure_content(output)["data_artists"] == 3
    assert empty_render_reason(output) is None


def test_zero_data_artist_output_is_rejected(tmp_path):
    from backend.services.figure_content import empty_render_reason

    jpg = tmp_path / "figure.jpg"
    pdf = tmp_path / "figure.pdf"
    jpg.write_bytes(valid_image_bytes("JPEG", min_bytes=4000))
    pdf.write_bytes(valid_pdf_bytes())
    output = "Saved JPG: figure.jpg\nSaved PDF: figure.pdf\n" + _census_line(
        0, axes=2, texts=["data not found"]
    )
    assert empty_render_reason(output)
    reason = _publication_output_error(jpg, pdf, output)
    assert reason and "0 data artists" in reason


def test_representative_data_artists_are_not_empty(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from backend.services.figure_content import FIGURE_CONTENT_SRC, FIGURE_CONTENT_SENTINEL

    namespace = {}
    exec(compile(FIGURE_CONTENT_SRC, "<figure-content>", "exec"), namespace, namespace)
    cases = []
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "bare text")
    cases.append(("bare text", fig, namespace["_scai_census_figure_content"](fig)))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter([], [])
    cases.append(("empty scatter", fig, namespace["_scai_census_figure_content"](fig)))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar([], [])
    cases.append(("empty bar", fig, namespace["_scai_census_figure_content"](fig)))
    plt.close(fig)

    for label, _fig, census in cases:
        assert census["data_artists"] == 0, (label, census)

    positive_cases = []
    fig, ax = plt.subplots()
    ax.scatter([1, 2], [2, 3])
    positive_cases.append(("scatter", fig, namespace["_scai_census_figure_content"](fig)))

    fig, ax = plt.subplots()
    ax.imshow([[1, 2], [3, 4]])
    positive_cases.append(("image", fig, namespace["_scai_census_figure_content"](fig)))

    fig, ax = plt.subplots()
    ax.table(cellText=[["a", "b"]], loc="upper right")
    positive_cases.append(("table", fig, namespace["_scai_census_figure_content"](fig)))

    fig, ax = plt.subplots()
    ax.annotate("arrow", xy=(1, 2), xytext=(2, 3), arrowprops={"arrowstyle": "->"})
    positive_cases.append(("arrow", fig, namespace["_scai_census_figure_content"](fig)))

    for label, _fig, census in positive_cases:
        assert census["data_artists"] > 0, (label, census)
    for _label, fig, _census in positive_cases:
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [2, 1, 3])
    census = namespace["_scai_census_figure_content"](fig)
    assert census["data_artists"] > 0, census
    jpg = tmp_path / "real-plot.jpg"
    pdf = tmp_path / "real-plot.pdf"
    fig.savefig(jpg)
    fig.savefig(pdf)
    output = (
        f"Saved JPG: {jpg}\nSaved PDF: {pdf}\n"
        + FIGURE_CONTENT_SENTINEL
        + json.dumps(census)
    )
    assert _publication_output_error(jpg, pdf, output) is None
    plt.close(fig)


def test_publication_wrapper_owns_save_and_restores_renderer(tmp_path, capsys):
    side_a = tmp_path / "user-side-a.jpg"
    side_b = tmp_path / "user-side-b.jpg"
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [2, 1, 3])\n"
        f"fig.savefig({str(side_a)!r})\n"
        f"plt.savefig({str(side_b)!r})\n"
        "plt.show()\n"
        "plt.close('all')\n"
    )
    output, namespace, original_close, original_show, original_plt_savefig, original_figure_savefig = _run_wrapper(
        tmp_path, code, capsys
    )
    from backend.services.figure_content import census_ran

    assert census_ran(output), output
    assert (tmp_path / "canonical.jpg").is_file()
    assert (tmp_path / "canonical.pdf").is_file()
    assert not side_a.exists()
    assert not side_b.exists()
    import matplotlib.pyplot as plt

    assert plt.close is original_close
    assert plt.show is original_show
    assert plt.savefig is original_plt_savefig
    assert plt.Figure.savefig is original_figure_savefig


@pytest.mark.parametrize("builder_name", [
    "_build_exec_autosave_wrapper",
    "_build_publication_figure_wrapper",
])
def test_user_code_traceback_retains_source_line(tmp_path, builder_name):
    side_a = tmp_path / f"{builder_name}-side-a.jpg"
    side_b = tmp_path / f"{builder_name}-side-b.jpg"
    source_lines = [
        "import matplotlib.pyplot as plt\n",
        "fig, ax = plt.subplots()\n",
        "ax.plot([1, 2], [2, 1])\n",
    ]
    if builder_name == "_build_publication_figure_wrapper":
        source_lines.extend([
            f"fig.savefig({str(side_a)!r})\n",
            f"plt.savefig({str(side_b)!r})\n",
            "plt.show()\n",
            "plt.close('all')\n",
        ])
    source_lines.append("raise RuntimeError('known render failure')\n")
    source = "".join(source_lines)
    if builder_name == "_build_exec_autosave_wrapper":
        wrapper = _build_exec_autosave_wrapper(source, str(tmp_path), "trace")
    else:
        wrapper = _build_publication_figure_wrapper(
            source, str(tmp_path), str(tmp_path / "trace.jpg"), str(tmp_path / "trace.pdf")
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_close = plt.close
    original_show = plt.show
    original_savefig = plt.Figure.savefig
    namespace = {
        "_original_plt_close": original_close,
        "_original_plt_show": original_show,
        "_original_plt_savefig": plt.savefig,
        "_capturing_show": original_show,
        "_combined_savefig": original_savefig,
    }
    with matplotlib.rc_context():
        with pytest.raises(RuntimeError) as excinfo:
            exec(compile(wrapper, "<wrapper>", "exec"), namespace, namespace)
    assert linecache.cache["<scai-user-code>"] == (
        len(source), None, source.splitlines(True), "<scai-user-code>"
    )
    assert "raise RuntimeError('known render failure')" in "".join(
        __import__("traceback").format_exception(
            type(excinfo.value), excinfo.value, excinfo.value.__traceback__
        )
    )
    assert excinfo.type is RuntimeError
    if builder_name == "_build_publication_figure_wrapper":
        assert not side_a.exists()
        assert not side_b.exists()
    assert plt.close is original_close
    assert plt.show is original_show
    assert plt.savefig is namespace["_original_plt_savefig"]
    assert plt.Figure.savefig is original_savefig


def test_vector_only_doubled_suffix_recovers_jpg(tmp_path, capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = tmp_path / "dotplot__markers.jpg.pdf"
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [1, 2, 3])\n"
        f"fig.savefig({str(target)!r})\n"
    )
    wrapper = _build_exec_autosave_wrapper(code, str(tmp_path), "vector")
    namespace = {
        "_original_plt_close": plt.close,
        "_original_plt_show": plt.show,
        "_original_plt_savefig": plt.savefig,
        "_capturing_show": plt.show,
        "_combined_savefig": plt.Figure.savefig,
    }
    with matplotlib.rc_context():
        exec(compile(wrapper, "<wrapper>", "exec"), namespace, namespace)
    assert target.is_file()
    assert target.with_name("dotplot__markers.jpg").is_file()
    assert not (tmp_path / "dotplot__markers.jpg.jpg").exists()
    assert not (tmp_path / "dotplot__markers.jpg.pdf.jpg").exists()


def test_content_event_precedes_layout_event(tmp_path, capsys):
    output, *_ = _run_wrapper(
        tmp_path,
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1, 2], [2, 1])\n",
        capsys,
    )
    from backend.services.figure_content import FIGURE_CONTENT_SENTINEL

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    content_at = max(i for i, line in enumerate(lines) if line.startswith(FIGURE_CONTENT_SENTINEL))
    layout_at = max(i for i, line in enumerate(lines) if line.startswith(LAYOUT_LINT_SENTINEL))
    assert content_at < layout_at


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_s3_renderer_contracts(tmp_path):
    from backend.agents.bioinformatics.kernel_templates import (
        _build_exec_autosave_wrapper,
        _build_publication_figure_wrapper,
    )
    from backend.utils.sandbox import LocalJupyterSandbox

    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    sandbox = LocalJupyterSandbox(
        session_id="s3-live-renderer",
        state_dir=str(tmp_path / "states"),
        working_dir=str(workspace),
        timeout=120,
    )
    await sandbox.start()
    try:
        forced_png = figures / "forced.png"
        forced_jpg = figures / "forced.jpg"
        forced_conversion = await sandbox.execute(
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            f"_forced_png = Path({str(forced_png)!r})\n"
            f"_forced_jpg = Path({str(forced_jpg)!r})\n"
            "_forced_seen = {}\n"
            "_forced_original = _original_savefig\n"
            "def _forced_probe(fig, filename, *args, **kwargs):\n"
            "    _forced_seen.update(kwargs)\n"
            "    return _forced_original(fig, filename, *args, **kwargs)\n"
            "_original_savefig = _forced_probe\n"
            "try:\n"
            "    _forced_fig, _forced_ax = plt.subplots()\n"
            "    _forced_ax.plot([1, 2], [2, 1])\n"
            "    _forced_fig.savefig(_forced_png, metadata={'drop': 'me'}, dpi=77, facecolor='yellow')\n"
            "finally:\n"
            "    _original_savefig = _forced_original\n"
            "assert _forced_jpg.is_file()\n"
            "assert not _forced_png.is_file()\n"
            "assert 'metadata' not in _forced_seen\n"
            "assert _forced_seen['dpi'] == 77\n"
            "assert _forced_seen['facecolor'] == 'yellow'\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert forced_conversion.success, forced_conversion.error

        scanpy_code = (
            "import anndata as ad\n"
            "import numpy as np\n"
            "import scanpy as sc\n"
            "adata = ad.AnnData(np.array([[1.0, 2.0], [2.0, 1.0]]))\n"
            "adata.var_names = ['x', 'y']\n"
            "sc.settings.figdir = r'" + str(figures) + "'\n"
            "sc.pl.scatter(adata, x='x', y='y', show=False, save='_markers.jpg')\n"
        )
        scanpy_wrapper = _build_exec_autosave_wrapper(
            scanpy_code, str(figures), "live"
        )
        scanpy_result = await sandbox.execute(
            scanpy_wrapper, records_history=False, skip_safety_check=True
        )
        assert scanpy_result.success, scanpy_result.error
        recovered = figures / "scatter_markers.jpg"
        assert recovered.is_file(), sorted(p.name for p in figures.iterdir())
        assert (figures / "scatter_markers.jpg.pdf").is_file()

        side = workspace / "user-side.jpg"
        publication_code = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2, 3], [3, 2, 1])\n"
            f"fig.savefig({str(side)!r})\n"
            f"plt.savefig({str(workspace / 'user-side-2.jpg')!r})\n"
            "plt.show()\n"
            "plt.close('all')\n"
        )
        publication_wrapper = _build_publication_figure_wrapper(
            publication_code,
            str(figures),
            str(figures / "canonical.jpg"),
            str(figures / "canonical.pdf"),
        )
        publication_result = await sandbox.execute(
            publication_wrapper, records_history=False, skip_safety_check=True
        )
        assert publication_result.success, publication_result.error
        assert (figures / "canonical.jpg").is_file()
        assert (figures / "canonical.pdf").is_file()
        assert not side.exists()
        assert not (workspace / "user-side-2.jpg").exists()
        identity = await sandbox.execute(
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.figure as _figure\n"
            "assert _figure.Figure.savefig is _combined_savefig\n"
            "assert plt.close is _original_plt_close\n"
            "assert plt.show is _capturing_show\n"
            "assert plt.savefig is _original_plt_savefig\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert identity.success, identity.error

        exception_side_a = workspace / "exception-side-a.jpg"
        exception_side_b = workspace / "exception-side-b.jpg"
        exception_code = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2], [2, 1])\n"
            f"fig.savefig({str(exception_side_a)!r})\n"
            f"plt.savefig({str(exception_side_b)!r})\n"
            "plt.show()\n"
            "plt.close('all')\n"
            "raise RuntimeError('live known render failure')\n"
        )
        exception_wrapper = _build_publication_figure_wrapper(
            exception_code,
            str(figures),
            str(figures / "exception-canonical.jpg"),
            str(figures / "exception-canonical.pdf"),
        )
        exception_result = await sandbox.execute(
            exception_wrapper, records_history=False, skip_safety_check=True
        )
        assert not exception_result.success
        assert "RuntimeError" in exception_result.error
        assert "<scai-user-code>" in exception_result.error
        assert "raise RuntimeError('live known render failure')" in strip_ansi(
            exception_result.error
        )
        assert not exception_side_a.exists()
        assert not exception_side_b.exists()
        assert (figures / "exception-canonical.jpg").is_file()
        assert (figures / "exception-canonical.pdf").is_file()
        exception_identity = await sandbox.execute(
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.figure as _figure\n"
            "assert _figure.Figure.savefig is _combined_savefig\n"
            "assert plt.close is _original_plt_close\n"
            "assert plt.show is _capturing_show\n"
            "assert plt.savefig is _original_plt_savefig\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert exception_identity.success, exception_identity.error
    finally:
        await sandbox.stop(save=False)
