from pathlib import Path

import pytest

from backend.agents.bioinformatics_agent import _build_exec_autosave_wrapper
from backend.utils.sandbox import validate_code_safety


def test_scientific_stack_is_pinned_to_bounded_compatible_versions():
    # decoupler is installed with --no-deps from requirements-nodeps.txt: its
    # own metadata caps numba below the pinned 0.61.2, so leaving it in the
    # resolved graph makes pip either downgrade numba or fail outright. It is
    # still exact-pinned, just in the other file -- see requirements-nodeps.txt
    # and backend/tests/test_decoupler_numba_pin.py.
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    nodeps_path = Path("requirements-nodeps.txt")
    nodeps = nodeps_path.read_text(encoding="utf-8") if nodeps_path.exists() else ""
    pinned = requirements + "\n" + nodeps
    for requirement in (
        "numpy==1.26.4",
        "numba==0.61.2",
        "llvmlite==0.44.0",
        "scanpy==1.11.5",
        "anndata==0.11.4",
        "scvelo==0.3.3",
        "cellrank==2.0.7",
        "decoupler==1.8.0",
        "scvi-tools==1.3.3",
    ):
        assert requirement in pinned


def test_autosave_cleanup_reimports_matplotlib_after_user_namespace_mutation(tmp_path):
    wrapper = _build_exec_autosave_wrapper(
        "plt = 'shadowed by generated code'",
        str(tmp_path),
        "test_namespace_cleanup",
    )
    namespace = {}
    exec(wrapper, namespace, namespace)

    import matplotlib.pyplot as pyplot

    assert callable(pyplot.close)
    assert "import matplotlib.pyplot as _scai_cleanup_plt" in wrapper


def test_autosave_cleanup_restores_capturing_show(tmp_path):
    import matplotlib.pyplot as pyplot

    previous_show = pyplot.show

    def capturing_show(*args, **kwargs):
        return None

    namespace = {
        "_capturing_show": capturing_show,
        "_original_plt_show": previous_show,
    }
    try:
        exec(_build_exec_autosave_wrapper(
            "print('done')", str(tmp_path), "test_show_cleanup"
        ), namespace, namespace)
        assert pyplot.show is capturing_show
    finally:
        pyplot.show = previous_show


def test_static_write_check_follows_paths_derived_from_workspace():
    code = """
from pathlib import Path
out_dir = Path(working_dir) / 'tables'
out_path = out_dir / 'summary.json'
with open(out_path, 'w', encoding='utf-8') as handle:
    handle.write('{}')
"""
    assert validate_code_safety(code) == (True, "")


def test_static_write_check_still_rejects_literal_outside_path():
    safe, reason = validate_code_safety("open('/etc/scai-output', 'w').write('x')")
    assert safe is False
    assert "file write" in reason


@pytest.mark.parametrize("path", [
    "tables/summary.json",
    "figures/metadata.txt",
    "./annotation_validation/result.json",
])
def test_static_write_check_accepts_safe_relative_literal_paths(path):
    assert validate_code_safety(f"open({path!r}, 'w').write('x')") == (True, "")
