import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.bioinformatics_agent import (
    _build_exec_autosave_wrapper,
    _build_resource_guard_preamble,
    _high_memory_referenced_names,
)
from backend.utils.sandbox import validate_code_safety


def test_scientific_stack_is_pinned_to_bounded_compatible_versions():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    for requirement in (
        "numpy==1.26.4",
        "numba==0.61.2",
        "llvmlite==0.44.0",
        "scanpy==1.11.5",
        "anndata==0.11.4",
        "scvelo==0.3.3",
        "cellrank==2.0.7",
        "decoupler==1.6.0",
        "scvi-tools==1.3.3",
    ):
        assert requirement in requirements


def test_velocity_code_gets_a_memory_preflight_for_referenced_anndata():
    code = "scv.pp.moments(adata_mes)\nscv.tl.velocity_graph(adata_mes)"
    assert "adata_mes" in _high_memory_referenced_names(code)
    preamble = _build_resource_guard_preamble(code)
    assert "ScAI resource preflight" in preamble
    assert "MemoryError" in preamble
    assert "cgroup" in preamble


def test_regular_scanpy_code_does_not_get_velocity_memory_guard():
    assert _build_resource_guard_preamble("sc.tl.umap(adata)") == ""


def test_full_matrix_scale_and_toarray_get_dense_memory_guard():
    assert _high_memory_referenced_names("sc.pp.scale(adata)") == ["adata"]
    assert _high_memory_referenced_names("dense = adata.X.toarray()") == ["adata"]
    assert "full-matrix dense transformation" in _build_resource_guard_preamble(
        "sc.pp.scale(adata)"
    )


def test_sparse_scale_without_centering_does_not_get_dense_guard():
    assert _build_resource_guard_preamble("sc.pp.scale(adata, zero_center=False)") == ""


def test_memory_preflight_targets_call_argument_not_source_used_to_build_subset():
    code = "adata_small = adata[mask].copy()\nscv.tl.velocity_graph(adata_small)"
    assert _high_memory_referenced_names(code) == ["adata_small"]


def test_cellrank_mention_without_operation_does_not_trigger_guard():
    code = "# cellrank may be considered later\nprint(adata.n_obs)"
    assert _high_memory_referenced_names(code) == []


def test_velocity_memory_preflight_fails_before_operation_when_headroom_is_low():
    class _Matrix:
        nbytes = 900 * 1024**2

    adata = SimpleNamespace(
        n_obs=100_000,
        n_vars=2_000,
        X=_Matrix(),
        layers={},
        obsp={},
        varp={},
        obsm={},
        varm={},
    )

    def _fake_open(path, *args, **kwargs):
        if path == "/proc/meminfo":
            return io.StringIO("MemAvailable: 1048576 kB\n")
        raise FileNotFoundError(path)

    namespace = {"adata_mes": adata, "open": _fake_open}
    with pytest.raises(MemoryError, match="Subset/downsample"):
        exec(_build_resource_guard_preamble("scv.tl.velocity_graph(adata_mes)"), namespace)


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
