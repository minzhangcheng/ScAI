import re
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest


def _synthetic_long_runtime_traceback(final_exception: str) -> str:
    synthetic_secret = "synthetic-fixture-credential-value"
    frames = "\n".join(
        f'  File "/home/synthetic-user/private-project/frame_{index}.py", '
        f"line {index}, in helper"
        for index in range(140)
    )
    return (
        "\x1b[31mTraceback (most recent call last):\x1b[0m\n"
        f"api_key='{synthetic_secret}'\n{frames}\n{final_exception}"
    )


def test_save_dill_template_rolls_back_failed_frame(tmp_path):
    """A failed frame must not poison the next residual object."""
    import dill

    from backend.services.snapshot_v2.runtime import _SAVE_DILL_TEMPLATE

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("intentional pickle failure")

    shared = ["GENE_A", "GENE_B"]
    namespace = {
        "results": {
            "large": list(range(100_000)),
            "failure": Unpicklable(),
        },
        "markers": {"up": shared, "down": shared},
    }
    residual = tmp_path / "residual.pkl"
    exec(
        _SAVE_DILL_TEMPLATE.format(
            names_repr=["results", "markers"],
            dill_path_repr=str(residual),
        ),
        namespace,
    )

    with residual.open("rb") as handle:
        name, restored = dill.load(handle)
        assert name == "markers"
        assert restored == {"up": shared, "down": shared}
        assert restored["up"] is restored["down"]
        with pytest.raises(EOFError):
            dill.load(handle)


@pytest.mark.asyncio
async def test_residual_capture_rejects_missing_structural_output(tmp_path):
    """Successful execution without saved/skipped metadata is not proof."""
    from backend.services.snapshot_v2.runtime import RuntimeCapture

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    (capture_dir / "residual.pkl").write_bytes(b"unverified")

    sandbox = AsyncMock()
    sandbox.execute.return_value = MagicMock(success=True, output="ok", error=None)
    object_store = MagicMock()
    capturer = RuntimeCapture(object_store, MagicMock(root=tmp_path))

    ref, saved, skipped = await capturer._capture_residual_dill(
        sandbox,
        ["results"],
        {"results": "builtins.dict"},
        capture_dir,
    )

    assert ref is None
    assert saved == []
    assert skipped == ["results"]
    object_store.put_file.assert_not_called()


class TestBuildRequiredRuntimeSet:
    def test_required_set_is_union_of_all_sources(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"adata": "AnnData", "custom_adata": "AnnData", "scratch": "Figure"},
            persistent_outputs={"scores"},
            checkpoint_dependencies={"model_input"},
        )
        assert required == {"adata", "custom_adata", "scores", "model_input"}

    def test_canonical_adata_names_included(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"adata": "AnnData", "adata_raw": "AnnData", "results": "DataFrame"},
        )
        assert "adata" in required
        assert "adata_raw" in required
        # Canonical non-AnnData names present in globals must be required.
        assert "results" in required

    def test_non_anndata_excluded_without_explicit_outputs(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"scratch": "Figure", "counts": "numpy.ndarray"},
        )
        assert required == set()

    def test_persistent_outputs_added(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"adata": "AnnData"},
            persistent_outputs={"results", "markers"},
        )
        assert "results" in required
        assert "markers" in required
        assert "adata" in required

    def test_checkpoint_dependencies_added(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"adata": "AnnData"},
            checkpoint_dependencies={"pca_embedding"},
        )
        assert "pca_embedding" in required

    def test_empty_inputs_produce_empty_set(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(globals_meta={})
        assert required == set()

    def test_all_none_inputs_produce_empty_set(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={}, persistent_outputs=None, checkpoint_dependencies=None
        )
        assert required == set()

    def test_deduplicates(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"adata": "AnnData"},
            persistent_outputs={"adata"},
            checkpoint_dependencies={"adata"},
        )
        assert required == {"adata"}

    def test_anndata_detection_via_module(self):
        from backend.services.snapshot_v2.runtime import build_required_runtime_set
        required = build_required_runtime_set(
            globals_meta={"adata": "anndata.core.AnnData", "ldata": "AnnData"},
        )
        assert required == {"adata", "ldata"}

    def test_parse_dill_skipped_empty(self):
        from backend.services.snapshot_v2.runtime import _parse_dill_skipped
        assert _parse_dill_skipped("") == []

    def test_parse_dill_skipped_found(self):
        from backend.services.snapshot_v2.runtime import _parse_dill_skipped
        output = "__DILL_SAVED__:1/3 size=100\n__DILL_SKIPPED__:['bad_var', 'other']"
        assert _parse_dill_skipped(output) == ["bad_var", "other"]

    def test_parse_dill_skipped_no_skip_line(self):
        from backend.services.snapshot_v2.runtime import _parse_dill_skipped
        output = "__DILL_SAVED__:3/3 size=100\nall good"
        assert _parse_dill_skipped(output) == []

    def test_parse_dill_skipped_structural_ast(self):
        """Structural parse must accept single-quoted Python lists without quote-replace."""
        from backend.services.snapshot_v2.runtime import _parse_dill_skipped
        output = '__DILL_SKIPPED__:["bad_var", "other"]'
        assert _parse_dill_skipped(output) == ["bad_var", "other"]
        output2 = "__DILL_SKIPPED__:['name_with_quote']"
        assert _parse_dill_skipped(output2) == ["name_with_quote"]


class TestRequiredRuntimeRegistry:
    def test_required_set_unions_canonical_anndata_tools_and_checkpoint(self):
        from backend.services.snapshot_v2.runtime_registry import build_required_runtime_manifest

        manifest = build_required_runtime_manifest(
            globals_meta={
                "adata": "anndata.AnnData",
                "results": "builtins.dict",
                "scratch": "io.TextIOWrapper",
            },
            persistent_outputs={"model"},
            checkpoint_dependencies={"selected_genes"},
        )
        assert set(manifest.required) == {"adata", "results", "model", "selected_genes"}
        assert "scratch" not in manifest.required

    def test_later_cell_read_promotes_prior_output_to_required(self):
        from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker

        registry = RuntimeDependencyTracker()
        registry.record_execution(created={"embedding"}, loaded=set(), declared=set())
        registry.record_execution(created=set(), loaded={"embedding"}, declared=set())
        assert registry.required_outputs == {"embedding"}

    def test_declared_outputs_are_required_immediately(self):
        from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker

        registry = RuntimeDependencyTracker()
        registry.record_execution(created={"model"}, loaded=set(), declared={"model"})
        assert registry.required_outputs == {"model"}

    def test_removed_global_is_pruned_from_required(self):
        """A declared/promoted output deleted (del) later must drop out of required.

        Regression: a loompy connection declared as a persistent output then
        `del`-eted stayed permanently required, breaking every snapshot capture.
        """
        from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker

        registry = RuntimeDependencyTracker()
        registry.record_execution(created={"loom_con"}, loaded=set(), declared={"loom_con"})
        assert registry.required_outputs == {"loom_con"}
        # A later cell deletes loom_con from the namespace.
        registry.record_execution(created=set(), loaded=set(), removed={"loom_con"})
        assert registry.required_outputs == set()
        assert "loom_con" not in registry.created_outputs

    def test_canonical_dict_list_artifacts_required(self):
        from backend.services.snapshot_v2.runtime_registry import build_required_runtime_manifest

        manifest = build_required_runtime_manifest(
            globals_meta={
                "results": "builtins.dict",
                "markers": "builtins.list",
                "deg_results": "pandas.core.frame.DataFrame",
                "noise": "numpy.ndarray",
            },
            persistent_outputs=None,
            checkpoint_dependencies=None,
        )
        assert set(manifest.required) == {"results", "markers", "deg_results"}
        assert "noise" not in manifest.required


class TestRuntimeCaptureIntrospection:
    @pytest.fixture
    def runtime_capture(self, tmp_path):
        from backend.services.snapshot_v2.runtime import RuntimeCapture

        staging = tmp_path / "staging"
        staging.mkdir()
        paths = MagicMock()
        paths.staging = staging
        return RuntimeCapture(MagicMock(), paths)

    @pytest.fixture
    def sandbox(self):
        sb = AsyncMock()
        sb.session_id = "test-session"
        sb.km = MagicMock()
        return sb

    @pytest.mark.asyncio
    async def test_introspection_failure_cannot_become_empty_complete(self, runtime_capture, sandbox):
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        sandbox.execute.return_value = MagicMock(
            success=False,
            error=_synthetic_long_runtime_traceback(
                "TimeoutError: synthetic introspection failure"
            ),
            output="",
        )
        with pytest.raises(
            SnapshotCaptureError, match="RUNTIME_INTROSPECT_FAILED"
        ) as exc_info:
            await runtime_capture.capture(sandbox)
        rendered = str(exc_info.value)
        assert "TimeoutError: synthetic introspection failure" in rendered
        assert len(rendered) <= 3200
        assert "\x1b" not in rendered
        assert "synthetic-fixture-credential-value" not in rendered
        assert "synthetic-user" not in rendered

    @pytest.mark.asyncio
    async def test_partial_introspection_is_not_empty_complete(self, runtime_capture, sandbox):
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        sandbox.execute.return_value = MagicMock(
            success=True, output="__RUNTIME_GLOBALS_BEGIN__", error=None
        )
        with pytest.raises(SnapshotCaptureError, match="RUNTIME_INTROSPECT_PARSE_ERROR"):
            await runtime_capture.capture(sandbox)

    @pytest.mark.asyncio
    async def test_live_kernel_empty_required_is_fail_closed(self, runtime_capture, sandbox):
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        sandbox.execute.return_value = MagicMock(
            success=True,
            output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
            error=None,
        )
        with pytest.raises(SnapshotCaptureError, match="NO_RUNTIME_VARIABLES"):
            await runtime_capture.capture(sandbox, kernel_present=True)

    @pytest.mark.asyncio
    async def test_no_kernel_empty_required_is_complete(self, runtime_capture, sandbox):
        from backend.services.snapshot_v2.models import RuntimeManifest

        sandbox.execute.return_value = MagicMock(
            success=True,
            output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
            error=None,
        )
        result = await runtime_capture.capture(sandbox, kernel_present=False)
        assert isinstance(result, RuntimeManifest)
        assert result.status == "complete"
        assert result.dependencies.required == {}


def _ensure_sandbox_file(code: str, content: bytes = b"dummy") -> None:
    """Create a dummy file at the .h5ad or .pkl path found in *code*."""
    m = re.search(r"'([^']+\.(?:h5ad|pkl))'", code)
    if m:
        p = Path(m.group(1))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def _write_valid_h5ad_stub(path: Path) -> None:
    """Write a real AnnData h5ad when available for unit tests."""
    try:
        import anndata as ad
        import numpy as np

        adata = ad.AnnData(X=np.zeros((2, 2)))
        adata.write(path)
    except Exception:
        path.write_bytes(b"not-a-real-h5ad")


def test_known_unpersistable_residual_type_detection():
    from backend.services.snapshot_v2.runtime_registry import is_known_unpersistable_residual
    assert is_known_unpersistable_residual("loompy.loompy.LoomConnection") is True
    assert is_known_unpersistable_residual("loompy.LoomConnection") is True
    assert is_known_unpersistable_residual("pandas.core.frame.DataFrame") is False
    assert is_known_unpersistable_residual("AnnData((10, 20))") is False


class TestRuntimeCapture:
    def test_capture_requires_valid_workflow_state(self):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        capturer = RuntimeCapture(MagicMock(), MagicMock())
        assert capturer is not None

    def test_capture_returns_runtime_generation_manifest(self):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.models import RuntimeGenerationManifest
        manifest = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            kernel_id="test-kernel",
        )
        assert manifest.capture_command == "runtime_capture_v2"
        assert manifest.kernel_id == "test-kernel"

    @pytest.mark.asyncio
    async def test_capture_with_introspect_failure(self):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        sandbox = AsyncMock()
        sandbox.execute = AsyncMock(
            return_value=MagicMock(success=True, output="no markers here", error=None)
        )
        sandbox.km = MagicMock()

        capturer = RuntimeCapture(MagicMock(), MagicMock())
        with pytest.raises(SnapshotCaptureError):
            await capturer.capture(sandbox, declared_outputs=set())

    @pytest.mark.asyncio
    async def test_capture_warning_on_adata_storage_failure(self, tmp_path):
        """When AnnData serialize succeeds but object store put fails, fail closed."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _write_valid_h5ad_stub(p)
                return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        obj_store.put_file = MagicMock(side_effect=RuntimeError("storage failed"))
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(sandbox, declared_outputs={"adata"})
        assert "required_runtime_unpersistable" in str(exc_info.value.code)
        assert "adata" in str(exc_info.value).lower() or "storage" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_required_variable_dill_failure_raises_error(self, tmp_path):
        """When a mandatory (checkpoint) residual fails dill serialization, an
        error is raised (not soft-skipped). Advisory declared outputs soft-skip."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"results": "list"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if "__DILL_SAVED__" in code or "dill" in code:
                return MagicMock(
                    success=True,
                    output="__DILL_SAVED__:0/1 size=0\n__DILL_SKIPPED__:['results']",
                    error=None,
                )
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(
                sandbox,
                declared_outputs=set(),
                checkpoint_dependencies={"results"},
            )
        assert "required_runtime_unpersistable" in str(exc_info.value.code)

    @pytest.mark.asyncio
    async def test_half_written_h5ad_rejected(self, tmp_path):
        """Half-written / invalid h5ad must not be put into object store."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"half-written-garbage")
                return MagicMock(success=True, output="__H5AD_SAVED__:12", error=None)
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(sandbox, declared_outputs={"adata"})
        assert "required_runtime_unpersistable" in str(exc_info.value.code)
        obj_store.put_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_h5ad_requires_result_success(self, tmp_path):
        """Serializer result.success must be required before put_file."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"fake")
                return MagicMock(
                    success=False,
                    output="",
                    error=_synthetic_long_runtime_traceback(
                        "OSError: synthetic h5ad write failure"
                    ),
                )
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(sandbox, declared_outputs={"adata"})
        assert "required_runtime_unpersistable" in str(exc_info.value.code)
        rendered = str(exc_info.value)
        assert "OSError: synthetic h5ad write failure" in rendered
        assert len(rendered) <= 3400
        assert "\x1b" not in rendered
        assert "synthetic-fixture-credential-value" not in rendered
        assert "synthetic-user" not in rendered
        obj_store.put_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_absent_advisory_declared_name_is_soft_skipped(self, tmp_path):
        """A declared (advisory) required name absent from globals must NOT fail closed.

        Regression: `loom_con` (a loompy connection) was declared as a persistent
        output, then deleted (del). It stayed in the declared set but was gone from
        the kernel. Capture must soft-skip it (record as incidental skip) and still
        succeed capturing the real objects — not raise required_runtime_unpersistable.
        """
        from backend.services.snapshot_v2.runtime import RuntimeCapture

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _write_valid_h5ad_stub(p)
                return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        from backend.services.snapshot_v2.models import ObjectRef

        obj_store = MagicMock()
        obj_store.put_file = MagicMock(
            return_value=MagicMock(
                ref=ObjectRef(sha256="a" * 64, size=1000, media_type="application/x-hdf5")
            )
        )
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        runtime = await capturer.capture(
            sandbox, declared_outputs={"adata", "loom_con"}
        )
        assert runtime.status == "complete"
        assert "adata" in runtime.adata_refs
        # loom_con was absent + advisory → soft-skipped, not a captured requirement.
        assert "loom_con" not in runtime.adata_refs
        assert "loom_con" not in runtime.captured_residual_names
        skip_names = {s.name for s in runtime.dependencies.incidental_skips}
        assert "loom_con" in skip_names

    @pytest.mark.asyncio
    async def test_absent_checkpoint_dependency_still_fails_closed(self, tmp_path):
        """A continuation-critical checkpoint dependency absent from globals must
        still fail closed — soft-skip only applies to advisory declarations."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _write_valid_h5ad_stub(p)
                return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        obj_store.put_file = MagicMock(return_value=MagicMock(ref=MagicMock(size=1000)))
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(
                sandbox,
                declared_outputs={"adata"},
                checkpoint_dependencies={"continuation_only"},
            )
        assert "required_runtime_unpersistable" in str(exc_info.value.code)
        assert "continuation_only" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_present_unpersistable_advisory_residual_is_soft_skipped(self, tmp_path):
        """Present loom connection declared as residual must soft-skip, not brick capture."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ObjectPutResult
        from unittest.mock import patch

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output=(
                        '__RUNTIME_GLOBALS_BEGIN__\n'
                        '{"adata": "AnnData(50, 10)", '
                        '"ds": "loompy.loompy.LoomConnection"}\n'
                        '__RUNTIME_GLOBALS_END__'
                    ),
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _write_valid_h5ad_stub(p)
                return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
            if "dill" in code or "__DILL_SAVED__" in code:
                # ds cannot pickle; if still requested, mark skipped
                m = re.search(r"'([^']+\.pkl)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"fake-dill")
                return MagicMock(
                    success=True,
                    output="__DILL_SAVED__:0/1 size=0\n__DILL_SAVED_NAMES__:[]\n__DILL_SKIPPED__:['ds']",
                    error=None,
                )
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        def put_file(path, media_type):
            size = Path(path).stat().st_size if Path(path).exists() else 1000
            return ObjectPutResult(
                ref=ObjectRef(sha256="b" * 64, size=size, media_type=media_type),
                created=True,
            )
        obj_store.put_file = MagicMock(side_effect=put_file)
        paths = MagicMock(staging=staging, root=tmp_path)

        capturer = RuntimeCapture(obj_store, paths)
        fake_usage = type("U", (), {"total": 100 * 1024**3, "used": 0, "free": 100 * 1024**3})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            runtime = await capturer.capture(
                sandbox,
                declared_outputs={"adata", "ds"},  # advisory declared residual
            )
        assert runtime.status == "complete"
        assert "adata" in runtime.adata_refs
        assert "ds" not in runtime.captured_residual_names
        skip_names = {s.name for s in runtime.dependencies.incidental_skips}
        assert "ds" in skip_names

    @pytest.mark.asyncio
    async def test_mandatory_unpersistable_residual_still_fails_closed(self, tmp_path):
        """Checkpoint-mandatory residual that cannot dill must still raise."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from unittest.mock import patch

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"ds": "loompy.loompy.LoomConnection"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if "dill" in code or "__DILL_SAVED__" in code:
                return MagicMock(
                    success=True,
                    output="__DILL_SAVED__:0/1 size=0\n__DILL_SAVED_NAMES__:[]\n__DILL_SKIPPED__:['ds']",
                    error=None,
                )
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()
        paths = MagicMock(staging=staging, root=tmp_path)
        capturer = RuntimeCapture(MagicMock(), paths)
        fake_usage = type("U", (), {"total": 100 * 1024**3, "used": 0, "free": 100 * 1024**3})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            with pytest.raises(SnapshotCaptureError) as exc_info:
                await capturer.capture(
                    sandbox,
                    declared_outputs=set(),
                    checkpoint_dependencies={"ds"},
                )
        assert "required_runtime_unpersistable" in str(exc_info.value.code)

    @pytest.mark.asyncio
    async def test_mark_runtime_committed(self, tmp_path):
        """mark_runtime_committed sets saved_runtime_revision to runtime_revision."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture

        sandbox = AsyncMock()
        sandbox.runtime_revision = 5
        sandbox.saved_runtime_revision = 0

        capturer = RuntimeCapture(MagicMock(), MagicMock())
        capturer.mark_runtime_committed(sandbox)
        assert sandbox.saved_runtime_revision == 5

    @pytest.mark.asyncio
    async def test_mark_runtime_committed_frozen_revision(self, tmp_path):
        """Explicit frozen revision must win over concurrent sandbox drift."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture

        sandbox = AsyncMock()
        sandbox.runtime_revision = 99
        sandbox.saved_runtime_revision = 0

        capturer = RuntimeCapture(MagicMock(), MagicMock())
        capturer.mark_runtime_committed(sandbox, runtime_revision=7)
        assert sandbox.saved_runtime_revision == 7

    @pytest.mark.asyncio
    async def test_exec_lock_held_uses_unlocked_path(self, tmp_path):
        """When exec_lock_held=True, capture must call _execute_unlocked."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()
        unlocked_calls = []
        execute_calls = []

        async def unlocked(code, save_to_history=True, skip_safety_check=False):
            unlocked_calls.append(True)
            return MagicMock(
                success=True,
                output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
                error=None,
            )

        async def execute(code, save_to_history=True, skip_safety_check=False):
            execute_calls.append(True)
            return MagicMock(
                success=True,
                output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
                error=None,
            )

        sandbox = AsyncMock()
        sandbox._execute_unlocked = unlocked
        sandbox.execute = execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        capturer = RuntimeCapture(MagicMock(), MagicMock(staging=staging))
        with pytest.raises(SnapshotCaptureError, match="NO_RUNTIME_VARIABLES"):
            await capturer.capture(
                sandbox, declared_outputs=set(), exec_lock_held=True,
            )
        assert unlocked_calls
        assert not execute_calls

    @pytest.mark.asyncio
    async def test_large_runtime_does_not_hit_hard_size_quota(self, tmp_path):
        """Captured runtime larger than the old 2 GiB constant must still succeed
        when post-write free space stays above reserve (mocked)."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ObjectPutResult
        from unittest.mock import patch

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _write_valid_h5ad_stub(p)
                return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        obj_store.put_file = MagicMock(return_value=ObjectPutResult(
            ref=ObjectRef(
                sha256="a" * 64,
                size=3 * 1024 * 1024 * 1024,  # 3 GiB — previously failed hard quota
                media_type="application/x-hdf5",
            ),
            created=True,
        ))
        paths = MagicMock()
        paths.staging = staging
        paths.root = tmp_path

        capturer = RuntimeCapture(obj_store, paths)
        # Post-write semantics: 100 GiB free after write, 5 GiB reserve → OK
        # (payload already accounted for in free; do not subtract total_size again)
        fake_usage = type("U", (), {"total": 200 * 1024**3, "used": 100 * 1024**3, "free": 100 * 1024**3})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            runtime = await capturer.capture(sandbox, declared_outputs={"adata"})
        assert runtime.status == "complete"
        assert "adata" in runtime.adata_refs

    @pytest.mark.asyncio
    async def test_runtime_capture_fails_when_disk_cannot_hold_payload(self, tmp_path):
        """When post-write free is below reserve, capture must fail closed."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ObjectPutResult
        from unittest.mock import patch

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"adata": "AnnData(50, 10)"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                m = re.search(r"'([^']+\.h5ad)'", code)
                if m:
                    p = Path(m.group(1))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _write_valid_h5ad_stub(p)
                return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()

        obj_store = MagicMock()
        obj_store.put_file = MagicMock(return_value=ObjectPutResult(
            ref=ObjectRef(
                sha256="a" * 64,
                size=2 * 1024 * 1024 * 1024,
                media_type="application/x-hdf5",
            ),
            created=True,
        ))
        paths = MagicMock()
        paths.staging = staging
        paths.root = tmp_path

        capturer = RuntimeCapture(obj_store, paths)
        capturer.FREE_SPACE_RESERVE_BYTES = 5 * 1024 ** 3
        # Post-write free below reserve (1 GiB free, 5 GiB reserve) → DISK_FULL.
        # Do NOT require free - total_size < reserve (payload already on disk).
        fake_usage = type("U", (), {"total": 10 * 1024**3, "used": 9 * 1024**3, "free": 1 * 1024**3})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            with pytest.raises(SnapshotCaptureError) as exc_info:
                await capturer.capture(sandbox, declared_outputs={"adata"})
        assert "RUNTIME_CAPTURE_DISK_FULL" in str(exc_info.value.code)

    @pytest.mark.asyncio
    async def test_anndata_write_failure_still_surfaces_required_error(self, tmp_path):
        """If even sanitized write fails, keep fail-closed required_runtime_unpersistable."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        staging = tmp_path / "staging"
        staging.mkdir()

        async def mock_execute(code, save_to_history=True, skip_safety_check=False):
            if "__RUNTIME_GLOBALS_BEGIN__" in code:
                return MagicMock(
                    success=True,
                    output='__RUNTIME_GLOBALS_BEGIN__\n{"dds": "AnnData((2, 13704))"}\n__RUNTIME_GLOBALS_END__',
                    error=None,
                )
            if ".h5ad" in code:
                return MagicMock(
                    success=False,
                    output="",
                    error="IORegistryError: No method registered for writing ModelMatrix",
                )
            return MagicMock(success=True, output="ok", error=None)

        sandbox = AsyncMock()
        sandbox.execute = mock_execute
        sandbox.session_id = "test-session"
        sandbox.km = MagicMock()
        capturer = RuntimeCapture(MagicMock(), MagicMock(staging=staging, root=tmp_path))
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(sandbox, declared_outputs={"dds"})
        assert "required_runtime_unpersistable" in str(exc_info.value.code)
        assert "dds" in str(exc_info.value)


def test_save_anndata_template_sanitizes_unwritable_slots():
    from backend.services.snapshot_v2 import runtime as runtime_mod
    tpl = runtime_mod._SAVE_ANNDATA_TEMPLATE
    assert "copy()" in tpl
    assert "obsm" in tpl
    # uns is now recursively sanitized (coerce/drop unwritable leaves such as a
    # bare pd.Series) while preserving nested metadata dicts.
    assert "_sanitize_uns_mapping" in tpl
    for slot in ("obsm", "varm", "layers", "obsp", "varp"):
        assert f"'{slot}'" in tpl or f'"{slot}"' in tpl
    # Must attempt conversion to numpy/DataFrame or drop on failure
    assert "IORegistryError" in tpl or "write" in tpl
    assert "to_numpy" in tpl or "asarray" in tpl
    assert "__H5AD_SANITIZE_DROP__" in tpl
    # Sparse must be preserved; exact-type checks for native frames
    assert "issparse" in tpl
    assert "type(_v) is _runtime_np.ndarray" in tpl or "type(_v) is" in tpl


def test_sanitize_anndata_preserves_sparse_and_converts_modelmatrix(tmp_path):
    """Behavioral host-side sanitize: sparse stays sparse; ModelMatrix becomes array; live unchanged."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    import scipy.sparse as sp
    import wrapt

    from backend.services.snapshot_v2.runtime import sanitize_anndata_for_h5ad

    n_obs, n_vars = 4, 3
    X = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars)
    sparse_layer = sp.csr_matrix(X)
    adata = ad.AnnData(X=X.copy())
    adata.layers["counts"] = sparse_layer
    adata.obsm["X_pca"] = np.random.randn(n_obs, 2)

    # Build a formulaic-like ModelMatrix (wrapt ObjectProxy): isinstance(..., DataFrame)
    # is True but type(...) is not DataFrame, and write_h5ad fails if it remains.
    try:
        from formulaic import model_matrix

        design_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=list(adata.obs_names))
        mm = model_matrix("1 + x", design_df)
        if hasattr(mm, "index") and not mm.index.equals(adata.obs_names):
            mm = mm.copy()
            mm.index = adata.obs_names
    except Exception:
        class ModelMatrix(wrapt.ObjectProxy):
            pass

        ModelMatrix.__module__ = "formulaic.materializers.types.model_matrix"
        ModelMatrix.__qualname__ = "ModelMatrix"
        ModelMatrix.__name__ = "ModelMatrix"
        base = pd.DataFrame(
            {"Intercept": [1, 1, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]},
            index=list(adata.obs_names),
        )
        mm = ModelMatrix(base)

    assert isinstance(mm, pd.DataFrame)
    assert type(mm) is not pd.DataFrame
    adata.obsm["design"] = mm

    # AnnData.copy() often unwraps ObjectProxy to plain DataFrame; re-inject ModelMatrix
    # on the copy so sanitize must convert (mirrors kernel slots that still hold proxy).
    _orig_copy = adata.copy

    def _copy_keep_modelmatrix():
        c = _orig_copy()
        c.obsm["design"] = mm
        return c

    adata.copy = _copy_keep_modelmatrix  # type: ignore[method-assign]

    live_layer_id = id(adata.layers["counts"])
    live_design_type = type(adata.obsm["design"])

    # Unsanitized write with ModelMatrix must fail (documents the bug we fix)
    dirty = _copy_keep_modelmatrix()
    with pytest.raises(Exception):
        dirty.write_h5ad(tmp_path / "dirty.h5ad")

    sanitized = sanitize_anndata_for_h5ad(adata)

    # Live original unchanged
    assert id(adata.layers["counts"]) == live_layer_id
    assert sp.issparse(adata.layers["counts"])
    assert type(adata.obsm["design"]) is live_design_type

    # Sparse layer preserved on sanitized copy
    assert sp.issparse(sanitized.layers["counts"])
    assert type(sanitized.layers["counts"]).__name__ in ("csr_matrix", "csc_matrix", "coo_matrix")

    # design converted to plain ndarray (not ModelMatrix / DF proxy)
    design = sanitized.obsm["design"]
    assert type(design) is np.ndarray, f"expected ndarray, got {type(design)}"
    assert design.ndim >= 1
    assert not (design.dtype == object and design.shape == ())

    # Must be writable to h5ad
    out = tmp_path / "sanitized.h5ad"
    sanitized.write_h5ad(out)
    assert out.stat().st_size > 0
    reloaded = ad.read_h5ad(out)
    assert "counts" in reloaded.layers
    assert "design" in reloaded.obsm


def test_sanitize_anndata_preserves_standard_uns_metadata(tmp_path):
    """Standard Scanpy-like uns nested dicts must survive sanitize + write_h5ad."""
    import anndata as ad
    import numpy as np

    from backend.services.snapshot_v2.runtime import sanitize_anndata_for_h5ad

    adata = ad.AnnData(X=np.random.rand(10, 5).astype(np.float32))
    adata.uns["neighbors"] = {"params": {"n_neighbors": 15, "method": "umap"}}
    adata.uns["log1p"] = {"base": None}
    adata.uns["pca"] = {"variance": np.array([1.0, 0.5], dtype=np.float32)}
    adata.uns["rank_genes_groups"] = {"params": {"groupby": "leiden"}}
    live_keys = set(adata.uns.keys())

    cleaned = sanitize_anndata_for_h5ad(adata)
    for k in ("neighbors", "log1p", "pca", "rank_genes_groups"):
        assert k in cleaned.uns, f"{k} dropped from uns"
    assert set(adata.uns.keys()) == live_keys  # live unchanged

    out = tmp_path / "scai_uns_sanitize_test.h5ad"
    cleaned.write_h5ad(out)  # must succeed
    reloaded = ad.read_h5ad(out)
    for k in ("neighbors", "log1p", "pca", "rank_genes_groups"):
        assert k in reloaded.uns


def test_sanitize_anndata_coerces_unwritable_uns_leaves(tmp_path):
    """uns leaves that anndata cannot write (bare Series/Index/Categorical, and
    the same nested inside dicts) must be coerced/dropped so write_h5ad succeeds.

    Reproduces the PyDESeq2 DeseqDataSet failure: uns['trend_coeffs'] is a
    pandas.Series, which has no registered h5ad writer under /uns.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    from backend.services.snapshot_v2.runtime import sanitize_anndata_for_h5ad

    adata = ad.AnnData(X=np.random.rand(6, 4).astype(np.float32))
    # The exact PyDESeq2 shape: a bare Series in uns.
    adata.uns["trend_coeffs"] = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    adata.uns["some_index"] = pd.Index(["g1", "g2", "g3"])
    adata.uns["cat"] = pd.Categorical(["x", "y", "x"])
    # Same unwritable type nested one level deep inside a metadata dict.
    adata.uns["fit"] = {"coeffs": pd.Series([1.0, 2.0]), "n": 6}
    # Legitimate nested Scanpy metadata must survive untouched.
    adata.uns["neighbors"] = {"params": {"n_neighbors": 15}}
    adata.uns["pca"] = {"variance": np.array([1.0, 0.5], dtype=np.float32)}
    live_keys = set(adata.uns.keys())

    # Unsanitized write must fail (documents the bug).
    with pytest.raises(Exception):
        adata.copy().write_h5ad(tmp_path / "dirty.h5ad")

    cleaned = sanitize_anndata_for_h5ad(adata)

    # Live original unchanged (no mutation of the analysis object).
    assert set(adata.uns.keys()) == live_keys
    assert isinstance(adata.uns["trend_coeffs"], pd.Series)

    # Coerced leaves became ndarray; nested dict metadata preserved.
    assert isinstance(cleaned.uns["trend_coeffs"], np.ndarray)
    assert isinstance(cleaned.uns["fit"]["coeffs"], np.ndarray)
    assert cleaned.uns["fit"]["n"] == 6
    assert cleaned.uns["neighbors"]["params"]["n_neighbors"] == 15
    assert isinstance(cleaned.uns["pca"]["variance"], np.ndarray)

    # Must now be writable to h5ad.
    out = tmp_path / "scai_uns_coerce_test.h5ad"
    cleaned.write_h5ad(out)
    assert out.stat().st_size > 0
    reloaded = ad.read_h5ad(out)
    assert "trend_coeffs" in reloaded.uns
    assert "neighbors" in reloaded.uns


def test_sanitize_anndata_coerces_unwritable_obs_var_columns(tmp_path):
    """obs/var object-dtype columns holding non-string objects (bool/None,
    e.g. scrublet 'predicted_doublet') must be coerced so write_h5ad succeeds.

    Reproduces: TypeError: Can't implicitly convert non-string objects to
    strings — Error raised while writing key 'predicted_doublet' to /obs.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    from backend.services.snapshot_v2.runtime import sanitize_anndata_for_h5ad

    adata = ad.AnnData(X=np.random.rand(4, 3).astype(np.float32))
    # Object column with Python bool + None → routed to vlen-str writer → fails.
    adata.obs["predicted_doublet"] = pd.Series(
        [True, False, None, True], index=adata.obs_names, dtype=object
    )
    # Object column that is all strings must be left as-is (still writable).
    adata.obs["cell_label"] = pd.Series(
        ["a", "b", "c", "d"], index=adata.obs_names, dtype=object
    )
    # A var object column with mixed types.
    adata.var["note"] = pd.Series([1, "x", None], index=adata.var_names, dtype=object)
    # datetime column has no h5ad writer either — must be stringified.
    adata.obs["ingest_time"] = pd.to_datetime(["2020-01-01"] * 4)
    live_obs_dtype = adata.obs["predicted_doublet"].dtype

    # Unsanitized write must fail (documents the bug).
    with pytest.raises(Exception):
        adata.copy().write_h5ad(tmp_path / "dirty.h5ad")

    cleaned = sanitize_anndata_for_h5ad(adata)

    # Live original unchanged.
    assert adata.obs["predicted_doublet"].dtype == live_obs_dtype
    assert adata.obs["predicted_doublet"].iloc[2] is None

    # Must now be writable to h5ad.
    out = tmp_path / "scai_obs_coerce_test.h5ad"
    cleaned.write_h5ad(out)
    assert out.stat().st_size > 0
    reloaded = ad.read_h5ad(out)
    assert "predicted_doublet" in reloaded.obs
    assert "cell_label" in reloaded.obs
    assert "note" in reloaded.var
    assert "ingest_time" in reloaded.obs
    # All-string object column preserved as its values (not mangled).
    assert list(reloaded.obs["cell_label"]) == ["a", "b", "c", "d"]


def test_sanitize_anndata_covers_raw_x_obsm_df_and_uns_lists(tmp_path):
    """Audit-driven: raw.var object cols, object-dtype X, DataFrame values in
    obsm with bad columns, and uns list[dict] must all be made writable, while
    uns list[str]/Series[str] keep their values."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from backend.services.snapshot_v2.runtime import sanitize_anndata_for_h5ad

    n = 6
    # raw carries an unwritable object column (set before HVG subsetting).
    adata = ad.AnnData(X=sp.csr_matrix(np.random.rand(n, 10)))
    adata.var_names = [f"g{i}" for i in range(10)]
    adata.var["flag"] = pd.Series(
        [True, None, "x"] + [False] * 7, index=adata.var_names, dtype=object
    )
    adata.raw = adata.copy()
    adata = adata[:, :4].copy()  # raw retains 10 vars + bad column

    # object-dtype dense X (numeric-in-object).
    adata.X = np.asarray(adata.X.toarray(), dtype=object)
    # DataFrame value in obsm with a bad object column.
    adata.obsm["meta"] = pd.DataFrame(
        {"b": pd.Series([True, None] * 3, dtype=object)}, index=adata.obs_names
    )
    # uns lists/series.
    adata.uns["markers"] = ["CD3", "CD4"]            # keep (as string array)
    adata.uns["weird"] = [{"k": 1}, {"k": 2}]        # drop (unwritable)
    adata.uns["sstr"] = pd.Series(["x", "y"])        # keep values

    cleaned = sanitize_anndata_for_h5ad(adata)

    out = tmp_path / "scai_audit_test.h5ad"
    cleaned.write_h5ad(out)  # must not raise
    reloaded = ad.read_h5ad(out)

    assert "flag" in reloaded.raw.var
    assert reloaded.X.dtype != object
    assert "meta" in reloaded.obsm
    assert list(reloaded.uns["markers"]) == ["CD3", "CD4"]
    assert list(reloaded.uns["sstr"]) == ["x", "y"]
    assert "weird" not in reloaded.uns


@pytest.mark.asyncio
async def test_post_dill_advisory_unknown_type_soft_skipped(tmp_path):
    """Advisory residual with unknown type that dill skips → complete + incidental_skips.

    Same name as a checkpoint dependency must still fail closed.
    """
    from unittest.mock import patch

    from backend.services.snapshot_v2.errors import SnapshotCaptureError
    from backend.services.snapshot_v2.models import ObjectRef
    from backend.services.snapshot_v2.object_store import ObjectPutResult
    from backend.services.snapshot_v2.runtime import RuntimeCapture
    from backend.services.snapshot_v2.runtime_registry import is_known_unpersistable_residual

    unknown_type = "some.package.MysteryHandle"
    assert not is_known_unpersistable_residual(unknown_type)

    staging = tmp_path / "staging"
    staging.mkdir()

    async def mock_execute(code, save_to_history=True, skip_safety_check=False):
        if "__RUNTIME_GLOBALS_BEGIN__" in code:
            return MagicMock(
                success=True,
                output=(
                    '__RUNTIME_GLOBALS_BEGIN__\n'
                    '{"adata": "AnnData(50, 10)", '
                    f'"mystery": "{unknown_type}"}}\n'
                    '__RUNTIME_GLOBALS_END__'
                ),
                error=None,
            )
        if ".h5ad" in code:
            m = re.search(r"'([^']+\.h5ad)'", code)
            if m:
                p = Path(m.group(1))
                p.parent.mkdir(parents=True, exist_ok=True)
                _write_valid_h5ad_stub(p)
            return MagicMock(success=True, output="__H5AD_SAVED__:1000", error=None)
        if "dill" in code or "__DILL_SAVED__" in code:
            m = re.search(r"'([^']+\.pkl)'", code)
            if m:
                p = Path(m.group(1))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fake-dill")
            return MagicMock(
                success=True,
                output=(
                    "__DILL_SAVED__:0/1 size=0\n"
                    "__DILL_SAVED_NAMES__:[]\n"
                    "__DILL_SKIPPED__:['mystery']"
                ),
                error=None,
            )
        return MagicMock(success=True, output="ok", error=None)

    sandbox = AsyncMock()
    sandbox.execute = mock_execute
    sandbox.session_id = "test-session"
    sandbox.km = MagicMock()

    def put_file(path, media_type):
        size = Path(path).stat().st_size if Path(path).exists() else 1000
        return ObjectPutResult(
            ref=ObjectRef(sha256="c" * 64, size=size, media_type=media_type),
            created=True,
        )

    obj_store = MagicMock()
    obj_store.put_file = MagicMock(side_effect=put_file)
    paths = MagicMock(staging=staging, root=tmp_path)
    capturer = RuntimeCapture(obj_store, paths)
    fake_usage = type(
        "U", (), {"total": 100 * 1024**3, "used": 0, "free": 100 * 1024**3}
    )()

    with patch("shutil.disk_usage", return_value=fake_usage):
        runtime = await capturer.capture(
            sandbox,
            declared_outputs={"adata", "mystery"},
        )
    assert runtime.status == "complete"
    assert "adata" in runtime.adata_refs
    assert "mystery" not in runtime.captured_residual_names
    skip_names = {s.name for s in runtime.dependencies.incidental_skips}
    assert "mystery" in skip_names

    # Same name as checkpoint dependency still fails closed.
    with patch("shutil.disk_usage", return_value=fake_usage):
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(
                sandbox,
                declared_outputs={"adata"},
                checkpoint_dependencies={"mystery"},
            )
    assert "required_runtime_unpersistable" in str(exc_info.value.code)
    assert "mystery" in str(exc_info.value)


@pytest.mark.asyncio
async def test_safe_point_methods_are_async_and_locked():
    from backend.services.snapshot_v2.safe_point import SafePointManager
    import inspect

    mgr = SafePointManager()
    for name in (
        "request_safe_point", "acknowledge_safe_point_request", "acquire",
        "update_descriptor", "cancel_safe_point_request", "release",
    ):
        assert inspect.iscoroutinefunction(getattr(mgr, name)), f"{name} must be async"

    req = await mgr.request_safe_point("s1", "owner-1")
    assert req is not None
    # A different owner cannot release owner-1's hold.
    assert await mgr.release("s1", "owner-2") is False
    assert await mgr.release("s1", "owner-1") is True
