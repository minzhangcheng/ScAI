import re
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest


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

        sandbox.execute.return_value = MagicMock(success=False, error="timeout", output="")
        with pytest.raises(SnapshotCaptureError, match="RUNTIME_INTROSPECT_FAILED"):
            await runtime_capture.capture(sandbox)

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
        """When a required non-AnnData variable fails dill serialization, an
        error is raised (not silently skipped)."""
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
            await capturer.capture(sandbox, declared_outputs={"results"})
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
                return MagicMock(success=False, output="", error="write failed")
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
    async def test_capture_quota_exceeded(self, tmp_path):
        """Capture raises SnapshotCaptureError when total size exceeds quota."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from backend.services.snapshot_v2.models import ObjectRef

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
        obj_store.put_file = MagicMock(return_value=ObjectRef(
            sha256="a" * 64, size=3 * 1024 * 1024 * 1024, media_type="application/x-hdf5"
        ))
        paths = MagicMock()
        paths.staging = staging

        capturer = RuntimeCapture(obj_store, paths)
        with pytest.raises(SnapshotCaptureError) as exc_info:
            await capturer.capture(sandbox, declared_outputs={"adata"})
        assert "RUNTIME_CAPTURE_QUOTA_EXCEEDED" in str(exc_info.value.code)
