# backend/tests/test_reproducibility.py
import io
import json
import zipfile
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

from backend.services.reproducibility import ReproducibilityBundle


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_ipynb(tmp_workspace):
    nb = new_notebook()
    nb.cells.append(new_code_cell("import scanpy as sc\nprint('hello')"))
    nb.cells.append(new_code_cell("# this is a comment\nsc.pp.filter_cells(adata, min_genes=200)"))
    nb.cells.append(new_code_cell("!pip install pandas"))
    nb.cells.append(new_code_cell(""))
    nb.cells.append(new_code_cell("# only a comment\n# another comment line"))
    path = tmp_workspace / "test.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    return path


@pytest.fixture
def sample_round_results():
    """bioinformatics_findings is List[str] per schemas.py:268."""
    return [
        type("RoundResult", (), {
            "round_number": 1,
            "bioinformatics_findings": [
                "QC filtering removed 15% low-quality cells with <200 genes or >20% mito.",
            ],
        })(),
        type("RoundResult", (), {
            "round_number": 2,
            "bioinformatics_findings": [
                "Cluster 0 expresses CD3D/CD3E, annotated as T cells (high confidence).",
                "Used Harmony for batch correction across 3 donor samples.",
            ],
        })(),
    ]


class TestEnvironmentCapture:
    def test_capture_environment_returns_valid_data(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        env_path = tmp_workspace / "environment_frozen.txt"
        sysinfo_path = tmp_workspace / "system_info.json"

        bundle._capture_environment(env_path, sysinfo_path)

        # env file must exist and contain pip freeze output
        assert env_path.exists()
        env_content = env_path.read_text()
        assert len(env_content) > 20
        # sysinfo must be valid JSON with required fields
        assert sysinfo_path.exists()
        info = json.loads(sysinfo_path.read_text())
        for key in ("python_version", "platform", "scai_version", "cpu_count"):
            assert key in info, f"Missing key: {key}"

    def test_capture_environment_pip_freeze_failure_graceful(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        env_path = tmp_workspace / "environment_frozen.txt"
        sysinfo_path = tmp_workspace / "system_info.json"

        with patch("subprocess.run", side_effect=FileNotFoundError("pip not found")):
            bundle._capture_environment(env_path, sysinfo_path)

        assert env_path.exists()
        assert "pip freeze failed" in env_path.read_text()

    def test_capture_environment_pip_freeze_success_mocked(self, tmp_workspace):
        """Mock pip freeze success for fast, reliable unit testing."""
        bundle = ReproducibilityBundle()
        env_path = tmp_workspace / "environment_frozen.txt"
        sysinfo_path = tmp_workspace / "system_info.json"

        canned = "scanpy==1.9.6\nanndata==0.10.3\nnumpy==1.26.4\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = canned
            bundle._capture_environment(env_path, sysinfo_path)

        assert env_path.exists()
        assert "scanpy==1.9.6" in env_path.read_text()
        assert sysinfo_path.exists()


class TestChecksums:
    def test_compute_checksums_empty_dir(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        result = bundle._compute_checksums(str(tmp_workspace), [])
        assert result == ""

    def test_compute_checksums_known_file(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        content = b"hello world"
        expected_sha = hashlib.sha256(content).hexdigest()
        (tmp_workspace / "data.h5ad").write_bytes(content)

        result = bundle._compute_checksums(str(tmp_workspace), [])
        assert "data.h5ad" in result
        assert expected_sha in result

    def test_compute_checksums_excludes_runtime_artifacts(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        (tmp_workspace / "session.pkl").write_bytes(b"\x80\x03")
        (tmp_workspace / "real_data.csv").write_text("a,b,c\n1,2,3")
        (tmp_workspace / "analysis.ipynb").write_text("{}")
        (tmp_workspace / "figure.png").write_bytes(b"\x89PNG")

        result = bundle._compute_checksums(str(tmp_workspace), [])
        # Runtime artifacts are excluded
        assert "session.pkl" not in result
        # Analysis outputs (ipynb, images, data) are included for integrity
        assert "real_data.csv" in result
        assert "analysis.ipynb" in result
        assert "figure.png" in result

    def test_compute_checksums_skips_reproducibility_subtree(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        repro_dir = tmp_workspace / "reproducibility"
        repro_dir.mkdir()
        (repro_dir / "manifest.json").write_text("{}")
        (repro_dir / "checksums.sha256").write_text("dummy")
        (tmp_workspace / "real_data.h5ad").write_bytes(b"data")

        result = bundle._compute_checksums(str(tmp_workspace), [])
        assert "manifest.json" not in result
        assert "checksums.sha256" not in result
        assert "real_data.h5ad" in result

    def test_compute_checksums_includes_uploads(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        upload_dir = tmp_workspace / "uploads"
        upload_dir.mkdir()
        (upload_dir / "input.h5ad").write_bytes(b"input data")

        result = bundle._compute_checksums(
            str(tmp_workspace),
            [str(upload_dir / "input.h5ad")],
        )
        assert "input.h5ad" in result


class TestCommandExtraction:
    def test_extract_commands_from_ipynb(self, sample_ipynb):
        bundle = ReproducibilityBundle()
        result = bundle._extract_commands(str(sample_ipynb))

        assert "import scanpy" in result
        assert "sc.pp.filter_cells" in result
        assert result.startswith("#!/usr/bin/env python3")
        # pip install should be filtered
        assert "pip install" not in result
        # Inline comments are preserved (they provide context for reproducibility)
        assert "# this is a comment" in result
        # Pure-comment cells (no executable code) are skipped
        assert "# only a comment" not in result
        assert "# another comment line" not in result

    def test_extract_commands_empty_notebook(self, tmp_workspace):
        nb = new_notebook()
        path = tmp_workspace / "empty.ipynb"
        with open(path, "w") as f:
            nbformat.write(nb, f)

        bundle = ReproducibilityBundle()
        result = bundle._extract_commands(str(path))
        assert result.startswith("#!/usr/bin/env python3")

    def test_extract_commands_skips_empty_cells(self, sample_ipynb):
        """Empty code cells should not produce blank lines in output."""
        bundle = ReproducibilityBundle()
        result = bundle._extract_commands(str(sample_ipynb))
        lines = result.split("\n")
        # no consecutive blank lines from empty cells
        assert "\n\n\n" not in result


class TestAgentDecisions:
    def test_compile_decisions_extracts_from_round_results(self, sample_round_results):
        bundle = ReproducibilityBundle()
        result = bundle._compile_agent_decisions(sample_round_results)

        assert "QC filtering" in result
        assert "T cells" in result
        assert "Harmony" in result
        assert "## Agent Decision Log" in result

    def test_compile_decisions_empty_rounds(self):
        bundle = ReproducibilityBundle()
        result = bundle._compile_agent_decisions([])
        assert "No agent decision records" in result


class TestManifest:
    def test_manifest_contains_all_entries(self, tmp_workspace):
        bundle = ReproducibilityBundle()
        manifest_path = tmp_workspace / "manifest.json"
        bundle._write_manifest(manifest_path, "sess-123", [
            "environment_frozen.txt", "system_info.json", "checksums.sha256",
            "commands.py", "agent_decisions.md",
        ])

        with open(manifest_path) as f:
            m = json.load(f)
        assert "scai_version" in m
        assert m["session_id"] == "sess-123"
        assert "contents" in m
        for fname in ("environment_frozen.txt", "checksums.sha256", "commands.py"):
            assert fname in m["contents"]


class TestFullBundle:
    def test_generate_creates_all_files(self, tmp_workspace, sample_ipynb, sample_round_results):
        bundle = ReproducibilityBundle()
        output_dir = tmp_workspace / "reproducibility"
        output_dir.mkdir()

        # Write files simulating generate() output
        bundle._capture_environment(
            output_dir / "environment_frozen.txt",
            output_dir / "system_info.json",
        )
        (output_dir / "checksums.sha256").write_text("abc123  data.h5ad\n")
        (output_dir / "commands.py").write_text(
            bundle._extract_commands(str(sample_ipynb))
        )
        (output_dir / "agent_decisions.md").write_text(
            bundle._compile_agent_decisions(sample_round_results)
        )
        bundle._write_manifest(
            output_dir / "manifest.json", "test-session",
            ["environment_frozen.txt", "system_info.json", "checksums.sha256",
             "commands.py", "agent_decisions.md"],
        )

        expected_files = [
            "environment_frozen.txt", "system_info.json", "checksums.sha256",
            "commands.py", "agent_decisions.md", "manifest.json",
        ]
        for fname in expected_files:
            assert (output_dir / fname).exists(), f"Missing: {fname}"

    def test_generate_cleans_old_files(self, tmp_workspace, sample_ipynb, sample_round_results):
        output_dir = tmp_workspace / "reproducibility"
        output_dir.mkdir()
        stale = output_dir / "old_file.txt"
        stale.write_text("should be removed")

        # simulate generate() with cleanup
        bundle = ReproducibilityBundle()
        bundle.generate(
            session_id="test",
            workspace_path=str(tmp_workspace),
            ipynb_path=str(sample_ipynb),
            round_results=sample_round_results,
            upload_paths=[],
        )
        assert not stale.exists()
        assert (output_dir / "manifest.json").exists()

    def test_create_zip(self, tmp_workspace):
        output_dir = tmp_workspace / "reproducibility"
        output_dir.mkdir()
        (output_dir / "manifest.json").write_text('{"scai_version":"0.22.1"}')
        (output_dir / "environment_frozen.txt").write_text("scanpy==1.9.6\n")

        bundle = ReproducibilityBundle()
        zip_path = tmp_workspace / "bundle.zip"
        bundle._create_zip(str(output_dir), str(zip_path))

        assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "environment_frozen.txt" in names