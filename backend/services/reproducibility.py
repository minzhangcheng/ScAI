# backend/services/reproducibility.py
"""
Reproducibility Bundle Generator

Generates a ClawBio-style reproducibility bundle for every completed analysis.
Includes: environment snapshot, file checksums, executable Python commands,
agent decision summary, and a manifest.

Environment capture uses subprocess.run in the backend process — no Jupyter
sandbox dependency.  The backend Python environment IS the analysis environment.
"""
import json
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

from backend.models.schemas import RoundResult

import nbformat

logger = logging.getLogger(__name__)


def _get_scai_version() -> str:
    """Read ScAI version from backend.main to avoid hardcoding."""
    try:
        from backend.main import app
        return app.version
    except Exception:
        return "unknown"


REPRODUCIBILITY_DIR_NAME = "reproducibility"

# File extensions to exclude from checksums (runtime artifacts, not analysis data).
# Only exclude ephemeral/regenerated files — ipynb, images, and PDFs are core
# analysis outputs that reviewers need to verify for reproducibility.
# Note: Path.suffix only returns the last component (".gz" for ".tar.gz"),
# so we store both single and compound suffixes and check both.
CHECKSUM_EXCLUDE_SUFFIXES = {
    ".pkl", ".pickle", ".tgz", ".zip",
}
CHECKSUM_EXCLUDE_COMPOUND = {".tar.gz"}


def _is_excluded(path: Path) -> bool:
    """Check if a file path has an excluded extension (handles compound suffixes like .tar.gz)."""
    if path.suffix in CHECKSUM_EXCLUDE_SUFFIXES:
        return True
    if "".join(path.suffixes) in CHECKSUM_EXCLUDE_COMPOUND:
        return True
    return False

# Key packages to highlight in the report appendix summary
KEY_PACKAGES = frozenset({
    "scanpy", "anndata", "scvelo", "cellrank", "harmonypy", "bbknn", "scvi-tools",
    "liana", "pydeseq2", "scikit-learn", "numpy", "pandas", "matplotlib",
    "seaborn", "plotly", "leidenalg", "decoupler", "gseapy", "squidpy",
})


class ReproducibilityBundle:
    """Generates and manages reproducibility artifacts for a session."""

    def generate(
        self,
        session_id: str,
        workspace_path: str,
        ipynb_path: str,
        round_results: List[RoundResult],
        upload_paths: List[str],
    ) -> str:
        """
        Generate the complete reproducibility bundle.

        Args:
            session_id: Session identifier.
            workspace_path: Path to session workspace directory.
            ipynb_path: Path to the exported Jupyter notebook.
            round_results: AnalysisState.round_results (list of RoundResult).
            upload_paths: List of absolute paths to uploaded input files.

        Returns:
            Path to the reproducibility output directory.
        """
        output_dir = Path(workspace_path) / REPRODUCIBILITY_DIR_NAME

        # Clean old bundle before regenerating (handles multi-round analysis)
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        env_txt = output_dir / "environment_frozen.txt"
        sysinfo_json = output_dir / "system_info.json"
        checksums_txt = output_dir / "checksums.sha256"
        commands_py = output_dir / "commands.py"
        decisions_md = output_dir / "agent_decisions.md"
        manifest_json = output_dir / "manifest.json"

        # File checksums (compute before writing bundle files so rglob skips an empty dir)
        try:
            checksums_txt.write_text(
                self._compute_checksums(workspace_path, upload_paths)
            )
        except Exception as e:
            logger.warning(f"Checksum computation failed: {e}")
            checksums_txt.write_text(f"# Checksum computation failed: {e}\n")

        # Environment snapshot (subprocess, no sandbox needed)
        try:
            self._capture_environment(env_txt, sysinfo_json)
        except Exception as e:
            logger.warning(f"Environment capture failed: {e}")
            env_txt.write_text(f"# Environment capture failed: {e}\n")
            self._write_json(
                {"error": str(e), "scai_version": _get_scai_version()},
                sysinfo_json,
            )

        # Extract Python commands from notebook
        if ipynb_path and Path(ipynb_path).exists():
            try:
                commands_py.write_text(self._extract_commands(ipynb_path))
            except Exception as e:
                logger.warning(f"Command extraction failed: {e}")
                commands_py.write_text(
                    "#!/usr/bin/env python3\n# Command extraction failed\n"
                )
        else:
            commands_py.write_text(
                "#!/usr/bin/env python3\n# No notebook available\n"
            )

        # Agent decision summary from round_results
        try:
            decisions_md.write_text(self._compile_agent_decisions(round_results))
        except Exception as e:
            logger.warning(f"Agent decision compilation failed: {e}")
            decisions_md.write_text("# Agent decision compilation failed\n")

        # Manifest
        file_list = [f.name for f in sorted(output_dir.iterdir()) if f.is_file()]
        self._write_manifest(manifest_json, session_id, file_list)

        logger.info(f"Reproducibility bundle generated at {output_dir}")
        return str(output_dir)

    # ------------------------------------------------------------------
    # Environment Capture (subprocess-based, no sandbox)
    # ------------------------------------------------------------------

    def _capture_environment(self, env_path: Path, sysinfo_path: Path) -> None:
        """Capture Python environment using subprocess in the backend process."""
        # System info
        info: Dict[str, Any] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "scai_version": _get_scai_version(),
            "cpu_count": os.cpu_count(),
        }
        try:
            import psutil
            info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            info["memory_gb"] = "unknown"
        self._write_json(info, sysinfo_path)

        # pip freeze
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--all"],
                capture_output=True, text=True, timeout=60,
            )
            env_path.write_text(result.stdout.strip() + "\n")
        except Exception as e:
            env_path.write_text(f"# pip freeze failed: {e}\n")

    # ------------------------------------------------------------------
    # Checksums
    # ------------------------------------------------------------------

    def _compute_checksums(
        self, workspace_path: str, upload_paths: List[str]
    ) -> str:
        """Compute SHA256 for all data/output files, skipping reproducibility/."""
        lines: List[str] = []

        wsp = Path(workspace_path)
        if wsp.exists():
            for f in sorted(wsp.rglob("*")):
                if not f.is_file():
                    continue
                if _is_excluded(f):
                    continue
                # Skip the reproducibility subtree itself
                rel = f.relative_to(wsp)
                if rel.parts and rel.parts[0] == REPRODUCIBILITY_DIR_NAME:
                    continue
                sha = self._sha256_file(str(f))
                lines.append(f"{sha}  {rel}")

        # Upload files
        for up in upload_paths:
            p = Path(up)
            if p.exists() and p.is_file() and not _is_excluded(p):
                sha = self._sha256_file(str(p))
                lines.append(f"{sha}  uploads/{p.name}")

        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _sha256_file(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Command Extraction
    # ------------------------------------------------------------------

    def _extract_commands(self, ipynb_path: str) -> str:
        """Extract Python code cells from notebook into a runnable script."""
        with open(ipynb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        lines = [
            "#!/usr/bin/env python3",
            "# Generated by ScAI Reproducibility Bundle",
            f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# Run: python commands.py",
            "# Requires the environment from environment_frozen.txt",
            '"""Reproducible analysis commands extracted from the Jupyter notebook."""',
        ]

        pip_prefixes = ("pip install", "!pip install", "%pip install", "pip3 install")

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source.strip()
            if not source:
                continue
            # Skip pip install (covered by environment snapshot)
            if any(source.startswith(p) for p in pip_prefixes):
                continue
            # Skip cells that contain only comments (no executable code)
            if all(ln.strip() == "" or ln.strip().startswith("#") for ln in source.split("\n")):
                continue

            lines.append("")
            lines.append(f"# === Cell {i + 1} ===")
            lines.extend(source.split("\n"))

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Agent Decision Summary
    # ------------------------------------------------------------------

    def _compile_agent_decisions(self, round_results: List[RoundResult]) -> str:
        lines = [
            "## Agent Decision Log",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This document records key analytical decisions made by ScAI agents.",
            "Each entry includes the agent's reasoning for the chosen approach.",
            "",
            "---",
            "",
        ]

        if not round_results:
            lines.append("No agent decision records available.\n")
            return "\n".join(lines)

        step = 0
        for rr in round_results:
            rn = getattr(rr, "round_number", "?")
            findings = getattr(rr, "bioinformatics_findings", []) or []

            if not findings:
                continue

            lines.append(f"## Round {rn}")
            lines.append("")

            for finding in findings:
                step += 1
                text = finding if isinstance(finding, str) else str(finding)
                if text.strip():
                    lines.append(f"### Step {step}")
                    lines.append("")
                    lines.append(f"**Decision:** {text}")
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _write_manifest(
        self, path: Path, session_id: str, file_list: List[str]
    ) -> None:
        manifest = {
            "scai_version": _get_scai_version(),
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "contents": {
                f: {
                    "description": self._describe_file(f),
                }
                for f in sorted(file_list)
            },
        }
        self._write_json(manifest, path)

    @staticmethod
    def _describe_file(filename: str) -> str:
        descriptions = {
            "environment_frozen.txt": "Complete pip freeze output with exact package versions",
            "system_info.json": "Python version, OS, CPU, memory, ScAI version",
            "checksums.sha256": "SHA256 hashes of all input and output files",
            "commands.py": "Executable Python script extracted from analysis notebook",
            "agent_decisions.md": "Agent reasoning for key analytical decisions",
            "manifest.json": "This file — bundle metadata",
        }
        return descriptions.get(filename, f"Reproducibility artifact: {filename}")

    # ------------------------------------------------------------------
    # Zip Creation (reused by download endpoint)
    # ------------------------------------------------------------------

    @staticmethod
    def create_zip_bytes(source_dir: str) -> BinaryIO:
        """Create a seekable ZIP stream that rolls large bundles to disk.

        The historical method name is kept for callers, but its backing store
        is no longer an unbounded ``BytesIO``.
        """
        buf = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
        source = Path(source_dir)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(source.iterdir()):
                if f.is_file() and not f.is_symlink():
                    zf.write(str(f), f.name)
        buf.seek(0)
        return buf

    def _create_zip(self, source_dir: str, zip_path: str) -> None:
        """Create a zip archive on disk of the reproducibility directory."""
        source = Path(source_dir)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for artifact in sorted(source.iterdir()):
                if artifact.is_file() and not artifact.is_symlink():
                    zf.write(artifact, artifact.name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(data: Any, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
