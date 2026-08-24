import re
import pathlib

import pytest

from backend.constants import VERSION


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_claude_md_version_matches():
    txt = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"CLAUDE.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"


def test_agents_md_version_matches():
    txt = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"AGENTS.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"


def test_readme_md_version_matches():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"README.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"


def test_python_client_version_matches():
    """The Python client follows the application release version."""
    txt = pathlib.Path("clients/python/scai_client/__init__.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r'^__version__ = "([^"]+)"', txt, re.MULTILINE)
    assert m and m.group(1) == VERSION, (
        f"scai_client version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"
    )


def test_handover_version_matches():
    txt = (ROOT / "docs/HANDOVER.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"docs/HANDOVER.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"


def test_agents_current_version_registry_row_matches():
    txt = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*Version\s*\|.*$", txt, re.MULTILINE)
    assert any(re.search(rf'Currently\s+`"{re.escape(VERSION)}"`', row) for row in rows), (
        f"AGENTS.md Version registry row does not record current VERSION {VERSION}"
    )


@pytest.mark.parametrize("manual", ["docs/admin-manual.md", "docs/admin-manual-zh.md"])
def test_admin_manual_build_examples_match_version(manual):
    txt = (ROOT / manual).read_text(encoding="utf-8")
    assert f"./build.sh --version {VERSION}" in txt, f"{manual} is missing the versioned build example"
    assert f"./build.sh --version {VERSION} --push" in txt, f"{manual} is missing the versioned push example"
