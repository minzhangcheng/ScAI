import re
import pathlib

import pytest

from backend.constants import VERSION


ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("document", ["README.md", "docs/HANDOVER.md"])
def test_document_version_matches(document):
    match = re.search(r"\*\*Version\*\*:\s*(\S+)", (ROOT / document).read_text())
    assert match and match.group(1) == VERSION, f"{document}: {match.group(1) if match else 'NOT FOUND'} != {VERSION}"


def test_python_client_version_matches():
    """The Python client follows the application release version."""
    txt = pathlib.Path("clients/python/scai_client/__init__.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r'^__version__ = "([^"]+)"', txt, re.MULTILINE)
    assert m and m.group(1) == VERSION, (
        f"scai_client version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"
    )


def test_changelog_starts_with_current_version():
    text = (ROOT / "docs/CHANGELOG.md").read_text(encoding="utf-8")
    releases = re.findall(r"^- `(\d+\.\d+\.\d+)`", text, re.MULTILINE)
    assert releases and releases[0] == VERSION, (
        f"CHANGELOG.md must list current VERSION {VERSION} first"
    )


@pytest.mark.parametrize("manual", ["docs/admin-manual.md", "docs/admin-manual-zh.md"])
def test_admin_manual_build_examples_match_version(manual):
    txt = (ROOT / manual).read_text(encoding="utf-8")
    assert f"./build.sh --version {VERSION}" in txt, f"{manual} is missing the versioned build example"
    assert f"./build.sh --version {VERSION} --push" in txt, f"{manual} is missing the versioned push example"
