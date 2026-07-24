import re
import pathlib
from backend.constants import VERSION


def test_claude_md_version_matches():
    txt = pathlib.Path("CLAUDE.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"CLAUDE.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"


def test_agents_md_version_matches():
    txt = pathlib.Path("AGENTS.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"AGENTS.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"


def test_readme_md_version_matches():
    txt = pathlib.Path("README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Version\*\*:\s*(\S+)", txt)
    assert m and m.group(1) == VERSION, f"README.md version {m.group(1) if m else 'NOT FOUND'} != {VERSION}"
