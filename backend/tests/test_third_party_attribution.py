"""Attribution for MIT-derived content must stay in place.

backend/data/method_rules.yaml is derived from bioSkills, which is MIT
licensed; this project is Apache-2.0. MIT requires that its copyright notice
and permission notice be retained. These tests fail loudly if the notice, the
licence text, or the README acknowledgement is removed or drifts away from the
corpus it describes.
"""
from pathlib import Path

import pytest

from backend.services.method_rules import corpus_meta

_REPO = Path(__file__).resolve().parent.parent.parent
_NOTICES = _REPO / "THIRD_PARTY_NOTICES.md"
_README = _REPO / "README.md"


def test_notices_file_exists():
    assert _NOTICES.exists(), "THIRD_PARTY_NOTICES.md is required for MIT attribution"


def test_notices_contain_the_full_mit_licence_text():
    """MIT requires the permission notice itself, not just a licence name."""
    text = _NOTICES.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in text
    assert "shall be included in all" in text


def test_notices_name_the_copyright_holder():
    text = _NOTICES.read_text(encoding="utf-8")
    assert "Copyright (c) 2026 GPTomics" in text


def test_notices_record_the_exact_commit_used():
    """A derivation is only checkable if the source revision is pinned."""
    text = _NOTICES.read_text(encoding="utf-8")
    commit = str(corpus_meta()["source_commit"])
    assert commit in text, (
        "THIRD_PARTY_NOTICES.md must cite the same commit as method_rules.yaml"
    )


def test_notices_state_what_was_derived_and_what_was_not():
    text = _NOTICES.read_text(encoding="utf-8")
    assert "method_rules.yaml" in text
    assert "not vendored" in text or "No bioSkills source file is vendored" in text


def test_notices_address_licence_compatibility():
    text = _NOTICES.read_text(encoding="utf-8")
    assert "Apache" in text and "MIT" in text


def test_readme_acknowledges_the_source():
    text = _README.read_text(encoding="utf-8")
    assert "bioSkills" in text
    assert "MIT" in text
    assert "THIRD_PARTY_NOTICES.md" in text


def test_readme_license_section_flags_the_mit_portions():
    text = _README.read_text(encoding="utf-8")
    tail = text[text.index("## License"):]
    assert "Apache License 2.0" in tail
    assert "MIT" in tail, (
        "the License section must disclose that some content is MIT-derived"
    )


def test_project_licence_is_still_apache():
    licence = (_REPO / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in licence


@pytest.mark.parametrize("resource", ["MSigDB", "Ensembl", "PanglaoDB", "CellMarker"])
def test_runtime_data_resources_are_disclosed(resource):
    """These are fetched, not redistributed, but users need to know they are used."""
    assert resource in _NOTICES.read_text(encoding="utf-8")


def test_no_bioskills_source_files_are_vendored():
    """The notice claims nothing is vendored; hold it to that."""
    offenders = [p for p in _REPO.rglob("SKILL.md")
                 if ".git" not in p.parts and "node_modules" not in p.parts]
    assert not offenders, f"vendored SKILL.md files found: {offenders}"
