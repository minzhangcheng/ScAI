"""P1-3: Bioinformatics agent identity must be derived from data files,
not hardcoded as single-cell RNA-seq."""
import re

import backend.agents.bioinformatics_agent as bio_mod


def test_static_system_prompt_uses_dynamic_modality():
    """P1-3: bioinfo agent identity must be derived from data, not hardcoded."""
    src = open(bio_mod.__file__).read()
    m = re.search(r"def _static_system_prompt\b.*?(?=\n    (?:async )?def |\Z)", src, re.DOTALL)
    assert m
    body = m.group(0)
    assert "You are conducting single-cell RNA-seq analysis" not in body, \
        "P1-3: identity must be derived from data, not hardcoded"
    assert "modality" in body.lower() or "_infer_modality" in body, \
        "P1-3: prompt must use a modality variable"


def test_infer_modality_helper_exists():
    src = open(bio_mod.__file__).read()
    assert "def _infer_modality" in src, \
        "P1-3: helper function _infer_modality must exist"
    inf = re.search(r"def _infer_modality\b.*?(?=\n    (?:async )?def |\Z)", src, re.DOTALL)
    assert inf
    inf_body = inf.group(0)
    for token in ["single-cell", "bulk", "transcriptomics"]:
        assert token in inf_body.lower(), f"P1-3: _infer_modality missing {token}"
