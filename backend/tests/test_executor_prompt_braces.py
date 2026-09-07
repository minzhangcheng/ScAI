"""The static executor prompt is assembled from f-string and plain sections.

Doubled braces are the f-string escape; in a plain section they reach the model
verbatim, and ``markers = {{}}`` is then copied into a cell as a set holding a
dict, which raises. The Harmony block is the other section models copy as-is.
"""
from backend.agents.bioinformatics.executor_prompt import build_static_executor_prompt


def _prompt() -> str:
    return build_static_executor_prompt("system prompt", "single-cell", "/tmp/workspace")


def test_no_escaped_braces_survive_into_the_rendered_prompt():
    text = _prompt()
    assert "{{" not in text and "}}" not in text
    assert "markers = {}" in text
    assert 'print(f"obs columns: {list(adata.obs.columns)}")' in text


def test_harmony_block_checks_both_orientations_and_never_blames_filtering():
    text = _prompt()
    assert "Z.shape == X_pca.shape[::-1]" in text
    assert "adata.obsm['X_pca_harmony'] = Z" in text
    assert "ho.Z_corr.T" not in text
    assert "filter cells between" not in text
