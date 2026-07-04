from backend.models.schemas import AnalysisState
from backend.agents.prompt_blocks import build_integration_block


def _state(reports):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.integration_reports = list(reports)
    return s


def test_empty_returns_empty_string_for_all_styles():
    s = _state([])
    assert build_integration_block(s, style="final") == ""
    assert build_integration_block(s, style="integrated") == ""
    assert build_integration_block(s, style="detailed") == ""


def test_cycle_headers_and_truncation():
    long = "x" * 3000
    s = _state(["first", long])
    for style in ("final", "integrated", "detailed"):
        out = build_integration_block(s, style=style)
        assert "### Integration Cycle 1" in out
        assert "### Integration Cycle 2" in out
        assert "x" * 2000 in out
        assert "x" * 2001 not in out  # truncated to 2000 chars


def test_detailed_style_has_no_wrapper_header():
    s = _state(["only"])
    out = build_integration_block(s, style="detailed")
    assert out.startswith("### Integration Cycle 1")


def test_build_integration_block_style_is_literal_typed():
    import inspect
    from backend.agents import prompt_blocks

    src = inspect.getsource(prompt_blocks.build_integration_block)
    assert "Literal[" in inspect.getsource(prompt_blocks)
    assert "style: IntegrationBlockStyle" in src
