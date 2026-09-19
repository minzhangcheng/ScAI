from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState, IntegratedReportFigureRef
from backend.services.report.integrated_report import sanitize_integrated_report
from backend.workflow.messaging import MessagingMixin


def _refs(count=3):
    return [
        IntegratedReportFigureRef(
            number=number,
            figure_id=f"figure-{number}",
            name=f"Panel {number}",
            render_sha256=str(number) * 64,
        )
        for number in range(1, count + 1)
    ]


def test_sanitizer_removes_method_and_declaration_sections_but_keeps_code():
    source = """# Report

## Results
See [Figure 1](#figure-2).

```markdown
### Methods Summary
example only
```

### Methods Summary
Invented wet-lab protocol.

## Declarations and Metadata
Untrusted declaration content is removed.

### Authoritative Declarations
- Figure 1 is a heatmap.

### PI Figure Dispositions and Reservations
- duplicate model rendering

## Conclusions
Done.
"""
    rendered = sanitize_integrated_report(
        source, _refs(1), ["Structured PI reservation"],
    )

    assert "Invented wet-lab protocol" not in rendered
    assert "Figure 1 is a heatmap" not in rendered
    assert "duplicate model rendering" not in rendered
    assert "Untrusted declaration content is removed" not in rendered
    assert "## Reporting Limitations" in rendered
    assert "```markdown\n### Methods Summary\nexample only\n```" in rendered
    assert "[Figure 1](#figure-1)" in rendered
    assert rendered.index("## Conclusions") < rendered.index("## Reporting Limitations")


def test_sanitizer_normalizes_multi_range_and_invalid_references_only():
    source = (
        "Finding A (Figures 1 and 2). Finding B uses Figures 2-3. "
        "Unsupported (Figure 9). The conclusion is unchanged."
    )
    rendered = sanitize_integrated_report(source, _refs(3))

    assert "[Figure 1](#figure-1)" in rendered
    assert rendered.count("[Figure 2](#figure-2)") == 2
    assert "[Figure 3](#figure-3)" in rendered
    assert "Figure 9" not in rendered
    assert "()" not in rendered
    assert "The conclusion is unchanged." in rendered


def test_structured_pi_limitations_are_appended_verbatim():
    rendered = sanitize_integrated_report(
        "# Report\n\n## Results\nNo numbered figures.",
        [],
        ["planned.panel: omit — source render was unavailable"],
    )
    assert "## Reporting Limitations" in rendered
    assert "planned.panel: omit — source render was unavailable" in rendered


def test_sanitizer_removes_setext_closed_atx_and_declaration_aliases():
    source = """# Report

Methods Summary
---------------
Invented sample preparation.

## Results
Supported result.

## Methodology ##
Invented sequencing protocol.

## Figure Inventory
- Figure 1 is a heatmap.

## Conclusions
Done.
"""
    rendered = sanitize_integrated_report(source, _refs(1))

    assert "Invented sample preparation" not in rendered
    assert "Invented sequencing protocol" not in rendered
    assert "is a heatmap" not in rendered
    assert "Figure Inventory" not in rendered
    assert "Supported result" in rendered
    assert "## Conclusions" in rendered


@pytest.mark.parametrize("heading", [
    "## 7. Methods Summary",
    "## **7. Methods Summary** ##",
    "### 7.1 Materials and Methods",
    "7) Methods Summary\n------------------",
])
def test_sanitizer_removes_numbered_methods_without_removing_results(heading):
    source = (
        f"# Report\n\n{heading}\nUnsupported experimental protocol.\n\n"
        "## Results\nRetained computational finding."
    )
    rendered = sanitize_integrated_report(source, [])
    assert "Unsupported experimental protocol" not in rendered
    assert "Retained computational finding" in rendered


def test_sanitizer_repairs_noncanonical_links_without_nesting_markdown():
    source = (
        "See [Figure 1](#wrong) and [Figure 2](https://example.test/figure). "
        "Keep [the source](https://example.test/Figure-1) and "
        "![Figure 1](https://example.test/image.png)."
    )
    rendered = sanitize_integrated_report(source, _refs(2))

    assert "[Figure 1](#figure-1)" in rendered
    assert "[Figure 2](#figure-2)" in rendered
    assert "[[Figure" not in rendered
    assert "[the source](https://example.test/Figure-1)" in rendered
    assert "![Figure 1](https://example.test/image.png)" in rendered


def test_sanitizer_protects_reference_links_nested_destinations_and_inline_code():
    source = (
        "See [Figure 1][panel] and [Figure 2](https://example.test/a_(nested)). "
        "Keep [the source](https://example.test/a_(Figure-1)) and `Figure 9`.\n\n"
        "[panel]: #wrong"
    )
    rendered = sanitize_integrated_report(source, _refs(2))

    assert "[Figure 1](#figure-1)" in rendered
    assert "[Figure 2](#figure-2)" in rendered
    assert "[the source](https://example.test/a_(Figure-1))" in rendered
    assert "`Figure 9`" in rendered
    assert "[panel]: #wrong" in rendered
    assert "[[Figure" not in rendered


@pytest.mark.asyncio
async def test_strict_state_sync_rejects_unacknowledged_persistence():
    class Harness(MessagingMixin):
        session_id = "report-test"
        message_callback = AsyncMock()
        state_callback = AsyncMock(return_value=False)

    state = AnalysisState(user_id="user", initial_instruction="test")
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        await Harness()._send_state_sync(
            state, "integrated_report_published", raise_on_save_error=True,
        )

    Harness.message_callback.assert_not_awaited()
