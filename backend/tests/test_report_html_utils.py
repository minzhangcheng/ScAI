"""
Smoke tests for report HTML utilities.
"""
import pytest
import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.report_html import markdown_to_html, HTML_TEMPLATE, _make_figure_block


class TestMarkdownToHtml:
    """Tests for markdown_to_html function."""

    def test_markdown_to_html_basic(self):
        """Assert headers, bold, and lists render correctly."""
        md = """# Main Title

## Section Header

This is **bold text** and some *italic text*.

- Item one
- Item two
- Item three
"""
        html = markdown_to_html(md)

        # Headers
        assert "<h1>Main Title</h1>" in html
        assert "<h2>Section Header</h2>" in html

        # Bold
        assert "<strong>bold text</strong>" in html

        # Lists
        assert "<ul>" in html
        assert "<li>Item one</li>" in html
        assert "<li>Item two</li>" in html
        assert "<li>Item three</li>" in html

    def test_markdown_to_html_emphasis(self):
        """Test italic rendering."""
        md = "This is *italic* text."
        html = markdown_to_html(md)
        assert "<em>italic</em>" in html

    def test_markdown_to_html_code(self):
        """Test inline code rendering."""
        md = "Use `print()` to output."
        html = markdown_to_html(md)
        assert "<code>print()</code>" in html


class TestHtmlTemplate:
    """Tests for HTML_TEMPLATE."""

    def test_html_template_placeholders(self):
        """Assert HTML_TEMPLATE contains the six required placeholders.

        The template uses toc_items (not toc_html) for the table of contents list.
        """
        required_placeholders = [
            "{title}",
            "{sections_html}",
            "{agent_stats_html}",
            "{score_definitions_html}",
            "{appendix_html}",
            "{toc_items}",  # Note: template uses toc_items, not toc_html
        ]

        for placeholder in required_placeholders:
            assert placeholder in HTML_TEMPLATE, f"Missing placeholder: {placeholder}"


class TestMakeFigureBlock:
    """Tests for _make_figure_block function."""

    def test_make_figure_block_basic(self):
        """Assert _make_figure_block returns figure with caption."""
        src = "data:image/png;base64,abc123"
        caption = "Test Figure Caption"
        result = _make_figure_block(src, caption)

        # Should contain figure element and caption
        assert "<figure" in result
        assert caption in result

    def test_make_figure_block_with_custom_class(self):
        """Assert custom CSS class is applied."""
        src = "image.png"
        caption = "My Figure"
        css_class = "custom-figure"
        result = _make_figure_block(src, caption, css_class=css_class)

        assert css_class in result
        assert caption in result
