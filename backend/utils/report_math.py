"""Protect TeX during Markdown conversion and render it in offline HTML reports."""

from functools import lru_cache
from html import escape
from pathlib import Path
import re


# Code is consumed first so examples containing dollar signs remain literal.
_TOKENS = re.compile(
    r"(?P<code>(?P<ticks>`+)[\s\S]*?(?P=ticks)|~~~[\s\S]*?~~~)"
    r"|(?<!\\)(?P<block>\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])"
    r"|(?<!\\)(?P<paren>\\\([^\n]+?\\\))"
    r"|(?<![\\$])(?P<inline>\$(?![\s$])(?:\\[^\n]|[^$\n\\])+?(?<!\s)\$(?!\$))"
)


def protect_report_math(text: str) -> tuple[str, dict[str, str]]:
    """Replace math with opaque tokens; restore only after Markdown processing."""
    replacements = {}

    def protect(match):
        if match.group("code") is not None:
            return match.group(0)
        raw = match.group(0)
        display = match.group("block") is not None
        width = 2 if display or match.group("paren") is not None else 1
        tex = raw[width:-width]
        token = f"\x00SCAIMATH{len(replacements)}\x00"
        replacements[token] = (
            f'<span data-scai-math="{escape(tex, quote=True)}" '
            f'data-display="{str(display).lower()}">{escape(raw)}</span>'
        )
        return token

    return _TOKENS.sub(protect, text), replacements


def restore_report_math(text: str, replacements: dict[str, str]) -> str:
    for token, markup in replacements.items():
        text = text.replace(token, markup)
    return text


@lru_cache(maxsize=1)
def _math_script() -> str:
    # MathML uses the browser's native layout: no CDN, fonts, or CSS downloads.
    static = Path(__file__).parents[1] / "static"
    source = (static / "vendor/katex/katex.min.js").read_text(encoding="utf-8")
    render = (static / "report_math.js").read_text(encoding="utf-8")
    return '<script id="scai-report-math">\n' + source + '\n' + render + '\n</script>'


def embed_report_math(document: str) -> str:
    """Add the bundled renderer once, after all report content is available."""
    if 'data-scai-math="' not in document or 'id="scai-report-math"' in document:
        return document
    return document.replace("</body>", _math_script() + "\n</body>", 1)
