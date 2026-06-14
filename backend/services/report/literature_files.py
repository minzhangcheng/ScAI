"""Render and persist the combined literature research report as a single
downloadable HTML file (linked from the report appendix, like the notebook).
"""
import html as html_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.utils.report_html import markdown_to_html

LITERATURE_REPORT_FILENAME = "literature_report.html"
LITERATURE_REPORT_LABEL = "Literature Research Report"


def render_literature_download_li(href: str = LITERATURE_REPORT_FILENAME) -> str:
    """Render the appendix download <li> for the literature report, consistently styled."""
    esc = html_module.escape
    return (
        f'<li><a href="{esc(href)}" download style="color:#4f46e5;">'
        f'{LITERATURE_REPORT_LABEL}</a></li>'
    )


def _get(obj: Union[Dict[str, Any], Any], key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _render_reference(ref: Dict[str, str]) -> str:
    """Render one reference dict as an escaped <li>. Links title via url if present."""
    esc = html_module.escape
    title = ref.get("title") or ref.get("name") or ""
    url = ref.get("url") or ref.get("link") or ""
    if not title and not url:
        # Fall back to joining whatever string values exist
        title = ", ".join(str(v) for v in ref.values() if v)
    if url:
        return f'<li><a href="{esc(url)}" target="_blank" rel="noopener">{esc(title or url)}</a></li>'
    return f"<li>{esc(title)}</li>"


def build_literature_report_html(literature_reports: List[Union[Dict[str, Any], Any]]) -> str:
    """Build a self-contained HTML document for all literature reports. '' if none."""
    if not literature_reports:
        return ""
    esc = html_module.escape
    blocks: List[str] = []
    for lr in literature_reports:
        topic = esc(str(_get(lr, "topic", "") or "Untitled topic"))
        body_md = str(_get(lr, "report", "") or "")
        body_html = markdown_to_html(body_md) if body_md else "<p>(No content)</p>"
        refs = _get(lr, "references", []) or []
        refs_html = ""
        if refs:
            items = "".join(_render_reference(r if isinstance(r, dict) else {"title": str(r)}) for r in refs)
            refs_html = f"<h3>References</h3><ol>{items}</ol>"
        blocks.append(
            f'<section class="lit-report"><h2>{topic}</h2>{body_html}{refs_html}</section>'
        )

    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Literature Research Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; line-height: 1.6;
         max-width: 860px; margin: 40px auto; padding: 0 20px; color: #1e293b; }}
  h1 {{ color: #15803d; }}
  section.lit-report {{ margin-bottom: 48px; padding-bottom: 24px; border-bottom: 1px solid #e2e8f0; }}
  section.lit-report h2 {{ color: #047857; }}
  ol, ul {{ padding-left: 22px; }}
  a {{ color: #4f46e5; }}
</style>
</head>
<body>
<h1>Literature Research Report</h1>
{body}
</body>
</html>"""


def write_literature_report(
    literature_reports: List[Union[Dict[str, Any], Any]],
    out_dir: Union[str, Path],
) -> Optional[Path]:
    """Write literature_report.html into out_dir. Returns the path, or None if empty."""
    html = build_literature_report_html(literature_reports)
    if not html:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LITERATURE_REPORT_FILENAME
    path.write_text(html, encoding="utf-8")
    return path
