"""Build report sections from chat messages (async)."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.utils.formatting import safe_parse_json_array

logger = logging.getLogger(__name__)


def _pair_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair user messages with their assistant responses, skipping delegation_active messages."""
    pairs = []
    filtered = [m for m in (messages or []) if not m.get("delegation_active", False)]
    
    i = 0
    while i < len(filtered):
        if filtered[i].get("role") == "user":
            user_msg = filtered[i]
            if i + 1 < len(filtered) and filtered[i + 1].get("role") == "assistant":
                assistant_msg = filtered[i + 1]
                pairs.append({"user": user_msg, "assistant": assistant_msg})
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return pairs


def _truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars with ellipsis."""
    if not text:
        return ""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def _figure_exists(fig: Dict[str, Any], figures_dir: str) -> bool:
    """Check if a figure file exists on disk. Tolerates both absolute and relative paths."""
    fig_path = fig.get("path") or fig.get("name", "")
    if not fig_path:
        return False
    p = Path(fig_path)
    if p.is_absolute():
        return p.exists()
    return (Path(figures_dir) / p).exists()


async def build_chat_sections_async(
    messages: Optional[List[Dict[str, Any]]],
    figures_dir: str,
    tables_dir: str,
    llm_client: Any = None,
    progress_callback: Optional[Callable] = None,
    max_chat_entries: int = 20,
    model: Optional[str] = None,
    context_budget_callback: Optional[Callable] = None,
    context_budget_ack_callback: Optional[Callable] = None,
    publication_figure_paths: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Process chat messages and return a list of report section dicts.

    Titles are generated via a single batch LLM call when both ``llm_client``
    and ``model`` are provided and no cached ``_summary_title`` exists. Falls
    back to truncated user message on any failure.
    """
    if not messages:
        return []

    pairs = _pair_messages(messages)
    if not pairs:
        return []

    if len(pairs) > max_chat_entries:
        pairs = pairs[-max_chat_entries:]

    needs_title = []
    for idx, pair in enumerate(pairs):
        user_msg = pair["user"]
        if not user_msg.get("_summary_title"):
            needs_title.append((idx, _truncate_text(user_msg.get("content", ""), 500)))

    if needs_title and llm_client is not None and model:
        try:
            prompt_parts = [f"{i+1}. {q}" for i, (_, q) in enumerate(needs_title)]
            prompt = (
                "For each user question below, provide a concise English title (max 10 words).\n"
                + "\n".join(prompt_parts)
                + '\n\nRespond with JSON: [{"index": 1, "title": "..."}, ...]'
            )

            title_messages = [{"role": "user", "content": prompt}]
            if context_budget_callback is None:
                response = await llm_client.chat(
                    model,
                    messages=title_messages,
                    prefer_stream=True,
                )
            else:
                dispatch = await context_budget_callback(
                    "supervisor",
                    title_messages,
                )
                response = await llm_client.chat(
                    model,
                    messages=dispatch.messages,
                    **(
                        {"max_tokens": dispatch.max_tokens}
                        if dispatch.max_tokens is not None
                        else {}
                    ),
                    prefer_stream=True,
                )
                if context_budget_ack_callback is not None:
                    await context_budget_ack_callback("supervisor", dispatch)

            content = getattr(response, "content", "")

            titles = safe_parse_json_array(content, default=[])
            if titles:
                for t in titles:
                    pair_idx = t.get("index", 0) - 1
                    if 0 <= pair_idx < len(needs_title):
                        original_idx = needs_title[pair_idx][0]
                        pairs[original_idx]["user"]["_summary_title"] = t.get("title", "")
        except Exception as e:
            logger.warning(f"Batch LLM title generation failed: {e}")
    
    sections = []
    for pair in pairs:
        user_msg = pair["user"]
        assistant_msg = pair["assistant"]
        
        title = user_msg.get("_summary_title") or _truncate_text(user_msg.get("content", ""), 500)
        content = assistant_msg.get("content", "")
        if not content:
            continue
        
        # Chat's direct renderer makes display plots without publication review.
        # Only registered, eligible artifacts may enter the manuscript report.
        figures = [
            fig for fig in (assistant_msg.get("figures") or [])
            if fig.get("figure_role") == "publication"
            and fig.get("artifact_id") in (publication_figure_paths or {})
            and Path(str(fig.get("path") or "")).resolve() == Path(
                publication_figure_paths[fig["artifact_id"]]
            ).resolve()
            and _figure_exists(fig, figures_dir)
        ]
        tables = assistant_msg.get("tables") or []
        
        sections.append({
            "title": title,
            "content": content,
            "figures": figures,
            "tables": tables,
            "analysis_type": "Chat Analysis",
        })
    
    return sections
