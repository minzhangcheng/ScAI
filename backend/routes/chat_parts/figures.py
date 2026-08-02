# Figure generation inside a chat turn and session figure collection.
# Moved verbatim out of backend/routes/chat.py.
"""Chat routes for PI Agent free-form chat with analysis sessions."""
import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from backend.config import get_session_workspace, get_session_reports, get_session_uploads, get_session_states, PI_AGENT_NAME
from backend.models.schemas import WebSocketMessage, AgentType, DataFile, AnalysisState
from backend.services.session_store import ensure_session_loaded, async_save_session, update_session_cache
from backend.state import (
    chat_tasks, chat_task_lock, chat_sandboxes,
    websockets, workflows, state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL


logger = logging.getLogger(__name__)


def _build_figure_setup_code(fig_dir, ts="test_ts", code="pass") -> str:
    dir_literal = repr(str(fig_dir))
    return (
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "import io as _io\n"
        "from IPython.display import display as _display, Image as _Image\n"
        + _TPL_NOOP_INSTALL +
        "_code_error = None\n"
        "try:\n"
        f"    {code.replace(chr(10), chr(10) + '    ')}\n"
        "except Exception as _e:\n"
        "    _code_error = str(_e)\n"
        "    print(f'CODE_ERROR: {_e}')\n"
        "    import traceback; traceback.print_exc()\n"
        "finally:\n"
        "    plt.show = _orig_show\n"
        "    plt.close = _orig_close\n"
        "# Capture all open figures as PNG + save PDF for download\n"
        "_fignums = plt.get_fignums()\n"
        "for _fn in _fignums:\n"
        "    _f = plt.figure(_fn)\n"
        "    # Save high-res PDF\n"
        f"    _f.savefig({dir_literal} + '/chat_{ts}_' + str(_fn) + '.pdf', "
        "format='pdf', dpi=300, bbox_inches='tight', facecolor='white')\n"
        "    # Capture PNG for display\n"
        "    _buf = _io.BytesIO()\n"
        "    _f.savefig(_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')\n"
        "    _buf.seek(0)\n"
        "    _display(_Image(data=_buf.getvalue()))\n"
        "plt.close('all')\n"
    )

async def _generate_figures_in_chat(
    session_id: str, state: Any, descriptions: List[str]
) -> List[Dict[str, str]]:
    """Generate figures via bioinformatics sandbox for chat responses."""
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.utils.agent_llm import get_agent_llm_client

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)

    generated_figures: List[Dict[str, str]] = []

    sandbox = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(get_session_states(session_id)),
        working_dir=str(get_session_workspace(session_id)),
    )
    chat_sandboxes[session_id] = sandbox
    try:
        start_info = await sandbox.start()
        if not start_info.get("session_restored"):
            logger.warning("Chat sandbox: session not restored, figure generation may fail")

        for desc_idx, desc in enumerate(descriptions):
            # Ask LLM to write figure code
            code_prompt = (
                "You are a bioinformatics expert. Write Python code using scanpy and matplotlib "
                "to generate the following figure. The variable `adata` is already loaded in the "
                "environment with all prior analysis results.\n"
                "IMPORTANT: End your code with plt.show() to display the figure. "
                "Do NOT call plt.close(). Do NOT use plt.savefig().\n"
                "Output ONLY a single ```python code block, no explanations.\n\n"
                f"Figure request: {desc}"
            )

            llm_client = get_agent_llm_client(state, config, "bioinformatics", session_id)
            provider = AgentSettingsProvider(state, config, None, session_id)
            bio_model = provider.get_model("bioinformatics")
            if not bio_model:
                raise HTTPException(
                    status_code=500,
                    detail="Bioinformatics model not configured for this session.",
                )
            code_resp = await llm_client.chat(
                model=bio_model,
                messages=[{"role": "user", "content": code_prompt}],
                temperature=0.1,
                max_tokens=32768,
            )

            code_text = code_resp.content or ""
            from backend.utils.sandbox import extract_code_from_markdown
            code = extract_code_from_markdown(code_text)
            if not code.strip():
                logger.warning(f"Chat sandbox: empty code for description '{desc[:80]}'")
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            fig_dir = get_session_workspace(session_id) / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)

            wrapped = _build_figure_setup_code(fig_dir, ts, code)

            result = await sandbox.execute(wrapped, save_to_history=False)
            logger.info(f"Chat sandbox execute: success={result.success}, "
                        f"kernel_figures={len(result.figures)}, error={result.error}")
            if result.output:
                logger.info(f"Chat sandbox output: {result.output[:2000]}")

            # Primary: use kernel-captured base64 figures (most reliable)
            if result.figures:
                import base64 as _b64
                for i, b64_data in enumerate(result.figures):
                    fig_name = f"chat_{ts}_{i + 1}"
                    fig_path = fig_dir / f"{fig_name}.png"
                    pdf_path = fig_dir / f"{fig_name}.pdf"
                    try:
                        fig_path.write_bytes(_b64.b64decode(b64_data))
                        logger.info(f"Chat sandbox: saved figure to {fig_path}")
                        entry: Dict[str, str] = {
                            "name": fig_name,
                            "path": str(fig_path),
                            "base64": b64_data,
                        }
                        # PDF was saved by the kernel code directly
                        if pdf_path.exists():
                            entry["pdfPath"] = str(pdf_path)
                        generated_figures.append(entry)
                    except Exception as save_err:
                        logger.warning(f"Chat sandbox: disk save failed ({save_err}), using base64 only")
                        generated_figures.append({
                            "name": fig_name,
                            "base64": b64_data,
                        })
            else:
                logger.warning(f"Chat sandbox: no figures captured from kernel output "
                               f"(success={result.success})")

    except Exception as e:
        logger.exception("Chat figure generation error")
    finally:
        try:
            from backend.services.session_lifecycle import is_session_tombstoned

            await sandbox.stop(
                save=not is_session_tombstoned(session_id)
            )
        except Exception:
            logger.warning("Failed to stop chat code sandbox during cleanup", exc_info=True)
        finally:
            # A strict final-save failure deliberately leaves km/kc live. Keep
            # the only registry handle so stale-session cleanup can retry it.
            if getattr(sandbox, "km", None) is None:
                if chat_sandboxes.get(session_id) is sandbox:
                    chat_sandboxes.pop(session_id, None)

    return generated_figures

def _collect_session_figures(
    session_id: str,
    since_time: Optional[float] = None
) -> List[Dict[str, str]]:
    """Collect figures from workspace, optionally filtering by creation time.

    Scans both the figures/ subdirectory (generate_publication_figure output)
    and the workspace root (execute_code explicit savefig output, e.g. UMAP plots).
    """
    figures: List[Dict[str, str]] = []
    seen_stems: set = set()

    def _add_png(png: Path) -> None:
        if since_time and png.stat().st_mtime < since_time:
            return
        if png.stem in seen_stems:
            return
        entry: Dict[str, str] = {"name": png.stem, "path": str(png)}
        pdf_path = png.with_suffix(".pdf")
        if pdf_path.exists():
            entry["pdfPath"] = str(pdf_path)
        figures.append(entry)
        seen_stems.add(png.stem)

    # 1. figures/ subdirectory (highest priority — publication figures)
    fig_dir = get_session_workspace(session_id) / "figures"
    if fig_dir.exists():
        for png in sorted(list(fig_dir.glob("*.jpg")) + list(fig_dir.glob("*.png")), key=lambda p: p.stat().st_mtime, reverse=True):
            _add_png(png)

    # 2. Workspace root — picks up UMAP and other plots saved via explicit plt.savefig()
    #    when LLM code calls plt.close() before the auto-save wrapper can capture them.
    root_dir = get_session_workspace(session_id)
    if root_dir.exists():
        for png in sorted(list(root_dir.glob("*.jpg")) + list(root_dir.glob("*.png")), key=lambda p: p.stat().st_mtime, reverse=True):
            _add_png(png)

    figures.sort(key=lambda e: Path(e["path"]).stat().st_mtime, reverse=True)
    return figures[:30]
