# Figure generation inside a chat turn and session figure collection.
# Moved verbatim out of backend/routes/chat.py.
"""Chat routes for PI Agent free-form chat with analysis sessions."""
import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from backend.config import get_session_workspace, get_session_reports, get_session_uploads, get_session_states, PI_AGENT_NAME
from backend.constants import NO_OUTPUT_TOKEN_CAP
from backend.models.schemas import WebSocketMessage, AgentType, DataFile, AnalysisState
from backend.services.session_store import ensure_session_loaded, async_save_session, update_session_cache
from backend.state import (
    chat_tasks, chat_task_lock, chat_sandboxes,
    websockets, workflows, report_workflows, state_lock,
    has_live_session_task,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ChatRuntimeReservation:
    """Exact owners installed at the shared workflow/runtime start barrier."""

    task: asyncio.Task
    initializing_token: object
    claimed_chat_task: bool


def _session_runtime_is_busy(
    session_id: str,
    *,
    allow_existing_chat_owner: bool = False,
) -> bool:
    from backend.state import initializing_sessions

    if (
        session_id in workflows
        or session_id in report_workflows
        or session_id in initializing_sessions
    ):
        return True
    live_sandbox = chat_sandboxes.get(session_id)
    if live_sandbox is not None and getattr(live_sandbox, "km", None) is not None:
        return True
    owner = chat_tasks.get(session_id)
    if owner is asyncio.current_task() or (owner is not None and allow_existing_chat_owner):
        return False
    return has_live_session_task(session_id)


async def _reserve_chat_runtime(
    session_id: str,
    *,
    allow_existing_chat_owner: bool = False,
) -> Optional[_ChatRuntimeReservation]:
    owner = asyncio.current_task()
    if owner is None:
        return None

    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.state import claim_initializing_session

    async with get_lifecycle_lock(session_id):
        async with chat_task_lock:
            existing = chat_tasks.get(session_id)
            owns_chat_task = existing is owner
            if _session_runtime_is_busy(
                session_id,
                allow_existing_chat_owner=allow_existing_chat_owner,
            ):
                return None
            token = claim_initializing_session(session_id)
            claimed_chat_task = existing is None or existing.done()
            if claimed_chat_task:
                chat_tasks[session_id] = owner
            return _ChatRuntimeReservation(
                task=owner,
                initializing_token=token,
                claimed_chat_task=claimed_chat_task and not owns_chat_task,
            )


async def _release_chat_runtime(
    session_id: str,
    reservation: Optional[_ChatRuntimeReservation],
) -> None:
    if reservation is None:
        return
    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.state import release_initializing_session

    async with get_lifecycle_lock(session_id):
        async with chat_task_lock:
            if (
                reservation.claimed_chat_task
                and chat_tasks.get(session_id) is reservation.task
            ):
                chat_tasks.pop(session_id, None)
            release_initializing_session(
                session_id,
                reservation.initializing_token,
            )


async def _release_chat_runtime_deferred(
    session_id: str,
    reservation: Optional[_ChatRuntimeReservation],
) -> None:
    from backend.utils.asyncio_tasks import await_with_deferred_cancellation

    task = asyncio.create_task(_release_chat_runtime(session_id, reservation))
    await await_with_deferred_cancellation(task)


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

    reservation = await _reserve_chat_runtime(session_id)
    if reservation is None:
        logger.warning(
            "Chat figure generation skipped for %s: another owner holds the session runtime",
            session_id,
        )
        return []

    generated_figures: List[Dict[str, str]] = []
    try:
        user_id = getattr(state, "user_id", "") if state else ""
        config = await resolve_config_for_user(user_id)
        sandbox = LocalJupyterSandbox(
            session_id=session_id,
            state_dir=str(get_session_states(session_id)),
            working_dir=str(get_session_workspace(session_id)),
        )
    except BaseException:
        await _release_chat_runtime_deferred(session_id, reservation)
        raise
    chat_sandboxes[session_id] = sandbox
    started = False
    try:
        start_info = await sandbox.start()
        started = True
        from backend.agents.bioinformatics.executor_runtime import (
            _runtime_marked_unavailable,
        )

        if start_info.get("restore_failed") or _runtime_marked_unavailable(state, session_id):
            # Never save an empty replacement over the unavailable prior runtime.
            started = False
            raise RuntimeError("The previous kernel runtime is unavailable for Chat figures")

        for desc_idx, desc in enumerate(descriptions):
            # Reuse the existing read-only inspector, under its required lock.
            # Inventory is context, not a scientific source-selection decision.
            async with sandbox._exec_lock:
                inventory = await sandbox.inspect_checkpoint_adata_locked()
            inventory_text = json.dumps(inventory, ensure_ascii=False)[:12000]
            code_prompt = (
                "You are a bioinformatics expert. Write Python code using scanpy and matplotlib "
                "to generate the following figure from existing results.\n"
                "Current kernel AnnData inventory (bounded metadata; not an assurance of "
                "result completeness):\n" + inventory_text + "\n"
                "Use the matching existing objects, annotations, and embeddings. Do not "
                "assume a variable named adata or any requested result exists. Inspect "
                "other required objects before using them. If the requested source or "
                "result is missing or ambiguous, report that limitation; do not reload "
                "raw data or recompute QC, clustering, embeddings, or annotations to "
                "satisfy a plotting request. Additional analysis requires PI planning.\n"
                "IMPORTANT: End your code with plt.show() to display the figure. "
                "Do NOT call plt.close(). Do NOT use plt.savefig().\n"
                "Output ONLY a single ```python code block, no explanations.\n\n"
                f"Figure request: {desc}"
            )

            llm_client = get_agent_llm_client(state, config, "bioinformatics", session_id)
            try:
                provider = AgentSettingsProvider(state, config, None, session_id)
                bio_model = provider.get_model("bioinformatics")
                if not bio_model:
                    raise HTTPException(
                        status_code=500,
                        detail="Bioinformatics model not configured for this session.",
                    )
                dispatch = await provider.budget_text_dispatch(
                    "bioinformatics",
                    [{"role": "user", "content": code_prompt}],
                    max_tokens=NO_OUTPUT_TOKEN_CAP,
                )
                code_resp = await llm_client.chat(
                    model=bio_model,
                    messages=dispatch.messages,
                    temperature=0.1,
                    max_tokens=dispatch.max_tokens,
                    prefer_stream=True,
                )
                await provider.acknowledge_text_dispatch(
                    "bioinformatics",
                    dispatch,
                )
            finally:
                try:
                    await llm_client.close()
                except Exception:
                    logger.warning("Failed to close chat figure LLM client", exc_info=True)

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

            result = await sandbox.execute(wrapped, )
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
                save=started and not is_session_tombstoned(session_id)
            )
        except Exception:
            logger.warning("Failed to stop chat code sandbox during cleanup", exc_info=True)
        finally:
            # A strict final-save failure deliberately leaves km/kc live. Keep
            # the only registry handle so stale-session cleanup can retry it.
            if getattr(sandbox, "km", None) is None:
                if chat_sandboxes.get(session_id) is sandbox:
                    chat_sandboxes.pop(session_id, None)
            await _release_chat_runtime_deferred(session_id, reservation)

    return generated_figures

def _collect_session_figures(
    session_id: str,
    since_time: Optional[float] = None,
    *, state=None, run_scope_ids=None,
) -> List[Dict[str, str]]:
    """Collect figures from workspace, optionally filtering by creation time.

    Scans both the figures/ subdirectory (generate_publication_figure output)
    and the workspace root (execute_code explicit savefig output, e.g. UMAP plots).
    """
    if run_scope_ids is not None:
        from backend.services.report.artifacts import (
            eligible_artifacts, verified_artifact_bytes,
        )

        # A reviewed Chat run has producer identities. Never promote its raw
        # autosaves/debug files to final figures through an mtime-only scan.
        scopes = set(run_scope_ids)
        artifacts = [figure for figure in getattr(state, "publication_figures", [])
                     if figure.producer_run_scope_id in scopes]
        figures = []
        for figure in eligible_artifacts(artifacts, state=state):
            if verified_artifact_bytes(figure.png_path, figure, kind="image") is None:
                continue
            entry = {"name": figure.name, "path": figure.png_path,
                     "artifact_id": figure.id,
                     "review_status": figure.review_status.value,
                     "review_unavailable": figure.review_unavailable}
            if verified_artifact_bytes(figure.pdf_path, figure, kind="pdf") is not None:
                entry["pdfPath"] = figure.pdf_path
            figures.append(entry)
        return figures

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
