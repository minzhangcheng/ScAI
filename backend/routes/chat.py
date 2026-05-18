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
from backend.utils.snapshot import SnapshotManager
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["chat"])



def _build_chat_context(state: Any) -> tuple:
    """Build analysis context summary for PI Agent chat.

    Returns:
        (context_str, figure_inventory) where figure_inventory is a list of
        {"name": str, "path": str} dicts for PNG files found on disk.
    """
    parts = []

    if state.initial_instruction:
        parts.append(f"Research Goal: {state.initial_instruction}")

    parts.append(f"Current Round: {state.current_round}")
    parts.append(f"Status: {state.status.value}")

    # Experimental design summary
    design_summary = state.build_experimental_design_summary()
    if design_summary:
        parts.append(design_summary)

    # Biochemistry indicators summary
    if state.biochemistry_indicators:
        indicators = [f"{r.indicator}={r.value}{r.unit}" for r in state.biochemistry_indicators]
        parts.append(f"Biochemistry: {', '.join(indicators)}")

    # Round results (last 3)
    for rr in state.round_results[-3:]:
        round_parts = [f"\n--- Round {rr.round_number} Summary ---"]
        if rr.pathology_findings:
            round_parts.append(f"Pathology: {'; '.join(rr.pathology_findings[:3])}")
        if rr.bioinformatics_findings:
            round_parts.append(f"Bioinformatics: {'; '.join(rr.bioinformatics_findings[:3])}")
        if rr.biochemistry_findings:
            round_parts.append(f"Biochemistry: {'; '.join(rr.biochemistry_findings[:3])}")
        if rr.literature_findings:
            round_parts.append(f"Literature: {'; '.join(rr.literature_findings[:3])}")
        if rr.pi_summary:
            round_parts.append(f"PI Summary: {rr.pi_summary[:500]}")
        parts.extend(round_parts)

    # Final summary
    if state.final_summary:
        parts.append(f"\nFinal Summary: {state.final_summary[:1000]}")

    # Literature reports
    if state.literature_reports:
        parts.append(f"\nLiterature Reports: {len(state.literature_reports)} available")
        for lr in state.literature_reports[-2:]:
            parts.append(f"- Topic: {lr.topic}")

    # Scan disk for actual figure files
    figure_inventory: List[Dict[str, str]] = []
    seen_names: set = set()

    def _add_pngs(directory: Path):
        if not directory.exists():
            return
        # Scan JPG first (new format), then PNG for backward compatibility
        for img in sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png")):
            if img.stem not in seen_names:
                seen_names.add(img.stem)
                figure_inventory.append({"name": img.stem, "path": str(img)})

    _add_pngs(get_session_workspace(state.id))
    _add_pngs(get_session_workspace(state.id) / "figures")
    _add_pngs(get_session_reports(state.id) / "figures")

    if figure_inventory:
        parts.append(f"\nAvailable Figures ({len(figure_inventory)} on disk):")
        for fig in figure_inventory:
            parts.append(f"  - {fig['name']}")

    return "\n".join(parts), figure_inventory


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
            bio_model = (state.custom_models or {}).get("bioinformatics") or getattr(config.models, "bioinformatics_model", "") or "google/gemini-3.1-pro-preview"
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

            # Wrap code: neutralize plt.show/plt.close, capture figures as PNG via IPython.display
            # Also save high-res PDF for download
            escaped_dir = str(fig_dir).replace("\\", "\\\\")
            wrapped = (
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
                f"    _f.savefig(r\"{escaped_dir}/chat_{ts}_\" + str(_fn) + \".pdf\", "
                "format='pdf', dpi=300, bbox_inches='tight', facecolor='white')\n"
                "    # Capture PNG for display\n"
                "    _buf = _io.BytesIO()\n"
                "    _f.savefig(_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')\n"
                "    _buf.seek(0)\n"
                "    _display(_Image(data=_buf.getvalue()))\n"
                "plt.close('all')\n"
            )

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
        chat_sandboxes.pop(session_id, None)
        try:
            await sandbox.stop()
        except Exception:
            logger.warning("Failed to stop chat code sandbox during cleanup", exc_info=True)

    return generated_figures


async def _send_chat_ws(session_id: str, msg_type: str, content: Any):
    """Send a WebSocket message for chat delegation events."""
    ws = websockets.get(session_id)
    if ws:
        try:
            await ws.send_json({
                "type": msg_type,
                "agent": None,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
            })
        except Exception as e:
            logger.warning(f"Chat WS send failed for {session_id} ({msg_type}): {e}")
    else:
        logger.warning(f"Chat WS: no connection for {session_id}, message {msg_type} not delivered")


async def _create_chat_snapshot(session_id: str, state: AnalysisState):
    """Create an automatic snapshot after chat exchange completes."""
    try:
        await SnapshotManager.create_and_register_snapshot(
            session_id, state, "post_chat", notify_fn=_send_chat_ws,
        )
    except Exception as e:
        logger.warning(f"[{session_id}] Chat auto-snapshot failed: {e}")


def _try_parse_delegation(content: str) -> Optional[Dict[str, Any]]:
    """Try to parse PI response as a delegation plan. Returns None if not delegation."""
    content = content.strip()
    # Direct JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("delegation") is True:
            delegates = parsed.get("delegates", [])
            if isinstance(delegates, list) and len(delegates) > 0:
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # JSON inside markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and parsed.get("delegation") is True:
                delegates = parsed.get("delegates", [])
                if isinstance(delegates, list) and len(delegates) > 0:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _build_synthesis_prompt(
    user_message: str,
    delegation: Dict[str, Any],
    agent_results: Dict[str, Dict[str, Any]],
) -> str:
    """Build PI synthesis prompt from delegation results."""
    parts = [
        "You are the PI Agent. Synthesize the following agent results into a clear, "
        "helpful response to the user's original request.",
        f"\n## User's Request\n{user_message}",
        f"\n## Delegation Plan\n{delegation.get('plan', '')}",
        "\n## Agent Results",
    ]
    for agent_name, result in agent_results.items():
        if "error" in result and not result.get("findings"):
            parts.append(f"\n### {agent_name.upper()} Agent: ERROR\n{result['error']}")
        else:
            # Extract the most useful content from agent result
            summary = result.get("synthesis", result.get("report", result.get("summary", "")))
            if isinstance(summary, dict):
                summary = json.dumps(summary, ensure_ascii=False, indent=2)
            findings = result.get("findings", [])
            if isinstance(findings, list) and findings:
                findings_text = "\n".join(f"- {f}" for f in findings[:20])
            else:
                findings_text = str(findings) if findings else ""
            agent_text = ""
            if summary:
                agent_text += f"Summary:\n{str(summary)[:6000]}\n"
            if findings_text:
                agent_text += f"Key Findings:\n{findings_text}\n"
            if not agent_text:
                agent_text = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            parts.append(f"\n### {agent_name.upper()} Agent\n{agent_text}")

    parts.append(
        "\n## Instructions\n"
        "- Synthesize these results into a comprehensive response\n"
        "- Reference specific findings from each agent\n"
        "- If figures were generated during the analysis, you can reference them with [[FIGURE:figure_name]]\n"
        "- Always respond in English\n"
    )
    if delegation.get("synthesis_instruction"):
        parts.append(f"Additional guidance: {delegation['synthesis_instruction']}")

    return "\n".join(parts)


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


async def _execute_chat_agent(
    session_id: str,
    state: Any,
    agent_name: str,
    instruction: str,
    llm_client: Any,
    message_callback,
    saved_files: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Any]]:
    """Execute a single agent in chat delegation context."""
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.utils.agent_llm import get_agent_llm_client

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)

    extra_clients: List[Any] = []

    # Provider for hot-reloadable settings; global_client is the per-agent client
    # already resolved by the caller via get_agent_llm_client.
    provider = AgentSettingsProvider(state, config, global_client=llm_client, session_id=session_id)

    async def _send_agent_msg(msg_type: str, content: Any, agent_type: AgentType):
        msg = WebSocketMessage(type=msg_type, agent=agent_type, content=content)
        await message_callback(msg)

    agent_type = AgentType(agent_name)
    await _send_agent_msg("status", f"Starting {agent_name} analysis...", agent_type)

    if agent_name == "bioinformatics":
        from backend.agents.bioinformatics_agent import BioinformaticsAgent

        # Create sandbox with session restore
        sandbox = LocalJupyterSandbox(
            session_id=session_id,
            state_dir=str(get_session_states(session_id)),
            working_dir=str(get_session_workspace(session_id)),
            timeout=config.sandbox_timeout if hasattr(config, 'sandbox_timeout') else 120
        )
        chat_sandboxes[session_id] = sandbox

        # Create literature agent for callbacks
        lit_config = state.literature_config or {}

        from backend.agents.literature_agent import LiteratureAgent
        literature_agent = LiteratureAgent(
            settings_provider=provider,
            literature_mode=lit_config.get("mode", "deep_research"),
            deep_research_model=lit_config.get("deep_research_model", ""),
            deep_research_llm_base_url=lit_config.get("deep_research_llm_base_url", ""),
            deep_research_api_key=lit_config.get("deep_research_api_key", ""),
            google_api_key=lit_config.get("google_api_key", ""),
            pubmed_email=lit_config.get("pubmed_email", ""),
            max_search_rounds=lit_config.get("max_search_rounds", 5),
            proxy_url=config.api.proxy_url if config.api else None,
        )

        async def lit_callback(topic: str) -> str:
            result = await literature_agent.process(state, topic, use_deep_research=True)
            return result.get("report", "")

        async def pi_callback(question: str, context: str) -> str:
            await _send_agent_msg("pi_question", f"Question: {question}\nContext: {context}", AgentType.BIOINFORMATICS)
            pi_model = provider.get_model(PI_AGENT_NAME)
            prompt = (
                f"The bioinformatics analyst is asking for your guidance:\n\n"
                f"QUESTION: {question}\n\nCONTEXT: {context}\n\n"
                f"Please provide clear, actionable guidance."
            )
            resp = await llm_client.chat(
                model=pi_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=provider.get_temperature(PI_AGENT_NAME),
                max_tokens=2048
            )
            answer = resp.content or "Proceed with standard bioinformatics practices."
            await _send_agent_msg("pi_response", answer, AgentType.PI)
            return answer

        async def annotate_callback(cluster_markers, tissue, organism):
            if not cluster_markers:
                return {
                    "status": "error",
                    "annotations": {},
                    "error": "cluster_markers is empty. Run rank_genes_groups and extract markers first."
                }
            await _send_agent_msg("status", f"Annotating {len(cluster_markers)} clusters via literature...", AgentType.LITERATURE)
            return await literature_agent.annotate_clusters(cluster_markers, tissue, organism, state)

        async def agent_msg_callback(msg_type: str, content: str):
            await _send_agent_msg(msg_type, content, AgentType.BIOINFORMATICS)

        bio_agent = BioinformaticsAgent(
            settings_provider=provider,
            sandbox=sandbox,
            literature_callback=lit_callback,
            pi_callback=pi_callback,
            annotate_clusters_callback=annotate_callback,
            message_callback=agent_msg_callback,
        )

        try:
            analysis_start_time = time.time()
            await sandbox.start()
            result = await bio_agent.run_autonomous_analysis(
                state, instruction, max_steps=30
            )
            # Collect only figures generated during THIS analysis (not historical ones)
            result["figures"] = _collect_session_figures(session_id, since_time=analysis_start_time)
            return result, extra_clients
        finally:
            chat_sandboxes.pop(session_id, None)
            try:
                await sandbox.stop()
            except Exception:
                logger.warning("Failed to stop delegation sandbox during cleanup", exc_info=True)

    elif agent_name == "pathology":
        from backend.agents.pathology_agent import PathologyAgent

        pathology_agent = PathologyAgent(
            settings_provider=provider,
            tile_size=config.image_tile_size if hasattr(config, 'image_tile_size') else 1024,
            max_tiles=50,
        )
        result = await pathology_agent.process(state, instruction)
        return result, extra_clients

    elif agent_name == "biochemistry":
        from backend.agents.biochemistry_agent import BiochemistryAgent

        biochem_agent = BiochemistryAgent(
            settings_provider=provider,
        )
        result = await biochem_agent.process(state, instruction)
        return result, extra_clients

    elif agent_name == "literature":
        from backend.agents.literature_agent import LiteratureAgent
        lit_config = state.literature_config or {}

        lit_agent = LiteratureAgent(
            settings_provider=provider,
            literature_mode=lit_config.get("mode", "deep_research"),
            deep_research_model=lit_config.get("deep_research_model", ""),
            deep_research_llm_base_url=lit_config.get("deep_research_llm_base_url", ""),
            deep_research_api_key=lit_config.get("deep_research_api_key", ""),
            google_api_key=lit_config.get("google_api_key", ""),
            pubmed_email=lit_config.get("pubmed_email", ""),
            max_search_rounds=lit_config.get("max_search_rounds", 5),
            proxy_url=config.api.proxy_url if config.api else None,
        )
        result = await lit_agent.process(state, instruction)
        return result, extra_clients

    else:
        raise ValueError(f"Unknown agent for chat delegation: {agent_name}")


async def _run_chat_delegation(
    session_id: str,
    state: Any,
    delegation: Dict[str, Any],
    user_message: str,
    image_b64_list: List[str],
    saved_files: List[Dict[str, str]],
):
    """Background task: orchestrate delegated agent work, then PI synthesizes results."""
    import traceback
    from backend.utils.llm_client import LLMManager

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)
    from backend.utils.agent_llm import get_agent_llm_client

    llm_manager = None
    start_time = time.time()
    _delegation_extra_clients: List[Any] = []
    try:
        # 1. Send started notification
        delegates = delegation.get("delegates", [])
        agent_names = [d.get("agent", "unknown") for d in delegates]
        await _send_chat_ws(session_id, "chat_delegation", {
            "status": "started",
            "plan": delegation.get("plan", ""),
            "agents": agent_names,
        })

        # 2. Create shared LLM client (default, used as fallback)
        llm_manager = LLMManager(
            api_key=config.api.api_key,
            base_url=config.api.llm_base_url,
            proxy_url=config.api.proxy_url,
            session_id=session_id,
            agent_name="chat_delegation",
        )
        # Per-agent clients for delegation

        # 3. Create message callback that sends to WebSocket + persists
        async def chat_msg_callback(msg: WebSocketMessage):
            ws = websockets.get(session_id)
            if ws:
                try:
                    await ws.send_json({
                        "type": msg.type,
                        "agent": msg.agent.value if msg.agent else None,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "id": msg.id,
                    })
                except Exception:
                    logger.warning(f"[{session_id}] Failed to send chat delegation message via WebSocket", exc_info=True)
            # Persist to realtime_messages
            if hasattr(state, 'realtime_messages'):
                state.realtime_messages.append(msg)

        # 4. Check for sandbox conflicts with running workflow
        workflow_running = session_id in workflows
        if workflow_running:
            # Filter out bioinformatics to avoid sandbox conflict
            delegates = [d for d in delegates if d.get("agent") != "bioinformatics"]
            if not delegates:
                await _send_chat_ws(session_id, "chat_error", {
                    "error": "Cannot run bioinformatics analysis while main workflow is active. "
                             "Please wait for the workflow to complete."
                })
                return

        # 5. Group by priority and execute
        priority_groups: Dict[int, List[Dict]] = {}
        for d in delegates:
            p = d.get("priority", 1)
            priority_groups.setdefault(p, []).append(d)

        agent_results: Dict[str, Dict[str, Any]] = {}
        all_figures: List[Dict[str, str]] = []

        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]

            if len(group) == 1:
                # Single agent — execute directly
                d = group[0]
                agent_name = d.get("agent", "unknown")
                try:
                    agent_client = get_agent_llm_client(state, config, agent_name, session_id)
                    if agent_client is not llm_manager.openrouter:
                        _delegation_extra_clients.append(agent_client)
                    result, extra = await _execute_chat_agent(
                        session_id, state, agent_name, d.get("instruction", ""),
                        agent_client, chat_msg_callback, saved_files
                    )
                    _delegation_extra_clients.extend(extra)
                    agent_results[agent_name] = result
                    if "figures" in result:
                        all_figures.extend(result["figures"])
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_complete", "agent": agent_name,
                    })
                except Exception as e:
                    logger.error(f"Chat delegation agent {agent_name} failed: {traceback.format_exc()}")
                    agent_results[agent_name] = {"error": str(e), "findings": []}
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_error", "agent": agent_name, "error": str(e),
                    })
            else:
                # Multiple agents at same priority — execute in parallel
                tasks = []
                for d in group:
                    a_name = d.get("agent", "unknown")
                    a_client = get_agent_llm_client(state, config, a_name, session_id)
                    if a_client is not llm_manager.openrouter:
                        _delegation_extra_clients.append(a_client)
                    tasks.append(_execute_chat_agent(
                        session_id, state, a_name, d.get("instruction", ""),
                        a_client, chat_msg_callback, saved_files
                    ))
                # Global timeout to prevent indefinite hang if an agent's LLM call stalls
                from backend.config import CHAT_DELEGATION_TIMEOUT
                try:
                    raw_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=CHAT_DELEGATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Chat delegation timed out after {CHAT_DELEGATION_TIMEOUT}s for session {session_id}")
                    # Cancel all tasks and wait for them to finish cleanup
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    for task in tasks:
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
                    raw_results = [TimeoutError(f"Delegation timed out after {CHAT_DELEGATION_TIMEOUT}s")] * len(tasks)
                for d, raw_result in zip(group, raw_results):
                    agent_name = d.get("agent", "unknown")
                    if isinstance(raw_result, Exception):
                        logger.error(f"Chat delegation agent {agent_name} failed: {raw_result}")
                        agent_results[agent_name] = {"error": str(raw_result), "findings": []}
                        await _send_chat_ws(session_id, "chat_delegation", {
                            "status": "agent_error", "agent": agent_name, "error": str(raw_result),
                        })
                    else:
                        result, extra = raw_result
                        _delegation_extra_clients.extend(extra)
                        agent_results[agent_name] = result
                        if "figures" in result:
                            all_figures.extend(result["figures"])
                        await _send_chat_ws(session_id, "chat_delegation", {
                            "status": "agent_complete", "agent": agent_name,
                        })

        # 6. PI Synthesis
        await _send_chat_ws(session_id, "chat_delegation", {
            "status": "synthesizing", "message": "PI synthesizing results...",
        })

        synthesis_prompt = _build_synthesis_prompt(user_message, delegation, agent_results)
        delegation_provider = AgentSettingsProvider(state, config, global_client=llm_manager.openrouter, session_id=session_id)
        pi_model = delegation_provider.get_model(PI_AGENT_NAME)
        pi_temp = delegation_provider.get_temperature(PI_AGENT_NAME)

        pi_response = await llm_manager.openrouter.chat(
            model=pi_model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=pi_temp,
            max_tokens=32768,
        )
        final_content = pi_response.content or "Analysis completed but synthesis failed."

        # 7. Post-process: FIGURE and GENERATE_FIGURE markers (reuse existing patterns)
        context, figure_inventory = _build_chat_context(state)
        figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
        referenced_names = figure_pattern.findall(final_content)
        synthesis_figures: List[Dict[str, str]] = []

        if referenced_names and figure_inventory:
            fig_lookup = {f["name"]: f for f in figure_inventory}
            for name in dict.fromkeys(referenced_names):
                fig = fig_lookup.get(name)
                if fig:
                    entry = {"name": fig["name"], "path": fig["path"]}
                    pdf_path = Path(fig["path"]).with_suffix(".pdf")
                    if pdf_path.exists():
                        entry["pdfPath"] = str(pdf_path)
                    synthesis_figures.append(entry)

        generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
        generate_requests = generate_pattern.findall(final_content)
        if generate_requests:
            try:
                generated = await _generate_figures_in_chat(session_id, state, generate_requests)
                synthesis_figures.extend(generated)
            except Exception as e:
                logger.exception("Chat delegation figure generation failed")

        # Merge: all_figures from agents + synthesis_figures
        # Deduplicate by name
        seen_names = set()
        merged_figures = []
        for fig in all_figures + synthesis_figures:
            if fig["name"] not in seen_names:
                seen_names.add(fig["name"])
                merged_figures.append(fig)

        # Clean markers from text
        cleaned_content = figure_pattern.sub('', final_content)
        cleaned_content = generate_pattern.sub('', cleaned_content).strip()
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

        # 8. Build final chat message
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": cleaned_content,
            "timestamp": datetime.now().isoformat(),
        }
        if merged_figures:
            assistant_msg["figures"] = merged_figures

        # Replace placeholder message in chat_messages
        replaced = False
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = assistant_msg
                replaced = True
                break
        if not replaced:
            state.chat_messages.append(assistant_msg)

        # 9. Send chat_result via WebSocket
        await _send_chat_ws(session_id, "chat_result", {
            "content": cleaned_content,
            "figures": merged_figures,
            "timestamp": datetime.now().isoformat(),
        })

        # 10. Persist state
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)

        # Auto-snapshot after chat delegation completes
        await _create_chat_snapshot(session_id, state)

        duration = time.time() - start_time
        logger.info(f"Chat delegation completed for {session_id} in {duration:.1f}s, "
                     f"{len(agent_results)} agents, {len(merged_figures)} figures")

    except asyncio.CancelledError:
        logger.info(f"Chat delegation cancelled by user for {session_id}")
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = {
                    "role": "assistant",
                    "content": "Cancelled",
                    "timestamp": datetime.now().isoformat(),
                }
                break
        try:
            await _send_chat_ws(session_id, "chat_cancelled", {"message": "Cancelled by user"})
            await update_session_cache(session_id, state)
            await async_save_session(session_id, state)
        except Exception:
            logger.warning(f"[{session_id}] Failed to save session during cancellation", exc_info=True)
        raise

    except Exception as e:
        logger.error(f"Chat delegation error for {session_id}: {traceback.format_exc()}")
        error_msg = f"Analysis failed: {str(e)}"
        # Update placeholder message
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = {
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat(),
                }
                break
        await _send_chat_ws(session_id, "chat_error", {"error": error_msg})
        await update_session_cache(session_id, state)
        await async_save_session(session_id, state)

    finally:
        async with chat_task_lock:
            chat_tasks.pop(session_id, None)
        chat_sandboxes.pop(session_id, None)
        if llm_manager:
            try:
                await llm_manager.close()
            except Exception:
                logger.warning("Failed to close LLM manager", exc_info=True)
        for extra_client in _delegation_extra_clients:
            try:
                await extra_client.close()
            except Exception:
                logger.debug("Failed to close extra LLM client", exc_info=True)


@router.post("/{session_id}/chat")
async def chat_with_pi(
    request: Request,
    session_id: str,
    message: str = Form(...),
    target_agents: str = Form(default="[]"),
    file_groups: str = Form(default="[]"),
    file_samples: str = Form(default="[]"),
    file_types: str = Form(default="[]"),
    files: List[UploadFile] = File(default=[]),
):
    """Chat with PI Agent using analysis context. Supports file uploads with per-file agent targeting."""
    from backend.utils.agent_llm import get_agent_llm_client
    import base64

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    config = await resolve_config_for_user(user_id)

    if config is None:
        raise HTTPException(status_code=400, detail="No API key configured. Please configure your LLM API key in Settings before sending a message.")

    # Parse per-file metadata arrays
    def _parse_json_list(raw: str) -> list:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    per_file_agents = _parse_json_list(target_agents)
    per_file_groups = _parse_json_list(file_groups)
    per_file_samples = _parse_json_list(file_samples)
    per_file_types = _parse_json_list(file_types)

    # --- Save uploaded files ---
    saved_files: List[Dict[str, str]] = []
    image_b64_list: List[str] = []
    data_file_contents: List[str] = []
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}

    chat_upload_dir = get_session_uploads(session_id) / "chat"
    chat_upload_dir.mkdir(parents=True, exist_ok=True)

    for idx, upload_file in enumerate(files):
        fname = safe_filename(upload_file.filename or "file")
        if not fname:
            continue
        dest = chat_upload_dir / fname
        # Deduplicate filenames
        counter = 1
        while dest.exists():
            stem = Path(fname).stem
            suffix = Path(fname).suffix
            dest = chat_upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        await stream_save_file(upload_file, dest)
        ext = dest.suffix.lower()
        is_image = ext in IMAGE_EXTS
        file_type = "image" if is_image else "file"
        file_agent = per_file_agents[idx] if idx < len(per_file_agents) else PI_AGENT_NAME

        file_group = per_file_groups[idx] if idx < len(per_file_groups) else None
        file_sample = per_file_samples[idx] if idx < len(per_file_samples) else None
        file_dtype = per_file_types[idx] if idx < len(per_file_types) else None

        saved_files.append({
            "filename": dest.name,
            "type": file_dtype or file_type,
            "path": str(dest),
            "target_agent": file_agent,
            "group_id": file_group,
            "sample_number": file_sample,
        })

        # Load image as base64 for vision API
        if is_image:
            with open(dest, "rb") as f:
                image_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))

        # Read data file content (first 5000 chars) for text inclusion
        if not is_image and ext in {'.csv', '.tsv', '.txt', '.json'}:
            try:
                with open(dest, "r", encoding="utf-8", errors="replace") as f:
                    content_preview = f.read(5000)
                data_file_contents.append(f"--- {dest.name} ---\n{content_preview}")
            except Exception as e:
                logger.warning(f"Failed to read file preview for {dest}: {e}", exc_info=True)

    # --- Register files to target agent state (per-file) ---
    _agent_section_map = {"pathology": "pathology", "bioinformatics": "sequencing", "biochemistry": "biochemistry"}
    for sf in saved_files:
        agent = sf["target_agent"]
        section = _agent_section_map.get(agent)
        if section:
            state.data_files.append(DataFile(
                filename=sf["filename"],
                file_path=sf["path"],
                section=section,
                data_type=sf.get("type") or "uploaded_via_chat",
                group_id=sf.get("group_id"),
                sample_number=sf.get("sample_number"),
                description="Uploaded via chat",
            ))

    # --- Build context & prompt ---
    context, figure_inventory = _build_chat_context(state)
    _custom_models = state.custom_models or {}
    _custom_temps = state.custom_temperatures or {}
    pi_model = _custom_models.get(PI_AGENT_NAME) or getattr(config.models, f"{PI_AGENT_NAME}_model", "")
    pi_temp = _custom_temps.get(PI_AGENT_NAME) if _custom_temps.get(PI_AGENT_NAME) is not None else getattr(config.temperatures, PI_AGENT_NAME, 0.5)

    system_prompt = (
        "You are the PI (Principal Investigator) Agent of a biomedical research analysis system. "
        "You have access to the current analysis state and can help the user understand results, "
        "answer questions about the data, and provide scientific insights.\n\n"
        "Current Analysis Context:\n" + context + "\n\n"
        "Instructions:\n"
        "- Answer the user's questions based on available analysis results\n"
        "- If relevant data exists in the analysis, reference it specifically\n"
        "- Be concise but thorough in scientific explanations\n"
        "- If you don't have enough information, say so clearly\n"
        "- You can suggest next steps or additional analyses if appropriate\n"
        "\n"
        "FIGURE DISPLAY RULES (CRITICAL — you MUST follow these):\n"
        "You have a built-in figure rendering system. You MUST use it. "
        "NEVER provide Python/R code for the user to run. NEVER tell the user to 'run this code'. "
        "The user CANNOT execute code — only you can display figures through these commands:\n\n"
        "1. To show an EXISTING figure from the analysis, write on its own line:\n"
        "   [[FIGURE:figure_name]]\n"
        "   Use exact names from the 'Available Figures' list in the context above.\n"
        "   Example: [[FIGURE:umap_cell_types]]\n\n"
        "2. To CREATE a NEW figure that doesn't exist, write on its own line:\n"
        "   [[GENERATE_FIGURE:detailed description]]\n"
        "   The system will automatically generate and display the figure.\n"
        "   Include specific details: what plot type, which genes/clusters, colors, labels, etc.\n"
        "   Example: [[GENERATE_FIGURE:UMAP plot colored by cell_type annotation with legend on the right margin, "
        "using tab20 palette, title 'Cell Type Clusters', frameon=False]]\n\n"
        "When the user asks to see, show, display, generate, or create any figure/plot/chart/image, "
        "you MUST use [[FIGURE:...]] or [[GENERATE_FIGURE:...]] — NEVER give code.\n"
        "\n"
        "- Always respond in English\n"
        "\n"
        "DELEGATION CAPABILITY:\n"
        "When the user's request requires running new computational analysis, processing new data, "
        "or using specialized agent capabilities beyond your own knowledge, you can delegate tasks "
        "to other agents. Available agents:\n"
        "- bioinformatics: Code execution, new plots, scRNA-seq analysis, clustering, DE analysis, trajectory\n"
        "- pathology: Pathology/histology image analysis or re-examination\n"
        "- biochemistry: Biochemistry data interpretation (Western blot, ELISA, etc.)\n"
        "- literature: Literature search, evidence gathering, Deep Research\n\n"
        "When delegation is needed, respond with ONLY a JSON block (no other text):\n"
        '{"delegation": true, "plan": "Brief description of what you will coordinate", "delegates": [\n'
        '  {"agent": "bioinformatics", "instruction": "Detailed task instruction with data paths and specifics", "priority": 1}\n'
        '], "synthesis_instruction": "How to summarize results for user"}\n\n'
        "Delegation rules:\n"
        "- Same priority number = parallel execution; lower number executes first\n"
        "- For simple questions answerable from existing results, respond normally (NO delegation)\n"
        "- For showing existing figures, use [[FIGURE:...]] (NO delegation)\n"
        "- Only delegate when NEW analysis, computation, or specialized processing is genuinely needed\n"
        "- Include specific data file paths and analysis requirements in the instruction field\n"
        "- The bioinformatics agent's notebook ALREADY has all previous analysis results loaded (adata with "
        "cell_type, clustering, PCA, UMAP, DE results, etc.). Do NOT include prerequisite steps like "
        "'first ensure cell annotations are available' or 'run clustering first' in the instruction. "
        "Focus ONLY on the NEW analysis the user is asking for. The agent will check existing results automatically.\n"
    )

    # Build messages: system + history + new message
    chat_messages_list = [{"role": "system", "content": system_prompt}]
    for msg in state.chat_messages[-20:]:
        chat_messages_list.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Build user content with file info
    user_content = message
    if saved_files:
        # Group files by target agent for readable description
        agent_files: Dict[str, List[str]] = {}
        for sf in saved_files:
            agent_files.setdefault(sf["target_agent"], []).append(sf["filename"])
        file_parts = []
        for agent, fnames in agent_files.items():
            label = agent.upper() if agent != PI_AGENT_NAME else "PI"
            file_parts.append(f"{label} Agent: {', '.join(fnames)}")
        user_content += f"\n\n[Attached {len(saved_files)} file(s) — {'; '.join(file_parts)}]"
    if data_file_contents:
        user_content += "\n\n" + "\n\n".join(data_file_contents)

    chat_messages_list.append({"role": "user", "content": user_content})

    # Store user message
    state.chat_messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat(),
        "files": saved_files,
    })

    # --- Call LLM ---
    try:
        llm_client = get_agent_llm_client(state, config, PI_AGENT_NAME, session_id)

        if image_b64_list:
            # Vision API for images
            full_prompt = system_prompt + "\n\nChat History:\n"
            for msg in state.chat_messages[-21:-1]:  # exclude the just-added user msg
                full_prompt += f"\n{msg.get('role', 'user')}: {msg.get('content', '')}"
            full_prompt += f"\n\nuser: {user_content}"

            response = await llm_client.chat_with_vision(
                model=pi_model,
                text_prompt=full_prompt,
                images=image_b64_list,
                temperature=pi_temp,
                max_tokens=32768
            )
        else:
            response = await llm_client.chat(
                model=pi_model,
                messages=chat_messages_list,
                temperature=pi_temp,
                max_tokens=32768
            )

        assistant_content = response.content if response.content else "I'm sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        logger.exception("Chat LLM error")
        assistant_content = f"Error communicating with LLM: {str(e)}"

    # --- Check for delegation ---
    delegation = _try_parse_delegation(assistant_content)
    if delegation:
        logger.info(f"Chat delegation detected for {session_id}: {delegation.get('plan', '')}")

        # Concurrency guard
        async with chat_task_lock:
            if session_id in chat_tasks and not chat_tasks[session_id].done():
                raise HTTPException(status_code=409, detail="Chat analysis already in progress for this session")

        # Add placeholder assistant message
        state.chat_messages.append({
            "role": "assistant",
            "content": f"Working on it: {delegation.get('plan', 'Coordinating agents...')}",
            "timestamp": datetime.now().isoformat(),
            "delegation_active": True,
        })
        await async_save_session(session_id, state)

        # Launch background task
        task = asyncio.create_task(
            _run_chat_delegation(
                session_id, state, delegation, user_content,
                image_b64_list, saved_files
            )
        )
        async with chat_task_lock:
            chat_tasks[session_id] = task

        return {"status": "processing", "plan": delegation.get("plan", "")}

    # --- Post-process: extract figure references ---
    figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
    referenced_names = figure_pattern.findall(assistant_content)
    logger.info(f"Chat post-process: found {len(referenced_names)} FIGURE refs, {len(figure_inventory)} in inventory")

    response_figures: List[Dict[str, str]] = []
    if referenced_names and figure_inventory:
        fig_lookup = {f["name"]: f for f in figure_inventory}
        for name in dict.fromkeys(referenced_names):  # deduplicate, preserve order
            fig = fig_lookup.get(name)
            if fig:
                entry = {"name": fig["name"], "path": fig["path"]}
                # Check for corresponding PDF
                pdf_path = Path(fig["path"]).with_suffix(".pdf")
                if pdf_path.exists():
                    entry["pdfPath"] = str(pdf_path)
                response_figures.append(entry)
        logger.info(f"Chat post-process: matched {len(response_figures)} existing figures")

    # --- Post-process: extract GENERATE_FIGURE requests ---
    generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
    generate_requests = generate_pattern.findall(assistant_content)
    logger.info(f"Chat post-process: found {len(generate_requests)} GENERATE_FIGURE requests")

    # Clean markers from text
    cleaned_content = figure_pattern.sub('', assistant_content)
    cleaned_content = generate_pattern.sub('', cleaned_content).strip()
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

    # Save text response IMMEDIATELY (before figure generation which can take minutes)
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
    }
    if response_figures:
        assistant_msg["figures"] = list(response_figures)
    state.chat_messages.append(assistant_msg)
    await async_save_session(session_id, state)
    logger.info(f"Chat response: text saved ({len(cleaned_content)} chars, {len(response_figures)} existing figures)")

    # Now try to generate new figures (may take long, but text is already saved)
    if generate_requests:
        try:
            generated = await _generate_figures_in_chat(session_id, state, generate_requests)
            logger.info(f"Chat post-process: generated {len(generated)} figures")
            if generated:
                existing = assistant_msg.get("figures", [])
                assistant_msg["figures"] = existing + generated
                response_figures.extend(generated)
                await async_save_session(session_id, state)
        except Exception as e:
            logger.exception("Chat figure generation failed")

    result: Dict[str, Any] = {"role": "assistant", "content": cleaned_content}
    if assistant_msg.get("figures"):
        result["figures"] = assistant_msg["figures"]

    # Auto-snapshot after chat exchange
    await _create_chat_snapshot(session_id, state)
    await async_save_session(session_id, state)

    return result


@router.post("/{session_id}/chat/cancel")
async def cancel_chat(request: Request, session_id: str):
    """Cancel an in-progress chat request (delegation or simple)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    cancelled = False

    # 1. Cancel delegation task if running
    async with chat_task_lock:
        task = chat_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            cancelled = True

    # 2. Interrupt sandbox kernel if running
    sandbox = chat_sandboxes.get(session_id)
    if sandbox:
        try:
            if sandbox.km:
                sandbox.km.interrupt_kernel()
                logger.info(f"Chat cancel: interrupted sandbox kernel for {session_id}")
        except Exception as e:
            logger.warning(f"Chat cancel: sandbox interrupt failed for {session_id}: {e}")

    return {"status": "cancelled", "had_delegation": cancelled}


@router.get("/{session_id}/chat")
async def get_chat_history(request: Request, session_id: str):
    """Get chat history for a session"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {"messages": state.chat_messages}
