# Delegation parsing, synthesis prompt, and the agent execution loop.
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
    chat_sandboxes,
    workflows, state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL
from backend.utils.websocket_delivery import send_analysis_json

from .core import _build_chat_context

from .figures import _collect_session_figures, _generate_figures_in_chat
from .snapshots import _create_chat_snapshot


logger = logging.getLogger(__name__)


async def _send_chat_ws(session_id: str, msg_type: str, content: Any):
    """Send a WebSocket message for chat delegation events."""
    delivered = await send_analysis_json(session_id, {
                "type": msg_type,
                "agent": None,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
    })
    if not delivered:
        logger.warning(f"Chat WS: no connection for {session_id}, message {msg_type} not delivered")

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

async def _execute_chat_agent(
    session_id: str,
    state: Any,
    agent_name: str,
    instruction: str,
    llm_client: Any,
    message_callback,
    saved_files: List[Dict[str, Any]],
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
        from backend.utils.runtime_apply import is_forward_resume_blocked

        if is_forward_resume_blocked(getattr(state, "runtime_restore_status", "complete")):
            # The workflow refuses to run forward on a kernel that does not match
            # the state (lifecycle resume and workflow.run/resume all check this).
            # Chat delegation used to bypass that entirely: it would start a fresh
            # sandbox with nothing in it while the delegation prompt asserts every
            # earlier result is still loaded, so the agent burns steps discovering
            # that `adata` is undefined.
            warning = (
                "This session's kernel runtime is unavailable (it was restored or "
                "imported without in-memory state), so bioinformatics analysis "
                "cannot run from chat -- there are no variables from the earlier "
                "run to build on. Use Re-run to recompute the analysis from the "
                "session's data files first."
            )
            await _send_agent_msg("error", warning, agent_type)
            return (
                {
                    "summary": warning,
                    "findings": [],
                    "figures": [],
                    "error": "runtime_unavailable",
                },
                extra_clients,
            )

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
            literature_mode=lit_config.get("mode", "pubmed_llm"),
            deep_research_model=lit_config.get("deep_research_model", "o4-mini-deep-research"),
            deep_research_llm_base_url=lit_config.get("deep_research_llm_base_url", ""),
            deep_research_api_key=lit_config.get("deep_research_api_key", ""),
            google_api_key=lit_config.get("google_api_key", ""),
            pubmed_email=lit_config.get("pubmed_email", ""),
            max_search_rounds=lit_config.get("max_search_rounds", 20),
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

        async def annotate_callback(cluster_markers, tissue, organism,
                                    analysis_log=None, database_priors=None):
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
            try:
                from backend.services.session_lifecycle import (
                    is_session_tombstoned,
                )

                await sandbox.stop(
                    save=not is_session_tombstoned(session_id)
                )
            except Exception:
                logger.warning("Failed to stop delegation sandbox during cleanup", exc_info=True)
            finally:
                if getattr(sandbox, "km", None) is None:
                    if chat_sandboxes.get(session_id) is sandbox:
                        chat_sandboxes.pop(session_id, None)

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
            literature_mode=lit_config.get("mode", "pubmed_llm"),
            deep_research_model=lit_config.get("deep_research_model", "o4-mini-deep-research"),
            deep_research_llm_base_url=lit_config.get("deep_research_llm_base_url", ""),
            deep_research_api_key=lit_config.get("deep_research_api_key", ""),
            google_api_key=lit_config.get("google_api_key", ""),
            pubmed_email=lit_config.get("pubmed_email", ""),
            max_search_rounds=lit_config.get("max_search_rounds", 20),
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
    saved_files: List[Dict[str, Any]],
):
    """Background task: orchestrate delegated agent work, then PI synthesizes results."""
    import traceback
    from backend.utils.llm_client import LLMManager
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)
    from backend.utils.agent_llm import get_agent_llm_client

    llm_manager = None
    start_time = time.time()
    _delegation_extra_clients: List[Any] = []
    pre_snapshot_status = None
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
            await send_analysis_json(session_id, {
                        "type": msg.type,
                        "agent": msg.agent.value if msg.agent else None,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "id": msg.id,
            })
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
        from backend.models.schemas import TaskStatus as _TaskStatus

        pre_snapshot_status = getattr(state, "status", _TaskStatus.COMPLETED)
        if not isinstance(pre_snapshot_status, _TaskStatus):
            try:
                pre_snapshot_status = _TaskStatus(str(pre_snapshot_status))
            except Exception:
                pre_snapshot_status = _TaskStatus.COMPLETED
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

    except RequiredSnapshotBlocked as e:
        # Coherent HITL: structured snapshot_failure, not generic chat_error.
        from backend.models.schemas import TaskStatus
        from backend.services.snapshot_v2.failure_gate import (
            enter_chat_snapshot_failure_gate,
        )

        logger.warning(
            f"[{session_id}] Post-chat snapshot blocked after delegation: {e}"
        )
        previous_status = pre_snapshot_status
        if previous_status is None:
            previous_status = getattr(state, "status", TaskStatus.COMPLETED)
        if not isinstance(previous_status, TaskStatus):
            try:
                previous_status = TaskStatus(str(previous_status))
            except Exception:
                previous_status = TaskStatus.COMPLETED
        # Always set chat retry_context (service may have written a bare failure).
        enter_chat_snapshot_failure_gate(state, e, previous_status)
        for i in range(len(state.chat_messages) - 1, -1, -1):
            if state.chat_messages[i].get("delegation_active"):
                state.chat_messages[i] = {
                    "role": "assistant",
                    "content": (
                        "Chat completed, but a required snapshot failed. "
                        "Use Retry, Manage, or Terminate."
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "snapshot_failure": state.snapshot_failure,
                }
                break
        await _send_chat_ws(session_id, "state_sync", {
            "change_type": "snapshot_failure",
            "status": getattr(state.status, "value", state.status),
            "wait_input_reason": "snapshot_failure",
            "snapshot_failure": state.snapshot_failure,
        })
        try:
            await update_session_cache(session_id, state)
            await async_save_session(session_id, state)
        except Exception:
            logger.warning(
                f"[{session_id}] Failed to persist snapshot_failure after delegation",
                exc_info=True,
            )

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
        active_sandbox = chat_sandboxes.get(session_id)
        if active_sandbox is not None and getattr(active_sandbox, "km", None) is None:
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
