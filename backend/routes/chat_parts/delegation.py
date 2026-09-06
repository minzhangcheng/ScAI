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
from backend.services.session_store import (
    _async_save_session_under_lifecycle,
    ensure_session_loaded,
)
from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.state import (
    chat_sandboxes,
    sessions,
    workflows,
    state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL
from backend.utils.websocket_delivery import (
    append_durable_realtime_message,
    send_analysis_json,
)

from .core import _build_chat_context

from .figures import (
    _collect_session_figures,
    _generate_figures_in_chat,
    _release_chat_runtime_deferred,
    _reserve_chat_runtime,
)
from .snapshots import _create_chat_snapshot


import sys
from . import delegation_flow


logger = logging.getLogger(__name__)


async def _persist_chat_terminal(session_id: str, state: AnalysisState) -> bool:
    """Force-save a terminal chat replacement only for the live generation."""
    try:
        async with get_lifecycle_lock(session_id):
            async with state_lock:
                if sessions.get(session_id) is not state:
                    return False
                assert_generation_writable(session_id, state)
            await _async_save_session_under_lifecycle(
                session_id, state, force=True,
            )
        return True
    except Exception:
        logger.warning(
            "Chat terminal persistence rejected for %s",
            session_id,
            exc_info=True,
        )
        return False


def _replace_latest_delegation_message(
    state: AnalysisState,
    replacement: Dict[str, Any],
) -> bool:
    """Replace the active placeholder without changing its message identity."""
    for index in range(len(state.chat_messages) - 1, -1, -1):
        current = state.chat_messages[index]
        if not current.get("delegation_active"):
            continue
        replacement.setdefault("id", current.get("id") or str(uuid.uuid4()))
        if current.get("reply_to"):
            replacement.setdefault("reply_to", current["reply_to"])
        state.chat_messages[index] = replacement
        return True
    return False


def _active_delegation_identity(state: AnalysisState) -> Dict[str, str]:
    """Return the assistant identity owned by this delegation's latest message."""
    for current in reversed(state.chat_messages):
        if current.get("delegation_active"):
            return {
                key: current[key]
                for key in ("id", "reply_to")
                if current.get(key)
            }
    for current in reversed(state.chat_messages):
        if current.get("role") == "assistant" and current.get("reply_to"):
            return {
                key: current[key]
                for key in ("id", "reply_to")
                if current.get(key)
            }
    return {}


def _replace_latest_chat_terminal(
    state: AnalysisState,
    replacement: Dict[str, Any],
) -> bool:
    """Replace an active placeholder or its already-published terminal message."""
    if _replace_latest_delegation_message(state, replacement):
        return True
    for index in range(len(state.chat_messages) - 1, -1, -1):
        current = state.chat_messages[index]
        if current.get("role") != "assistant" or not current.get("reply_to"):
            continue
        replacement.setdefault("id", current.get("id") or str(uuid.uuid4()))
        replacement.setdefault("reply_to", current["reply_to"])
        state.chat_messages[index] = replacement
        return True
    return False


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
    """Reserve the shared runtime; the inner path enforces is_forward_resume_blocked."""
    reservation = None
    if agent_name == "bioinformatics":
        reservation = await _reserve_chat_runtime(
            session_id,
            allow_existing_chat_owner=True,
        )
        if reservation is None:
            raise HTTPException(
                status_code=409,
                detail="Another owner already holds this session runtime",
            )
    try:
        return await _execute_chat_agent_inner(
            session_id,
            state,
            agent_name,
            instruction,
            llm_client,
            message_callback,
            saved_files,
        )
    finally:
        if reservation is not None:
            await _release_chat_runtime_deferred(session_id, reservation)


async def _execute_chat_agent_inner(
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
    extra_clients: List[Any] = []
    config = await resolve_config_for_user(user_id)

    # Provider for hot-reloadable settings; global_client is the per-agent
    # client already resolved by the caller via get_agent_llm_client.
    provider = AgentSettingsProvider(
        state, config, global_client=llm_client, session_id=session_id
    )

    async def _send_agent_msg(msg_type: str, content: Any, agent_type: AgentType):
        msg = WebSocketMessage(type=msg_type, agent=agent_type, content=content)
        await message_callback(msg)

    agent_type = AgentType(agent_name)
    await _send_agent_msg("status", f"Starting {agent_name} analysis...", agent_type)

    if agent_name == "bioinformatics":
        from backend.agents.bioinformatics_agent import BioinformaticsAgent
        from backend.utils.runtime_apply import (
            is_forward_resume_blocked,
            state_runtime_coherence,
        )

        if is_forward_resume_blocked(
            getattr(state, "runtime_restore_status", "complete"),
            coherence=state_runtime_coherence(state),
        ):
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
            dispatch = await provider.budget_text_dispatch(
                PI_AGENT_NAME,
                [{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            resp = await llm_client.chat(
                model=pi_model,
                messages=dispatch.messages,
                temperature=provider.get_temperature(PI_AGENT_NAME),
                max_tokens=dispatch.max_tokens,
                prefer_stream=True,
            )
            await provider.acknowledge_text_dispatch(PI_AGENT_NAME, dispatch)
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

        chat_sandboxes[session_id] = sandbox
        started = False
        try:
            analysis_start_time = time.time()
            await sandbox.start()
            started = True
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
                    save=started and not is_session_tombstoned(session_id)
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

async def _run_chat_delegation(session_id: str, state: Any, delegation: Dict[str, Any], user_message: str, image_b64_list: List[str], saved_files: List[Dict[str, Any]]):
    delegation_flow.bind_facade(sys.modules[__name__])
    return await delegation_flow._run_chat_delegation(session_id, state, delegation, user_message, image_b64_list, saved_files)
