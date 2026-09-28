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
from backend.utils.formatting import loads_llm_json
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
        for key in ("followup_id", "branch_id"):
            if current.get(key):
                replacement.setdefault(key, current[key])
        state.chat_messages[index] = replacement
        return True
    return False


def _active_delegation_identity(state: AnalysisState) -> Dict[str, str]:
    """Return the assistant identity owned by this delegation's latest message."""
    for current in reversed(state.chat_messages):
        if current.get("delegation_active"):
            return {
                key: current[key]
                for key in ("id", "reply_to", "followup_id", "branch_id")
                if current.get(key)
            }
    for current in reversed(state.chat_messages):
        if current.get("role") == "assistant" and current.get("reply_to"):
            return {
                key: current[key]
                for key in ("id", "reply_to", "followup_id", "branch_id")
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
        for key in ("followup_id", "branch_id"):
            if current.get(key):
                replacement.setdefault(key, current[key])
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
        parsed = loads_llm_json(content)
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
            parsed = loads_llm_json(match.group(1))
            if isinstance(parsed, dict) and parsed.get("delegation") is True:
                delegates = parsed.get("delegates", [])
                if isinstance(delegates, list) and len(delegates) > 0:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None

def _synthesis_evidence_text(value: Any, *, limit: int) -> str:
    """Bound prompt evidence without silently presenting excerpts as complete."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text
    return text[:limit] + (
        f"\n[Evidence truncated: {len(text) - limit} characters omitted. "
        "Missing details cannot be verified from this excerpt.]"
    )


def _build_execution_context(result: Dict[str, Any]) -> str:
    """Expose executor records that are not part of the top-level findings."""
    summaries = []
    errors = []
    figure_attempts = []
    for step in result.get("analysis_log") or []:
        if not isinstance(step, dict):
            step = step.model_dump(mode="json")
        if step.get("step_type") == "tool_error" or step.get("code_error"):
            errors.append({
                key: step.get(key)
                for key in (
                    "step_number", "step_type", "execution_id", "text_content",
                    "code_error",
                )
            })
        if step.get("findings_summary"):
            summaries.append({
                key: step.get(key)
                for key in (
                    "step_number", "findings_summary", "evidence_execution_ids",
                )
            })
        if step.get("pub_figure_name") or step.get("pub_figure_result"):
            figure_attempts.append({key: step.get(key) for key in (
                "id", "step_number", "pub_figure_name", "pub_figure_code",
                "pub_figure_result",
            )})

    executions = []
    for execution in result.get("code_history") or []:
        if not isinstance(execution, dict):
            execution = execution.model_dump(mode="json", exclude={"figures"})
        # Keep failed/partial attempts visible so the PI can distinguish them
        # from the computation supporting a result. Do not copy image payloads.
        executions.append({
            key: execution.get(key)
            for key in (
                "id", "round_number", "plan_id", "execution_status",
                "code", "output", "error",
                "figure_method_annotations",
            )
        })

    score_definitions = []
    for score in result.get("score_definitions") or []:
        if not isinstance(score, dict):
            score = score.model_dump(mode="json")
        score_definitions.append(score)

    scoring_plans = []
    for plan in result.get("scoring_plans") or []:
        if not isinstance(plan, dict):
            plan = plan.model_dump(mode="json")
        scoring_plans.append(plan)

    parts = []
    outcome = {
        key: result[key]
        for key in (
            "status", "success", "steps_executed", "error", "forced_stop",
            "plan_revision_needed", "attempt_execution_ids",
        )
        if key in result
    }
    if outcome:
        parts.append(
            "Delegate execution outcome (reported by the executor):\n"
            + json.dumps(outcome, ensure_ascii=False, indent=2)
        )
    for key, title in (
        ("figure_review_records", "Figure attempts, including rejected/unreviewed renders"),
        ("chat_review", "PI Chat review and correction history (not a technical execution status)"),
    ):
        if key in result:
            parts.append(title + ":\n" + json.dumps(result[key], ensure_ascii=False, indent=2))
    if errors:
        parts.append(
            "Executor tool/code errors (attempt records; later recovery is possible):\n"
            + _synthesis_evidence_text(errors, limit=16000)
        )
    if figure_attempts:
        parts.append(
            "Publication figure tool attempts (including failed renders):\n"
            + json.dumps(figure_attempts, ensure_ascii=False, indent=2)
        )
    if "figures" in result:
        figures = [
            {key: figure[key] for key in ("name", "path", "pdfPath") if key in figure}
            for figure in (result["figures"] or [])
            if isinstance(figure, dict)
        ]
        parts.append(
            "Figure artifacts returned for this delegation:\n"
            + json.dumps(figures, ensure_ascii=False, indent=2)
        )
    if summaries:
        parts.append(
            "Executor report summaries (agent-authored):\n"
            + _synthesis_evidence_text(summaries, limit=16000)
        )
    if executions:
        parts.append(
            "Execution records returned by the delegate (including any restored "
            "history; match execution IDs to the reported results):\n"
            + json.dumps(executions, ensure_ascii=False, indent=2)
        )
    if score_definitions:
        parts.append(
            "Registered score definitions (session-wide structured method records; "
            "may predate this delegation):\n"
            + _synthesis_evidence_text(score_definitions, limit=16000)
        )
    if scoring_plans:
        parts.append(
            "Frozen scoring plans (session-wide structured pre-specified method records; "
            "may predate this delegation):\n"
            + _synthesis_evidence_text(scoring_plans, limit=16000)
        )
    if (
        result.get("analysis_log")
        or result.get("code_history")
        or result.get("score_definitions")
        or result.get("scoring_plans")
    ):
        parts.append(
            "Record availability: "
            f"{len(summaries)} report summaries; {len(executions)} execution records; "
            f"{len(score_definitions)} score definitions; {len(scoring_plans)} scoring plans. "
            "Absent records do not establish how a computation was performed."
        )
    return "\n".join(parts)


def _build_synthesis_prompt(
    user_message: str,
    delegation: Dict[str, Any],
    agent_results: Dict[str, Dict[str, Any]],
    source_context: str = "",
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
            execution_context = _build_execution_context(result)
            if execution_context:
                parts.append(execution_context)
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
                agent_text += (
                    "Summary:\n"
                    + _synthesis_evidence_text(str(summary), limit=6000)
                    + "\n"
                )
            if findings_text:
                agent_text += f"Key Findings:\n{findings_text}\n"
            execution_context = _build_execution_context(result)
            if execution_context:
                agent_text += f"\n{execution_context}\n"
            if not agent_text:
                agent_text = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            parts.append(f"\n### {agent_name.upper()} Agent\n{agent_text}")

    parts.append(
        "\n## Instructions\n"
        "- Synthesize these results into a comprehensive response\n"
        "- Honor the recorded PI Chat review, including unresolved issues, corrections, "
        "and unexecuted revision requests. Explicitly disclose unavailable review or "
        "exhausted correction budget. Do not present superseded findings as current "
        "or unreviewed figures as visually approved.\n"
        "- For reviewed Chat analyses, reference only figures returned by that "
        "delegation. Do not emit GENERATE_FIGURE requests after the review; disclose "
        "missing figures as unfinished work.\n"
        "- Reference specific findings from each agent\n"
        "- Report the delegate's actual stopping status and unresolved execution errors. "
        "A success flag alone does not establish completion: max_steps_reached means "
        "execution did not reach completion. Preserve usable partial results, but "
        "do not describe unexecuted work as completed.\n"
        "- Distinguish session-wide historical results and plans from work executed "
        "in this delegation, using execution IDs where available. Registration alone "
        "does not establish a new computation. A proposed figure layout or legend "
        "is not a generated figure; ground figure claims in the returned artifacts.\n"
        "- Before writing Methods or figure legends, reconcile agent summaries and "
        "findings with the supplied execution records. Describe what actually ran, "
        "distinguishing successful computations from failed attempts and proposed work.\n"
        "- Preserve the recorded feature lists, input representation, parameters, "
        "sample units, and execution IDs. Do not substitute a familiar method or "
        "reconstruct missing details from domain knowledge. Variable or layer names "
        "alone do not establish how the data were transformed.\n"
        "- If records conflict, explain the discrepancy and anchor the method "
        "description to the relevant execution. If a detail or library default "
        "cannot be verified from the supplied records, explicitly say it is "
        "unverified rather than inventing a value, including when evidence is "
        "missing or truncated. Scientific interpretation remains your responsibility.\n"
        "- If figures were generated during the analysis, you can reference them with [[FIGURE:figure_name]]\n"
        "- Always respond in English\n"
    )
    if source_context and source_context.strip():
        parts.append(
            "\n## Server-resolved selected-direction context\n"
            + source_context.strip()
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
        from backend.agents.pi_agent import PIAgent
        from .reviews import ChatReviewCallbacks, run_reviewed_analysis
        # Reviewers must resolve their own endpoints, not inherit the executor's
        # per-agent endpoint as though it were the installation default.
        review_client = get_agent_llm_client(state, config, "workflow", session_id)
        review_provider = AgentSettingsProvider(
            state, config, global_client=review_client, session_id=session_id,
        )
        literature_pi_agent = PIAgent(settings_provider=review_provider)
        literature_pi_agent.set_session(session_id)

        async def lit_callback(topic: str) -> str:
            result = await literature_agent.process(
                state,
                topic,
                use_deep_research=True,
                pi_plan_fn=literature_pi_agent.plan_literature_search,
                pi_review_fn=literature_pi_agent.review_literature_search,
                research_context=f"Project question: {state.initial_instruction}",
            )
            report = str(result.get("report") or "")
            note = str(result.get("note") or result.get("source_provenance") or "")
            return f"{report}\n\nSource note: {note}".strip() if note else report

        async def pi_callback(question: str, context: str) -> str:
            await _send_agent_msg("pi_question", f"Question: {question}\nContext: {context}", AgentType.BIOINFORMATICS)
            prompt = (
                f"The bioinformatics analyst is asking for your guidance:\n\n"
                f"QUESTION: {question}\n\nCONTEXT: {context}\n\n"
                f"Please provide clear, actionable guidance."
            )
            answer = await literature_pi_agent.chat_without_history(prompt)
            await _send_agent_msg("pi_response", answer, AgentType.PI)
            return answer

        async def annotate_callback(cluster_markers, tissue, organism,
                                    analysis_log=None, database_priors=None,
                                    annotation_source=None):
            if not cluster_markers:
                return {
                    "status": "error",
                    "annotations": {},
                    "error": "cluster_markers is empty. Run rank_genes_groups and extract markers first."
                }
            await _send_agent_msg("status", f"Annotating {len(cluster_markers)} clusters via literature...", AgentType.LITERATURE)
            # The executor owns source validation and applies the returned labels
            # to annotation_source; the literature lookup needs biological context.
            return await literature_agent.annotate_clusters(
                cluster_markers=cluster_markers,
                tissue=tissue,
                organism=organism,
                state=state,
            )

        async def agent_msg_callback(msg_type: str, content: str):
            await _send_agent_msg(msg_type, content, AgentType.BIOINFORMATICS)

        reviews = ChatReviewCallbacks(
            state, review_provider, literature_pi_agent, _send_agent_msg,
            pi_callback, sandbox,
        )
        bio_agent = BioinformaticsAgent(
            settings_provider=provider,
            sandbox=sandbox,
            literature_callback=lit_callback,
            pi_callback=pi_callback,
            annotate_clusters_callback=annotate_callback,
            message_callback=agent_msg_callback,
            supervisor_callback=reviews.supervise,
            figure_review_callback=reviews._figure_review_callback,
        )
        bio_agent.set_session(session_id)
        reviews.bioinformatics_agent = bio_agent

        chat_sandboxes[session_id] = sandbox
        started = False
        try:
            await sandbox.start()
            started = True
            from .source_selection import select_chat_source

            source_decision = await select_chat_source(
                literature_pi_agent, sandbox, state, instruction,
            )
            async def persist_review():
                if not await _persist_chat_terminal(session_id, state):
                    raise RuntimeError("Chat review state could not be persisted for its owner")

            result = await run_reviewed_analysis(
                state=state, instruction=instruction, bio_agent=bio_agent,
                reviews=reviews, persist=persist_review,
                collect_figures=lambda scopes: _collect_session_figures(
                    session_id, state=state, run_scope_ids=scopes,
                ),
                execution_context=_build_execution_context,
                source_decision=source_decision,
            )
            # These records live on AnalysisState. Include them in the PI
            # handoff so Methods are grounded in registered panels and
            # parameters instead of only in prose findings.
            result["score_definitions"] = [
                score.model_dump(mode="json")
                if hasattr(score, "model_dump") else score
                for score in (getattr(state, "score_definitions", None) or [])
            ]
            result["scoring_plans"] = [
                plan.model_dump(mode="json")
                if hasattr(plan, "model_dump") else plan
                for plan in (getattr(state, "scoring_plans", None) or [])
            ]
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
                try:
                    await review_provider.cleanup()
                finally:
                    await review_client.close()

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
    *,
    followup_id: str = "",
    branch_id: str = "main",
    source_context: str = "",
):
    delegation_flow.bind_facade(sys.modules[__name__])
    return await delegation_flow._run_chat_delegation(
        session_id,
        state,
        delegation,
        user_message,
        image_b64_list,
        saved_files,
        followup_id=followup_id,
        branch_id=branch_id,
        source_context=source_context,
    )
