"""Phases for _run_chat_delegation."""
from __future__ import annotations

from backend.constants import NO_OUTPUT_TOKEN_CAP

_OWNED_NAMES = frozenset({"_run_chat_delegation"})

def bind_facade(facade) -> None:
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


_DELEGATION_STOP = object()


async def _execute_delegation_success(
    session_id, state, delegation, user_message, saved_files,
    llm_manager, config, start_time, extra_clients,
    *, followup_id: str = "", branch_id: str = "main", source_context: str = "",
):
    import traceback
    from backend.utils.agent_llm import get_agent_llm_client

    delegates = delegation.get("delegates", [])

    def _delegation_instruction(raw: Any) -> str:
        instruction = (
            "Original user request (scope for final review):\n" + user_message
            + "\n\nThe latest user request above defines the authorized scope and "
            "takes precedence over the PI delegation details below. Earlier Chat "
            "requests are historical context, not additional tasks. Reuse prior "
            "results matching this request; do not repeat earlier deliverables "
            "unless the user requests a revision. Do not execute extra analyses "
            "or produce extra deliverables just because the delegation mentions "
            "them. If a prerequisite needs work beyond this scope, consult the "
            "PI and disclose it as a separate follow-up.\n\n"
            "PI delegation details:\n" + str(raw or "")
        )
        if source_context and source_context.strip():
            instruction += (
                "\n\n## Server-resolved selected-direction context\n"
                + source_context.strip()
            )
        return instruction

    # 3. Create message callback that sends to WebSocket + persists
    async def chat_msg_callback(msg: WebSocketMessage):
        # Persist before delivery so cancellation cannot lose history.
        if hasattr(state, 'realtime_messages'):
            append_durable_realtime_message(state, msg)
        await send_analysis_json(session_id, {
                    "type": msg.type,
                    "agent": msg.agent.value if msg.agent else None,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "id": msg.id,
        })

    # 4. Check for sandbox conflicts with running workflow
    workflow_running = session_id in workflows
    if workflow_running:
        # Filter out bioinformatics to avoid sandbox conflict
        delegates = [d for d in delegates if d.get("agent") != "bioinformatics"]
        if not delegates:
            identity = _active_delegation_identity(state)
            terminal = {
                "role": "assistant",
                "content": (
                    "Cannot run bioinformatics analysis while main workflow is active. "
                    "Please wait for the workflow to complete."
                ),
                "pending_figures": False,
                "timestamp": datetime.now().isoformat(),
                **identity,
            }
            _replace_latest_chat_terminal(state, terminal)
            if not await _persist_chat_terminal(session_id, state):
                return _DELEGATION_STOP
            await _send_chat_ws(session_id, "chat_error", {
                "error": "Cannot run bioinformatics analysis while main workflow is active. "
                "Please wait for the workflow to complete.",
                **terminal,
            })
            return _DELEGATION_STOP

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
                    extra_clients.append(agent_client)
                result, extra = await _execute_chat_agent(
                    session_id, state, agent_name,
                    _delegation_instruction(d.get("instruction", "")),
                    agent_client, chat_msg_callback, saved_files
                )
                extra_clients.extend(extra)
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
            # Own every child so cancellation and setup errors settle cleanup
            # before the caller closes clients. Individual model/code budgets
            # apply inside each agent, just as in the single-agent branch.
            tasks = []
            try:
                for d in group:
                    a_name = d.get("agent", "unknown")
                    a_client = get_agent_llm_client(state, config, a_name, session_id)
                    if a_client is not llm_manager.openrouter:
                        extra_clients.append(a_client)
                    tasks.append(asyncio.create_task(_execute_chat_agent(
                        session_id, state, a_name,
                        _delegation_instruction(d.get("instruction", "")),
                        a_client, chat_msg_callback, saved_files
                    )))
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for d, raw_result in zip(group, raw_results):
                agent_name = d.get("agent", "unknown")
                if isinstance(raw_result, asyncio.CancelledError):
                    raise raw_result
                if isinstance(raw_result, Exception):
                    logger.error(f"Chat delegation agent {agent_name} failed: {raw_result}")
                    agent_results[agent_name] = {"error": str(raw_result), "findings": []}
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_error", "agent": agent_name, "error": str(raw_result),
                    })
                else:
                    result, extra = raw_result
                    extra_clients.extend(extra)
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

    synthesis_prompt = _build_synthesis_prompt(
        user_message,
        delegation,
        agent_results,
        source_context=source_context,
    )
    delegation_provider = AgentSettingsProvider(state, config, global_client=llm_manager.openrouter, session_id=session_id)
    pi_model = delegation_provider.get_model(PI_AGENT_NAME)
    pi_temp = delegation_provider.get_temperature(PI_AGENT_NAME)

    dispatch = await delegation_provider.budget_text_dispatch(
        PI_AGENT_NAME,
        [{"role": "user", "content": synthesis_prompt}],
        max_tokens=NO_OUTPUT_TOKEN_CAP,
    )
    pi_response = await llm_manager.openrouter.chat(
        model=pi_model,
        messages=dispatch.messages,
        temperature=pi_temp,
        max_tokens=dispatch.max_tokens,
        prefer_stream=True,
    )
    await delegation_provider.acknowledge_text_dispatch(PI_AGENT_NAME, dispatch)
    final_content = pi_response.content or "Analysis completed but synthesis failed."

    # 7. Post-process: FIGURE and GENERATE_FIGURE markers (reuse existing patterns)
    context, figure_inventory = _build_chat_context(state)
    reviewed_chat = any("chat_review" in result for result in agent_results.values())
    if reviewed_chat:
        # Final synthesis cannot reintroduce a rejected/raw file through the
        # session-wide inventory after the producer review has selected artifacts.
        figure_inventory = all_figures
    figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
    referenced_names = figure_pattern.findall(final_content)
    synthesis_figures: List[Dict[str, str]] = []

    if referenced_names and figure_inventory:
        fig_lookup = {f["name"]: f for f in figure_inventory}
        for name in dict.fromkeys(referenced_names):
            fig = fig_lookup.get(name)
            if fig:
                entry = dict(fig)
                # Reviewed artifacts already carry their verified PDF path;
                # the image may be a record preview with a different filename.
                if (not reviewed_chat and fig.get("figure_role") != "publication"
                        and fig.get("path")):
                    pdf_path = Path(fig["path"]).with_suffix(".pdf")
                    if pdf_path.exists():
                        entry["pdfPath"] = str(pdf_path)
                synthesis_figures.append(entry)

    generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
    generate_requests = generate_pattern.findall(final_content)
    if generate_requests and reviewed_chat:
        final_content += (
            "\n\nAdditional figure requests were not executed after the completed "
            "review cycle; those figures remain unfinished."
        )
    elif generate_requests:
        try:
            generated = await _generate_figures_in_chat(session_id, state, generate_requests)
            synthesis_figures.extend(generated)
        except Exception as e:
            logger.exception("Chat delegation figure generation failed")

    # Explicit PI selection controls reviewed attachments, not the stored artifacts.
    # Preserve legacy delivery when synthesis supplies no selection markers.
    figure_candidates = all_figures + synthesis_figures
    if reviewed_chat and referenced_names:
        figure_candidates = synthesis_figures
        if set(referenced_names) - {fig["name"] for fig in synthesis_figures}:
            final_content += (
                "\n\nSome referenced figures were unavailable among the reviewed "
                "delegation artifacts and were not attached."
            )
    seen_names = set()
    merged_figures = []
    for fig in figure_candidates:
        if fig["name"] not in seen_names:
            seen_names.add(fig["name"])
            merged_figures.append(fig)

    # Clean markers from text
    cleaned_content = figure_pattern.sub('', final_content)
    cleaned_content = generate_pattern.sub('', cleaned_content).strip()
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

    chat_reviews = [
        result["chat_review"] for result in agent_results.values()
        if result.get("chat_review")
    ]
    incomplete_reviews = [
        review for review in chat_reviews if review.get("execution_incomplete")
    ]
    if incomplete_reviews:
        review_ids = ", ".join(review["id"] for review in incomplete_reviews)
        cleaned_content += (
            "\n\nBioinformatics reached its step limit before completion. "
            f"The unfinished Chat review ID is {review_ids}. "
            "Ask Chat to continue this review; saved execution records can be "
            "reused, and kernel objects can be reused if restored."
        )

    # 8. Build final chat message
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
        "pending_figures": False,
    }
    if followup_id:
        assistant_msg.update({
            "followup_id": followup_id,
            "branch_id": branch_id,
        })
    if merged_figures:
        assistant_msg["figures"] = merged_figures
    if chat_reviews:
        assistant_msg["chat_review_ids"] = [review["id"] for review in chat_reviews]
        assistant_msg["execution_incomplete"] = bool(incomplete_reviews)

    # Replace placeholder message in chat_messages
    replaced = _replace_latest_delegation_message(state, assistant_msg)
    if not replaced:
        assistant_msg["id"] = str(uuid.uuid4())
        state.chat_messages.append(assistant_msg)

    # Persist the identity-preserving terminal replacement before delivery.
    if not await _persist_chat_terminal(session_id, state):
        return _DELEGATION_STOP

    # 9. Send chat_result via WebSocket
    await _send_chat_ws(session_id, "chat_result", {
        "id": assistant_msg.get("id"),
        "content": cleaned_content,
        "figures": merged_figures,
        "timestamp": datetime.now().isoformat(),
        "reply_to": assistant_msg.get("reply_to"),
        "pending_figures": False,
        **({"chat_review_ids": assistant_msg["chat_review_ids"],
            "execution_incomplete": assistant_msg["execution_incomplete"]}
           if chat_reviews else {}),
        **({"followup_id": followup_id, "branch_id": branch_id} if followup_id else {}),
    })

    duration = time.time() - start_time
    logger.info(f"Chat delegation completed for {session_id} in {duration:.1f}s, "
                 f"{len(agent_results)} agents, {len(merged_figures)} figures")
    return None


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
    """Background task: orchestrate delegated agent work, then PI synthesizes results."""
    import traceback
    from backend.utils.llm_client import LLMManager

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)
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
        outcome = await _execute_delegation_success(
            session_id, state, delegation, user_message, saved_files,
            llm_manager, config, start_time, _delegation_extra_clients,
            followup_id=followup_id,
            branch_id=branch_id,
            source_context=source_context,
        )
        if outcome is _DELEGATION_STOP:
            return

    except asyncio.CancelledError:
        logger.info(f"Chat delegation cancelled by user for {session_id}")
        identity = _active_delegation_identity(state)
        _replace_latest_chat_terminal(state, {
            "role": "assistant",
            "content": "Cancelled",
            "timestamp": datetime.now().isoformat(),
            "pending_figures": False,
        })
        if await _persist_chat_terminal(session_id, state):
            try:
                await _send_chat_ws(session_id, "chat_cancelled", {
                    "message": "Cancelled by user",
                    "content": "Cancelled",
                    "pending_figures": False,
                    **identity,
                })
            except Exception:
                logger.warning(
                    f"[{session_id}] Failed to send cancellation terminal",
                    exc_info=True,
                )
        else:
            logger.info(
                "[%s] Skipping stale cancellation terminal delivery", session_id,
            )
        raise

    except Exception as e:
        logger.error(f"Chat delegation error for {session_id}: {traceback.format_exc()}")
        error_msg = f"Analysis failed: {str(e)}"
        identity = _active_delegation_identity(state)
        # Update placeholder message
        _replace_latest_chat_terminal(state, {
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now().isoformat(),
            "pending_figures": False,
        })
        if await _persist_chat_terminal(session_id, state):
            await _send_chat_ws(session_id, "chat_error", {
                "error": error_msg,
                "content": error_msg,
                "pending_figures": False,
                **identity,
            })

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
