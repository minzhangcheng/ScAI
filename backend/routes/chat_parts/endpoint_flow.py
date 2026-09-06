"""Phases for chat_with_pi."""
from __future__ import annotations

from fastapi import File, Form

from backend.constants import NO_OUTPUT_TOKEN_CAP

_OWNED_NAMES = frozenset({"chat_with_pi"})

def bind_facade(facade) -> None:
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


def _build_chat_system_prompt(context: str) -> str:
    return 'You are the PI (Principal Investigator) Agent of a biomedical research analysis system. You have access to the current analysis state and can help the user understand results, answer questions about the data, and provide scientific insights.\n\nCurrent Analysis Context:\n' + context + '\n\nInstructions:\n- Answer the user\'s questions based on available analysis results\n- If relevant data exists in the analysis, reference it specifically\n- Be concise but thorough in scientific explanations\n- If you don\'t have enough information, say so clearly\n- You can suggest next steps or additional analyses if appropriate\n\nFIGURE DISPLAY RULES (CRITICAL — you MUST follow these):\nYou have a built-in figure rendering system. You MUST use it. NEVER provide Python/R code for the user to run. NEVER tell the user to \'run this code\'. The user CANNOT execute code — only you can display figures through these commands:\n\n1. To show an EXISTING figure from the analysis, write on its own line:\n   [[FIGURE:figure_name]]\n   Use exact names from the \'Available Figures\' list in the context above.\n   Example: [[FIGURE:umap_cell_types]]\n\n2. To CREATE a NEW figure that doesn\'t exist, write on its own line:\n   [[GENERATE_FIGURE:detailed description]]\n   The system will automatically generate and display the figure.\n   Include specific details: what plot type, which genes/clusters, colors, labels, etc.\n   Example: [[GENERATE_FIGURE:UMAP plot colored by cell_type annotation with legend on the right margin, using tab20 palette, title \'Cell Type Clusters\', frameon=False]]\n\nWhen the user asks to see, show, display, generate, or create any figure/plot/chart/image, you MUST use [[FIGURE:...]] or [[GENERATE_FIGURE:...]] — NEVER give code.\n\n- Always respond in English\n\nDELEGATION CAPABILITY:\nWhen the user\'s request requires running new computational analysis, processing new data, or using specialized agent capabilities beyond your own knowledge, you can delegate tasks to other agents. Available agents:\n- bioinformatics: Code execution, new plots, scRNA-seq analysis, clustering, DE analysis, trajectory\n- pathology: Pathology/histology image analysis or re-examination\n- biochemistry: Biochemistry data interpretation (Western blot, ELISA, etc.)\n- literature: Literature search, evidence gathering, Deep Research\n\nWhen delegation is needed, respond with ONLY a JSON block (no other text):\n{"delegation": true, "plan": "Brief description of what you will coordinate", "delegates": [\n  {"agent": "bioinformatics", "instruction": "Detailed task instruction with data paths and specifics", "priority": 1}\n], "synthesis_instruction": "How to summarize results for user"}\n\nDelegation rules:\n- Same priority number = parallel execution; lower number executes first\n- For simple questions answerable from existing results, respond normally (NO delegation)\n- For showing existing figures, use [[FIGURE:...]] (NO delegation)\n- Only delegate when NEW analysis, computation, or specialized processing is genuinely needed\n- Include specific data file paths and analysis requirements in the instruction field\n- The bioinformatics agent\'s notebook ALREADY has all previous analysis results loaded (adata with cell_type, clustering, PCA, UMAP, DE results, etc.). Do NOT include prerequisite steps like \'first ensure cell annotations are available\' or \'run clustering first\' in the instruction. Focus ONLY on the NEW analysis the user is asking for. The agent will check existing results automatically.\n'


async def chat_with_pi(
    request: Request,
    session_id: str,
    message: str = Form(...),
    target_agents: str = Form(default="[]"),
    file_groups: str = Form(default="[]"),
    file_samples: str = Form(default="[]"),
    file_types: str = Form(default="[]"),
    message_id: str = Form(default=""),
    files: List[UploadFile] = File(default=[]),
):
    """Chat with PI Agent using analysis context. Supports file uploads with per-file agent targeting."""
    from backend.utils.agent_llm import get_agent_llm_client

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
    user_message_id = _new_chat_message_id(message_id)

    # --- Commit uploaded files and their state projection as one generation ---
    if files:
        saved_files, image_b64_list, data_file_contents = (
            await _commit_chat_uploads(
                session_id,
                state,
                files,
                per_file_agents=per_file_agents,
                per_file_groups=per_file_groups,
                per_file_samples=per_file_samples,
                per_file_types=per_file_types,
                message=message,
                message_id=user_message_id,
            )
        )
    else:
        saved_files = []
        image_b64_list = []
        data_file_contents = []
        state.chat_messages.append({
            "id": user_message_id,
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "files": [],
        })

    # --- Build context & prompt ---
    context, figure_inventory = _build_chat_context(state)
    _custom_models = state.custom_models or {}
    _custom_temps = state.custom_temperatures or {}
    pi_model = _custom_models.get(PI_AGENT_NAME) or getattr(config.models, f"{PI_AGENT_NAME}_model", "")
    pi_temp = _custom_temps.get(PI_AGENT_NAME) if _custom_temps.get(PI_AGENT_NAME) is not None else getattr(config.temperatures, PI_AGENT_NAME, 0.5)

    system_prompt = _build_chat_system_prompt(context)

    # Build messages: system + history + new message
    chat_messages_list = [{"role": "system", "content": system_prompt}]
    for msg in state.chat_messages[-21:-1]:
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

    # --- Call LLM ---
    llm_client = None
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
                max_tokens=NO_OUTPUT_TOKEN_CAP,
                prefer_stream=True,
            )
        else:
            from backend.utils.settings_provider import AgentSettingsProvider
            from backend.config import SystemConfig

            provider = AgentSettingsProvider(
                state,
                (
                    config
                    if hasattr(config, "api") and hasattr(config, "context_windows")
                    else SystemConfig()
                ),
                global_client=llm_client,
                session_id=session_id,
            )
            dispatch = await provider.budget_text_dispatch(
                PI_AGENT_NAME,
                chat_messages_list,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
            )
            response = await llm_client.chat(
                model=pi_model,
                messages=dispatch.messages,
                temperature=pi_temp,
                max_tokens=dispatch.max_tokens,
                prefer_stream=True,
            )
            await provider.acknowledge_text_dispatch(PI_AGENT_NAME, dispatch)

        assistant_content = response.content if response.content else "I'm sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        logger.exception("Chat LLM error")
        assistant_content = f"Error communicating with LLM: {str(e)}"
    finally:
        if llm_client is not None:
            try:
                await llm_client.close()
            except Exception:
                logger.warning("Failed to close main chat LLM client", exc_info=True)

    # --- Check for delegation ---
    delegation = _try_parse_delegation(assistant_content)
    if delegation:
        logger.info(f"Chat delegation detected for {session_id}: {delegation.get('plan', '')}")

        await _publish_chat_delegation(
            session_id,
            state,
            delegation,
            user_content,
            image_b64_list,
            saved_files,
            user_message_id,
        )

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
        "id": _new_chat_message_id(),
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
        "reply_to": user_message_id,
        "pending_figures": bool(generate_requests),
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
        except Exception as e:
            logger.exception("Chat figure generation failed")
        finally:
            assistant_msg["pending_figures"] = False
            if not await _persist_chat_terminal(session_id, state):
                logger.info(
                    "[%s] Skipping stale final figure completion save",
                    session_id,
                )

    result: Dict[str, Any] = dict(assistant_msg)

    await async_save_session(session_id, state)

    if not await _persist_chat_terminal(session_id, state):
        raise HTTPException(
            status_code=409,
            detail="Chat session generation changed before terminal save",
        )
    return result
