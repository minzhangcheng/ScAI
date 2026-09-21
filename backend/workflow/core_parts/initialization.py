"""SingleCellAnalysisWorkflow initialization method owners."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def _set_status(state: AnalysisState, status: TaskStatus):
    state.status = status
    if status in (TaskStatus.WAITING_INPUT, TaskStatus.WAITING_SPECIES_INPUT):
        state.waiting_input_started_at = datetime.now()
    else:
        state.waiting_input_started_at = None

def _consume_restore_routing_metadata(self, analysis_state: AnalysisState) -> None:
    """Clear one-shot restore routing fields only (not provenance).

    Clears: restore_resume_phase/kind/node/gate/workflow_state and
    restore_start_intent. Does not clear last_restored_snapshot_id or
    other provenance fields owned by Task 4.
    """
    analysis_state.restore_resume_phase = None
    analysis_state.restore_resume_kind = None
    analysis_state.restore_resume_node = None
    analysis_state.restore_resume_gate = None
    analysis_state.restore_resume_workflow_state = None
    analysis_state.restore_start_intent = None

def _consume_restore_resume_metadata(self, analysis_state: AnalysisState) -> None:
    """Clear restore routing metadata only (not provenance).

    Provenance is consumed only after the canonical first restored
    RoundResult is recorded in synthesis/integration. Prefer
    ``_consume_restore_routing_metadata`` for new call sites.
    """
    self._consume_restore_routing_metadata(analysis_state)

def _restore_entry(self, state: WorkflowState) -> WorkflowState:
    """Graph entry node that determines and applies the restore entry plan.

    Uses ``restore_entry_loader`` (injectable for testing, defaults to
    ``build_restore_entry_plan``) to produce a ``RestoreEntryPlan`` from
    the analysis state's restore metadata, then:
    - Deep-copies AnalysisState and consumes routing metadata only on the copy
    - Seeds exact frozen WorkflowState for graph_node resumes
    - Sets ``_restore_entry_plan`` for the routing method
    - Handles ``preserve_round_on_initialize`` for explicit_start

    The original input AnalysisState is not mutated before node success so
    loader/validation failures leave metadata intact for retry.
    """
    import copy

    analysis_state = state["analysis_state"]
    loader = self.restore_entry_loader or build_restore_entry_plan

    # Pass restore_start_intent to the plan builder (read-only on original).
    start_intent = getattr(analysis_state, "restore_start_intent", None)
    plan = loader(analysis_state, restore_start_intent=start_intent)

    logger.info(
        f"[{self.session_id}] _restore_entry: intent={plan.intent} "
        f"target={plan.target_node} preserve_round={plan.preserve_round_on_initialize}"
    )

    # Deep-copy AnalysisState before clearing routing metadata so the
    # original input object remains intact until LangGraph checkpoints
    # the successful node return.
    returned_analysis = analysis_state.model_copy(deep=True)
    self._consume_restore_routing_metadata(returned_analysis)

    returned: WorkflowState = {**state, "analysis_state": returned_analysis}

    # Graph-node restore: seed exact frozen WorkflowState (not initialize_round).
    if plan.intent == "graph_node" and plan.target_node:
        returned["_restore_resume_node"] = plan.target_node

    for key, value in plan.workflow_state.items():
        if key not in {"analysis_state", "messages"}:
            returned[key] = copy.deepcopy(value)

    # Explicit-start restore controls same-round initialization. Emitted on
    # every entry so an earlier invocation cannot leave a stale True behind in
    # the declared channel. This is the only same-round control; durable
    # restore origin must never decide round arithmetic.
    returned["_preserve_round_on_initialize"] = bool(
        plan.preserve_round_on_initialize
    )

    # Store the plan for the routing method.
    returned["_restore_entry_plan"] = plan.model_dump(mode="json")

    return returned

async def resume_restored_wait_state(self, analysis_state: AnalysisState) -> AnalysisState:
    """Continue a restored HITL gate from its exact post-action node.

    A semantic capsule describes where the user was waiting, not a node to
    execute again.  Once an action has been accepted, convert that capsule
    to the gate's declared graph-node continuation and let ``restore_entry``
    seed the frozen WorkflowState.  Re-running the generic convergence
    predicates here changes the captured control flow (and used to skip
    annotation / plan-revision nodes entirely).
    """
    logger.info("[%s] Continuing restored wait-state snapshot", self.session_id)
    if self.compiled_graph is None:
        self._state = analysis_state
        await self.initialize()

    # Mirror the post-wait max-round extension from _wait_for_user_input().
    has_real_text = bool(
        analysis_state.pending_human_input
        and analysis_state.pending_human_input.strip() not in ("", "[CONTINUE]", "[PI_DECIDE]")
    )
    has_uploaded_files = bool(analysis_state.pending_uploaded_files)
    max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
    if (has_real_text or has_uploaded_files) and analysis_state.current_round >= max_rounds:
        analysis_state.max_rounds = analysis_state.current_round + 1

    from ..restore_targets import prepare_restored_semantic_action

    target = prepare_restored_semantic_action(analysis_state)
    if target is None:
        raise ValueError(
            "Restored semantic gate has no exact post-action target"
        )
    logger.info(
        "[%s] Restored semantic action routes exactly to %s",
        self.session_id,
        target,
    )
    return await self._invoke_restore_entry(analysis_state, self.session_id)

def __init__(
    self,
    config: SystemConfig,
    session_id: str,
    message_callback: Optional[Callable] = None,
    state_callback: Optional[Callable] = None  # Callback to save state after each round
):
    # Fail closed: unresolved recovery blocks workflow construction.
    from ...services.session_store import assert_recovery_clear
    assert_recovery_clear(session_id=session_id)

    self.config = config
    self.session_id = session_id
    self.message_callback = message_callback
    self.state_callback = state_callback

    # Initialize components
    self.llm_manager: Optional[LLMManager] = None
    self.sandbox: Optional[LocalJupyterSandbox] = None

    # Agents
    self.pi_agent: Optional[PIAgent] = None
    self.pathology_agent: Optional[PathologyAgent] = None
    self.bioinformatics_agent: Optional[BioinformaticsAgent] = None
    self.biochemistry_agent: Optional[BiochemistryAgent] = None
    self.literature_agent: Optional[LiteratureAgent] = None
    self.painter_agent: Optional[PainterAgent] = None
    self.supervisor_agent: Optional[SupervisorAgent] = None

    # Workflow graph
    self.graph = None
    self.compiled_graph = None
    self.checkpointer = None
    self._db_conn = None
    self._close_lock = asyncio.Lock()
    self._close_complete = False
    self._close_status: Dict[str, bool] = {}
    self._close_task: Optional[asyncio.Task] = None
    self._close_retry_task: Optional[asyncio.Task] = None
    self._discard_runtime_on_close = False
    self._close_discard_event = asyncio.Event()

    # Current analysis state for callbacks (updated during workflow execution)
    self._current_analysis_state: Optional[AnalysisState] = None
    self._current_graph_state: Optional[WorkflowState] = None

    # Cooperative pause gate. SET = running (await passes immediately);
    # CLEAR = pause requested (waiters block until resume).
    # Checked by @with_retry at node entry and by the bioinformatics agent's
    # autonomous tool-call loop between steps. Cleared in pause(), set in
    # signal_resume(). Defaults to running so first run() is not blocked.
    self._pause_event = asyncio.Event()
    self._pause_event.set()
    # Serializes pause persistence with the workflow-owned work that follows
    # the autonomous Bioinformatics loop (review, figure follow-up, PI summary).
    self._pause_bioinformatics_tail_lock = asyncio.Lock()

    # Event for blocking between rounds (wait for user input or continue signal)
    self._continue_event = asyncio.Event()

    # Event for blocking during annotation confirmation (exploration phase)
    self._annotation_event = asyncio.Event()

    # Event for blocking during plan review
    self._plan_review_event = asyncio.Event()
    self._plan_review_action = None  # "approve" | "skip" | dict with user modifications

    # Event for blocking during species input (inference or gene validation)
    self._species_event = asyncio.Event()
    self._species_user_response: Optional[str] = None

    # LLM failure intervention — serialized via Lock
    self._llm_failure_lock = asyncio.Lock()
    self._llm_failure_events: Dict[str, asyncio.Event] = {}
    self._llm_failure_actions: Dict[str, Optional[str]] = {}
    self._llm_failure_generations: Dict[str, int] = {}  # monotonic counter per node for idempotency
    self._unattended_gate_retries: Dict[str, int] = {}  # auto-retries spent per node at an unanswered gate
    # Supplementary analysis requests queue (set by API, consumed by HITL loop).
    # Each item is (request_text, target) where target is "annotation" | "direction" | "round".
    # Use a deque so consecutive rapid requests are preserved in order rather than overwritten.
    self._supplementary_queue: deque[Tuple[str, str]] = deque()

    # Settings provider — set in initialize(), provides hot-reloadable settings
    self._settings_provider: Optional[AgentSettingsProvider] = None

    # Technique agent (optional, created in initialize())
    self.technique_agent: Optional[TechniqueAgent] = None

    # Restore entry loader — injectable callable for testing; defaults
    # to build_restore_entry_plan.  Called by _restore_entry() to
    # determine the graph entry plan.
    self.restore_entry_loader: Optional[Callable] = None

def _vision_settings(self) -> Dict[str, bool]:
    """Vision settings — read from Provider so changes are always fresh."""
    return {
        t.value: self._settings_provider.get_vision_setting(t.value)
        for t in AgentType
    }

def _create_optional_agent(self, agent_name: str):
    """Factory — instantiate a new optional agent by name."""
    if agent_name == "pathology":
        agent = PathologyAgent(
            settings_provider=self._settings_provider,
            tile_size=self.config.image_tile_size,
            max_tiles=50,
        )
        agent.set_session(self.session_id)
        agent.llm_intervention_callback = self._pathology_llm_intervention_callback
        return agent
    if agent_name == "biochemistry":
        agent = BiochemistryAgent(settings_provider=self._settings_provider)
        agent.set_session(self.session_id)
        return agent
    if agent_name == "painter":
        agent = PainterAgent(settings_provider=self._settings_provider)
        agent.set_session(self.session_id)
        return agent
    if agent_name == "supervisor":
        agent = SupervisorAgent(
            settings_provider=self._settings_provider,
            pi_callback=self._pi_callback,
            message_callback=self._supervisor_message_callback,
            session_id=self.session_id,
            technique_agent=self.technique_agent,
        )
        agent.set_sandbox(self.sandbox)
        return agent
    if agent_name == "technique":
        agent = TechniqueAgent(
            settings_provider=self._settings_provider,
            message_callback=self._technique_message_callback,
        )
        agent.set_session(self.session_id)
        return agent
    raise ValueError(f"Unknown optional agent: {agent_name}")

def ensure_agent_ready(self, agent_name: str) -> bool:
    """Materialize or destroy an optional agent based on its current config.

    Returns True if the agent is ready for use, False if it should be
    skipped (model is empty).
    """
    def sync_technique_dependents():
        if agent_name != "technique":
            return
        for dependent_name in ("bioinformatics_agent", "supervisor_agent"):
            dependent = getattr(self, dependent_name, None)
            if dependent is not None and hasattr(dependent, "technique_agent"):
                dependent.technique_agent = self.technique_agent

    model = self._settings_provider.get_model(agent_name)
    attr = f"{agent_name}_agent"
    agent = getattr(self, attr, None)
    if not model or not model.strip():
        if agent is not None:
            logger.info(f"[{self.session_id}] {agent_name} agent disabled by empty model")
            setattr(self, attr, None)
            sync_technique_dependents()
        return False
    if agent is None:
        logger.info(f"[{self.session_id}] Creating {agent_name} agent after config enable")
        agent = self._create_optional_agent(agent_name)
        setattr(self, attr, agent)
        sync_technique_dependents()
    if hasattr(agent, "refresh_runtime_config_if_needed"):
        agent.refresh_runtime_config_if_needed()
    return True

async def _rebuild_supplementary_queue(self) -> None:
    """Restore durable queued requests into the in-memory FIFO (R1.7).

    Only nonterminal records whose target still matches the restored semantic
    gate are made runnable.  A record whose graph position has already passed
    is marked failed rather than routing backward.
    """
    state = getattr(self, "_state", None)
    if state is None:
        return
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    requests = dict(sandbox_state.get("supplementary_requests_v1") or {})
    if not requests:
        return
    terminal = {"settled", "failed"}
    existing = {key for key, _target in self._supplementary_queue}
    ordered = sorted(
        requests.items(),
        key=lambda item: (
            str((item[1] or {}).get("created_at") or ""),
            str(item[0]),
        ),
    )
    changed = False
    from backend.constants import GATE_DIRECTION_SELECTION, GATE_ROUND_WAIT
    from backend.models.schemas import TaskStatus
    from backend.services.round_snapshots.gates import round_boundary_gate

    restored_gate = None
    if getattr(state, "status", None) == TaskStatus.WAITING_INPUT:
        restored_gate = round_boundary_gate(state)

    def resumable_checkpoint(request_id: str, record: dict) -> str | None:
        import json

        from backend.agents.bioinformatics.executor_checkpoint import (
            find_executor_checkpoint_paths,
            is_executor_checkpoint_name,
        )
        from backend.config import get_session_states

        states_dir = get_session_states(str(state.id))
        requested_name = str(record.get("executor_checkpoint_name") or "")
        paths = (
            [states_dir / requested_name]
            if requested_name
            and is_executor_checkpoint_name(
                requested_name, session_id=str(state.id),
            )
            else find_executor_checkpoint_paths(
                states_dir, session_id=str(state.id),
            )
        )
        for path in paths:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                payload.get("run_kind") == "supplementary"
                and str(payload.get("supplementary_request_id") or "")
                == str(request_id)
                and not payload.get("completed")
                and payload.get("_scope_digest")
            ):
                return path.name
        return None

    for request_id, raw in ordered:
        if not isinstance(raw, dict):
            continue
        record = dict(raw)
        status = str(record.get("status") or "queued")
        if status in terminal or request_id in existing:
            continue
        target = str(record.get("target") or "")
        request_round = int(record.get("round_number") or 0)
        state_status = getattr(state, "status", None)
        continuation_name = None
        if status in {"planning", "running", "interrupted"}:
            continuation_name = resumable_checkpoint(request_id, record)
            if not continuation_name:
                record["status"] = "failed"
                record["error_summary"] = (
                    "The interrupted supplementary request has no valid scoped "
                    "executor continuation."
                )[:500]
                record["updated_at"] = datetime.now().isoformat()
                requests[request_id] = record
                changed = True
                continue
            if record.get("executor_checkpoint_name") != continuation_name:
                record["executor_checkpoint_name"] = continuation_name
                requests[request_id] = record
                changed = True
        match = False
        if target == "annotation":
            match = state_status == TaskStatus.WAITING_ANNOTATION
        elif target == "direction":
            match = (
                state_status == TaskStatus.WAITING_INPUT
                and restored_gate == GATE_DIRECTION_SELECTION
            )
        elif target == "round":
            match = (
                state_status == TaskStatus.WAITING_INPUT
                and restored_gate == GATE_ROUND_WAIT
                and request_round == int(getattr(state, "current_round", 0) or 0)
            )
        if not match:
            record["status"] = "failed"
            record["error_summary"] = (
                "Restored graph is no longer at the request's semantic gate; "
                "the request was not routed backward."
            )[:500]
            record["updated_at"] = datetime.now().isoformat()
            requests[request_id] = record
            changed = True
            continue
        self._supplementary_queue.append((request_id, target))
        existing.add(request_id)
        if target == "annotation":
            self._annotation_event.set()
        elif target in {"direction", "round"}:
            self._continue_event.set()
    if changed:
        sandbox_state["supplementary_requests_v1"] = requests
        state.sandbox_state = sandbox_state
        await state.increment_version()
        from backend.services.session_store import async_save_session

        await async_save_session(str(state.id), state, force=True)


async def initialize(self):
    """Initialize all components"""
    logger.info(f"[{self.session_id}] Initializing workflow components")
    # Close leftover provider from previous initialize() call (e.g. resume)
    if hasattr(self, '_settings_provider') and self._settings_provider:
        await self._settings_provider.cleanup()

    # Initialize checkpointer if not exists
    if self._db_conn is None:
        states_dir = get_session_states(self.session_id)
        states_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(states_dir, 0o700)
        db_path_obj = states_dir / "checkpoints.sqlite"
        db_path = str(db_path_obj)
        self._db_conn = await aiosqlite.connect(db_path)
        os.chmod(db_path_obj, 0o600)
        serde = DurableJsonPlusSerializer(allowed_json_modules=[
            ("backend.models.schemas", "AgentType"),
            ("backend.models.schemas", "AnalysisState"),
            ("backend.models.schemas", "AnalysisPhase"),
            ("backend.models.schemas", "RoundResult"),
            ("backend.models.schemas", "TaskStatus"),
            ("backend.models.schemas", "MessageRole"),
        ])
        self.checkpointer = AsyncSqliteSaver(self._db_conn, serde=serde)
        # Make sure tables are created
        await self.checkpointer.setup()
        try:
            await configure_checkpoint_retention(
                self._db_conn,
                keep=CHECKPOINTS_TO_KEEP_PER_NAMESPACE,
            )
        except Exception:
            # A readable checkpoint remains resumable if maintenance cannot
            # install its trigger; close-time compaction retries it.
            logger.warning(
                "[%s] Failed to configure checkpoint retention",
                self.session_id,
                exc_info=True,
            )

    # Initialize LLM manager with session context
    self.llm_manager = LLMManager(
        api_key=self.config.api.api_key,
        base_url=self.config.api.llm_base_url,
        proxy_url=self.config.api.proxy_url,
        session_id=self.session_id,
        agent_name="workflow",
    )

    # Initialize local Jupyter sandbox with session-specific paths
    from backend.utils.runtime_budget import (
        apply_runtime_budget_environment,
        resolve_effective_runtime_budgets,
    )

    configured_runtime_budgets = apply_runtime_budget_environment({
        "max_anndata_total_bytes": self.config.runtime_max_anndata_total_bytes,
        "max_rss_bytes": self.config.runtime_max_rss_bytes,
    })
    runtime_budgets = resolve_effective_runtime_budgets(
        configured_runtime_budgets
    )
    logger.info(
        "[%s] Effective runtime persistence budgets (%s, capacity=%s, "
        "configured=%s): %s",
        self.session_id,
        runtime_budgets.get("effective_limit_basis"),
        runtime_budgets.get("effective_memory_capacity_bytes"),
        configured_runtime_budgets,
        {
            key: value for key, value in runtime_budgets.items()
            if key.startswith("max_")
        },
    )
    self.sandbox = LocalJupyterSandbox(
        session_id=self.session_id,
        state_dir=str(get_session_states(self.session_id)),
        working_dir=str(get_session_workspace(self.session_id)),
        timeout=self.config.sandbox_timeout,
        runtime_budgets=runtime_budgets,
    )
    self.sandbox.adata_checkpoint_candidate_byte_budget = getattr(
        self.config, "adata_checkpoint_candidate_byte_budget", None
    )
    self._continuation_kernel_epoch = None

    def latch_first_kernel_epoch(epoch):
        if self._continuation_kernel_epoch is None:
            self._continuation_kernel_epoch = epoch

    self.sandbox.runtime_epoch_callback = latch_first_kernel_epoch
    self.sandbox.runtime_persistence_commit_callback = (
        self._runtime_persistence_commit_callback
    )

    # Create the settings provider — all agents read model/temp/prompt/client from it
    self._settings_provider = AgentSettingsProvider(
        state=self._state,
        config=self.config,
        global_client=self.llm_manager.openrouter,
        session_id=self.session_id,
    )

    # Install retry callbacks before agents are created so eager external
    # clients, such as Literature deep-research clients, inherit them.
    self._install_retry_notifiers()

    # Technique Agent (lightweight doc lookup, used by Supervisor and Bioinformatics)
    self.technique_agent = None
    self.ensure_agent_ready("technique")

    # Initialize required agents (PI and Bioinformatics — always active)
    self.pi_agent = PIAgent(
        settings_provider=self._settings_provider,
    )
    self.pi_agent.set_session(self.session_id)

    self.bioinformatics_agent = BioinformaticsAgent(
        settings_provider=self._settings_provider,
        sandbox=self.sandbox,
        literature_callback=self._literature_callback,
        pi_callback=self._pi_callback,
        annotate_clusters_callback=self._annotate_clusters_callback,
        message_callback=self._agent_message_callback,
        supervisor_callback=self._supervisor_callback,
        figure_plan_binding_callback=self._figure_plan_binding_callback,
        technique_agent=self.technique_agent,
        species_validation_callback=self._species_validation_callback,
        marker_db_intervention_callback=self._marker_db_intervention_callback,
    )
    self.bioinformatics_agent.set_session(self.session_id)
    self.bioinformatics_agent._pause_event = self._pause_event

    # Initialize optional agents — disabled if model is empty
    self.pathology_agent = None
    self.ensure_agent_ready("pathology")
    self.biochemistry_agent = None
    self.ensure_agent_ready("biochemistry")

    # Literature agent with mode-specific configuration
    literature_config = self._settings_provider.get_literature_config()
    self.literature_agent = LiteratureAgent(
        settings_provider=self._settings_provider,
        literature_mode=literature_config.get('mode', 'pubmed_llm'),
        deep_research_model=literature_config.get('deep_research_model', 'o4-mini-deep-research'),
        deep_research_llm_base_url=literature_config.get('deep_research_llm_base_url', ''),
        deep_research_api_key=literature_config.get('deep_research_api_key', ''),
        google_api_key=literature_config.get('google_api_key', ''),
        pubmed_email=literature_config.get('pubmed_email', ''),
        max_search_rounds=literature_config.get('max_search_rounds', 20),
        proxy_url=self.config.api.proxy_url,
    )
    self.literature_agent.set_session(self.session_id)

    # Initialize Painter Agent for graphic abstract generation (optional)
    self.painter_agent = None
    self.ensure_agent_ready("painter")

    # Initialize Supervisor Agent for real-time analysis monitoring (optional)
    self.supervisor_agent = None
    self.ensure_agent_ready("supervisor")

    # Log which agents are active
    pi_model = self._settings_provider.get_model("pi")
    bioinformatics_model = self._settings_provider.get_model("bioinformatics")
    literature_model = self._settings_provider.get_model("literature")
    path_model = self._settings_provider.get_model("pathology")
    biochem_model = self._settings_provider.get_model("biochemistry")
    painter_model = self._settings_provider.get_model("painter")
    sup_model = self._settings_provider.get_model("supervisor")
    logger.info(f"[{self.session_id}] Agents initialized: PI={pi_model}, Pathology={path_model or 'DISABLED'}, Bioinformatics={bioinformatics_model}, Biochemistry={biochem_model or 'DISABLED'}, Literature={literature_model}, Painter={painter_model or 'DISABLED'}, Supervisor={sup_model or 'DISABLED'}")

    # Build workflow graph
    self._build_graph()
    logger.info(f"[{self.session_id}] Workflow graph built")
    await _rebuild_supplementary_queue(self)

async def _wait_for_retry_pause_gate(self):
    """Park retries unless the current Bioinformatics phase must drain first."""
    bioinformatics_agent = getattr(self, "bioinformatics_agent", None)
    if (
        getattr(bioinformatics_agent, "_pause_turn_active", False)
        or getattr(bioinformatics_agent, "_pause_gate_deferred", False)
    ):
        return
    await self._pause_event.wait()


def _install_retry_notifiers(self):
    """Set retry notification and pause-gate callbacks on all LLM clients.

    When an LLM request fails and enters retry-with-backoff, the callback
    pushes a status message through the WebSocket so the frontend knows the
    system is still alive (prevents disconnect due to apparent inactivity).
    """
    def _retry_agent(agent_name: str) -> Optional[AgentType]:
        """Map an LLM client's agent_name onto the AgentType the UI labels with.

        Clients carry the agent name as a plain string, set by set_context
        from AgentType.value, so the enum lookup is exact. An unrecognised
        name (a client whose context was never set) stays unlabelled rather
        than being attributed to an agent that did not make the request.
        """
        try:
            return AgentType(agent_name)
        except ValueError:
            return None

    async def _on_retry(attempt: int, max_retries: int, delay: float, error_type: str,
                        agent_name: str = ""):
        try:
            await self._send_message(
                "status",
                f"LLM request retry {attempt + 1}/{max_retries + 1} "
                f"({error_type}, waiting {delay:.0f}s)...",
                _retry_agent(agent_name),
            )
        except Exception:
            pass  # Best-effort; never break retry logic

    async def _before_retry():
        # This is a lifecycle gate, not a notification.  Do not swallow
        # cancellation or other failures: a paused workflow must not begin
        # another request until signal_resume() releases this event.
        await _wait_for_retry_pause_gate(self)

    # Set on the global client
    self.llm_manager.openrouter.on_retry = _on_retry
    self.llm_manager.openrouter.before_retry = _before_retry
    # Set on the provider so all future lazy-created clients also get it
    self._settings_provider.set_retry_callback(_on_retry, _before_retry)
    # Also set on any already-cached clients
    for client in self._settings_provider.cached_clients():
        client.on_retry = _on_retry
        client.before_retry = _before_retry

async def _build_and_save_detailed_segment(
    self, state: AnalysisState, phase: str, title: str,
    graph_state=None,
    round_result=None,
) -> DetailedReportSegment:
    """Build a detailed report segment and save the combined HTML."""
    from ...services.report import (
        generate_detailed_report_segment_html,
        generate_combined_detailed_report_html,
    )

    from ...services.report.round_chapters import round_chapter_digest, round_segment_index
    from ...services.report.report_outline import report_result_identity

    content_html = generate_detailed_report_segment_html(
        state, phase, title,
        graph_state=dict(graph_state) if graph_state else None,
        round_result=round_result,
    )
    decision_content_html = ""
    if round_result is not None:
        from ...services.report.decision_reviews import render_round_decision_content
        decision_content_html = render_round_decision_content(state, round_result, phase, title)
    segment = DetailedReportSegment(
        decision_content_html=decision_content_html,
        result_identity_key=report_result_identity(round_result) if round_result is not None else "",
        chapter_sha256=round_chapter_digest(round_result),
        phase=phase,
        title=title,
        content_html=content_html,
        round_number=state.current_round,
    )
    previous_index = round_segment_index(state, phase, round_result)
    if previous_index is None:
        state.detailed_report_segments.append(segment)
    else:
        # Refresh content in place so already-delivered download links stay valid.
        segment.id = state.detailed_report_segments[previous_index].id
        state.detailed_report_segments[previous_index] = segment
    if round_result is not None:
        round_result.report_segment_id = segment.id
        # Annotation-gate supplements can predate the first RoundResult.
        # Bind only their captured identity, never a round-number-only guess.
        round_result.report_supplement_ids = list(dict.fromkeys(
            round_result.report_supplement_ids + [
                item.id for item in state.detailed_report_segments
                if item.supplement_owner_key == report_result_identity(round_result)
            ]
        ))

    try:
        generate_combined_detailed_report_html(state)
    except Exception as e:
        logger.warning(f"[{self.session_id}] Combined detailed report generation failed: {e}")

    return segment

def _make_initial_workflow_state(analysis_state, current_round=0):
    return make_initial_workflow_state(analysis_state, current_round)
