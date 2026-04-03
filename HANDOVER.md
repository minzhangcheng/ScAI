# ScAI Project Handover Document

This document is intended for developers and operations personnel, providing technical details, architecture design, development guides, and operations instructions.

**Version**: 0.18.13
**Last Updated**: 2026-04-03

## Project Overview

**ScAI** (Single Cell Sequencing Analysis Driven by AI) is a multi-agent collaboration system for biomedical data analysis. Through the coordination of seven specialized agents (PI, Bioinformatics, Pathology, Biochemistry, Literature, Painter, and Supervisor) plus a lightweight Technique Agent for API documentation lookup, it automates comprehensive analysis of single-cell sequencing data with sequential execution, state version control, structured error tracking, and optimized real-time communication.

## Technical Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Port 3000)                  │
│         Zustand State + Tailwind CSS + Recharts             │
│  ┌──────┐ ┌───────┐ ┌────────┐ ┌───────┐ ┌──────┐ ┌──────┐ │
│  │ Data │ │Prompts│ │Analysis│ │Results│ │Report│ │ Chat │ │
│  │Panel │ │ Panel │ │+Errors │ │ Panel │ │Panel │ │Panel │ │
│  └──────┘ └───────┘ └────────┘ └───────┘ └──────┘ └──────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API + WebSocket (state_sync, parallel_errors)
┌──────────────────────────▼──────────────────────────────────┐
│                FastAPI Backend (Port 8000)                  │
├─────────────────────────────────────────────────────────────┤
│                  LangGraph Workflow Engine                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ┌─────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐      │ │
│  │  │   PI    │ │Pathol. │ │ Bioinfo │ │ Biochem │      │ │
│  │  │  Agent  │ │ Agent  │ │  Agent  │ │  Agent  │      │ │
│  │  └────▲────┘ └────────┘ └──┬───┬──┘ └─────────┘      │ │
│  │       │  consult/          │   │  annotate/           │ │
│  │       │  fig review        │   │  lit search          │ │
│  │       └────────────────────┘   │                      │ │
│  │       │                        ▼                      │ │
│  │  ┌────┴──────┐ ┌──────────┐ ┌────────────┐           │ │
│  │  │Supervisor │ │ Painter  │ │ Literature │           │ │
│  │  │  Agent    │ │  Agent   │ │   Agent    │           │ │
│  │  └───────────┘ └──────────┘ └────────────┘           │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  OpenRouter │  │   Jupyter   │  │   Google APIs       │ │
│  │   LLM API   │  │ Kernel+dill │  │ (Research + Imagen) │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
 sessions/             uploads/             reports/
 *.json               {id}/*              {id}/figures/
                                          {id}/tables/
```

### Directory Structure

```
scai/                           # Code directory
├── backend/
│   ├── main.py                 # FastAPI entry point (~5260 lines)
│   │                           # - REST API endpoints
│   │                           # - WebSocket real-time communication
│   │                           # - Session management & persistence
│   │                           # - _update_session_cache(): thread-safe session dict update (acquires _state_lock)
│   │                           # - Unified data upload (/upload/data) + treatment group CRUD
│   │                           # - Chunked file upload (10GB support)
│   │                           # - Report generation task scheduling
│   │                           # - Base64 stripping for message optimization
│   │                           # - PI Agent chat (figure display + generation)
│   │                           # - Chat delegation (PI delegates to agents)
│   │                           # - Session export/import (tar.gz/zip/JSON, incl. .pkl)
│   │                           # - Two-phase export (prepare + download)
│   │                           # - In-place session info editing (name + instruction)
│   │                           # - File download (/api/download with path containment)
│   │                           # - GET /api/session/{id}/annotation-report/html endpoint
│   │                           # - Direction validation + resume-without-direction guard
│   │                           # - HumanInputRequest.clear_queue: abort multi-direction queue mid-run
│   │                           # - User override sanitization (injection prevention)
│   │                           # - PI_DECIDE resume: extends max_rounds by 1 when round limit reached
│   │                           # - Figure path resolution: uses explicit png_path from figure data if present
│   │                           # - Empty cluster_markers guard in annotation endpoint
│   │
│   ├── workflow.py             # LangGraph workflow (~3270 lines)
│   │                           # - Multi-Agent coordination
│   │                           # - Sequential execution with error recovery
│   │                           # - Human-in-the-loop (wait_for_user_input)
│   │                           # - Retry with exponential backoff
│   │                           # - Convergence detection
│   │                           # - State version sync callbacks
│   │                           # - File upload handling during pause
│   │                           # - Round context enrichment (previous findings + code history)
│   │                           # - annotate_clusters_callback for literature-based annotation
│   │                           # - Deep Research topic merging + gap coverage fallback
│   │                           # - Report-only resume via generate_report()
│   │                           # - Selective agent activation (skip_nodes from agents_for_next_round)
│   │                           # - Agent auto-activation when PI tasks exist but agent not in agents_for_next_round
│   │                           # - Dynamic PI planning: refines downstream agent tasks based on completed results
│   │                           # - Cross-agent context: biochemistry → pathology → bioinformatics findings sharing
│   │                           # - Analysis log propagation and deduplication (by step ID)
│   │                           # - Graphic abstract revision loop (PI review + Painter revision + best selection)
│   │                           # - Annotation HTML report generation at annotation pause
│   │                           # - Innovation phase: user guidance merged as supplementary tasks (not direction override)
│   │                           # - Direction pop persistence: state saved immediately after queue pop
│   │                           # - key_markers supplemented to 10 genes from rank_genes_groups before PI review
│   │                           # - _normalize_tasks(): normalizes dict/str task items returned by LLMs
│   │                           # - annotate_clusters_callback: returns cached result if annotation already confirmed
│   │                           # - WebSocket heartbeat: ping every 30s in disconnect listener to survive proxies
│   │                           # - ws_lock held for entire check-and-send sequence (thread-safe WS send)
│   │
│   ├── report_generator.py     # Report generator (~5460 lines)
│   │                           # - No-sandbox report compilation (v0.7.3 rewrite)
│   │                           # - PI analysis review pre-step (reads notebook, views figures via vision API)
│   │                           # - Builds analysis record from code_history
│   │                           # - Scans workspace for existing figures/tables
│   │                           # - PI plans narrative around existing assets
│   │                           # - Cluster annotation table extraction
│   │                           # - Report appendix (project info, agent config, data summary, round history)
│   │                           # - Full analysis log timeline (chronological steps with code/figures/reasoning)
│   │                           # - Figure/table scientific interpretation rendering
│   │                           # - generate_annotation_context_report_html(): standalone HTML annotation context report
│   │                           # - Per-round narrative with phase/direction labels; innovation sections required 300-500+ words
│   │
│   ├── regenerate_report_html.py # Report HTML regeneration utility (~1600 lines, appendix + analysis log support)
│   │
│   ├── config.py               # Configuration management (~360 lines)
│   │                           # - APIConfig: LLM keys + proxy (google_api_key moved to literature_config)
│   │                           # - ModelConfig: Model selection per agent (incl. literature_model, supervisor_model)
│   │                           # - AgentTemperatures: Per-agent temperatures (incl. supervisor)
│   │                           # - AgentPrompts: System prompts (English, batch effect + PDF guidelines, painter_prompt + supervisor_prompt placeholders)
│   │                           # - SystemConfig: System parameters
│   │                           # - Data directory path definitions
│   │                           # - Literature agent defaults (mode, max_search_rounds)
│   │
│   ├── agents/
│   │   ├── base.py             # Agent base class (rollback_last_turn for safe history cleanup)
│   │   ├── pi_agent.py         # PI Agent (coordinator + figure reviewer + graphic abstract reviewer + dynamic task planner)
│   │   │                       #   annotation review: returns alternative_cell_types + low_confidence_reason; 10 markers shown
│   │   │                       #   innovation planning: mandates 4-6 tasks with concrete method/target specification
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent (autonomous, figure step budget, pip install, cumulative revision history, analysis log)
│   │   │                       #   register_score tool: documents score methodology (method, gene list, source, parameters)
│   │   │                       #   mandatory score workflow: consult PI → compute → register_score
│   │   │                       #   _WORKSPACE_DIR kernel variable: pre-defined pathlib.Path; agent always uses it for file paths
│   │   │                       #   obs/var names auto-fix: make_unique() called after every execute_code + at data load time
│   │   │                       #   annotate_clusters must be separate step from code that generates markers
│   │   │                       #   gene module scoring: use_raw=True, preserve adata.raw before HVG, species-aware gene names
│   │   │                       #   security restrictions: globals()/exec()/eval()/subprocess blocked in system prompt
│   │   │                       #   checkpoint round validation: rejects checkpoints with mismatched round_number
│   │   │                       #   step budget fix: remaining = original_total - step_count (not step_count + original_total)
│   │   │                       #   message trim: walks cut point backward to avoid orphaning tool_result messages
│   │   │                       #   cell type dict uses json.dumps for safe serialization (injection prevention)
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── biochemistry_agent.py    # Biochemistry Agent (3-step v2, ~785 lines)
│   │   ├── literature_agent.py # Literature Agent (2 research modes + cluster annotation, ~1270 lines)
│   │   │                       #   result dict includes search_mode field (actual mode used, including fallback chain)
│   │   │                       #   annotate_clusters cache: keyed by organism+tissue+top-gene fingerprint, skips redundant DR calls
│   │   │                       #   empty cluster_markers guard: returns error immediately without calling LLM
│   │   │                       #   previous literature reports included in annotate_clusters context
│   │   ├── painter_agent.py    # Painter Agent (graphic abstract + revision support)
│   │   │                       #   token budget management: estimates input tokens, auto-trims to 30K safe limit for gemini-3-pro-image-preview
│   │   ├── supervisor_agent.py # Supervisor Agent (real-time bioinformatics step monitoring, sampled LLM evaluation, PI escalation, custom system_prompt, ~2645 lines)
│   │   └── technique_agent.py  # Technique Agent (API doc lookup via sandbox introspection + LLM, ~340 lines)
│   │                           #   _set_llm_context(): called before each LLM call to bind shared client to session+agent
│   │                           #   per-round narrative with phase/direction headers; publication figures in figure map
│   │                           #   dynamic report sections: [innovation_N] per direction, [exploration_findings] if multi-phase
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic data models (~640 lines)
│   │                           # - AnalysisState: Core state with version control + treatment_groups + data_files + biochemistry_indicators + analysis_log
│   │                           #   + score_definitions (List[ScoreDefinition]) + annotation_report_html_path
│   │                           # - ScoreDefinition: Score methodology record (score_name, method, description, parameters, gene_list, gene_list_source, round_number)
│   │                           # - AnalysisStep: Comprehensive step log (code/PI/literature/annotation/findings/figure/text)
│   │                           # - TreatmentGroup: Experimental group definition (name, sample_count)
│   │                           # - DataFile: Universal file record (section: pathology/sequencing/biochemistry, group_id, sample_number)
│   │                           # - BiochemistryIndicator: Manual biochemistry entry with group/sample
│   │                           # - TaskStatus: GENERATING_REPORT status
│   │                           # - AgentError: Structured error tracking
│   │                           # - RoundResult: agents_for_next_round for selective activation + direction field
│   │                           # - ClusterAnnotation: + alternative_cell_types + low_confidence_reason fields
│   │                           # - PublicationFigure/Table: Publication outputs (+ interpretation field)
│   │                           # - GraphicAbstract: Painter agent output
│   │                           # - BioinformaticsCheckpoint: Step-level recovery (+ analysis_log)
│   │                           # - Data helper methods: get_files_by_section/group, build_experimental_design_summary
│   │
│   ├── utils/
│   │   ├── llm_client.py       # LLM client (~980 lines, OpenRouter + Google DR, precise retry with InternalServerError)
│   │   ├── sandbox.py          # Jupyter execution environment (~1320 lines, auto PDF, code safety validation, extract_code_from_markdown)
│   │   │                       #   _WORKSPACE_DIR (pathlib.Path) pre-defined in kernel; agent uses this variable for all paths
│   │   │                       #   savefig redirect: Figure.savefig monkey-patched to auto-redirect workspace-root saves → figures/
│   │   │                       #   numba cache: /tmp/numba_cache_{session_id} per session to prevent cross-session conflicts
│   │   │                       #   open() write safety check: allows writes referencing _WORKSPACE_DIR/working_dir/figures/tables//tmp
│   │   │                       #   execute() params: save_to_history=False, skip_safety_check=True for internal/housekeeping calls
│   │   │                       #   working_dir set to absolute path; kernel init sets os.chdir + creates figures/tables subdirs
│   │   │                       #   I/O sandbox: intercepts open()/Path.mkdir()/write_text()/write_bytes() to block writes outside session dir
│   │   │                       #   _kernel_dead flag: tracks failed restart, subsequent execute() calls return error immediately
│   │   ├── checkpoint.py       # Generic checkpoint manager (analysis + report resumption)
│   │   ├── image_utils.py      # Image processing
│   │   ├── upload.py           # Chunked upload manager (~398 lines)
│   │   ├── error_parsing.py    # Error type extraction utility (~19 lines)
│   │   └── llm_logger.py       # LLM logging (~216 lines)
│   │
│   └── static/                 # Static files (backup)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main application component (+ Chat tab + About dialog)
│   │   ├── components/         # React components
│   │   │   ├── ChatPanel.tsx   # PI Agent free-form chat + delegation (~657 lines)
│   │   │   ├── AboutDialog.tsx # About dialog (project info, author, contact)
│   │   │   ├── AnnotationConfirmationDialog.tsx  # Cell annotation review dialog (Phase 1 pause, ~400 lines)
│   │   │   │                                    # - expandable cluster rows: alternative_cell_types + low_confidence_reason
│   │   │   │                                    # - download button linking to annotation context HTML report
│   │   │   ├── ExplorationReportDialog.tsx       # Phase 1 exploration report & multi-direction selection (multi-select, notes, score badges, countdown)
│   │   │   └── ...             # Other panels (Analysis, Data, Prompts, Results, Report)
│   │   ├── stores/             # Zustand state (+ chatMessages, chatDelegation, generating_report status)
│   │   ├── hooks/              # API Hooks
│   │   ├── i18n/               # Internationalization
│   │   └── styles/             # CSS styles
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── requirements.txt            # Python dependencies
├── start.sh                    # Startup script
├── stop.sh                     # Stop script
├── README.md
└── HANDOVER.md

data/                    # Data directory (sibling to code directory)
├── sessions/                   # Session-centric data (new layout since v0.12.4)
│   └── {session_id}/
│       ├── session.json        # Session metadata
│       ├── workspace/          # Analysis code, figures
│       ├── reports/            # HTML reports, figures, tables
│       ├── uploads/            # Uploaded data files
│       ├── states/             # Checkpoints, notebook (.ipynb), env (.pkl)
│       └── logs/               # Session LLM interaction logs (llm.jsonl)
├── logs/                       # Global system logs (app.log, error.log, debug.log)
│   └── llm/                    # Daily aggregated LLM logs ({YYYY-MM-DD}.jsonl)
├── upload_temp/                # Global upload temp (chunked uploads)
├── uploads/                    # (legacy, kept for backward compatibility)
├── states/                     # (legacy, kept for backward compatibility)
├── workspace/                  # (legacy, kept for backward compatibility)
└── reports/                    # (legacy, kept for backward compatibility)

dep/                     # Dependencies directory (sibling to code directory)
├── venv/                       # Python virtual environment
│   ├── bin/                    # Python executables
│   ├── lib/                    # Python packages
│   └── ...
└── node_modules/               # Node.js dependencies
    └── (symlinked to frontend/node_modules)
```

## Core Module Details

### 1. Workflow Engine (workflow.py)

Multi-Agent coordination system based on LangGraph with sequential execution and error recovery:

```python
class SingleCellAnalysisWorkflow:
    def __init__(self, config, session_id, message_callback, state_callback):
        # message_callback: Real-time message push (WebSocket)
        # state_callback: Save state after each round

    async def run(self, state: AnalysisState):
        # Iteratively execute analysis rounds
        # Each round: PI planning → Biochemistry → Pathology → Bioinformatics → Literature → PI synthesis
```

**Workflow Graph Structure:**

```
User creates Session → Upload data → WebSocket connect → Start analysis
                                                 │
  ┌──────────────────────────────────────────────┘
  ▼
Phase 1: Exploration (Round 1)
  │
  ├─ initialize_round       Process pending inputs, current_round++, status=RUNNING
  ├─ pi_planning            initial_exploration_instruction(): unbiased comprehensive plan
  ├─ biochemistry_analysis  Extract indicators → Image analysis → Synthesize interpretation
  ├─ pathology_analysis     Tiling → Vision API → Synthesis (depends on biochemistry results)
  ├─ bioinformatics_analysis Jupyter autonomous execution (200-step budget)
  │    ├── QC → Normalization → Clustering → Cell Annotation
  │    │       └── [BLOCKING] annotation_confirmation → User confirms/corrects
  │    └── DEG → GO/KEGG → Cell Communication
  ├─ literature_research    Deep Research / PubMed+LLM / Quick search
  └─ pi_synthesis           Synthesize findings → Exploration Report → Phase switch → WAITING_INPUT
                                │
                  [exploration_report dialog: User selects direction]
                                │
Phase 2: Innovation (Round 2+)
  │
  ├─ Direction scheduling
  │    ├── Single direction: user_chosen_direction
  │    ├── Multiple directions: directions_queue (auto-advance between rounds, skip wait)
  │    └── PI-selected: [PI_DECIDE] → select_best_directions() → directions_queue
  │
  ├─ initialize_round       (same as Phase 1)
  ├─ pi_planning            innovation_instruction() + selective agent activation
  ├─ [biochemistry/pathology/bioinformatics/literature] Run per PI instruction (may skip)
  ├─ pi_synthesis
  │    └── When directions_queue empty: trigger multi-direction correlation analysis
  │
  ├─ wait_for_user_input  ◄── configurable (-1=manual, 0=no-wait, >0=countdown+60s)
  │    │   When directions_queue non-empty: auto-skip wait
  │    │
  ┌───┴───┐
  │       │
(continue) (stop)
  │       │
  │       ▼
  │  generate_final_report
  │       │
  │       ▼
  │  generate_graphic_abstract  (PI review + Painter revision x2 + best selection)
  │       │
  │       ▼
  │  generate_detailed_report   (9-chapter HTML report, checkpoint resume)
  │       │
  │       ▼
  │      END
  │
  └───► check_convergence ──────────────────► initialize_round (next round)
           │  2 consecutive rounds with no new findings → stop
           └─ max_rounds reached → stop (unless user input extends)
```

**Key Features:**

1. **Two-Phase Workflow**: Phase 1 (Exploration, Round 1) runs all agents for unbiased discovery; Phase 2 (Innovation, Round 2+) focuses on user-selected directions with selective agent activation.
2. **Sequential Execution**: Biochemistry → Pathology → Bioinformatics → Literature run sequentially within each round. Biochemistry runs first so its results inform downstream agents. PI dynamically refines each agent's tasks based on already-completed results.
3. **Annotation Confirmation Blocking**: After Bioinformatics Agent completes cluster annotation, workflow pauses (`WAITING_ANNOTATION` status), sends `annotation_confirmation` WebSocket message, and blocks on `_annotation_event`. Resumes when user confirms/corrects annotations via the dialog.
4. **Multi-Direction Queue**: When user selects multiple directions or PI decides (`[PI_DECIDE]`), directions are queued in `directions_queue`. Each direction gets its own innovation round. `wait_for_user_input` auto-skips when the queue is non-empty. `max_rounds = len(selected_dirs) + 1`. After the queue empties, PI synthesizes a multi-direction correlation analysis.
5. **Retry Mechanism**: `@with_retry(max_retries=3)` decorator with exponential backoff. On exhaustion, returns `{"error": ..., "findings": [], "status": "failed"}` so PI synthesis can distinguish agent failures from skipped agents.
6. **Structured Error Tracking**: Errors captured in `AgentError` model with agent, type, message, details
7. **State Version Control**: `_send_state_sync()` method ensures atomic version updates; status guard in `_wait_for_user_input` ensures `WAITING_INPUT` is synced before blocking (prevents pause/resume deadlock)
8. **Figure Review Loop**: Bioinformatics Agent's figures are reviewed by PI before approval (up to 3 revisions). PI review includes experimental design context with a Data Fidelity criterion to catch fabricated groups. Revision uses `chat_with_vision()` so the bioinformatics agent can see its own figure alongside PI feedback. Full cumulative revision history is tracked in `PublicationFigure.revision_history` and passed to the code reviser to prevent re-introducing fixed issues.
9. **AI-Curated Brief Figures**: After all analysis rounds complete, PI Agent's `select_brief_figures()` curates the figure set shown in the Results tab — filters out QC/diagnostic plots (e.g. qc_violin, pca_variance, hvg_plot) and superseded intermediate versions, keeping only key result figures. Falls back to all publication figures if AI selection returns empty. Workspace figures (workspace root + figures/) are scanned and compressed (`compress_image_to_base64`) for inclusion alongside existing publication figures.
10. **Figure Step Budget**: Figure-only iterations (using `generate_publication_figure` tool) are excluded from the bioinformatics agent's step count. Safety cap of `MAX_FIGURE_ITERATIONS=5` prevents infinite figure generation loops — after exhaustion, agent refuses new figures and redirects to analysis. `max_tokens=32000` for figure revision code generation prevents truncation.
11. **Message History Trimming**: Bioinformatics Agent trims its message history to the last 80 messages (preserving the system message) before each `chat_with_tools()` call to prevent unbounded context growth during 200-step analyses.
12. **Tool Message History Consistency**: Bioinformatics Agent appends exactly one assistant message per LLM response (using `raw_assistant_message` if available) before iterating over tool calls. Eliminates duplicate or missing assistant messages that previously caused API errors with Gemini's `thought_signature` requirement.
13. **Figure Data Fidelity**: All figure generation and revision prompts include explicit experimental design constraints (`CRITICAL DATA CONSTRAINT`) with actual group names from `state.treatment_groups`. This prevents LLM hallucination of non-existent treatment groups. Constraint is injected in: `_build_data_context()`, `generate_publication_figure` tool description, `_revise_figure_code()` prompt, and PI `review_figure()` prompt.
14. **Convergence Detection**: Stops after 2 consecutive rounds with no new findings
15. **Human-in-the-Loop**: `wait_for_user_input` node blocks between rounds via `asyncio.Event`. Frontend shows `RoundFeedbackDialog` with configurable countdown (default 300s; -1=manual-only, 0=no-wait auto-continue, stored in `round_wait_seconds`). Users can provide text guidance, upload files (targeted to specific agents), or skip. Backend respects `round_wait_seconds`: manual mode waits indefinitely, no-wait proceeds immediately, countdown mode uses frontend timer + 60s buffer as backend safety timeout. `POST /api/session/{id}/human-input` calls `signal_continue()` when status is `WAITING_INPUT` so the workflow wakes immediately on submission. User input or file uploads on the last round extends `max_rounds` by 1 (only if not true multi-direction mode, i.e., `len(user_chosen_directions) <= 1`).
16. **File Upload During Pause**: Users can upload pathology images while workflow is in `waiting_input` or `completed` state. Files are tracked in `pending_uploaded_files` and processed in the next round.
17. **Round Context Enrichment**: `_build_previous_findings_context()` and `_build_code_history_summary()` methods provide each agent with previous round findings and code execution history, preventing redundant analyses in multi-round workflows.
18. **Literature-based Cluster Annotation**: `_annotate_clusters_callback()` routes bioinformatics agent's `annotate_clusters` tool calls to the literature agent for efficient batch cluster annotation.
19. **Deep Research Topic Merging**: When using Gemini Deep Research mode, multiple topics are merged into a single comprehensive query (one DR call instead of N). After the DR call, `_evaluate_dr_coverage()` checks which topics were insufficiently covered and falls back to PubMed+LLM for gaps.
20. **Report-Only Resume**: `generate_report()` method runs only the report generation phase (final report → graphic abstract → detailed report), bypassing analysis rounds. Used when a session completed all analysis but was interrupted during report generation. WebSocket handler detects this condition automatically.
21. **GENERATING_REPORT Status**: `TaskStatus.GENERATING_REPORT` marks sessions in report generation phase, enabling detection of interrupted report generation on resume.
22. **Selective Agent Activation**: From round 2 onwards, PI Agent decides which agents to activate via `RoundResult.agents_for_next_round`. Bioinformatics Agent always runs. `_pi_planning()` builds `skip_nodes` from the previous round's `agents_for_next_round`, and `@with_retry` skips deactivated nodes with WebSocket notification. Each round's `skip_nodes` is rebuilt fresh (not accumulated), so previously skipped agents can be re-activated. **Auto-activation**: If PI's `next_steps` contains tasks prefixed with an agent name (e.g., `[Pathology] ...`) but that agent is not in `agents_for_next_round`, the workflow auto-activates it.
23. **Analysis Log Propagation**: `_post_bioinformatics()` collects `analysis_log` from the bioinformatics agent result, deduplicates by step ID, and appends to `AnalysisState.analysis_log`. Code history is similarly deduped by `CodeExecution.id`.
24. **Cross-Agent Context Sharing**: Biochemistry findings are injected into pathology and bioinformatics agent prompts. Pathology findings are injected into bioinformatics agent prompts. `_collect_current_round_results()` collects findings from already-completed agents for dynamic PI planning. Each agent benefits from the accumulated results of all preceding agents in the same round.
25. **Dynamic PI Planning**: PI Agent's `plan_agent_tasks()` method refines each downstream agent's tasks based on already-completed results within the round. Uses `self.chat()` to accumulate context across successive planning calls, building a coherent chain of thought. PI history is reset at the start of each round's planning phase. For round 2+, previous round PI summary is passed as strategic overview context. Round 3+ falls back to `next_steps` → `pi_summary` if prefix-tagged tasks are not found.
26. **Docker Pip Install**: Bioinformatics Agent's `_validate_code()` allows `subprocess` calls when all usages are pip/pip3 install commands (safe in Docker environments). System prompt includes package installation guidelines.
27. **Graphic Abstract Revision Loop**: The graphic abstract generation uses a 3-phase revision process: (1) initial generation by Painter Agent; (2) up to 2 rounds of PI review via vision API (`review_graphic_abstract()`) + Painter revision with accumulated feedback history (`revise_graphic_abstract()`); (3) PI selects the best candidate from all versions using side-by-side vision comparison (`select_best_graphic_abstract()`). Metadata stored on `GraphicAbstract`: `revision_count`, `revision_history`, `all_candidates`, `selected_index`.
28. **Score Registration Workflow**: Bioinformatics Agent has a `register_score` tool that records how a score was computed (method, gene list, source, key parameters) as a `ScoreDefinition` in `AnalysisState.score_definitions`. The agent is instructed to call `consult_pi → execute → register_score` for every scoring operation. Scores are deduplicated by `score_name` (update existing if same name). Score definitions surface in the final report's methodology section.
29. **Jupyter I/O Sandbox**: Kernel initialization injects a runtime I/O sandbox that intercepts `open()`, `Path.mkdir()`, `Path.write_text()`, and `Path.write_bytes()` to block writes outside the session directory. Allowed roots: session root, `/tmp`, system paths (`~/.cache`, `~/.local`, etc.). Blocked writes raise `PermissionError` with a clear redirect message. Also sets `os.chdir()` to workspace, creates `figures/` and `tables/` subdirs, configures `sc.settings.figdir` to absolute path.
30. **Kernel Dead Detection**: `_kernel_dead` flag on `LocalJupyterSandbox`. Set when kernel restart fails after timeout. All subsequent `execute()` calls immediately return `success=False` with a clear error message instead of hanging.
31. **Thread-Safe Session Cache**: `_update_session_cache(session_id, state)` in `main.py` acquires `_state_lock` before updating `sessions` dict and `_session_access_times`. All direct `sessions[id] = state` assignments replaced with this helper throughout `main.py`.
32. **Annotation Context HTML Report**: During the annotation confirmation pause, `generate_annotation_context_report_html()` generates a standalone HTML report saved to the session's reports directory. Path stored in `state.annotation_report_html_path`. Accessible via `GET /api/session/{id}/annotation-report/html`. `annotation_report_url` returned in session load API.
33. **Innovation Phase: User Guidance Merging**: In round 2+, `adjusted_instructions` (from user input processing) are no longer allowed to replace direction-based PI planning. Instead, they are merged as supplementary `[User guidance]` tasks appended to existing task lists after direction-based planning completes.
34. **Direction Pop Persistence**: After popping a direction from `directions_queue`, the workflow immediately calls `state_callback` to persist the updated queue, preventing direction loss on crash between direction pop and round completion.

**WebSocket Message Types:**

| Type | Purpose |
|------|---------|
| `status` | Status updates |
| `state_sync` | Version control and state synchronization |
| `parallel_errors` | Structured error list from parallel execution |
| `message` | General agent messages |
| `progress` | Progress updates |
| `result` | Analysis results |
| `error` | Error messages |
| `round_summary` | End-of-round summary |
| `integrated_report` | Complete analysis report |
| `graphic_abstract` | Painter Agent output |
| `detailed_report` | Detailed report output |
| `figure_review` | PI reviewing generated figures |
| `figure_approved` | Figure approved |
| `figure_revision` | Revision requested |
| `completed` | Analysis complete (includes round_results + final_summary on reconnect) |
| `message_history` | Batch message restore (reconnect to completed session) |
| `human_input_requested` | Workflow waiting for user input (round feedback) |
| `human_input_received` | User input received, continuing workflow |
| `annotation_result` | Literature-based cluster annotation result |
| `annotation_confirmation` | Phase 1 pause — sends cluster annotations (incl. alternative_cell_types, low_confidence_reason) + report_html_url for review |
| `exploration_report` | Phase 1 completion — sends exploration report and requests research direction selection |
| `data_request` | Agent requesting additional data |
| `chat_delegation` | Chat delegation progress (started/agent_complete/agent_error/synthesizing) |
| `chat_result` | Final chat delegation result |
| `chat_error` | Chat delegation error |
| `supervisor_feedback` | Supervisor Agent monitoring feedback |
| `supervisor_checkpoint` | Supervisor biological checkpoint notification |
| `supervisor_summary` | Supervisor round summary |
| `direction_correlation` | Multi-direction correlation analysis result (v0.13.1) |

### 1.1 Biochemistry Agent (v2 - 3-Step Workflow)

Since v0.5.1, the biochemistry agent uses a 3-step analysis pipeline:

1. **Step 1**: Process structured indicators and data files (CSV/XLSX)
2. **Step 2**: Full visual analysis of images (comprehensive prompt, not strict JSON extraction)
3. **Step 3**: Summarize findings, draw conclusions, and suggest follow-up experiments

This approach is more robust against vision API inconsistencies (see Known Issues).

### 1.2 Literature Agent (2 Research Modes + Cluster Annotation, ~1270 lines)

The Literature Agent supports 2 research modes with automatic fallback:

| Mode | API | Model/Endpoint | Max Runtime |
|------|-----|----------------|-------------|
| `gemini_deep_research` | Google Interactions API | `deep-research-pro-preview-12-2025` | 30 min |
| `pubmed_llm` (default) | NCBI E-utilities + LLM | PubMed search + any LLM model | ~2 min |

**Fallback chain**: Configured mode → PubMed + LLM → Quick search (LLM knowledge only)

**Search behavior**:
- **Gemini Deep Research**: Single call per topic — DR handles multi-round search/synthesis internally on Google's servers. Gap coverage for multi-topic queries is handled at the workflow level (`_evaluate_dr_coverage()` in `workflow.py`), falling back to PubMed+LLM for insufficiently covered topics.
- **PubMed + LLM**: Multi-round iterative search (configurable 1-20 rounds). After each round, LLM evaluates whether results are sufficient and generates refined queries if needed.

**Per-session configuration**: Literature mode, Google API key, PubMed email, and max search rounds are stored in `AnalysisState.literature_config` (not global `APIConfig`). The `GoogleDeepResearchClient` is created lazily inside the agent only when a Google API key is provided.

**Key methods:**
- `_multi_round_gemini()`: Single Google Interactions API call (start + poll, 5s interval)
- `_multi_round_pubmed()`: E-Search → E-Fetch → XML parsing → LLM synthesis
- `_validate_and_degrade_query()`: Progressive query simplification (strip field tags → simple keywords) when PubMed returns zero results
- `_build_research_context()`: Collects raw experimental context (treatment groups, biochemistry indicators, previous/current round findings) for LLM-based query generation, replacing hardcoded `_build_research_query()`
- `_evaluate_search_results()` / `_evaluate_pubmed_results()`: LLM judges sufficiency
- `annotate_clusters()`: Two-phase cluster annotation — single literature search + LLM matching for all clusters at once. Called by bioinformatics agent via `annotate_clusters_callback`

### 1.3 PI Agent Chat (main.py, Chat Endpoints)

Free-form conversational interface with the PI Agent, separate from the workflow analysis:

**Endpoint**: `POST /api/session/{session_id}/chat`

**Features:**
- Accepts text message + file uploads with per-file agent targeting (PI, Pathology, Bioinformatics, Biochemistry)
- Builds analysis context from session state (round results, figures, literature reports) for informed responses
- **Figure Display**: PI references existing figures via `[[FIGURE:name]]` markers — matched against on-disk figure inventory (workspace + reports directories)
- **Figure Generation**: PI requests new figures via `[[GENERATE_FIGURE:description]]` — system spins up Jupyter sandbox, generates code via LLM, executes it, captures PNG/PDF output
- File routing: Uploaded files → `data_files` with section based on target agent (pathology/sequencing/biochemistry), with group_id, sample_number, and data_type metadata
- Chat history stored in `AnalysisState.chat_messages` (last 20 messages used as LLM context)
- Vision API used when images are attached

**Key functions:**
- `_build_chat_context()`: Builds analysis summary + scans disk for available PNG figures
- `_generate_figures_in_chat()`: LLM writes code → sandbox executes → captures figures (neutralizes `plt.show/close`)
- `chat_with_pi()`: Main endpoint handler — file saving, context building, LLM call, figure post-processing

### 1.4 Chat Delegation (main.py)

When a user's chat request requires specialized analysis beyond PI's direct knowledge, PI can delegate to other agents:

**Flow:**
1. User sends message via `POST /api/session/{id}/chat`
2. PI Agent receives the message with a delegation-aware system prompt
3. If PI returns a delegation plan (`{"delegation": true, "delegates": [...], "plan": "..."}`), the backend:
   - Returns `{"status": "processing"}` immediately to the frontend
   - Spawns a background task `_run_chat_delegation()` to orchestrate agents
4. Each delegated agent is executed via `_execute_chat_agent()`:
   - **Bioinformatics**: Creates full sandbox with session restore, literature/PI callbacks, runs `run_autonomous_analysis()`
   - **Pathology**: Creates `PathologyAgent.process()` call
   - **Biochemistry**: Creates `BiochemistryAgent.process()` call
   - **Literature**: Creates `LiteratureAgent.process()` call
5. WebSocket messages track progress (`chat_delegation` type with status: started/agent_complete/agent_error/synthesizing)
6. After all agents complete, PI synthesizes results via `_build_synthesis_prompt()`
7. Final response saved to `chat_messages` and sent via `chat_result` WebSocket message

**Cancellation**: `POST /api/session/{id}/chat/cancel` cancels the background delegation task and stops any active sandbox.

**Key functions:**
- `_try_parse_delegation()`: Parses PI response as delegation plan (JSON or markdown code block)
- `_execute_chat_agent()`: Executes a single agent with full callback wiring
- `_run_chat_delegation()`: Background task orchestrating delegation flow
- `_build_synthesis_prompt()`: Builds PI synthesis prompt from agent results
- `_collect_session_figures()`: Collects figures from workspace (both `figures/` subdir and workspace root, deduped by stem, optionally filtered by creation time)

### 1.5 Technique Agent (technique_agent.py, ~340 lines)

Lightweight API documentation lookup agent. Not autonomous — called by other agents as a utility.

**Purpose:** Extracts docstrings from installed Python packages via Jupyter sandbox introspection, then summarizes them into concise API references using a cheap LLM (`google/gemini-3.1-flash-lite-preview`).

**Two entry points:**
- `lookup_docs(sandbox, function_path)`: Direct API doc lookup. Executes `inspect.getdoc()` and `inspect.signature()` in the sandbox, returns formatted reference (signature, key parameters, example)
- `lookup_from_error(sandbox, error_text, code)`: Error-driven lookup. Parses error tracebacks (`AttributeError`, `TypeError`, `ImportError`, `ModuleNotFoundError`) to identify the problematic function, then looks up correct API usage

**Integration points:**
- **Bioinformatics Agent**: `lookup_documentation` tool in TOOLS array. Agent can proactively look up API docs before writing code for unfamiliar packages (decoupler, gseapy, scvelo, cellrank, etc.)
- **Supervisor Agent**: Auto-triggered when the same API error type occurs in 2 consecutive steps (`_should_trigger_technique()`). Injects corrective guidance as `[Technique Support]` feedback

**Key helper:**
- `_guess_function_path()`: Resolves short function names to full dotted paths using import statements and common aliases (sc→scanpy, dc→decoupler, etc.)
- `_parse_error_for_lookup()`: Extracts module/attribute/function from error tracebacks
- Uses `asyncio.gather` for parallel sandbox calls when looking up both attribute-level and module-level docs

### 1.6 Session Export/Import (main.py)

**Export** (Two-phase flow, since v0.9.0):
- **Phase 1: Prepare** (`POST /api/session/{session_id}/export/prepare`):
  - Builds tar.gz archive server-side containing: `manifest.json`, `sessions/{id}.json`, `states/{id}*` (.ipynb + .pkl), `uploads/{id}/`, `workspace/{id}/`, `reports/{id}/`
  - Stores archive path in in-memory `_prepared_exports` dict
  - Returns `{"status": "ready", "size_bytes": N, "filename": "...tar.gz"}`
  - Filename derived from sanitized session name + session ID prefix
- **Phase 2: Download** (`GET /api/session/{session_id}/export/download`):
  - Pops prepared export, streams pre-built tar.gz via `FileResponse`
  - Deletes archive from disk after sending (via `BackgroundTask`)
- Frontend shows file size confirmation before initiating download
- **Legacy endpoint** (`GET /api/session/{session_id}/export`): Still available for backward compatibility

**Import** (`POST /api/session/import`):
- Supports tar.gz (async background extraction), zip, and plain JSON formats
- **Async tar.gz import**: Upload is streamed to disk, backend returns `{"import_id", "status": "pending"}` immediately via `BackgroundTasks`. Frontend polls `GET /api/session/import/{import_id}` every 3s until completed/failed. Avoids nginx `proxy_read_timeout` on large archives
- **Proxy timeout resilience**: Frontend pre-generates `import_id` (via `crypto.randomUUID()` with fallback) and sends it as a form field. Backend registers the import job before saving the upload file, so the polling endpoint is available immediately. If the proxy times out before the HTTP response arrives, the frontend detects that upload bytes were fully sent (`uploadFinished` flag) and falls through to polling instead of throwing an error. Polling includes retry logic (up to 5 retries per poll, tolerates transient 404s during initial registration) and a 30-minute overall timeout
- Path remapping: replaces archived `data_base_dir` with current installation's `DATA_BASE_DIR`
- Session ID validation (no path traversal)
- Extracts files (including .pkl for sandbox restore) to correct directories, loads state into memory

### 2. Session Persistence

**Storage Structure:**

```
sessions/{session_id}.json     # Complete AnalysisState
uploads/{session_id}/          # Uploaded data files
states/{session_id}.ipynb      # Jupyter notebook
states/{session_id}_session.pkl # dill-serialized Python environment
```

**Key Functions (main.py):**

```python
async def async_save_session(session_id: str, state: AnalysisState):
    """Serialize state to JSON on the event-loop thread (consistent snapshot),
    then write to disk in a thread pool. Prevents race conditions where
    asyncio.to_thread would read mutable state while the event loop mutates it."""

def save_session_to_disk(session_id: str, state: AnalysisState):
    """Save complete state to disk (synchronous, for legacy callers)"""

def load_session_from_disk(session_id: str) -> Optional[AnalysisState]:
    """Load state from disk"""

def ensure_session_loaded(session_id: str) -> Optional[AnalysisState]:
    """Ensure session is in memory (load from disk if necessary)"""
```

**Save Timing:**

- After each analysis round (via state_callback)
- When WebSocket disconnects
- When analysis completes
- When user pauses

**Memory Management:**

- `MAX_CACHED_SESSIONS = 50`: Maximum sessions kept in memory
- `MAX_REALTIME_MESSAGES = 1000`: Cap per-session realtime messages to prevent memory bloat
- `SESSION_CACHE_TTL = 3600`: Inactive session eviction after 1 hour
- `_IMPORT_JOB_TTL = 3600`: Stale import job cleanup after 1 hour
- Session eviction re-checks activation status, closes Jupyter kernels and chat sandboxes
- Figure message base64 data stripped before persisting to realtime_messages

### 3. Jupyter Code Execution (sandbox.py)

```python
# Centralized code safety validation
DANGEROUS_PATTERNS = [...]  # ~40 patterns for system commands, code execution, file ops, network, etc.

def validate_code_safety(code: str) -> Tuple[bool, str]:
    """Validate code for security before execution. Allows subprocess for pip install."""

def extract_code_from_markdown(text: str, language: str = "python") -> str:
    """Extract code from markdown-formatted LLM response using state-machine parsing."""

class LocalJupyterSandbox:
    async def start(self):
        """Start Jupyter kernel"""

    async def execute(self, code: str, save_to_history=True, skip_safety_check=False) -> CodeExecutionResult:
        """Execute code with safety validation. skip_safety_check=True for trusted internal code."""

    async def stop(self):
        """Stop kernel"""

    def save_state(self, path: str):
        """Save complete Python environment using dill"""

    def load_state(self, path: str):
        """Restore Python environment"""
```

**Features:**

- Local Jupyter kernel, no external service needed
- Supports large memory analysis
- Auto-saves as .ipynb format
- dill serialization supports pause/resume
- Fixed random seed (42) ensures reproducibility
- **Auto PDF generation**: Monkey-patches `Figure.savefig` to automatically generate companion high-res PDFs (300 DPI) alongside every PNG saved. Uses vectorized Type 42 fonts for editable PDFs
- **Code history extraction**: `get_code_history()` returns executed cells from the notebook, used by bioinformatics agent to populate code_history on session restore
- **Centralized code safety**: `DANGEROUS_PATTERNS` and `validate_code_safety()` moved here from bioinformatics agent, enforced at sandbox level so all callers are protected. Internal code (kernel init, session restore/save) uses `skip_safety_check=True`
- **Code extraction**: `extract_code_from_markdown()` shared utility for extracting code from LLM responses, replacing duplicated regex-based extraction across agents

### 4. Report Generator (report_generator.py, ~5460 lines)

**9 Chapters + Cluster Annotation Table:**

| Chapter | Analysis Items |
|---------|---------------|
| Quality Control | QC violin plots, scatter plots, gene distribution, statistics table |
| Cell Clustering | UMAP clusters, groups, sample distribution, dendrogram |
| Cell Annotation | Marker heatmap, dotplot, UMAP, feature plots |
| Cell Proportion | Stacked bar, grouped bar, heatmap |
| Differential Expression | Volcano plots, heatmap, gene lists |
| Pathway Enrichment | GO/KEGG enrichment, GSEA |
| Cell Communication | Communication heatmap, ligand-receptor analysis |
| Trajectory | PAGA graph, trajectory analysis |
| Targeted | Research objective-related analysis |

**Generation Flow (v0.9.0 — No Sandbox, PI Analysis Review):**

Since v0.7.3, the report generator compiles reports from existing analysis artifacts without re-executing code. Since v0.9.0, a PI analysis review pre-step grounds the report in actual notebook content. Since v0.12.1, checkpoint-based resumption allows interrupted reports to continue from the last completed step:

```python
class DetailedReportGenerator:
    async def generate_report(self, state: AnalysisState, resume_from_checkpoint: bool = True):
        # 0. Initialize CheckpointManager and try loading existing checkpoint
        checkpoint_mgr = CheckpointManager(session_id=state.id, state_dir=..., prefix="report")

        # 1. Build structured analysis record from state.code_history
        analysis_record = self._build_analysis_record(state)

        # 2. Scan workspace for existing figures and tables
        existing_figures = self._scan_existing_figures()
        existing_tables = self._scan_existing_tables()

        # 3. Extract findings from round results
        findings = self._extract_findings(state)

        # 3.5 PI reviews analysis — checkpoint after (expensive LLM call)
        analysis_summary = await self._pi_review_analysis(state, ...)

        # 4. PI plans report structure — checkpoint after (expensive LLM call)
        report_plan = await self._plan_report_from_analysis(...)

        # 5. Generate each section — checkpoint after each section
        for section_plan in report_plan["sections"]:
            section = await self._generate_section_from_existing_figures(...)

        # 6. Abstract and conclusions — checkpoint after each
        abstract = await self._generate_abstract(...)
        conclusions = await self._generate_conclusions(...)

        # 7. Build appendix (no LLM — pure data extraction)
        appendix_html = self._build_appendix_html(state, report, language)

        # 8. Clear checkpoint on successful completion
        checkpoint_mgr.clear()
```

**PI Analysis Review (v0.9.0 — Step 3.5):**
- `_read_notebook()`: Reads `.ipynb` file directly via `nbformat`, extracting code, text output, inline figures, and saved figure filenames from every cell. Falls back to `state.code_history` if .ipynb not found
- `_build_notebook_digest()`: Builds compact chronological digest of notebook cells, intelligently truncating long cells via `_truncate_code_smartly()` (retains comments, scanpy calls, savefig/write calls, key assignments)
- `_extract_output_highlights()`: Extracts quantitative highlights from cell outputs (cell counts, gene counts, clusters, p-values, filtering results, AnnData dimensions)
- `_prepare_figure_images()`: Loads up to 20 figures for vision API, prioritized by type (annotation > umap > clustering > heatmap > volcano)
- `_prepare_table_summaries()`: Reads CSV tables with shape, columns, data previews; for DE tables: counts significant genes (|logFC|>1, padj<0.05)
- `_pi_comprehensive_review()`: Single comprehensive PI review of the entire analysis — reads full code digest and all figures together, returns structured JSON (methodology_overview, workflow_summary, key_results, figure_descriptions, parameters_used)
- `_extract_analysis_code()`: Classifies code cells into canonical phases (qc, normalization, integration, clustering, annotation, differential_expression, pathway, trajectory, communication, visualization)

**Report Appendix (v0.9.0):**
- `_build_appendix_html()`: Generates HTML appendix with no LLM calls, containing:
  1. Project Information (name, description, creation time, report generation time)
  2. Agent Configuration (model + temperature for all 7 agents)
  3. Uploaded Data (treatment groups, data files, biochemistry indicator count)
  4. Analysis Rounds (timestamp, per-agent finding counts)
  5. User Inputs Between Rounds (messages with round tags, timestamps, attached files)
- Raw appendix data also saved to `report.json` as `"appendix"` key for regeneration
- `regenerate_report_html.py` includes `_build_appendix_from_report()` to rebuild appendix from stored data

**Key methods:**
- `_build_analysis_record()`: Parses `code_history` to find cells that saved figures, connects code → figures → round findings
- `_plan_report_from_analysis()`: PI plans report structure based on analysis record, available figures, and `analysis_summary`
- `_generate_section_from_existing_figures()`: Writes narrative around existing figures (no code execution)
- `_scan_existing_figures()` / `_scan_existing_tables()`: Scans workspace + reports directories for PNG/PDF/CSV files

**Cluster Annotation Extraction**: `_extract_cluster_annotations_from_state()` extracts cluster annotations from state data (no sandbox needed). Stored in `self.cluster_annotations` dict, rendered as HTML table in report, and included in LLM prompts so narrative uses cell type names instead of cluster numbers.

### 5. LLM Client (llm_client.py, ~980 lines)

**Three client classes:**

```python
class OpenAICompatibleClient:
    """Primary LLM client for OpenRouter/OpenAI-compatible APIs"""
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    async def chat(self, model, messages, temperature=0.7, max_tokens=4096,
                   stream=False, timeout=None) -> LLMResponse:
        # Standard chat with per-request timeout override
    async def chat_with_tools(self, model, messages, tools, ...) -> LLMResponse:
        # Function calling support
    async def chat_stream(self, model, messages, ...) -> AsyncGenerator:
        # Streaming response
    async def chat_with_vision(self, model, text_prompt, images, ...) -> LLMResponse:
        # Vision/multimodal support
    def _extract_content_from_message(self, message, response=None) -> str:
        # Robust content extraction from various response formats

class GoogleDeepResearchClient:
    """Client for Google Gemini Deep Research via Interactions API"""
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    async def start_research(self, query) -> str:
        # Start background research task, returns interaction_id
    async def poll_research(self, interaction_id, poll_interval=5.0, max_wait=1800.0):
        # Poll until completed/failed (30min max)
    async def research(self, query) -> Dict:
        # Convenience: start + poll, returns {report, sources, thinking}

class LLMManager:
    """Manager wrapper, provides .llm / .openrouter aliases"""
    # Note: no longer creates GoogleDeepResearchClient; that is now created
    # lazily inside LiteratureAgent when google_api_key is provided
```

**Key Features:**

- `_model_supports_temperature()`: Module-level function that skips temperature for reasoning models (`o1`/`o3`/`o4` series); these models use `max_completion_tokens` instead of `max_tokens`
- `_extract_content_from_message()`: Robust content extraction handling multiple formats — string content, list content (multimodal), None content with model_dump fallback, reasoning model responses (`reasoning`/`reasoning_content` fields), refusal detection, and debug logging for empty content with non-zero completion tokens
- `_detect_image_mime_type()`: Module-level helper that inspects the first 9 bytes of base64-encoded image data to identify JPEG/PNG/GIF/WebP, falling back to `image/png`. Used by `chat_with_vision()` to send correct `Content-Type` in multimodal requests
- `LLMToolResponse.raw_assistant_message`: Stores the serialized assistant message (via `model_dump()`) so callers can echo it back verbatim. Required for providers such as Gemini that attach extra fields (e.g. `thought_signature`) to tool-call responses and must receive them unmodified in subsequent requests
- `retry_with_backoff()`: Max 3 retries with exponential backoff (2x delay), rate limit handling. Only retries transient errors (`InternalServerError`, timeouts, connection errors); non-transient errors (`AuthenticationError`, `BadRequestError`) fail immediately
- Per-request `timeout` override: Deep research models pass `timeout=660.0` without changing client default (120s)

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | Get all session list |
| `/api/session` | POST | Create new session |
| `/api/session/{id}` | GET | Get session status |
| `/api/session/{id}/info` | PUT | Update session name and/or instruction |
| `/api/session/{id}/load` | GET | Load complete session data |
| `/api/session/{id}/delete` | DELETE | Delete session |
| `/api/session/{id}/upload/data` | POST | Unified data file upload (section + data_type + group_id + sample_number) |
| `/api/session/{id}/groups` | POST | Create treatment group |
| `/api/session/{id}/groups/{gid}` | PUT | Update treatment group |
| `/api/session/{id}/groups/{gid}` | DELETE | Delete treatment group (and associated files) |
| `/api/session/{id}/detailed-report` | GET | Get generated report |
| `/api/session/{id}/generate-detailed-report` | POST | Start report generation |
| `/api/session/{id}/report/download-all-figures` | POST | Download all figures |
| `/api/session/{id}/report/download-all-tables` | POST | Download all tables |
| `/api/session/{id}/report/download-full` | POST | Download complete report |
| `/api/session/{id}/literature-config` | POST | Save literature agent configuration (mode, keys, rounds) |
| `/api/session/{id}/round-wait` | POST | Update round wait countdown seconds (-1=manual, 0=no wait, positive=countdown) |
| `/api/session/{id}/resume` | POST | Resume analysis (paused/completed/waiting_input/failed/generating_report) |
| `/api/session/{id}/chat` | POST | Chat with PI Agent (supports file uploads with per-file agent targeting) |
| `/api/session/{id}/chat` | GET | Get chat history for a session |
| `/api/session/{id}/chat/cancel` | POST | Cancel active chat delegation task |
| `/api/session/{id}/export` | GET/HEAD | Export complete session as tar.gz archive (legacy, browser-native download) |
| `/api/session/{id}/export/prepare` | POST | Build export archive server-side, returns size and filename |
| `/api/session/{id}/export/download` | GET | Download prepared export archive (streaming) |
| `/api/session/import` | POST | Import session from tar.gz/zip/JSON archive (tar.gz returns import_id for async polling) |
| `/api/session/import/{import_id}` | GET | Poll async import job status (pending/completed/failed) |
| `/api/config` | GET/POST | Get/save configuration |
| `/api/download` | GET | Download file by path (with path containment security check) |
| `/api/file` | GET | Get file content |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{session_id}` | Analysis real-time messages |
| `/ws/report/{session_id}` | Report generation progress |

**WebSocket Message Types:**

```typescript
interface WebSocketMessage {
  id: string;           // Unique message ID (for deduplication)
  type: MessageType;    // See table below
  agent?: AgentType;    // Source agent (if applicable)
  content: any;         // Message payload
  timestamp: string;    // ISO8601 timestamp
}

// Message Types
type MessageType =
  // System messages
  | 'status'            // Status update
  | 'state_sync'        // Version control sync
  | 'parallel_errors'   // Structured error list
  | 'completed'         // Analysis complete (with round_results on reconnect)
  | 'message_history'   // Batch message restore (reconnect)
  // Analysis process
  | 'message'           // General message
  | 'progress'          // Progress update
  | 'result'            // Analysis result
  | 'error'             // Error message
  // Results
  | 'round_summary'     // Round summary
  | 'integrated_report' // Integrated report
  | 'graphic_abstract'  // Graphic abstract
  | 'detailed_report'   // Detailed report
  // Figure review
  | 'figure_review'     // PI reviewing figure
  | 'figure_approved'   // Figure approved
  | 'figure_revision'   // Revision requested
  // Cluster annotation
  | 'annotation_result'       // Literature-based cluster annotation result
  | 'annotation_confirmation' // Phase 1 pause — cluster annotations for user review
  | 'exploration_report'      // Phase 1 completion — exploration report + direction selection
  // Chat delegation
  | 'chat_delegation'   // Delegation progress (started/agent_complete/agent_error/synthesizing)
  | 'chat_result'       // Final delegation result
  | 'chat_error'        // Delegation error
  // Supervisor monitoring
  | 'supervisor_feedback'    // Supervisor monitoring feedback
  | 'supervisor_checkpoint'  // Biological checkpoint notification
  | 'supervisor_summary'     // Supervisor round summary
```

**State Sync Message:**

```json
{
  "type": "state_sync",
  "content": {
    "change_type": "round_complete|status_change|error_added",
    "version": 5,
    "status": "running",
    "current_round": 2,
    "timestamp": "2026-02-04T10:30:00Z",
    "unresolved_errors": 0
  }
}
```

**Parallel Errors Message:**

```json
{
  "type": "parallel_errors",
  "content": {
    "errors": [
      {
        "id": "uuid",
        "agent": "pathology",
        "error_type": "execution_error",
        "message": "Image analysis failed",
        "details": "...",
        "round": 1
      }
    ],
    "total_errors": 1,
    "completed_agents": ["bioinformatics", "biochemistry"]
  }
}
```

## Data Models

### AnalysisState (Core State)

```python
class AnalysisState(BaseModel):
    id: str
    name: str
    status: TaskStatus  # pending, running, completed, failed, paused, waiting_input, waiting_annotation, generating_report
    initial_instruction: str

    # Version control
    state_version: int = 0
    last_sync_timestamp: datetime

    # Treatment groups and uploaded data (multi-group, multi-sample)
    treatment_groups: List[TreatmentGroup]       # Experimental group definitions
    data_files: List[DataFile]                   # All uploaded files (pathology, sequencing, biochemistry)
    biochemistry_indicators: List[BiochemistryIndicator]  # Manual biochemistry entries

    # Analysis results
    round_results: List[RoundResult]
    messages: List[AgentMessage]
    code_history: List[CodeExecution]
    final_summary: Optional[str]

    # Publication outputs
    publication_figures: List[PublicationFigure]
    publication_tables: List[PublicationTable]
    graphic_abstract: Optional[GraphicAbstract]
    integrated_report: Optional[str]

    # Error tracking
    agent_errors: List[AgentError]

    # Human-in-the-loop
    pending_human_input: Optional[str]
    pending_uploaded_files: List[Dict]
    max_rounds: int = 0
    round_wait_seconds: int = 300

    # Two-phase workflow state (v0.13.0+)
    analysis_phase: str = "exploration"  # "exploration" (Round 1) / "innovation" (Round 2+)
    annotation_state: Optional[AnnotationConfirmationState] = None  # Cell annotation confirmation state
    exploration_report: Optional[ExplorationReport] = None  # Phase 1 structured report
    exploration_report_html_path: Optional[str] = None  # Path to exploration report HTML (v0.13.1)
    user_chosen_direction: Optional[str] = None  # User's chosen research direction (backward compat)
    current_round_direction: str = ""  # Active direction for current innovation round

    # Multi-direction workflow (v0.13.1)
    user_chosen_directions: List[Dict[str, str]] = []  # All selected directions: [{"direction": str, "notes": str}]
    directions_queue: List[Dict[str, str]] = []         # Remaining directions to process (FIFO)
    multi_direction_correlation: Optional[str] = None   # Correlation analysis result (after all directions complete)

    # Literature config (per-session)
    literature_config: Dict[str, Any]

    # Chat messages (PI Agent free-form chat)
    chat_messages: List[Dict[str, Any]] = []

    # Configuration
    custom_prompts: Dict[str, str]
    custom_models: Dict[str, str]
    custom_temperatures: Dict[str, float]

    # Data helper methods
    def get_files_by_section(section: str) -> List[DataFile]
    def get_files_by_group(group_id, section=None) -> List[DataFile]
    def has_sequencing_data() -> bool
    def get_group_by_id(group_id) -> Optional[TreatmentGroup]
    def build_experimental_design_summary() -> str
    def increment_version()
    def add_error(agent, error_type, message, details) -> AgentError
    def get_unresolved_errors() -> List[AgentError]
    def resolve_error(error_id, resolution)
```

### TreatmentGroup

```python
class TreatmentGroup(BaseModel):
    id: str
    name: str                          # e.g., "Control", "NASH Model", "Treatment A"
    description: str = ""
    sample_count: int = 1              # Number of samples/replicates
    order: int = 0                     # Display order
```

### DataFile (Universal File Record)

```python
class DataFile(BaseModel):
    id: str
    filename: str
    file_path: str                     # Absolute path on disk
    section: Literal["pathology", "sequencing", "biochemistry"]
    data_type: str                     # pathology: "H&E"/"Other"/custom; sequencing: "scRNA-seq"/"loom"; biochemistry: "image"/"file"
    group_id: Optional[str] = None     # TreatmentGroup.id
    sample_number: Optional[int] = None  # 1-based sample number
    tiles: List[str] = []              # For pathology: tile paths
    analysis: Optional[str] = None     # For pathology: analysis result
    description: Optional[str] = None
    extracted_data: Optional[Dict] = None  # For biochemistry: extracted values
    timestamp: datetime
```

### BiochemistryIndicator

```python
class BiochemistryIndicator(BaseModel):
    id: str
    indicator: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None
    group_id: Optional[str] = None
    sample_number: Optional[int] = None
    timestamp: datetime
```

### AgentError

```python
class AgentError(BaseModel):
    id: str                      # Unique ID
    agent: AgentType             # Which agent failed
    error_type: str              # execution_error, timeout, data_missing, llm_error
    message: str                 # Error message (truncated to 200 chars)
    details: Optional[str]       # Full error details
    round_number: int            # Which round
    timestamp: datetime
    resolved: bool = False       # Resolution status
    resolution: Optional[str]    # How it was resolved
```

### RoundResult (Single Round Result)

```python
class RoundResult(BaseModel):
    round_number: int
    phase: str = "exploration"  # "exploration" or "innovation" (v0.13.0)
    exploration_report: Optional[ExplorationReport] = None  # Only present for Phase 1 round result
    pathology_findings: List[str]
    bioinformatics_findings: List[str]
    biochemistry_findings: List[str]
    literature_findings: List[str]
    pi_summary: str
    next_steps: List[str]
    conflicts: List[Dict[str, Any]]
    new_findings: bool = True    # For convergence detection
    agents_for_next_round: List[str]  # PI decides which agents to activate next round
    timestamp: datetime
```

### GraphicAbstract

```python
class GraphicAbstract(BaseModel):
    id: str
    prompt: str           # Generation prompt
    image_base64: str     # Base64 encoded image (PI-selected best candidate)
    revision_count: int = 0            # Number of revision rounds performed
    revision_history: List[Dict[str, Any]] = []  # PI feedback per revision round
    all_candidates: List[str] = []     # All candidate images (base64)
    selected_index: int = 0            # Index of PI-selected best candidate
    created_at: datetime
```

### Two-Phase Workflow Models (v0.13.0)

```python
class ClusterAnnotation(BaseModel):
    """Single cluster annotation for user review during exploration phase"""
    cluster_id: str
    cell_type: str
    subtypes: List[str]
    key_markers: List[str]
    confidence: str        # "high" / "medium" / "low"
    evidence: str
    user_confirmed: bool = False
    user_override: Optional[str] = None  # User-provided cell type override

class AnnotationConfirmationState(BaseModel):
    """State for cell annotation confirmation pause during exploration phase"""
    cluster_annotations: List[ClusterAnnotation]
    tissue: str
    organism: str
    tissue_summary: str
    pending: bool = False
    confirmed: bool = False

class ExplorationReport(BaseModel):
    """Structured exploration report generated after Phase 1 (Round 1)"""
    cluster_overview: str
    data_quality_warnings: List[str]
    strongest_heterogeneity: List[str]
    key_pathway_changes: List[str]
    abnormal_markers: List[str]
    low_confidence_annotations: List[str]
    pi_direction_assessment: str
    research_direction_candidates: List[Dict[str, Any]]
    # Each candidate: {"direction": str, "rationale": str, "expected_innovation": str,
    #                  "expected_novelty": str, "data_reliability": str}
```

## Development Guide

### Environment Setup

**Recommended (using startup script):**

```bash
# Startup script automatically completes all environment configuration
./start.sh
```

The script will automatically:
1. Create Python virtual environment at `../dep/venv/`
2. Install frontend dependencies at `../dep/node_modules/`
3. Create data directories at `../data/`
4. Start frontend and backend services

**Manual setup (if needed):**

```bash
# Backend (dependencies installed to ../dep/venv)
python -m venv ../dep/venv
source ../dep/venv/bin/activate
pip install -r requirements.txt

# Frontend (dependencies installed to ../dep/node_modules)
mkdir -p ../dep/node_modules
cd frontend
ln -s ../../dep/node_modules node_modules
npm install
```

### Local Development

```bash
# Backend (hot reload)
uvicorn backend.main:app --reload --port 8000

# Frontend (hot reload)
cd frontend
npm run dev
```

### Adding New Agents

1. Create new Agent file in `backend/agents/`
2. Inherit from `BaseAgent` class
3. Implement `analyze()` method
4. Register new Agent in `workflow.py`
5. Update `AgentType` enum in `schemas.py`

### Adding New Report Sections

1. Add section definition in `REPORT_SECTIONS` in `report_generator.py`
2. Implement corresponding analysis item generation logic
3. Update frontend `ReportPanel.tsx` to display new section

### Debugging Tips

```python
# View LLM interaction logs
# Logs located in data/logs/ directory

# Debug Jupyter execution
# View states/{session_id}.ipynb

# Debug Session state
# View sessions/{session_id}.json
```

## Operations Guide

### Starting Services

```bash
# Using script
./start.sh

# Or manual startup
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
cd frontend && npm run build && npx serve -s dist -l 3000 &
```

### Stopping Services

```bash
./stop.sh
```

### Docker Image Publishing

To push to GitHub Container Registry:

```bash
# Login (requires a PAT with write:packages permission)
echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Push
docker push ghcr.io/minzhangcheng/scai:latest
```

### Data Backup

**Directories to back up** (located in parent of code directory):

```
data/                # Data directory (must backup)
├── sessions/               # All session data (session-centric layout since v0.12.4)
│   └── {session_id}/       # Each session's workspace, reports, uploads, states, logs
├── logs/                   # Global system logs
└── upload_temp/            # Temporary upload chunks (can skip)

dep/                 # Dependencies directory (optional backup, can reinstall)
├── venv/                   # Python virtual environment
└── node_modules/           # Node.js dependencies
```

**Backup Recommendations:**

- `data/` directory contains user data and analysis results, must backup
- `dep/` directory can be skipped, reinstall via `./start.sh`
- Code directory `scai/` managed via git

### Log Locations

```
data/logs/   # LLM interaction logs
uvicorn stdout      # API logs
```

### Configuration Items

**config.py main configurations:**

| Config | Default | Description |
|--------|---------|-------------|
| `max_rounds` | 3 | Max analysis rounds |
| `convergence_rounds` | 2 | Consecutive rounds without new findings |
| `sandbox_timeout` | 300 | Code execution timeout (seconds) |
| `image_tile_size` | 1024 | Pathology image tile size |
| `image_overlap` | 64 | Tile overlap pixels |
| `blank_threshold` | 0.95 | Threshold for detecting blank tiles |
| `round_wait_seconds` | 300 | Countdown between rounds (-1=manual, 0=no wait, positive=seconds) |

**Literature Agent Defaults:**

| Config | Default | Description |
|--------|---------|-------------|
| `literature_mode` | `pubmed_llm` | Research mode (`gemini_deep_research` or `pubmed_llm`) |
| `max_search_rounds` | 5 | Max iterative search rounds (1-20) |

**LLM Model Defaults:**

| Agent | Default Model |
|-------|--------------|
| PI | `google/gemini-3.1-pro-preview` |
| Pathology | `google/gemini-3.1-pro-preview` |
| Bioinformatics | `anthropic/claude-sonnet-4.6` |
| Biochemistry | `google/gemini-3.1-pro-preview` |
| Painter | `google/gemini-3-pro-image-preview` |
| Literature | `google/gemini-3-flash-preview` (chat); Deep Research via mode config |
| Supervisor | `google/gemini-3.1-flash-lite-preview` |

**Agent Temperatures:**

| Agent | Default | Rationale |
|-------|---------|-----------|
| PI | 0.5 | Balanced creativity for synthesis |
| Pathology | 0.2 | Objective, consistent analysis |
| Bioinformatics | 0.1 | Highly deterministic code generation |
| Biochemistry | 0.2 | Objective interpretation |
| Literature | 0.6 | Broader exploration |
| Painter | 0.7 | Creative image generation |
| Supervisor | 0.1 | High-certainty evaluation |

### Common Issues

**1. Session Load Failure**

Check if `sessions/{id}.json` file exists and is properly formatted.

**2. Report Generation Failure**

- Check if data file paths are correct
- Check backend logs for specific errors
- Confirm Jupyter kernel starts normally

**3. WebSocket Disconnection**

- Frontend implements auto-reconnect with exponential backoff (1s, 2s, 4s... max 30s)
- Message history is automatically restored on reconnection
- Check network connection and proxy settings

**4. Code Execution Failure**

- Check `states/{id}.ipynb` for execution history
- Confirm required Python packages are installed

**5. State Version Conflicts**

- If frontend shows outdated data, check `state_version` in session JSON
- Frontend detects conflicts when received version < current version
- Resolution: Trigger full session reload via `loadSession()`

**6. Parallel Analysis Errors**

- Errors are tracked per-agent in `agent_errors` list
- Check `AnalysisPanel` for error display
- Errors persist until manually resolved
- Debug via: `sessions/{id}.json` → `agent_errors` array

**7. Vision API Empty Responses**

- Gemini via OpenRouter may return empty content with high token count
- Affects pathology and biochemistry image analysis
- Biochemistry Agent v2 mitigates this with comprehensive prompts
- Check `data/logs/error.log` for details

**8. Stale Session After Server Restart**

- If browser shows blank/error state after server restart, the session may no longer exist on server
- v0.6.1+ auto-detects this and clears stale session from localStorage
- Manual fix: Clear localStorage in browser DevTools → Application → Local Storage → delete `scai-storage`

**9. Deep Research Timeout**

- Gemini Deep Research via Google Interactions API can take up to 30 minutes per request
- v0.6.1+ uses per-request `timeout=660s` for deep research calls (vs default 120s)
- Since v0.7.3, Deep Research is single-call per topic (DR handles multi-round internally). Gap coverage is handled at the workflow level by falling back to PubMed+LLM
- If timeout errors persist, check network/proxy connection and Google API service status

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.109.0 | Web framework |
| langgraph | >=0.0.40 | Workflow orchestration |
| openai | >=1.12.0 | LLM API (OpenAI-compatible) |
| pydantic | >=2.5.0 | Data models |
| jupyter_client | 8.6.0 | Code execution |
| dill | >=0.3.7 | State serialization |

### Bioinformatics Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scanpy | >=1.9.6 | Single-cell analysis core |
| anndata | >=0.10.3 | Data structures |
| muon | >=0.1.5 | Multi-omics data |
| liana | >=1.0.0 | Cell communication analysis |
| scvelo | >=0.3.1 | RNA velocity |
| cellrank | >=2.0.0 | Cell fate analysis |
| decoupler | >=1.5.0 | Pathway activity |
| gseapy | >=1.1.0 | GO/KEGG enrichment |
| celltypist | >=1.6.0 | Cell annotation |
| scvi-tools | >=1.0.0 | Deep learning models |
| pySCENIC | >=0.12.0 | Gene regulation |

### Frontend Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| react | ^18.2.0 | UI framework |
| typescript | ^5.3.3 | Type safety |
| zustand | ^4.5.0 | State management |
| vite | ^5.0.12 | Build tool |
| tailwindcss | ^3.4.1 | Styling |
| recharts | ^2.10.4 | Charts |
| react-markdown | ^9.0.1 | Markdown rendering |
| react-syntax-highlighter | ^15.5.0 | Code highlighting |
| react-dropzone | ^14.2.3 | File upload |

## Key Code Locations

### Backend

| Feature | File | Approximate Size |
|---------|------|-----------------|
| FastAPI server, session mgmt, chat, delegation, export/import & session editing | `main.py` | ~5260 lines |
| Report generator (no-sandbox, PI analysis review, appendix) | `report_generator.py` | ~5460 lines |
| Workflow graph, orchestration, DR optimization, selective activation & GA revision loop | `workflow.py` | ~3270 lines |
| Supervisor Agent (real-time step monitoring + PI escalation + detailed report generation) | `supervisor_agent.py` | ~2645 lines |
| Bioinformatics Agent (autonomous + figure budget + pip install + vision revision + cumulative history) | `bioinformatics_agent.py` | ~2396 lines |
| PI Agent (coordinator + reviewer + GA reviewer + agent activation + data fidelity + brief figure curation) | `pi_agent.py` | ~2353 lines |
| Report HTML regeneration (appendix + analysis log support) | `regenerate_report_html.py` | ~1600 lines |
| Literature Agent (2 modes + annotation) | `literature_agent.py` | ~1270 lines |
| Jupyter sandbox (+ auto PDF + I/O sandbox) | `sandbox.py` | ~1316 lines |
| LLM client (OpenRouter + Google DR + MIME detection + precise retry) | `llm_client.py` | ~980 lines |
| Biochemistry Agent (3-step v2) | `biochemistry_agent.py` | ~785 lines |
| Painter Agent (graphic abstract + revision + dual-path image generation) | `painter_agent.py` | ~647 lines |
| Data models & schemas | `schemas.py` | ~642 lines |
| Pathology Agent (batch size 5 + dynamic timeout) | `pathology_agent.py` | ~460 lines |
| Chunked upload manager | `upload.py` | ~400 lines |
| Configuration & prompts (+ supervisor/painter/technique prompt fields) | `config.py` | ~364 lines |
| Technique Agent (API doc lookup via sandbox introspection + LLM) | `technique_agent.py` | ~342 lines |
| Agent base class | `base.py` | ~299 lines |
| Error type extraction utility | `error_parsing.py` | ~19 lines |

### Frontend

| Feature | File | Approximate Size |
|---------|------|-----------------|
| API hooks + WebSocket + session editing | `useApi.ts` | ~1230 lines |
| Data upload (queue + per-file progress) + download interface | `DataPanel.tsx` | ~1098 lines |
| Analysis control + agent status | `AnalysisPanel.tsx` | ~1072 lines |
| Real-time message streaming + delegation | `MessageStream.tsx` | ~963 lines |
| Agent prompt + literature config + per-agent LLM server | `PromptsPanel.tsx` | ~960 lines |
| Report generation & download | `ReportPanel.tsx` | ~945 lines |
| Session mgmt + export/import + session editing | `Sidebar.tsx` | ~759 lines |
| PI Agent chat + delegation | `ChatPanel.tsx` | ~674 lines |
| Round feedback dialog (configurable timer) | `RoundFeedbackDialog.tsx` | ~634 lines |
| Zustand global state (+ delegation) | `appStore.ts` | ~615 lines |
| Exploration report & direction selection dialog | `ExplorationReportDialog.tsx` | ~486 lines |
| Cell annotation confirmation dialog (Phase 1) | `AnnotationConfirmationDialog.tsx` | ~386 lines |
| Round results + publication outputs | `ResultsPanel.tsx` | ~372 lines |
| Main application (+ About dialog) | `App.tsx` | ~347 lines |
| API + literature configuration + DR warning | `ConfigPanel.tsx` | ~210 lines |
| About dialog (project info) | `AboutDialog.tsx` | ~108 lines |

## Known Issues

### 1. Vision API Empty Content

**Symptom**: Gemini via OpenRouter sometimes returns empty content despite high completion token count (e.g., 4093 tokens but content=None).

**Affected Agents**: Pathology, Biochemistry (image analysis steps)

**Root Cause**: Likely OpenRouter relay or response format mismatch with Gemini vision API.

**Workaround**: Biochemistry Agent v2 uses comprehensive prompts instead of strict JSON extraction for more robustness. Retry mechanisms with exponential backoff help mitigate intermittent failures.

### 2. Large Memory Usage

Single-cell datasets can be very large. The local Jupyter kernel may require significant memory (8GB+) for larger datasets. dill serialization of the full Python environment can also produce large `.pkl` files.

### 3. localStorage Overflow (Fixed in v0.6.1)

**Symptom**: Page refresh fails to load session; browser console shows `QuotaExceededError`.

**Root Cause**: Zustand persist stores state to localStorage (5MB limit). Large sessions with many base64 images overflow.

**Fix**: Base64 data is now stripped from persisted messages and session objects in `appStore.ts` partialize function. API response also strips base64 via `_strip_message_base64()` in `main.py`.

### 4. Figure Gallery Preview Loss (Fixed in v0.7.4)

**Symptom**: Figure Gallery shows grey placeholder icons instead of figure previews after page reload.

**Root Cause**: `appStore.ts` partialize function strips `pngBase64` from both `currentSession.publicationFigures` and `integrated_report` messages for localStorage. On page reload, `ResultsPanel.tsx` useEffect first sets figures from `currentSession.publicationFigures` (fresh from `/load` API with base64), then processes the `integrated_report` message from localStorage (with stripped base64), overwriting the good data.

**Fix**: `integrated_report` message handler now uses functional update to merge figures, preserving `pngBase64` from existing data when the message's figure data is missing it.

### 5. Figure Hallucination of Non-Existent Treatment Groups (Fixed in v0.8.1)

**Symptom**: Publication figures (e.g., figure_8) show 4 genotype groups (WT, Ctr1 KO, ZnT1 KO, DKO) when only 2 groups (WT, DKO) were uploaded. Figures contain fabricated data.

**Root Cause**: Three cascading failures: (1) Bioinformatics agent's LLM inferred extra groups from the biological context in `initial_instruction` (which mentioned Ctr1 and ZnT1 genes); (2) PI figure review only received `initial_instruction[:500]` without actual treatment group info, so it reinforced the hallucination by suggesting "add Ctr1 KO columns"; (3) `_revise_figure_code()` blindly executed PI feedback without experimental design context, and couldn't see the figure (text-only `chat()`).

**Fix**: (1) Added `CRITICAL DATA CONSTRAINT` with actual group names to `_build_data_context()`, `_revise_figure_code()` prompt, and `generate_publication_figure` tool description; (2) PI `review_figure()` now includes `build_experimental_design_summary()` and Data Fidelity criterion; (3) `_revise_figure_code()` now uses `chat_with_vision()` to see the current figure image during revision.

### 6. Stale Session 404 Loop (Fixed in v0.6.1)

**Symptom**: After server restart or session cleanup, frontend enters infinite 404 + WebSocket reconnect loop.

**Root Cause**: Browser localStorage has a session ID that no longer exists on server. Every API call returns 404, WebSocket gets "Session not found" error, reconnect fires, repeat.

**Fix**: Added stale session detection in 3 places: Sidebar auto-reload, WebSocket `onmessage`, and WebSocket `onopen` reconnect. On 404 detection, clears session from Zustand state.

## Version History

#### v0.18.13 (2026-04-03)

- **Docker PyPI mirror support** (`docker-entrypoint.sh`): Docker startup now supports `-e PIP_INDEX_URL` parameter to pass a PyPI mirror URL (e.g., Aliyun, Tsinghua). When set, pip install uses the specified mirror for faster package installation in containers.

#### v0.18.12 (2026-04-03)

- **Bioinformatics total error hard limit** (`agents/bioinformatics_agent.py`): Added `total_errors` counter with `MAX_TOTAL_ERRORS = 30` hard limit. Forces `report_findings` after 30 total code errors, preventing infinite error loops. Added try/except around matplotlib savefig restore to avoid masking real user code errors. Added `raw_response` preview in annotation failure messages for debugging.
- **Painter image API response handling** (`agents/painter_agent.py`): Removed redundant `n=1` from direct generation call. Added debug logging for image API raw responses. Added `content_filter` detection for content moderation interception. Better handling of SiliconFlow (`images` array) vs standard Zhipu (`data` array) response formats.
- **PI annotation JSON retry + reanalysis agents** (`agents/pi_agent.py`): Added retry logic for annotation JSON parsing with a more explicit prompt on first failure. Added `reanalysis_agents: List[str]` field to `RoundResult` schema (`models/schemas.py`) for re-analysis of existing data with different parameters/perspectives. Updated PI prompt documentation to clarify distinction between `agents_for_next_round` (new data processing) vs `reanalysis_agents` (existing data re-analysis).
- **WebSocket observer disconnect handling** (`backend/main.py`): Added `WebSocketDisconnect` exception handling for observer ping/receive loops, preventing spurious error logs when observers disconnect.
- **JSON repair for LLM responses** (`utils/formatting.py`): Added optional `json_repair` import. `safe_parse_json()` now falls back to `repair_json()` when standard JSON parsing fails, handling malformed JSON from LLM responses.
- **Sandbox savefig redirect recovery** (`utils/sandbox.py`): Improved error handling in savefig redirect with recovery mechanism; original path is tried as last resort if redirect fails.
- **Selective agent activation from round 2+** (`workflow.py`): Track biochemistry data in `new_data_files`. From round 2 onwards, pathology and biochemistry agents are disabled by default unless new data arrives or PI explicitly requests reanalysis via `reanalysis_agents`.
- **Dev/prod mode for startup scripts** (`docker-entrypoint.sh`, `start.sh`): Added `--dev`/`--prod` mode flags. Production mode builds the frontend and uses `vite preview`. Development mode uses `vite dev`. Default is production mode.
- **DataPanel type annotation** (`frontend/src/components/DataPanel.tsx`): Changed `typeOptions` prop type from `string[]` to `readonly string[]` for better TypeScript compatibility.
- **New dependency: json-repair** (`requirements.txt`): Added `json-repair>=0.30,<1.0` for repairing malformed JSON from LLM responses.

#### v0.18.11 (2026-04-02)

- **Literature agent main model retry** (`agents/literature_agent.py`): Added main model retry logic with `_MAX_MAIN_RETRIES = 2` and `_RETRY_DELAY_BASE = 2.0`. `_chat_with_fallback()` now retries main model failures before falling back, with special handling for RateLimitError to skip directly to fallback. Fallback error chaining now preserves the original exception context.
- **Subprocess pip install in sandbox** (`utils/sandbox.py`): Improved `_is_safe_subprocess()` with better multiline regex matching for subprocess calls. Combined Figure.savefig monkey-patch with PNG→JPG auto-conversion and workspace root path redirect to prevent writes outside session directory.
- **WebSocket timeout increased** (`config.py`): `WS_RECEIVE_TIMEOUT` increased from 30.0s to 60.0s to handle longer-running operations.
- **Graph recursion limit** (`workflow.py`): Added `GRAPH_RECURSION_LIMIT = 100` constant and applied it to workflow config in both `run()` and `resume()` methods to prevent infinite recursion in LangGraph.
- **Bioinformatics subprocess pip install** (`agents/bioinformatics_agent.py`): Updated system prompts to use `subprocess.run(['pip', 'install', 'package_name'])` instead of `!pip install` magic command, aligning with sandbox security policy.

#### v0.18.10 (2026-04-01)

- **LLM temperature handling for kimi-k2.5** (`llm_client.py`): Added `_NO_CUSTOM_TEMP_MODELS = {"kimi-k2.5"}` set for models that prohibit custom temperature. `_get_effective_temperature()` now returns `None` for these models, omitting the temperature parameter entirely instead of causing BadRequestError.
- **max_tokens for o-series models** (`llm_client.py`): Fixed `_prepare_request_params()` to correctly use `max_completion_tokens` (not `max_tokens`) for OpenAI o-series reasoning models when temperature is skipped.
- **Zhipu AI support** (`agents/painter_agent.py`): Added `_ZHIPU_BASE_URL = "open.bigmodel.cn"` constant. `_is_zhipu` detection skips chat modalities strategy (same as Google AI Studio), going directly to the Images API. `_skip_chat_modalities = _is_google_ai_studio or _is_zhipu` consolidates both providers.
- **Literature fallback LLM client** (`agents/literature_agent.py`): Added `fallback_llm_client: Optional["OpenAICompatibleClient"]` parameter to `__init__()`. When a fallback model is specified, the quick search now uses this client instead of the agent's own client, ensuring the correct LLM endpoint is used.
- **Supervisor report LLM client** (`agents/supervisor_agent.py`): Added `report_llm_client: Optional["OpenAICompatibleClient"]` parameter to `generate_detailed_report()`. Uses the provided client (matching report_model's endpoint) instead of the supervisor's own client.
- **Agent LLM client caching** (`workflow.py`): Added `_agent_client_cache` dict to cache per-agent LLM clients. `_get_agent_llm_client()` now checks the cache before creating new clients, avoiding duplicate client creation on repeated calls.
- **Workflow LLM client fixes** (`workflow.py`): `create_agents()` now passes `fallback_llm_client=self._get_agent_llm_client('pi')` to LiteratureAgent, and `report_llm_client=self._get_agent_llm_client('pi')` to supervisor's `generate_detailed_report()`, ensuring correct endpoints are used.

#### v0.18.8 (2026-03-31)

- **LLM temperature auto-fallback** (`llm_client.py`): New `_call_with_temp_fallback()` method automatically retries requests without temperature if the model rejects it via BadRequestError. `_temp_unsupported_models` set caches such models to avoid repeated failures. Refactored `_build_request_pair()` and `_build_retry_kwargs()` to unify request construction across all chat methods.
- **Technique Agent message callback** (`technique_agent.py`): Added `message_callback: Optional[Callable[[str, str], Awaitable[None]]]` parameter to `__init__()`. `_notify()` helper sends status/result messages during doc extraction and error analysis stages.
- **Round counter guard on snapshot restore** (`workflow.py`): `start_new_round()` now skips `current_round += 1` when `last_restored_snapshot_id` is set, preventing double-counting rounds after resuming from snapshot.
- **Agent session initialization** (`workflow.py`): Added `set_session(self.session_id)` calls to PI, Bioinformatics, Pathology, Biochemistry, Literature, and Technique agents. Added `_technique_message_callback()` bridge for WebSocket notifications from Technique Agent.

#### v0.18.7 (2026-03-31)

- **Agent LLM Statistics** (`regenerate_report_html.py`): New `get_agent_llm_stats(session_id)` function reads `llm.jsonl` and aggregates per-agent statistics (llm_server, model, temperature, call_count, input_tokens, output_tokens, avg_response_time_ms). New `build_agent_stats_html(stats)` and `get_agent_stats_html(session_id)` functions render the data as an HTML table section.
- **Supervisor report integration** (`supervisor_agent.py`): `generate_html_report()` now calls `get_agent_stats_html(self.session_id)` and injects `{agent_stats_html}` into the report template.
- **Report generator integration** (`report_generator.py`): `generate_html_report()` now calls `get_agent_stats_html(state.id)` and injects `{agent_stats_html}` into the report template.
- **LLM logger base_url field** (`llm_logger.py`): `LLMInteraction` gains `base_url: Optional[str] = None` field; `start_interaction()` accepts and records the `base_url` parameter.
- **LLM client max_tokens default None** (`llm_client.py`): `send_message()`, `send_message_with_tools()`, `stream_chat()`, and `send_message_with_images()` all changed `max_tokens` default from `32768` to `None`. All call sites now pass `base_url=self.base_url` to `start_interaction()`.

#### v0.18.6 (2026-03-30)

- **Annotation snapshot timing fix** (`workflow.py`): `_create_auto_snapshot(state, "pre_annotation")` moved after `state.annotation_state = ann_state`, so rollback restores the `AnnotationConfirmationState` dialog data the frontend requires.
- **Pre-annotation snapshot backward compat** (`main.py`): `restore_snapshot()` now detects `pre_annotation` phase snapshots lacking `annotation_state` and falls back to `TaskStatus.PAUSED`, preventing the workflow from hanging in HITL-hold with no dialog data.
- **PI Agent constructor fix** (`workflow.py`): Removed `session_id=self.session_id` keyword from `PIAgent()` construction — `PIAgent.__init__` does not accept this parameter.
- **Multimodal text-first ordering** (`utils/llm_client.py`): `send_message_with_images()` now appends the text content block before image blocks (previously images came first), following OpenRouter's recommended ordering.
- **Painter Google AI Studio constant** (`agents/painter_agent.py`): Extracted `_GOOGLE_AI_STUDIO_BASE_URL = "generativelanguage.googleapis.com"` module-level constant. `_generate_image()` uses it to skip Strategy 1 (chat modalities) entirely for Google AI Studio, avoiding a guaranteed-fail request.
- **Supervisor error tail tracking** (`agents/supervisor_agent.py`): `_recent_errors.append(step.code_error[-300:])` replaces `[:200]` — Python tracebacks place the error type at the end, so the tail is more useful for stuck detection than the head.
- **Deep Research configurable model** (`utils/llm_client.py`): `GoogleDeepResearchClient.__init__` gains `model: Optional[str] = None` parameter; defaults to `DEFAULT_MODEL = "deep-research-pro-preview-12-2025"`. The `start_research()` payload uses `self.model` instead of a hardcoded string.

#### v0.18.5 (2026-03-30)

- **`safe_parse_json` shared utility** (`utils/formatting.py`): New `safe_parse_json(response, default=None)` function consolidates the balanced-bracket JSON extraction logic previously duplicated in `BaseAgent._safe_parse_json` and `report_generator._safe_parse_json`. Both callers now delegate to this single implementation. ~60 lines of duplicate code removed.
- **Supervisor stuck detection broadened** (`agents/supervisor_agent.py`): `is_stuck` now also fires when the same error type appears 3+ times across `_recent_errors` (not only when the last two errors match), allowing PI escalation for non-consecutive repeated errors.
- **Supervisor PI escalation on stuck technical errors** (`agents/supervisor_agent.py`): The Layer 1 (exception-type) and Layer 2 (LLM-classified) technical-error PI-escalation skips now include `and not is_stuck`, so a frozen agent causing repeated technical errors is still escalated to PI.
- **Supervisor null response guard** (`agents/supervisor_agent.py`): `response.content or ""` changed to `response.content if response else ""` in both `_evaluate_step()` call sites, preventing AttributeError when the LLM client returns None.
- **Biochemistry invalid value skip** (`agents/biochemistry_agent.py`): Non-finite float values (NaN, Inf) from LLM responses now `continue` (skip the entry) instead of being silently replaced with 0.0, preventing corrupt indicator records.
- **Annotation cache size cap** (`agents/literature_agent.py`): `_ANNOTATION_CACHE_MAX = 100` added; when `_annotation_cache` reaches the limit, the oldest key is evicted before inserting a new entry, bounding memory use in long sessions.
- **DashScope painter revision fix** (`agents/painter_agent.py`): `revise_image()` now detects `_DASHSCOPE_BASE_URL` and calls `_generate_image_dashscope()` with a text-only prompt (stripped of the "look at attached image" instruction), matching the native API's text-only contract.
- **`review_status` None guard in serialization** (`models/schemas.py`): `AnalysisState.to_api_dict()` wraps `_enum(fig.review_status)` / `_enum(tbl.review_status)` with `if … else None`, preventing AttributeError when `review_status` is None.
- **WebSocket reconnect deduplication** (`frontend/src/App.tsx`): Removed the competing auto-reconnect `useEffect` (exponential backoff loop) from `App.tsx`; reconnection is now handled solely by the `useWebSocket` hook, eliminating race conditions between two independent reconnect loops. `devLog` import and `chatDelegationActive`/`connected` destructuring removed accordingly.
- **WebSocket reconnect timer cleanup** (`frontend/src/hooks/useApi.ts`): `reconnectTimerRef` tracks the pending `setTimeout` handle; `disconnect()` cancels it, preventing ghost reconnect attempts after manual disconnection.
- **Persisted `chatMessages` TTL eviction** (`frontend/src/stores/appStore.ts`): `chatMessages` is now cleared alongside `messages` and `currentSession` when the persistence TTL expires, preventing stale chat history from surfacing after a browser refresh.

#### v0.18.4 (2026-03-30)

- **DashScope image generation** (`agents/painter_agent.py`): Added `_DASHSCOPE_BASE_URL` constant and `_generate_image_dashscope()` method. The method submits a task via `POST /api/v1/services/aigc/text2image/image-synthesis` (with `X-DashScope-Async: enable`), polls `GET /api/v1/tasks/{task_id}` every 5 s (up to 60 attempts), downloads the result URL, and returns base64. `_generate_image()` gains Strategy 0: if `_DASHSCOPE_BASE_URL` is in `self.llm_client.base_url`, delegate to the native API and skip all OpenAI-compat strategies.
- **Snapshot pkl backup** (`utils/snapshot.py`): `create_snapshot()` is now `async`; copies `{session_id}_session.pkl` from `session_states/` to `snapshots/{snapshot_id}_session.pkl` via `asyncio.to_thread(shutil.copy2)`. `restore_snapshot()` reverses the copy. `delete_snapshot()` and `cleanup_old_snapshots()` unlink pkl backups. `get_session_states` import moved to module level.
- **Session export includes snapshots** (`main.py`): New `_add_snapshots_to_tar(tf, session_id)` helper iterates `snapshots/` and adds all files to the tar archive under `{session_id}/snapshots/`. Called as step 6 in `export_state`.
- **Vision auto-disable** (`agents/bioinformatics_agent.py`, `agents/pi_agent.py`): Bioinformatics Agent broadens vision-unsupported detection to include ZhipuAI error code "1210" and sets `self.has_vision = False` on any vision failure. PI Agent similarly sets `self.has_vision = False` when vision call fails.
- **Pre-round-wait snapshot + kernel save** (`workflow.py`): `_create_auto_snapshot()` calls `await self.sandbox.save_session()` before snapshotting. New `pre_round_wait` snapshot phase created before the round-wait HITL loop.
- **Supplementary analysis in wait loop** (`workflow.py`): `_run_round_wait_loop()` handles `_supplementary_request` with `_supplementary_target == "round"`, executes supplementary analysis inline, then resumes `WAITING_INPUT`.
- **Multi-direction auto-skip removed** (`workflow.py`): `directions_queue` filtering and auto-continue logic replaced with comment; all rounds now fall through to normal wait logic regardless of queue state.
- **Annotation context report** (`report_generator.py`): Per-step rendering: each `StepAnalysis` emits an `.analysis-step` block with step marker, description, inline figure (`.step-figure`), and expandable raw output. `orphan_blocks` collects `publication_figures` not already shown; rendered as "Additional Figures" with dynamic section number. CSS gains `.analysis-step`, `.step-header`, `.step-marker`, `.step-desc`, `.step-figure`, `.step-figure-caption`, `.step-explanation`, `.step-output-details` rules.
- **AnalysisPanel snapshot button** (`frontend/src/components/AnalysisPanel.tsx`): When `isCompleted`, renders a `<button>` calling `setShowSnapshotPanel(true)` with Camera icon beside the green "Analysis Complete" badge. `finalSummary` card: `from-accent-500 to-accent-600` → `from-accent-50 to-accent-100`.

#### v0.18.3 (2026-03-29)

- **PI-guided annotation** (`workflow.py`, `pi_agent.py`, `literature_agent.py`): Three new PI Agent methods — `generate_annotation_search_query()` builds an optimised PubMed query using MeSH terms and Boolean operators; `evaluate_annotation_search_results()` assesses coverage per cluster and proposes refined queries; `annotate_clusters_from_literature()` maps cluster markers to cell types using the strong PI model. `literature_agent.annotate_clusters()` gains `pi_initial_query`, `pi_evaluate_fn`, and `skip_matching` params. `workflow._annotate_clusters_step()` now runs a two-step flow: Literature executes the search, PI handles matching. Non-pubmed modes (deep_research/gemini) also skip Literature matching and delegate to PI. Fallback chain preserved: PI query failure → full Literature Agent; PI matching failure → Literature Agent fallback; both fail → `_build_fallback_annotations()` stub.
- **Supervisor issue-type classification** (`supervisor_agent.py`): Evaluation prompt and default response dict gain `issue_type` field (`technical`/`biological`/`ambiguous`). `_evaluate_step()` adds a Layer 2 check: LLM-classified `technical` issues skip PI escalation even when no Python exception is present (e.g. blank plot from wrong column name).
- **Supervisor-only reviewer degradation** (`workflow.py`): `_supervisor_figure_review()` detects when `pi_vision=False` and `bio_vision=False`. In that mode: score ≥ 5 auto-approves (relaxed from 7); consecutive rejections tracked in `_sup_rejections`; after 2 rejections sets `requires_reanalysis=True` to break revision loops.
- **Supervisor feedback dedup** (`bioinformatics_agent.py`): `_recent_sup_feedback` set (reset per-round) skips re-injecting supervisor messages whose first 120 chars match a previously injected message, reducing token waste on repeated warnings.
- **Figure revision stall detection** (`bioinformatics_agent.py`): After each PI revision, `revision_history` is checked; if `quality_score` did not improve across the last two rounds, the figure is accepted with reservations and the tool returns immediately.
- **LLM retry WebSocket notifications** (`utils/llm_client.py`, `workflow.py`): `retry_with_backoff` gains optional `on_retry: Callable[[attempt, max_retries, delay, error_type], Awaitable[None]]` parameter, fired before each retry sleep. `OpenAICompatibleClient` exposes `self.on_retry` attribute. `workflow._install_retry_notifiers()` wires all agent LLM clients with a callback that pushes `status` messages to the WebSocket.
- **Message buffer removal** (`main.py`, `config.py`): `_message_buffers`, `_message_buffer_lock`, and `MAX_MESSAGE_BUFFER_SIZE` removed. WebSocket observer path now calls `_serialize_realtime_messages(state.realtime_messages)` (new helper, limit=500) instead of the capped buffer. Three inline message-serialization loops replaced with the helper.
- **Checkpoint write dedup** (`utils/checkpoint.py`): `_last_saved_step` attribute tracks the last successfully written `step_name`; `save()` returns early if the same step is re-submitted, eliminating redundant disk writes during figure-revision loops.
- **Pathway enrichment library fallback** (`config.py`): Bioinformatics system prompt now documents preferred gseapy/Enrichr library order: `GO_Biological_Process_2023/2021/2018` → `Reactome_2022`, `WikiPathways_2024_Human`, `BioPlanet_2019` → `KEGG_2021_Human`, with instructions to catch errors and try the next library automatically.
- **`_tracking_savefig` brace escape fix** (`bioinformatics_agent.py`): Format string `{_track_err}` changed to `{{_track_err}}` to prevent f-string interpolation inside injected code template.
- **`_build_fallback_annotations` extracted** (`workflow.py`): Inline fallback dict construction (repeated in two except blocks) replaced with static helper `_build_fallback_annotations(cluster_markers, error)`.

#### v0.18.2 (2026-03-29)

- **Fine-grained session concurrency** (`main.py`): Per-session `asyncio.Lock` via `_get_session_lock()`; removed TOCTTOU window between `_active_sessions` check and modification. Session-level constants (`MAX_CACHED_SESSIONS`, `SESSION_CACHE_TTL`, etc.) extracted to `config.py`.
- **WebSocket reconnect after snapshot restore** (`main.py`): `_pause_and_restart_ws()` closes WS on confirmed HITL with no running workflow, triggering client reconnect.
- **LLM temperature & retry** (`utils/llm_client.py`): `_get_effective_temperature()` returns `None` for o1/o3/o4, `1.0` for Gemini 3.x thinking models, requested value otherwise. Retry count 3→4; delays `[5, 10, 20, 40]` s.
- **AST sandbox layer** (`utils/sandbox.py`): Second-pass AST visitor catches reflection escapes (`getattr`, `__globals__`, `__builtins__`, `__dict__[...]`, `__subclasses__`) that regex alone misses.
- **Annotation guard** (`agents/bioinformatics_agent.py`): `_CELL_TYPE_ASSIGN_RE` regex blocks direct `adata.obs['cell_type'] =` assignment; matplotlib figures closed safely with `try/except plt.close('all')`.
- **Annotation recovery** (`workflow.py`): Already-confirmed annotations skip re-confirmation on restart; PI review failures fall back to original annotation rather than aborting.
- **Upload concurrency** (`utils/upload.py`): `MAX_CONCURRENT_UPLOADS_PER_SESSION = 5`; counter rolled back on metadata init failure via `try/finally`.
- **Report CSS extraction** (`utils/report_css.py`): 538-line report stylesheet extracted from `supervisor_agent.py` into shared module.
- **Image utils extraction** (`utils/image_utils.py`): `compress_image_to_base64()` and `categorize_figures()` moved from `report_generator.py`.
- **AnalysisState serialisation** (`models/schemas.py`): `to_api_dict()` centralises snake→camelCase, Enum/datetime conversion, and URL generation.
- **Download utilities** (`frontend/src/utils/download.ts`): `downloadBlob()` and `fetchAndDownload()` replace inline blob-download code in ReportPanel, ResultsPanel, DownloadButtons.
- **Parallel figure download** (`analysis/DownloadButtons.tsx`): `Promise.all()` parallelises figure and table downloads.
- **Upload blocklist** (`DataPanel.tsx`): `BLOCKED_EXTENSIONS` set rejects executable/script files client-side.
- **Tailwind colour system** (`tailwind.config.js`): primary → Material red, accent → modern blue; Sidebar migrated to light theme.
- **Abstract method guard** (`agents/base.py`): `process()` raises `NotImplementedError` instead of silently passing.
- **Config constants** (`config.py`): `_ensure_directories()` extracted; session/upload/WS constants centralised.

#### v0.18.1 (2026-03-27)

- Version bump and documentation sync.

#### v0.18.0 (2026-03-27)

- **Snapshot tree view** (`SnapshotPanel.tsx`, `schemas.py`): Snapshots now display as a tree with parent-child relationships using `parent_id` field. DFS traversal builds depth-based indentation and detects branch points (multiple children of the same parent). `GitBranch` icon marks divergence points. Snapshot count shown in panel header. Manual snapshot creation UI (input + create button) and backend endpoint (`POST /snapshots`) removed — snapshots are now auto-created at key analysis points only. Empty state message updated to explain auto-creation.
- **Snapshot restore reliability** (`main.py`, `workflow.py`, `snapshot.py`): Restoring a snapshot now (1) closes the workflow DB connection before cleanup to avoid SQLite lock, (2) calls new `SnapshotManager.delete_checkpoint()` to remove LangGraph checkpoint files (`checkpoints.sqlite*`), and (3) sets phase-aware status: `pre_direction` → `WAITING_INPUT` (re-opens direction selection), `pre_annotation` → `WAITING_ANNOTATION` (re-opens annotation dialog), others → `PAUSED`. New `last_restored_snapshot_id` field tracks restore provenance for parent chain. `workflow.resume()` now handles missing checkpoint gracefully by starting a fresh graph invocation via `_make_initial_workflow_state()`. WebSocket resume condition extended to include `WAITING_ANNOTATION`.
- **Snapshot parent chain tracking** (`workflow.py`): `_create_auto_snapshot()` sets `parent_id` from `last_restored_snapshot_id` (if just restored) or from the last snapshot in the list. Restore marker cleared after first snapshot to avoid polluting subsequent snapshots. Removed `post_annotation` auto-snapshot (only `pre_annotation` and `pre_direction` remain).
- **Annotation context report figure scanning** (`report_generator.py`): `generate_annotation_context_report_html()` now scans workspace and `figures/` subdirectory for `.jpg`/`.png` files directly, with `seen_stems` deduplication. Fixed `publication_figures` attribute access (`png_base64`/`name` instead of `base64_data`/`title`). This fixes empty figure sections when the report is generated mid-execution before bioinformatics agent returns.
- **Analysis log passthrough for annotation** (`workflow.py`, `bioinformatics_agent.py`): `annotate_clusters_callback` now accepts `analysis_log` parameter. During annotation, the in-progress bioinformatics log is temporarily populated into `state.analysis_log` so the context report can access figures from the current round, then restored after report generation.
- **ANSI escape code stripping** (`bioinformatics_agent.py`): Imported `strip_ansi` from `utils/error_parsing`. Applied to `result.error` in code execution, figure generation, and figure revision paths — producing clean error messages without terminal color codes.
- **Explicitly saved figure tracking** (`bioinformatics_agent.py`): Auto-save code now emits `[Auto-saved] JPG/PNG` lines for user-saved images (previously only emitted for auto-converted images), ensuring they appear in `analysis_log.saved_figure_filenames`. Regex updated to match `.jpeg` extension (`jpe?g`).
- **Status badge styles** (`globals.css`): Added `.status-waiting_annotation`, `.status-waiting_input` (amber), and `.status-generating_report` (indigo) CSS classes matching existing badge pattern.
- **Workflow refactor** (`workflow.py`): Extracted `_make_initial_workflow_state()` static method to eliminate duplicated `WorkflowState` dict construction between `run()` and `resume()`.

#### v0.17.5 (2026-03-26)

- **Fix: Loading spinner not animating** (`globals.css`, `AnalysisPanel.tsx`): Loader2 spinner icons were rendered as static icons because the `@media (prefers-reduced-motion: reduce)` rule in `globals.css` applied `animation-duration: 0.01ms !important` to all elements, overriding both inline styles and Tailwind classes. Fixed by excluding `.animate-spin` elements from the reduced-motion selector. Additionally, standardized 3 inline `style={{ animation: 'spin 1s linear infinite' }}` usages in `AnalysisPanel.tsx` to Tailwind's `animate-spin` class for consistency.
- **Improvement: Sidebar project description adaptive height** (`Sidebar.tsx`): Long project descriptions previously pushed Export/Import/API Settings buttons off-screen. Merged the Current Session section and spacer into a single `flex-1 min-h-0` flex container. Title bar and status bar use `shrink-0` to stay visible; the description `<p>` uses `overflow-y-auto min-h-0` to scroll only when exceeding available space. Short descriptions display fully without wasted whitespace.
- **Improvement: Sidebar description readability** (`Sidebar.tsx`): Increased description font size from `text-xs` (12px) to `text-sm` (14px). Changed text color from `text-accent-300/70` (low-contrast translucent purple) to `text-surface-300` (clear light gray) for better readability on the dark sidebar background.

#### v0.17.4 (2026-03-26)

- **Frontend visual redesign — "Scientific Elegance"** (`index.html`, `tailwind.config.js`, `globals.css`, `App.tsx`, `Sidebar.tsx`, `MessageStream.tsx`, all dialog and panel components): Complete UI beautification with zero new npm dependencies. All changes are CSS/Tailwind-only.
  - **Typography**: Google Fonts updated from Inter to Source Serif 4 (display/headings, serif) + IBM Plex Sans (body, technical sans-serif) + JetBrains Mono (code, unchanged). Tailwind `fontFamily` extended with `display`, `sans`, `mono`. New `font-display` class applied to all panel headings, modal titles, app branding, and session names for academic journal aesthetic.
  - **Color system**: New `surface` color scale (blue-gray, 50–950) for backgrounds/text/borders replaces raw `slate-*` in chrome areas. New `accent` color scale (indigo #6366f1, 50–900) replaces `violet-*` as primary accent throughout all components. Agent identity colors (pi=violet, pathology=red, bioinformatics=green, biochemistry=amber, literature=blue) preserved unchanged. Agent CSS variables extended with `--light` variants for gradient effects.
  - **Dark sidebar** (`Sidebar.tsx`): Background changed from `bg-white` to `bg-surface-900`. All text/borders/inputs/buttons adapted to dark palette. Logo gets `shadow-glow-accent`. "New Analysis" button gets gradient + hover lift micro-interaction (`hover:-translate-y-0.5`). Session card uses `bg-accent-900/30` with `border-accent-700/40`. Custom `.dark-scrollbar` class applied.
  - **Header & tabs** (`App.tsx`): Header height reduced to `h-14` with enhanced `backdrop-blur-md` and subtle box-shadow. Tabs redesigned from pill-style (`bg-slate-100 rounded-lg p-1`) to underline-style with animated bottom indicator (`animate-tab-indicator`). Footer text set to `uppercase tracking-wider` at 10px.
  - **Loading screen** (`App.tsx`): Dark background (`bg-surface-900`) with DNA icon, accent gradient badge, and `shadow-glow-accent`.
  - **Content area** (`App.tsx`): `.bg-science-grid` CSS class adds subtle dot-grid pattern (`radial-gradient`, 24px spacing, indigo at 5% opacity) evoking graph paper / microscopy grid.
  - **Message stream** (`MessageStream.tsx`): New `agentGradients` record maps each agent to gradient classes. Agent headers now show gradient-background icon badges with white icons. Message cards get `animate-fade-in-up` entrance animation and `shadow-card`. Stream header uses `backdrop-blur-sm` with accent-colored message count badge. Background changed to `bg-surface-50`.
  - **Animations** (`tailwind.config.js`, `globals.css`): New keyframes — `fade-in-up` (0.4s cubic-bezier for cards), `scale-in` (0.25s for modals), `shimmer` (loading glow), `glow-pulse` (active agent indicator), `tab-indicator` (tab underline). Stagger utilities (`.stagger-1` to `.stagger-5`). `@media (prefers-reduced-motion: reduce)` disables all animations.
  - **Shadows** (`tailwind.config.js`): Five new tokens — `subtle`, `card`, `elevated`, `modal`, `glow-accent`. Applied to message cards (`shadow-card`), all modals (`shadow-modal`), logo (`shadow-glow-accent`).
  - **Modals** (all dialog components): `animate-scale-in` added to all 6 dialog containers. `shadow-modal` replaces `shadow-2xl`/`shadow-xl`. Gradient headers unified to `accent-*` palette.
  - **Scrollbars** (`globals.css`): Reduced to 6px width, transparent track, rounded thumb. `.dark-scrollbar` variant for dark backgrounds.
  - **Status badges** (`globals.css`): Redesigned from flat backgrounds to pill-shaped with fine borders (`rounded-full`, `border`, `inline-block`).
  - **Glass effects** (`globals.css`): `.glass` enhanced with `saturate(1.5)` and translucent border. New `.glass-dark` variant.
  - **Input styling** (`globals.css`): New `.input-science` component class with unified focus ring, border transitions.
  - **Resizable divider** (`App.tsx`): Width increased to 3px, accent color on hover.

#### v0.17.3 (2026-03-26)

- **Technique Agent** (`technique_agent.py`, new file): Lightweight API documentation lookup agent. Executes `inspect.getdoc()`/`inspect.signature()` in the Jupyter sandbox to extract docstrings from installed packages, then summarizes via a cheap LLM (`google/gemini-3.1-flash-lite-preview`) into concise API references (signature, key parameters, example). Two entry points: `lookup_docs(sandbox, function_path)` for direct lookups and `lookup_from_error(sandbox, error_text, code)` for error-driven lookups. Includes `_parse_error_for_lookup()` to extract module/attribute/function from AttributeError, TypeError, ImportError, and ModuleNotFoundError tracebacks. `_guess_function_path()` resolves short function names to full dotted paths using import statements and common aliases (sc→scanpy, dc→decoupler, etc.). Uses `asyncio.gather` for parallel sandbox calls when looking up both attribute-level and module-level docs.
- **Bioinformatics Agent integration** (`bioinformatics_agent.py`): New `lookup_documentation` tool added to TOOLS array. Handler delegates to `self.technique_agent.lookup_docs()` when available, falling back to "not available" message. Hardcoded DECOUPLER API and GSEAPY PARAMETER RULES sections in the system prompt replaced with a generic PACKAGE API LOOKUP instruction directing the agent to call `lookup_documentation` before writing code for unfamiliar APIs.
- **Supervisor Agent integration** (`supervisor_agent.py`): New Phase 1.5 in `_evaluate_step()` — after rule checks but before LLM evaluation, `_should_trigger_technique()` checks if (1) technique_agent and sandbox are available, (2) current step has an API-related error, (3) previous step had the same error type. If triggered, calls `technique_agent.lookup_from_error()` and injects the result as `[Technique Support]` feedback. `set_sandbox()` method added for deferred sandbox injection. Local `_extract_error_type()` replaced with shared `extract_error_type()` from `utils/error_parsing.py`.
- **Error parsing utility** (`utils/error_parsing.py`, new file): Extracted `extract_error_type(error_text)` function and `API_ERROR_TYPES` frozenset (`AttributeError`, `TypeError`, `ImportError`, `ModuleNotFoundError`) shared between Supervisor and Technique agents.
- **Configuration** (`config.py`, `schemas.py`): `ModelConfig.technique_model` (default: `google/gemini-3.1-flash-lite-preview`); `AgentTemperatures.technique` (default: 0.1); `AgentType.TECHNIQUE` enum value.
- **Workflow wiring** (`workflow.py`): `TechniqueAgent` instantiated before other agents and injected into both `BioinformaticsAgent` and `SupervisorAgent`. `supervisor_agent.set_sandbox()` called after sandbox initialization.

#### v0.17.2 (2026-03-26)

- **Docker detached mode** (`docker-entrypoint.sh`, `README.md`): Docker run command updated to use `-d` (detached) and `--name scai` flags so the container runs as a background service without occupying the terminal. All stdout/stderr output from the entrypoint script (including backend and frontend logs) is redirected via `exec > >(tee -a "$LOG_FILE") 2>&1` to `/data/logs/scai.log` in append mode, preserving logs across container restarts. Users can view logs via `docker logs -f scai` or by reading the log file directly.

#### v0.17.1 (2026-03-26)

- **Per-agent LLM server configuration** (`schemas.py`, `workflow.py`, `main.py`, `PromptsPanel.tsx`, `appStore.ts`): New `agent_server_configs` field in `AnalysisState` maps agent names to per-agent `llm_base_url` and `api_key` overrides. New `_get_agent_llm_client()` helper in both `workflow.py` and `main.py` creates per-agent clients that fall back to global config when no override is set. All 7 agents now receive per-agent clients during workflow initialization. Extra per-agent clients are tracked and cleaned up on workflow end. Two new API endpoints: `POST/GET /api/session/{session_id}/agent-server-config`. PromptsPanel UI adds LLM Server and API Key fields per agent, persisted to `localStorage` and backend. Reset clears per-agent overrides to global defaults.
- **Painter Agent dual-path image generation** (`painter_agent.py`): Image generation now uses a two-strategy approach: (1) chat completions with `modalities=["image", "text"]` (works on OpenRouter), (2) Images API fallback via `/v1/images/generations` (works on OpenAI, Google AI Studio, SiliconFlow). Also handles both `inlineData` and `inline_data` response formats. Error messages combine failures from both strategies.
- **Config panel simplification** (`ConfigPanel.tsx`, `PromptsPanel.tsx`): Google AI API Key input removed from ConfigPanel and moved to Literature agent section in PromptsPanel (shown only when `gemini_deep_research` mode is selected). "LLM Server" label renamed to "Default LLM Server".
- **i18n updates** (`translations.ts`): Added `agentLlmServer`, `agentApiKey`, `agentServerHint`, `agentModelHint` translation keys for both Chinese and English.

#### v0.16.2 (2026-03-24)

- **Docker packaging** (`Dockerfile`, `docker-entrypoint.sh`, `build.sh`, `.dockerignore`): Added Docker support for building and publishing the ScAI image to `ghcr.io/minzhangcheng/scai`. The image is based on `python:3.11-slim` with Node.js 20 and build-essential. Dependencies are not baked into the image — they are installed at first container run and persisted via volume mounts (`/dep` for venv + node_modules, `/data` for application data). Only the frontend port (3000) is exposed; the backend (8000) remains internal with Vite proxying API/WebSocket requests. The `build.sh` script supports `--version` and `--push` flags for CI/CD integration.

#### v0.16.1 (2026-03-24)

- **Vision fallback for figure revision** (`bioinformatics_agent.py`): The `chat_with_vision` call in the figure revision loop is now wrapped in a try-except. If the model raises a `NotFoundError` or HTTP 404 (indicating no vision support), the image attachment is dropped, the prompt's image reference line is replaced with a text-only fallback notice, and the request retries via `chat()`. All other exceptions are re-raised unchanged.
- **Data integrity check patterns** (`supervisor_agent.py`): Two module-level compiled regex patterns added: `_PERCENTAGE_PATTERN` (matches `\d+(?:\.\d+)?%`) and `_INF_NAN_PATTERN` (matches `\b(inf|-inf|nan)\b`, case-insensitive). These constants support upcoming data integrity validation logic in supervisor step monitoring.
- **Extended sandbox execution timeout** (`config.py`): `SystemConfig.sandbox_timeout` increased from 3600 seconds to 5400 seconds (1.5 hours) to prevent premature timeout failures during long-running bioinformatics computations.

#### v0.15.9 (2026-03-23)

- **Figure revision loop indentation fix** (`bioinformatics_agent.py`): Code blocks for `_validate_code`, sandbox `execute()`, and auto-retry logic in the figure revision loop were indented one level too deep — erroneously nested inside a non-existent block. Corrected indentation aligns these blocks directly under the `while` loop iteration, restoring correct Python scope and execution flow.
- **Auto-sync API config on startup** (`App.tsx`): On startup, if `checkStatus()` reports the backend is not configured but the frontend `apiConfig.apiKey` is non-empty (saved in localStorage), the app automatically calls `saveConfig()` and sets `configured = true` without showing the config panel. A warning is logged if auto-sync fails and the normal config panel flow proceeds.
- **Auto-sync API config before analysis** (`AnalysisPanel.tsx`): Before starting an analysis, the config-check logic is split into two cases: missing `apiConfig.apiKey` on the frontend immediately shows the config panel; a backend `not configured` status triggers a silent `saveConfig()` + `checkStatus()` retry, preventing unnecessary config panel interruptions after a backend restart.
- **Config status check error logging** (`useApi.ts`): The `catch` block in `checkStatus()` now logs a `console.warn` with the caught error instead of silently returning `{ configured: false, has_api_key: false }`.

#### v0.15.8 (2026-03-23)

- **Figure save path tracking** (`bioinformatics_agent.py`): `_explicitly_saved_figs` changed from `set` to `dict` mapping figure names to full file paths. `_tracking_savefig` now captures the resolved path on each `savefig` call. After saving, if no companion PDF exists, one is auto-generated at 600 dpi preserving the original layout. Revision counter (`_figure_revision_count`) is incremented only after revised code executes successfully, preventing inflated counts on failed attempts.
- **Supervisor feedback deduplication** (`supervisor_agent.py`): New `_recent_rule_feedback_hashes` set stores SHA hashes of rule-based feedback messages. Before sending feedback, a hash is computed; if already present the message is suppressed. The set is cleared alongside `_recent_error_hashes` when the error stream resets, giving fresh deduplication context for the next analysis segment.
- **Figure extension fallback** (`supervisor_agent.py` new `_alt_ext_name()`): When a figure lookup fails, `_alt_ext_name()` returns the alternate common extension (`.png` → `.jpg`, `.jpg`/`.jpeg` → `.png`). The caller retries with the alternate path before reporting a missing figure.
- **Dynamic download button labels** (`supervisor_agent.py`, `report_generator.py`): Download buttons for figures now extract the actual file extension from the filename and display it in uppercase (e.g., "JPG", "PNG") instead of a hardcoded "PNG" label. Consistent across supervisor HTML reports and report generator output.
- **Unified `.jpg` default in report generator** (`report_generator.py`): JSON schema examples and LLM prompt guidance updated to use `.jpg` as the default figure extension, matching actual pipeline output. PDF companion path is now derived using `Path(fig).stem + ".pdf"` for correct handling of any source extension.
- **WebSocket log noise reduced** (`main.py`): `_safe_send_ws()` send failures are now logged at `debug` level (was `exception`) with a message indicating normal connection-close, reducing log noise during routine disconnects.
- **Model recommendation updates** (`PromptsPanel.tsx`): PI Agent description now notes Vision capability is required for figure quality review. Bioinformatics Agent description explicitly states Vision is required for figure revision and removes outdated model name. Supervisor Agent description emphasizes Vision capability for real-time monitoring and labels it as a lightweight real-time component.

#### v0.15.7 (2026-03-22)

- **Mandatory exploration pipeline enforcement** (`bioinformatics_agent.py`): Added `annotation_completed` boolean tracker (initialized to `False`). `report_findings` now checks this flag in exploration phase and rejects the call with a `[REJECTED]` tool response if `annotate_clusters` has not yet been called, instructing the agent to complete `rank_genes_groups → annotate_clusters` first. `annotation_completed` is set to `True` when `annotate_clusters` tool call succeeds.
- **Smarter circuit breaker in exploration phase** (`bioinformatics_agent.py`): When `consecutive_errors >= MAX_CONSECUTIVE_ERRORS` and the analysis is in exploration phase with annotation not yet done, the circuit breaker now sends a redirect message asking the agent to try a simpler approach (`sc.tl.rank_genes_groups` + `annotate_clusters`) rather than forcing a premature report. The original force-report behavior is preserved for all other cases.
- **Inline table support in HTML reports** (`supervisor_agent.py`): `_build_table_html()` helper reads up to 10 rows from CSV files and renders them as styled HTML tables with a download button. Report renderer now replaces `[TABLE: filename]` (optionally followed by `[TABLE_CAPTION: text]`) markers inline. Any tables referenced in text but not yet placed are appended as a supplementary section at the end of the report.
- **Supervision log restoration for report regeneration** (`supervisor_agent.py`, `main.py`): New `SupervisorAgent.restore_log_from_saved()` deserializes supervision records from a list of dicts (e.g., loaded from `report.json`), converting ISO timestamp strings back to `datetime` objects and populating `self.supervision_log`. `_generate_report_task` in `main.py` now instantiates `SupervisorAgent`, restores the log from `report.json` if present, then calls `supervisor.generate_detailed_report()`. `DetailedReportGenerator` path temporarily disabled (commented out).
- **Sandbox safety refinements** (`sandbox.py`): `locals()` removed from `DANGEROUS_PATTERNS` — it is read-only and commonly used in data-science code. `os.system()` validation now checks whether ALL usages in the code are pip install commands (regex: `os.system\(["'](?:pip|pip3)\s+install`); if so, the block is skipped. `importlib.reload` added to the importlib safe-use allowlist alongside `import_module` and `util.find_spec`.
- **Annotation confirmation decoupled from analysis phase** (`workflow.py`): The guard for triggering PI review and user confirmation of cluster annotations changed from `if state.analysis_phase == "exploration"` to `if not (state.annotation_state and state.annotation_state.confirmed)`, ensuring the confirmation dialog fires whenever annotations are pending regardless of the current phase.

#### v0.14.6 (2026-03-21)

- **Dynamic AnnData discovery** (`bioinformatics_agent.py` new `_discover_adata_in_sandbox()`): New helper method iterates over sandbox global variables to find AnnData objects containing leiden/louvain/cluster columns. Returns `(var_name, cluster_key)` pairs. Cluster annotation workflow now calls this instead of hardcoding `"adata"` and `"leiden"`, supporting any user-defined variable names.
- **Faster consecutive error recovery** (`bioinformatics_agent.py`): `MAX_CONSECUTIVE_ERRORS` reduced from 5 to 3, triggering forced report generation sooner when analysis is stuck. Checkpoint save is now wrapped in try-except to prevent checkpoint failures from aborting the main pipeline.
- **Full output visibility** (`bioinformatics_agent.py`, `literature_agent.py`): Removed truncation on code execution output (`output[:1500]` → full), literature search results (`[:2000]` → full), and PubMed paper abstracts (`[:200]` → full). LLM receives complete content for better decision-making.
- **Cluster annotation auto-injects publication figures** (`bioinformatics_agent.py`): When `publication_figures` are available in state, they are injected into the annotation prompt for Supervisor visual review, without requiring bioinformatics agent to regenerate them.
- **Smart PI escalation filter** (`supervisor_agent.py` new `_is_technical_error()`): New `_TECHNICAL_ERROR_TYPES` frozenset contains Python runtime exception names (KeyError, TypeError, IndexError, etc.). `_is_technical_error()` checks if error message starts with any known exception type. Errors matching this list are no longer escalated to PI — PI provides biological guidance, not Python debugging.
- **Versioned figure deduplication** (`supervisor_agent.py`): When collecting figures for the report, if a base figure name (e.g., `umap`) and a versioned variant (e.g., `umap_v2`, `umap_final`) both exist, only the versioned variant is retained. Prevents duplicate figures in report.
- **Score definitions HTML table** (`supervisor_agent.py` new `_build_score_definitions_html()`): Renders a structured HTML table of all score definitions including score name, method, description, parameters, gene list, and source. Included in `score_definitions` field of the final report.
- **Faster stall detection** (`supervisor_agent.py`): Consecutive error threshold reduced from 3 to 2 for faster detection of stuck analysis loops.
- **Truncation limits removed** (`supervisor_agent.py`): `agent_reasoning` (was 500 chars), `code_output` (was 500 chars), and `annotation_result_summary` (was 2000 chars) now display without truncation.
- **PubMed XML `itertext()` fix** (`literature_agent.py`): Abstract and title extraction changed from `findtext()` to `"".join(element.itertext())` to correctly reconstruct text containing inline XML tags (`<i>`, `<sub>`, `<sup>`, etc.) that `findtext()` would silently drop.
- **Deep Research connection retry** (`literature_agent.py`): Added retry wrapper around the Deep Research API call (max 2 retries, 5s/10s delays) catching `OpenAIConnectionError`, `httpx.ConnectError`, `httpx.ConnectTimeout`, `OSError`. Addresses transient network failures in long-running 1800s-timeout requests.
- **Cluster annotation table removed from report** (`report_generator.py`): Removed TOC entry and table generation for cluster annotations. The `n_cells` field was not correctly populated by the pipeline, making the table incomplete. Analysis log display also removed from the main report body.

#### v0.14.5 (2026-03-20)

- **`directions_queue` type mismatch fix** (`workflow.py` `_wait_for_user_input`): Queue items are `Dict[str, str]` (`{"direction": ..., "notes": ...}`), but the filter used `d.strip()` causing `AttributeError` when multi-direction mode was used. Fix: added a pre-filter step using `d.get("direction", "").strip()` before the skip-wait decision.
- **Countdown timeout status desync** (`workflow.py` `_wait_for_user_input`): After `asyncio.wait_for` timeout, `analysis_state.status` was not reset to `RUNNING` and no state sync was sent. Frontend remained showing `WAITING_INPUT`. Fix: set status + call `_send_state_sync("auto_continue")` after timeout.
- **PI callback context accumulation** (`workflow.py` `_pi_callback`): Used `chat()` which accumulates history, causing unrelated prior bioinformatics context to pollute PI consultation responses. Changed to `chat_without_history()`.
- **Annotation callback field name** (`workflow.py` `_annotate_clusters_callback`): Returned `tissue_context` (does not exist on `AnnotationConfirmationState`); correct field is `tissue_summary`. Also added `"already_confirmed": True` to the cached-annotation return path to prevent double confirmation prompts.
- **Literature search with no prior findings** (`workflow.py` `_literature_research`): When no findings exist from other agents (e.g., early round), topic building fell into an empty `else` branch, silently skipping literature search. Fix: build topics from `research_objective` or treatment group names; log a warning and skip only if truly nothing is available.
- **Multi-direction correlation direction mismatch** (`workflow.py` `_pi_synthesis`): Correlation analysis matched each `user_chosen_direction` to an innovation round by list index (`innovation_rounds[i]`). If any round was skipped or reordered, summaries were paired with wrong directions. Fix: build a `direction_summaries` dict keyed by `r.direction` and look up by name.
- **`analysis_log` unbounded growth** (`workflow.py` `_bioinformatics_analysis`): Log accumulated indefinitely across rounds; no truncation. Fix: cap at `_MAX_ANALYSIS_LOG = 2000` entries (keep latest) after each append pass.
- **`skip_nodes` lost on resume** (`workflow.py` `resume()`): `skip_nodes` decisions are stored in `analysis_state.sandbox_state["skip_nodes"]` but `with_retry` checks the top-level `WorkflowState["skip_nodes"]`. On resume, the top-level key was never populated. Fix: extract `skip_nodes` from `sandbox_state` and inject into graph state via `aupdate_state` before `ainvoke`.
- **`JsonPlusSerializer` too permissive** (`workflow.py` `initialize()`): `allowed_msgpack_modules=True` allows deserialization of any module. Changed to explicit allowlist: `backend.models.schemas` types only (`AgentType`, `AnalysisState`, `AnalysisPhase`, `RoundResult`, `TaskStatus`, `MessageRole`).
- **Figure review always auto-approved** (`workflow.py` `_generate_integrated_report`): `PublicationFigure` was initialized with `review_status: "approved"`, bypassing the PI review loop. Changed to `"pending"`.
- **`_db_conn` not cleared after close** (`workflow.py` `cleanup()`): Added `finally: self._db_conn = None` to prevent potential double-close if `cleanup()` is called multiple times.
- **PI merge strategy passed to bioinformatics** (`workflow.py` `_bioinformatics_analysis`): `merge_strategy` and `cluster_comparison_strategy` from `pi_instructions` are now appended to the bioinformatics instruction string, enabling biologically informed cluster merging (e.g., "stress hepatocytes + hepatocytes → hepatocytes") before DEG analysis.
- **Exploration phase per-cell-type figure requirement**: Exploration round instruction now explicitly requires a volcano plot (or bar chart) for each cell type's DEG results and a pathway enrichment bar plot for each cell type's GO/KEGG results.
- **PI English-only enforcement** (`workflow.py` `_pi_callback`): Added `IMPORTANT: Respond in English only.` to the PI consultation prompt to prevent language mixing when non-English data is in the context.

#### v0.14.4 (2026-03-17)

- **Project Renamed** (all files): InteSc → ScAI (Single Cell Sequencing Analysis Driven by AI). All display names, branding, titles, localStorage keys, package names, and domain references updated.
- **Directory Restructure** (`config.py`, `start.sh`, docs): Data directory `intesc_data/` → `data/`, dependency directory `intesc_dep/` → `dep/`, code directory `intesc/` → `scai/`. Environment variable `INTESC_DATA_DIR` → `SCAI_DATA_DIR`.
- **Configurable Startup** (`start.sh`, `vite.config.ts`): `start.sh` now accepts `--host` (default 127.0.0.1), `--port` (default 3000), `--backport` (default 8000) CLI arguments. Vite proxy reads `SCAI_BACKPORT` env var for backend port.
- **GitHub Repo Renamed**: `minzhangcheng/intesc` → `minzhangcheng/scai`.

#### v0.14.3 (2026-03-17)

- **JPG Figure Output** (`bioinformatics_agent.py`): All auto-saved figures and `generate_figure` / `revise_figure` tool calls now save JPG (quality 85, optimized) instead of PNG for web display. File size and bandwidth are reduced. Companion high-res PDF still generated. Regex for `[Auto-saved]` log lines updated to match both `.jpg` and `.png`. `PublicationFigure.png_path` now stores the JPG path; field name preserved for backward compatibility.
- **Larger Default Plot Fonts** (`bioinformatics_agent.py`, `config.py`): matplotlib rcParams updated in all three figure-saving templates: `font.size` 12→14, `axes.titlesize` 14→16, `axes.labelsize` 12→14, `xtick.labelsize` 10→12, `ytick.labelsize` 10→12, `legend.fontsize` 10→11. System prompt minimum text requirement raised from 10pt to 12pt (title 14pt).
- **decoupler v2.x API** (`bioinformatics_agent.py`): System prompt updated from deprecated v1.x API (`dc.get_collectri`, `dc.run_ulm`) to v2.x API (`dc.op.collectri/progeny/dorothea` + `dc.mt.ulm`). Results now in `adata.obsm['score_ulm']` / `adata.obsm['padj_ulm']`. Old method names explicitly listed as "no longer exist".
- **gseapy Organism Case** (`bioinformatics_agent.py`): New `GSEAPY PARAMETER RULES` section in system prompt: organism must be lowercase (`'mouse'`, `'human'`) — not `'Mouse'` or `'Human'`.
- **Defensive Empty-DataFrame Coding** (`bioinformatics_agent.py`): New `DEFENSIVE CODING` section in system prompt with pattern for guarding per-cluster DE/enrichment loops: check `df.empty` or `'names' not in df.columns` before accessing results; applies to `rank_genes_groups_df`, `gseapy.enrichr`, `gseapy.prerank`, and any per-group iteration. Applied to cell type annotation workflow example too.
- **Harmony Dimension-Mismatch Prevention** (`bioinformatics_agent.py`): New `HARMONY BATCH CORRECTION` section in system prompt: always verify `ho.Z_corr.T.shape[0] == adata.n_obs` before assignment; never filter cells between `sc.tl.pca()` and `hm.run_harmony()`.
- **`importlib` Blocked** (`bioinformatics_agent.py`): Added `importlib` to the security restrictions blocked list. New guidance: use `!pip install package_name` in a code cell for missing packages, then import normally. Do NOT use `subprocess.run(['pip', ...])` or `importlib`.
- **Supervisor Inline Figure Embedding** (`supervisor_agent.py`): `_generate_html_report()` now processes `[FIGURE: filename]` markers in section content and replaces them with inline `<div class="figure-inline">` blocks. Figures include JPEG/PNG MIME detection, PDF download links (bilingual labels), and interpretation text from `PublicationFigure`. Unplaced figures appended as a `Supplementary Figures` section. Previously figures were matched by filename substring and appended after each section.
- **Supervisor Report: Monitoring Stats Bar Removed** (`supervisor_agent.py`): Stats bar (steps monitored, issues, PI escalations, critical issues) removed from HTML output. `supervision_stats` removed from appendix data. Report prompt no longer includes summary statistics block.
- **Supervisor Report: Scientific Framing** (`supervisor_agent.py`): `[analysis_overview]` section description changed to "Analysis methodology overview (study design, data processing pipeline, computational methods, key parameters)". Prompt instructs to write as a formal scientific research report, not a monitoring log. `[supervisor_log]` and `[figure_appendix]` sections removed from base section list; these are now skipped in `generate_report()` parsing.
- **Supervisor Report: User Interaction History** (`supervisor_agent.py`): `_build_appendix()` now collects `cluster_annotations` (cluster, cell type, user override, confirmed, confidence) and `direction_selection` (direction, notes) from `state`. HTML appendix renders these as tables with bilingual headers. `related_reports` (exploration, annotation HTML) and notebook filename also added; rendered as download links.
- **Supervisor Report: Notebook in Output** (`supervisor_agent.py`): After HTML generation, the session notebook (`{session_id}.ipynb`) is copied to the report output directory using `shutil.copy2`. Served via the new catch-all report file endpoint.
- **`_set_llm_context()` in `finally`** (`supervisor_agent.py`): `_set_llm_context()` is now called in a `finally` block after the report-generation LLM call, ensuring supervisor context is restored even if the call raises.
- **`AnalysisState` passed to HTML generation** (`supervisor_agent.py`): `report["_analysis_state"]` temporarily set before calling `_generate_html_report()`; popped after. This allows `_generate_html_report` to read `publication_figures` for rich descriptions without serializing the state to JSON.
- **Painter Agent Retry Logic** (`painter_agent.py`): Both `generate()` and `revise()` now retry the image generation API call up to 3 times with exponential backoff (1s, 2s). 4xx client errors (400, 401, 403, 404, 413, 422) are detected via `_is_client_error()` and break the loop immediately. 429 (rate limit) is considered retriable. Token-estimate log calls downgraded from `INFO` to `DEBUG`.
- **Literature Cache Key: hashlib.md5** (`literature_agent.py`): Annotation cache key now uses `hashlib.md5(",".join(sorted(all_genes_for_key)).encode()).hexdigest()[:16]` instead of Python's built-in `hash()`, making keys deterministic across process restarts.
- **Pathology: Image Loaded Once** (`pathology_agent.py`): `_analyze_single_image()` loads the PIL image once and passes it to both `process_pathology_image()` (via new `preloaded_image` parameter in `image_utils.py`) and `create_overview_image()`, avoiding a redundant disk read.
- **`image_utils.process_pathology_image`: `preloaded_image` param** (`image_utils.py`): New optional `preloaded_image: PIL.Image | None = None` parameter; skips the `load_image()` call when a pre-loaded image is provided.
- **`schemas.py`: Async Version Lock** (`schemas.py`): `_version_lock` changed from `threading.RLock` to `asyncio.Lock` (lazily initialized on first use for Python 3.10/3.11 compatibility). `increment_version()` is now `async def`. `n_cells: int = 0` field added to `ClusterAnnotation`.
- **`main.py`: 100MB Generic File Upload Limit** (`main.py`): `upload_generic_file()` now stream-reads the file in 64KB chunks, accumulating up to 100MB. Raises HTTP 413 if exceeded.
- **`main.py`: Stop Handles GENERATING_REPORT** (`main.py`): `stop_analysis()` now accepts `GENERATING_REPORT` status. `immediate` stop cancels the report task and sets status to PAUSED; `summarize` returns early with a message. Previously raised HTTP 400 for this state. Also returns success (not 400) when already PAUSED.
- **`main.py`: Path Traversal Validation** (`main.py`): `get_exploration_report_html()` and `get_annotation_report_html()` now call `validate_path_within(html_path, session_dir)` before serving.
- **`main.py`: Catch-All Report File Endpoint** (`main.py`): New `GET /api/session/{id}/report/{filename:path}` endpoint serves any file from the report directory (notebook, sub-reports, CSV, etc.) with correct MIME type.
- **`main.py`: Resume Cache Consistency** (`main.py`): Multiple `_update_session_cache(session_id, state)` calls added in `resume_analysis()` after direction guard logic, PI_DECIDE extension, and other state mutations to keep the in-memory cache consistent.
- **`main.py`: Figure Scan JPG-First** (`main.py`): `_build_chat_context()` and `_collect_session_figures()` now scan `*.jpg` files first, then `*.png`, for backward compatibility with older sessions.
- **`main.py`: Stale Session Cleanup** (`main.py`): `cleanup_stale_sessions()` now also removes entries from `report_websockets` and `detailed_reports` dicts when cleaning up expired sessions.
- **Frontend: JPG MIME Auto-Detection** (`FigureGallery.tsx`, `GraphicAbstract.tsx`, `MessageStream.tsx`, `ChatPanel.tsx`): Base64 strings starting with `iVBOR` are treated as PNG; all others as JPEG. Applied to `<img src>` attributes and download link generation. Figure gallery download button label updated PNG→JPG.
- **`useApi.ts`: Message Merge Race Condition Fix** (`useApi.ts`): On WebSocket reconnect, `setMessages()` is only called when there are genuinely new server messages. Merge reads the latest store state (not the stale closure) and only adds messages not already present locally.
- **`ChatPanel.tsx`: Stale Closure Fix** (`ChatPanel.tsx`): Message-sync effect reads latest chat messages from `useAppStore.getState()` instead of the stale `chatMessages` closure value. Optimistic local-only messages are preserved when merging server state.
- **`MessageStream.tsx`: Bilingual TruncatedText** (`MessageStream.tsx`): "展开全文"/"收起" toggle now renders "Expand"/"Collapse" in English when `language !== 'zh'`. `language` prop threaded through all `TruncatedText` usages. Memoization condition expanded to re-render on content or timestamp change.
- **`RoundFeedbackDialog.tsx`: Finally Block** (`RoundFeedbackDialog.tsx`): Submission handler now has a `finally` block to ensure loading state is always cleared on error.

#### v0.14.2 (2026-03-16)

- **OBS/VAR Names Auto-Fix After Execute** (`bioinformatics_agent.py`, `sandbox.py`): After every `execute_code` tool call, the agent now runs a housekeeping snippet (via `execute(save_to_history=False, skip_safety_check=True)`) that iterates all globals and calls `obs_names_make_unique()` / `var_names_make_unique()` on every AnnData object. Also applied at data-load time in `_build_data_context()`. Prevents `InvalidIndexError` from duplicate indices after `ad.concat()`, subsetting, or merging.
- **`_WORKSPACE_DIR` Kernel Variable** (`sandbox.py`): Kernel init now defines `_WORKSPACE_DIR = Path('<workspace>')` as a pre-built pathlib.Path variable. Bioinformatics Agent system prompt instructs to always use this variable for file paths rather than string literals, preventing path resolution errors.
- **savefig Redirect** (`sandbox.py`): `matplotlib.figure.Figure.savefig` is monkey-patched at kernel startup to auto-redirect saves targeting the workspace root (e.g., `figure.png`) into `figures/` subdirectory. Agent no longer needs to manually manage figure output paths.
- **Numba Cache Isolation** (`sandbox.py`): Kernel init sets `NUMBA_CACHE_DIR` to `/tmp/numba_cache_{session_id}` to prevent cache collisions between concurrent sessions.
- **`open()` Write Safety Check Refined** (`sandbox.py`): Static code safety validator now allows `open()` write calls whose lines reference known safe path variables (`_WORKSPACE_DIR`, `working_dir`, `figures`, `tables`, `/tmp`), reducing false positives for legitimate file I/O that the runtime sandbox already enforces.
- **`execute()` New Parameters** (`sandbox.py`): `execute()` now accepts `save_to_history: bool = True` and `skip_safety_check: bool = False`, enabling internal housekeeping calls that bypass history recording and the static safety validator.
- **Checkpoint Round Validation** (`bioinformatics_agent.py`): Checkpoint loader now verifies that `checkpoint['round_number']` matches `state.current_round`. Mismatched checkpoints (from a previous round) are rejected, preventing cross-round state contamination on resume.
- **Fixed Step Budget on Resume** (`bioinformatics_agent.py`): Corrected budget calculation: `max_steps = original_total` (remaining steps = `original_total - step_count`). Previously, `max_steps = step_count + original_total` doubled the effective budget on resume.
- **`annotate_clusters` Must Follow Code Output** (`bioinformatics_agent.py`): System prompt now explicitly requires the agent to execute and observe marker-extraction code output before calling `annotate_clusters`, not in the same response. Added to both security restrictions and cell type annotation workflow steps.
- **Gene Module Scoring Guidance** (`bioinformatics_agent.py`): System prompt adds a `GENE MODULE SCORING` section: preserve `adata.raw` before HVG, check gene overlap, use `use_raw=True` for `sc.tl.score_genes`, investigate if overlap < 50%. Companion `SPECIES-AWARE GENE NAMES` section explains mouse (Title case) vs. human (UPPER CASE) naming conventions.
- **Security Restrictions Block in System Prompt** (`bioinformatics_agent.py`): New `SECURITY RESTRICTIONS` section explicitly lists `globals()`, `exec()`, `eval()`, `subprocess`, and bare-filename `open()` as blocked operations, reducing wasted steps from sandbox rejections.
- **`round_number` in Checkpoint Data** (`bioinformatics_agent.py`): Checkpoint dict now includes `round_number: state.current_round` so the validator added above can compare it on resume.
- **Iteration Tracking Reworked** (`bioinformatics_agent.py`): Replaced the `figure_only_iteration` boolean with three flags: `has_substantive_work`, `has_figure_work`, `has_register_score`. This enables finer-grained step-budget decisions.
- **Literature Agent: `annotate_clusters` Cache** (`literature_agent.py`): In-memory `_annotation_cache` dict keyed by `f"{organism}:{tissue}:{hash(frozenset(top_genes))}"`. Cache hits return the previous result without calling the LLM or Deep Research API.
- **Literature Agent: Empty Cluster Guard** (`literature_agent.py`): `annotate_clusters()` now returns an error dict immediately when `cluster_markers` is empty, without making any API calls.
- **Literature Agent: Previous Reports in Context** (`literature_agent.py`): Previous round `LiteratureReport` summaries are now appended to `finding_parts` in `annotate_clusters()`, giving the LLM context to avoid redundant searches.
- **Painter Agent: Token Budget Management** (`painter_agent.py`): Module-level constants `_TOTAL_CONTEXT_TOKENS = 65_536`, `_MAX_OUTPUT_TOKENS = 32_800`, `_SAFE_INPUT_TOKEN_LIMIT = 30_000` define the budget for `google/gemini-3-pro-image-preview`. `_estimate_tokens(text)` provides a rough estimate (len/3). Both `generate()` and `revise()` paths now estimate input tokens and truncate prompts/summaries when the limit is exceeded, with logged warnings.
- **Supervisor Agent: `_set_llm_context()`** (`supervisor_agent.py`): New helper sets `llm_client.set_context(session_id, "supervisor")` before each LLM call, ensuring the shared client is bound to the correct session and agent for logging.
- **`_normalize_tasks()` Utility** (`workflow.py`): New module-level function converts task items to strings. Handles plain strings, dicts with `task`/`description`/`topic`/`name` keys, and appends `rationale`/`recommended_methods`/`target_readouts` as inline annotations. Applied consistently across all task list usages in `workflow.py`.
- **`annotate_clusters_callback`: Confirmed Annotation Cache** (`workflow.py`): When `state.annotation_state.confirmed` is True, returns cached annotations from `cluster_annotations` immediately without invoking the literature agent pipeline.
- **`annotate_clusters_callback`: Empty Cluster Guard** (`workflow.py`): Returns an error dict immediately when `cluster_markers` is empty, with a message directing the agent to run `rank_genes_groups` first.
- **WebSocket Heartbeat** (`main.py`): `_listen_for_disconnect()` now sends a `{"type": "ping"}` message every 30 seconds (on `asyncio.TimeoutError`) to keep connections alive through proxies. Frontend silently ignores `ping` messages.
- **Thread-Safe WebSocket Send** (`main.py`): Both `message_callback` and `_send_ws_direct()` now acquire `ws_lock` for the entire check-and-send sequence, preventing concurrent coroutines from racing on a closing connection.
- **PI_DECIDE Resume: `max_rounds` Extension** (`main.py`): When `POST /api/session/{id}/resume` is called while the session is in `WAITING_INPUT`, if the user supplied a `[PI_DECIDE]` signal and `current_round >= max_rounds`, `max_rounds` is incremented by 1 before re-entering the workflow.
- **Figure Path Resolution** (`main.py`): Figure lookup now checks `fig_data.get('png_path', '')` first (explicit path stored when generating figures), falling back to workspace path conventions. Ensures figures saved with non-standard names are still served correctly.
- **Empty `cluster_markers` Guard in Annotation Endpoint** (`main.py`): The `/annotate` endpoint now returns HTTP 400 with a descriptive error when `cluster_markers` is empty, matching the guard added in `workflow.py`.
- **Exploration Report Dialog Timing Fix** (`AnalysisPanel.tsx`): Dialog suppression condition changed from `analysisPhase === 'innovation'` to `(currentRound ?? 0) > 1`. The backend sets `analysisPhase` to `'innovation'` at Phase 1 completion (before direction selection), which was incorrectly blocking the dialog before the user acted.
- **Painter Model Recommendation Updated** (`PromptsPanel.tsx`): Model selector tooltip for the Painter Agent now recommends `google/gemini-3-pro-image-preview` and `google/gemini-3.1-flash-image-preview` as models with image generation capability.
- **WebSocket `ping` Silently Ignored** (`useApi.ts`): Frontend WebSocket message handler now returns early on `data.type === 'ping'` without logging or state updates.

#### v0.14.1 (2026-03-15)

- **Human Input Wakes Workflow** (`main.py`): `POST /api/session/{id}/human-input` now calls `workflow.signal_continue()` when the session status is `WAITING_INPUT`. Previously the endpoint stored `pending_human_input` and `pending_uploaded_files` but never signalled the workflow, so the workflow would only resume after the countdown timeout. Both `/choose-direction` and `/resume` already called `signal_continue()`; `/human-input` was the only endpoint that didn't.
- **Stale `should_continue` Cleared on Round Start** (`workflow.py`): `_initialize_round()` now resets `state["should_continue"] = True` before incrementing the round counter. Fixes a scenario where `_check_convergence` wrote `should_continue=False` in one round, `_wait_for_user_input` extended `max_rounds` on user input, but `_should_continue` in the next round read the residual `False` value and stopped the analysis prematurely.
- **`current_round_direction` Set Before PI Instruction Call** (`workflow.py`): In multi-direction mode, `current_round_direction` is now assigned immediately after `directions_queue.pop(0)`, before calling `pi_agent.innovation_instruction()`. If the PI call raises, the direction is re-queued and `current_round_direction` is cleared to `None`. Previously, the assignment was after the try-except block, so retries saw the previous round's direction in `current_round_direction`.
- **Annotation Confirm Respects PAUSED Status** (`workflow.py`): After `_annotation_event.wait()` completes, status is only restored to `original_status` if the current status is not `PAUSED`. If the user paused the analysis during the annotation blocking window, the pause is now preserved correctly instead of being overwritten.
- **`multi_dir_completed` Uses `len() > 1`** (`workflow.py`): The guard preventing `max_rounds` extension after multi-direction completion now checks `len(analysis_state.user_chosen_directions) > 1` (true multi-direction mode) instead of truthiness. Single-direction mode also populates `user_chosen_directions` with a 1-element list, which was incorrectly blocking the extra-round extension when the user submitted input on the final round of a single-direction analysis.
- **Exploration Report Failure Sent to Frontend** (`workflow.py`): If `generate_exploration_report()` raises, an `"error"` WebSocket message is now sent to inform the user, rather than silently logging and continuing. Message text: `"Exploration report generation failed: {err}. You can still select a research direction based on the analysis findings above."`.

#### v0.14.0 (2026-03-15)

- **`register_score` Tool** (`bioinformatics_agent.py`, `schemas.py`): New tool for the Bioinformatics Agent to register how a score was computed. Tool arguments: `score_name`, `method`, `description`, `parameters`, `gene_list`, `gene_list_source`. Creates a `ScoreDefinition` Pydantic model, deduplicates by `score_name` in `AnalysisState.score_definitions`. Agent system prompt mandates a `consult PI → execute → register_score` workflow for all scoring operations (sc.tl.score_genes, gseapy, decoupleR, custom scoring). Score-only `register_score` calls do not consume a bioinformatics step (`step_count -= 1`).
- **Jupyter I/O Sandbox** (`sandbox.py`): Kernel `_initialize_environment()` now injects a second setup block that: (1) sets `os.chdir()` to the absolute workspace path; (2) creates `figures/` and `tables/` subdirectories; (3) configures `sc.settings.figdir`; (4) monkey-patches `builtins.open`, `pathlib.Path.mkdir`, `Path.write_text`, and `Path.write_bytes` to block writes outside allowed roots (session root, `/tmp`, system paths). Blocked writes raise `PermissionError` with a redirect message listing correct subdirectories.
- **Kernel Dead Detection** (`sandbox.py`): Added `_kernel_dead: bool = False` to `LocalJupyterSandbox`. Set to `True` when `restart_kernel()` raises an exception after a timeout. All subsequent `execute()` calls check this flag first and return `CodeExecutionResult(success=False, error="Kernel is unavailable — restart failed")` immediately.
- **Working Directory as Absolute Path** (`sandbox.py`): `self.working_dir = Path(working_dir).absolute()` — prevents relative-path issues when the kernel changes cwd internally.
- **Annotation Dialog: Alternative Cell Types & Low Confidence Reason** (`pi_agent.py`, `schemas.py`, `workflow.py`, `AnnotationConfirmationDialog.tsx`): PI Agent's `review_annotations()` prompt now requests `alternative_cell_types` (list) and `low_confidence_reason` (str) for medium/low confidence clusters. These fields are stored in `ClusterAnnotation` and propagated to the frontend via `annotation_confirmation` WebSocket message. The dialog shows expandable rows per cluster with this information, using chevron toggle icons.
- **Annotation Marker Supplementation** (`workflow.py`): Before sending annotations to PI for review, `key_markers` lists shorter than 10 genes are supplemented from `cluster_markers` (rank_genes_groups top genes), ensuring PI and user always see at least 10 genes per cluster.
- **Annotation Context HTML Report** (`workflow.py`, `report_generator.py`, `main.py`): `generate_annotation_context_report_html()` produces a standalone HTML annotation context report during the annotation pause. Saved to `reports/{session_id}/annotation_context_report.html`; path stored in `state.annotation_report_html_path`. New endpoint `GET /api/session/{id}/annotation-report/html` serves it. `annotation_report_url` included in session load API response. Annotation dialog shows a download button linking to this report.
- **Per-Round Report Narrative** (`supervisor_agent.py`): `_build_analysis_narrative()` now groups records into per-round blocks by detecting step number resets, emitting `ROUND N (PHASE)` headers with the research direction for innovation rounds. Exploration report direction candidates and user-chosen directions are appended to the narrative. Context limit raised from 60k to 80k characters.
- **Dynamic Report Sections** (`supervisor_agent.py`): `_build_report_prompt()` now builds section templates dynamically. If any innovation round exists, the template includes `[exploration_findings]`, one `[innovation_N]` section per direction (replacing `[key_findings]`), and `[integrated_findings]`. Innovation sections are marked as CRITICAL with 300-500 word minimum and requirement to include all figures, genes, pathways, and quantitative statistics. Publication figures are included in `_collect_figure_paths()` alongside supervision records.
- **Thread-Safe Session Cache** (`main.py`): `_update_session_cache(session_id, state)` acquires `_state_lock` before updating `sessions[session_id]` and `_session_access_times[session_id]`. All direct `sessions[id] = state` assignments (~15 call sites) replaced with `await _update_session_cache(session_id, state)`.
- **Innovation Phase User Guidance Merging** (`workflow.py`): In `_pi_planning()`, `adjusted_instructions` from user input now only replaces direction-based planning in round 1 (`current_round <= 1`). In round 2+, adjusted instructions are merged as supplementary `[User guidance]` tasks appended to existing task lists after innovation-phase planning completes.
- **Direction Pop Persistence** (`workflow.py`): After `directions_queue.popleft()`, the workflow immediately calls `await self.state_callback(analysis_state)` to persist the updated queue. Prevents direction loss if the process crashes between the pop and round completion.
- **Message Trim Orphan Prevention** (`bioinformatics_agent.py`): The 80-message trim now walks the cut point backward until the message at `cut` is not a `tool` role message. This avoids orphaning tool result messages that require a preceding assistant message with the matching `tool_use` block, which causes API errors on some providers.
- **Direction Validation** (`main.py`): `POST /api/session/{id}/choose-direction` now validates that at least one of `direction`, `directions` (non-empty list), or `let_pi_decide` is provided, returning HTTP 400 with a clear message if not.
- **Resume Without Direction Guard** (`main.py`): When resuming a `WAITING_INPUT` session in innovation phase (round ≥ 1) without a `user_chosen_direction` or `directions_queue`, the handler defaults `user_chosen_direction = "[PI_DECIDE]"` so analysis does not run directionless.
- **Clear Direction Queue** (`main.py`): `HumanInputRequest` gains `clear_queue: bool = False`. When set, `directions_queue` is cleared and `max_rounds` is capped to `current_round + 1`, allowing users to abort the multi-direction sequence and jump to report generation.
- **User Override Sanitization** (`main.py`): Cell type name overrides in `POST /api/session/{id}/confirm-annotations` are now sanitized: strips `{}[]\\;` `` ` `` `$` `__dunder__` `import`, `exec`, `eval` patterns and caps length at 200 characters. Prevents code injection when the override value is embedded in generated Python code.
- **Literature Search Mode Tracking** (`literature_agent.py`): All paths in `search_literature()` now set `result["search_mode"]` to the actual mode used (including fallback chain, e.g., `"pubmed_llm (fallback from gemini_deep_research)"`). Previously only the final fallback paths set this field.
- **Innovation Phase Analysis Requirements** (`workflow.py`, `bioinformatics_agent.py`): The innovation phase bioinformatics system prompt now lists 10 concrete requirements: execute all PI tasks, ≥1 figure per task, use specified advanced methods, add exploratory analyses, include volcano + heatmap for DEG, bar plot for pathways, UMAP + violin/box for scoring, report detailed statistics, consult PI before scoring, call `register_score` after scoring.
- **`RoundResult.direction` Field** (`schemas.py`): New `direction: str = ""` field on `RoundResult`. Set at PI synthesis time from `analysis_state.current_round_direction`. Used by supervisor agent report prompt and narrative to label which direction each innovation round investigated.

#### v0.13.4 (2026-03-13)

- **Pause/Resume Deadlock Fix** (`workflow.py`): `_wait_for_user_input` now checks `analysis_state.status != WAITING_INPUT` before entering the blocking `asyncio.Event.wait()` in both manual mode (`wait_seconds == -1`) and countdown mode (`wait_seconds > 0`). If status is not already `WAITING_INPUT`, it is force-set and synced via `_send_state_sync()`. Fixes a deadlock where Resume restored status to `RUNNING`, leaving the frontend with no waiting-input UI while the backend remained blocked on `_continue_event.wait()`.
- **Multi-Direction max_rounds Guard** (`workflow.py`): The `max_rounds` extension on last-round user input now checks whether multi-direction mode has completed all directions (`not directions_queue and user_chosen_directions`). If so, the extension is skipped to prevent a spurious extra round after multi-direction correlation analysis is already done.
- **Retry `status: "failed"` Field** (`workflow.py`): `@with_retry` decorator's exhaustion return dict now includes `"status": "failed"` alongside `"error"` and `"findings": []`. This allows PI synthesis to distinguish failed agents from skipped agents (which return `"status": "skipped"`).
- **Round 3+ Task Extraction Fallback** (`workflow.py`): Replaced the single-fallback (pi_summary only) with a two-tier fallback: (1) if prefix-tagged tasks are absent from `next_steps`, assign the raw `next_steps` list to bioinformatics; (2) if `next_steps` is also empty, fall back to `pi_summary`. Prevents silent empty task lists when LLM does not use the `[Agent]` prefix format.
- **Bioinformatics Message History Cap** (`bioinformatics_agent.py`): Added a `MAX_MESSAGES = 80` trim before each `chat_with_tools()` call: `messages = [messages[0]] + messages[-(MAX_MESSAGES-1):]`. Preserves the system message and the most recent 79 messages. Prevents unbounded context growth during 200-step autonomous analyses.
- **Supervisor Sampled LLM Evaluation** (`supervisor_agent.py`): LLM evaluation in `evaluate_step()` is now gated by `should_llm_evaluate`: True only when `step.code_error` exists, `figures` is non-empty, `step.step_type == "annotate_clusters"`, or `self._step_count % 5 == 0`. All other steps receive rule-based checks only (`_rule_check()`), reducing per-step LLM calls by ~80%.
- **Report Generator Dead Code Removal** (`report_generator.py`): Deleted ~700 lines of unreachable code: `_setup_environment()` (never called), `_auto_load_processed_data()` (only called by the above), `_extract_cluster_annotations()` sandbox version (only called by the above), and `_generate_section_autonomous_legacy()` (explicitly marked legacy, never called). Removed `self.sandbox: Optional[LocalJupyterSandbox] = None` field and the now-unused `from .utils.sandbox import LocalJupyterSandbox` and `get_session_workspace` imports.
- **Pre-existing Syntax Fix** (`workflow.py`): Fixed a merged comment+`if` statement (`# comment            if analysis_state:`) in the Resume path (~line 2698) that caused `IndentationError` under Python 3.13's stricter parser.
- **Documentation**: Updated README and HANDOVER with accurate two-phase analysis flow (7 sequential steps, bioinformatics sub-flow, annotation confirmation blocking, multi-direction dispatch), a new bilingual workflow graph, and renumbered/expanded Key Features list.

#### v0.13.1 (2026-03-12)

- **Multi-Direction Research Workflow**: `AnalysisState` gains `user_chosen_directions` (list of `{"direction", "notes"}` dicts), `directions_queue` (FIFO queue popped one-by-one by `_pi_planning()`), and `multi_direction_correlation`. When PI decides or user selects multiple directions, `select_best_directions()` (replaces `innovation_instruction_pi_directed()`) returns 2 best directions; these are enqueued and processed in sequential innovation rounds. `wait_for_user_input` auto-continues (skips countdown) while `directions_queue` is non-empty. After the queue is exhausted and `multi_direction_correlation is None`, the workflow triggers `pi_agent.analyze_direction_correlations()` and broadcasts a `direction_correlation` WebSocket message.
- **Enhanced Exploration Report Direction Candidates**: `research_direction_candidates` entries now require five fields: `direction`, `rationale`, `expected_innovation` (High/Medium/Low + evidence), `expected_novelty` (High/Medium/Low + explanation), `data_reliability` (High/Medium/Low + data citations). PI prompt enforces all 5 fields.
- **Innovation Instruction Notes**: `pi_agent.innovation_instruction()` accepts optional `notes: str` parameter; notes are injected as a "User Notes for This Direction" section in the planning prompt.
- **Multi-Direction API** (`main.py`): `POST /api/session/{id}/choose-direction` now accepts `directions: List[Dict[str, str]]` (multi-direction) in addition to the original `direction: str` (single direction, backward compat). Single direction is normalized to multi-direction format internally. `max_rounds` is set to `len(directions) + 1`.
- **Exploration Report HTML** (`report_generator.py`): Added `generate_exploration_report_html()` — a pure-Python (no LLM) HTML renderer for the Phase 1 Exploration Report. Called at end of Round 1; path stored in `state.exploration_report_html_path`. New endpoint `GET /api/session/{id}/exploration-report/html` serves the file. URL passed via `exploration_report` WebSocket message as `html_url`.
- **Redesigned ExplorationReportDialog** (`frontend/`): Multi-select checkboxes for direction candidates with per-direction notes fields. Score badges (`ScoreBadge` component) for innovation/novelty/reliability. Custom direction entries (add/remove). Built-in countdown timer (wall-clock based, same as RoundFeedbackDialog). Link to standalone exploration report HTML. Submits `SelectedDirection[]` to `onChooseDirections()`.
- **Bioinformatics LLM Error Status**: LLM failures are captured in `llm_error: Optional[str]` and result in `status = "error"` (not `"max_steps_reached"`), `success: False`, and error string in result dict for upstream error handling.
- **Session Eviction Race Fix** (`main.py`): `cleanup_stale_sessions()` now snapshots `access_time_at_capture` before I/O. After saving, if the access time changed (session was re-activated concurrently), eviction is aborted. Removes a window where recently-resumed sessions could be incorrectly deleted from memory.
- **Sidebar Max Rounds Removed**: Removed the "Analysis Rounds" input/slider from the sidebar. `max_rounds` is now dynamically managed by the multi-direction queue system and is no longer user-editable directly.
- **New WebSocket Type**: `direction_correlation` rendered in MessageStream with direction tags and a formatted pre-block for the correlation narrative.

#### v0.13.0 (2026-03-12)

- **Two-Phase Analysis Workflow**: Restructured the analysis workflow into two distinct phases. Round 1 is the "Exploration Phase" — a comprehensive full-landscape analysis. Round 2+ is the "Innovation Phase" — focused, hypothesis-driven analysis guided by a user-selected or AI-determined research direction. New `AnalysisPhase` enum (`"exploration"` / `"innovation"`); `AnalysisState.analysis_phase` tracks current phase.
- **Cell Annotation Confirmation (WAITING_ANNOTATION)**: After the Bioinformatics Agent completes cluster annotation in Phase 1, the workflow pauses with a new `TaskStatus.WAITING_ANNOTATION` status. A WebSocket `annotation_confirmation` message is sent to the frontend, containing the full list of `ClusterAnnotation` objects (cluster_id, cell_type, confidence, key_markers, evidence). Frontend displays `AnnotationConfirmationDialog` where users can confirm, edit, or override low-confidence annotations. The `_annotation_event` asyncio event in `workflow.py` blocks the workflow until confirmation is received. Low-confidence clusters block auto-confirm; high/medium-only results auto-confirm after a countdown.
- **Exploration Report & Direction Selection**: At the end of Phase 1, `pi_agent.generate_exploration_report()` produces a structured `ExplorationReport` (cluster_overview, data_quality_warnings, strongest_heterogeneity, key_pathway_changes, abnormal_markers, pi_direction_assessment, research_direction_candidates). A WebSocket `exploration_report` message sends this to the frontend, which displays `ExplorationReportDialog`. Users choose a research direction (from AI candidates or custom text) or let PI decide. The chosen direction is stored in `AnalysisState.user_chosen_direction` and consumed at the start of Round 2.
- **New Data Models** (`schemas.py`): `ClusterAnnotation` (cluster annotation with confidence + user override), `AnnotationConfirmationState` (annotation pause state tracker), `ExplorationReport` (structured Phase 1 output with direction candidates). `RoundResult` gains `phase` and `exploration_report` fields.
- **New Frontend Components**: `AnnotationConfirmationDialog.tsx` (cell annotation review with editable cell types, confidence badges, auto-countdown); `ExplorationReportDialog.tsx` (exploration findings display, direction candidate selection, custom direction input, "Let PI Decide" option).
- **Backend Wiring**: `bioinformatics_agent.py` calls `annotate_clusters_callback()` which routes through `workflow.py`'s `_request_annotation_confirmation()` to pause and await user input. `main.py` adds `WAITING_ANNOTATION` to valid resumable statuses and provides the annotation confirmation submit endpoint.

#### v0.12.4 (2026-03-11)

- **Session-Centric Data Organization**: Restructured `data/` from type-based layout (`uploads/`, `states/`, `workspace/`, `reports/` as top-level dirs) to session-centric layout (`sessions/{session_id}/` containing all subdirs). Each session now has its own isolated directory with `workspace/`, `reports/`, `uploads/`, `states/`, and `logs/` subdirectories.
- **Path Helper Functions**: Added `get_session_dir()`, `get_session_workspace()`, `get_session_states()`, `get_session_uploads()`, `get_session_reports()`, `get_session_logs()`, `get_session_metadata_path()`, and `ensure_session_dirs()` to `config.py`.
- **Backward Compatibility**: All path resolution falls back to legacy directories when new-structure files are not found. Session listing, loading, and deletion handle both layouts.
- **Migration Script**: Added `backend/migrate_to_session_dirs.py` with `--dry-run` support to migrate existing data. Handles session metadata, uploads, workspace, reports, state files, and LLM logs. Remaps embedded absolute paths in session JSON.
- **Files Changed**: `config.py`, `main.py`, `workflow.py`, `sandbox.py`, `llm_logger.py`, `report_generator.py`.

#### v0.12.3 (2026-03-10)

- **Report Markdown Rendering Fix**: Enhanced `markdown_to_html()` in `report_generator.py`, `supervisor_agent.py`, and `regenerate_report_html.py` to convert markdown headers (`#`–`####`) to `<h1>`–`<h4>` HTML tags and added bullet list (`- item` → `<ul><li>`) support. Previously headers appeared as raw text like `<p>### 3. Conclusions</p>` in generated HTML reports.
- **Conclusions/Abstract Prompt Optimization**: Updated LLM prompts in `_generate_conclusions()` and `_generate_abstract()` to explicitly instruct LLM not to generate redundant section headers (report template already provides them). Added defensive regex cleanup to strip any remaining `### Conclusions` or `**Abstract**` prefixes from LLM output.
- **Frontend ReactMarkdown Rendering**: `ReportPanel.tsx` Detail view now renders abstract, section content, and conclusions via `ReactMarkdown` component (already installed as dependency) instead of raw `whitespace-pre-wrap` text, properly displaying bold, italic, headers, and lists.
- **Frontend Appendix Display**: Added `ReportAppendix` interface and appendix rendering section to `ReportPanel.tsx` showing project info, agent model configurations with temperatures, data files, treatment groups, and user instructions in a styled card layout.
- **Supervisor Agent HTML Appendix**: Added `_build_appendix_html_section()` static method to `SupervisorAgent`, inserting a full appendix (project info, agent config, data files, treatment groups, analysis rounds) into the HTML report between conclusions and footer.

#### v0.12.2 (2026-03-10)

- **Analysis Stop/Pause Support**: Added UI buttons and backend endpoint (`POST /api/session/{session_id}/stop`) to pause or terminate analysis. Support immediate termination (pause/save) or termination with summary (stop-and-summarize report generation). Implemented task tracking (`_run_tasks`) and cancellation support for workflow execution.
- **Publication-Quality Bioinformatics Figures**: Added mandatory checklist (UMAP axis labels, font sizes, legend placement) and strict composition barplot rules to bioinformatics agent's system prompt. Updated default matplotlib/seaborn themes in sandbox for professional "publication" aesthetics.
- **Figure Revision Auto-Retry**: Bioinformatics Agent now automatically attempts to fix runtime errors during figure code revision once using the error message as context, improving reliability of figure generation.
- **Progressive Figure Review Leniency**: PI Agent's figure review now employs progressive leniency (from revision 2+) to focus on scientific accuracy over minor aesthetics, avoiding infinite revision loops for minor cosmetic issues.
- **WebSocket Reconnection Improvements**: WebSocket handler now updates the global reference for active workflows, ensuring users can reliably observe real-time progress after a connection drop or refresh.
- **UI UX Polish**: Feedback dialog "Skip" button renamed to "Continue" for clarity; added Stop/Summarize buttons to feedback dialog and Analysis panel.

#### v0.12.1 (2026-03-10)

- **Centralized Code Safety**: Moved safety validation to `sandbox.py`, protecting all components (Bioinfo, Report, Chat).
- **Report Resumption**: Added `CheckpointManager` for report generation, allowing resumption from last completed step.
- **Race Condition Fixes**: Serialized session saves to prevent JSON mutation; improved session eviction logic to prevent re-activation conflicts.
- **Memory Optimization**: Added message caps and base64 stripping for realtime message history.

#### v0.12.0 (2026-03-10)

- **Supervisor Agent**: Real-time monitoring of bioinformatics analysis steps for issue detection and PI escalation.
- **Bug Fixes**: Fixed LLM retry logic (InternalServerError only), settings sync, WebSocket version conflict recovery, and UUID generation in non-HTTPS contexts.
- **UX Improvements**: Expand/collapse buttons for truncated messages; last-round guidance indicators; automatic cleanup of partial failed uploads.

#### v0.11.0 - v0.11.1 (2026-03-09)

- **Sequential Execution**: Reordered agents to Biochemistry → Pathology → Bioinformatics → Literature for better context flow.
- **Dynamic PI Planning**: PI Agent refines downstream tasks based on completed results within the same round.
- **Cross-Agent Context**: findings shared between agents in real-time (e.g., Biochemistry results injected into Bioinfo).
- **Session Improvements**: Deleting session now removes all disk artifacts; fixed duplicate figure generation via upsert logic; improved session persistence timing.

#### v0.10.3 (2026-03-09)

- **Comprehensive Analysis Log**: New `AnalysisStep` Pydantic model records every bioinformatics agent step (7 types: `execute_code`, `consult_pi`, `search_literature`, `annotate_clusters`, `report_findings`, `generate_publication_figure`, `text_response`). Each step captures agent reasoning, timestamps, tool-specific fields (code/output/error, PI Q&A, literature topics, cluster markers, figure metadata). Stored in `AnalysisState.analysis_log` and `BioinformaticsCheckpoint.analysis_log`. Deduplication by step ID handles node retries
- **Analysis Log in Reports**: Reports include a full chronological analysis log timeline section (collapsible code blocks, inline figure thumbnails, PI consultations, literature results). Implemented in both `report_generator.py` (`_build_full_analysis_log` + `_build_analysis_log_html`) and `regenerate_report_html.py` (`_build_analysis_log_from_report`). Falls back to `code_history` for older sessions without `analysis_log`
- **Figure & Table Scientific Interpretation**: Added `interpretation` field to `PublicationFigure` and `PublicationTable` models. Reports render styled interpretation sections (blue for figures, green for tables) below each figure/table
- **PI Agent Memory Optimization**: `reset_history()` called before `synthesize_round()` and `generate_final_summary()` — these methods embed full state context in their prompts, making prior chat history unnecessary and saving tokens
- **Bioinformatics Agent Code History Cap**: `MAX_CODE_HISTORY = 100` prevents unbounded memory growth across multi-round sessions. Figure attempt counter (`_figure_attempts`) reset between rounds so previously failed figures can be retried with updated data
- **Workflow Agent Auto-Activation**: When PI's `next_steps` contain prefixed tasks for an agent (e.g., `[Pathology] ...`) but the agent was not included in `agents_for_next_round`, the workflow now auto-activates that agent with a log message
- **Pathology → Bioinformatics Context Sharing**: Current round pathology findings (up to 5) are now included in the bioinformatics agent's instruction prompt for cross-agent collaboration
- **Code History Deduplication by ID**: `_post_bioinformatics()` now deduplicates code history entries by `CodeExecution.id` instead of equality comparison, correctly handling node retries that produce identical objects
- **Deep Research Refactoring**: `is_deep_research_available` property added to `LiteratureAgent`, replacing inline checks in `workflow.py`. PubMed gap fallback now temporarily switches agent mode to ensure `_do_research()` routes correctly
- **Report Section Mapping Fix**: Removed "annotation" from `clustering_overview` and `key_findings` figure category lists to prevent duplicate figure placement
- **Task Prefix Parsing Refactor**: Replaced 4 `if/elif` branches with a `prefix_map` dict loop for cleaner next_steps task extraction

#### v0.10.2 (2026-03-08)

- **Sandbox Kernel Timeout Recovery**: When kernel fails to return to idle after interrupt, now performs full recovery: `restart_kernel()` via `asyncio.to_thread`, client reconnection (`stop_channels` → new `client()` → `start_channels`), `_wait_for_ready`, `_initialized = False` → `_initialize_kernel()` → `_restore_session()`. Previously left a zombie client connection causing all subsequent executions to hang
- **Figure plt.close/plt.show Safety**: Temporarily disables `plt.close` and `plt.show` during code execution to prevent premature figure closure before auto-save. Wrapped in `try/finally` to guarantee restoration even if user code throws exceptions. Applied to all 3 code templates: `execute_code`, `generate_publication_figure`, and figure revision
- **Blank Figure Detection**: Publication figures with file size < 2KB are flagged as blank. Applied to both initial generation and PI-requested revisions. System prompt updated to instruct LLM not to call `plt.close()`/`plt.show()`/`plt.savefig()` in figure code
- **Sandbox Timeout**: Increased from 300s (5 min) to 1800s (30 min) to accommodate long-running bioinformatics analyses (pathway analysis, large dataset processing)
- **Architecture Documentation**: Updated HANDOVER.md and README.md diagrams to reflect current sequential execution model (Pathology → Bioinformatics → Biochemistry) with Bioinformatics consulting PI and Literature via callbacks

#### v0.10.1 (2026-03-07)

- **Literature Fallback Model**: Added `fallback_model` parameter to `LiteratureAgent`. Quick search fallback now uses the PI model instead of the literature agent's own deep research model, reducing costs
- **Deep Research max_tokens Fix**: Changed deep research call from `max_tokens=262144` to `max_tokens=None`; `OpenAICompatibleClient.chat()` `max_tokens` parameter is now `Optional[int]`, omitting the parameter when `None`
- **Deep Research Model Passthrough**: `deep_research_model` is now passed from `lit_config` when initializing literature agent in chat mode
- **Upload Cancellation**: Added cancel button (X icon) for upload queue items in `DataPanel.tsx`. Supports aborting in-progress uploads via `AbortController` and removing pending items from queue. `uploadData` in `useApi.ts` now accepts optional `AbortSignal`

#### v0.10.0 (2026-03-07)

- **Deep Research Mode (OpenRouter)**: Added `deep_research` literature mode that calls deep research models (e.g., `openai/o4-mini-deep-research`, `perplexity/sonar-deep-research`) via standard OpenRouter chat API. Single long-running call with 30-min timeout and 256K max tokens. Three modes now coexist: `deep_research` (default), `gemini_deep_research`, `pubmed_llm`
- **Fallback Chain**: `deep_research` → `pubmed_llm` → `quick_search`; timeout propagation to `retry_with_backoff` with `max_retries=0` for requests > 300s
- **LLM-Based Query Generation**: Removed hardcoded `_NON_GENE_WORDS` blacklist and regex-based species/tissue/gene extraction. Query generation now delegates entity identification entirely to the LLM
- **Model Upgrades**: Bioinformatics agent default model upgraded from `anthropic/claude-sonnet-4.5` to `anthropic/claude-sonnet-4.6`
- **Cross-Agent Context**: Deep research prompt now includes `current_round_findings` from pathology/bioinformatics agents in the same round
- **Bug Fixes**: Fixed useEffect overriding user's manual literature mode selection; fixed skip_nodes returning stale previous-round results; fixed fallback task assignment not activating all agents; added `Literal` type validation for literature mode field

#### v0.9.1 (2026-03-07)

- **Cross-Agent Context Passing**: Biochemistry Agent now receives current round's pathology and bioinformatics findings (not just previous round), enabling real-time cross-referencing. Literature Agent receives current round findings from all other agents for better-informed PubMed query generation
- **PubMed Search Query Degradation**: New `_validate_and_degrade_query()` method progressively simplifies PubMed queries that return zero results (strip field tags → simple keywords), reducing empty search failures
- **LLM-Driven Research Context**: Replaced hardcoded keyword matching (`_build_research_query`) with `_build_research_context()` that passes raw experimental context to LLM for entity extraction, producing more relevant PubMed queries
- **PubMed Query Refinement Improvements**: Search evaluation now includes truncated abstracts for better relevance judgment, explicitly lists previous queries to prevent duplication, and provides clearer evaluation criteria
- **PI Prompt Optimization**: Literature search topic guidance in PI Agent prompts now requests focused, searchable research questions with biological context instead of broad topics or raw gene lists
- **Skip Nodes Bug Fix**: Fixed stale `skip_nodes` carrying over between rounds when user provides adjusted instructions — agents that should run could be incorrectly skipped in round 2+. Added `skip_nodes: []` reset in both `_initialize_round` and `_pi_planning` early return
- **Skipped Node Result Clarity**: `@with_retry` decorator now returns explicit `{"findings": [], "status": "skipped"}` for skipped nodes instead of passing through stale previous-round results
- **Race Condition Fix**: Moved `_continue_event.clear()` from `_wait_for_user_input` to `_pi_synthesis`, preventing a race where frontend signals between status announcement and event wait
- **PI Synthesis Error Awareness**: PI synthesis now receives error/status context from each agent (skipped, no_data, error) instead of just empty findings, enabling better-informed round summaries
- **Literature Findings in Round Summary**: Round summary `findings` array now correctly includes `*round_result.literature_findings` (previously omitted)
- **Fallback Task Assignment**: When PI's `next_steps` contain no prefixed tasks (e.g., `[Bioinformatics] ...`), system falls back to PI summary as context for all agents, preventing empty-instruction rounds. Sets `fallback_all_agents = True` to activate all agents
- **Human Guidance Source Fix**: Bioinformatics Agent now reads human guidance from `state["human_guidance"]` (set by `_initialize_round`) instead of `analysis_state.pending_human_input` which may already be cleared
- **Max Tokens Unification**: Unified `max_tokens` default from 4096 to 32768 across all LLM calls (AgentConfig, OpenAICompatibleClient.chat/chat_with_tools/chat_stream/chat_with_vision, LLMInteraction, report_generator._pi_chat, main.py chat endpoints)
- **Bioinformatics Agent Improvements**: Reset `text_response_count` after substantive tool calls to prevent false "stuck in text loop" detection; new `completed_with_fallback` status distinguishes normal completion from fallback extraction; checkpoint cleared on `completed_with_fallback`
- **JSON Parse Failure Logging**: Biochemistry and Pathology agents now log warnings when `_safe_parse_json` returns `raw_response` key, improving debuggability

#### v0.9.0 (2026-03-04)

- **PI Analysis Review Before Report**: Report generation now includes a PI review pre-step — PI reads the actual Jupyter notebook, views figures via vision API, and reads table summaries before writing. Produces a grounded methodology/results summary used throughout the report narrative
- **Report Appendix**: HTML reports now include an amber-styled appendix section with project information, agent configuration (model + temperature per agent), uploaded data summary (treatment groups, files, biochemistry indicators), analysis round table, and user inputs between rounds
- **Graphic Abstract Revision Loop**: Graphic abstracts now go through a 3-phase process — initial generation, up to 2 PI review+revision rounds (Painter revises with accumulated feedback history), and PI selects the best candidate from all versions using vision comparison
- **Cumulative Figure Revision History**: Publication figures now track all rounds of PI feedback in `revision_history`. The code reviser sees all previous rounds to avoid re-introducing previously fixed issues
- **In-Place Session Editing**: Session name and instruction can be edited directly in the sidebar via a pencil icon, saved on blur via `PUT /api/session/{id}/info`
- **Two-Phase Export**: Export now uses a prepare+download flow — server builds the archive first (showing file size), then streams the download. Replaces the old single-step export
- **Upload Queue with Per-File Progress**: Multiple files can be queued for upload with individual progress bars. The drop zone stays active during uploads. Failed uploads show error tooltips
- **"Other" Sequencing Type**: Supports bulk RNA-seq and custom sequencing formats with no file extension restriction
- **Sample Count Safety Guard**: Backend rejects reducing a group's sample count below the highest sample number that already has data; UI shows the error and reverts the field
- **Workspace Figure Downloads**: Figures generated in the Jupyter workspace (not just publication figures) can now be downloaded as PDF/PNG via the API
- **Figure Gallery API Fallback**: After page refresh, figures without cached base64 data are loaded from the API automatically
- **Deep Research Cost Warning**: Both ConfigPanel and PromptsPanel show an amber warning about Deep Research time (~60 min) and cost (~$2-5/task) with a link to Google docs

#### v0.8.9 (2026-03-03)

- **Import Proxy Timeout Resilience**: Frontend pre-generates `import_id` before upload and sends it with the request, so even if a reverse proxy (nginx) timeout prevents the HTTP response from reaching the client, it can still poll for extraction status. Backend registers the import job before saving the file, ensuring the polling endpoint is available immediately
- **Import UX Improvements**: Success/failure states with proper icons (CheckCircle for success, AlertCircle for errors). Polling includes retry logic that tolerates transient network errors and early 404s. 30-minute timeout for very large archives. Success confirmation dialog replaces silent dismissal
- **Branding Update**: App subtitle updated to "Integrated & Intelligent Single Cell Analysis" / "整合与智能单细胞分析" across footer, About dialog, Disclaimer dialog, and i18n translations

#### v0.8.7 (2026-03-02)

- **Data Page Scroll Fix**: Added `overflow: hidden` on `html, body` to prevent page-level scrolling when Data panel content overflows — standard fix for full-screen SPA
- **Footer i18n**: Footer copyright text now supports Chinese/English switching
- **About Page Update**: Updated version number and date

#### v0.8.6 (2026-03-02)

- **Async Session Import**: Large tar.gz imports no longer block the HTTP response (avoiding nginx proxy timeouts). Backend returns `import_id` immediately after upload, extracts in background; frontend polls `GET /api/session/import/{import_id}` for status
- **First-Visit Disclaimer Dialog**: Shows an AI reliability warning on first visit (stored in localStorage), requiring user acknowledgment before entering the system
- **About Button Text**: Added "About"/"关于" text label next to the info icon
- **Footer Copyright**: Added persistent footer with project name and copyright notice

#### v0.8.5 (2026-02-28)

- **Tool Message History Fix**: Bioinformatics Agent now appends one assistant message before processing all tool calls, instead of each handler branch appending its own copy. Preserves Gemini's `thought_signature` via `raw_assistant_message` field for correct multi-turn conversations
- **AI-Curated Brief Figures**: New `PI.select_brief_figures()` method uses AI to curate which figures appear in the Results tab brief report — excludes QC/diagnostic plots and intermediate versions, keeps key results that directly answer the research question
- **Workspace Figure Collection**: `_collect_session_figures()` now scans both `figures/` subdirectory and workspace root, capturing UMAP and other plots saved via explicit `plt.savefig()` that previously only appeared in the detailed report
- **Brief Report Figure Expansion**: The integrated report phase now includes AI-selected workspace figures (not just publication figures) in the Results display, with automatic image compression and base64 encoding
- **Image MIME Type Detection**: LLM client now detects actual image format (JPEG/PNG/GIF/WebP) from base64 magic bytes instead of always sending `image/png`, fixing vision API errors for JPEG pathology images
- **Export UX**: Session export now uses browser-native GET-based download instead of streaming POST, eliminating memory buffering for large archives. UI shows packaging state → download-started confirmation with dismiss button

#### v0.8.4 (2026-02-21)

- **About Dialog**: Added About page accessible from the header, showing project info, author, affiliation, contact, and GitHub link

#### v0.8.3 (2026-02-21)

- **Manual Round Wait Fix**: Fixed `_wait_for_user_input` ignoring `round_wait_seconds` setting — previously hardcoded 600s timeout caused auto-proceed even in manual mode (`-1`). Now respects all modes: manual (wait indefinitely), no-wait (proceed immediately), countdown (frontend timer + 60s backend buffer)

#### v0.8.2 (2026-02-21)

- **Default Model Update**: PI, Pathology, and Biochemistry agents upgraded from `google/gemini-3-pro-preview` to `google/gemini-3.1-pro-preview`
- **Proxy Fix**: Fixed 4 `OpenAICompatibleClient` instances (report generation, chat figure generation, PI chat, workflow report) missing explicit `proxy_url` parameter — previously relied on fragile environment variable fallback

#### v0.8.1 (2026-02-20)

- **Figure Data Fidelity**: Fixed publication figure generation hallucinating non-existent treatment groups. PI figure review and bioinformatics code revision now receive actual experimental design context (`build_experimental_design_summary()`), with explicit constraints to only reference uploaded groups
- **Vision-Based Figure Revision**: Bioinformatics Agent can now see its own generated figure during revision via `chat_with_vision()` (previously "blind revision" using only PI text feedback and original code). PI feedback + current figure image are now both provided to the revision LLM
- **Data Constraint Propagation**: Added `CRITICAL DATA CONSTRAINT` to `_build_data_context()`, `_revise_figure_code()` prompt, `generate_publication_figure` tool description, and PI `review_figure()` prompt — covering the full pipeline from initial code generation through review and revision
- **Data Fidelity Review Criterion**: PI figure review now includes a 6th criterion ("Data Fidelity") checking that figures show only actual experimental groups

#### v0.8.0 (2026-02-16)

- **Unified Data Model**: Replaced per-type data models (`PathologyImage`, `BiochemistryResult`/`BiochemistryFile`, `control_seq_path`/`treatment_seq_path`) with unified `DataFile` (section: pathology/sequencing/biochemistry), `TreatmentGroup`, and `BiochemistryIndicator` models
- **Treatment Groups & Multi-Sample Support**: Users can define experimental groups (e.g., Control, Treatment, NASH Model) with configurable sample counts. All uploaded data is assigned to a group and sample number, enabling structured multi-group comparisons. `build_experimental_design_summary()` generates text descriptions for agent prompts
- **Unified Upload Endpoint**: Replaced separate `/upload/pathology`, `/upload/snrna_seq`, `/upload/biochemistry` endpoints with single `/upload/data` endpoint. New group CRUD endpoints (`/groups`)
- **Upload Selectors in Feedback & Chat**: Round Feedback Dialog and Chat Panel file uploads now include treatment group, sample number, and data type selectors (matching DataPanel's existing pattern)
- **Simplified Pathology Types**: Pathology staining type options simplified to H&E and Other (with custom stain name input for Other)
- **DataPanel Rewrite**: Complete rewrite of DataPanel for group-centric data organization with inline editing, collapsible sections, and per-file download

#### v0.7.5 (2026-02-15)

- **Selective Agent Activation**: From round 2 onwards, PI Agent autonomously decides which agents (Pathology, Biochemistry, Literature) to activate based on analysis needs. Bioinformatics Agent always runs. Skipped agents receive WebSocket notification. Uses existing `skip_nodes` mechanism in `@with_retry` decorator. New `agents_for_next_round` field in `RoundResult`
- **Docker Pip Install Support**: Bioinformatics Agent can now install pip packages via subprocess in Docker environments. `_validate_code()` adds special exception for pip/pip3 install commands. System prompt includes package installation guidelines

#### v0.7.4 (2026-02-14)

- **Figure Step Budget**: Figure generation (`generate_publication_figure`) no longer counts toward bioinformatics agent's step budget. A safety cap of 5 figure-only iterations prevents infinite loops — after exhaustion, agent refuses new figure generation and redirects to remaining analysis steps
- **Figure Revision Fix**: Increased `max_tokens` for figure revision code generation from 2048 to 32000, preventing code truncation (SyntaxError) when revising complex publication figures
- **Data Download**: All uploaded data (pathology images, sequencing files, loom files, biochemistry files) can now be downloaded from the Data panel via individual download buttons
- **Agent Status Display Fix**: Fixed bug where completed analysis agents showed spinning icons during report generation. `getAgentStatus()` now correctly handles `generating_report` session status (analysis agents stay "done") and recognizes additional completion message types (`completed`, `integrated_report`, `graphic_abstract`, etc.)
- **Figure Gallery Fix**: Fixed figure previews not displaying after page reload (localStorage persistence stripped base64 data, which then overwrote fresh API data). Fixed MIME type mismatch (JPEG data displayed as PNG). Session-completed figures with `revision_requested` status now display as green "Final" instead of red "Revision Requested"
- Bug fixes: `TaskStatus` type now includes `generating_report`, backend `/load` and upload endpoints return `filePath` for pathology and biochemistry files

#### v0.7.3 (2026-02-13)

- **Chat Delegation**: PI Agent can now delegate complex chat requests to specialized agents (Bioinformatics, Pathology, Biochemistry, Literature). Agents execute analysis in background, PI synthesizes results — with WebSocket progress updates and stop/cancel support
- **Report Generation Rewrite**: Detailed report no longer uses Jupyter sandbox for code re-execution. Instead builds structured analysis record from existing code history, scans workspace for figures/tables, and PI writes narrative around existing assets — faster and more reliable
- **Report-Only Resume**: Sessions interrupted during report generation (after all analysis rounds completed) are automatically detected and resume with report generation only, skipping redundant analysis
- **Auto PDF Generation**: Monkey-patches `Figure.savefig` to automatically generate companion high-resolution PDFs (300 DPI, vectorized Type 42 fonts) alongside every PNG saved during analysis
- **Deep Research Optimization**: Gemini Deep Research now merges multiple topics into a single comprehensive query. After DR completes, evaluates topic coverage and falls back to PubMed+LLM for insufficiently covered gaps
- **Bioinformatics Session Restore**: When resuming analysis, code history is populated from restored notebook cells — prevents agent from re-running completed analysis steps. New "REUSE EXISTING RESULTS" guidelines check for existing PCA, UMAP, clustering, annotations before re-executing
- **Batch Effect Assessment**: Bioinformatics agent now checks for batch effects after PCA/UMAP when multiple samples are present, and consults PI about correction strategy (Harmony, BBKNN, scVI, etc.)
- **Configurable Round Wait Timer**: Round feedback countdown is now user-configurable (default 5 minutes; -1 for manual-only mode; 0 for no-wait auto-continue). Stored per-session in `round_wait_seconds`
- **Pathology Agent Reliability**: Reduced tile batch size (20 → 5) and added dynamic timeouts (120s base + 30s per image) for more reliable vision API calls
- **Export/Import with Sandbox State**: Session export/import now includes .pkl files for full sandbox environment restore
- **Publication Outputs in Results Tab**: Publication figures, integrated report, graphic abstract, and download buttons moved from Analysis tab to Results tab for cleaner organization
- **Model Defaults Updated**: Bioinformatics agent now defaults to Claude Sonnet 4.5; Literature agent gets dedicated `literature_model` field defaulting to Gemini 3 Flash Preview
- Bug fixes: chat stop/cancel, progress message rendering, snake_case→camelCase transform for publication outputs, sandbox kernel cleanup

#### v0.7.2 (2026-02-11)

- **Chat Panel**: New Chat tab for free-form conversation with PI Agent — ask questions about analysis results, request scientific insights, and discuss next steps
  - File attachment support with per-file agent targeting (PI, Pathology, Bioinformatics, Biochemistry)
  - Inline figure display: PI can reference existing figures (`[[FIGURE:name]]`) or generate new ones on-the-fly (`[[GENERATE_FIGURE:description]]`) via Jupyter sandbox
  - Markdown-rendered responses with PNG/PDF download buttons for figures
  - Resizable input textarea with drag handle
- **Session Export/Import**: Download complete sessions as tar.gz archives and restore them on any instance
  - Export includes: session state, uploads, workspace files, reports (excludes large .pkl files)
  - Import supports tar.gz, zip, and plain JSON formats with automatic path remapping
  - Session browser modal with status badges and data type indicators
- Bug fixes: chat file deduplication, sandbox message filtering, import path validation

#### v0.6.3 (2026-02-11)

- **Literature-based Cluster Annotation**: New `annotate_clusters` tool for bioinformatics agent — single literature search + LLM matching annotates all clusters at once, replacing per-cluster PI consultation
- **Round Context Enrichment**: Each agent now receives previous round findings and code history, preventing redundant analyses in multi-round workflows
- **Literature Config per Session**: Literature mode (Gemini/PubMed), Google API key, PubMed email, and search rounds are now stored per-session in `literature_config` (moved from global `APIConfig`)
- **ConfigPanel Redesign**: PubMed email (required) and Google API key (optional, auto-enables Gemini mode) moved to main config panel
- **LLM Client Improvements**: Robust content extraction from various response formats (string, list, None, reasoning models); reasoning model support (`o1`/`o3`/`o4` series use `max_completion_tokens` instead of `max_tokens`)
- **RoundFeedbackDialog Timer Fix**: Countdown timer now uses wall-clock time so background browser tabs don't freeze the countdown
- Bug fixes: literature agent cleanup, PromptsPanel literature config save, base64 stripping improvements

#### v0.6.2 (2026-02-10)

- **Literature Agent Rewrite**: Rebuilt with 2 research modes (removed OpenRouter deep research)
  - Gemini Deep Research (Google Interactions API) - fixed REST payload, status polling, and response extraction
  - PubMed + LLM (NCBI E-utilities + LLM synthesis)
  - Multi-round iterative search with LLM-guided query refinement
  - Automatic fallback: Gemini → PubMed → LLM knowledge only
- **Literature-based Cluster Annotation**: `annotate_clusters()` method for two-phase cluster annotation (literature search + LLM matching)
- **Bioinformatics Agent `annotate_clusters` Tool**: New function-calling tool to annotate all clusters via literature evidence in a single call
- **Report Cluster Annotations**: Reports now include a cell cluster annotation table (Cluster → Cell Type → Cell Count) extracted from processed adata
- **Session Recovery on Page Refresh**: Auto-reload session from server after page refresh; stale session detection clears sessions that no longer exist on server
- **localStorage Overflow Prevention**: Base64 image data stripped from WebSocket messages, API responses, and Zustand persisted state
- **Read Mode**: New read-only mode for reviewing existing analyses without starting new ones
- **File Upload During Pause**: Users can upload pathology images while workflow is paused; uploads on the last round extend max_rounds by 1

#### v0.6.1 (2026-02-10)

- Bug fixes: null safety in partialize, WebSocket "session not found" detection, report findings dict key safety

#### v0.6.0 (2026-02-09)

- **Human-in-the-Loop Round Feedback**: New `RoundFeedbackDialog` modal appears after each analysis round with 2-minute countdown timer
  - Displays per-agent findings (Pathology, Bioinformatics, Biochemistry, Literature)
  - Users can provide text guidance and upload additional data files targeted to specific agents
  - Timer auto-pauses when user starts typing; auto-continues on expiry
  - User input on the last round extends max_rounds by 1
- **Workflow `wait_for_user_input` node**: Backend blocks between rounds via `asyncio.Event`, respecting `round_wait_seconds` setting (-1=wait indefinitely, 0=no wait, >0=countdown+60s buffer)
- **Cell type annotation improvements**: Strengthened bioinformatics agent prompt for mandatory cell type annotation with UMAP visualization
- **Report generator fixes**: Now searches workspace directory for annotated data; supports both `adata_annotated.h5ad` and `annotated_adata.h5ad` filenames
- **WebSocket reconnection fix**: Completed sessions now send full round_results and message_history on reconnect
- **Report loading retry**: ReportPanel retries loading after 3 seconds if report is null

#### v0.5.x (2026-02-06 – 2026-02-09)

- **Biochemistry Agent v2**: Rebuilt with 3-step analysis workflow (structured indicators → visual image analysis → synthesis & conclusions)
- **Real-time Message Optimization**: Improved WebSocket message streaming and deduplication
- Enhanced frontend-backend state synchronization
- Bug fixes for bioinformatics agent, session handling, sandbox timeout, UI components

#### v0.3.0 – v0.4.x (2026-02-04 – 2026-02-05)

- **Painter Agent**: Generates publication-quality graphic abstracts using Google Gemini image generation
- **State Version Control**: Atomic state versioning prevents frontend-backend synchronization issues
- **WebSocket Reconnection Recovery**: Automatic message history restoration after connection loss
- **Structured Error Tracking**: Per-agent error tracking with detailed information display
- **Parallel Analysis**: Pathology, Bioinformatics, and Biochemistry agents now run in parallel
- **Figure Review Loop**: PI Agent reviews and approves publication-quality figures before inclusion
- **Report Generation Rebuilt**: Complete overhaul of the detailed report generation system

## Contact

For issues, please contact the project maintainers.

---

*Document updated: 2026-03-31*
*Version: 0.18.8*
