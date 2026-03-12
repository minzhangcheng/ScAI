# InteSc Project Handover Document

This document is intended for developers and operations personnel, providing technical details, architecture design, development guides, and operations instructions.

**Version**: 0.13.2
**Last Updated**: 2026-03-12

## Project Overview

**InteSc** (Integrated and Intelligent Single Cell Sequencing Analysis) is a multi-agent collaboration system for biomedical data analysis. Through the coordination of seven specialized agents (PI, Bioinformatics, Pathology, Biochemistry, Literature, Painter, and Supervisor), it automates comprehensive analysis of single-cell sequencing data with parallel execution, state version control, structured error tracking, and optimized real-time communication.

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
intesc/                         # Code directory
├── backend/
│   ├── main.py                 # FastAPI entry point (~4423 lines)
│   │                           # - REST API endpoints
│   │                           # - WebSocket real-time communication
│   │                           # - Session management & persistence
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
│   │
│   ├── workflow.py             # LangGraph workflow (~2265 lines)
│   │                           # - Multi-Agent coordination
│   │                           # - Parallel execution (Fan-out/Fan-in)
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
│   │
│   ├── report_generator.py     # Report generator (~5100 lines)
│   │                           # - No-sandbox report compilation (v0.7.3 rewrite)
│   │                           # - PI analysis review pre-step (reads notebook, views figures via vision API)
│   │                           # - Builds analysis record from code_history
│   │                           # - Scans workspace for existing figures/tables
│   │                           # - PI plans narrative around existing assets
│   │                           # - Cluster annotation table extraction
│   │                           # - Report appendix (project info, agent config, data summary, round history)
│   │                           # - Full analysis log timeline (chronological steps with code/figures/reasoning)
│   │                           # - Figure/table scientific interpretation rendering
│   │
│   ├── regenerate_report_html.py # Report HTML regeneration utility (~1479 lines, appendix + analysis log support)
│   │
│   ├── config.py               # Configuration management (~300 lines)
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
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent (autonomous, figure step budget, pip install, cumulative revision history, analysis log)
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── biochemistry_agent.py    # Biochemistry Agent (3-step v2, ~785 lines)
│   │   ├── literature_agent.py # Literature Agent (2 research modes + cluster annotation, ~1060 lines)
│   │   ├── painter_agent.py    # Painter Agent (graphic abstract + revision support)
│   │   └── supervisor_agent.py # Supervisor Agent (real-time bioinformatics step monitoring, PI escalation, custom system_prompt)
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic data models (~540 lines)
│   │                           # - AnalysisState: Core state with version control + treatment_groups + data_files + biochemistry_indicators + analysis_log
│   │                           # - AnalysisStep: Comprehensive step log (code/PI/literature/annotation/findings/figure/text)
│   │                           # - TreatmentGroup: Experimental group definition (name, sample_count)
│   │                           # - DataFile: Universal file record (section: pathology/sequencing/biochemistry, group_id, sample_number)
│   │                           # - BiochemistryIndicator: Manual biochemistry entry with group/sample
│   │                           # - TaskStatus: GENERATING_REPORT status
│   │                           # - AgentError: Structured error tracking
│   │                           # - RoundResult: agents_for_next_round for selective activation
│   │                           # - PublicationFigure/Table: Publication outputs (+ interpretation field)
│   │                           # - GraphicAbstract: Painter agent output
│   │                           # - BioinformaticsCheckpoint: Step-level recovery (+ analysis_log)
│   │                           # - Data helper methods: get_files_by_section/group, build_experimental_design_summary
│   │
│   ├── utils/
│   │   ├── llm_client.py       # LLM client (~872 lines, OpenRouter + Google DR, precise retry with InternalServerError)
│   │   ├── sandbox.py          # Jupyter execution environment (~946 lines, auto PDF, code safety validation, extract_code_from_markdown)
│   │   ├── checkpoint.py       # Generic checkpoint manager (analysis + report resumption)
│   │   ├── image_utils.py      # Image processing
│   │   ├── upload.py           # Chunked upload manager (~398 lines)
│   │   └── llm_logger.py       # LLM logging
│   │
│   └── static/                 # Static files (backup)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main application component (+ Chat tab + About dialog)
│   │   ├── components/         # React components
│   │   │   ├── ChatPanel.tsx   # PI Agent free-form chat + delegation (~657 lines)
│   │   │   ├── AboutDialog.tsx # About dialog (project info, author, contact)
│   │   │   ├── AnnotationConfirmationDialog.tsx  # Cell annotation review dialog (Phase 1 pause, ~314 lines)
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

intesc_data/                    # Data directory (sibling to code directory)
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

intesc_dep/                     # Dependencies directory (sibling to code directory)
├── venv/                       # Python virtual environment
│   ├── bin/                    # Python executables
│   ├── lib/                    # Python packages
│   └── ...
└── node_modules/               # Node.js dependencies
    └── (symlinked to frontend/node_modules)
```

## Core Module Details

### 1. Workflow Engine (workflow.py)

Multi-Agent coordination system based on LangGraph with parallel execution and error recovery:

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
initialize_round ◄────────────────────────┐
      │                                   │
      ▼                                   │
  pi_planning                             │
      │                                   │
      ▼ (sequential)                      │
  biochemistry_analysis                   │
      │                                   │
      ▼  (PI refines tasks)               │
  pathology_analysis                      │
      │                                   │
      ▼  (PI refines tasks)               │
  bioinformatics_analysis                 │
      │                                   │
      ▼  (PI refines tasks)               │
  literature_research                     │
      │                                   │
      ▼                                   │
  pi_synthesis                            │
      │                                   │
      ▼                                   │
  wait_for_user_input  ◄── configurable   │
      │                    (default 5min) │
  ┌───┴───┐                               │
  │       │                               │
(continue) (stop)                         │
  │       │                               │
  │       ▼                               │
  │  generate_final_report                │
  │       │                               │
  │       ▼                               │
  │  generate_graphic_abstract            │
  │       │                               │
  │       ▼                               │
  │  generate_detailed_report             │
  │       │                               │
  │       ▼                               │
  │      END                              │
  │                                       │
  └───► check_convergence ────────────────┘
```

**Key Features:**

1. **Sequential Execution**: Biochemistry → Pathology → Bioinformatics Agents run sequentially, with Bioinformatics consulting PI and Literature via callbacks. Biochemistry runs first so its results inform downstream agents.
2. **Retry Mechanism**: `@with_retry(max_retries=3)` decorator with exponential backoff
3. **Structured Error Tracking**: Errors captured in `AgentError` model with agent, type, message, details
4. **State Version Control**: `_send_state_sync()` method ensures atomic version updates
5. **Figure Review Loop**: Bioinformatics Agent's figures are reviewed by PI before approval (up to 3 revisions). PI review includes experimental design context with a Data Fidelity criterion to catch fabricated groups. Revision uses `chat_with_vision()` so the bioinformatics agent can see its own figure alongside PI feedback. Full cumulative revision history is tracked in `PublicationFigure.revision_history` and passed to the code reviser to prevent re-introducing fixed issues.
18. **AI-Curated Brief Figures**: After all analysis rounds complete, PI Agent's `select_brief_figures()` curates the figure set shown in the Results tab — filters out QC/diagnostic plots (e.g. qc_violin, pca_variance, hvg_plot) and superseded intermediate versions, keeping only key result figures. Falls back to all publication figures if AI selection returns empty. Workspace figures (workspace root + figures/) are scanned and compressed (`compress_image_to_base64`) for inclusion alongside existing publication figures.
6. **Figure Step Budget**: Figure-only iterations (using `generate_publication_figure` tool) are excluded from the bioinformatics agent's step count. Safety cap of `MAX_FIGURE_ITERATIONS=5` prevents infinite figure generation loops — after exhaustion, agent refuses new figures and redirects to analysis. `max_tokens=32000` for figure revision code generation prevents truncation.
19. **Tool Message History Consistency**: Bioinformatics Agent appends exactly one assistant message per LLM response (using `raw_assistant_message` if available) before iterating over tool calls. Eliminates duplicate or missing assistant messages that previously caused API errors with Gemini's `thought_signature` requirement.
17. **Figure Data Fidelity**: All figure generation and revision prompts include explicit experimental design constraints (`CRITICAL DATA CONSTRAINT`) with actual group names from `state.treatment_groups`. This prevents LLM hallucination of non-existent treatment groups. Constraint is injected in: `_build_data_context()`, `generate_publication_figure` tool description, `_revise_figure_code()` prompt, and PI `review_figure()` prompt.
7. **Convergence Detection**: Stops after 2 consecutive rounds with no new findings
8. **Human-in-the-Loop**: `wait_for_user_input` node blocks between rounds via `asyncio.Event`. Frontend shows `RoundFeedbackDialog` with configurable countdown (default 300s; -1=manual-only, 0=no-wait auto-continue, stored in `round_wait_seconds`). Users can provide text guidance, upload files (targeted to specific agents), or skip. Backend respects `round_wait_seconds`: manual mode waits indefinitely, no-wait proceeds immediately, countdown mode uses frontend timer + 60s buffer as backend safety timeout. User input or file uploads on the last round extends `max_rounds` by 1.
9. **File Upload During Pause**: Users can upload pathology images while workflow is in `waiting_input` or `completed` state. Files are tracked in `pending_uploaded_files` and processed in the next round.
10. **Round Context Enrichment**: `_build_previous_findings_context()` and `_build_code_history_summary()` methods provide each agent with previous round findings and code execution history, preventing redundant analyses in multi-round workflows.
11. **Literature-based Cluster Annotation**: `_annotate_clusters_callback()` routes bioinformatics agent's `annotate_clusters` tool calls to the literature agent for efficient batch cluster annotation.
12. **Deep Research Topic Merging**: When using Gemini Deep Research mode, multiple topics are merged into a single comprehensive query (one DR call instead of N). After the DR call, `_evaluate_dr_coverage()` checks which topics were insufficiently covered and falls back to PubMed+LLM for gaps.
13. **Report-Only Resume**: `generate_report()` method runs only the report generation phase (final report → graphic abstract → detailed report), bypassing analysis rounds. Used when a session completed all analysis but was interrupted during report generation. WebSocket handler detects this condition automatically.
14. **GENERATING_REPORT Status**: New `TaskStatus.GENERATING_REPORT` marks sessions in report generation phase, enabling detection of interrupted report generation on resume.
15. **Selective Agent Activation**: From round 2 onwards, PI Agent decides which agents to activate via `RoundResult.agents_for_next_round`. Bioinformatics Agent always runs. `_pi_planning()` builds `skip_nodes` from the previous round's `agents_for_next_round`, and `@with_retry` skips deactivated nodes with WebSocket notification. Each round's `skip_nodes` is rebuilt fresh (not accumulated), so previously skipped agents can be re-activated. **Auto-activation**: If PI's `next_steps` contains tasks prefixed with an agent name (e.g., `[Pathology] ...`) but that agent is not in `agents_for_next_round`, the workflow auto-activates it.
21. **Analysis Log Propagation**: `_post_bioinformatics()` collects `analysis_log` from the bioinformatics agent result, deduplicates by step ID, and appends to `AnalysisState.analysis_log`. Code history is similarly deduped by `CodeExecution.id`.
22. **Cross-Agent Context Sharing**: Biochemistry findings are injected into pathology and bioinformatics agent prompts. Pathology findings are injected into bioinformatics agent prompts. `_collect_current_round_results()` collects findings from already-completed agents for dynamic PI planning. Each agent benefits from the accumulated results of all preceding agents in the same round.
23. **Dynamic PI Planning**: PI Agent's `plan_agent_tasks()` method refines each downstream agent's tasks based on already-completed results within the round. Uses `self.chat()` to accumulate context across successive planning calls, building a coherent chain of thought. PI history is reset at the start of each round's planning phase. For round 2+, previous round PI summary is passed as strategic overview context.
16. **Docker Pip Install**: Bioinformatics Agent's `_validate_code()` allows `subprocess` calls when all usages are pip/pip3 install commands (safe in Docker environments). System prompt includes package installation guidelines.
20. **Graphic Abstract Revision Loop**: The graphic abstract generation now uses a 3-phase revision process: (1) initial generation by Painter Agent; (2) up to 2 rounds of PI review via vision API (`review_graphic_abstract()`) + Painter revision with accumulated feedback history (`revise_graphic_abstract()`); (3) PI selects the best candidate from all versions using side-by-side vision comparison (`select_best_graphic_abstract()`). Metadata stored on `GraphicAbstract`: `revision_count`, `revision_history`, `all_candidates`, `selected_index`.

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
| `annotation_confirmation` | Phase 1 pause — sends cluster annotations for user review and confirmation |
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

### 1.2 Literature Agent (2 Research Modes + Cluster Annotation, ~1060 lines)

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

### 1.5 Session Export/Import (main.py)

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

### 4. Report Generator (report_generator.py, ~4683 lines)

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

### 5. LLM Client (llm_client.py, ~872 lines)

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
1. Create Python virtual environment at `../intesc_dep/venv/`
2. Install frontend dependencies at `../intesc_dep/node_modules/`
3. Create data directories at `../intesc_data/`
4. Start frontend and backend services

**Manual setup (if needed):**

```bash
# Backend (dependencies installed to ../intesc_dep/venv)
python -m venv ../intesc_dep/venv
source ../intesc_dep/venv/bin/activate
pip install -r requirements.txt

# Frontend (dependencies installed to ../intesc_dep/node_modules)
mkdir -p ../intesc_dep/node_modules
cd frontend
ln -s ../../intesc_dep/node_modules node_modules
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
# Logs located in intesc_data/logs/ directory

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

### Data Backup

**Directories to back up** (located in parent of code directory):

```
intesc_data/                # Data directory (must backup)
├── sessions/               # All session data (session-centric layout since v0.12.4)
│   └── {session_id}/       # Each session's workspace, reports, uploads, states, logs
├── logs/                   # Global system logs
└── upload_temp/            # Temporary upload chunks (can skip)

intesc_dep/                 # Dependencies directory (optional backup, can reinstall)
├── venv/                   # Python virtual environment
└── node_modules/           # Node.js dependencies
```

**Backup Recommendations:**

- `intesc_data/` directory contains user data and analysis results, must backup
- `intesc_dep/` directory can be skipped, reinstall via `./start.sh`
- Code directory `intesc/` managed via git

### Log Locations

```
intesc_data/logs/   # LLM interaction logs
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
- Check `intesc_data/logs/error.log` for details

**8. Stale Session After Server Restart**

- If browser shows blank/error state after server restart, the session may no longer exist on server
- v0.6.1+ auto-detects this and clears stale session from localStorage
- Manual fix: Clear localStorage in browser DevTools → Application → Local Storage → delete `intesc-storage`

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
| FastAPI server, session mgmt, chat, delegation, export/import & session editing | `main.py` | ~4423 lines |
| Report generator (no-sandbox, PI analysis review, appendix) | `report_generator.py` | ~4683 lines |
| Workflow graph, orchestration, DR optimization, selective activation & GA revision loop | `workflow.py` | ~2133 lines |
| Bioinformatics Agent (autonomous + figure budget + pip install + vision revision + cumulative history) | `bioinformatics_agent.py` | ~1628 lines |
| PI Agent (coordinator + reviewer + GA reviewer + agent activation + data fidelity + brief figure curation) | `pi_agent.py` | ~1642 lines |
| Literature Agent (2 modes + annotation) | `literature_agent.py` | ~1060 lines |
| LLM client (OpenRouter + Google DR + MIME detection + precise retry) | `llm_client.py` | ~914 lines |
| Jupyter sandbox (+ auto PDF) | `sandbox.py` | ~793 lines |
| Biochemistry Agent (3-step v2) | `biochemistry_agent.py` | ~785 lines |
| Pathology Agent (batch size 5 + dynamic timeout) | `pathology_agent.py` | ~453 lines |
| Chunked upload manager | `upload.py` | ~398 lines |
| Data models & schemas | `schemas.py` | ~486 lines |
| Painter Agent (graphic abstract + revision) | `painter_agent.py` | ~440 lines |
| Supervisor Agent (real-time step monitoring + PI escalation) | `supervisor_agent.py` | ~442 lines |
| Configuration & prompts (+ supervisor/painter prompt fields) | `config.py` | ~300 lines |
| Agent base class | `base.py` | ~246 lines |

### Frontend

| Feature | File | Approximate Size |
|---------|------|-----------------|
| API hooks + WebSocket + session editing | `useApi.ts` | ~1184 lines |
| Data upload (queue + per-file progress) + download interface | `DataPanel.tsx` | ~1070 lines |
| Real-time message streaming + delegation | `MessageStream.tsx` | ~846 lines |
| Report generation & download | `ReportPanel.tsx` | ~789 lines |
| Analysis control + agent status | `AnalysisPanel.tsx` | ~751 lines |
| Session mgmt + export/import + session editing | `Sidebar.tsx` | ~778 lines |
| PI Agent chat + delegation | `ChatPanel.tsx` | ~657 lines |
| Agent prompt + literature config | `PromptsPanel.tsx` | ~521 lines |
| Round feedback dialog (configurable timer) | `RoundFeedbackDialog.tsx` | ~600 lines |
| Zustand global state (+ delegation) | `appStore.ts` | ~509 lines |
| Round results + publication outputs | `ResultsPanel.tsx` | ~359 lines |
| Main application (+ About dialog) | `App.tsx` | ~313 lines |
| About dialog (project info) | `AboutDialog.tsx` | ~135 lines |
| Cell annotation confirmation dialog (Phase 1) | `AnnotationConfirmationDialog.tsx` | ~314 lines |
| Exploration report & direction selection dialog | `ExplorationReportDialog.tsx` | ~264 lines |
| API + literature configuration + DR warning | `ConfigPanel.tsx` | ~261 lines |

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

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.13.1 | 2026-03-12 | Multi-direction research workflow (directions_queue, auto-continue between directions), enhanced direction candidate evaluation (innovation/novelty/reliability scores), exploration report HTML export, PI select_best_directions(), bioinformatics LLM error status, session eviction race fix, sidebar max-rounds removed |
| 0.13.0 | 2026-03-12 | Two-phase workflow (Exploration Phase + Innovation Phase), cell annotation confirmation dialog (WAITING_ANNOTATION status), exploration report with research direction selection, new data models (ClusterAnnotation, AnnotationConfirmationState, ExplorationReport) |
| 0.12.4 | 2026-03-11 | Session-centric data directory restructure, backward-compatible path resolution, data migration script |
| 0.12.3 | 2026-03-10 | Report markdown rendering fix (headers/lists in HTML), conclusions formatting cleanup, frontend ReactMarkdown rendering, frontend appendix display, supervisor agent HTML appendix |
| 0.12.2 | 2026-03-10 | Analysis stop/pause support, publication figure checklist, figure revision auto-retry, progressive figure review leniency, WebSocket reconnection improvements |
| 0.10.3 | 2026-03-09 | Comprehensive analysis log (AnalysisStep model, all tool call types), analysis log timeline in reports, figure/table scientific interpretation, PI history reset on synthesis, code history cap (100), agent auto-activation from PI tasks, pathology→bioinformatics context sharing, code history dedup by ID, deep research availability refactor |
| 0.10.2 | 2026-03-08 | Sandbox kernel timeout recovery (full reconnection + session restore), plt.close/plt.show safety (try/finally in 3 code templates), blank figure detection (<2KB threshold), sandbox timeout 300s→1800s, architecture docs updated (sequential execution) |
| 0.10.1 | 2026-03-07 | Literature fallback model (quick search uses PI model), deep research max_tokens fix (None instead of 262144), deep research model passthrough in chat mode, upload cancellation (abort in-progress + remove pending) |
| 0.10.0 | 2026-03-07 | Deep Research mode (OpenRouter: o4-mini-deep-research, sonar-deep-research), 3 literature modes (deep_research/gemini/pubmed_llm), fallback chain, LLM-based query generation, model upgrades (claude-sonnet-4.6), cross-agent context, bug fixes |
| 0.9.1 | 2026-03-07 | Cross-agent context passing (biochemistry + literature receive current round findings), PubMed query degradation (`_validate_and_degrade_query`), LLM-driven research context (`_build_research_context` replaces hardcoded `_build_research_query`), skip_nodes reset bug fix, skipped node result clarity, race condition fix (`_continue_event.clear()` moved to `_pi_synthesis`), PI synthesis error awareness, literature findings in round summary, fallback task assignment, human guidance source fix, max_tokens unified to 32768, bioinformatics text_response_count reset + `completed_with_fallback` status, JSON parse failure logging |
| 0.9.0 | 2026-03-04 | PI analysis review before report (reads notebook + views figures via vision API), report appendix (project info, agent config, data summary, round history, user inputs), graphic abstract revision loop (PI review + Painter revision + best selection), cumulative figure revision history, in-place session editing (name + instruction via sidebar), two-phase export (prepare + download with size preview), upload queue with per-file progress, "Other" sequencing type (bulk RNA-seq support), sample count safety guard, workspace figure downloads (PDF/PNG), figure gallery API fallback on reload, Deep Research cost warning |
| 0.8.9 | 2026-03-03 | Import proxy timeout resilience (pre-generated `import_id`, early job registration, upload-finished detection, polling retry with 404 tolerance), import UX (success/failure icons, confirmation dialog), branding update ("Integrated & Intelligent Single Cell Analysis") |
| 0.8.7 | 2026-03-02 | Data page scroll fix (`overflow: hidden` on html/body), footer i18n (Chinese/English), About page version/date update |
| 0.8.6 | 2026-03-02 | Async session import (background tar.gz extraction + polling endpoint, avoids proxy timeout), first-visit disclaimer dialog (AI reliability warning, localStorage), About button text label, footer copyright |
| 0.8.5 | 2026-02-28 | Tool message history fix (single raw_assistant_message per response, Gemini thought_signature), AI-curated brief figures (`select_brief_figures`), workspace figure collection (root + figures/ dirs), image MIME type detection, export GET endpoint + browser-native download UX |
| 0.8.4 | 2026-02-21 | About dialog (project info, author, affiliation, contact, GitHub link) |
| 0.8.3 | 2026-02-21 | Manual round wait fix (`_wait_for_user_input` now respects `round_wait_seconds`: -1=indefinite, 0=immediate, >0=countdown+60s buffer; previously hardcoded 600s timeout) |
| 0.8.2 | 2026-02-21 | Default model update (PI/Pathology/Biochemistry → `gemini-3.1-pro-preview`), proxy fix (4 `OpenAICompatibleClient` instances missing explicit `proxy_url`) |
| 0.8.1 | 2026-02-20 | Figure data fidelity (experimental design constraints in figure review/revision pipeline), vision-based figure revision (`chat_with_vision` in `_revise_figure_code`), PI Data Fidelity review criterion |
| 0.8.0 | 2026-02-16 | Unified data model (`DataFile` + `TreatmentGroup` + `BiochemistryIndicator` replace per-type models), multi-group multi-sample support, unified `/upload/data` endpoint, group CRUD, upload selectors in feedback dialog & chat, simplified pathology types (H&E + Other), DataPanel rewrite |
| 0.7.5 | 2026-02-15 | Selective agent activation (PI decides which agents to activate from round 2+, `agents_for_next_round`), Docker pip install support for bioinformatics agent |
| 0.7.4 | 2026-02-14 | Figure step budget (MAX_FIGURE_ITERATIONS=5), figure revision max_tokens fix, data download buttons, agent status display fix (generating_report), figure gallery preview + review status fix, MIME type correction |
| 0.7.3 | 2026-02-13 | Chat delegation, no-sandbox report generation, report-only resume, auto PDF, Deep Research optimization, bioinformatics session restore, batch effect check, configurable round timer, pathology reliability, .pkl in export/import |
| 0.7.2 | 2026-02-11 | Bug fixes for chat file handling, sandbox message filtering, import path validation |
| 0.7.1 | 2026-02-11 | ChatPanel UX improvements (per-file agent selector, resizable textarea), Sidebar export/import UX polish |
| 0.7.0 | 2026-02-11 | Chat tab (PI Agent free-form chat with figure display/generation), session export/import (tar.gz), session browser |
| 0.6.3 | 2026-02-11 | Literature-based cluster annotation tool, round context enrichment, per-session literature config, ConfigPanel redesign, LLM reasoning model support, timer wall-clock fix |
| 0.6.2 | 2026-02-10 | Literature Agent rewrite (2 modes), annotate_clusters method, cluster annotations in report, session recovery, localStorage overflow fix, read mode |
| 0.6.1 | 2026-02-10 | Bug fixes (partialize null safety, session not found, report dict keys) |
| 0.6.0 | 2026-02-09 | Human-in-the-loop round feedback dialog, workflow wait_for_user_input node, cell annotation + report fixes |
| 0.5.5 | 2026-02-09 | WebSocket reconnect with full session data, report loading retry, annotation workflow improvements |
| 0.5.4 | 2026-02-09 | Bug fixes (bioinformatics, session handling, sandbox, UI) |
| 0.5.3 | 2026-02-09 | Report generation and bioinformatics enhancements |
| 0.5.1 | 2026-02-08 | Biochemistry Agent v2 (3-step workflow), major refactoring |
| 0.5.0 | 2026-02-06 | Real-time message optimization |
| 0.4.1 | 2026-02-05 | Painter agent improvements, bug fixes |
| 0.4.0 | 2026-02-05 | Report generation rebuilt |
| 0.3.0 | 2026-02-04 | Parallel execution, Painter Agent, state versioning, error tracking |
| 0.2.x | 2026-01 | Report features development |
| 0.1.x | 2026-01 | Initial functional version |

### Detailed Changelog

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

- **Session-Centric Data Organization**: Restructured `intesc_data/` from type-based layout (`uploads/`, `states/`, `workspace/`, `reports/` as top-level dirs) to session-centric layout (`sessions/{session_id}/` containing all subdirs). Each session now has its own isolated directory with `workspace/`, `reports/`, `uploads/`, `states/`, and `logs/` subdirectories.
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

*Document updated: 2026-03-12*
*Version: 0.13.2*
