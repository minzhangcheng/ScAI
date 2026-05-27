# ScAI Handover

Architecture, module map, deployment, and developer guide for operators and contributors. For end-user workflow, see [user-guide.md](user-guide.md); for the project pitch and installation, see [README.md](README.md).

**Version**: 0.25.2 · **Last updated**: 2026-05-27

## 1. Runtime Architecture

ScAI is a single-container web application: a React/Vite single-page frontend talks to a FastAPI backend over REST and a WebSocket. The backend orchestrates eight specialized agents, each making LLM calls against any **OpenAI-compatible endpoint** — verified endpoints include cloud aggregators (OpenRouter, n1n.ai), direct provider APIs (DeepSeek, Moonshot, Zhipu, MiniMax, Aliyun/Qwen), Google Gemini (Deep Research literature mode), and locally deployed open-source models via **Ollama** — other OpenAI-compatible endpoints are likely to work but have not been officially tested. Endpoint and model are configurable **per agent**, so flagship and open-source models can be mixed freely. The Bioinformatics Agent additionally drives a **local Jupyter kernel** for code execution. Sessions are persisted to disk (per-session directory) and cached in memory; users and auth live in a small SQLite database.

```
┌──────────────────────────────────────────────────────────────┐
│           Frontend  (React · TypeScript · Vite · Zustand)    │
│  Tabbed shell — Data / Prompts / Analysis / Results /        │
│                 Report / Chat; WebSocket client              │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST (JWT) + WebSocket
┌───────────────────────────▼──────────────────────────────────┐
│           Backend  (FastAPI · Uvicorn · Pydantic)            │
│                                                              │
│  middleware/auth  →  routes/*  →  services/*  →  workflow.py │
│                                          │                   │
│                                          ▼                   │
│                               Agents (backend/agents/*)      │
│                                          │                   │
│                    ┌─────────────────────┼─────────────────┐ │
│                    ▼                     ▼                 ▼ │
│       LLM endpoints               Jupyter kernel      Disk   │
│  (any OpenAI-compatible;         (sandbox subprocess) store  │
│   cloud or local Ollama;                                     │
│   optional Google)                                           │
│                                                              │
│                             SQLite (users)                   │
└──────────────────────────────────────────────────────────────┘
```

Long-running analysis runs on a background asyncio task owned by the backend; the frontend observes progress over the WebSocket and interacts through HITL endpoints (annotation confirmation, direction selection, round feedback).

## 2. Repository Layout

```
scai/
├── backend/            FastAPI app, agents, workflow engine
├── frontend/           React + Vite UI
├── clients/python/     Python SDK (scai_client)
├── docs/               End-user documentation
├── Dockerfile          Multi-stage build image
├── docker-entrypoint.sh Container startup
├── start.sh / stop.sh  Native launch / shutdown
├── build.sh            Build helper
├── requirements.txt    Python dependencies
└── LICENSE             Apache 2.0
```

## 3. Backend Module Map (`backend/`)

| Path                                   | Purpose                                                                                                                                                                            |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main.py`                              | FastAPI app factory: CORS, auth middleware, lifespan hooks for background cleanup tasks, route registration. Version string lives here.                                            |
| `workflow.py`                          | Core orchestration engine — multi-agent round execution, HITL gates, state transitions, integrated-report generation.                                                              |
| `state.py`                             | Per-process mutable state: active sessions cache, per-session locks, access-time tracking, prepared exports, import jobs, chat delegation tracking, WebSocket monitor tasks.       |
| `agents/`                              | `base.py` + eight agents: `pi_agent`, `bioinformatics_agent`, `pathology_agent`, `biochemistry_agent`, `literature_agent`, `painter_agent`, `supervisor_agent`, `technique_agent`. |
| `routes/`                              | 17 route modules (see §4). Each module is mounted by `main.py`; all mutable endpoints require auth unless the path is in `AUTH_OPTIONAL_PATHS`.                                    |
| `services/`                            | Business logic between routes and agents. `config_io`, `session_store`, `workflow_runner`, `upload`, `user_cleanup`, and the `report/` sub-package (see §5).                       |
| `models/`                              | Pydantic schemas: `schemas` (analysis state), `api_requests` (request/response bodies), `config_schema` (agent + LLM config), `user` (auth entities).                              |
| `middleware/auth.py`                   | JWT validation; `AUTH_OPTIONAL_PATHS` tuple (`/api/session/`, `/api/file/`, `/api/download/`) delegates ownership checks to the route layer via `require_session_ownership`.       |
| `utils/`                               | `llm_client`, `agent_llm`, `sandbox` (Jupyter execution), `error_parsing` (incl. shared `_TECHNICAL_ERROR_TYPES`), `report_html` / `report_css`, `image_utils`, `checkpoint`, `files`, `auth`, `db`, `formatting`, `upload`, `llm_logger`. |
| `tasks/`                               | Background jobs started in lifespan: `session_cleanup` (idle + overflow eviction, stale import jobs), `ttl_cleanup` (upload TTL).                                                  |
| `config/`                              | Package: `_config.py` (paths, timeouts, LLM defaults, `PI_AGENT_NAME`), `__init__.py` (backward-compatible re-exports), `agent_prompts.py` (`AgentPrompts` Pydantic model).        |
| `tests/`                               | Pytest suite; shared `app_db` fixture in `conftest.py`; `route_snapshot.json` pins the route registry.                                                                             |
| `migrate_to_session_dirs.py`           | One-shot migration helper for older data layouts.                                                                                                                                  |
| `regenerate_report_html.py`            | CLI helper to rebuild a report's HTML from saved state.                                                                                                                            |

## 4. Routes (`backend/routes/`)

| Module             | Endpoint prefix / purpose                                                                 |
|--------------------|-------------------------------------------------------------------------------------------|
| `root.py`          | `/` landing, health.                                                                      |
| `setup.py`         | First-run admin bootstrap.                                                                |
| `auth.py`          | `/api/auth/*` — register, login, logout, `me`, profile (email/password change).           |
| `admin.py`         | `/api/admin/*` — user management, cascade cleanup.                                        |
| `config.py`        | `/api/config/*` — LLM API key / proxy / PubMed email, with validation.                    |
| `sessions.py`      | `/api/sessions/*` — session CRUD, listing, load.                                          |
| `session_config.py`| Per-session agent configs (models, temperatures, prompts, vision flags) and round-wait setting. |
| `indicators.py`    | Biochemistry indicator create/update/delete.                                              |
| `upload.py`        | Chunked upload endpoints with integrity verification.                                     |
| `files.py`         | File serving (uploaded data, generated figures).                                          |
| `lifecycle.py`     | Start / pause / resume / stop analysis.                                                   |
| `interaction.py`   | HITL inputs: human guidance, annotation confirmation, direction selection.                |
| `snapshots.py`     | Create / restore analysis checkpoints.                                                    |
| `export_import.py` | Session export (`.tar.gz` / `.zip`) and import with background job tracking.              |
| `chat.py`          | PI-Agent chat endpoints, including on-the-fly file uploads and agent delegation.          |
| `reports.py`       | Report generate / list / download / HTML / regenerate.                                    |
| `websockets.py`    | `/ws/{session_id}` real-time stream; `/ws/report/{session_id}` report-generation stream.  |
| `v1.py`            | Versioned API for the Python client.                                                      |

## 5. Services (`backend/services/`)

| Module                | Responsibility                                                                                                    |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| `config_io.py`        | Load/save session config (agent models, temps, prompts, LLM server overrides).                                    |
| `session_store.py`    | In-memory cache + atomic disk writes, save throttling, per-session locks.                                         |
| `workflow_runner.py`  | Launches and supervises headless workflow tasks (used by API-based runs).                                         |
| `upload.py`           | `ChunkedUploadManager` for multi-part uploads with resume and integrity checks.                                   |
| `user_cleanup.py`     | Admin-driven purge of a user's sessions and files.                                                                |
| `report/`             | Modular report pipeline:                                                                                          |
| ├ `generator.py`      | Orchestrates end-to-end report generation.                                                                        |
| ├ `regenerator.py`    | Rebuild report HTML from existing state; hosts `render_table_html()`, `get_agent_stats_html()`.                   |
| ├ `planning.py`       | Derive report structure from analysis state.                                                                      |
| ├ `sections.py`       | Section definitions (summary, methods, results, etc.).                                                            |
| ├ `figures.py`        | Figure selection, annotation, and review workflow.                                                                |
| ├ `analysis_log.py`   | Human-readable analysis transcript.                                                                               |
| ├ `llm_calls.py`      | LLM interactions used during report generation.                                                                   |
| ├ `pi_review.py`      | PI feedback pass before final rendering.                                                                          |
| ├ `scoring.py`        | Figure / table quality scoring.                                                                                   |
| └ `variants.py`       | Theme variants and visual options (uses shared `render_table_html`).                                              |

## 6. Frontend Module Map (`frontend/src/`)

| Path                 | Purpose                                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------------------|
| `main.tsx`           | Vite entry — mounts React 18 root.                                                                                |
| `App.tsx`            | Root layout: tabbed shell (Data, Prompts, Analysis, Results, Report, Chat), WebSocket setup, startup config gate. |
| `components/`        | Panel components (`DataPanel`, `PromptsPanel`, `AnalysisPanel`, `ResultsPanel`, `ReportPanel`, `ChatPanel`, `Sidebar`, `ConfigPanel`, `MessageStream`, `UserMenu`) and dialogs (`StopDialog`, `ValidationDialog`, `ExplorationReportDialog`, `RoundFeedbackDialog`, annotation dialogs, …). |
| `hooks/useApi.ts`    | REST client + WebSocket management + session lifecycle helpers (the bulk of frontend/backend plumbing).           |
| `hooks/useCountdownTimer.ts` | Wall-clock-deadline countdown with pause/resume; robust to backgrounded tabs.                               |
| `hooks/useClickOutside.ts`   | Reusable click-outside handler.                                                                            |
| `stores/appStore.ts` | Zustand store with `localStorage` persistence: auth, messages (pruned to ~7 days, images stripped), session data, per-agent configs, LLM overrides, vision/literature settings, UI state. |
| `utils/`             | `messageTransform` (shared `transformRawMessage` / `restoreAgentErrors`), `image`, `download`, `logger`, `authStorage`, `dataTypes`. |
| `constants/`         | `agents.ts` (icons/colors/gradients), auth constants, etc.                                                        |
| `i18n/`              | Localization resources (Chinese / English, auto-switched).                                                        |
| `pages/`             | Legacy route pages (Login, Register, Setup, Admin, ApiKeys) — most functionality has moved into tabbed panels.    |
| `types.ts`           | Shared TypeScript interfaces (`RoundSummary`, session/response shapes).                                           |
| `styles/`            | Global CSS.                                                                                                       |

## 7. Data & State

| Where                                   | What                                                                                                           |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `backend/state.py` (in-process)         | Active sessions cache, per-session locks, WebSocket monitor tasks, prepared exports, import jobs.              |
| Per-session disk directory              | `state.json`, workflow checkpoints, uploaded files, generated figures (PNG + PDF), notebooks, reports.         |
| SQLite (users DB)                       | Users, credentials (bcrypt), status (`pending` / `readonly` / `active`), admin flag, profile metadata.         |
| Frontend `localStorage`                 | JWT, API-key config (sent to backend), per-agent prompts/models/temps, recent session id, recent messages.     |

**Session lifecycle**: the in-memory cache is authoritative during activity; `session_store` writes through to disk with throttling. `tasks/session_cleanup` evicts idle sessions (and handles overflow) back to disk. Pause / resume / snapshot / export all go through the same store so they compose cleanly.

## 8. Request & Data Flow

1. **Config** — first-run dialog posts API keys to `/api/config`; backend validates and persists.
2. **Session create** — `POST /api/sessions` creates the session directory and DB row.
3. **Group + data upload** — biochemistry indicators via `/api/session/.../indicators`; files via chunked `/api/session/.../upload*`.
4. **Start analysis** — `POST /api/session/.../start` schedules the workflow task; WebSocket `/ws/{id}` streams agent messages, figures, status.
5. **HITL gates** — annotation confirmation, exploration-report review, round feedback — all posted to `routes/interaction.py`; the workflow task awaits these.
6. **Report** — `POST /api/session/.../report/generate` runs the report pipeline in `services/report/`; `/ws/report/{id}` streams progress; final HTML is saved and downloadable.
7. **Export / import** — tarball / zip created in the background, tracked as a job; import reverses the process.

## 9. Deployment & Operations

**Docker (recommended)**

- Image: `ghcr.io/minzhangcheng/scai:latest`; `Dockerfile` is multi-stage (Python + Node build).
- Volumes: `/dep` (Python venv + Node modules, installed on first boot), `/data` (session store, uploads, reports, logs).
- Entry: `docker-entrypoint.sh` — installs deps if `/dep` is empty, starts backend + serves built frontend.
- Port: `3000` (frontend; backend sits behind it).

**Native**

- `start.sh` (flags: `--host`, `--port`, `--backport`, `--dev`, `--prod`) creates a venv, installs `requirements.txt`, builds or serves the frontend, launches Uvicorn.
- `stop.sh` kills the PIDs and cleans temp dirs.
- Requires Python 3.11+ and Node LTS.

**Configuration**

- API keys and proxy live in the per-user config saved through `/api/config`.
- First-time installs require an admin password via the setup route; new user registrations default to `pending` until approved.
- Logs: stdout of the process; Docker installs write `scai.log` under the `/data` volume.

## 10. Development Guide

- **Dev loop**: `./start.sh --dev` — hot reload for both frontend and backend.
- **Tests**:
  - Backend: `pytest backend/tests` (uses shared `app_db` fixture in `conftest.py`; route registry pinned by `route_snapshot.json`).
  - Python client: `pytest clients/python/tests`.
- **Adding a route**: create `backend/routes/<name>.py`, register it in `main.py`, update `route_snapshot.json` (`test_route_registry.py` will tell you if it drifts).
- **Adding an agent**: subclass `backend/agents/base.py`, register defaults in `backend/config/agent_prompts.py`, wire it into `workflow.py`.
- **Adding a report section**: extend `backend/services/report/sections.py` and, if needed, `planning.py` / `generator.py`.
- **Conventions**: Pydantic models validate at route boundaries; services contain logic; agents are thin LLM orchestrators; `utils/` holds cross-cutting helpers; config is centralized in `backend/config/` with `PI_AGENT_NAME` as the single source of truth for the PI identifier.

## 11. Python Client SDK (`clients/python/`)

`scai_client` exposes a `Session` client for programmatic use: create a session, upload data, start analysis, poll status, export / import. `ImportJob` and `ExportJob` wrap the long-running background jobs with polling helpers. Chunked upload is handled automatically. Errors surface through the `errors` module as typed exceptions. See `clients/python/README.md` for usage.

## 12. Version History (condensed)

Detailed release notes for every version are available through `git log` and release tags. The summary below highlights the architectural arc of the project.

**0.24.x — collaborative analysis planning (2026-05-15)**

- `0.24.23` — report config table & websocket reconnect fix (2026-05-19):
  - **Agent config table with LLM stats**: Added `render_agent_config_table_html()` in regenerator.py for building agent configuration table with LLM usage statistics (calls, input/output tokens, avg response time). Updated `sections.py` to use this new function instead of manual HTML construction.
  - **WebSocket reconnect state sync**: Added `_resend_wait_input_state_sync()` in websockets.py to re-emit state_sync for pending HITL gates (llm_failure, agent_disabled) after a reconnect. Without this, LlmTimeoutDialog and disabled-agent dialog would not reopen on page reload.
  - **AGENT_DISPLAY_NAMES mapping**: Added display name mapping for all agents (pi, pathology, bioinformatics, biochemistry, literature, painter, supervisor, technique) in regenerator.py.
  - **Cleanup**: Removed unused `llm_server` field from agent stats and dead code in supervisor_agent.py.

- `0.24.22` — snapshot label consistency (2026-05-19):
  - **Snapshot "Experimental" label**: Added "(Experimental)" suffix to SnapshotPanel title and translation key `analysisSnapshots` in both English and Chinese for consistency with other experimental features.

- `0.24.21` — core agent disabled handling (2026-05-19):
  - **Core agents concept**: Added `CORE_AGENTS` constant in constants.py (pi, bioinformatics, literature) - these agents must be enabled for workflow to proceed.
  - **Agent disabled intervention**: Added `AGENT_DISABLED` reason and `TERMINATE` action for LLM failure handling. When a core agent is disabled, workflow blocks and shows intervention dialog instead of silently skipping.
  - **Settings provider fix**: Fixed agent disabled detection logic - now checks if key exists in custom_models/custom_prompts (including empty/whitespace values) instead of just checking truthiness.
  - **Workflow intervention method**: Added `_require_core_agent()` method in agents.py that blocks until core agent is configured, with retry/skip/terminate options.
  - **LlmTimeoutDialog integration**: Updated frontend LlmTimeoutDialog to handle "agent_disabled" reason - shows different title, body message, and removes model/server fields.
  - **Test additions**: Added `test_core_agent_disabled_intervention.py`, `test_disabled_agent_e2e.py`, `test_optional_agent_hot_reload.py`, `test_settings_provider_disable.py`.

- `0.24.20` — PlanReviewDialog countdown UI improvements (2026-05-19):
  - **Countdown color change**: Added dynamic color change for countdown timer - turns red when <= 30 seconds remaining to alert users of time running out.
  - **Layout refactor**: Restructured PlanReviewDialog header to organize countdown timer and download button in a horizontal flex layout alongside the title.

- `0.24.19` — per-round agent result cache (2026-05-18):
  - **AgentRoundCache model**: Added new Pydantic model in schemas.py for per-round per-agent intermediate result caching. Lives in AnalysisState so snapshot/restore captures it automatically.
  - **cache_hit/cache_store helpers**: Added functions in helpers.py for checking and storing agent results in the current round cache. Returns None if no cache exists or cache is for a different round.
  - **Pathology agent caching**: Added cache hit check at start of `_pathology_analysis` to skip re-execution if result already exists. Stores result on completion.
  - **Bioinformatics agent caching**: Added cache hit check at start of `_bioinformatics_analysis` to skip re-execution if result already exists. Preserves planning_stage transition on cache hit.
  - **Cache lifecycle management**: Clear `current_round_cache` on new round start in `_start_new_round` and after appending to `round_results` in `_synthesize_round` to prevent stale cache on snapshot restore.
  - **Snapshot labeling**: Updated translation to mark snapshots as "Experimental" in both Chinese and English.
  - **Test additions**: Added `test_agent_round_cache.py` and `test_snapshot_rollback_idempotency.py`.

- `0.24.18` — unified HITL wait gate (2026-05-18):
  - **wait_with_skip_time helper**: Added unified `wait_with_skip_time()` function in helpers.py that provides consistent HITL wait gate behavior across all gates (plan review, annotation, direction selection, round wait). Semantics: -1=manual (block forever), 0=no-wait (immediate), >0=countdown (wait_seconds + 60s buffer).
  - **Planning integration**: Updated `_user_review_gate` to use `wait_with_skip_time` instead of hardcoded timeout.
  - **Synthesis integration**: Updated `_wait_for_user_input` to use `wait_with_skip_time` for both direction selection gate and round wait. Auto-skip now sets user_chosen_direction to "[PI_DECIDE]".
  - **Annotation integration**: Updated `_annotation_confirmation` to use `wait_with_skip_time`. Auto-skip now accepts PI annotations as-is and marks them as confirmed.

- `0.24.17` — prompt template improvements (2026-05-18):
  - **Figure saving clarification**: Clarified that execute_code tool allows `plt.savefig('name.jpg')` and `sc.pl.*(save='_name.jpg')` for figure saving, while generate_prompts_figure tool wraps saving internally with different rules.
  - **Template placeholders in PI prompts**: Updated PI agent prompts to use template placeholders (e.g., `<pathway>`, `<cell type>`, `<condition>`) instead of hardcoded examples for research questions, literature topics, and graphic abstract feedback.
  - **Stage split note**: Added note in PI agent about Stage 1 vs Stage 2 task splitting to prevent task compression.
  - **MeSH tissue guidance**: Updated PubMed query guidance to include tissue MeSH terms (e.g., "Liver"[MeSH], "Kidney"[MeSH]).
  - **Organism-specific pathway libraries**: Updated agent_prompts.py to use organism-specific gene set libraries for pathway enrichment (Human: KEGG_2021_Human, Mouse: KEGG_2019_Mouse, Other: query gseapy).
  - **GO libraries cross-species**: Added guidance that GO libraries work across species when gene symbols match the organism convention.

- `0.24.16` — stage scope validation in planning (2026-05-18):
  - **Enhanced Stage 1 guidance**: Expanded `_STAGE1_GUIDANCE` in bioinformatics agent to explicitly exclude loading/preprocessing data for Stage 2 analyses, including: Loom files (RNA velocity), trajectory inputs, cell-cell communication preprocessing, and pre-staging gene sets for Stage 2 enrichment.
  - **PI plan review stage scope validation**: Added "Plan Stage" field and "Stage Scope" section to PI plan review prompt. Stage 1 reviews now reject steps that prep data for Stage 2; Stage 2 reviews allow Stage 2-specific steps.
  - **Stage scope as #1 review criterion**: PI plan review now evaluates stage scope first, with explicit criteria for detecting "scope leak" steps (methods referencing "for downstream", "for velocity", "for trajectory", etc.).
  - **Test additions**: Added `test_stage1_guid ance_scope.py` and `test_pi_review_stage_scope.py` for stage scope validation coverage.

- `0.24.15` — plan review action leak fix (2026-05-18):
  - **Planning action leak fix**: Clear `_plan_review_action` and `_plan_review_event` at the start of `_user_review_gate` to prevent stale actions from previous gate invocations (e.g., stage1's approve still sitting in the slot when stage2's gate is entered).
  - **Consume after action**: Clear the slot and event after consuming the action so the next loop iteration (clarify path) or the next gate invocation starts from a clean state.
  - **Test additions**: Added `test_user_review_gate_action_leak.py` (142 lines) to test the action leak fix.

- `0.24.14` — lifecycle & routing fixes (2026-05-18):
  - **Plan review resume requires explicit action**: Lifecycle resume endpoint no longer auto-skips plan review. Instead returns message directing user to `/plan-review` endpoint for approve/skip/modify.
  - **Reset pi_instructions on stage1→stage2 transition**: Added `_reset_pi_instructions_for_stage2()` to clear bioinformatics/pathology/biochemistry/literature tasks when transitioning from stage1 to stage2.
  - **SHA-1 for code_history dedup**: Replaced Python's built-in `hash()` with `hashlib.sha1()` for stable code content deduplication across sessions/restarts.
  - **plan_revision_resume one-shot**: Cleared `plan_revision_resume` flag immediately after routing to prevent stale True from mis-routing subsequent re-entries.
  - **Honor pending user work over convergence**: Added `_should_force_next_round()` in synthesis to override convergence when there's real human text input, uploaded files, or remaining directions queue.
  - **Clear user_plan_feedback after planning**: Planning now clears `user_plan_feedback` after consuming it to prevent stale feedback persisting.
  - **Close signal-loss race in manual mode**: Fixed `_continue_event` clearing/waiting order in `wait_for_user_input` to prevent signal loss.
  - **Test additions**: Added tests for code_history dedup hash, convergence override, resume plan review flow, stage transition instructions, user review gate abort signal, and wait for user input signal.

- `0.24.13` — empty workflow short-circuit & planning abort handling (2026-05-18):
  - **Empty workflow short-circuit**: Added early exit in `generate_final_report`, `generate_graphic_abstract`, and `generate_detailed_report` when `analysis_log` and `code_history` are empty (no analysis work recorded). Prevents generating empty reports when user stops before any analysis runs.
  - **Planning abort handling**: When bioinformatics LLM fails unrecoverably during planning and there's no existing plan to fall back to, planning now sets `should_continue=False` and `status=FAILED` instead of letting `failed_node` loop indefinitely. Added `_after_review` routing to handle "abort" case.
  - **Graph abort edge**: Added `abort → generate_final_report` edge in graph routing for the new abort flow.
  - **Test additions**: Added `test_reporting_shortcircuit.py` for empty workflow report short-circuit.

- `0.24.12` — report generation LLM failure handling (2026-05-18):
  - **Report generation LLM retry loop**: Wrapped `generate_integrated_report` in retry loop with `UpstreamExhaustedError` handling, similar to other LLM-calling methods. Uses `_wait_for_llm_intervention()` for consistent failure recovery.
  - **Test additions**: Added `test_generate_final_report_llm_failure.py`, `test_annotation_completed_restore.py`, `test_pi_planning_early_return.py`, `test_workflow_state_init.py`.
  - **State field cleanup**: Removed unused `skip_nodes` from WorkflowState TypedDict.

- `0.24.11` — HITL pause/resume with state restoration (2026-05-18):
  - **pre_pause_status field**: Added `pre_pause_status` field to save original status when pausing from WAITING_INPUT/WAITING_ANNOTATION/WAITING_PLAN_REVIEW states.
  - **Resume signal restoration**: Resume now properly signals continue/annotation/plan_review events based on restored status, allowing HITL-gated workflows to resume in place without frontend reconnection.
  - **Template noop install**: Extracted matplotlib stub unwinding logic into `_TPL_NOOP_INSTALL` template string for reuse across sandbox and chat code generation; ensures clean unwinding of stale stubs.
  - **Test additions**: Added `test_pause_resume_hitle.py` for HITL pause/resume coverage, `test_tpl_noop_install.py` for template installation, `test_initialize_plan_consistency.py`.

- `0.24.10` — cooperative pause gate & matplotlib stub fixes (2026-05-18):
  - **Cooperative pause gate**: Added `_pause_event` asyncio.Event to workflow for cooperative pausing. Cleared on `pause()` to block bioinformatics agent's autonomous loop between steps; set on `signal_resume()` to release. Wired into bioinformatics agent via `_pause_event` attribute.
  - **Lifecycle pause improvements**: Pause endpoint now accepts RUNNING and all WAITING_* states; added idempotent check for already-paused state (double-click safe); clear cooperative gate first before snapshot.
  - **Lifecycle resume improvements**: Resume now releases cooperative pause gate in-place when workflow is alive (no frontend reconnect needed); only cold reload when workflow was lost (server restart).
  - **Matplotlib stub unwinding**: Chat code generation now unwinds any leftover SCAI no-op stubs before re-installing to prevent wrapper stacking; marks stubs with `_scai_noop` and `_scai_prev` attributes for clean unwinding.
  - **Original plt.close capture**: Saved baseline `_original_plt_close` before any monkey-patching in sandbox for defensive restore.

- `0.24.9` — reasoning field expansion & API serialization refactor (2026-05-17):
  - **LLM reasoning field expansion**: Added "thinking" and "thought" keys to content extraction for DeepSeek/Claude compatibility; also handles list values in reasoning fields.
  - **to_api_dict refactor**: Replaced manual snake_case → camelCase mapping with `model_dump(mode="json")` + `_API_KEY_MAP` dictionary for maintainability; fields not in map are excluded from API response.
  - **Empty content diagnostic extraction**: Extracted `_log_empty_content_diagnostic()` helper from `llm_client.py` for reusability.
  - **Literature mode_override refactor**: Pass mode_override as parameter instead of mutating agent state directly.
  - **Config cleanup**: Removed dead config code from `main.py`; updated `_config.py`.
  - **New tests**: Added test files for content extraction reasoning, error parsing, literature mode_override, to_api_dict coverage, and base agent history.

- `0.24.8` — workflow modularization (2026-05-17):
  - **Workflow package refactoring**: Split the monolithic `backend/workflow.py` (~4000 lines) into a modular `backend/workflow/` package with 13+ specialized modules:
    - `core.py` — Main workflow class preserving backward-compatible API
    - `agents.py` — Bioinformatics, PI, supervisor agent orchestration
    - `planning.py` — Plan generation, revision, and PI review dialogue
    - `annotation.py` — Cluster annotation and confirmation flow
    - `callbacks.py` — PI callback, figure review, supervisor callbacks
    - `synthesis.py` — Round synthesis and supervisor integration
    - `reporting.py` — Exploration report and report generation
    - `routing.py` — Node routing logic (after_bioinformatics, should_continue, etc.)
    - `initialize.py` — Round initialization and plan checking
    - `state.py` — WorkflowState TypedDict definition
    - `retry.py` — @with_retry decorator and backoff logic
    - `helpers.py` — Shared utility functions
    - `context.py` — Planning context building
    - `graph.py` — LangGraph state graph definition
    - `messaging.py` — WebSocket message sending
  - **Backward compatibility**: The `__init__.py` exports `SingleCellAnalysisWorkflow`, `WorkflowState`, `with_retry`, and `_retry_backoff_delay` to preserve existing import paths.
  - **New tests**: Added `test_workflow_helpers.py` and `test_workflow_routing.py` for the refactored code.

- `0.24.27` — stringified tool-call recovery & unknown-tool dispatch guard (2026-05-21):
  - **Stringified tool-call recovery**: Added `_recover_stringified_tool_call()` helper that detects when an upstream gateway (notably litellm proxying n1n.ai for Claude/Gemini) loses the structured `tool_calls` field and serializes the tool-use JSON into `message.content`. Recognizes three shapes: n1n.ai (`{"tool":...,"parameters":...}`), OpenAI (`{"name":...,"arguments":...}`), and Anthropic native (`{"type":"tool_use","name":...,"input":...}`). Strips ` ```json ``` ` fences, synthesizes a ToolCall with a `recovered_`-prefixed ID, clears content, and syncs `raw_assistant_message` for Gemini thought-signature preservation.
  - **Unknown-tool dispatch guard (orphan prevention)**: Added `KNOWN_TOOL_NAMES` frozenset derived from TOOLS in the bioinformatics agent. A new `else` branch in the dispatch chain catches hallucinated or recovered tool names that don't match any known tool, emitting a `role=tool` response listing available tools. Prevents orphan tool_call IDs that would cause 400 errors on subsequent requests.
  - **Tests**: Added `test_stringified_tool_recovery.py` (12 tests) and `test_unknown_tool_dispatch.py` (6 tests) covering all recovery shapes, fence handling, plain-text rejection, native tool-call passthrough, and dispatch chain completeness.

- `0.24.26` — modality-branched workflow, concurrency hardening, LLM client resilience (2026-05-21):
  - **Modality-branched workflow (P1-4)**: `workflow_for_modality()` in `agent_prompts.py` provides separate analysis workflows for scRNA-seq, bulk RNA-seq, and multimodal data. Bioinformatics agent's `_infer_modality()` drives modality-aware system prompt identity so the LLM knows what kind of analysis it's conducting.
  - **Bulk-only instruction isolation**: Agent nodes no longer impose scRNA-dependent cross-validation or deconvolution on bulk-only sessions. New `_has_stage1_keyword()` helper excludes "re-" prefixed terms (re-cluster, re-annotate) from stage1 filtering. Bulk RNA-seq terms added to `STAGE1_KEYWORDS`.
  - **PI unilateral final plan (P1-10)**: `PI.generate_final_plan_from_dialogue()` produces a plan when `MAX_PLANNING_ROUNDS` is exhausted without agreement, replacing infinite revise loops.
  - **Direction queue commit on round entry (P1-1)**: Directions are now consumed at `_initialize_round` only after the prior round's LangGraph checkpoint confirms success, avoiding pre-mature pop with no rollback path.
  - **PI_DECIDE moved to workflow invariant (P3-5)**: Directionless-INNOVATION default moved from the resume route into `_initialize_round` so it applies consistently regardless of how the workflow resumes.
  - **Per-session lock on HITL endpoints (P1-9)**: All interaction/lifecycle mutation endpoints wrapped with session-scoped `asyncio.Lock` to prevent race conditions from concurrent pause/resume/stop/llm-failure calls.
  - **Resume no longer auto-signals HITL gates (P0-5)**: Generic resume now awaits explicit user action at the restored WAITING_* gate rather than blindly emitting signals.
  - **Supplementary queue FIFO**: Replaced single `_supplementary_request`/`_supplementary_target` fields with `deque` so consecutive rapid requests are preserved in order.
  - **Cache invalidate on mid-execution plan revision (P0-8)**: Bioinfo cache cleared so post-revision re-execution doesn't return stale pre-revision results.
  - **Exploration success whitelist**: `_exploration_round_succeeded()` now only accepts `status=="completed"` — all other outcomes (including `completed_with_fallback`, `max_steps_reached` with findings) block EXPLORATION→INNOVATION transition.
  - **No double-clear of _continue_event (P0-6)**: Prevents signals delivered during LangGraph checkpoint I/O from being silently discarded, fixing the "click twice" bug.
  - **should_continue=False on planning failure (P0-7)**: `_after_review` no longer loops infinitely on "revise" when the LLM fails.
  - **LLM client resilience**:
    - **Cache lock serialization (P2-1)**: `asyncio.Lock` guards on `_temp_unsupported_models` and `_toolchoice_unsupported_models` to prevent race conditions in concurrent first-time probes.
    - **max_tokens field override cache (P3-4)**: Adaptive tracking of which field (`max_tokens` vs `max_completion_tokens`) a provider accepts, handling o-series variants routed through providers the static `_is_o_series` heuristic misses.
    - **ToolCall parse_error propagation (P2-2)**: JSON parse failures now surface `parse_error`/`raw_arguments` to the caller so the bioinfo agent can retry with valid JSON instead of silent KeyErrors.
    - **Reasoning extraction refactor**: `_collect_reasoning_parts()` helper handles str/dict/list elements robustly across both `model_dump` and `getattr` code paths.
  - **Data-driven system prompt**: Bioinformatics agent's `_static_system_prompt` caches by `(system_prompt, modality)` tuple so hot-reload and data-driven identity compose correctly.
  - **Multiple file format load hints**: Added format-appropriate load hints for `.h5ad`, `.h5` (10x), `.loom`, `.csv`, `.tsv`, `.mtx`, `.zarr`, `.rds`, `.h5seurat` in bioinformatics agent's data context building.
  - **Configurable DEG thresholds (P2-11)**: `deg_thresholds` field on `AnalysisState` for session-configurable logFC/padj cutoffs; propagated to report figures and bioinformatics annotate_clusters.
  - **Sandbox error hardening (P1-11)**: dill restore failure now raises `RuntimeError` instead of silently continuing with empty state; `save_session` tracks and logs unpicklable variables via `__SCAI_SAVE_SKIPPED__` markers.
  - **Snapshot parent chain fix**: `parent_id` no longer falls back to `snapshots[-1]` by time — leaves as `None` (new root) to prevent parent-chain corruption in multi-branch restore.
  - **Auth & security hardening**:
    - Session auth suffix matching anchored to path segment position; defense-in-depth against traversal segments and nested `sessions/` paths.
    - `validate_file_path_ownership()` rejects multiple `sessions/` segments and verifies resolved path is within `get_session_dir()`.
    - Import archive rejects symlinks/hardlinks/devices and uses `is_relative_to()` for traversal check.
    - Chunked upload endpoints now enforce session ownership validation.
  - **ANSI regex expansion**: Covers CSI, OSC (BEL-terminated), and charset designations.
  - **Figure type from agent metadata (P2-10)**: `_infer_figure_type()` prefers agent-declared `figure_type` over filename heuristics; `_scan_existing_figures()` builds declared-type map from `publication_figures`.
  - **Report DEG table summary**: Honors session-configurable `deg_thresholds` for significant-gene counts.
  - **Session list dedup (P3-8)**: Merged disk+memory sessions by ID to prevent duplicate entries.
  - **Export filename length cap**: `MAX_EXPORT_NAME=200` with UUID suffix when truncation is needed.
  - **Frontend hardening**:
    - Typed HITL data interfaces (AnnotationData, PlanReviewData, ExplorationReportData) replace `any`.
    - Plan review countdown timer resets on plan identity change.
    - Message merge uses transactional `setState` to avoid loss during async fetch.
    - Aborts analysis on agent config sync failure.
    - Handles negative clock delta in message TTL persistence check.
  - **40 new test files** covering: auth optional paths, bioinfo modality, direction gating, continue event, data format dispatch, DEG thresholds, directions queue commit, exploration success whitelist, reasoning extraction robustness, figure type metadata, file path ownership, import archive safety, LLM client cache locking, max planning rounds fallback, modality branched workflow, no silent validation error, ANSI/max-tok handling, batch modality recognition, routes housekeeping, pathology generic grading, PI review failure routing, plan revision cache invalidation, resume with no auto signal, sandbox failure surfacing, session locking, snapshot parent chain, supplementary deque/target round, tool call parse failure, upload ownership, user plan feedback preservation.

- `0.25.2` — temp-file cleanup on failure (2026-05-27):
  - **Orphaned temp file cleanup**: `externalize_blobs` wraps write+replace in try/except — on failure, removes the orphaned `.tmp` file before re-raising.
  - **Snapshot failure cleanup**: `create_snapshot` `_externalize_and_write` cleans up blob dir and temp JSON on failure, preventing stale snapshot artifacts.

- `0.25.1` — zstd streaming compression for session pkl (2026-05-27):
  - **Zstd-compressed pkl**: Session save/restore uses `zstandard.stream_writer`/`stream_reader` for streaming compression. Plain dill fallback for backward compat. Corrupt zstd raises clear `RuntimeError("zstd decompression failed (corrupt pkl)")`.
  - **Atomic pkl write**: Save writes to temp file first, then `os.replace()` — no half-written compressed file on crash.
  - **New dependency**: `zstandard>=0.23,<1.0` added to `requirements.txt`.
  - **Lazy zstd import**: `zstandard` imported only inside the zstd branch of restore code — plain-dill restore stays dependency-free.
  - **Tests**: Added `test_pkl_compression.py` (magic detection, round-trip dict/sparse, compression ratio, backward compat, corrupt zstd, kernel code compilation, exec round-trip) and `test_snapshot_rollback.py` (snapshot copies compressed/plain pkl byte-identical, detection round-trip).

- `0.24.30` — atomic blob write & snapshot I/O offload (2026-05-27):
  - **Atomic blob write**: `externalize_blobs` writes `.b64` files via temp file + `os.replace()` — no half-written file on crash.
  - **Empty-response backoff**: `_empty_retry_backoff()` adds incremental delay (capped at 5s) between empty/truncated upstream retries.
  - **Snapshot blob I/O offloaded to worker thread**: `create_snapshot` runs `externalize_blobs`, JSON serialization, and atomic file write inside `asyncio.to_thread()` — the event loop stays responsive for WebSocket pushes and LLM streaming.
  - **Tests**: Added atomic write crash-safety test (`test_blob_store.py`), backoff behavior test (`test_empty_response_handling.py`), and snapshot I/O thread verification (`test_snapshot_blob_offload.py`).

- `0.24.29` — blob externalization & retry hardening (2026-05-27):
  - **Base64 blob externalization**: New `blob_store.py` module extracts large base64 strings (images, figures) from session JSON into SHA256-deduplicated `.b64` files. Integrated into `session_store.py` and `snapshot.py` for both sync/async save and load. Reduces session.json size and write overhead significantly.
  - **Hard quota vs transient rate-limit classification**: `retry_with_backoff` now distinguishes hard 429s (monthly max, insufficient_quota) from transient per-window limits; 401 `AuthenticationError` and `PermissionDeniedError` surface immediately as `UpstreamExhaustedError` without burning retries.
  - **Empty/truncated response handling**: `_is_empty_response()` detects zero-token, whitespace-only, and extraction-placeholder responses as transient upstream truncation; retries up to 3 times without counting toward the text-only HITL escalation. New test suite (`test_empty_response_handling.py`).
  - **Kernel protocol error recovery**: `_is_kernel_protocol_error()` detects ZMQ signature corruption (`Invalid Signature`, `<IDS|MSG>`) and triggers `_restart_kernel()`. Refactored kernel restart into a dedicated method with reentrancy guard, proper dead-state reset, and best-effort session restore. New test suite (`test_kernel_recovery.py`).
  - **Defensive routing hardening**: `_check_convergence_condition` missing key defaults to `continue` (not `stop`); `_initialize_round` wrapped with `@with_retry`; `_user_review_gate` sets `should_continue=False` when no plan exists. New test suite (`test_routing_defensive.py`).
  - **Supplementary plan safety**: `plan_supplementary_analysis` result guarded with `or {}` to prevent `None` crash.
  - **Tests**: Added `test_blob_store.py`, `test_empty_response_handling.py`, `test_kernel_recovery.py`, `test_retry_classification.py`, `test_routing_defensive.py` (5 new test files, ~400 lines).

- `0.24.28` — unknown tool defense & stringified tool recovery (2026-05-27):
  - **Orphan tool call defense**: Bioinformatics agent now handles unknown/hallucinated tool names gracefully by responding with an error message rather than crashing. Added `KNOWN_TOOL_NAMES` frozenset derived from TOOLS; `run_autonomous_analysis()` now emits a role=tool error response for unknown tool calls, keeping conversation well-formed.
  - **Stringified tool recovery**: New regression tests for recovering from serialized tool_call IDs via dill checkpoint restore (`test_stringified_tool_recovery.py`).
  - **Unknown tool dispatch tests**: New test suite for handling unknown tool names dispatched by LLM (`test_unknown_tool_dispatch.py`).
  - **LLM client improvements**: Enhanced `OpenAICompatibleClient` with better error handling and retry logic.

- `0.24.26` — multi-agent modality support & robustness fixes (2026-05-22):
  - **Multi-modality analysis**: Full support for multi-omics data (scRNA-seq, snRNA-seq, bulk RNA-seq, pathology images, biochemical indicators) with dynamic modality detection and branched workflow routing.
  - **PI Agent enhancements**: Added NMF gene-expression programs and RandomForest biomarker discovery to Round-2+ toolkit; restructured task-generation prompts with cascading storyline (Task/Rationale/Expected Outcome); literature verification loop for each bioinformatics hypothesis.
  - **Bioinformatics agent improvements**: Multi-modal fallback rules with `print()` of raw data backing every plot; code-level self-repair with try/except and adaptive DEG thresholds; serendipity check (variance partitioning) before targeted analyses.
  - **Sandbox robustness**: dill restore failure now raises `RuntimeError` instead of silently continuing; session save tracks unpicklable variables via `__SCAI_SAVE_SKIPPED__`.
  - **Auth middleware**: Session auth suffix matching anchored to path segment position; defense-in-depth against traversal segments.
  - **Configurable DEG thresholds**: `deg_thresholds` field on `AnalysisState` for session-configurable logFC/padj cutoffs.
  - **40+ new test files** covering: auth optional paths, bioinfo modality, direction gating, continue event, data format dispatch, DEG thresholds, directions queue, exploration success, reasoning extraction, figure type metadata, file path ownership, import archive safety, LLM client cache locking, max planning rounds, modality branched workflow, snapshot parent chain, tool call parse failure, and more.

- `0.24.25` — tool-choice fallback & exploration failure guard (2026-05-21):
  - **tool_choice BadRequestError fallback**: Added `is_tool_choice_error()` detection and `_toolchoice_unsupported_models` cache in `OpenAICompatibleClient`. When a provider rejects `tool_choice="required"`, the client caches the model and retries with `tool_choice="auto"`; subsequent calls are pre-emptively demoted. Fixes crashes on providers that don't support forced tool calling.
  - **Text-only exhaustion escalates properly**: Bioinformatics agent now raises `UpstreamExhaustedError` instead of silently fabricating fallback findings when MAX_TEXT_RESPONSES consecutive text-only responses occur. A checkpoint is saved before escalation so Retry from the HITL gate resumes the conversation rather than starting fresh.
  - **Last-chance tool forcing**: `_compute_tool_choice()` returns `"required"` on the last text-only budget attempt to nudge models (especially Gemini-3.x-Pro) into actually calling a tool instead of replying with prose.
  - **Exploration round success guard**: Added `_exploration_round_succeeded()` predicate — blocks Report A generation and EXPLORATION→INNOVATION phase transition when bioinformatics failed/errored or exhausted steps without findings. Prevents empty exploration reports and lost retry opportunities.
  - **Tests**: Added `test_bioinfo_text_only_exit.py`, `test_exploration_round_succeeded.py`, and `test_tool_choice_fallback.py` with unit and integration tests covering all paths.

- `0.24.24` — plan review timer pause/resume (2026-05-21):
  - **Timer pause/resume**: PlanReviewDialog now supports pausing and resuming the countdown timer. Users can pause the timer during plan review and resume when ready.
  - **Timer display**: Countdown now displays in MM:SS format with visual styling for urgency (red pulse when ≤30s) and paused state (amber indicator).
  - **New hook features**: `useCountdownTimer` hook now exposes `isPaused`, `pause`, `resume`, and `hasCountdown` for timer control.

- `0.24.7` — stage-aware planning & stop/resume fixes (2026-05-15):
  - **PI stage context**: Added stage context to PI callback prompts so PI knows whether agent is in stage1 (QC+clustering+annotation only) or stage2 (downstream analysis).
  - **PLAN_REVISION_KEYWORDS**: Extracted plan revision keyword list to constants.py for reuse across codebase.
  - **Frontend countdown refactor**: AnalysisPanel now uses captured `roundWaitSeconds` and `maxRounds` from store hooks instead of calling `useAppStore.getState()` directly in JSX.
  - **Stop/resume max_rounds restoration**: Added `original_max_rounds` field to save original value during stop, restored on resume. Fixes round limit being permanently reduced after stop operations.
  - **PI_REVIEW_FALLBACK_DECISION**: Added constant for default PI review decision when LLM fails.

- `0.24.6` — (skipped — consolidated into 0.24.7)

- `0.24.5` — model serialization fix & planning improvements (2026-05-15):
  - **Pydantic model_dump mode="json"**: Changed all `model_dump()` calls to `model_dump(mode="json")` across workflow, bioinformatics_agent, schemas, and snapshots to properly serialize datetime objects and avoid Pydantic serialization warnings.
  - **Collect all PI feedback**: Planning dialogue now collects PI feedback from all rounds, not just the last one, for better context in plan revisions.
  - **Plan status guard**: Fixed plan status handling to not reinstate a PI-rejected plan as "pi_reviewing".
  - **Bioinformatics findings in context**: Added `bioinformatics_findings` to planning context aggregation alongside biochemistry and pathology.
  - **PlanReviewDialog countdown fix**: Refactored to use `useCountdownTimer` hook with proper submit guards to prevent double-submissions and ensure onSkip triggers correctly.
  - **stage2 edge case handling**: Added warning and explicit routing when plan_approved=True is unexpectedly reached in stage2.
  - **Plan approval state sync**: Added `_set_status(analysis_state, TaskStatus.RUNNING)` and `_send_state_sync` when plan is approved to properly update state.

- `0.24.4` — custom data type support (2026-05-15):
  - **Custom data type input**: Added ability to enter custom data type names in file uploads. Users can now type custom values (e.g., "Sirius Red", "IHC") beyond the predefined dropdown options for pathology images.
  - **New utility functions**: Added `CUSTOM_TYPE_SENTINEL` sentinel value ("Other"), `isCustomDataType()`, and `selectValueForDataType()` in `dataTypes.ts`.
  - **New CustomTypeInput component**: Created reusable `CustomTypeInput.tsx` component in `components/shared/` that shows a text input when dataType is custom.
  - **Frontend integration**: Updated `ChatPanel.tsx` and `RoundFeedbackDialog.tsx` to use the new custom type input with proper value derivation.
  - **Pathology cleanup**: Reorganized imports in `pathology_agent.py`.

- `0.24.3` — planning dialogue improvements (2026-05-14):
  - **Refactored retry logic**: Extracted shared `_retry_backoff_delay()` helper used by `@with_retry` decorator and inline retries in workflow and report generation.
  - **Plan approval propagation**: `_make_initial_workflow_state()` now propagates plan approval from `analysis_state` when resuming without active workflow, ensuring downstream routing detects approved plans.
  - **Planning context isolation**: `_start_new_round()` now resets `planning_context` to prevent stage1 data from leaking between rounds.
  - **Skip re-planning for approved plans**: `_should_plan()` detects when `current_plan.status == "approved"` and preserves approval instead of regenerating, enabling proper downstream routing.
  - **Configurable planning rounds**: Changed from hardcoded `MAX_ROUNDS=5` to configurable `MAX_PLANNING_ROUNDS=5` with while loop for cleaner iteration.
  - **Plan revision signaling**: Introduced `PLAN_REVISION_NEEDED_TAG` constant in `constants.py` (replaces hardcoded `"[__PLAN_REVISION_NEEDED__]"`); updated bioinformatics agent and workflow to use it.
  - **Dialogue history serialization**: Refactored `_serialize_dialogue_history()` to recursively handle nested datetime objects.
  - **Plan review fix**: Fixed `submit_plan_review` endpoint to set correct final status directly instead of being overwritten; added validation preventing "modify" action without active workflow.
  - **Async optimizations**: Moved snapshot atomic write and notebook state save to `asyncio.to_thread()` to avoid blocking event loop.
  - **Cleanup**: Removed unused `pending_plan_revision` field from `WorkflowState`; removed unused `random` import from reports.py.

- `0.24.2` — version bump (2026-05-13): Version bumped to 0.24.2 across all files; HANDOVER.md modification history updated.
- `0.24.1` — collaborative analysis planning (2026-05-13): Bioinformatics agent: `generate_analysis_plan()` and `revise_plan_incrementally()` methods for method-level analysis planning with stage-aware decomposition (stage1: QC+clustering+annotation, stage2: downstream); plan injected into agent context during autonomous execution. New Pydantic models: `AnalysisPlan`, `PlanStep` with field types (`SnapshotPhase`, `AnalysisPhaseType`, `PlanStage`, `PlanStatus`). New `WAITING_PLAN_REVIEW` task status for plan review HITL gate. Frontend: `PlanReviewDialog` component with expandable steps, countdown auto-skip, approve/modify/skip actions, and dialogue download; integrated into `AnalysisPanel`. Workflow: `planning_dialogue` and `user_review_gate` nodes added to `NODE_TO_AGENT`/`NODE_TO_RESULT_KEY` routing; `current_plan` and `plan_dialogue_history` fields in `AnalysisState`. API: `POST /session/{id}/plan-review` endpoint for approve/skip/modify actions; lifecycle resume/stop extended for `WAITING_PLAN_REVIEW`. Snapshot: strongly typed `SnapshotPhase` with `pre_execution_stage1`, `pre_execution_stage2`, `pre_execution_full`, `pre_major_revision`, `post_report` phases; expanded docstring in `SnapshotManager`. Tests: `test_planning_ordering.py` with regression tests for plan review workflow and planning context ordering; `test_bugs_confirmed.py` with PlanStep mutable default safety test.

**0.23.x — report enhancements, chat sections, round tracking (2026-05-08 → 2026-05-10)**

- `0.23.6` — **Report download & export improvements**: Export now saves to `UPLOAD_TEMP_DIR` instead of reports dir; filename format changed to `export_{session_id[:8]}_{safe_name}.tar.gz`; `_should_skip_in_report_zip()` helper excludes report archives and orphan .zip.tmp files from report ZIP downloads; `download_full_report` now uses streaming temp file with background cleanup; frontend shows download errors inline instead of console errors.

- `0.23.5` — **LLM failure intervention & report robustness**: Extracted `_wait_for_llm_intervention()` helper for LLM failure handling; `UpstreamExhaustedError` now propagates from `generate_final_summary`, `generate_detailed_report`, `generate_graphic_abstract`, and report route (with 3 retries + exponential backoff); supervisor report generation also propagates UpstreamExhaustedError; section title safety net replaces placeholder text with descriptive defaults; improved retry messaging via WebSocket (`retrying` type).

- `0.23.4` — **Bioinformatics upstream error propagation**: Fixed `UpstreamExhaustedError` from bioinformatics agent not propagating to `@with_retry` decorator; bioinformatics agent now raises instead of catching; workflow caller no longer catches and returns early; ensures LlmTimeoutDialog triggers for upstream exhaustion in bioinformatics node; comprehensive regression tests added.

- `0.23.3` — **Python client session export**: Added `export_session()` method to Python client for exporting sessions as tar.gz archives with streaming download, atomic write (`.partial` + rename), configurable timeout and chunk size.

- `0.23.2` — **Snapshot restore synchronization & config sync**: Refactored snapshot restore with per-session locking to prevent race conditions; `_write_state_if_current()` helper prevents stale writes from superseded workflows; improved error message "No API key configured..."; removed unnecessary API key check from session creation endpoints (`start_session`, `ensure_session`); WebSocket observer skips sending stale status during snapshot restore; workflow defers to HITL-hold path when status is WAITING_INPUT/WAITING_ANNOTATION; frontend shows alert on session creation failure; frontend config sync: App startup now syncs non-admin user's llmBaseUrl from backend to store; ConfigPanel saves both user LLM config to backend and updates store for in-session use.

- `0.23.1` — **Report enhancements, chat sections, round tracking**: Added report generation concurrency guard (409 on duplicate requests); automatic report archiving before regeneration (keeps last 5 versions); archive listing/download API endpoints; `round_number` field added to `AnalysisStep` schema for round tracking; supervisor log restored from report.json now groups by round (`all_rounds_log` vs current `supervision_log`); new chat sections generation from chat messages (`chat_sections.py`); config resolver now allows per-user config without global config; new `/user/config/validate-saved` endpoint for validating saved credentials; frontend validation flow: admin validates with frontend config, non-admin validates saved config; validation on report generation (check API key before start); session cleanup clears report generation pending flag.

**0.22.x — per-user LLM config, config persistence, reproducibility toggle (2026-05-03 → 2026-05-08)**

- `0.21.5` — Session priority loading & automatic tab sync: new `sessionPriority.ts` utility with intelligent session selection (priority tiers: waiting_input > active > paused > completed > pending); `waiting_input` sessions sorted by deadline urgency; `syncTabToSessionStatus()` auto-switches tab based on session status (pending→data, active→analysis, completed→report); Sidebar auto-loads highest-priority session on startup (via `doPriorityLoad()`); Zustand hydration tracking with `_hydrated` flag in `appStore.ts` (deferred flag flip via `onRehydrateStorage()`); `setActiveTab()` optimized to skip no-ops; backend: `waiting_input_started_at` timestamp in `AnalysisState`; `DEFAULT_ROUND_WAIT_SECONDS` extracted to `schemas.py`; session list API includes `waiting_input_started_at`, `round_wait_seconds`, and `created_at`; user guide fully reorganized and updated (both English and Chinese versions).
- `0.21.4` — Settings provider hardening & user guide refactor: moved `from backend.state import sessions` to module-level import in `settings_provider.py` (removed deferred import); simplified `get_temperature()` to read custom temperatures directly instead of `merge_custom()` round-trip; inlined model/temperature reads in `chat.py` for PI and bioinformatics figure generation (bypassed `AgentSettingsProvider` for those two call sites); fixed `PainterAgent` log to use `self.model` instead of stale `model` param; removed unused `AgentConfig` import from `pi_agent.py` and redundant `from ..utils.settings_provider` import from `literature_agent.py`; fixed missing trailing newline in `settings_provider.py`. Added standalone Chinese `user-guide.md` at project root; updated all user-guide link references in `README.md` and `HANDOVER.md` from `docs/user-guide.md` to `user-guide.md`.
- `0.21.3` — Unified Agent Settings Provider: new `AgentSettingsProvider` class in `backend/utils/settings_provider.py` that unifies reading of model, temperature, system_prompt, vision setting, and LLM client for all agents with hot-reload support (reads fresh state from global `sessions[session_id]` on each call); `BaseAgent` refactored to take a single `settings_provider` argument with all settings exposed as `@property`s; all 8 agents updated; `workflow.py` simplified (-233 lines net); `chat.py` and `reports.py` use provider; documentation cleanup (`docs/user-guide.md` and design spec removed).
- `0.21.2` — Username-based authentication: login and registration now require a username (3–64 chars, alphanumeric/underscore/hyphen); email is optional. New `UsernameStr` / `PasswordStr` Pydantic types; `check_user_uniqueness` validation; unique indexes on `users.username` and `users.email` (where not null). Admin setup accepts optional email. JWT payload stores `username` instead of `email`. Frontend: Login, Register, Setup, Admin, Settings, ApiKeys, UserMenu all updated; i18n (zh/en) for auth pages. Removed legacy `get_admin_user`; `create_admin_if_missing` returns `(id, created)`.
- `0.21.1` — Reproducibility bundle generation for analysis sessions: `ReproducibilityBundle` service captures environment info (Python version, platform, CPU, memory, ScAI version), frozen pip requirements, SHA256 checksums for all session files, executable Python commands, and agent decision logs; bundle included in session export/import and report generation; download-reproducibility API endpoint; Reproducibility appendix section in HTML report with environment summary, key packages, checksum table, and download button; `import_snapshot.py` utility script.

**0.20.x — auth, modularization, and admin (2026-04-21 → 2026-04-28)**

- `0.20.12` — Python client: added `list_sessions()` to list all analysis sessions and `import_session()` to import session archives with polling support; added `ImportFailed` exception class; version bumped to 0.2.0. Backend: refactored WebSocket route imports to top-level.
- `0.20.11` — Bioinformatics Agent: multi-modal fallback rules requiring `print()` of raw data backing every plot for downstream agents; code-level self-repair with try/except and adaptive DEG thresholds; serendipity check (variance partitioning) before targeted analyses to flag hidden covariates. PI Agent: added lightweight NMF gene-expression programs and RandomForest biomarker discovery to Round-2+ toolkit; task-generation prompt restructured to require cascading storyline with Task/Rationale/Expected Outcome fields; literature verification loop for each bioinformatics hypothesis.
- `0.20.10` — session import now requires authentication and stamps `user_id` on imported sessions (tar.gz / zip / JSON paths); per-session round-wait setting editable from session panel via new `/session/{id}/round-wait` endpoint; round-wait options extracted to shared `constants/roundWait.ts`; `loadSession` syncs `round_wait_seconds` from server.
- `0.20.9` — annotate_clusters robustness (cluster_key/rank_key hints, expanded cell-type guard regex, retry loop with supervisor nudge); upstream LLM exhaustion handling (`UpstreamExhaustedError`, extended 5xx retries); WebSocket reconnect on resume; HANDOVER/README rewrite.
- `0.20.8` — `backend/config/` package split; new `AgentPrompts` model; `StopDialog` / `ValidationDialog` components; `useCountdownTimer` and `messageTransform` utilities; auth-optional paths now pass through missing DB users (ownership enforced at route layer); dead `BioinformaticsCheckpoint` removed.
- `0.20.7` — `AUTH_OPTIONAL_PATHS` middleware; shared `PI_AGENT_NAME` constant; `_TECHNICAL_ERROR_TYPES` extracted to `utils/error_parsing`; session-cleanup refactor.
- `0.20.5–0.20.6` — Report generator split into `backend/services/report/` (analysis_log, figures, generator, llm_calls, pi_review, planning, regenerator, scoring, sections, variants); dead `report_generator.py` removed.
- `0.20.4` — Centralized frontend auth init; passlib → direct `bcrypt` (72-byte limit); structured API-key validation responses.
- `0.20.3` — Admin management panel; first-run admin bootstrap; user statuses (`pending` / `readonly` / `active`); user profile self-service.
- `0.20.1–0.20.2` — User authentication with JWT; WebSocket auth; session ownership; large backend modularization (routes/, services/, middleware/, utils/, tasks/); session-persistence service; workflow runner; session TTL cleanup; chat delegation system.

**0.19.x — bulk RNA-seq & literature (2026-04-09 → 2026-04-19)**

- Full bulk RNA-seq support with cross-validation against single-cell pseudobulk; mandatory in Innovation when both are present.
- Cell-type deconvolution mandatory when bulk + scRNA-seq present; standardized library names (pydeseq2, harmonypy, bbknn, scvi, decoupler) and PyDESeq2 fallback.
- Dynamic PI task generation based on detected data types; added GIF/BMP/WebP/SVG image support.
- Improved LLM prompt caching and cache-hit tracking; round-progress UI uses the real max-rounds setting.
