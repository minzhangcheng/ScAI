# ScAI Handover

Architecture, module map, deployment, and developer guide for operators and contributors. For end-user workflow, see [user-guide.md](user-guide.md); for the project pitch and installation, see [README.md](README.md).

**Version**: 0.22.3 · **Last updated**: 2026-05-04

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

**0.22.x — reproducibility toggle, config-driven feature flags, import hardening (2026-05-03 → 2026-05-04)**

- `0.22.3` — Import security hardening & memory optimization: `validate_path_within()` added to `backend/utils/files.py` for path-traversal protection; `_resolve_import_dest()` extracts archive-to-session path mapping into a testable helper; `shutil.copyfileobj()` with `UPLOAD_CHUNK_SIZE` replaces whole-file `write_bytes()` in tar.gz extraction to reduce peak memory for large archives; explicit `validate_session_id()` calls replace inline regex checks in both tar.gz and zip import paths; structured logging (`logger.info`/`logger.error`/`logger.exception`) added to `_process_import_job` for import lifecycle observability; snapshot JSON path remapping consolidated to `_resolve_import_dest` dispatch.
- `0.22.2` — Snapshot rollback enhancement: full-state restoration (ipynb notebook backup, workspace file manifest, report cleanup); `Snapshot` model extended with `notebook_file`, `workspace_manifest`, `branch_label` fields; `SnapshotManager.scan_workspace_manifest()` tracks workspace files at snapshot time; `restore_notebook()` and `cleanup_workspace_files()` enable precise rollback to snapshot state; `cleanup_reports()` removes stale generated reports and clears report path fields; `GET /rollback-preview` endpoint shows affected files, notebook cells, and descendant snapshots before restore; `DELETE /branch` endpoint deletes a snapshot and all its descendants; branch labels auto-computed in workflow (`main` / `branch-A` / `branch-B`) for tree visualization; `pre_round_wait` phase label added to frontend; frontend SnapshotPanel shows rollback preview with affected-file/cell/descendant counts and branch badge, with AbortController cleanup on unmount; comprehensive test suite (`test_snapshot_rollback.py`, 321 lines) covering manifest creation, workspace/report cleanup, branch labels, and branch deletion.
- `0.22.1` — ReproducibilityBundle opt-in toggle: added `enable_reproducibility` boolean field to `SystemConfig` (controlled by `SCAI_ENABLE_REPRODUCIBILITY` env var, default `False`); report generator's bundle creation extracted into `_maybe_generate_reproducibility_bundle()` method that checks the toggle before generating; download endpoint (`/reports/{session_id}/reproducibility`) returns 404 when toggle is off; `DataFile` and `RoundResult` imports added to generator; lazy import of `ReproducibilityBundle` inside the method; comprehensive test suite (`test_config_toggle.py`) covering config defaults, env var parsing (true/yes/1/false/0/no/empty), subprocess isolation, and generator branching logic. Aligns with ScAI's human-in-the-loop free exploration philosophy — reproducibility bundles are now opt-in rather than always-on.

**0.21.x — reproducibility bundle, username-based auth, unified settings (2026-04-30 → 2026-05-01)**

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

**0.18.x — snapshots, figure review, and providers (2026-03-27 → 2026-04-08)**

- Snapshot tree view and reliable state restoration with full Jupyter kernel preservation.
- Supervisor figure review with issue classification; auto-revision with vision-disable fallback.
- Development / production mode switching; hard limit on code-execution errors.
- Multiple image-generation providers (DashScope, Zhipu AI); per-agent LLM provider overrides; improved provider compatibility and LLM client caching.
- Advanced analysis modules: gene signature scoring, trajectory analysis, TF networks, cell-cell communication, stemness, tumor estimation, compositional analysis, immune repertoire.

**0.17.x — UI redesign and proactive docs (2026-03-26)**

- Complete UI visual redesign.
- Automatic API-documentation lookup on errors; Bioinformatics Agent can read package docs proactively.

**0.16.x — Docker and resilience (2026-03-24)**

- Official Docker image published; vision-aware figure-revision fallback; analysis timeout extended to 1.5 h.

**0.15.x — report & annotation polish (2026-03-22 → 2026-03-23)**

- Auto-generated PDF companions for every figure.
- Cell-type annotation required before the exploration report; inline tables with preview/download in reports.

**0.14.x — identity and annotation depth (2026-03-15 → 2026-03-21)**

- Project renamed from InteSc to **ScAI** (0.14.4).
- Annotation dialog surfaces alternative cell types and confidence reasons; AnnData variable-name auto-detection; improved error recovery.

**0.13.x — two-phase workflow (2026-03-12 → 2026-03-13)**

- Exploration → Innovation phase split; cell-annotation confirmation; multi-direction analysis with cross-direction correlation.

**0.12.x — supervisor & figure review (2026-03-10 → 2026-03-11)**

- Supervisor Agent for real-time monitoring; pause / stop controls; PI-reviewed publication figures with auto-revision.

**0.11.x — cross-modality biochemistry (2026-03-09)**

- Biochemistry results feed into other agents for cross-referenced findings.

**0.10.x — literature depth & logs (2026-03-07 → 2026-03-09)**

- Deep Research mode for comprehensive literature surveys; full chronological analysis log.

**0.9.x — publication-ready reports (2026-03-04)**

- PI reviews notebooks and figures before reports; reports include project-info appendix; graphic-abstract generation and revision.

**0.3.x – 0.8.x — foundation (2026-02-04 → 2026-03-03)**

- Core multi-agent architecture; treatment-group system; chat interface with PI; Painter Agent; report generation; literature-based annotation.
