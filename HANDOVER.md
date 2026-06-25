# ScAI Handover

Architecture, module map, deployment, and developer guide for operators and contributors. For end-user workflow, see [user-guide.md](user-guide.md); for the project pitch and installation, see [README.md](README.md).

**Version**: 0.29.3 · **Last updated**: 2026-06-25

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

**0.29.x — (2026-06-24 → 2026-06-25)**

- `0.29.3` — Refactoring pass: HITL gate scaffold, dead-code removal, UI component extraction, lazy init, query dedup (2026-06-25):
  - **Shared HITL gate scaffold (`backend/workflow/core.py`)**: Extracted `_drive_hitl_gate()` unifying lock/event/generation/state_sync pattern duplicated across `_wait_for_llm_intervention`, `_wait_for_disabled_agent`, and `_wait_for_marker_db_intervention`. Each caller parameterizes only what differs (reason, payload, change types, cache-on-retry). Replaces ~120 lines of near-identical code.
  - **Direction evidence shared factory (`backend/utils/direction_evidence.py`)**: New `build_fallback_direction_evidence()` in utils — both `PIAgent` and `SynthesisMixin` consume it without the agent layer importing the orchestration layer.
  - **PI annotation summary helper (`backend/agents/pi_agent.py`)**: Extracted `_format_cluster_annotation_summary()` replacing inline annotation-summary construction duplicated across `generate_raw_direction_candidates` and `score_directions_with_evidence`.
  - **Score badge & species-label unification (`backend/services/report/variants.py`)**: Extracted module-level `_render_score_badge()` and `_SPECIES_TISSUE_LABELS` map — removes two inline score-badge helpers (`score_badge`, `_sb`) and inline `_species_tissue_labels` dict.
  - **Query batch loop dedup (`backend/agents/literature_agent.py`)**: Extracted `_run_query_batches()` replacing duplicated inline loop across standard and deep search paths. Extracted `_dedup_by_pmid()` static method replacing hardcoded dedup inline.
  - **Lazy MarkerDB (`backend/agents/bioinformatics_agent.py`)**: Replaced eager `_MARKER_DB = MarkerDB()` with `_get_marker_db()` lazy accessor. Hoisted `ora_kernel` source to module-level constant.
  - **O(1) candidate lookup (`backend/agents/pi_agent.py`)**: `cand_by_id` dict in prerequisite closure loop replaces O(n) `next()` scan.
  - **Focused `exhausted()` helper (`backend/agents/literature_agent.py`)**: Local closure in `_focused_rare_search` consolidates 9-line exhausted-result construction.
  - **Dead `mouse_liver_*` qualification (`backend/models/schemas.py`)**: Removed `mouse_liver_available` / `no_mouse_liver_evidence` from `SpeciesTissueQualification` — unused by evidence layer.
  - **Frontend `DirectionEvidence` component extraction**: Inline 130-line evidence rendering IIFE from `ExplorationReportDialog.tsx` moved to standalone `components/analysis/DirectionEvidence.tsx` with `language`/`expanded`/`onToggle` props.
  - **Frontend `LlmTimeoutDialog` refactoring**: Unifies marker-db and agent-disabled branches via `isMarkerDb`/`isAgentDisabled` booleans instead of duplicate JSX trees.
  - **Cleanup**: Removed redundant `asyncio` imports, dead assignments, no-op map, unused imports across `backend/agents/literature_agent.py`, `backend/services/marker_priors.py`, `backend/workflow/`; fixed unbalanced JSX in `LlmTimeoutDialog`.

- `0.29.2` — Evidence-gated direction selection: PI raw candidate generation, multi-axis literature evidence review, scored direction table with full audit trail (2026-06-25):
  - **Direction selection evidence gating (`backend/workflow/synthesis.py`)**: New `_generate_evidence_gated_exploration_report()` splits exploration report creation into three phases — (1) PI generates raw direction candidates (`generate_raw_direction_candidates`), (2) literature agent evaluates each candidate via multi-axis PubMed queries (`evaluate_direction_novelty_candidates`), (3) PI scores candidates with evidence (`score_directions_with_evidence`). Skips literature gate when Literature Agent is disabled. Stays in EXPLORATION phase on failure instead of advancing to INNOVATION. Dependency graph validated via `_validate_direction_dependency_graph()`.
  - **PI two-phase direction generation (`backend/agents/pi_agent.py`)**: `generate_exploration_report()` refactored into `generate_raw_direction_candidates()` (3-5 candidates with biology context, hypothesis, key genes/cell types/pathways, prerequisites) and `score_directions_with_evidence()` (assigns overall_priority, data_reliability, literature_novelty, biology_expectedness, data_novelty, mechanistic_importance, functional_impact, data_sufficiency, methodological_gap, risk_flags per candidate). `select_best_directions()` returns structured `DirectionSelectionDecision` with selected/rejected directions and rationale. `_validate_direction_dependency_graph()` enforces prerequisite closure with cycle detection and reachability.
  - **Direction literature evidence models (`backend/models/schemas.py`)**: New `DirectionSearchQuery` (per-axis query with result/relevant counts, degraded flag), `PubMedArticle` (article reference), `PubMedEvidenceItem` (PMID with relevance/support/source/species/tissue context), `DirectionLiteratureEvidence` (per-direction: status, search_tier, literature_novelty, confidence, negative_validation_axes, evidence counts by support type, direct/analogous/contradictory/methodological evidence lists, species_tissue_qualification, gap_assessment, summary, limitations), `ResearchDirectionCandidate` (scored candidate with all evaluation fields, evidence_ref), `DirectionSelectionDecision` (user's selection with rejected directions). `ExplorationReport` extended with `raw_direction_candidates`, `direction_literature_evidence`, `direction_dependency_graph`, `direction_selection_summary`.
  - **Multi-axis PubMed query engine (`backend/agents/literature_agent.py`)**: `evaluate_direction_novelty_candidates()` drives 8 query builders per direction — `_build_direct_prior_work()` (gene/term scoped to species + tissue), `_build_analogous_mechanism()` (pathway/gene across species+tissue), `_build_contradictory_expectedness()` (gene/pathway + adverse/negative keywords), `_build_methodological_precedent()` (methods scoped to species+tissue), `_build_gene_model_relaxed()` (gene-only fallback), `_build_cell_type_focused()` (cell type ± tissue), `_build_pathway_focused()` (pathway + gene), `_build_cross_species_or_tissue()` (core mechanism excluding target organism). `_build_degraded_fallback()` comprehensive fallback when meaningful term count is low (`_meaningful_term_count()`).
  - **Negative validation & species/tissue qualification**: `_evaluate_negative_validation_axes()` scores evidence for each axis (confirms, contradicts, no evidence) with `_judge_axis_evidence()` LLM gating. `_determine_species_tissue_qualification()` classifies evidence as `matched_species_tissue_available`, `mouse_liver_available`, `no_matched_species_tissue_evidence`, `no_mouse_liver_evidence`, `cross_tissue_only`, or `cross_species_only` — threaded through evidence items and stored as `species_context`/`tissue_context` per PMID.
  - **Novelty gating & judge**: `_judge_direction_novelty()` LLM call scores each direction along literature_novelty (high/medium/low), confidence (high/medium/low/failed), gap_assessment, summary, limitations. Implements `gating_precedent_high_novelty` rule: low relevance or failed `methodological_precedent` blocks high novelty assignments (`literature_novelty` capped at `medium`). Novelty requires `confidence ≥ medium` AND `negative_validation_confidence ≥ low`.
  - **Cache-based article dedup & term overlap**: `_deduplicate_evidence_by_pmid()` merges high-relevance duplicates keeping the best relevance/support, low-relevance duplicates merged into best count. `_cache_article_matching()` caches PMID→article mappings per session for cross-session reuse; `_article_matches_candidate()` returns `False` when no term overlap (gene/cell/pathway) exists despite PMID match.
  - **Deep search escalation**: `_should_escalate_to_deep_search()` escalates when (a) direct_prior_work count < 2 AND (b) any axis is degraded, or (c) methodological_precedent count < 1. `escalate_to_deep_search_on_degraded` / `escalate_to_deep_search_on_missing_methodological_precedent` flags. Deep research tasks built with consolidated queries per candidate plus fused topic list.
  - **Report evidence rendering (`backend/services/report/variants.py`)**: `_render_direction_evidence_section()` produces summary table (direction, overall_priority, data_reliability, literature_novelty, evidence status) with badge styling (High/Medium/Low, Completed/Partial/Failed). Per-direction detail section with species/tissue qualification label, direct/analogous/contradictory/methodological evidence lists (PMID, title, relevance, why_it_matters), negative validation axes, gap assessment, prerequisite dependencies, risk flags. `evidence-failed-row`/`evidence-partial-row` CSS classes for visual status.
  - **Planning direction selection (`backend/workflow/planning.py`)**: `_selected_directions_from_decision()` helper extracts selected directions list from `DirectionSelectionDecision` dict. PI direction selection stores full decision in `analysis_state.exploration_report.direction_selection_summary`.
  - **Frontend evidence-gated direction cards (`frontend/src/components/ExplorationReportDialog.tsx`)**: Direction cards display scored fields (overall_priority, data_reliability, literature_novelty, biology_expectedness), evidence-sourced species badge, merged PMID list with expandable detail, complete evidence details including DOI links, negative validation axes status, prerequisite dependencies. Vitest test file. Dependencies: `package.json`/`package-lock.json` updated.
  - **24 new test files** (~1450 total lines): `test_pi_direction_decision.py` (selection decisions, rejected directions, empty fallback), `test_pi_prerequisite_closure.py` (single-tier, multi-hop, cycle rejection), `test_pi_selection_hardening.py` (dependency resolution, closure validation), `test_pi_fallback_candidate.py` (empty candidate fallback), `test_pi_score_field_contract.py` (score field completeness, fallback values), `test_pi_cycle_propagation.py` (selection cycle with propagation, multi-hop closure), `test_direction_evidence_models.py` (all new model serialization/validation), `test_direction_evidence_gate.py` (end-to-end evidence gating, skip/fallback), `test_direction_literature_triage.py` (multi-axis query builders, negative validation, species/tissue qualification), `test_literature_cache_match.py` (cache article matching, term overlap), `test_literature_dedup.py` (PMID dedup, merge rules), `test_literature_escalation.py` (degraded query escalation, missing precedent escalation), `test_literature_novelty_gating.py` (high novelty gating, low confidence block), `test_literature_negative_axes.py` (axis evaluation, confidence scoring), `test_literature_cross_species.py` (cross-species query derivation), `test_synthesis_phase_guard.py` (exploration phase retention on failure), `test_variants_candidate_table.py` (rendered candidate fields, badge styling), `test_variants_evidence_ref.py` (evidence_ref linking), `test_exploration_report_direction_evidence_html.py` (HTML rendering of evidence section). 3 frontend tests: `ExplorationReportDialog.test.tsx` (Vitest, JSX rendering).

- `0.29.1` — Rare cell type focused literature search: four-bucket triage system, focused PubMed query pipeline, exhaustion reason classification (2026-06-24):
  - **Four-bucket triage in `PIAgent.evaluate_annotation_search_results()`**: Clusters now classified into `covered`, `rare_unresolved`, `suspected_contamination`, and `context_dependent_state` buckets. Hallucinated cluster IDs filtered against ground-truth markers; cross-bucket duplicates resolved with priority rules (covered > context_dependent_state > suspected_contamination > rare_unresolved). `triage_complete`/`triage_low_confidence` guards enable semantic early termination. `state` parameter carries experimental context (initial instruction, experimental design summary) into the triage prompt.
  - **Focused query builder in `LiteratureAgent`**: Pure-function `build_focused_query()` generates two-tier PubMed queries — tier 1 scoped to tissue + candidate/markers, tier 2 drops tissue (rare) or adds tissue-specific plausibility terms (contamination, e.g. `"portal tract"[tiab]` for liver). New helpers: `_gene_case_variants()`, `_sanitize_synonyms()`, `status_from_judge()`, `plausibility_terms_for_tissue()` (tissue-parameterized vocabulary with explicit override support).
  - **Focused relevance judge (`_judge_focused_relevance()`)**: LLM call scoring articles along identity/marker/tissue/method dimensions, returning strong/limited/exhausted/contradictory via deterministic `status_from_judge()` mapping.
  - **Search orchestration (`_focused_rare_search()`)**: Rare clusters run tier 1 → escalate to tier 2 if insufficient; contamination clusters run both tiers in parallel, taking the higher result. Distinct `exhaustion_reason` values per failure mode: `search_failed`, `judge_failed`, `no_pubmed_results`, `no_relevant_evidence`, `empty_query`. `_focused_result_score()` prioritizes runtime failures over genuine absence on ties.
  - **Multi-round pipeline integration (`_multi_round_pubmed()`)**: `triage_complete=True && triage_low_confidence=False` triggers semantic termination + focused search + single synthesis (skips standard multi-round loop). Low confidence keeps standard rounds. Focused articles prioritized at synthesis input (first-position, 25-main + 5-focused budget). `context_dependent_state` buckets skip focused search.
  - **Annotation prompt enrichment (`annotate_clusters_from_literature()`)**: Injects per-cluster focused evidence summaries (with article abstracts), differentiates contradictory evidence by bucket (rare → alternative identity, contamination → contamination), classifies exhaustion by reason (runtime failure → "do not interpret as absence", no PubMed → "evidence scarcity", no relevant evidence → article titles).
  - **Review enhancement (`review_cluster_annotations()`)**: Injects triage context (suspected contamination reason, candidate cell type) and biological principles (single-nucleus guidance, marker set coherence, negative markers). Deterministic post-processing: forces low confidence + reason passthrough + candidate into `alternative_cell_types` for contamination buckets.
  - **Workflow wiring (`annotation.py`)**: `state` passed to `evaluate_annotation_search_results()`; `triage` + `focused_evidence` threaded through to `annotate_clusters_from_literature()` and both `review_cluster_annotations()` call sites.
  - **11 new test files** (1077 total lines): `test_focused_query_builder.py`, `test_focused_status.py`, `test_focused_rare_search.py`, `test_multi_round_triage.py`, `test_evaluate_four_bucket.py`, `test_match_prompt_hints.py`, `test_review_four_class.py`, `test_rare_celltype_batch1_replay.py`, `test_annotation_threading.py`, plus updated `test_bioinfo_text_only_exit.py` and `test_supervisor_stuck_escalation.py` imports. Covers query building, relevance judge, rare/contamination search orchestration, semantic termination, bucket dedup, hallucination filtering, prompt hint formatting, review post-processing, and full pipeline replay.
  - **Code quality**: `_focused_result_score()` ensures runtime failures surface over genuine exhaustion on ties. Import cleanup (`remove duplicate import re`).
  - **Offline ORA marker database priors** (`backend/services/ora_kernel.py`, `marker_priors.py`): Self-contained `compute_ora_priors()` performs hypergeometric enrichment + BH-FDR in the sandbox kernel. `MarkerDB` class loads PanglaoDB and CellMarker2 TSV with species normalization, column projection, and dedup. `_compute_database_priors()` in `BioinformaticsAgent` executes ORA against scRNA-seq marker genes and passes results to annotation callbacks. `annotation_database_priors_summary` on `AnalysisStep` for report logging. `_format_database_priors_block()` formats priors into the PI annotation prompt as supportive evidence alongside literature.
  - **Marker DB fetch HITL gate** (`backend/constants.py::WaitInputReason.MARKER_DB_MISSING`): When `backend/data/markers/` lacks TSV files, `_wait_for_marker_db_intervention()` in `workflow/core.py` raises a marker-db-missing dialog through the `LlmTimeoutDialog` channel. `_marker_db_intervention_callback()` triggers `trigger_fetch()` (async download via `asyncio.to_thread`), skip (literature-only annotation), or terminate. Frontend `LlmTimeoutDialog` shows Fetch/Skip/Terminate with Chinese/English i18n. WebSocket reconnect resends `marker_db_missing` state sync (`_resend_wait_input_state_sync`). Stale marker DB failure cleared on session load (`ensure_session_loaded`). `resolve_hitl_skip_target` and `_llm_failure_action_locked` in `lifecycle.py` extended for `MARKER_DB_MISSING`.
  - **`scripts/fetch_markers.py`**: Atomic dual-source fetch of PanglaoDB (gzip TSV) and CellMarker 2.0 (xlsx) via `pd.read_csv`/`pd.read_excel`. Projects both to a fixed 7-column schema with species normalization. File-lock concurrency guard (`O_CREAT|O_EXCL`) with 30-minute stale-lock reclamation. Atomically written via `os.replace` to prevent partial-read races. `CLI: python scripts/fetch_markers.py`; backend entry: `fetch_and_write(dest_dir)`.
  - **9 new test files** (additional): `test_marker_priors.py`, `test_ora_kernel.py`, `test_marker_db_missing_hitl.py`, `test_marker_db_ws_restart.py`, `test_marker_startup_script.py`, `test_fetch_markers.py`, `test_db_priors_injection.py`, `test_db_priors_prompt.py`, `test_chat_annotate_callback_sig.py`.

**0.28.x — (2026-06-20 → 2026-06-24)**

- `0.28.8.1` — Anthropic `/v1` bare suffix handling, protocol probe startup delay (2026-06-24):
  - **`_anthropic_sdk_base_url` now strips bare `/v1`**: Added `"/v1"` to the suffix-strip tuple so base URLs like `https://api.anthropic.com/v1` (without `/messages`) are normalized correctly — the SDK's own `/v1/messages` append is no longer doubled.
  - **Protocol probe startup delay**: Added `await asyncio.sleep(1.0)` before the protocol probe loop in `_dispatch_non_streaming()` to prevent race conditions when concurrent requests trigger simultaneous probes with the same `(base_url, model)` lock key.

- `0.28.8` — Snapshot parent-chain fix: forward (non-restore) auto-snapshots now chain instead of all becoming isolated roots (2026-06-23):
  - **Problem**: On a normal forward run with no manual restores, every auto-snapshot got `parent_id=None` / `branch_label="main"`, producing a flat list of isolated roots (observed: 26 snapshots, zero parent links). Root cause: `compute_snapshot_context()` sourced `parent_id` only from `state.last_restored_snapshot_id`, which is set only immediately after a restore (`snapshots.py`) and cleared after a single snapshot (`create_and_register_snapshot`). Outside a restore it is always `None`, so forward snapshots never linked. This was the over-correction of the P2-4 fix (which correctly removed the `snapshots[-1]` time-based fallback but removed forward chaining with it).
  - **New lineage head pointer `AnalysisState.last_snapshot_id`**: Advances to the new snapshot id on every create, and is set to the restore target on restore. `compute_snapshot_context()` now sets `parent_id = last_restored_snapshot_id or last_snapshot_id` — a fresh restore still takes precedence (anchors the new diverging branch on the restored node), but normal forward runs chain onto the head. Does **not** reintroduce `snapshots[-1]`, so the P2-4 regression test still passes.
  - **`branch_label` continuation**: Forward snapshots now inherit the parent snapshot's `branch_label` instead of reverting to `"main"`, so a branch keeps its label as it extends (only a fresh restore allocates a new `branch-X`).
  - **Head updates**: `create_and_register_snapshot` sets `state.last_snapshot_id = snapshot.id` after append; snapshot restore route (`snapshots.py`) sets `restored_state.last_snapshot_id = snapshot.id`. Both production creation sites (`workflow/core.py`, `routes/chat.py`) go through `create_and_register_snapshot`, so head advancement is centralized.
  - **API exclusion**: `last_snapshot_id` is internal lineage state (like `last_restored_snapshot_id`) — added to `_EXCLUDED_FIELDS` in `test_to_api_dict_coverage.py`.
  - **New tests** (`test_snapshot_forward_chain.py`): first forward snapshot is a root; second chains to first; a long forward run forms a single linear chain with exactly one root; restore-then-forward chains onto the restore point and stays on the diverging branch.
  - **Note**: Pre-existing flat snapshots cannot be auto-relinked (they lack ordering parent info); the fix only applies to snapshots created after the upgrade.

- `0.28.7` — Species HITL WebSocket reconnect fix, test stub hardening across 5 test files (2026-06-23):
  - **Species HITL resend on WebSocket reconnect**: `_resend_wait_input_state_sync` extended to cover `species_input` and `gene_species_mismatch` reasons (previously only `llm_failure`/`agent_disabled`); `species_input_info` now included in the state-sync payload so the species dialog reopens after reconnect.
  - **Test stub hardening**: Added `_provider`/`_last_config_revision`/`agent_type` stubs to `PathologyAgent._bare_agent` and `BioinformaticsAgent` test shell in `test_bioinfo_text_only_exit.py`. Bound real `ensure_agent_ready` in hot-reload test stand-in. Added `ensure_agent_ready` mock to `test_snapshot_rollback_idempotency.py`. Resolved file paths relative to test file in `test_tpl_noop_install.py`.

- `0.28.6` — Literature search tuning: PubMed max rounds 20→50 (default 5→10), Deep Research LLM default 1 call (max 5→2) with prompt inversion (2026-06-23):
  - **PubMed max_search_rounds upper bound 20→50, default 5→10**: All backend layers updated — Pydantic validators in `config_schema.py` (export/import) and `api_requests.py` (runtime API), `LiteratureAgent` constructor/clamp/hot-reload, `AnalysisState` default dict, `workflow/core.py` instantiation, `config_io.py` export/import fallbacks, `chat.py` delegation. Frontend store default and `PromptsPanel` input synced (clamp `20→50`, default `10`). Upper-bound guard test added (50 passes, 51 rejects). Default-alignment tests updated in `test_config_schema.py`, `test_config_io.py`, `test_settings_provider.py`.
  - **Deep Research LLM decomposition capped at 2 tasks (was 5), prompt inverted to favor single merged query**: `MAX_DR_TASKS` in `PiAgent.plan_deep_research_tasks()` reduced from 5 to 2. Prompt rewritten to default to single merged query (`decompose=false`); split only when genuinely necessary, producing exactly 2 tasks. Docstring updated. Tests aligned in `test_plan_deep_research_tasks.py` and `test_literature_dr_decompose.py`.
  - **New doc files**: Implementation plan `docs/superpowers/plans/2026-06-23-literature-search-tuning.md` and design spec `docs/superpowers/specs/2026-06-23-literature-search-tuning-design.md`.

- `0.28.5` — Full cluster marker pass-through in PI agent, inline markdown link extraction in literature citations, citation footer made format-agnostic (2026-06-23):
  - **`_summarize_cluster_markers()` changed from 3-tuple to 2-tuple**: Removed `gene_sample` (sampled `[:10]` genes per cluster capped at 20 total). The `gene_sample` line was too sparse to be useful and consumed context. Callers now receive only `(n_clusters, markers_summary)`.
  - **Removed `[:15]` cluster truncation in PI prompts**: `generate_annotation_research_brief()`, `evaluate_annotation_search_results()`, and `build_search_direction_brief()` now pass ALL clusters instead of capping at 15. Each cluster still limited to 8 genes per line. This ensures the LLM sees the full cluster landscape for annotation and direction decisions.
  - **`_DEEP_RESEARCH_CITATION_FOOTER` made format-agnostic**: Changed from rigid `"[N] Author et al. (Year). Title. Journal. URL_or_DOI"` mandate to guidance supporting both inline markdown links `[text](URL)` and numbered reference lists. The parser already accepts both formats.
  - **`_extract_references_from_text()` now extracts inline markdown links**: Added a second regex for `[text](URL)` patterns (with `(?<!!)` lookbehind to exclude image markdown `![alt](url)`). Structured extraction order: (1) end-of-text numbered list `[N] Title... URL`, (2) inline markdown links, (3) bare URL fallback (only when structured < 3 refs). Added `_norm()` to strip `#:~:text=...` PMC text-fragment suffixes before dedup.
  - **Gene sample cap raised in LiteratureAgent**: `get_exploration_findings()` gene_sample cap bumped from 20 to 40 (fallback topic only).
  - **Supervisor test fix**: `_mk_supervisor` in `test_supervisor_stuck_escalation.py` now mocks `_provider` and `_last_config_revision` to prevent TypeError.
  - **`analysis_state_api_dict` test**: Added `"literature_config"` to expected API dict fields.
  - **Route snapshot updated**: Added `/api/admin/users/{uid}/role` PATCH and `/api/session/{session_id}/species-input` POST.
  - **New tests**: `test_evaluate_annotation_search.py` — evaluator sees all 22 clusters; `test_summarize_cluster_markers.py` — 2-tuple return, all clusters beyond 15 included, genes capped at 8 per cluster. `test_pi_research_briefs.py` — annotation brief includes all clusters, no `gene_sample` line. `test_literature_citations.py` — inline markdown link extraction, image markdown exclusion, PMC fragment stripping, mixed numbered+inline.

- `0.28.4.1` — Deep Research background polling: Responses `background=True` submission + retrieve polling to avoid gateway idle timeouts, `_NO_BACKGROUND_ENDPOINT` per-endpoint cache for `background`-rejecting providers (2026-06-22):
  - **Problem**: Deep research models run for several minutes. A blocking `responses.create()` keeps the connection idle long enough for upstream gateways (e.g. Cloudflare in front of `api.n1n.ai`) to sever it with HTTP 524 while the job keeps running server-side. Submitting with `background=True` returns immediately, then we poll `responses.retrieve()` with short requests that never trip the gateway's idle timeout.
  - **`_ResponsesAdapter.__init__`**: Now accepts `base_url` parameter, passed by `OpenAICompatibleClient._get_responses_adapter()`, to enable per-endpoint background support caching.
  - **`_build_params()` background logic**: Accepts new `background_ok` parameter. When `background_ok=True`, `stream=False`, model is deep-research, and `base_url not in _NO_BACKGROUND_ENDPOINT`, adds `"background": True` to params. Deep research calls with tools or vision go through `complete_with_tools`/`complete_vision` which do not pass `background_ok` (they use their own `_build_params` without it), so background is only used for the text‑only `complete()` path.
  - **`_await_completion()`**: Polls `responses.retrieve(resp_id)` at `_DR_BACKGROUND_POLL_INTERVAL` (5s) until status `"completed"` or a terminal failure in `_DR_BACKGROUND_TERMINAL_FAILED`. Handles transient poll errors (counted up to `_DR_BACKGROUND_MAX_POLL_ERRORS=5` then raises). Deadline at `_DR_BACKGROUND_MAX_WAIT=1800` (30 min). Used by `complete()`, `complete_with_tools()`, and `complete_vision()`.
  - **`_create_responses_with_fallback()` background rejection fallback**: When the provider rejects `background` (`_is_unsupported_background_error`), the endpoint is added to `_NO_BACKGROUND_ENDPOINT`, `background` is stripped from params, and the call is retried synchronously — the fallback is cached so every subsequent call avoids the failed round‑trip.
  - **`_is_unsupported_background_error()`**: String‑based detection (like `_is_unknown_web_search_tool_error`) requiring the word `"background"` AND a rejection keyword (`"unknown parameter"`, `"unsupported"`, `"not supported"`, `"unexpected"`, `"not allowed"`, `"invalid parameter"`), so prose mentioning background in other contexts does not match.
  - **Constants**: `_DR_BACKGROUND_POLL_INTERVAL=5.0`, `_DR_BACKGROUND_MAX_WAIT=1800.0`, `_DR_BACKGROUND_MAX_POLL_ERRORS=5`, `_DR_BACKGROUND_TERMINAL_FAILED="failed","cancelled","canceled","incomplete","expired"`, `_NO_BACKGROUND_ENDPOINT=set()`.
  - **Tests** (20 new in `test_responses_adapter.py`): `test_build_params_adds_background_for_deep_research`, `test_build_params_no_background_for_normal_model`, `test_build_params_no_background_when_streaming_deep_research`, `test_build_params_no_background_without_background_ok`, `test_build_params_skips_background_when_endpoint_cached`, `test_complete_with_tools_deep_research_no_background`, `test_complete_vision_deep_research_no_background`, `test_await_completion_returns_immediately_when_completed`, `test_await_completion_polls_until_completed`, `test_await_completion_raises_on_failed`, `test_await_completion_tolerates_transient_poll_errors`, `test_await_completion_raises_after_repeated_poll_errors`, `test_complete_deep_research_uses_background_polling`, `test_background_polling_constants_exist`, `test_is_unsupported_background_error_detects`, `test_is_unsupported_background_error_ignores_unrelated`, `test_no_background_endpoint_cache_exists`, `test_create_fallback_strips_background_and_retries`, `test_create_fallback_does_not_strip_on_unrelated_error`.

- `0.28.4` — `is_transient_probe_error` / `retry_with_backoff` blacklist retry model: unknown/unexpected errors retried instead of immediately propagated, protocol/parameter adaptation signals bypass retry, hang-safety fix in LLM failure test helper (2026-06-22):
  - **`is_transient_probe_error()` rewritten from whitelist to blacklist model**: Previously only specific exception types (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError, httpx timeouts, asyncio.TimeoutError) were transient; everything else was non-transient. Now only `UpstreamExhaustedError` and `_is_protocol_adaptation_signal()` return `False` (non-retryable). Everything else — including unknown/unexpected error types and error codes — is treated as potentially transient and retried. Upstream LLM providers return error codes inconsistently, so a whitelist is unreliable; retry exhaustion always produces `UpstreamExhaustedError` → workflow HITL gate.
  - **`_is_protocol_adaptation_signal()` added**: Groups all protocol/parameter adaptation signals that must bypass generic retry: `is_responses_endpoint_absent`, `is_responses_chat_only_model`, `is_responses_only_model`, `is_temperature_error`, `is_tool_choice_error`. These have dedicated upstream handlers (protocol redirect, temperature fallback, tool_choice fallback) and must never be caught by `retry_with_backoff`.
  - **`retry_with_backoff()` changed from `except RETRYABLE_EXCEPTIONS` to `except Exception`**: Broad catch with explicit filters. `UpstreamExhaustedError` (nested exhaustion) propagates as-is. `_is_protocol_adaptation_signal()` propagates to dedicated handlers. All other errors are retried; on exhaustion they surface as `UpstreamExhaustedError` → user-facing HITL gate. Comment updated to reflect the "broad except with adaptation signal gating" model.
  - **Test rewrites**: `test_retry_classification.py` completely rewritten — removed `_is_hard_quota_error` / auth/hard-quota tests, replaced with `test_service_unavailable_400_is_retried` (generic 400 retried 3×), `test_temperature_400_not_retried_propagates` (temperature signal propagates immediately), `test_tool_choice_400_not_retried_propagates` (tool_choice signal propagates immediately).
  - **Test classification changes**: `test_transient_probe_classifier.py` — `test_json_decode_error_is_not_transient` → `test_json_decode_error_is_transient`, `test_plain_bad_request_is_not_transient` → `test_plain_bad_request_is_transient`. Added `test_temperature_error_is_not_transient` and `test_tool_choice_error_is_not_transient`.
  - **Test divergence fixed**: `test_tool_choice_fallback.py` — non-tool-choice BadRequestError now surfaces as `UpstreamExhaustedError` (retried then exhausted) instead of raw `BadRequestError`. `test_client_protocol_dispatch.py` — JSONDecodeError now counts as 3 probe retries with sleep.
  - **Hang-safety fix in `test_llm_failure_workflow.py`**: `_signal_action_once` helper now has bounded `timeout=5.0` parameter (was: infinite polling). Prevents suite-wide hang when the code under test diverts before creating the LLM-failure Event. Added `has_resumable_checkpoint = AsyncMock(return_value=False)` mock to `test_bioinformatics_node_raises_upstream_exhausted` to prevent flow diversion before `UpstreamExhaustedError` path.

- `0.28.3` — Protocol probe rewrite: multi-protocol ordered probing via `_route_protocol_order`, per-protocol transient retry with PROBE_RETRY_DELAYS, `is_transient_probe_error` classifier, fixed confirmed-responses no-retry bug, keyword-based probe order hints, anthropic third-in-order support (2026-06-21):
  - **`_route_protocol_order()` replaces single-protocol routing**: Returns an ordered list of protocols to probe (`["responses", "chat", "anthropic"]` by default) instead of a single decided protocol. `_route_protocol()` retained as head-of-list wrapper. Probe order built from: (0) explicit URL path suffix → single hard-pinned, (1) cached probe decision → single decided, (2) deep-research model → `["responses", "chat"]`, (3) base order reordered by keyword hints (`anthropic`/`responses`/`chat` substrings in base_url) + `/v1`-absent floats anthropic front + known no-responses sites drop `responses`.
  - **`_keyword_protocol_hints()`**: Soft substring-match hints from base_url that reorder the probe sequence without hard-pinning. Covers `api.anthropic.com` URLs missed by the no-`/v1` heuristic.
  - **`PROBE_RETRY_DELAYS`** `(2.0, 15.0, 10.0)`: Per-protocol transient-retry waits in seconds before switching to the next protocol in the probe order. Each protocol gets 3 single-shot attempts with these delays on transient errors (429, timeout, 5xx), then the probe moves on.
  - **`is_transient_probe_error()` classifier**: Distinguishes transient upstream blips (retry same protocol) from deterministic "wrong protocol / endpoint" signals (switch immediately). Deterministic: endpoint-absent, chat-only-model, responses-only-model. Transient: APITimeoutError, APIConnectionError, InternalServerError, RateLimitError, httpx connect/read timeout, asyncio.TimeoutError. Deterministic errors take precedence.
  - **`_probe_in_order()`**: New probe loop — tries protocols from the order list, for each protocol does up to 3 transient-retry attempts with PROBE_RETRY_DELAYS sleep, then switches. On deterministic wrong-protocol error, switches immediately (may inject redirect protocol at front). Only a SUCCESS writes `_PROBE_CACHE`. Per-(base_url, model) lock ensures concurrent first-call probes once. Exhaustion raises `UpstreamExhaustedError` with all tried protocols.
  - **`_call_decided_protocol()`**: For cached/decided protocols — full retry/backoff on transient errors (old behavior). On deterministic wrong-protocol error (cache wrong), redirects once to the correct protocol with full retry. Fixes the confirmed-responses no-retry bug where a cached `"responses"` decision would not retry transient 5xx before the old `_responses_or_fallback`.
  - **`_dispatch_non_streaming()` rewritten**: Uses `call_for()` closure that generates protocol-specific factories. In probe mode: `max_retries=0` + `non_retryable_predicate=lambda e: True` (fail fast, probe loop owns retry). In decided mode: full retry + `non_retryable_predicate` excludes transient errors (so retry_with_backoff works normally). Removed `reactive_log_message` parameter (defunct).
  - **Deleted**: `_responses_or_fallback()`, `_responses_probe_decision()`, `_record_responses_probe_failure()`, `PROTOCOL_PROBE_ROUNDS` — all replaced by the new probe infrastructure.
  - **Bug fix in `_build_retry_kwargs()`**: `max_retries_override` was only applied inside the `if request_timeout is not None` block — extracted to unconditional check. Adapter `_get_responses_adapter()` / `_get_anthropic_adapter()` changed from `is None` to `getattr(self, ...) is None` for property-init safety.
  - **Test updates**: Existing probe tests updated for new behavior — `test_protocol_probe_persistent_500_does_not_cache_protocol` now includes anthropic mock and 3 calls per protocol. New tests: `test_probe_responses_transient_retries_3x_then_switches_to_chat` (verifies PROBE_RETRY_DELAYS sleep sequence), `test_probe_responses_recovers_on_second_attempt_stays_responses`, `test_probe_endpoint_absent_switches_immediately_no_retry`, `test_probe_anthropic_third_in_order_succeeds_caches_anthropic`, `test_cached_responses_uses_full_retry_on_transient`. Routing tests: 9 new `_route_protocol_order` tests covering order defaults, anthropic keyword, /v1 absence, `/messages` suffix hard-pin, cached decision, DR model, no-responses endpoint drop. New `test_transient_probe_classifier.py` (61 lines) with 10 test cases.

- `0.28.2.1` — Deep Research web_search tool injection, Responses adapter tools passthrough, tool name fallback retry (2026-06-20):
  - **Web search tool injection**: `LiteratureAgent._deep_research_via_chat()` now passes `tools=[{"type": "web_search_preview"}]` to the DR-LLM chat call — enables OpenAI Responses API's built-in web search capability for deep-research models (o4-mini-deep-research, o3-deep-research, etc). The model retrieves live web content during generation instead of relying solely on training data.
  - **Responses adapter tools passthrough**: `_ResponsesAdapter.complete()` now accepts a `tools` parameter and forwards it through `_build_params()` to `client.responses.create()`. `OpenAICompatibleClient.chat()` also now accepts `tools` and passes them to the Responses adapter call, so plain-text chat requests can request web search via the Responses protocol.
  - **Tool name fallback retry**: `_ResponsesAdapter._create_responses_with_fallback()` already had retry logic for the built-in `web_search` tool when `web_search_preview` was unknown. New test `test_responses_adapter_complete_retries_explicit_web_search_tool_name()` confirms the fallback chain works when explicit tools are passed outside the auto-injection path.
  - **3 test additions** (~50 lines): `test_literature_dr_system_prompt.py` + `test_deep_research_via_chat_requests_web_search_tool()`, `test_protocol_crosscutting.py` + `test_chat_responses_path_passes_explicit_tools_to_adapter()`, `test_responses_adapter.py` + 2 tests (tool forwarding + fallback retry).

- `0.28.2` — Hot-reloadable agent config: runtime config revision tracking, base agent refresh hook, literature agent live reconfiguration, workflow ensure_agent_ready, session-config route invalidation, frontend literature config sync, secret masking (2026-06-20):
  - **Settings provider config revision**: `AgentSettingsProvider.get_config_signature()` / `get_config_revision()` produce a deterministic tuple and SHA-256-based integer revision for each agent's effective config (model, temperature, system_prompt, vision, server, literature_config). Secrets are hashed in signatures via `_secret_safe_signature()`. `clear_agent_client_cache(agent_name)` drops per-agent OpenAI clients. `mask_literature_config()` / `resolve_literature_config_update()` handle the `"********"` masked-secret pattern for round-trip safe API responses.
  - **Base agent refresh hook**: `BaseAgent.refresh_runtime_config_if_needed()` compares `_last_config_revision` with the current provider revision; on mismatch, clears the provider's per-agent client cache and calls `on_runtime_config_changed()` (optional subclass hook). Called at the top of `chat()`, `chat_without_history()`, `chat_stream()` so every LLM call sees fresh config.
  - **Literature agent live reconfiguration**: `on_runtime_config_changed()` → `_sync_literature_config_from_provider()` syncs `mode`, `deep_research_model`, `deep_research_llm_base_url`, `deep_research_api_key`, `google_api_key`, `pubmed_email`, `max_search_rounds` from the provider. Rebuilds the deep-research `OpenAICompatibleClient` or `GoogleDeepResearchClient` on signature change, closing stale clients asynchronously. Clears `_annotation_cache`. Dedicated `close_runtime_clients()` for cleanup. Also calls `refresh_runtime_config_if_needed()` at entry to `process()`, `_deep_research_via_chat()`, `_literature_search_for_cell_type()`, `annotate_clusters()`, `validate_pathway()`, `search_mechanism()`.
  - **Workflow `ensure_agent_ready()`**: Extracted `_create_optional_agent()` factory and `ensure_agent_ready()` — materializes or destroys optional agents (pathology, biochemistry, painter, supervisor, technique) based on model config, calling `refresh_runtime_config_if_needed()` when the agent exists. All `initialize()` agent-creation blocks replaced with `ensure_agent_ready()` calls. `_install_retry_notifiers()` moved before agent creation so eager external clients (literature DR client) inherit retry callbacks. Cleanup: literature runtime clients closed before provider cleanup.
  - **Session-config route invalidation**: `_invalidate_active_agent_config()` notifies the active workflow's provider and agent when prompts/models/temperatures/vision/server/literature config are updated via API. All mutation endpoints (`/prompts`, `/models`, `/temperatures`, `/vision-settings`, `/max-rounds`, `/round-wait`, `/literature-config`, `/agent-server`) now: validate session id, acquire `get_session_lock`, use `update_session_cache`, and call `_invalidate_active_agent_config()`. Literature config uses `AgentSettingsProvider.resolve_literature_config_update()` for masked-secret round-trip.
  - **`AnalysisState` literature config exposure**: `literature_config` added to `_API_KEY_MAP` as a read-only frontend field. `to_api_dict()` masks secrets via `AgentSettingsProvider.mask_literature_config()` before serialization.
  - **Frontend literature config sync**: New `frontend/src/utils/literatureConfig.ts` — `buildLiteratureConfigPayload()`, `resolveDeepResearchServerConfig()`, `isMaskedSecret()`, `MASKED_SECRET_VALUE`. `useApi.ts`: `applyStoreUpdates()` handles all store-update fields (including literature config) in one function replacing duplicated inline blocks. Session-start sync uses `buildLiteratureConfigPayload()`. `transformLoadResponse.ts` parses `literature_config` into `StoreUpdates` with proper snake_case→camelCase mapping. `types.ts` and `appStore.ts` updated for literature config store fields. `setSecretFromLoad()` preserves masked secrets on load by comparing current store value.
  - **Other workflow wiring**: `callbacks.py` — literature WS callback calls `refresh_runtime_config_if_needed()`. `_check_supervisor_agent` calls `ensure_agent_ready("supervisor")`. `messaging.py` — `_supervisor_callback` uses `ensure_agent_ready("supervisor")` for lazy materialization.
  - **10+ new/updated test files** (~990 lines): `test_settings_provider_config_revision.py`, `test_session_config_routes.py`, `test_session_api_dict.py`, `test_agent_entrypoint_refresh.py`, `test_workflow_optional_agent_hot_reload.py`, `test_workflow_retry_notifier_init.py`, `test_literature_agent_runtime_config.py`, `test_literature_config_hot_reload.py`, `test_literature_config_path_param.py` (implied), updated `test_settings_provider.py`, `test_to_api_dict_coverage.py`, `test_plan_deep_research_tasks.py`, `test_research_brief_threading.py`. Frontend: `literatureConfig.test.ts` (128 lines), `useApi.test.ts` (46 lines), `transformLoadResponse.test.ts` (59 lines).
  - **New config files**: `AGENTS.md` (685 lines) and `CLAUDE.md` (286 lines) for OpenCode/Claude agent contexts.

- `0.28.1.1` — Protocol probe rewrite: alternating Responses/Chat with PROTOCOL_PROBE_ROUNDS, non-retryable predicate, Gemini/n1n.ai Responses 500 detection (2026-06-20):
  - **Protocol probe rewrite**: `_responses_or_fallback()` now alternates between Responses and chat up to `PROTOCOL_PROBE_ROUNDS=3` before raising `UpstreamExhaustedError` with a detailed diagnostic. Previously tried Responses once and fell back to chat on any error — which masked transient flakiness on both sides. The probe loop tracks `last_responses_error` and `last_chat_error` and clears `_PROBE_CACHE` on exhaustion so a future call re-probes.
  - **`is_responses_endpoint_absent()` enhancement**: Now inspects `error.body` dict for `"not implemented"` combined with `"responses"` or `"local:convert_request_failed"` — handles n1n.ai's Gemini Responses 500 error (returns `InternalServerError`, not `NotFoundError`). Previously this 500 was treated as a transient server error and retried (wasteful); now it's classified as endpoint-absent and triggers chat fallback.
  - **`retry_with_backoff()` `non_retryable_predicate` parameter**: When set and returns True for an exception, the exception is re-raised immediately (no retry, no sleep). Used by the protocol probe to fail fast from the Responses adapter and alternate to chat instead of wasting retry attempts on a server that doesn't support Responses.
  - **`_invoke_adapter()` and `_dispatch_non_streaming()`**: `non_retryable_predicate` threaded through; `make_probe_fallback_call` parameter added to `_dispatch_non_streaming()`. When the protocol is "responses", the initial adapter call passes `non_retryable_predicate=lambda _e: True` (fail fast) and `make_probe_fallback_call` uses zero-retry inline calls (`max_retries=0`), so the probe loop alternates quickly without internal retry/sleep delays.
  - **All three dispatch methods updated**: `chat()`, `chat_with_tools()`, and `chat_with_vision()` now pass `make_probe_fallback_call` with `max_retries=0` and `probe_timeout` for the chat-side probe calls. `chat_with_tools()` and `chat_with_vision_inline()` accept `max_retries` parameter to support this.
  - **11 new test cases** (~230 lines) in `test_client_protocol_dispatch.py`: Responses 500 fallback for chat/vision/tools/stream, both-protocols-500 no-cache, responses-failure-then-chat-success caches chat, probe retries responses after chat failure, probe exhausts after 3 alternating rounds, rate-limit and JSON-error alternate immediately, 404 endpoint-absent + chat-fail clears model cache.

- `0.28.1` — Multi-protocol LLM client: OpenAI Responses API + Anthropic Messages API adapters, protocol auto-routing, deep-research structured citations (2026-06-20):
  - **Protocol routing system**: `_route_protocol()` auto-selects between `chat` (OpenAI SDK `/chat/completions`), `responses` (OpenAI SDK `/responses`), and `anthropic` (Anthropic SDK `/v1/messages`) based on `base_url` and `model`. URL path suffix override (`/responses`, `/messages`, `/chat/completions`), no-`/v1` → Anthropic heuristic, deep-research model → responses, probe caching with per-(base_url, model) lock for concurrent first-call safety. Two-level cache: `_NO_RESPONSES_ENDPOINT` (base_url-level, no Responses endpoint at all) + `_PROBE_CACHE` (model-level protocol). Lazy adapter construction via `_get_responses_adapter()` / `_get_anthropic_adapter()`.
  - **OpenAI Responses API adapter**: `ResponsesLLMAdapter` — implements `complete()`, `complete_with_tools()`, `complete_vision()`, `stream()` using the OpenAI SDK's `client.responses.*` interface. Translates messages via `_responses_messages_to_input()` (system → `instructions`, rest → `input[]`), tools via `_responses_translate_tools()` (flattened function format), tool_choice via `_responses_translate_tool_choice()`. Supports web search built-in tool. `_responses_or_fallback()` pattern catches endpoint-absent / chat-only-model errors and falls back to the inline chat path with automatic cache write.
  - **Anthropic Messages API adapter**: `AnthropicLLMAdapter` — implements the same interface (`complete()`, `complete_with_tools()`, `complete_vision()`, `stream()`) using the `anthropic` SDK. Translates messages via `_anthropic_messages_translate()` (system split, tool_use/tool_result content blocks), tools via `_anthropic_translate_tools()` and `_anthropic_translate_tool_choice()`. Handles `thinking`/`redacted_thinking` content blocks for Claude extended thinking.
  - **Deep-research structured citations**: `LiteratureAgent._select_references()` — prefers structured `response.citations` from the Responses API over regex extraction from text. Both `_extract_references_from_text()` and `_select_references()` made `@staticmethod`.
  - **URL normalization**: `_openai_sdk_base_url()` / `_anthropic_sdk_base_url()` strip endpoint suffixes so SDK path append does not double-append. `_strip_path_suffix()` shared helper.
  - **Model detection**: `_is_deep_research_model()` normalizes hyphens to match both `o3-deep-research` and `o4-deepresearch-mini`.
  - **All three high-level methods** (`chat()`, `chat_with_tools()`, `chat_stream()`, `chat_with_vision()`) now auto-route through `_route_protocol()` with inline fallback: chat path first, `BadRequestError` with `is_responses_only_model()` → cache + retry via Responses adapter. `chat_with_tools()` extracted from monolithic method → `_chat_with_tools_inline()`; same for vision.
  - **Dependency**: `anthropic>=0.40.0` added to `requirements.txt` (lazy-imported, only needed for Anthropic Messages API endpoints).
  - **Follow-up fix**: `_AnthropicAdapter.complete_vision()` now passes `timeout` parameter through to `self.client.messages.create()` (was silently ignored). Test `test_anthropic_adapter_complete_vision_passes_timeout()` added.
  - **7 new test files** (~1010 lines): `test_protocol_routing.py` (215 lines — routing priority, probe cache, concurrent lock, URL suffix override, URL normalization), `test_client_protocol_dispatch.py` (231 lines — chat/stream/vision/tools dispatch for all 3 protocols), `test_anthropic_adapter.py` (176 lines — Anthropic message translation, tool format, thinking blocks, timeout pass-through), `test_responses_adapter.py` (160 lines — Responses message/tool/stream translation), `test_protocol_crosscutting.py` (171 lines — english enforcement, rate-limit retry, error mapping across protocols), `test_protocol_url_normalize.py` (38 lines), `test_literature_citations.py` (19 lines — structured citation from response object).

**0.27.x — (2026-06-17 → 2026-06-19)**

- `0.27.12` — Report figure collection deep search (scai-data-5 regression fix), figure count audit logging, prompt figure steering (2026-06-19):
  - **`resolve_figure_in_workspace()` shared helper**: New function in `backend/utils/files.py` — searches for a figure by basename across the entire workspace tree (root → `figures/` → `rglob` deep search), with `seen_stems` dedup. Returns the first match deterministically, logs a warning on multiple matches.
  - **Supervisor figure collection fix**: `_collect_figure_paths()` now uses `resolve_figure_in_workspace()` instead of checking only `workspace_path / fname` and `workspace_path / "figures" / fname`. Figures saved into `results/<dir>/figures/` subdirectories (via code that ignored the bare-filename guidance) are now found for supervisor review.
  - **Report figure scanner `_scan_existing_figures()`**: New `analysis_log` parameter — iterates all `saved_figure_filenames` from analysis steps, resolves each via `resolve_figure_in_workspace()`, copies to the report figures directory. Caught the scai-data-5 case (34 figures generated, only 2 reached the report). Integrated into `DetailedReportGenerator._scan_existing_figures()` call site.
  - **`count_saved_figures()` helper**: New function in `bioinformatics_agent.py` — counts actual disk-saved figures from `analysis_log.saved_figure_filenames`. Completion log lines in both `BioinformaticsAgent.run()` and `AgentNodesMixin._bioinformatics_analysis()` now report "X saved figures" alongside "Y inline figures", replacing the misleading `0 figures` inline-only count.
  - **Bioinformatics prompt steering**: Prompt now recommends `generate_publication_figure` for report-quality figures and warns against saving into nested `results/<dir>/figures/` paths. Both `_static_system_prompt` and kernel-initialization prompt updated.
  - **4 new test files** (215 total lines): `test_bioinformatics_figure_count_log.py`, `test_figure_prompt_guidance.py`, `test_scan_logged_figures.py` (regression: subdir figure found and copied), `test_supervisor_figure_collection.py` (regression: subdir figure + shared helper tests).

- `0.27.11` — Gemini Deep Research poll transient-failure resilience (2026-06-19):
  - **`GoogleDeepResearchClient._poll_research()`**: Wraps the HTTP GET in `try/except` for `httpx.TransportError` and `httpx.TimeoutException`. On transient network errors (disconnects, timeouts, DNS failures) during polling, logs a warning and retries after `poll_interval` instead of crashing the entire deep research call. Sustained failures still raise `TimeoutError` via the existing `max_wait` deadline.
  - **New test**: `test_poll_research_resilience.py` (57 lines) — `test_poll_recovers_after_transient_disconnects()` (2 failures then success via `httpx.RemoteProtocolError`), `test_poll_sustained_failure_times_out_not_raises_raw()` (persistent `ConnectError` → `TimeoutError`, not a raw transport exception).

- `0.27.10` — Literature Agent system prompt injection, PubMed eval year-sorted sampling, PI topic format rule, dict-repr task normalization, DR coverage single-topic gap fill (2026-06-19):
  - **Deep Research system prompt injection**: `_deep_research_via_chat()` now sends `self.system_prompt` as a `{"role": "system"}` message before the user prompt — the DR-LLM (OpenAI o4, Perplexity sonar, etc.) receives the proper role definition instead of having it mixed into the user message. `_FINDINGS_EXTRACT_MAX_CHARS = 100000` (was hardcoded 10k) so long reports aren't aggressively truncated before findings extraction.
  - **Literature prompt rewritten as system role**: `agent_prompts.literature_prompt` changed from a user-level request ("Please help me conduct literature research on {topic}") to a proper role-definition system prompt ("You are a professional biomedical literature researcher..."). Removes the unformatted `{topic}` placeholder; operating principles enforce citation rigor and output-format compliance.
  - **PubMed eval year-sorted sampling**: `_evaluate_pubmed_results()` sorts articles by year descending before sampling up to 30 (was first 15), ensuring recent rounds' articles are not dropped. Prompt now reports `len(sampled)` instead of the hardcoded "first 15".
  - **PI `_LITERATURE_TOPIC_FORMAT_RULE` constant**: Reusable rule string injected into `innovation_instruction` — instructs the PI to output `literature_topics` as plain natural-language strings (not `{"Topic": ..., "Rationale": ...}` dicts). Tested for constant existence and instruction reference.
  - **Dict-repr task normalization**: `_normalize_tasks()` now parses dict-representation strings (e.g. `"{'Topic': 'X', 'Rationale': 'Y'}"`) via `ast.literal_eval` and renders them through the same `_dict_to_task_text()` path as real dicts. Extracted `_dict_to_task_text()` from `_normalize_tasks()` for reuse.
  - **DR coverage single-topic gap fill**: Removed `len(limited_topics) > 1` guard on DR coverage evaluation — single-topic merged calls now also get PubMed gap-fill when the DR output is under-covered.
  - **Literature topic context enrichment**: When building topics from other agents' findings (no PI decomposition), prepend `initial_instruction[:120]` as context prefix so the literature search is anchored to the research objective.
  - **safe_pi_brief empty logging**: Added `logger.info` when the PI brief is empty (was silent), aiding debugging of brief-fallback paths.
  - **7 new/updated test files** (total +160 lines): `test_literature_dr_system_prompt.py` (system message injection, max chars constant), `test_literature_prompt_role.py` (no `{topic}`, role-def pattern), `test_literature_topic_format_rule.py` (constant + instruction reference), `test_pubmed_eval_sampling.py` (year-sorted top-30), `test_safe_pi_brief_empty_log.py` (logging), `test_workflow_helpers.py` (dict-repr parsing), `test_research_brief_threading.py` (system message position update).

- `0.27.9` — Multi-direction correlation trigger fix (P1-1 queue leak), innovation detailed-report segments, detailed_report_url exposure (2026-06-19):
  - **Multi-direction correlation trigger fix**: Extracted `_should_run_direction_correlation()` gating on completed innovation rounds ≥ chosen directions instead of `directions_queue` being empty. P1-1 peek-commit leaves the final direction's queue entry unpopped at the terminal round (no next-round init), so the old empty-queue criterion silently suppressed cross-direction correlation analysis for all terminal-round sessions. Same `innovation_rounds_done ≥ len(chosen)` logic applied to `max_rounds` extension guard in `_synthesis_check_convergence_condition`.
  - **Innovation detailed-report segments**: New `_should_build_innovation_segment()` — each innovation round now generates an incremental `DetailedReportSegment` (phase `innovation_round_{N}`). Keys off `round_result.phase` (not `analysis_state.analysis_phase`) to avoid a spurious `innovation_round_1` segment at the exploration round's synthesis (where `analysis_phase` is already flipped to `INNOVATION`). Deduplicates by phase for node-retry idempotency. Called from `_pi_synthesis` alongside the existing exploration-complete segment.
  - **detailed_report_url exposure**: `AnalysisState.to_api_dict()` now emits `detailed_report_url` (the `/api/session/{id}/detailed-report/html` URL) when `detailed_report_html_path` is set, matching the existing `exploration_report_url` / `annotation_report_url` pattern. New test `test_detailed_report_url_exposed.py` covers both present and absent cases.
  - **3 new test files** (143 total lines): `test_multi_direction_correlation_trigger.py` (bug replication, single-direction, no-retrigger, non-innovation guards), `test_innovation_detailed_segment.py` (new-segment, duplicate-guard, exploration-phase regression), `test_detailed_report_url_exposed.py`. Updated `test_analysis_state_api_dict.py` expected fields.

- `0.27.8` — Auth-optional exact-path fix for /api/file and /api/download, unauthenticated file-path ownership delegation, report table horizontal scroll (2026-06-19):
  - **Middleware auth-optional paths**: Changed `PROTECTED_PREFIXES` and `AUTH_OPTIONAL_PATHS` from `/api/file/` and `/api/download/` (trailing-slash match, sub-path only) to `/api/file` and `/api/download` (exact-path match). The actual Starlette routes are `/api/file` and `/api/download` with the path passed as query string, so the trailing-slash version never matched the exact route — causing 403s on `<img src>` and download links in HTML reports where browsers fetch without auth headers.
  - **Unauthenticated file-path ownership delegation**: `agent_llm.validate_file_path_ownership()` no longer rejects `user_id=None` outright. Instead, it delegates ownership to `require_session_ownership(user_id=None)` which falls back to UUID-guard (session-exists) verification — mirroring the auth-optional figure/report endpoints so embedded resources render without auth headers. Path-containment validation is unchanged for both authenticated and unauthenticated requests.
  - **Report table horizontal scroll**: Added `max-width: 100%`, `display: block`, `overflow-x: auto` to all report tables — `table` (global CSS), `.direction-table`, `.steps-table`, and `.ann-table` — preventing wide tables with many columns from overflowing the page.
  - **Test updates**: `test_auth_optional_paths.py` — new tests for `/api/file` and `/api/download` exact-path auth-optional. `test_file_path_ownership.py` — renamed `test_unauthenticated_rejected` → `test_unauthenticated_legit_path_delegates_uuid_guard` (now expects allow + UUID-guard), added `test_unauthenticated_invalid_path_still_rejected`. `test_ownership.py` — migrated from test route `/api/session/sess1/detail` to `/api/session/sess1/report/html` (a real auth-optional session suffix).

- `0.27.7` — Test resilience: missing mocks in PI planning early-return test, new model field in API dict coverage (2026-06-18):
  - **`test_pi_planning_early_return.py`**: Added `infer_species` AsyncMock return value and `_send_state_sync` AsyncMock to `_Harness` — the PI planning node calls `infer_species` unconditionally since the 0.26.x species-inference addition, but the harness never provided a mock return, causing `TypeError: 'coroutine' object is not subscriptable` when `pi_planning` runs during agent-nodes-mixin round transitions.
  - **`test_to_api_dict_coverage.py`**: Added `"literature_actual_model"` to `_EXCLUDED_FIELDS` — the `AnalysisState` model gained this field, and the coverage test tracks excluded (non-API-exported) fields explicitly.

- `0.27.6` — PI-generated Deep Research briefs, zero direct context injection (2026-06-18):
  - **PI-authored research briefs**: Two new `PIAgent` methods — `generate_annotation_research_brief()` (focused marker-driven brief for cell-type annotation) and `generate_round_research_brief()` (breadth-judging brief for regular research rounds). Both present experimental background as "reference only" — PI decides relevance instead of mechanical concatenation. Shared `_summarize_cluster_markers()` and `_brief_background_block()` helpers. `_BRIEF_RULES` footer prohibits citation-format directives (the Literature Agent appends its own parser-coupled footer).
  - **Literature Agent consumption**: `research_brief` parameter threaded through `process()` → `_do_research()` → `_multi_round_gemini()` / `_deep_research_via_chat()`. When brief is present, it replaces the old injected-context prompt body verbatim (Gemini: `query = brief`; DR-LLM: `body = brief` + citation footer). When absent (PI failure/empty), falls back to the focused topic — never reverts to the old injection behavior. `_deep_research_via_chat()` removes the hardcoded experimental-context template. `_build_research_query()` deleted. New `_DEEP_RESEARCH_CITATION_FOOTER` constant preserves the load-bearing parser contract.
  - **Workflow wiring**: `annotation.py` generates per-mode briefs for `deep_research`/`gemini_deep_research` modes and passes `pi_research_brief` to `annotate_clusters()`. `agents.py` generates per-sub-task and merged round briefs passed as `research_brief` to `literature_agent.process()`. Both use `safe_pi_brief()` from `helpers.py` — wraps brief factory with try/except + empty→None fallback.
  - **Deleted design spec**: `docs/superpowers/specs/2026-06-18-pi-generated-research-prompts-design.md` (implemented and superseded).
  - **3+1 new/updated test files**: `test_pi_research_briefs.py` (annotation/round brief content, background reference-only, whitespace), `test_research_brief_threading.py` (DR-LLM brief+footer, Gemini brief, fallback no-injection), `test_brief_wiring_helpers.py` (brief→None on empty/exception), `test_literature_dr_decompose.py` extended (brief threading through decomposed/merged calls, empty brief fallback).

- `0.27.5` — Deep Research citation URL resolution & title backfill, concurrency-safe enrichment (2026-06-18):
  - **Vertex grounding redirect resolution**: `_resolve_redirect()` reads the `Location` header from 302 responses for `grounding-api-redirect` URLs, returning the real article URL. Handles relative `Location` headers, non-redirect responses, and network errors gracefully (falls back to original URL). Time-sensitive expired redirect links degrade safely.
  - **HTML title backfill**: `_fetch_title()` fetches each source URL and extracts its `<title>` tag from HTML responses. Unescapes HTML entities (`&amp;` → `&`), collapses internal whitespace, and returns `""` on non-HTML responses, missing `<title>`, or network errors. Uses a browser-ish User-Agent to avoid trivial bot blocks.
  - **`enrich_sources()`**: Concurrently resolves redirects and backfills missing titles for all sources, preserving order so inline `[cite: N]` markers stay aligned. Concurrency capped at 5 via `asyncio.Semaphore` to avoid rate-limiting academic sites.
  - **`research()` integration**: Every research call now automatically enriches sources as the final step before returning — no caller-side changes needed.
  - **17 new tests**: `test_deep_research_enrich.py` with `httpx.MockTransport` — covers redirect resolution (302 follow, skip, missing Location, HTTP error, relative URL, expired), title extraction (HTML, attributes, newlines, non-HTML, error, missing), source enrichment (resolve+titles, preserve existing), order preservation, empty input, failure fallback, and end-to-end `research()` integration.

- `0.27.4` — Exploration-failed HITL dialog, sandbox exec lock, phase instruction refactoring, gate skip notes (2026-06-18):
  - **Exploration-failed user dialog**: When Stage 1 (QC/clustering/annotation) does not complete, the workflow sends an `exploration_failed` WebSocket message with a stable `reason` code and surfaces an actionable dialog (`ExplorationFailedDialog`) with three options: "Go to Data Panel" (review/re-upload data), "Continue" (retry Stage 1), or "Stop Analysis". Rather than silently advancing rounds toward `max_rounds`, the system stays in `EXPLORATION` phase — the user must explicitly act. `resolve_round_phase()` in `planning.py` only advances to `INNOVATION` when `exploration_report` exists.
  - **`classify_exploration_failure()`**: Pure function in `synthesis.py` maps internal bioinformatics result to stable reason codes (`"analysis_error"` or `"did_not_converge"`). The dialog displays localized reason text via new i18n keys (`explorationFailedReason_did_not_converge` / `explorationFailedReason_analysis_error`). Never leaks raw internal status strings to the frontend. Tested in `test_exploration_failure_reason.py`.
  - **`_check_gate_keyword` escape tracking**: Returns a third `escaped` boolean, True only when a mandatory gate is force-released after max rejections without the analysis actually being performed. `build_gate_skip_note()` appends a `[LIMITATION]` note to findings for each escaped gate (bulk/scRNA cross-validation, deconvolution, innovation bulk validation). `test_escaped_notes_scope.py` guards against `NameError` from nested-scope initialization.
  - **Sandbox exec lock (`_exec_lock`)**: New `asyncio.Lock` preventing concurrent `execute()` and `save_session()` coroutines from reading the same ZMQ socket simultaneously (which would trigger libzmq abort). `execute()` acquires the lock; internal methods (`_execute_unlocked`) skip it for trusted kernel-init/restore code. `save_session()` now holds the lock while writing.
  - **Phase instruction extraction**: Inline bioinformatics phase instructions moved from `agents.py` into module-level pure functions for unit-testability. Stage2 now recovers executable tasks from the approved plan rather than defaulting to generic goals.
  - **STAGE1_KEYWORDS cleanup**: Removed bulk-keyword filter tokens — Stage 1 is strictly QC/clustering/annotation; condition-level DEG belongs to Stage 2.
  - **Pathology severity score**: `severity_label_to_score()` maps textual severity to canonical 0-4 scale. Overview data enriched with `severity_score`. Agent prompt updated to mention the 0-4 scale. `_summarize_status()` returns `"error"` only when every image failed.
  - **Shared `save_graphic_abstract_image()`**: New function in `image_utils.py` decodes base64, flattens RGBA/palette modes onto white background, saves JPEG. Supervisor `persist_graphic_abstract()` delegates to this shared helper (removed inline PIL code) and reads base64 directly from `analysis_state.graphic_abstract` instead of a dict pass-through. `services/report/generator.py` also uses it. `test_save_graphic_abstract_image.py` verifies RGBA→RGB JPEG.
  - **Agent prompts cleanup**: Removed modality-branched workflow constants from `agent_prompts.py`. Added proper painter and technique agent system prompts.
  - **`with_retry` clears `failed_node`**: On successful execution, sets `result["failed_node"] = None` so stale failure state from previous retries does not persist downstream.
  - **Context mixin includes literature**: `_collect_current_round_results` and previous-round fallback now include `literature_findings` alongside biochemistry/pathology/bioinformatics findings.
  - **15 new test files**: `test_agent_prompts_nonempty`, `test_escaped_notes_scope`, `test_exploration_failure_reason`, `test_exploration_retry_guard`, `test_gate_keyword_skip_note`, `test_no_dead_modality_helpers`, `test_pathology_severity_scale`, `test_pathology_soft_failure_status`, `test_planning_context_literature`, `test_retry_clears_failed_node`, `test_sandbox_exec_lock`, `test_save_graphic_abstract_image`, `test_stage1_keyword_filter`, `test_stage2_instruction`, `test_supervisor_report_ga`.

- `0.27.3` — Pathology Agent image-level LLM-failure HITL with per-image skip, terminate propagation (2026-06-18):
  - **Image-level LLM-failure HITL**: New `PathologyAgent._call_llm_with_hitl()` wraps every LLM vision/chat call (`chat_with_vision` for overview, tile batches, and findings synthesis) with an LLM-failure dialog. On `UpstreamExhaustedError`, the user can choose RETRY (re-attempt only the failing call), SKIP (drop only the current image), or TERMINATE (abort the whole analysis).
  - **Per-image skip via `_ImageSkipRequested`**: New internal sentinel exception caught by `process()` to skip just the failing image and continue with the remainder. The skipped image is recorded in the results with a clear user-facing error message.
  - **`WorkflowTerminatedError` moved to `helpers.py`**: Moved from `core.py` to the dependency-free `helpers.py` so low-level modules like `retry.py` can catch it without circular imports. Re-exported from `core.py` for backward compatibility.
  - **`with_retry` TERMINATE handling**: `with_retry` now handles `LlmFailureAction.TERMINATE` (raises `WorkflowTerminatedError`) and catches `WorkflowTerminatedError` before the generic exception handler so it propagates cleanly without being retried as a regular error.
  - **`_wait_for_llm_intervention` returns TERMINATE**: Extended return values to include `LlmFailureAction.TERMINATE` alongside the existing `RETRY` and `SKIP`.
  - **Callback wiring**: `_pathology_llm_intervention_callback()` drives the HITL dialog via the existing `_wait_for_llm_intervention` infrastructure, with a fallback to SKIP when `_current_analysis_state` is unavailable.

- `0.27.2` — Google Deep Research response extraction rewrite, frontend round parsing fix (2026-06-18):
  - **Deep Research `visualization: "off"`**: Set `visualization` to `"off"` in `agent_config` to suppress agent-generated charts/images — only the textual report is consumed, so keeping images out keeps the response light and the parser simple.
  - **Deep Research `_extract_research_result` rewrite**: Rewrote response extraction for the current Interactions API schema (`steps[]` timeline with `model_output` steps containing `content[]` text blocks and `url_citation` annotations, `thought` steps with `summary[]` arrays). Legacy `output_text`/`outputs[]`/`output` paths retained as defensive fallbacks.
  - **Frontend `currentRound` parsing fix**: `useApi.ts` WebSocket handler now only matches `"Starting round X"` for `currentRound` updates, preventing the planning dialogue's `"Planning dialogue: round 3/5"` from incorrectly overwriting the analysis round counter.
  - **Version corrections**: HANDOVER.md, user-guide.md, and user-guide-zh.md retroactively corrected from `0.28.1` to `0.27.1` for the previous release.

- `0.27.1` — Admin management, last-admin guard, error_parsing hyphenated skip, frontend refinements (2026-06-17):
  - **Admin custom username**: Setup flow requires a custom admin username instead of hardcoded `"admin"`. `create_admin_if_missing()` accepts a required `username` parameter; logs a warning if it differs from an existing admin's username. Setup page gets a username input with validation (3-64 chars, alphanumeric/underscore/hyphen). Tests updated.
  - **Admin role management**: New `PATCH /admin/users/{uid}/role` endpoint for promoting/demoting user roles. Role dropdown on admin page (`EnumCell` reusable component) for non-self users. Promoting a pending user to admin auto-activates them. Cannot demote the last remaining admin (409 Conflict via `count_admins()`).
  - **Admin delete guard**: Admins can now delete other admin users (previously forbidden), but deleting the last remaining admin returns 409 Conflict.
  - **Admin status guard**: Admin users cannot be set to `pending` status; admin status dropdown excludes `pending` option.
  - **Self user row**: Role/status/delete cells show read-only display for the current user's own row.
  - **Login routing fix**: Admin users no longer auto-redirect to `/admin` on login — go to main app root instead.
  - **Admin sidebar link**: "User Admin" (用户管理) button in sidebar, visible only to admin users, navigates to `/admin`.
  - **Back button on Admin page**: Added "Back" (返回) button to navigate back to main app.
  - **Version centralized**: `VERSION` constant extracted to `backend/constants.py`; `main.py` and `root.py` import from it. AboutDialog fetches version dynamically from backend `/api` endpoint. Version fetch failure logged via `devWarn` instead of silent catch.
  - **Swallowed-error hyphenated skip detection**: `_SKIP_DECISION_RE` regex extended to match `skip-velocity` (hyphenated) in addition to `skip_velocity` and `skip velocity`. Two new tests in `test_error_parsing.py`.
  - **Minor fixes**: `test_literature_dr_decompose.py` missing trailing newline added.
  - **ProtectedRoute admin redirect removed**: Admin users were previously
    force-redirected to `/admin` by `ProtectedRoute` for every non-login
    page. Removed this guard so admins can use the normal analysis app; they
    reach the admin panel via the sidebar "User Admin" link. Complements the
    login routing fix above.

**0.26.x — Species identification HITL & gene validation (2026-06-11 → 2026-06-17)**

- `0.26.9` — Gemini Deep Research decomposition short-circuit (2026-06-17):
  - **Workflow short-circuit**: When `literature_mode == "gemini_deep_research"`, `_literature_research` skips the PI agent's `plan_deep_research_tasks()` call and directly merges all topics into a single comprehensive Deep Research query. Gemini Deep Research self-decomposes internally, so the extra PI call is unnecessary and could over-split topics.
  - **Test**: `test_literature_dr_decompose.py` — `test_gemini_mode_short_circuits_to_merged` verifies the PI decomposition is skipped and a single merged DR call is made.
  - **Cleanup**: Deleted 3 completed implementation plan files (`pi-decide-fast-path`, `painter-images-api-retry`, `bioinformatics-swallowed-error-tech-agent`).

- `0.26.8` — swallowed-error detection & Technique Agent doc lookup (2026-06-17):
  - **`detect_swallowed_error()` in `error_parsing.py`**: New function that detects technical errors the agent caught in try/except and then chose to skip. Requires both a skip/abandon decision keyword AND a known `TECHNICAL_ERROR_TYPES` token (repr or str form). Prevents false positives for legit data-driven skips and transient errors the agent later fixes.
  - **Bioinfo agent integration**: `_detect_and_lookup_swallowed_error()` method consults the Technique Agent (`lookup_from_error`) for API documentation when a swallowed error is detected. `_build_swallowed_error_feedback()` generates steering messages that guide the agent to fix and retry instead of skipping.
  - **Execution loop injection**: Swallowed-error detection runs unconditionally after every code execution step. When detected, the feedback is appended to the tool result before it's sent back to the LLM, guiding the agent toward the correct API call.
  - **Test**: `test_error_parsing.py` — 5 tests covering scvelo case, str-form error, no-skip-keyword (returns None), data-driven skip (returns None), empty/None output.

- `0.26.7` — painter retry extraction, _images_api_attempt method, shared _retry_with_backoff (2026-06-16):
  - **Shared `_retry_with_backoff()` helper**: Extracted the 3× inline exponential-backoff retry loops (Revision, Chat generation, Images API) into a single reusable `_retry_with_backoff()` function. Client errors (4xx) break out immediately without retrying; transient errors use `2^attempt` delay. All call sites unified under the same retry logic.
  - **`_images_api_attempt()` method**: Extracted the single-attempt Images API call (`/v1/images/generations`) with `b64_json`→URL response-format fallback, standard/non-standard response extraction, and `content_filter`/error-field diagnostics into a dedicated method. Images API strategy now wrapped in `_retry_with_backoff()` instead of inline retry.
  - **Test**: `test_painter_retry.py` — 4 tests covering transient→success retry, 4xx client-error skip, exhaustion after 3 failures, and b64_json→no-format fallback preservation.

- `0.26.7` — painter retry extraction, _images_api_attempt method, shared _retry_with_backoff (2026-06-16):
  - **Shared `_retry_with_backoff()` helper**: Extracted the 3× inline exponential-backoff retry loops (Revision, Chat generation, Images API) into a single reusable `_retry_with_backoff()` function. Client errors (4xx) break out immediately without retrying; transient errors use `2^attempt` delay. All call sites unified under the same retry logic.
  - **`_images_api_attempt()` method**: Extracted the single-attempt Images API call (`/v1/images/generations`) with `b64_json`→URL response-format fallback, standard/non-standard response extraction, and `content_filter`/error-field diagnostics into a dedicated method. Images API strategy now wrapped in `_retry_with_backoff()` instead of inline retry.
  - **Test**: `test_painter_retry.py` — 4 tests covering transient→success retry, 4xx client-error skip, exhaustion after 3 failures, and b64_json→no-format fallback preservation.

- `0.26.6` — supervisor report path aligned with services/report shared modules (2026-06-12):
  - **Supervisor `_build_score_definitions_html` → shared module**: Replaced inline HTML table with `build_score_definitions_section_html()` from `score_section.py`, so the supervisor-generated report appendix uses the same enhanced rendering (Computation Method column, collapsible gene list, CSV download link) as the main report.
  - **Supervisor `_build_report_files` writes score CSV + literature report**: `write_score_definitions_csv()` now called during supervisor report file assembly, and `write_literature_report()` writes `literature_report.html` when `literature_reports` exist — both previously only done by the main report generator.
  - **Supervisor appendix literature model label**: `_build_appendix` now calls `resolve_literature_model()` for the literature agent model, showing the actually-used mode (Deep Research vs PubMed) instead of the configured synthesis model.
  - **Supervisor appendix literature download link**: `_build_appendix_html_section` includes `render_literature_download_li()` when `literature_report` is present in appendix metadata.
  - **`friendly_method` scanpy prefix normalization**: `score_section.py` now normalizes `scanpy.tl.*` → `sc.tl.*` before lookup, so `scanpy.tl.score_genes` resolves to the same friendly description as `sc.tl.score_genes`.
  - **Test**: `test_score_section.py` extended with `scanpy.tl.score_genes` normalization assertion.

- `0.26.5` — *(skipped — consolidated into 0.26.6)*

- `0.26.4` — score definitions shared module, literature report download, FetchError class (2026-06-12):
  - **Shared `score_section.py`**: New module extracting score definitions HTML rendering and CSV export from `variants.py`, `regenerator.py`, and `generator.py` into a single DRY source. `build_score_definitions_section_html()` and `write_score_definitions_csv()` are the two public functions, used by all three callers plus the exploration report.
  - **Enhanced HTML display**: New "Computation Method" column with biologist-friendly plain-language descriptions (e.g., "Mean expression of the gene set, relative to a random control gene set" for `sc.tl.score_genes`) derived from a `METHOD_AGGREGATION` map. Full gene list in collapsible `<details>` element (no longer truncated at 10). CSV download link above the table.
  - **CSV to `tables/` directory**: `write_score_definitions_csv()` now writes to `tables/score_definitions.csv` instead of report root, so it flows into the download-all-tables ZIP archive automatically.
  - **Download buttons always attempt tables**: `DownloadButtons.tsx` now always fires the tables-download request (catches 404 gracefully via `FetchError` instead of skipping when `tables.length === 0`), ensuring `score_definitions.csv` is always downloadable.
  - **`FetchError` class**: New `frontend/src/utils/download.ts` export — `FetchError` extends `Error` with a `status` property, enabling callers to distinguish HTTP errors (e.g., 404) from network failures.
  - **Literature report download**: New `backend/services/report/literature_files.py` module — `build_literature_report_html()` renders all `LiteratureReport` entries into a single self-contained HTML document with references; `write_literature_report()` persists it as `literature_report.html` in the report output dir. Generator writes it during report creation; regenerator and sections appendix both include a download link via `render_literature_download_li()`.
  - **Regenerator sync**: `generate_html_report()` now also regenerates `score_definitions.csv` into `tables/` during re-gen, keeping the CSV in sync with the HTML.
  - **CSS**: New `.score-definitions details.score-genes` styles for the collapsible gene list (indigo cursor, break-word, muted text).
  - **Test**: `test_score_section.py` — 10 tests covering friendly_method, normalize, HTML rendering (all genes present, download link, friendly computation, XSS escape), CSV writing (empty, content, full gene list).

- `0.26.3` — literature DR decomposition, report model label tracking, refactoring (2026-06-12):
  - **Literature DR task decomposition**: `PIAgent.plan_deep_research_tasks()` uses a PI LLM call to decide whether literature topics are complex enough to split into 2-5 independent Deep Research sub-tasks (serial calls, each autonomous). `_literature_research` in `agents.py` refactored: multi-task DR path (one serial call per decomposed sub-task) vs legacy merged DR path with PubMed gap fallback. Topic limit raised from 3→5.
  - **Report literature model label**: New `literature_actual_model` field on `AnalysisState` records the resolved display label (e.g., "Google Deep Research", "openai/o4-mini-deep-research", or the literature synthesis model). `LiteratureAgent.resolve_model_label()` maps search_mode to label. Report generator appendix and config table use `build_agent_models_appendix()` / `resolve_literature_model()` to show the actually-used model instead of the configured default.
  - **Refactoring**: `PIAgent._format_completed_results()` extracted as a static helper, reused by `plan_agent_tasks` and `plan_deep_research_tasks` to keep the findings-summary formatting consistent.
  - **Cache restoration**: Literature cache-hit path now restores `literature_actual_model` from cached search_modes so the report stays accurate when execution is served entirely from cache.
  - **5 new test files**: `test_literature_dr_decompose.py`, `test_literature_model_label.py`, `test_plan_deep_research_tasks.py`, `test_report_appendix_literature_model.py`, `test_report_literature_model.py`.

- `0.26.2` — current_round_direction None→"" coercion, session file chmod, default project description revision (2026-06-12):
  - **`current_round_direction` None coercion**: Added `field_validator` on `AnalysisState.current_round_direction` that coerces `None` to `""` at deserialization. Prevents `AttributeError` when legacy sessions or downstream code assigns `None`. Workflow `initialize.py` and `planning.py` changed all `= None` assignments to `= ""` to match the declared `str` type.
  - **Session file chmod alignment**: `_write_session_json` now calls `os.chmod(tmp_path, 0o644)` before `os.replace` so the temp file from `mkstemp` (which hardcodes 0600) matches the permissions of other session files (state.json, blobs). Isolation is at the app layer, not via mode bits.
  - **Default project description revision**: `Sidebar.tsx` default project description shortened and cleaned up — removed redundant line breaks, simplified wording (e.g., "copper and iron ions" → "copper metabolism", "ZnT1/Ctr1" → "ZnT1/CTR1").
  - **New test**: `test_current_round_direction_coercion.py` — verifies None→"" coercion and valid string preservation.

- `0.26.1` — species identification HITL, gene-species mismatch validation, WAITING_HITL_STATUSES refactor (2026-06-11):
  - **Species inference at Round 1**: PI agent infers species from project info with HITL loop (max 3 attempts) — `infer_species()`, `infer_species_from_hint()`, `decide_species_action()`. Low-confidence inference pauses for user input via `TextInputDialog`.
  - **Gene-species mismatch HITL**: New `_species_validation_callback()` in bioinformatics agent pauses workflow when detected, allowing user to correct species mid-analysis. `decide_species_action()` classifies user intent (change/keep/custom).
  - **New state fields**: `species`, `species_resolved`, `species_genes_validated`, `species_input_info` in `AnalysisState`; new `TaskStatus.WAITING_SPECIES_INPUT`; new `WaitInputReason` and `StateSyncChangeType` values for `SPECIES_INPUT` / `GENE_SPECIES_MISMATCH` / `SPECIES_RESOLVED` / `SPECIES_AUTO_SKIPPED`.
  - **`WAITING_HITL_STATUSES` constant**: Single `frozenset` in `lifecycle.py` replaces scattered inline `WAITING_*` tuple checks across pause/resume/stop.
  - **`prepend_species()` helper**: Injects species context into bioinformatics/pathology/biochemistry/literature agent task prompts.
  - **`SpeciesInputRequest` model**: New Pydantic request body with `Annotated[str, StringConstraints]`.
  - **`POST /{session_id}/species-input` endpoint**: New interaction route for user species input.
  - **Frontend**: `SpeciesInputInfo` store type, `TextInputDialog` integration, `state_sync` handlers for species events.
  - **i18n**: New `speciesInputTitle`, `speciesGeneValidationTitle`, `speciesInputPlaceholder`, `speciesGeneValidationPlaceholder`, `waitingSpeciesInput` keys.
  - **Lifecycle integration**: Pause/resume/stop all handle `WAITING_SPECIES_INPUT` status — empty signal on resume auto-skips, stop with summarize forces completion.

**0.25.x — LLM resilience, sandbox hardening, frontend refactoring (2026-05-27 → 2026-06-04)**

- `0.25.18` — import request logging, vite & CLI timeout hardening (2026-06-04):
  - **Import request logging**: `import_state()` logs structured metadata on every request (filename, content_length, trust, client IP, X-Forwarded-For, user_id, import_id) with `import_start_time` timing. Upload failure logs include `exc_info=True` for full stack traces.
  - **Vite timeout hardening**: New `disableRequestTimeout()` plugin sets `requestTimeout`/`headersTimeout` to 0 on both dev and preview servers. Proxy timeouts raised to 24h (86400000ms) so multi-GB uploads through `vite dev`/`vite preview` don't get silently aborted.
  - **CLI timeout**: `session_manager.py` creates the SDK client with `timeout=86400` (24h) to match the server-side import polling behavior — large archives can take hours to transfer and extract.

- `0.25.17` — import job `created_at` survival, SDK default timeout, vite preview proxy (2026-06-04):
  - **Import job `created_at` preservation**: Every job state transition (pending/extracting/completed/failed/cancelled) now carries forward `created_at` from the previous state instead of dropping it. Fixes a bug where the stale-job cleanup task in `session_cleanup.py` could never evict failed/cancelled jobs because `created_at` was lost on transition.
  - **SDK default timeout**: `Client.__init__` default `timeout` raised from 300s to 3600s — multi-GB archive uploads + extraction can legitimately exceed 5 minutes.
  - **Vite preview proxy**: Preview server configuration now proxies `/api` and `/ws` through to the backend, so production-like `vite preview` sessions work without CORS issues.

- `0.25.15` — import progress tracking, live cancel, trust-based pickle restore, blob round-trip & SDK overhaul (2026-06-04):
  - **Raw octet-stream streaming upload**: Import route reads `request.stream()` with `X-Filename`/`X-Import-Id`/`X-Trust` headers instead of multipart FormData. Eliminates Starlette's `SpooledTemporaryFile` which caused ~50 min of redundant disk I/O for multi-GB archives.
  - **Import progress tracking**: `_import_tar_gz` reports live `phase` (uploading/extracting/completed), `bytes_processed`, `files_processed`, `current_file` to the in-memory job dict. Frontend renders a real-time progress bar with cancel button for both upload and extraction phases.
  - **Live import cancel**: New `POST /api/session/import/{import_id}/cancel` endpoint sets `cancel_requested` flag on the job. The streaming upload loop and tar member loop check this flag cooperatively. `ImportCancelled` exception triggers `_cleanup_partial_session()` to remove half-extracted session directories.
  - **Page-refresh resilience**: Frontend persists `pendingImport` to `localStorage`; a mount-effect probe resumes polling and surfaces the progress modal on navigation refresh, preventing orphaned backend jobs.
  - **Trust-based sandbox pickle restore**: Import accepts `X-Trust: true` header / `trust` parameter. When true, sandbox pickles (`_session.pkl`, `_session.pkl.zst`) are written from the archive instead of dropped. Default `trust=false` maintains S4 security (dill.load = arbitrary code execution). Frontend shows `window.confirm()` dialog before import.
  - **Export `include_runtime`**: `GET /api/session/{id}/export?include_runtime=true` bundles sandbox session pickles (live kernel state + per-snapshot pkl.zst) in the archive. Default false for smaller portable archives. Frontend shows `window.confirm()` dialog before export.
  - **Python SDK `import_session` overhaul**: Raw byte-stream POST (no files/multipart). New `on_progress` callback, `trust` parameter, `cancel_import()` method. `_wait_for_import` handles `cancelled` status and invokes the progress callback on each poll.
  - **Python SDK `export_session` update**: New `include_runtime` parameter; passes `?include_runtime=true` query param to server.
  - **CLI (`session_manager.py`)**: `import` subcommand gains `--trust` flag; `export` subcommand gains `--include-runtime` flag. Live progress callback with `_print_progress()` single-line status to stderr.
  - **Session-level blob store packing**: `_add_session_blobs_to_tar()` packs `<session>/blobs/<hash>.b64` sidecars into tar exports so imported `@scai-blob` tokens resolve to real content instead of dangling as raw text in the UI.
  - **Recursive snapshot tar inclusion**: `_add_snapshots_to_tar()` uses `rglob` instead of `iterdir` so per-snapshot blob subdirs (`<snap_id>_blobs/<hash>.b64`) are included alongside top-level state/notebook/pkl files.
  - **Import `inline_blobs` resolution**: `_import_tar_gz` and `_import_zip` now call `inline_blobs()` on the deserialized `state_data` before constructing `AnalysisState`, restoring externalized blobs from extracted sidecars.
  - **Zip legacy path remapping**: `_import_zip` remaps legacy `<UPLOAD_DIR>/<id>`, `<WORKSPACE_DIR>/<id>`, `<REPORTS_DIR>/<id>` paths in the JSON payload to the current session-centric layout.
  - **Import path resolution for nested snapshots**: `_resolve_import_dest` handles `blobs/` entries and nested `snapshots/<subdir>/` paths (from rglob), not just top-level snapshot files.
  - **DeepResearch structured error logging**: New `_check_response()` on `GoogleDeepResearchClient` uses `response.is_error` (4xx/5xx only, lets 3xx pass) with structured warning logs, replacing bare `raise_for_status()` in `start_research` and `poll_research`.
  - **Stale import job cleanup**: `session_cleanup` handles `cancelled` status in TTL eviction alongside `completed`/`failed`.
  - **Route registry**: New `POST /api/session/import/{import_id}/cancel` endpoint pinned in `route_snapshot.json`.
  - **Auth test update**: `test_export_import_auth.py` uses raw octet-stream payload instead of multipart FormData for the 401 check.
  - **Inline size guard test cleanup**: `read_capped` / `_FakeUpload` removed (replaced by streaming approach). `MAX_INLINE_IMPORT_BYTES` constant is still exported.
  - **New test file**: `test_export_import_blob_roundtrip.py` — verifies session-blob and snapshot-blob content preserved losslessly through export→import.
  - **Extended test**: `test_import_pkl_skip.py` — tests nested pkl detection under snapshot blob directories.

- `0.25.14` — Google Deep Research API revision upgrade (2026-06-01):
  - **Model & API revision bump**: `GoogleDeepResearchClient.DEFAULT_MODEL` upgraded from `deep-research-pro-preview-12-2025` to `deep-research-preview-04-2026`; new `API_REVISION = "2026-05-20"` constant is now sent on every Interactions API call via the `Api-Revision` header (cached on the client as `_default_headers`).
  - **Structured `agent_config` payload**: Interaction creation now sends `agent_config: {type: "deep-research", thinking_summaries: "auto", visualization: "auto", collaborative_planning: false}` and reorders `agent` before `input` to match the new schema.
  - **New response schema extraction**: `_extract_research_result()` first reads the top-level `output_text` and walks `steps[]` for `thinking_summary` / per-step `groundingMetadata`; legacy `outputs[]` and `output.parts[].text` paths are retained as defensive fallbacks.
  - **Top-level grounding compatibility**: Accepts both `groundingMetadata` (camelCase) and `grounding_metadata` (snake_case) at the response root so source citations keep working across schema variants.

- `0.25.13` — transformLoadResponse absent-field preservation (2026-05-31):
  - **Partial Session updates**: `transformLoadResponse()` no longer fills missing fields with empty strings/arrays/zeros — uses `opt()` helper that omits the key entirely when raw value is null/undefined.
  - **Null-safe store updates**: `setStateVersion`/`setSnapshotCount` now guarded with `!= null` checks in `useApi.ts` to prevent unintentional resets to 0.
  - **Optional API types**: `SessionLoadResponse` fields (`name`, `status`, `current_round`, `created_at`, `initial_instruction`, `treatment_groups`, `data_files`, `biochemistry_indicators`, `round_results`) now typed as optional to match server behavior.
  - **Session defaults**: `Session` construction provides sensible defaults in `useApi.ts` for `name`, `status`, `currentRound`, `instruction`, `treatmentGroups`, `dataFiles`, `biochemistryIndicators`, and `roundResults`.

- `0.25.12` — volume-based context compaction & reactive empty-response escalation (2026-05-28):
  - **Volume-based context compaction**: `_compact_context()` applies three-phase compaction at 150K estimated tokens: (1) truncate large tool/user content with decreasing budgets 64K/32K/16K chars, (2) drop+summarize oldest non-protected messages, (3) fallback truncate protected window. Protects anchor system/dynamic messages and the 6 most recent messages.
  - **Reactive compaction on empty response**: Empty responses trigger immediate compaction to `COMPACT_TARGET_TOKENS=105K` before retrying, replacing the previous nudge-only approach. The nudge now only fires when compaction didn't occur (compaction alone already changes the request sufficiently).
  - **Empty-response tool_choice escalation**: When `empty_response_count > 0`, forces `tool_choice="required"` — the model must respond with a concrete tool call instead of another zero-content turn.
  - **New static methods**: `_estimate_context_tokens()`, `_compact_context()`.
  - **New test file**: `test_context_compaction.py`.

- `0.25.11` — smart retry resume & checkpoint stage awareness (2026-05-28):
  - **Resumable checkpoint detection**: `has_resumable_checkpoint()` returns True when a non-completed checkpoint for the current round+stage exists. `_bioinformatics_analysis` uses this to skip the PI planning prelude (goal definition, task refinement, snapshot, instruction display) on Retry — re-running PI LLM calls wastes tokens and confuses the user.
  - **Checkpoint stage guard**: Checkpoints now store `planning_stage`. `run_autonomous_analysis` rejects checkpoints whose `planning_stage` doesn't match the current stage, preventing a stale stage1 checkpoint from being wrongly resumed when stage2 begins (both share the same `round_number`).
  - **Pre-exhaustion checkpoint save**: Before raising `UpstreamExhaustedError` on empty responses, the full conversation state (messages, findings, figures, code_history, analysis_log) is saved as a checkpoint. A Retry from the HITL gate then resumes with the perturbed messages instead of replaying the identical prefix.
  - **Empty-response nudge perturbation**: Each empty-response retry appends a distinct `[System]` nudge message before retrying — the changing suffix busts both prefix caching and deterministic re-failure in reasoning models.
  - **New helper**: `_checkpoint_state_dir()` resolves checkpoint path, shared by `has_resumable_checkpoint` and `run_autonomous_analysis`.

- `0.25.10` — ZMQ thread leak fix (2026-05-28):
  - **Orphaned reader thread elimination**: Replaced `asyncio.wait_for(asyncio.to_thread(get_iopub_msg))` with `asyncio.to_thread(get_iopub_msg, timeout=...)` across the entire `execute()` method. The old pattern only cancelled the coroutine — the underlying thread stayed blocked on ZMQ recv forever, leaking orphaned reader threads. Multiple threads reading the same ZMQ socket is unsafe and eventually triggers libzmq `abort()` in `src/msg.cpp:387`.
  - **Timeout detection via QueueEmpty**: All `except asyncio.TimeoutError` changed to `except QueueEmpty` since the ZMQ-level poll timeout now surfaces as `queue.Empty` instead of `asyncio.TimeoutError`. `queue.Empty` imported as `QueueEmpty`.
  - **Restart channel drain**: `_restart_kernel` drains stale ZMQ messages (up to 10) from new channels after restart, preventing leftover kernel-info broadcasts from confusing the first `execute()`.
  - **Restart sleep guard**: Added `await asyncio.sleep(0.3)` before stopping channels during kernel restart, letting in-flight reader threads finish their current poll cycle.
  - **wait_for_ready simplified**: Uses `get_shell_msg(timeout=5)` directly with `except QueueEmpty` handler instead of nested `asyncio.wait_for`/`asyncio.to_thread`.

- `0.25.9` — streaming session save/restore & out-of-kernel snapshot compression (2026-05-28):
  - **Per-variable streaming dill**: `save_session` writes each variable as an independent `dill.dump((name, obj), f)` frame. Unpicklable variables skipped via `tell()/seek()/truncate()` rollback — file always parseable. Peak kernel memory drops from "N full pickle byte strings" to one variable's buffer, eliminating OOM-kills.
  - **Out-of-kernel snapshot compression**: Snapshot pkls stored as `<id>_session.pkl.zst`. Compression/decompression runs in the backend process (via `asyncio.to_thread` + `zstd.copy_stream`), not the kernel. Old kernel-level `ZSTD_MAGIC`/`_is_zstd`/`COMPRESSION_LEVEL` constants removed from `sandbox.py`.
  - **Import security**: `is_sandbox_pickle` now matches `_session.pkl.zst` in addition to `_session.pkl`.
  - **NOT backward compatible**: Sessions saved before this version cannot be resumed.
  - **Deleted test file**: `test_pkl_compression.py` (kernel-level zstd approach removed). Tests in `test_snapshot_rollback.py` and `test_sandbox_save_restore_streaming.py` rewritten for the new format.
  - **New test file**: `test_sandbox_save_restore_streaming.py`.

- `0.25.8` — supervisor force-skip, kernel death guard, figure template hardening (2026-05-28):
  - **SupervisorEvalResult**: New dataclass with `feedback` and `force_skip` fields. `evaluate_step()` returns structured result instead of bare string. `force_skip=True` when agent is stuck AND PI has already intervened.
  - **Bioinfo force-skip escalation**: On first 2 force-skips, injects MANDATORY skip instructions to agent messages. On 3rd force-skip (same error type), forces `report_findings` and hard-exits the tool-call loop with partial results.
  - **Figure template hardening**: All code templates (auto-save, initial plot, revision, retry) restructured — `plt.close`/`plt.show` restore + figure save moved into `finally` block so JPG+PDF are always written even on user code error. Individual save formats wrapped in try/except so one failure doesn't block the other. PDF supplement deduplicates targets. Figure-less execution prints a diagnostic message.
  - **Kernel death guard**: `execute()` now checks `km.is_alive()` before running code and on iopub poll timeout — dead kernel triggers restart early, preventing libzmq abort from corrupt ZMQ channels after OOM kill.
  - **savefig-redirect diagnostics**: Logs when resolved path is outside workspace root.
  - **Tests**: `test_supervisor_stuck_escalation.py` extended with force_skip assertion tests.

- `0.25.7.5` — cleanup & refactoring (2026-05-28):
  - **Rate limiter lazy init**: `SlidingWindowRateLimiter` switched from `defaultdict(deque)` to `dict` + manual `get()` — no auto-created entries for non-existent keys. `_prune` returns `None` for unseen keys; `is_allowed` short-circuits to True.
  - **Node-map assertion refined**: `constants.py` drift guard now separately tracks `undocumented` vs `stale` nodes with clearer error messages.
  - **PI agent cleanup**: `STOP_KEYWORDS` import moved to top-level; restored accidentally removed `user_lower = user_message.lower()` in `process_user_input`.
  - **Unknown WaitInputReason handling**: `resolve_hitl_skip_target` logs a warning for unrecognized `wait_input_reason` values so new gate types don't silently deadlock.
  - **Minor simplifications**: `_import_zip` removes redundant `is_sandbox_pickle` check; `chat_sections` simplifies content extraction; `user_cleanup` calls `shutil.rmtree` directly via `to_thread`; `ttl_cleanup` uses `unlink(missing_ok=True)`; `_evaluate_dr_coverage` removes unreachable `return []`; bioinfo auto-retry re-raises `UpstreamExhaustedError`.

- `0.25.7.4` — node-map drift guard, STOP_KEYWORDS consolidation, cleanup async offload (2026-05-28):
  - **Node-map drift guard**: `constants.py` adds `RESULT_KEYLESS_NODES` and import-time assertions ensuring `NODE_TO_AGENT` and `NODE_TO_RESULT_KEY` stay in sync. New test `test_node_maps_consistent.py`.
  - **STOP_KEYWORDS consolidation**: Moved inline stop-keyword lists from `pi_agent.py`'s `process_human_guidance` and `process_user_input` into a single `constants.STOP_KEYWORDS` frozenset. Both paths now import from the shared source.
  - **Supervisor appendix uses resolved models**: `_build_appendix` reads resolved provider models/temperatures (via `AgentSettingsProvider`) instead of raw `custom_models` overrides, so the appendix now shows all agents even without overrides.
  - **chat_sections title generation fix**: Uses `llm_client.chat(model=model, messages=...)` signature and reads `LLMResponse.content` attribute. Title generation guards on `model` being non-None.
  - **Cleanup async offload**: `user_cleanup.delete_user_data` routes `shutil.rmtree` and state-file `glob`/`unlink` through `asyncio.to_thread`. `ttl_cleanup` offloads chunk-file deletion. `db.list_expired_uploads` no longer filters `completed=0`, so completed-but-expired uploads also get cleaned.
  - **New tests**: `test_chat_sections_title_generation.py`, `test_supervisor_appendix_models.py`, `test_pi_stop_keywords.py`, `test_node_maps_consistent.py`, `test_cleanup_async.py`.

- `0.25.7.3` — image MIME detection, independent 429 budget, wall-clock sandbox timeout (2026-05-28):
  - **WebP MIME fix**: `detect_image_mime()` checks the WEBP fourcc at bytes 8:12, not just the RIFF prefix — RIFF-container WAV/AVI are no longer misclassified as `image/webp`.
  - **Independent 429 retry budget**: `compute_rate_limit_delay()` tracks the 429 counter separately from other error types (5xx/timeout/JSON). A burst of 5xx can no longer shift or consume the rate-limit retry budget. JSON retry count is now correctly bounded to `MAX_JSON_RETRIES` (uses `<=` instead of `<`).
  - **Wall-clock sandbox timeout**: Iopub loop uses a fixed `deadline` before the loop; `compute_remaining_timeout()` is checked each iteration with `min(remaining, IOPUB_POLL_CAP=30s)` per poll. A kernel that keeps streaming output (tqdm/scanpy verbose) can no longer reset the timeout indefinitely.
  - **New tests**: `test_image_mime.py`, `test_json_retry_budget.py`, `test_rate_limit_backoff.py`, `test_sandbox_timeout.py`.

- `0.25.7.2` — tool-message pairing & HITL gate routing fixes (2026-05-28):
  - **Tool-call pairing helpers**: New `tool_messages.py` with `tool_calls_are_paired()` (detect orphaned tool calls) and `append_missing_tool_results()` (backfill placeholders). Used by checkpoint validation and plan-revision pause to prevent upstream HTTP 400 from orphaned tool_call IDs.
  - **Disabled-agent HITL routing**: `resolve_hitl_skip_target()` in `lifecycle.py` returns `(node_name, generation)` for both LLM-failure AND disabled-agent gates. `stop_analysis` "summarize" action now releases both gate types via `signal_llm_failure_action` instead of deadlocking on disabled-agent gates.
  - **WebSocket init failure recovery**: Workflow construction in `websockets.py` wrapped in try/except — cleans up `initializing_sessions` and sends error message on failure, preventing observer connections from spinning forever.
  - **UpstreamExhaustedError re-raise**: `_evaluate_dr_coverage` (agents.py) and `_revise_figure_code` (bioinformatics_agent.py) now re-raise `UpstreamExhaustedError` instead of swallowing it, correctly delegating to `@with_retry`.
  - **New files**: `backend/utils/tool_messages.py`, `backend/tests/test_tool_messages.py`, `backend/tests/test_dr_coverage_reraise.py`, `backend/tests/test_hitl_skip_target.py`.

- `0.25.7.1` — login rate-limit & import security hardening (2026-05-28):
  - **Login brute-force protection**: `SlidingWindowRateLimiter` — thread-safe in-process rate limiter (defaultdict(deque) + Lock). Applied to `/api/auth/login`: max 5 failed attempts per (username, client-ip) / 60s → HTTP 429. Resets on successful login.
  - **Import size guard**: `check_inline_import_size()` fast-rejects oversized inline imports via Content-Length; `read_capped()` streaming read with 100 MB cap for requests that omit/lie about Content-Length → HTTP 413.
  - **Import pickle security (S4)**: `is_sandbox_pickle()` detects session pkl files; both tar.gz and zip import paths now skip sandbox pickles instead of writing them, preventing arbitrary code execution via dill.load on untrusted archives.
  - **New files**: `backend/utils/rate_limit.py`, `backend/tests/test_rate_limit.py`, `backend/tests/test_import_pkl_skip.py`, `backend/tests/test_import_size_guard.py`.

- `0.25.6` — mid-execution plan revision routing fix (2026-05-28):
  - **`decide_bioinfo_post_execution`**: New pure function in `helpers.py` that returns one of `"revision"`, `"stage1_complete"`, or `"complete"` — replacing inline stage checks that masked mid-plan revision.
  - **Mid-execution revision detection**: `_bioinformatics_analysis` now checks the raw agent `result` (before it's rebuilt into `final_result`, which drops `plan_revision_needed`). On `"revision"`, stays in current stage, keeps `current_plan` intact, and invalidates cache — the agent resumes the same plan from its checkpoint after revision is approved.
  - **Stage1 revision pause**: Previous code checked `plan_revision_needed` on `final_result` (which never carried it), making mid-execution revision detection impossible. Now correctly pauses in stage1 without advancing to stage2.
  - **Tests**: `test_bioinfo_post_execution_decision.py` and `test_stage1_revision_pause.py`.

- `0.25.5` — reasoning echo refactor (2026-05-27):
  - **Structured reasoning echo**: Added `_REASONING_ECHO_KEYS`, `_reasoning_echo_allowed()`, and `_build_assistant_echo()` — single source of truth for echoing thinking-mode reasoning on tool-call turns. Preserves all known reasoning field names verbatim (`reasoning_content`, `reasoning`, `reasoning_details`, `thinking`, `thought`), matching DeepSeek v4, MiMo, and Anthropic/OpenRouter thinking requirements.
  - **R1 exclusion**: `_reasoning_echo_allowed` returns False for `deepseek-reasoner` / `-r1` models — R1 rejects multi-turn requests containing `reasoning_content` with HTTP 400.
  - **Tool-turn scoping**: Reasoning is echoed ONLY on turns with `tool_calls`, matching the official API contract. Plain-text turns never carry reasoning, even on echo-required models.
  - **Simplified base**: `BaseAgent.add_to_history` reverted to plain 2-arg method — reasoning echo is handled at the `raw_assistant_message` layer, not in `LLMResponse` or message_history.
  - **Tests**: `test_reasoning_echo.py` (158 lines) covering model detection, key set completeness, tool/plain/R1 echo behavior, structured array preservation, and end-to-end pipeline.

- `0.25.4` — workflow teardown refactor & edge-guard (2026-05-27):
  - **Shared teardown helper**: New `routes/_helpers.py` with `teardown_running_workflow()` — extracts duplicated workflow-stop, task-cancel, and checkpoint-delete logic from `snapshots.py` and `lifecycle.py` into a single reusable function.
  - **Re-analysis teardown**: `lifecycle.py` `_resume_analysis_locked` now calls `teardown_running_workflow` before resetting a COMPLETED session, so WS reconnect triggers a fresh graph run instead of resuming a terminated graph (which caused infinite loop).
  - **Graph-at-END guard**: `core.py` `_resume_analysis` detects when `state.next` is empty (graph already terminated) and marks COMPLETED immediately, preventing infinite resume loop on WebSocket reconnect.
  - **Snapshot restore cleanup**: `snapshots.py` `restore_snapshot` delegates to `teardown_running_workflow`; `delete_checkpoint` now runs inside the shared helper.

- `0.25.3` — session loading overlay (2026-05-27):
  - **Session loading overlay**: Full-screen backdrop with animated DNA icon and i18n text (`loadingAnalysis`) shown while `loadSession` is fetching, providing users visual feedback during session load.
  - **Load session state**: `useApi.ts` sets `sessionLoading=true` before fetch, `false` in finally block. New `sessionLoading` / `setSessionLoading` in zustand `appStore`.
  - **i18n**: Added `loadingAnalysis` key to both zh and en translations.

- `0.25.2` — temp-file cleanup on failure (2026-05-27):
  - **Orphaned temp file cleanup**: `externalize_blobs` wraps write+replace in try/except — on failure, removes the orphaned `.tmp` file before re-raising.
  - **Snapshot failure cleanup**: `create_snapshot` `_externalize_and_write` cleans up blob dir and temp JSON on failure, preventing stale snapshot artifacts.

- `0.25.1` — zstd streaming compression for session pkl (2026-05-27):
  - **Zstd-compressed pkl**: Session save/restore uses `zstandard.stream_writer`/`stream_reader` for streaming compression. Plain dill fallback for backward compat. Corrupt zstd raises clear `RuntimeError("zstd decompression failed (corrupt pkl)")`.
  - **Atomic pkl write**: Save writes to temp file first, then `os.replace()` — no half-written compressed file on crash.
  - **New dependency**: `zstandard>=0.23,<1.0` added to `requirements.txt`.
  - **Lazy zstd import**: `zstandard` imported only inside the zstd branch of restore code — plain-dill restore stays dependency-free.
  - **Tests**: Added `test_pkl_compression.py` (magic detection, round-trip dict/sparse, compression ratio, backward compat, corrupt zstd, kernel code compilation, exec round-trip) and `test_snapshot_rollback.py` (snapshot copies compressed/plain pkl byte-identical, detection round-trip).

**0.24.x — collaborative analysis planning (2026-05-15)**

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
  - **Test additions**: Added `test_stage1_guidance_scope.py` and `test_pi_review_stage_scope.py` for stage scope validation coverage.


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

- `0.23.7` — **Frontend read-only hardening & HITL dialog reopen**: Sidebar now shows "Open Saved Session" and export actions in read-only mode while gating delete, edit, clear, new session, and API settings behind `!isReadOnly`; `DataConfigMenu` accepts a `readOnly` prop to conditionally hide import/configure actions; AnalysisPanel status badges ("Waiting for Annotation" / "Waiting for Input") are now clickable buttons that reopen the corresponding interaction dialog via `handleReopenDialog`, improving HITL workflow accessibility.

- `0.23.6` — **Report download & export improvements**: Export now saves to `UPLOAD_TEMP_DIR` instead of reports dir; filename format changed to `export_{session_id[:8]}_{safe_name}.tar.gz`; `_should_skip_in_report_zip()` helper excludes report archives and orphan .zip.tmp files from report ZIP downloads; `download_full_report` now uses streaming temp file with background cleanup; frontend shows download errors inline instead of console errors.

- `0.23.5` — **LLM failure intervention & report robustness**: Extracted `_wait_for_llm_intervention()` helper for LLM failure handling; `UpstreamExhaustedError` now propagates from `generate_final_summary`, `generate_detailed_report`, `generate_graphic_abstract`, and report route (with 3 retries + exponential backoff); supervisor report generation also propagates UpstreamExhaustedError; section title safety net replaces placeholder text with descriptive defaults; improved retry messaging via WebSocket (`retrying` type).

- `0.23.4` — **Bioinformatics upstream error propagation**: Fixed `UpstreamExhaustedError` from bioinformatics agent not propagating to `@with_retry` decorator; bioinformatics agent now raises instead of catching; workflow caller no longer catches and returns early; ensures LlmTimeoutDialog triggers for upstream exhaustion in bioinformatics node; comprehensive regression tests added.

- `0.23.3` — **Python client session export**: Added `export_session()` method to Python client for exporting sessions as tar.gz archives with streaming download, atomic write (`.partial` + rename), configurable timeout and chunk size.

- `0.23.2` — **Snapshot restore synchronization & config sync**: Refactored snapshot restore with per-session locking to prevent race conditions; `_write_state_if_current()` helper prevents stale writes from superseded workflows; improved error message "No API key configured..."; removed unnecessary API key check from session creation endpoints (`start_session`, `ensure_session`); WebSocket observer skips sending stale status during snapshot restore; workflow defers to HITL-hold path when status is WAITING_INPUT/WAITING_ANNOTATION; frontend shows alert on session creation failure; frontend config sync: App startup now syncs non-admin user's llmBaseUrl from backend to store; ConfigPanel saves both user LLM config to backend and updates store for in-session use.

- `0.23.1` — **Report enhancements, chat sections, round tracking**: Added report generation concurrency guard (409 on duplicate requests); automatic report archiving before regeneration (keeps last 5 versions); archive listing/download API endpoints; `round_number` field added to `AnalysisStep` schema for round tracking; supervisor log restored from report.json now groups by round (`all_rounds_log` vs current `supervision_log`); new chat sections generation from chat messages (`chat_sections.py`); config resolver now allows per-user config without global config; new `/user/config/validate-saved` endpoint for validating saved credentials; frontend validation flow: admin validates with frontend config, non-admin validates saved config; validation on report generation (check API key before start); session cleanup clears report generation pending flag.

**0.22.x — snapshot rollback, LLM failure intervention, per-user config (2026-05-03 → 2026-05-08)**

- `0.22.12` — Database inline foreign keys (2026-05-08): Removed `_connect()` helper; PRAGMA foreign_keys enabled directly in `delete_user()` and `set_user_llm_config()` for consistency.

- `0.22.11` — PubMed Email for all users (2026-05-08): PubMed Email field in ConfigPanel made available to non-admin users, enabling literature search credential configuration for all roles.

- `0.22.10` — Minor fixes & cleanup (2026-05-07): Removed `configSkipped` state; deleted stray `next-env.d.ts`; version bump.

- `0.22.9` — Per-user LLM configuration & config persistence (2026-05-07): `user_llm_config` DB table with Fernet encryption; `/api/user/config` CRUD routes; `resolve_config_for_user()` merges per-user credentials over global config; `config_resolver.py`, `crypto.py`, `llm_probe.py` added; frontend `useUserConfig()` hook; ConfigPanel split between admin/user modes; global config persists to DB.

- `0.22.8` — Auth-optional path refinement (2026-05-07): `SESSION_AUTH_OPTIONAL_SUFFIXES` tuple for report/export/figure paths; `_is_auth_optional()` refactored into helper.

- `0.22.7` — Auth ownership enforcement (2026-05-07): `require_session_ownership()` replaces manual `ensure_session_loaded()` across all routes; `Request` parameter added to all session-specific endpoints; chat routes rename `request` to `body`; report WebSocket uses auth token/protocol; admin initialization check.

- `0.22.6` — LLM failure workflow (2026-05-07): `@with_retry` while-loop with user intervention via per-node `asyncio.Event`; generation-based idempotency; `UpstreamExhaustedError` with context fields (`agent_name`, `model`, `llm_server`); `POST /llm-failure-action` route; frontend `LlmTimeoutDialog` and `LlmModifySubDialog` with countdown auto-skip; stale LLM failure cleanup on session load; `constants.py` created.

- `0.22.5` — Supervisor stuck escalation & resilience (2026-05-05): Supervisor detects stuck agents (same error >=3 times) and escalates to PI; `CODE_GEN_ERROR_TYPES` added with static hints; bioinformatics agent error tolerance increased (MAX_CONSECUTIVE_ERRORS 3→5, MAX_TOTAL_ERRORS 30→50); last-chance attempt messaging.

- `0.22.4` — Snapshot logic refactor & UI overhaul (2026-05-05): Extracted `compute_snapshot_context()` and `create_and_register_snapshot()` static methods shared by workflow and chat; chat auto-snapshots added; removed `cleanup_old_snapshots()`; frontend CSS-grid snapshot tree with SVG rails/fork paths; snapshot count reminder dialog; `snapshot_count` field in WebSocket messages; favicon; report auto-snapshot; start.sh build ordering fix.

- `0.22.3` — Import hardening & memory optimization (2026-05-04): Added `validate_path_within()` for path-traversal protection; `_resolve_import_dest()` extracts archive-to-session mapping; `shutil.copyfileobj()` with `UPLOAD_CHUNK_SIZE` reduces peak memory; structured logging for import lifecycle.

- `0.22.2` — Snapshot rollback enhancement (2026-05-03): Extended `Snapshot` model with `notebook_file`, `workspace_manifest`, `branch_label`; added `restore_notebook()`, `cleanup_workspace_files()`, `cleanup_reports()`; new `GET /rollback-preview` and `DELETE /branch` endpoints; branch labels auto-computed; frontend CSS-grid snapshot graph with SVG rails/fork paths and rollback preview.

- `0.22.1` — Reproducibility toggle (2026-05-03): Added `enable_reproducibility` boolean to `SystemConfig` (env var `SCAI_ENABLE_REPRODUCIBILITY`, default `False`); extracted `_maybe_generate_reproducibility_bundle()` with lazy import; download endpoint returns 403 when disabled.

**0.21.x — session priority loading, settings provider, username auth (2026-05-03 → 2026-05-08)**

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
