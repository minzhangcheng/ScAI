# InteSc - Integrated and Intelligent Single Cell Sequencing Analysis

**InteSc** is a multi-agent AI system for comprehensive biomedical research data analysis. It brings together pathology image analysis, single-cell sequencing data, biochemical indicators, and literature research into a unified, iterative workflow — powered by seven specialized AI agents that collaborate like a research team.

**Version**: 0.13.3

## Core Features

### Multi-Agent Collaboration

Seven specialized AI agents work together to analyze your data from different angles:

- **PI Agent** — The project coordinator. Plans analysis strategy, synthesizes results from all agents, resolves conflicts, and reviews generated figures for publication quality
- **Pathology Agent** — Analyzes histological section images (H&E and custom staining types), identifying tissue structures, cellular patterns, and pathological changes
- **Bioinformatics Agent** — Performs autonomous single-cell sequencing data analysis (scRNA-seq, snRNA-seq, bulk RNA-seq). Executes code in a local Jupyter environment, generates publication-quality figures, and can run up to 200 analysis steps per round
- **Biochemistry Agent** — Interprets biochemical experimental results from structured indicator values and data images, cross-referencing findings from other agents
- **Literature Agent** — Searches and synthesizes relevant scientific literature. Supports three modes: Deep Research Models (e.g., openai/o4-mini-deep-research), Google Gemini Deep Research, and PubMed + LLM synthesis for targeted searches
- **Painter Agent** — Generates graphic abstracts summarizing the research findings
- **Supervisor Agent** — Monitors bioinformatics analysis steps in real-time using a lightweight vision LLM, detecting issues, identifying biological checkpoints, and escalating critical problems to the PI Agent

### Iterative Analysis with Human-in-the-Loop

- Multi-round analysis that progressively deepens — each round builds on previous findings
- After each round, a feedback dialog shows findings from all agents. You can provide guidance, upload additional data, or let analysis continue automatically
- From round 2 onwards, PI Agent intelligently decides which agents need to run, saving time and API costs
- You can pause and resume analysis at any time; all state is preserved

### Interactive Chat

- Chat freely with the PI Agent after analysis to ask questions, request explanations, or explore results further
- PI can delegate complex questions to specialized agents in the background
- Upload images and data files directly in the chat for on-the-fly analysis
- PI can display existing analysis figures or generate new custom figures

### Comprehensive Report Generation

Generate detailed HTML reports with 9 chapters and 40+ analysis items:

1. Quality Control (QC)
2. Cell Clustering
3. Cell Type Annotation
4. Cell Proportion Analysis
5. Differential Expression Analysis
6. Pathway Enrichment Analysis (GO/KEGG/GSEA)
7. Cell-Cell Communication Analysis
8. Trajectory Analysis
9. Targeted Analysis (custom)

Reports include high-resolution PNG/PDF figures, data tables, and a cell cluster annotation table.

### Session Management

- **Pause/Resume**: Pause analysis at any time; resume from where you left off with all data and variables intact
- **Auto-save**: State is automatically saved after each round
- **Export/Import**: Download complete sessions as archives and restore them on any instance
- **Session Browser**: Browse and manage all sessions with status and metadata display

### Multi-language Support

- Chinese/English interface switching
- Agent prompts auto-switch based on language setting

## Use Cases

- **Animal Model Phenotype Analysis**: KO/KI mice, disease models, etc.
- **Tumor Research**: Comprehensive analysis of tumor vs normal tissue
- **Drug Toxicology**: Tissue damage assessment after drug treatment
- **Organ-specific Research**: Liver, heart, kidney, brain, etc.
- **Developmental Biology**: Comparative analysis across developmental stages
- **Disease Mechanism Research**: Multi-omics data integration analysis

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Web Frontend (Port 3000)                   │
│  ┌──────┐ ┌───────┐ ┌────────┐ ┌───────┐ ┌──────┐ ┌──────┐ │
│  │ Data │ │Prompts│ │Analysis│ │Results│ │Report│ │ Chat │ │
│  │Panel │ │ Panel │ │ Panel  │ │ Panel │ │Panel │ │Panel │ │
│  └──────┘ └───────┘ └────────┘ └───────┘ └──────┘ └──────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ WebSocket / REST API
┌─────────────────────────▼───────────────────────────────────┐
│                    Backend (Port 8000)                       │
├─────────────────────────────────────────────────────────────┤
│                  Multi-Agent Workflow Engine                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │   PI    │  │ Pathol. │  │ Bioinfo │  │ Biochem │       │
│  │  Agent  │  │  Agent  │  │  Agent  │  │  Agent  │       │
│  └────▲────┘  └─────────┘  └──┬───┬──┘  └─────────┘       │
│       │                       │   │                         │
│       │  consult/fig review   │   │  annotate/lit search    │
│       └───────────────────────┘   │                         │
│       │                           ▼                         │
│  ┌────┴──────┐ ┌───────────┐ ┌─────────────┐               │
│  │Supervisor │ │  Painter  │ │  Literature │               │
│  │   Agent   │ │   Agent   │ │    Agent    │               │
│  └───────────┘ └───────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  OpenRouter  │  │Local Jupyter │  │ Google APIs  │      │
│  │  (LLM API)   │  │   Kernel     │  │ (Research +  │      │
│  │              │  │              │  │  Imagen)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Clone the Project

```bash
git clone <repository-url>
cd intesc
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Start the Application

**Option A: Using Startup Script (Recommended)**

```bash
./start.sh
```

This will automatically create a virtual environment, install dependencies, and start both backend (http://localhost:8000) and frontend (http://localhost:3000).

To stop:
```bash
./stop.sh
```

**Option B: Manual Startup**

```bash
# Terminal 1: Start backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Access **http://localhost:3000** to open the web interface.

## User Guide

### 1. Configure API Keys

On first launch, a configuration dialog will appear. You need to set up:

- **OpenRouter API Key** (required): Used by all AI agents for analysis. Get one at https://openrouter.ai/keys
- **PubMed Email** (required): Required by NCBI for PubMed literature searches. Use any valid email address
- **Google AI API Key** (optional): Enables Gemini Deep Research mode for comprehensive literature surveys. Get one at https://aistudio.google.com/app/apikey. When not set, the system uses PubMed + LLM mode instead
- **Proxy URL** (optional): HTTP/HTTPS proxy if your network requires one for API access

You can update these settings at any time via the gear icon in the header.

### 2. Create an Analysis Session

Click **"New Analysis"** in the sidebar. Enter a clear description of your research objective. The more specific you are, the better the agents can tailor their analysis. Examples:

> Analyze the liver tissue phenotype of Alb-Cre; Genex-flox/flox knockout mice, focusing on cell type composition changes, fibrosis-related signaling pathways, and immune cell infiltration

> Compare NASH model mouse liver with control, identify key differentially expressed genes and enriched pathways in hepatocytes and macrophages

### 3. Define Treatment Groups

In the **Data** panel, define your experimental groups before uploading data:

- Click **"Add Group"** to create groups (e.g., "Control", "NASH Model", "Treatment A")
- Set the **sample count** for each group (e.g., 3 biological replicates)
- Each uploaded file will be assigned to a specific group and sample number

This group structure is used by all agents to organize comparisons and ensure figures reference the correct experimental conditions.

### 4. Upload Data

Upload your research data in the **Data** panel. Each file is assigned to a treatment group and sample:

**Pathology Images**
- Supported formats: JPEG, PNG, TIFF, SVG
- Staining types: H&E or custom (specify the stain name)
- Large images are automatically split into tiles for analysis
- Assign each image to its treatment group and sample number

**Sequencing Data**
- Primary format: `.h5ad` (AnnData) — recommended for single-cell data
- Also supports: `.loom` files
- Sequencing types: scRNA-seq, snRNA-seq, bulk RNA-seq, or other
- Assign to treatment group and sample number

**Biochemistry Data**
- **Manual entry**: Enter indicator values directly (e.g., ALT = 45 U/L)
- **File upload**: Upload data tables (CSV/Excel) or images of experimental results
- Assign to treatment group and sample number

All uploaded files can be downloaded back via the download buttons in the Data panel.

### 5. Start Analysis

Click **"Start Analysis"** in the Analysis panel. The system runs a **two-phase analysis workflow**:

**Phase 1 — Exploration (Round 1)**

1. **PI Planning** — PI Agent reviews your data and creates an analysis plan, assigning specific tasks to each agent
2. **Agent Execution** — Biochemistry, Pathology, Bioinformatics, and Literature agents run their analyses sequentially
3. **Cell Annotation Confirmation** — After the Bioinformatics Agent completes cluster annotation, analysis pauses and a dialog appears showing each cluster's cell type, confidence level (high/medium/low), and key markers. Low-confidence clusters require manual confirmation or correction before analysis continues. High-confidence results auto-confirm after a countdown
4. **PI Synthesis** — PI Agent synthesizes findings from all agents
5. **Exploration Report** — At the end of Round 1, PI generates a structured Exploration Report summarizing cluster overview, data quality warnings, heterogeneity findings, pathway changes, and research direction candidates. A dialog allows you to select a direction for Phase 2 (from AI-suggested candidates, a custom direction, or let PI decide automatically)

**Phase 2 — Innovation (Round 2+)**

Following rounds use your chosen research direction as the focal point for deeper, hypothesis-driven analysis. PI Agent and specialized agents conduct in-depth investigation of the selected direction. From round 2 onwards, PI intelligently decides which agents need to run, saving time and API costs.

The Bioinformatics Agent runs code in a local Jupyter kernel — you can see the code and outputs in real-time. All figures are automatically saved as both PNG and high-resolution PDF.

### 6. Provide Feedback Between Rounds

After each round, a **Round Feedback Dialog** appears showing:
- Summary of findings from each agent
- A configurable countdown timer (default: 5 minutes)

You have three options:
- **Provide guidance**: Type instructions to steer the next round (e.g., "Focus on the Wnt signaling pathway" or "Compare macrophage subtypes between groups")
- **Upload additional data**: Add more files targeted to specific agents
- **Skip / Auto-continue**: Let the countdown expire or click "Continue" to proceed with automatic analysis

Providing input or uploading files on the final round automatically extends the analysis by one additional round.

### 7. View Results

Switch to the **Results** panel to see:

- **Round-by-round findings** from each agent
- **Integrated report** summarizing the overall analysis
- **Publication figures** generated by the Bioinformatics Agent — with PI review status and download buttons (PNG/PDF)
- **Graphic abstract** — an AI-generated visual summary of the research

### 8. Generate Detailed Report

In the **Report** panel, click **"Generate Report"** to create a comprehensive HTML report:

- PI Agent first reviews all analysis notebooks and figures
- The system then generates a structured report with 9 chapters covering all analysis aspects
- Each chapter includes relevant figures, data tables, and narrative text
- The report appendix documents project settings, agent configurations, uploaded data, and analysis history

Download the complete report including all figures and tables via the download button.

### 9. Chat with PI Agent

The **Chat** panel provides a free-form conversational interface:

- Ask questions about the analysis results (e.g., "What are the most significant findings in the macrophage population?")
- Request additional analysis (e.g., "Generate a volcano plot comparing group A vs group B for hepatocytes")
- Upload images or data files for on-the-fly analysis — you can target files to specific agents
- PI can display figures from the analysis or generate new ones in real-time
- For complex questions, PI automatically delegates to specialized agents and synthesizes their responses

### 10. Customize Agent Prompts

In the **Prompts** panel, you can customize each agent's behavior:

- Modify the system prompt for any agent to adjust its focus
- Configure which LLM model each agent uses
- Adjust temperature settings for more creative or deterministic outputs
- Set literature search parameters (mode, max search rounds)

Common customizations:
- Adjust Pathology Agent's focus for specific organs or staining methods
- Add domain-specific terminology for Bioinformatics Agent
- Change Literature Agent search strategy for your research area

### 11. Session Management

**Pause and Resume**
- Click "Pause" to stop analysis at any time — all state is preserved
- Click "Resume" to continue from where you left off; the Jupyter environment is fully restored

**Export and Import**
- **Export**: Click the export button in the sidebar to download your session as a compressed archive. This includes all data, analysis state, figures, and reports
- **Import**: Click "Import Session" in the sidebar to restore a previously exported session

**Session Browser**
- All sessions are listed in the sidebar with status indicators
- Click any session to view or resume it
- Session name and description can be edited directly in the sidebar

## Supported Data Formats

| Data Type | Formats | Notes |
|-----------|---------|-------|
| Pathology images | JPEG, PNG, TIFF, SVG | Large images auto-split to 1024px tiles |
| Sequencing data | .h5ad (AnnData), .loom | scRNA-seq, snRNA-seq, bulk RNA-seq |
| Biochemistry data | Manual entry, CSV/Excel, images | Indicator values with units and reference ranges |
| Session archives | .tar.gz, .zip, .json | For session import/export |

## Notes

1. Sequencing data analysis may take a while depending on dataset size — please be patient
2. Deep Research mode may take up to 60 minutes per query and costs ~$2-5 per task
3. All code executions are automatically saved as Jupyter Notebooks (.ipynb)
4. When pausing/resuming analysis, the complete Python environment is automatically restored
5. Network interruptions are handled gracefully — the interface automatically reconnects and restores message history

## Changelog

### v0.13.3 (2026-03-12)

- **Annotation Context Panel**: The cell annotation confirmation dialog now shows a left-side "Analysis Context" panel with a structured markdown summary covering experimental design, biochemistry/pathology findings, previous round highlights, and recent literature searches — giving context for reviewing cluster annotations.
- **`analysis_summary` in Annotation State**: Added `analysis_summary` field to `AnnotationConfirmationState`; populated by new `_build_annotation_context_report()` in `SingleCellAnalysisWorkflow`; stored in `_current_graph_state` (updated at round initialization and bioinformatics agent entry).
- **Exploration Report Link in Results Panel**: Results panel now shows a "View Exploration Report" button linking to the standalone exploration report HTML when available.
- **`exploration_report_url` in Session Load API**: `/api/session/{id}` now returns `exploration_report_url` when an exploration report HTML exists; `useApi.ts` maps it to `explorationReportUrl` in session state.
- **Exploration Context in Report Prompts**: Both the bioinformatics summary and the full report generation prompts in `report_generator.py` now include Phase 1 exploration direction candidates and user-chosen directions, so LLM-generated report sections properly reference the two-phase analysis flow.
- **Annotation Dialog Wider & More Markers**: Dialog widened to `w-[1200px] max-w-[90vw]`; key markers shown per cluster increased from 5 to 10.

### v0.13.2 (2026-03-12)

- **Checkpoint Directory Auto-Creation**: Fixed a crash when the session states directory does not exist — `states_dir.mkdir(parents=True, exist_ok=True)` is now called before creating the SQLite checkpoint file in `SingleCellAnalysisWorkflow`.

### v0.13.1 (2026-03-12)

- **Multi-Direction Research Workflow**: Users can now select multiple research directions from the Exploration Report. Each direction is processed in a dedicated innovation round sequentially (auto-continuing between rounds without waiting for user input). After all direction rounds complete, PI automatically performs a **Multi-Direction Correlation Analysis** that synthesizes findings across all directions.
- **Enhanced Direction Candidate Evaluation**: Exploration report direction candidates now include three scored evaluation fields: `expected_innovation` (how unexplored the area is), `expected_novelty` (how surprising the finding is), and `data_reliability` (confidence in supporting evidence) — each with a High/Medium/Low rating and brief explanation.
- **Redesigned Exploration Report Dialog**: The direction selection UI now supports multi-select of AI-suggested candidates (with per-direction notes), custom direction entries (with notes), score badges for innovation/novelty/reliability, and a built-in countdown timer that auto-triggers PI selection on expiry. A link to the standalone Exploration Report HTML is included.
- **Exploration Report HTML Export**: Added `generate_exploration_report_html()` in `report_generator.py` to produce a standalone HTML exploration report (Report A) with no LLM calls. Accessible via new endpoint `GET /api/session/{id}/exploration-report/html`.
- **PI Direction Selection Revamp**: Replaced `innovation_instruction_pi_directed()` with `select_best_directions()` — PI now autonomously picks 2 best directions (with notes and rationale) using the multi-direction queue, so 2 sequential innovation rounds run when PI decides.
- **Innovation Instruction Notes**: `innovation_instruction()` now accepts an optional `notes` parameter, allowing user-provided annotations to be injected into the PI's innovation planning prompt.
- **Bioinformatics Agent LLM Error Propagation**: LLM call failures are now tracked separately (`llm_error` field) and result in `status = "error"` (distinct from `"max_steps_reached"`), with `success: false` and the error string propagated to the workflow.
- **Session Eviction Race Fix**: Fixed a race condition in `cleanup_stale_sessions()` where a session accessed during eviction I/O could be incorrectly evicted. Added access-time snapshot comparison to abort eviction if the session was modified concurrently.
- **Sidebar Max Rounds Control Removed**: Removed the manual "Analysis Rounds" slider/input from the sidebar. Max rounds is now set dynamically based on the number of selected research directions (1 exploration round + N direction rounds).
- **New WebSocket Message Type**: Added `direction_correlation` message type, rendered in the message stream with direction badges and a formatted correlation analysis block.

### v0.13.0 (2026-03-12)

- **Two-Phase Analysis Workflow**: Introduced a structured two-phase analysis workflow. Round 1 is now an "Exploration Phase" that comprehensively maps the full cellular landscape. After Round 1, an Exploration Report is generated summarizing cluster overview, data quality warnings, strongest heterogeneity signals, key pathway changes, abnormal markers, and AI-assessed research direction candidates. Users choose a direction (from AI suggestions or a custom input) before entering the "Innovation Phase" (Round 2+) for focused, hypothesis-driven analysis.
- **Cell Annotation Confirmation Dialog**: During the Exploration Phase, the Bioinformatics Agent pauses after completing cluster annotation to present annotations for user review. A new dialog shows each cluster's cell type, confidence level (high/medium/low), and key marker genes. Low-confidence clusters require manual confirmation or correction before analysis continues. High-confidence-only results auto-confirm after a countdown timer.
- **Exploration Report Dialog**: A new dialog presented at the end of Phase 1 shows structured exploration findings and offers research direction candidates with rationale, expected innovation value, and data reliability notes. Users can select a suggested direction, enter a custom direction, or let the PI Agent decide automatically.
- **New Backend State**: Added `analysis_phase` ("exploration"/"innovation"), `annotation_state`, `exploration_report`, `user_chosen_direction`, and `current_round_direction` fields to `AnalysisState`. New `WAITING_ANNOTATION` task status for annotation confirmation pauses. New data models: `ClusterAnnotation`, `AnnotationConfirmationState`, `ExplorationReport`.

### v0.12.4 (2026-03-11)

- **Session-Centric Data Organization**: Restructured `intesc_data/` directory from type-based layout to session-centric layout. All per-session files (workspace, reports, uploads, states, LLM logs) are now co-located under `intesc_data/sessions/{session_id}/`, making multi-session management clean and predictable. Session deletion is now a single directory removal.
- **Backward Compatibility**: The system automatically falls back to legacy paths when new-structure files are not found, ensuring existing sessions continue to work without migration.
- **Migration Script**: Added `backend/migrate_to_session_dirs.py` to migrate existing data from the old layout to the new session-centric layout. Supports `--dry-run` mode for safe preview.

### v0.12.3 (2026-03-10)

- **Improved Report Markdown Rendering**: Enhanced `markdown_to_html()` across all three report generators (`report_generator.py`, `supervisor_agent.py`, `regenerate_report_html.py`) to convert markdown headers (`#`/`##`/`###`/`####`) to proper HTML heading tags, and added bullet list support. Previously, markdown headers appeared as raw text in HTML reports.
- **Fixed Conclusions Formatting**: Updated LLM prompts for abstract and conclusions generation to prevent redundant section headers (e.g., `### 3. Conclusions`). Added defensive post-processing to strip any remaining header artifacts from LLM output.
- **Frontend Markdown Rendering**: ReportPanel Detail view now uses ReactMarkdown (already installed) for rendering abstract, section content, and conclusions, replacing raw `whitespace-pre-wrap` text display.
- **Frontend Appendix Display**: Added appendix section to Report Detail view showing project information, agent model configurations, data files, treatment groups, and user instructions — data already existed in backend but was not rendered in the frontend.
- **Supervisor Agent HTML Appendix**: Supervisor Agent's HTML report now includes a full appendix section (project info, agent config, data files, treatment groups, analysis rounds), matching the depth of the main report generator.

### v0.12.2 (2026-03-10)

- **Analysis Stop/Pause Support**: Added UI buttons and backend endpoint (`POST /api/session/{id}/stop`) to pause or terminate analysis. Support immediate termination (pause/save) or termination with summary (stop-and-summarize report generation).
- **Publication-Quality Bioinformatics Figures**: Added mandatory checklist (UMAP axis labels, font sizes, legend placement) and strict composition barplot rules to bioinformatics agent's system prompt. Updated default matplotlib/seaborn themes in sandbox for professional "publication" aesthetics.
- **Figure Revision Auto-Retry**: Bioinformatics Agent now automatically attempts to fix runtime errors during figure code revision once using the error message as context.
- **Progressive Figure Review Leniency**: PI Agent's figure review now employs progressive leniency (from revision 2+) to focus on scientific accuracy over minor aesthetics, avoiding infinite revision loops.
- **WebSocket Reconnection Improvements**: WebSocket handler now updates the global reference for active workflows, ensuring users can reliably observe real-time progress after a connection drop.
- **UI UX Polish**: Feedback dialog "Skip" button renamed to "Continue"; added Stop/Summarize buttons to feedback dialog for easier control.
- **Code Stability**: Fixed WebSocket handling for completed sessions; improved LLM retry logic; centralized code safety validation in sandbox.

### v0.12.1 (2026-03-10)

- Centralized code safety validation: moved `DANGEROUS_PATTERNS` and `validate_code_safety()` to `sandbox.py`, enforced at sandbox level so all callers (bioinformatics agent, report generator, chat sandbox) are protected
- Extracted `CheckpointManager` to standalone `backend/utils/checkpoint.py` module, now shared by bioinformatics analysis and report generation
- Added checkpoint-based resumption for report generation: interrupted reports (e.g. API rate limit) resume from last completed step instead of restarting
- Fixed session save race condition: `async_save_session()` serializes JSON on the event-loop thread before writing in thread pool, preventing concurrent mutation during serialization
- Fixed session eviction race: cleanup task now re-checks if session was re-activated before evicting, and properly closes Jupyter kernels and chat sandboxes for evicted sessions
- Fixed completed session WebSocket handling: multiple clients can now connect to completed sessions without overwriting global WebSocket reference
- Added `MAX_REALTIME_MESSAGES = 1000` cap to prevent unbounded memory growth from realtime messages
- Strip base64 data from figure messages before persisting to realtime_messages to reduce memory usage
- Added stale import job cleanup (1-hour TTL) to prevent memory leak from completed/failed import jobs
- Added `rollback_last_turn()` method to BaseAgent, replacing duplicated inline rollback patterns across PI Agent
- Changed output truncation strategy from head (first N chars) to tail (last N chars) across bioinformatics agent, supervisor agent, and analysis log — more useful for debugging as recent output is preserved
- Supervisor Agent: added `system_prompt` parameter for user-customizable monitoring prompts; added explicit "Execution status" field and instruction to ignore truncation artifacts
- Added `VALID_AGENTS` Literal type for API endpoint validation (prompt/model/temperature updates)
- Added shared `extract_code_from_markdown()` utility with state-machine parsing to sandbox.py
- Fixed bioinformatics checkpoint resume budget: uses `original_total` to avoid budget inflation on repeated resumes
- Fixed bioinformatics checkpoint clearing: only clears on explicit completion via `report_findings`, preserving checkpoint for `max_steps_reached` and `completed_with_fallback` to allow resumption
- Reset bioinformatics per-run state (code_history, analysis_context, last_result) at start of fresh runs to prevent cross-round pollution
- Added graceful error handling for figure revision failures — keeps current version instead of crashing
- Added `supervisor_feedback`, `supervisor_checkpoint`, `supervisor_summary` to WebSocket message types in schema
- Supervisor Agent now appears in Prompts panel with model/temperature configuration (Eye icon, cyan theme)
- Added sandbox-level code safety check with `skip_safety_check` flag for trusted internal code (kernel init, session restore/save)
- Workflow now passes user-configured supervisor prompt to SupervisorAgent
- Grab workflow reference under lock in pause/resume endpoints to avoid race with cleanup task

### v0.12.0 (2026-03-10)

- Added Supervisor Agent: real-time monitoring of bioinformatics analysis steps with issue detection, biological checkpoint identification, and PI escalation
- Fixed Supervisor Agent settings never syncing to backend — model, prompt, and temperature are now included in pre-analysis sync and GET API endpoints
- Fixed LLM retry logic: replaced overly broad `APIError` with `InternalServerError` in retryable exceptions, preventing unnecessary retries on authentication failures and bad requests
- Fixed settings sync failures being silently ignored — failures are now collected and displayed as error messages in the message stream so users are informed
- Fixed round feedback dialog potentially triggering multiple times for the same round by adding a ref-based guard
- Fixed WebSocket `state_sync` version conflict having no recovery — outdated versions now trigger a full session state reload from the server
- Fixed `crypto.randomUUID()` usage that could crash in non-HTTPS contexts — replaced with a browser-safe UUID generator
- Fixed WebSocket `connect()` using stale closure references for session ID — now reads fresh state from the store
- Improved pre-analysis settings sync: all agent models/prompts/temperatures and remaining config now sync in parallel via `Promise.allSettled()`, reducing startup latency
- Improved last-round UX: the waiting-for-input message now indicates it is the final round and that users can provide data to extend analysis
- Added expand/collapse buttons to truncated message content (prompts, responses, outputs, literature) so users can view full text
- Added localStorage hydration validation to prevent crashes from corrupted or outdated persisted state
- Added `max_length=50000` validation for custom prompt updates to prevent oversized payloads
- Added automatic cleanup of partial files when uploads fail mid-stream
- Added `painter_prompt` and `supervisor_prompt` fields to backend `AgentPrompts` config for API consistency
- Fixed Chinese translation for Painter Agent name

### v0.11.1 (2026-03-09)

- Fixed duplicate publication figures: bioinformatics agent now upserts figures by name instead of always appending, with additional dedup in workflow sync
- Fixed session cleanup: deleting a session now also removes workspace, reports, and state files from disk
- Fixed session eviction: initializing sessions are no longer evicted by the cleanup task
- Fixed literature model config: get-models endpoint now correctly reads `literature_model` instead of `pi_model`
- Fixed hardcoded model names: report generation and chat now use configured model from settings instead of hardcoded defaults
- Improved session persistence: status changes (start, pause, resume) are now immediately saved to disk
- Improved WebSocket reconnection: message buffer now strips base64 data to reduce memory usage; history send moved outside lock to avoid I/O blocking
- Improved LLM retry logic: `RateLimitError` is now caught before generic `APIError` to ensure proper backoff handling
- Improved session restore detection: sandbox now verifies restore output instead of relying solely on exit code
- Improved markdown code fence stripping in literature gap analysis JSON parsing
- Added `generating_report` as a valid session status in the frontend

### v0.11.0 (2026-03-09)

- Changed agent execution order to biochemistry → pathology → bioinformatics → literature. Biochemistry now runs first so its results inform downstream agents
- Added dynamic PI planning: PI Agent uses `plan_agent_tasks()` to refine each downstream agent's tasks based on already-completed results within the same round, building a coherent chain of thought
- Enhanced cross-agent context: biochemistry findings are now injected into pathology and bioinformatics agent prompts for cross-referencing
- PI resets message history at the start of each round's planning phase; dynamic planning calls accumulate context within the round
- Round 2+ planning now includes previous round PI summary as strategic overview for dynamic task refinement
- Literature topics are dynamically refined based on all completed agent results before search

### v0.10.3 (2026-03-09)

- Added comprehensive analysis log: every bioinformatics agent step (code execution, PI consultation, literature search, cluster annotation, figure generation) is recorded with agent reasoning, timestamps, and results
- Reports now include a full chronological analysis log timeline with collapsible code, outputs, and inline figures
- Added scientific interpretation sections for publication figures and tables in reports
- PI Agent resets message history before synthesis and final summary, reducing token waste
- Bioinformatics Agent: capped code history at 100 entries to prevent memory growth; figure attempt counter resets between rounds for retry
- Workflow auto-activates agents when PI assigns tasks but forgets to include them in `agents_for_next_round`
- Pathology findings now shared with bioinformatics agent for cross-agent collaboration within the same round
- Code history deduplication by ID to handle node retries correctly
- Refactored deep research availability check; improved PubMed gap fallback mode switching

### v0.10.2 (2026-03-08)

- Fixed critical bug: sandbox kernel now fully recovers after timeout (reconnects client, re-initializes environment, restores session variables)
- Figure plotting is now exception-safe: `plt.close`/`plt.show` are temporarily disabled during execution and guaranteed to restore via `try/finally`
- Added blank figure detection for publication figures (rejects files < 2KB)
- Increased sandbox execution timeout from 5 minutes to 30 minutes for long-running analyses
- Updated architecture documentation to reflect current sequential agent execution model

### v0.10.1 (2026-03-07)

- Literature agent quick search now uses PI model as fallback, avoiding expensive deep research model costs
- Fixed deep research mode max_tokens handling for better API compatibility
- Added cancel button for upload queue items (abort in-progress or remove pending uploads)

### v0.10.0 (2026-03-07)

- Added Deep Research mode for literature agent: supports OpenAI o4-mini-deep-research, Perplexity sonar-deep-research, and other deep research models via OpenRouter
- Three literature search modes: Deep Research Models (default), Gemini Deep Research, and PubMed + LLM
- Upgraded bioinformatics agent default model to anthropic/claude-sonnet-4.6
- Improved cross-agent collaboration: agents now share findings within the same round for better-informed analysis
- Improved literature search: smarter PubMed queries with automatic retry when initial search returns no results
- Improved PI planning: more focused and actionable literature search topics
- Fixed a bug where some agents could be incorrectly skipped in multi-round analysis
- Fixed round summary missing literature findings
- Various stability and performance improvements

### v0.9.0 (2026-03-04)

- PI now reviews actual analysis notebooks and figures before writing reports, producing more grounded narratives
- Reports include an appendix with project info, agent settings, data summary, and analysis history
- Graphic abstracts go through a revision process with PI feedback for higher quality
- Session name and description can be edited directly in the sidebar
- Export now shows file size before download
- Upload queue supports multiple files with individual progress bars
- Added support for bulk RNA-seq and custom sequencing formats
- Deep Research cost warning displayed in settings

### v0.8.9 (2026-03-03)

- Improved session import reliability behind reverse proxies
- Better import progress feedback with success/failure indicators

### v0.8.7 (2026-03-02)

- Fixed page scrolling issues in the Data panel
- Added Chinese/English switching for footer text

### v0.8.6 (2026-03-02)

- Large session imports now run in the background to avoid timeouts
- Added first-visit disclaimer dialog about AI limitations

### v0.8.5 (2026-02-28)

- AI-curated figure selection in brief reports — only key results are shown
- Improved session export with browser-native download

### v0.8.4 (2026-02-21)

- Added About dialog with project information

### v0.8.3 (2026-02-21)

- Fixed round wait timer not respecting manual mode setting

### v0.8.2 (2026-02-21)

- Upgraded default models for improved analysis quality
- Fixed proxy configuration for API calls

### v0.8.1 (2026-02-20)

- Fixed publication figures referencing non-existent treatment groups
- Bioinformatics Agent can now see its own figures during revision for more accurate corrections

### v0.8.0 (2026-02-16)

- New treatment group system: define experimental groups with sample counts, assign all data to groups
- Unified data upload interface with group/sample assignment
- Complete redesign of the Data panel

### v0.7.5 (2026-02-15)

- PI Agent now decides which agents to activate each round, saving time and API costs

### v0.7.4 (2026-02-14)

- Figure generation no longer counts toward analysis step budget
- All uploaded data can now be downloaded from the Data panel

### v0.7.3 (2026-02-13)

- PI can delegate chat questions to specialized agents for in-depth answers
- Report generation rewritten for better reliability
- Sessions interrupted during report generation now resume correctly
- Auto PDF generation for all figures (300 DPI, vectorized fonts)
- Automatic batch effect detection for multi-sample datasets
- Configurable round wait timer

### v0.7.2 (2026-02-11)

- New Chat panel for free-form conversation with PI Agent
- Session export/import support

### v0.6.x (2026-02-09 – 2026-02-11)

- Literature-based cluster annotation
- Human-in-the-loop round feedback dialog
- Session recovery on page refresh
- Two literature search modes with automatic fallback

### v0.5.x (2026-02-06 – 2026-02-08)

- Biochemistry Agent rebuilt with 3-step analysis workflow
- Real-time message optimization

### v0.3.0 – v0.4.x (2026-02-04 – 2026-02-05)

- Painter Agent for graphic abstract generation
- Report generation system
- Parallel agent execution

## License

MIT License
