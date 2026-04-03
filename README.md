# ScAI - Single Cell Sequencing Analysis Driven by AI

**ScAI** is a multi-agent AI system for comprehensive biomedical research data analysis. It brings together pathology image analysis, single-cell sequencing data, biochemical indicators, and literature research into a unified, iterative workflow — powered by seven specialized AI agents that collaborate like a research team.

**Version**: 0.18.13

## Installation

### Option A: Using Docker (Recommended)

Pull and run the pre-built image from GitHub Container Registry:

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v ./scai-dep:/dep \
  -v ./scai-data:/data \
  ghcr.io/minzhangcheng/scai:latest
```

View logs:

```bash
docker logs -f scai
# or check the log file inside the data volume:
# docker exec scai cat /data/logs/scai.log
```

- `/dep` — stores Python venv and Node.js dependencies (installed automatically on first run, persisted across restarts)
- `/data` — stores session data, uploads, reports, and logs

To build the image locally:

```bash
docker build -t ghcr.io/minzhangcheng/scai:latest .
```

Access **http://localhost:3000** to open the web interface.

### Option B: Using Startup Script

```bash
git clone <repository-url>
cd scai
./start.sh
```

You can customize the host, port, and backend port:

```bash
./start.sh --host 0.0.0.0 --port 3000 --backport 8000
```

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Listen address | `127.0.0.1` |
| `--port` | Frontend port | `3000` |
| `--backport` | Backend port | `8000` |

This will automatically create a virtual environment, install dependencies, and start both backend and frontend.

To stop:
```bash
./stop.sh
```

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

Seven steps run sequentially to build a complete, unbiased picture of your data:

1. **Round Initialization** — Pending user inputs are processed, round counter increments, status sets to RUNNING
2. **PI Planning** — PI Agent generates an unbiased, comprehensive analysis plan with specific tasks for each agent
3. **Biochemistry Analysis** — Extracts and interprets biochemical indicators from structured data and uploaded images
4. **Pathology Analysis** — Tiles histology images, runs visual analysis, synthesizes findings cross-referenced with biochemistry results
5. **Bioinformatics Analysis** — Autonomous Jupyter-based single-cell analysis (up to 200 steps):
   - QC → Normalization → Clustering → Cell Annotation
   - **Cell Annotation Confirmation** — Analysis pauses here. A dialog shows each cluster's cell type, confidence level (high/medium/low), and key marker genes. Low-confidence clusters require manual confirmation or correction. High-confidence results auto-confirm after a countdown
   - DEG → GO/KEGG Pathway Enrichment → Cell-Cell Communication
6. **Literature Research** — Searches and synthesizes scientific literature (Deep Research, PubMed+LLM, or quick search mode)
7. **PI Synthesis** — PI Agent synthesizes findings from all agents, generates a structured **Exploration Report** covering cluster overview, data quality warnings, heterogeneity findings, pathway changes, and research direction candidates. A dialog appears for you to select a direction for Phase 2 (AI-suggested candidates with innovation/novelty/reliability scores, a custom direction, or let PI decide automatically)

**Phase 2 — Innovation (Round 2+)**

Following rounds focus on the selected research direction for deeper, hypothesis-driven analysis:

- **Direction Modes**: Single direction, multiple directions (processed sequentially via an automatic queue), or PI-selected best directions
- **Selective Agent Activation**: PI decides which agents run each round — saving time and API costs. Bioinformatics always runs
- **Multi-Direction Correlation**: After all direction rounds complete, PI automatically synthesizes findings across directions
- **Convergence Detection**: Analysis stops automatically after 2 consecutive rounds with no new findings
- **Feedback Loop**: After each round, provide text guidance, upload new data, or let the countdown auto-continue

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

### v0.18.13 (2026-04-03)

- **Improvement: Docker PyPI mirror support** — Docker startup now supports `-e PIP_INDEX_URL` parameter to pass a PyPI mirror URL, significantly speeding up Python package installation in containers.

### v0.18.12 (2026-04-03)

- **Fix: Bioinformatics error handling** — Added a hard limit on code execution errors to prevent analysis from getting stuck in infinite error loops.
- **Improvement: Painter image generation** — Improved compatibility with more image generation API providers.
- **Improvement: PI annotation parsing** — Enhanced cell type annotation JSON parsing reliability.
- **Improvement: Selective agent activation** — From round 2 onwards, PI automatically determines which agents need to run, avoiding unnecessary runs.
- **Improvement: Dev/prod mode** — Added development/production mode switching; startup scripts support `--dev` and `--prod` flags.

### v0.18.10 (2026-04-01)

- **Fix: LLM temperature handling for kimi-k2.5** — Models that prohibit custom temperature now omit the parameter entirely instead of failing.
- **Fix: max_tokens for o-series models** — `max_completion_tokens` is now correctly used instead of `max_tokens` for OpenAI o-series reasoning models.
- **Improvement: Zhipu AI support** — Painter Agent now supports Zhipu AI image generation.
- **Improvement: Literature fallback LLM client** — Literature Agent uses the correct LLM endpoint when calling fallback models.
- **Improvement: Agent LLM client caching** — Workflow caches per-agent LLM clients to avoid creating duplicates.

### v0.18.6 (2026-03-30)

- **Fix: Annotation snapshot timing** — Auto-snapshot before annotation HITL is now created after `annotation_state` is set, so rollback restores the dialog data needed by the frontend instead of leaving it empty.
- **Fix: Pre-annotation snapshot backward compatibility** — Restoring a legacy `pre_annotation` snapshot that lacks `annotation_state` now falls back to PAUSED status, preventing the workflow from hanging in HITL-hold with no dialog data.
- **Fix: PI Agent constructor** — Removed erroneous `session_id` keyword argument from PI Agent construction that could cause a TypeError.
- **Improvement: Multimodal content text-first ordering** — Vision messages now place the text prompt before image attachments, following OpenRouter's recommended ordering for better model comprehension.
- **Improvement: Painter Google AI Studio detection** — Extracted `_GOOGLE_AI_STUDIO_BASE_URL` constant; Google AI Studio endpoints now skip the chat modalities strategy entirely (Strategy 1) and go directly to the Images API, avoiding unnecessary failed requests.
- **Improvement: Supervisor error tracking uses traceback tail** — `_recent_errors` now keeps the last 300 characters (tail) of each traceback instead of the first 200, capturing the Python error type that appears at the end of tracebacks for more accurate stuck detection.
- **Improvement: Deep Research configurable model** — `GoogleDeepResearchClient` now accepts an optional `model` parameter (defaults to `deep-research-pro-preview-12-2025`), allowing model selection without code changes.

### v0.18.5 (2026-03-30)

- **Improvement: `safe_parse_json` extracted to shared utility** — Duplicate JSON-extraction logic in `BaseAgent._safe_parse_json` and `report_generator._safe_parse_json` now delegates to a single `backend/utils/formatting.safe_parse_json` implementation, eliminating ~60 lines of duplicated bracket-matching code.
- **Improvement: Supervisor stuck detection broadened** — "Stuck" condition now triggers when the same error type appears 3 or more times across recent error history (in addition to the existing two-consecutive check), ensuring the agent escalates to PI even when identical errors are not strictly back-to-back.
- **Improvement: Supervisor PI escalation on stuck technical errors** — Technical errors (KeyError, TypeError, etc.) and LLM-classified technical issues now also escalate to PI when the agent is stuck, overriding the previous unconditional skip that could leave the analysis frozen.
- **Fix: Supervisor null response guard** — `response.content or ""` replaced with `response.content if response else ""` to prevent AttributeError when the LLM client returns None.
- **Fix: Biochemistry invalid value skip** — Non-finite float values (NaN, Inf) returned by the LLM are now skipped entirely rather than silently coerced to 0.0, preventing corrupt indicator records from propagating into downstream analysis.
- **Improvement: Annotation cache size cap** — `LiteratureAgent._annotation_cache` is now capped at 100 entries; the oldest entry is evicted when the limit is reached, preventing unbounded memory growth in long multi-round sessions.
- **Fix: DashScope painter revision text-only prompt** — When revising a graphic abstract via DashScope's native image API, the "look at attached image" instruction is stripped from the revision prompt, removing a misleading reference that the text-only endpoint cannot honour.
- **Fix: `review_status` serialization None guard** — `AnalysisState.to_api_dict()` now guards `review_status` with `if fig.review_status` / `if tbl.review_status` before calling `_enum()`, preventing AttributeError when the field is None.
- **Improvement: WebSocket reconnect deduplication** — Removed competing auto-reconnect `useEffect` from `App.tsx`; all reconnection logic is now consolidated in the `useWebSocket` hook, eliminating race conditions between two independent reconnect loops.
- **Fix: WebSocket reconnect timer cleanup** — `disconnect()` now cancels any pending reconnect `setTimeout` via a tracked `reconnectTimerRef`, preventing ghost reconnect attempts after manual disconnection.
- **Fix: Persisted `chatMessages` TTL eviction** — `chatMessages` is now cleared alongside `messages` and `currentSession` when the persistence TTL expires, preventing stale chat history from surfacing in new sessions after a browser refresh.

### v0.18.4 (2026-03-30)

- **Improvement: DashScope native image generation** — Painter Agent now detects DashScope base URLs and calls the native async task API (`/api/v1/services/aigc/text2image/image-synthesis` → poll `/api/v1/tasks/{task_id}`) directly, bypassing the OpenAI compatibility layer which does not expose image endpoints.
- **Improvement: Snapshot kernel session backup** — Snapshots now copy the Jupyter kernel session pkl file alongside the state JSON, enabling full kernel state rollback on restore. Export (tar.gz) includes the snapshots directory. Delete and cleanup operations remove pkl backups.
- **Improvement: Vision auto-disable on failure** — Bioinformatics Agent and PI Agent now set `has_vision = False` when a vision API call fails (including ZhipuAI error 1210), preventing repeated failures for the rest of the session.
- **Improvement: Pre-round-wait auto-snapshot** — Workflow creates an automatic snapshot before each round wait HITL, preserving state at the exact moment the user is prompted to continue.
- **Improvement: Supplementary analysis in round wait loop** — Round wait loop now handles supplementary analysis requests, running them inline and resuming the wait state afterward.
- **Improvement: Multi-direction auto-skip removed** — Directions queue no longer auto-continues rounds; all rounds now follow normal wait logic, giving the user control at each round boundary.
- **Improvement: Annotation context report redesign** — Per-step rendering with figures inlined alongside step descriptions; "Additional Figures" section with dynamic numbering replaces the flat figures grid; improved CSS for step layout and expandable raw output.
- **Improvement: Analysis complete snapshot shortcut** — "Analysis Complete" banner in AnalysisPanel now shows a Snapshot button alongside the green badge, making it easy to capture the final state.
- **Fix: Final summary card contrast** — `finalSummary` card background changed from `accent-500/600` to `accent-50/100` for improved text readability.

### v0.18.3 (2026-03-29)

- **Improvement: PI-guided cluster annotation** — PI Agent now generates the PubMed search query and evaluates search results for cell-type annotation, leveraging the stronger PI model instead of the Literature Agent's weaker LLM. In the final matching step, PI also maps cluster markers to cell types directly, replacing the Literature Agent's LLM matching.
- **Improvement: Supervisor LLM issue-type classification** — Supervisor prompt now returns an `issue_type` field (`technical`/`biological`/`ambiguous`). Technical issues (e.g. blank plot from wrong column name, logic errors with no exception) are skipped for PI escalation without relying solely on Python exception inspection.
- **Improvement: Supervisor-as-sole-reviewer degradation** — When PI and bioinformatics agents both lack vision, the supervisor uses a relaxed approval threshold (score ≥ 5 instead of 7). After 2 consecutive rejections of the same figure, `requires_reanalysis` is set to break infinite revision loops.
- **Improvement: Supervisor feedback deduplication** — Bioinformatics Agent skips re-injecting identical supervisor feedback (same first 120 chars) within a round, reducing wasted tokens on repeated warnings.
- **Improvement: Figure revision stall detection** — If the PI quality score does not improve across two consecutive revisions, the figure is accepted with reservations and the agent moves on, preventing infinite revision loops.
- **Improvement: LLM retry notifications via WebSocket** — `retry_with_backoff` now accepts an `on_retry` callback; all agent LLM clients are wired with a notifier that pushes status messages to the WebSocket during retry waits, preventing apparent dead silence.
- **Improvement: Message buffer replaced with realtime_messages** — The in-memory `_message_buffers` dict (capped at 100 entries) is removed; WebSocket observers now receive the full `state.realtime_messages` history (up to 500 entries) via `_serialize_realtime_messages()`, eliminating the separate buffer and its lock.
- **Improvement: Checkpoint write deduplication** — `CheckpointManager` tracks the last saved `step_name` and skips redundant writes, reducing I/O during figure-revision loops that revisit the same step.
- **Improvement: Pathway enrichment library fallback order** — `config.py` now documents the preferred gene set library sequence for gseapy/Enrichr (GO_Biological_Process_2023 → Reactome_2022 → KEGG_2021_Human), with instructions to try each in order on failure.

### v0.18.2 (2026-03-29)

- **Improvement: Fine-grained session concurrency** — Per-session mutex locks replace the single global lock, eliminating TOCTTOU race windows and enabling truly parallel operations across sessions.
- **Improvement: LLM client rate-limit handling** — Retry count raised to 4 with fixed escalating delays (5/10/20/40 s) for 429 errors; reasoning models (OpenAI o-series, Gemini 3.x) now receive correct temperature parameters automatically.
- **Improvement: Sandbox AST security layer** — A second AST-based inspection pass detects reflection-based sandbox escapes (`getattr`, `__globals__`, `__subclasses__`, etc.) that regex patterns miss.
- **Improvement: Centralised download utilities** — A new `frontend/src/utils/download.ts` module de-duplicates blob-download logic across ReportPanel, ResultsPanel, and DownloadButtons; parallel `Promise.all` download added to DownloadButtons.
- **Improvement: Report CSS and image utils extraction** — Report stylesheet moved to `backend/utils/report_css.py` and image helpers to `backend/utils/image_utils.py`, enabling reuse between report generators.
- **Improvement: Annotation workflow guard** — The bioinformatics agent now blocks direct `adata.obs['cell_type']` assignment, enforcing the `annotate_clusters` workflow and preventing silent annotation overwrites.
- **Fix: Annotation recovery after restart** — Restoring a snapshot with a previously confirmed annotation no longer re-blocks on confirmation; overrides are applied directly.
- **Improvement: Upload concurrency limit** — Per-session upload slots capped at 5; excess requests are rejected early with a clear error before any file I/O begins.
- **Improvement: Centralised AnalysisState serialisation** — `AnalysisState.to_api_dict()` replaces scattered camelCase conversion logic across endpoints.
- **Improvement: Frontend colour system** — Tailwind palette updated to Material Design red (primary) and modern blue (accent); Sidebar migrated to light theme for improved contrast.
- **Improvement: Upload file extension blocklist** — DataPanel now rejects executable and script file types (.exe, .sh, .dll, etc.) client-side before upload begins.

### v0.18.1 (2026-03-27)

- Version bump and documentation sync.

### v0.18.0 (2026-03-27)

- **Improvement: Snapshot tree view** — Snapshots now display as a tree with parent-child relationships and branch indicators, making it easy to visualize analysis history and restore points. Manual snapshot creation removed; snapshots are now auto-created at key analysis points only.
- **Improvement: Snapshot restore reliability** — Restoring a snapshot now properly deletes the LangGraph checkpoint and closes the DB connection first, enabling a clean fresh run from the restored state. Phase-aware status setting: restoring a pre-direction snapshot re-opens the direction selection dialog; restoring a pre-annotation snapshot re-opens the annotation dialog.
- **Fix: Annotation context report missing figures** — The annotation context report now scans the workspace directory for images directly, fixing empty figure sections when the report is generated mid-execution before the bioinformatics agent returns its analysis log.
- **Fix: ANSI escape codes in error output** — Jupyter traceback errors are now stripped of ANSI escape codes before display, producing clean readable error messages.
- **Improvement: Status badge styles** — Added missing CSS styles for `waiting_annotation`, `waiting_input`, and `generating_report` status badges.

### v0.17.5 (2026-03-26)

- **Fix: Loading spinner not animating** — Spinner icons (Loader2) were static due to `prefers-reduced-motion` CSS override killing all animations with `!important`. Fixed by exempting functional `.animate-spin` elements from the reduced-motion rule. Also standardized all inline `style={{ animation: 'spin ...' }}` to Tailwind's `animate-spin` class.
- **Improvement: Sidebar project description adaptive height** — Long project descriptions no longer push Export/Import/API Settings off-screen. The description area now dynamically uses available sidebar space and scrolls only when content exceeds it.
- **Improvement: Sidebar description readability** — Increased description font size (12px → 14px) and improved text contrast for better readability on the dark sidebar.

### v0.17.4 (2026-03-26)

- **Improvement: Frontend visual redesign ("Scientific Elegance")** — Complete UI beautification with no new dependencies. New typography (Source Serif 4 for headings, IBM Plex Sans for body), refined color system (indigo accent + surface gray scale), dark sidebar theme, underline-style tab navigation, scientific dot-grid content background, gradient agent badges in message stream, modal scale-in animations, card fade-in-up entrance animations, enhanced scrollbars, pill-shaped status badges, and `prefers-reduced-motion` accessibility support.

### v0.17.3 (2026-03-26)

- **New: Automatic API documentation lookup** — When the analysis hits the same type of API error twice in a row, the system now automatically looks up the correct API documentation from installed packages and provides corrective guidance, significantly reducing stuck loops caused by outdated API usage.
- **New: On-demand documentation tool** — Bioinformatics Agent can now proactively look up API docs for any installed package (decoupler, gseapy, scvelo, cellrank, etc.) before writing code, replacing hardcoded API rules with dynamic, always-up-to-date references.

### v0.17.2 (2026-03-26)

- **Improvement: Docker run as background service** — Container now runs in detached mode, freeing the terminal. Logs persisted to `/data/logs/scai.log` for access across restarts.

### v0.17.1 (2026-03-26)

- **New: Per-agent LLM server configuration** — Each agent can now use a different LLM provider and API key via the Prompts Panel.
- **Improvement: Painter Agent dual-path image generation** — Supports more image generation providers (OpenAI, Google AI Studio, SiliconFlow, etc.) via automatic fallback between two API strategies.
- **Improvement: Config panel simplification** — Google AI API Key moved to Literature agent settings; "LLM Server" label renamed to "Default LLM Server".

### v0.16.2 (2026-03-24)

- **New: Docker packaging** — Build and run ScAI via Docker (`ghcr.io/minzhangcheng/scai`). Dependencies install at first run and persist across restarts.

### v0.16.1 (2026-03-24)

- **Improvement**: Figure revision auto-falls back to text-only mode when the model doesn't support vision input.
- **Improvement**: Sandbox execution timeout extended to 1.5 hours for long-running analyses.

### v0.15.9 (2026-03-23)

- **Fix**: Figure revision loop execution error corrected.
- **Improvement**: API config auto-syncs from browser to backend on startup and before analysis, avoiding unnecessary config prompts after server restarts.

### v0.15.8 (2026-03-23)

- **Improvement**: Auto-generates companion PDF for manually saved figures; supervisor feedback deduplication; figure file extension auto-detection in download buttons and reports.
- **Improvement**: Clarified vision model requirements in Prompts Panel recommendations.

### v0.15.7 (2026-03-22)

- **New**: Cluster annotation must complete before exploration report — prevents premature reporting.
- **New**: Inline table support in HTML reports with row preview and download.
- **Improvement**: Smarter error recovery in exploration redirects to annotation steps instead of aborting.

### v0.14.6 (2026-03-21)

- **Improvement**: Auto-detect AnnData variable names and clustering keys — no more hardcoded assumptions.
- **Improvement**: Faster error recovery (max 3 consecutive errors); full output visibility (no more truncation); technical errors no longer escalated to PI.
- **Improvement**: Score definitions table in report; versioned figure deduplication.
- **Fix**: PubMed XML parsing; Deep Research connection retry on transient network errors.

### v0.14.5 (2026-03-20)

- **Fix**: Multiple crash and desync bugs — multi-direction queue, countdown timer, annotation callback, literature search without prior findings, correlation direction mismatch.
- **Improvement**: PI's cluster merge strategy passed to bioinformatics; per-cell-type volcano plots and pathway charts in exploration.

### v0.14.4 (2026-03-17)

- **Project renamed**: InteSc → ScAI. Directory and environment variable restructure. Configurable startup host/port.

### v0.14.3 (2026-03-17)

- **JPG figure output** with companion PDF for publication; larger plot fonts for readability.
- **Supervisor report redesign**: Figures embedded inline; report framed as scientific document; user interaction history and notebook in appendix.
- **Painter agent reliability**: Retry with exponential backoff on image generation failures.

### v0.14.2 (2026-03-16)

- **Improvement**: Automatic duplicate index repair; cached cluster annotations; graphic abstract token management; WebSocket heartbeat for connection stability.

### v0.14.1 (2026-03-15)

- **Fix**: Round feedback now immediately resumes workflow; stale convergence flag cleared; pause during annotation preserved; exploration report failure shown to user.

### v0.14.0 (2026-03-15)

- **New**: Score methodology transparency — each score's computation method, gene list, and source tracked and shown in report.
- **New**: Richer annotation dialog with alternative cell types, confidence reasons, and downloadable context report.
- **Improvement**: Jupyter sandbox security; kernel failure detection; report organized by round with phase labels.

### v0.13.0 – v0.13.4 (2026-03-12 – 2026-03-13)

- **New: Two-phase analysis** — Round 1 explores the full landscape; Exploration Report with scored direction candidates; user selects direction(s) for focused Innovation Phase.
- **New: Cell annotation confirmation** — Review, correct, or confirm each cluster's cell type before analysis continues.
- **New: Multi-direction research** — Select multiple directions; each runs a dedicated round; automatic cross-direction correlation analysis.
- **Improvement**: Annotation context panel with experimental design and literature; Exploration Report HTML export; pause/resume stability fixes.

### v0.12.0 – v0.12.4 (2026-03-10 – 2026-03-11)

- **New: Supervisor Agent** — Real-time monitoring of bioinformatics analysis steps with issue detection and PI escalation.
- **New: Pause/Stop controls** with optional summary report on stop.
- **New: Publication-quality figures** with PI review and auto-revision loop.
- **Improvement**: Session-centric data layout; report resume from last chapter; Markdown rendering in reports.

### v0.11.0 – v0.11.1 (2026-03-09)

- **Improvement**: Biochemistry runs first to inform other agents; PI dynamically refines tasks based on completed results; cross-agent context sharing within rounds.

### v0.10.0 – v0.10.3 (2026-03-07 – 2026-03-09)

- **New: Deep Research mode** for literature agent (OpenAI, Perplexity, OpenRouter deep research models).
- **New: Analysis log** with full chronological timeline and inline figures in reports.
- **Improvement**: Cross-agent collaboration; sandbox timeout recovery; figure interpretation narratives.

### v0.9.0 (2026-03-04)

- PI reviews actual notebooks and figures before writing reports.
- Reports include appendix with project info, agent settings, and analysis history.
- Graphic abstract revision with PI feedback; bulk RNA-seq support.
- Session name editing; multi-file upload queue; export file size preview.

### v0.3.0 – v0.8.9 (2026-02-04 – 2026-03-03)

- Treatment group system, unified data model, multi-group multi-sample support
- Chat panel for PI Agent conversation with delegation to specialized agents
- Painter Agent for graphic abstracts; comprehensive report generation with PI review
- Literature-based cluster annotation; human-in-the-loop round feedback
- Biochemistry Agent 3-step workflow; session export/import; auto PDF generation
- Selective agent activation from round 2+; batch effect detection

## License

Apache License 2.0
