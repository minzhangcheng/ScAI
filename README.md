# ScAI - Single Cell Sequencing Analysis Driven by AI

**ScAI** is a multi-agent AI system for comprehensive biomedical research data analysis. It brings together pathology image analysis, single-cell sequencing data, biochemical indicators, and literature research into a unified, iterative workflow — powered by eight specialized AI agents that collaborate like a research team.

**Version**: 0.20.2

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
- Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP, SVG
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

Eight specialized AI agents work together to analyze your data from different angles:

- **PI Agent** — The project coordinator. Plans analysis strategy, synthesizes results from all agents, resolves conflicts, and reviews generated figures for publication quality
- **Pathology Agent** — Analyzes histological section images (H&E and custom staining types), identifying tissue structures, cellular patterns, and pathological changes
- **Bioinformatics Agent** — Performs autonomous single-cell sequencing data analysis (scRNA-seq, snRNA-seq, bulk RNA-seq). Executes code in a local Jupyter environment, generates publication-quality figures, and can run up to 200 analysis steps per round
- **Biochemistry Agent** — Interprets biochemical experimental results from structured indicator values and data images, cross-referencing findings from other agents
- **Literature Agent** — Searches and synthesizes relevant scientific literature. Supports three modes: Deep Research Models (e.g., openai/o4-mini-deep-research), Google Gemini Deep Research, and PubMed + LLM synthesis for targeted searches
- **Painter Agent** — Generates graphic abstracts summarizing the research findings
- **Supervisor Agent** — Monitors bioinformatics analysis steps in real-time using a lightweight vision LLM, detecting issues, identifying biological checkpoints, and escalating critical problems to the PI Agent
- **Technique Agent** — Provides API documentation lookup and error summarization for the Supervisor and Bioinformatics agents

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
│  ┌────┴──────┐ ┌───────────┐ ┌─────────────┐ ┌─────────┐  │
│  │Supervisor │ │  Painter  │ │  Literature │ │Technique│  │
│  │   Agent   │ │   Agent   │ │    Agent    │ │  Agent  │  │
│  └───────────┘ └───────────┘ └─────────────┘ └─────────┘  │
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
| Pathology images | PNG, JPEG, GIF, BMP, TIFF, WebP, SVG | Large images auto-split to 1024px tiles |
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

> Technical details for each release are documented in [HANDOVER.md](./HANDOVER.md).

### v0.20.2 (2026-04-21)

- **Advanced Backend Modularization**: All API routes extracted from main.py into 14 dedicated route modules (root, config, sessions, session_config, indicators, upload, lifecycle, interaction, snapshots, export_import, files, chat, websockets, reports)
- **Session Persistence Service**: New `services/session_store.py` centralizes memory cache + atomic disk writes with save throttling and per-session locks
- **Workflow Runner Service**: New `services/workflow_runner.py` manages headless analysis workflows for API-based usage
- **Session Cleanup Task**: New `tasks/session_cleanup.py` provides periodic TTL-based in-memory cache eviction for idle sessions
- **Agent LLM Utility**: New `utils/agent_llm.py` provides per-agent LLM client factory and file-path ownership validation
- **Extended State Management**: `state.py` expanded with per-session locks, access time tracking, prepared exports, import jobs tracking, chat delegation tracking, and WebSocket monitor tasks
- **Chat Delegation**: Full agent delegation system in chat routes allowing PI Agent to coordinate bioinformatics, pathology, biochemistry, and literature agents in parallel

### v0.20.1 (2026-04-21)

- **User Authentication**: New registration and login system, all API requests require JWT token auth
- **Session Ownership**: Each analysis session is now tied to its creator, users can only access their own sessions
- **WebSocket Authentication**: WebSocket connections support token auth, auth failures disconnect without retry
- **Backend Modularization**: Refactored into modular structure with routes/, services/, middleware/, utils/, tasks/
- **Auto Cleanup Tasks**: New TTL-based automatic expiration cleanup for upload files

### v0.19.9 (2026-04-19)

- Round progress now shows the actual max rounds setting
- Improved reliability of text-mode graphic abstract generation with automatic retries
- Better error messages when graphic abstract generation fails
- Improved message display robustness to gracefully handle non-text content

### v0.19.8 (2026-04-15)

- Bulk/scRNA-seq cross-validation is now enforced as a mandatory analysis step in the Exploration phase
- Cell-type deconvolution is now mandatory when both bulk and scRNA-seq data are present
- Standardized bioinformatics library names across all agents (pydeseq2, harmonypy, bbknn, scvi, decoupler)
- Automatic fallback for bulk RNA-seq DEG analysis when PyDESeq2 is unavailable

### v0.19.7 (2026-04-14)

- Bulk RNA-seq validation is now mandatory in the INNOVATION phase when both bulk and scRNA-seq data are present
- Analysis failed state now shows in the UI with a snapshot button
- Report progress messages now display checkmark icons for completed steps

### v0.19.6 (2026-04-12)

- Text-mode graphic abstracts are now generated as a fallback when the Painter Agent is unavailable

### v0.19.5 (2026-04-12)

- Cell-type deconvolution is now mandatory when both bulk and scRNA-seq data are present
- Standardized bioinformatics library names across all agents
- Improved literature search performance by reducing unnecessary context accumulation

### v0.19.4 (2026-04-12)

- Bulk/scRNA-seq cross-validation is now enforced as a mandatory analysis step
- Added automatic fallback for bulk RNA-seq DEG analysis when PyDESeq2 is unavailable

### v0.19.3 (2026-04-11)

- Improved LLM prompt caching for faster bioinformatics analysis
- Added cache hit tracking for monitoring API efficiency

### v0.19.2 (2026-04-11)

- PI Agent now dynamically generates analysis tasks based on detected data types
- Bulk RNA-seq data is now required to be analyzed before reporting findings
- Added support for GIF, BMP, WebP, and SVG image formats

### v0.19.1 (2026-04-09)

- Added complete support for bulk RNA-seq data analysis with cross-validation against single-cell pseudobulk results

### v0.18.15 (2026-04-08)

- Added advanced analysis modules: gene signature scoring, trajectory analysis, transcription factor networks, cell-cell communication, stemness analysis, tumor estimation, compositional analysis, and immune repertoire profiling
- Errors are now shown via alert dialog instead of only in console

### v0.18.14 (2026-04-05)

- Improved handling of malformed cluster marker data
- Expanded literature search gene sampling for more comprehensive results

### v0.18.12 (2026-04-03)

- Added hard limit on code execution errors to prevent infinite loops
- Improved Painter Agent compatibility with additional image generation providers
- PI now automatically selects which agents run each round from round 2 onwards
- Added development and production mode switching

### v0.18.10 (2026-04-01)

- Improved LLM provider compatibility
- Added Zhipu AI as an image generation provider
- Improved LLM client caching for better performance

### v0.18.6 (2026-03-30)

- Improved snapshot timing and legacy snapshot compatibility
- Enhanced supervisor monitoring for stuck analysis detection
- Added configurable Deep Research model selection

### v0.18.5 (2026-03-30)

- Improved WebSocket stability and reconnection handling
- Better handling of invalid data values in biochemistry
- Fixed chat history cleanup on session expiry

### v0.18.4 (2026-03-30)

- Added DashScope image generation support
- Full Jupyter kernel state now preserved in snapshots
- Vision now automatically disables after repeated failures
- Improved annotation context report layout

### v0.18.3 (2026-03-29)

- PI Agent now guides the cell type annotation process
- Enhanced supervisor figure review with issue classification
- Improved figure revision handling

### v0.18.2 (2026-03-29)

- Improved session concurrency handling
- Better API rate limit retry logic
- Enhanced sandbox security
- Blocked executable file uploads

### v0.18.0 (2026-03-27)

- Added snapshot tree view for analysis checkpoint history
- Improved snapshot state restoration reliability

### v0.17.4 (2026-03-26)

- Complete UI visual redesign with new typography, colors, and animations

### v0.17.3 (2026-03-26)

- System now automatically finds correct API documentation when encountering errors
- Bioinformatics Agent can look up package documentation proactively

### v0.17.1 (2026-03-26)

- Each agent can now use different LLM providers
- Added support for more image generation providers

### v0.16.2 (2026-03-24)

- Official Docker image now available

### v0.16.1 (2026-03-24)

- Automatic fallback for figure revision when vision is not supported
- Extended analysis timeout to 1.5 hours

### v0.15.8 (2026-03-23)

- Auto-generated PDF companions for all figures

### v0.15.7 (2026-03-22)

- Cell type annotation now required before the exploration report
- Inline tables with preview and download in reports

### v0.14.6 (2026-03-21)

- Automatic detection of AnnData variable names
- Improved error recovery

### v0.14.4 (2026-03-17)

- Project renamed from InteSc to ScAI

### v0.14.0 (2026-03-15)

- Detailed score computation documentation
- Richer annotation dialog with alternative cell types and confidence reasons

### v0.13.0 – v0.13.4 (2026-03-12 – 2026-03-13)

- Two-phase analysis: Exploration followed by Innovation phase
- Cell annotation confirmation with manual review
- Multiple research directions with correlation analysis

### v0.12.0 – v0.12.4 (2026-03-10 – 2026-03-11)

- Supervisor Agent for real-time monitoring
- Pause/Stop controls for analysis execution
- PI-reviewed publication figures with auto-revision

### v0.11.0 – v0.11.1 (2026-03-09)

- Biochemistry results now inform other agents

### v0.10.0 – v0.10.3 (2026-03-07 – 2026-03-09)

- Deep Research mode for comprehensive literature surveys
- Full chronological analysis log

### v0.9.0 (2026-03-04)

- PI reviews notebooks and figures before reports
- Reports include appendix with project info
- Graphic abstract generation and revision

### v0.3.0 – v0.8.9 (2026-02-04 – 2026-03-03)

- Core multi-agent architecture
- Treatment group system
- Chat interface with PI Agent
- Painter Agent for graphic abstracts
- Report generation
- Literature-based annotation

## License

Apache License 2.0