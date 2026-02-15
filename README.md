# InteSc - Integrated and Intelligent Single Cell Sequencing Analysis

**InteSc** (Integrated and Intelligent Single Cell Sequencing Analysis) is a multi-agent collaboration system based on LangGraph for comprehensive analysis of biomedical research data. The system integrates pathology section analysis, high-throughput sequencing data, biochemical indicators, and literature research, completing comprehensive phenotype analysis through multi-round iterative collaboration.

**Version**: 0.7.5

## Core Features

### Six Specialized Agent Collaboration

- **PI Agent**: Project coordinator, manages overall workflow, makes decisions, and reviews publication-quality figures
- **Pathology Agent**: Analyzes histological sections (supports H&E, Sirius Red, Prussian Blue, and other staining types)
- **Bioinformatics Agent**: Performs autonomous high-throughput sequencing data analysis (scRNA-seq, snRNA-seq, bulk RNA-seq) with checkpoint recovery
- **Biochemistry Agent**: Interprets biochemical experimental results (3-step analysis: structured indicators → visual image analysis → synthesis & conclusions)
- **Literature Agent**: Provides literature support (2 modes: Google Gemini Deep Research, PubMed + LLM synthesis) with multi-round iterative search, automatic fallback chain, and literature-based cluster annotation
- **Painter Agent**: Generates graphic abstracts using Google Gemini image generation

### Interactive Chat with PI Agent

- **Chat Tab**: Free-form conversational interface with the PI Agent after analysis
- Ask questions about results, request explanations, or discuss next steps
- **Chat Delegation**: PI can delegate complex requests to specialized agents (Bioinformatics, Pathology, Biochemistry, Literature) — agents execute analysis in the background, then PI synthesizes results into a comprehensive response
- **File Attachments**: Upload images and data files directly in chat, with per-file agent targeting (PI, Pathology, Bioinformatics, Biochemistry)
- **Figure Display & Generation**: PI can reference existing analysis figures or generate new custom figures on-the-fly via Jupyter sandbox
- **Stop/Cancel**: Cancel ongoing chat requests or delegation tasks with a stop button
- Markdown-rendered responses with inline figure display and PNG/PDF download

### Iterative Analysis Workflow with Human-in-the-Loop

- Multi-round collaboration, progressively deepening analysis
- **Selective Agent Activation**: From round 2 onwards, PI Agent autonomously decides which agents to activate based on analysis needs — Bioinformatics Agent always runs, while Pathology, Biochemistry, and Literature agents are activated only when needed, saving API costs and time
- **Round Feedback Dialog**: After each round, a modal shows per-agent findings with a configurable countdown timer (default 5 minutes; supports manual-only mode and no-wait auto-continue). Users can provide guidance, upload additional data (targeted to specific agents), or skip to auto-continue
- User input or file uploads on the last round automatically extends analysis by one additional round
- Automatic conflict detection and resolution
- Convergence detection for automatic stopping

### Local Code Execution

- Uses local Jupyter kernel for code execution (supports large memory analysis)
- Each execution is automatically saved as Jupyter Notebook (.ipynb format)
- **Auto PDF Generation**: Every PNG figure saved automatically gets a companion high-resolution PDF with vectorized fonts (Type 42) for publication-quality editing
- Supports scanpy and other bioinformatics tools
- **Docker Pip Install**: Bioinformatics Agent can install pip packages on-the-fly when running in Docker environments
- Reproducible code (fixed random seed 42)
- **Batch Effect Assessment**: Automatic batch effect detection step when multiple samples/batches are present — visualizes cells by batch before clustering

### State Persistence & Synchronization

- Pause/resume analysis at any time
- Uses dill to save complete Python runtime environment (including all variables, AnnData objects, etc.)
- Resumes without re-executing code, directly loading all variables
- Notebook history saved as .ipynb files
- **State Version Control**: Atomic version increments ensure frontend-backend consistency
- **WebSocket Reconnection Recovery**: Automatic message history restoration after connection loss
- **Structured Error Tracking**: Per-agent error tracking with detailed information and resolution support

### Detailed Report Generation

The system supports generating comprehensive reports with 9 chapters, 40+ analysis items, and cell cluster annotation table:

1. **Quality Control (QC)**: QC violin plots, scatter plots, gene expression distribution, statistics table
2. **Cell Clustering**: UMAP clustering, grouping, sample distribution, cluster dendrogram
3. **Cell Type Annotation**: Marker gene heatmap, dotplot, feature plots
4. **Cell Proportion Analysis**: Stacked bar charts, grouped bar charts, heatmaps
5. **Differential Expression Analysis**: Volcano plots, heatmaps, DE gene lists
6. **Pathway Enrichment Analysis**: GO/KEGG enrichment, GSEA analysis
7. **Cell-Cell Communication Analysis**: Communication heatmaps, ligand-receptor analysis (built-in 100+ ligand-receptor pairs database)
8. **Trajectory Analysis**: PAGA graphs, trajectory analysis
9. **Targeted Analysis**: Custom analysis based on research objectives

### Session Import/Export

- **Export**: Download complete session as tar.gz archive (state, uploads, workspace, reports, sandbox state)
- **Import**: Restore sessions from tar.gz, zip, or JSON archives with automatic path remapping and full sandbox restore
- Session browser with metadata display (status, round count, data types)

### State Persistence & Session Recovery

- Auto-reload session on page refresh (restores full state from server)
- Stale session detection: automatically clears sessions that no longer exist on server
- Base64 image stripping in WebSocket messages and localStorage to prevent overflow
- Read mode for reviewing existing analyses without starting new ones

### Real-time Communication

- Optimized WebSocket message streaming with deduplication
- State version control for frontend-backend consistency
- Automatic reconnection with exponential backoff
- Message history restoration after connection loss

### Multi-language Support

- Supports Chinese/English interface switching
- Prompts auto-switch based on language setting

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
│            React Web Frontend (localhost:3000)              │
│           (Zustand + Tailwind CSS + Recharts)               │
│  ┌──────┐ ┌───────┐ ┌────────┐ ┌───────┐ ┌──────┐ ┌──────┐ │
│  │ Data │ │Prompts│ │Analysis│ │Results│ │Report│ │ Chat │ │
│  │Panel │ │ Panel │ │ Panel  │ │ Panel │ │Panel │ │Panel │ │
│  └──────┘ └───────┘ └────────┘ └───────┘ └──────┘ └──────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ WebSocket / REST API
┌─────────────────────────▼───────────────────────────────────┐
│               FastAPI Backend (localhost:8000)              │
├─────────────────────────────────────────────────────────────┤
│                    LangGraph Workflow                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   PI    │  │ Pathol. │  │ Bioinfo │  │ Biochem │        │
│  │  Agent  │◄─┤  Agent  │◄─┤  Agent  │◄─┤  Agent  │        │
│  └────┬────┘  └─────────┘  └────┬────┘  └─────────┘        │
│       │                         │                           │
│       │    ┌─────────────┐      │      ┌─────────────┐     │
│       └───►│  Literature │◄─────┴─────►│   Painter   │     │
│            │    Agent    │             │    Agent    │     │
│            └─────────────┘             └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  OpenRouter  │  │Local Jupyter │  │ Google APIs  │      │
│  │  (LLM API)   │  │   Kernel     │  │ (Research +  │      │
│  │              │  │   + dill     │  │  Imagen)     │      │
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

**Main dependencies include:**

- **Web Framework**: FastAPI, uvicorn
- **Workflow Orchestration**: langgraph
- **LLM**: openai (OpenAI-compatible API)
- **Core Bioinformatics**: scanpy, anndata, muon, pandas
- **Visualization**: matplotlib, seaborn, plotly
- **Cell Communication**: liana, omnipath, squidpy
- **RNA Velocity**: scvelo, cellrank
- **Pathway Analysis**: decoupler, gseapy
- **Deep Learning**: scvi-tools, pytorch
- **Code Execution**: jupyter_client, dill

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Configure API Keys

Configure the following in the web interface:

- **OpenRouter API Key** (required): For calling Gemini and Claude models
  - Get it at: https://openrouter.ai/keys

- **PubMed Email** (required): Email for NCBI E-utilities PubMed access (NCBI requirement)

- **Google AI API Key** (optional): For Gemini Deep Research mode — auto-enables when set
  - Get it at: https://aistudio.google.com/app/apikey

- **Proxy URL** (optional): HTTP/HTTPS proxy for API requests

**Note**: Code execution uses local Jupyter kernel, no additional sandbox service configuration needed.

## Startup

### Method 1: Using Startup Script (Recommended)

```bash
./start.sh
```

This script will automatically:
1. Create Python virtual environment
2. Install dependencies
3. Start backend service (http://localhost:8000)
4. Start frontend service (http://localhost:3000)

Stop services:
```bash
./stop.sh
```

### Method 2: Manual Startup

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

A configuration window will pop up on first startup. Enter the API keys for each platform.

### 2. Create Analysis Session

Click "New Analysis", enter your research objective, for example:

> Analyze the tissue phenotype of a specific gene knockout mouse, focusing on cell type changes and related signaling pathway abnormalities

### 3. Upload Data

Upload in the interface:

- **Pathology Sections**: H&E or special staining images (large images auto-split to 1024px tiles)
- **Sequencing Data**: h5ad format AnnData files, .loom files
- **Biochemical Indicators**: Manual data entry or upload data tables/images

All uploaded data can be downloaded back to local via the download buttons in the Data panel.

### 4. Start Analysis

Click "Start Analysis", the system will:

1. PI creates analysis plan
2. Each Agent executes tasks in parallel
3. PI synthesizes results
4. **Round Feedback Dialog** appears — review findings, provide guidance or upload additional data, or skip
5. Repeat until convergence (default max 3 rounds, extendable via user input)

### 5. View Results

After analysis completion, you can view:

- Detailed findings from each round
- Final analysis report
- Generated charts and data tables

### 6. Generate Detailed Report

In the "Report" tab, click "Generate Report", the system will:

1. Load single-cell data to Jupyter environment
2. Execute 40+ analysis tasks
3. Generate PNG/PDF charts and CSV data tables
4. Support batch download of all results

## Customize Agent Prompts

You can customize each Agent's system prompt based on specific research needs:

- Adjust Pathology Agent's focus points for specific organs
- Adjust Bioinformatics Agent's analysis workflow for specific data types
- Add domain-specific terminology

## State Management

- **Pause**: Pause analysis at any time, state auto-saves
- **Resume**: Resume analysis from pause point (fully restore Jupyter environment)
- **Auto-save**: Auto-save to disk after each analysis round
- **Service Restart**: Load previous sessions from disk after restart

## Project Structure

```
intesc/                         # Code directory
├── backend/
│   ├── agents/                 # Agent implementations
│   │   ├── base.py             # Base Agent class
│   │   ├── pi_agent.py         # PI Agent (coordinator + figure reviewer)
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent (autonomous)
│   │   ├── biochemistry_agent.py    # Biochemistry Agent
│   │   ├── literature_agent.py      # Literature Agent (2 research modes + cluster annotation)
│   │   └── painter_agent.py         # Painter Agent (graphic abstract)
│   ├── models/
│   │   └── schemas.py          # Pydantic data models (AgentError, state_version)
│   ├── utils/
│   │   ├── llm_client.py       # LLM client (OpenRouter + Google Deep Research)
│   │   ├── sandbox.py          # Jupyter code execution environment
│   │   ├── image_utils.py      # Pathology image processing
│   │   ├── upload.py           # Chunked file upload manager
│   │   └── llm_logger.py       # LLM interaction logging
│   ├── static/                 # Frontend static files
│   ├── config.py               # Configuration management
│   ├── workflow.py             # LangGraph workflow (parallel execution)
│   ├── report_generator.py     # Detailed report generator
│   └── main.py                 # FastAPI service
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main application
│   │   ├── components/         # React components
│   │   │   ├── AnalysisPanel.tsx         # Analysis control + error display
│   │   │   ├── ChatPanel.tsx            # PI Agent free-form chat
│   │   │   ├── RoundFeedbackDialog.tsx  # Round-end feedback modal
│   │   │   ├── DataPanel.tsx            # Data upload
│   │   │   ├── PromptsPanel.tsx         # Agent configuration
│   │   │   ├── ResultsPanel.tsx         # Round results
│   │   │   ├── ReportPanel.tsx          # Report generation
│   │   │   ├── MessageStream.tsx        # Real-time messages
│   │   │   └── analysis/               # Report sub-components
│   │   ├── stores/             # Zustand state (agentErrors, stateVersion)
│   │   ├── hooks/              # API hooks (WebSocket reconnection)
│   │   ├── i18n/               # Internationalization
│   │   └── styles/             # CSS styles
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt
├── start.sh
├── stop.sh
├── README.md
└── HANDOVER.md

intesc_data/                    # Data directory (sibling to code directory)
├── sessions/                   # Session persistence storage
├── uploads/                    # Uploaded file storage
├── reports/                    # Generated reports
├── states/                     # Jupyter notebook and dill states
├── workspace/                  # Jupyter working directory
└── logs/                       # LLM interaction logs

intesc_dep/                     # Dependencies directory (sibling to code directory)
├── venv/                       # Python virtual environment
└── node_modules/               # Node.js dependencies
```

## Tech Stack

### Backend

- **Language**: Python 3
- **Web Framework**: FastAPI
- **Workflow Orchestration**: LangGraph
- **Data Models**: Pydantic v2
- **Code Execution**: jupyter_client (local kernel)
- **State Persistence**: dill

### Frontend

- **Language**: TypeScript
- **Framework**: React 18
- **State Management**: Zustand
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **Charts**: Recharts

### LLM

- **Main Models**: Gemini 3 Pro Preview, Claude Sonnet 4.5 (via OpenRouter)
- **Literature Research**: 2 modes with automatic fallback
  - Google Gemini Deep Research (via Interactions API, single call with workflow-level gap coverage)
  - PubMed + LLM synthesis (NCBI E-utilities search + LLM review)
- **Literature Chat**: Gemini 3 Flash Preview (fast model for literature agent)
- **Image Generation**: Gemini 3 Pro Image Preview for graphic abstracts

### Bioinformatics

- **Core**: scanpy, anndata, muon
- **Cell Communication**: liana, omnipath, squidpy
- **Trajectory Analysis**: scvelo, cellrank
- **Pathway Analysis**: decoupler, gseapy
- **Deep Learning**: scvi-tools

## Notes

1. Large pathology images are automatically split into 1024x1024 pixel tiles for analysis
2. Sequencing data analysis may take a while, please be patient
3. Deep Research may consume significant tokens per call
4. Recommended to run in a stable network environment
5. All code executions are automatically saved as Jupyter Notebooks (.ipynb) in the `states/` directory
6. When pausing/resuming analysis, the previous Python environment state is automatically restored
7. WebSocket reconnection automatically restores message history - no data loss on network interruptions
8. Analysis errors are tracked per-agent and displayed persistently until resolved
9. State version conflicts are detected automatically to ensure data consistency

## Changelog

### v0.7.5 (2026-02-15)

- **Selective Agent Activation**: From round 2 onwards, PI Agent autonomously decides which agents (Pathology, Biochemistry, Literature) to activate based on analysis needs. Bioinformatics Agent always runs. Skipped agents receive WebSocket notification. Uses existing `skip_nodes` mechanism in `@with_retry` decorator. New `agents_for_next_round` field in `RoundResult`
- **Docker Pip Install Support**: Bioinformatics Agent can now install pip packages via subprocess in Docker environments. `_validate_code()` adds special exception for pip/pip3 install commands. System prompt includes package installation guidelines

### v0.7.4 (2026-02-14)

- **Figure Step Budget**: Figure generation (`generate_publication_figure`) no longer counts toward bioinformatics agent's step budget. A safety cap of 5 figure-only iterations prevents infinite loops — after exhaustion, agent refuses new figure generation and redirects to remaining analysis steps
- **Figure Revision Fix**: Increased `max_tokens` for figure revision code generation from 2048 to 32000, preventing code truncation (SyntaxError) when revising complex publication figures
- **Data Download**: All uploaded data (pathology images, sequencing files, loom files, biochemistry files) can now be downloaded from the Data panel via individual download buttons
- **Agent Status Display Fix**: Fixed bug where completed analysis agents showed spinning icons during report generation. `getAgentStatus()` now correctly handles `generating_report` session status (analysis agents stay "done") and recognizes additional completion message types (`completed`, `integrated_report`, `graphic_abstract`, etc.)
- **Figure Gallery Fix**: Fixed figure previews not displaying after page reload (localStorage persistence stripped base64 data, which then overwrote fresh API data). Fixed MIME type mismatch (JPEG data displayed as PNG). Session-completed figures with `revision_requested` status now display as green "Final" instead of red "Revision Requested"
- Bug fixes: `TaskStatus` type now includes `generating_report`, backend `/load` and upload endpoints return `filePath` for pathology and biochemistry files

### v0.7.3 (2026-02-13)

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

### v0.7.2 (2026-02-11)

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

### v0.6.3 (2026-02-11)

- **Literature-based Cluster Annotation**: New `annotate_clusters` tool for bioinformatics agent — single literature search + LLM matching annotates all clusters at once, replacing per-cluster PI consultation
- **Round Context Enrichment**: Each agent now receives previous round findings and code history, preventing redundant analyses in multi-round workflows
- **Literature Config per Session**: Literature mode (Gemini/PubMed), Google API key, PubMed email, and search rounds are now stored per-session in `literature_config` (moved from global `APIConfig`)
- **ConfigPanel Redesign**: PubMed email (required) and Google API key (optional, auto-enables Gemini mode) moved to main config panel
- **LLM Client Improvements**: Robust content extraction from various response formats (string, list, None, reasoning models); reasoning model support (`o1`/`o3`/`o4` series use `max_completion_tokens` instead of `max_tokens`)
- **RoundFeedbackDialog Timer Fix**: Countdown timer now uses wall-clock time so background browser tabs don't freeze the countdown
- Bug fixes: literature agent cleanup, PromptsPanel literature config save, base64 stripping improvements

### v0.6.2 (2026-02-10)

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

### v0.6.1 (2026-02-10)

- Bug fixes: null safety in partialize, WebSocket "session not found" detection, report findings dict key safety

### v0.6.0 (2026-02-09)

- **Human-in-the-Loop Round Feedback**: New `RoundFeedbackDialog` modal appears after each analysis round with 2-minute countdown timer
  - Displays per-agent findings (Pathology, Bioinformatics, Biochemistry, Literature)
  - Users can provide text guidance and upload additional data files targeted to specific agents
  - Timer auto-pauses when user starts typing; auto-continues on expiry
  - User input on the last round extends max_rounds by 1
- **Workflow `wait_for_user_input` node**: Backend blocks between rounds via `asyncio.Event`, with 10-minute safety timeout
- **Cell type annotation improvements**: Strengthened bioinformatics agent prompt for mandatory cell type annotation with UMAP visualization
- **Report generator fixes**: Now searches workspace directory for annotated data; supports both `adata_annotated.h5ad` and `annotated_adata.h5ad` filenames
- **WebSocket reconnection fix**: Completed sessions now send full round_results and message_history on reconnect
- **Report loading retry**: ReportPanel retries loading after 3 seconds if report is null

### v0.5.5 (2026-02-09)

- Bug fixes for cell annotation workflow and report data loading
- WebSocket reconnection sends full session data for completed sessions

### v0.5.4 (2026-02-09)

- Bug fixes for bioinformatics agent analysis logic
- Improved session handling and sandbox timeout configuration
- Enhanced workflow state management
- UI improvements across AnalysisPanel, ConfigPanel, PromptsPanel, Sidebar
- WebSocket hook stability improvements
- i18n translation updates

### v0.5.3 (2026-02-09)

- Report generation and bioinformatics agent enhancements

### v0.5.1 (2026-02-08)

- **Biochemistry Agent v2**: Rebuilt with 3-step analysis workflow (structured indicators → visual image analysis → synthesis & conclusions)
- Major refactoring (1,325 insertions, 723 deletions)
- Improved image analysis using comprehensive prompts instead of strict JSON extraction

### v0.5.0 (2026-02-06)

- **Real-time Message Optimization**: Improved WebSocket message streaming and deduplication
- Enhanced frontend-backend state synchronization

### v0.4.0 - v0.4.1 (2026-02-05)

- **Report Generation Rebuilt**: Complete overhaul of the detailed report generation system
- Painter Agent improvements
- Various bug fixes

### v0.3.0 (2026-02-04)

- **Painter Agent**: Generates publication-quality graphic abstracts using Google Gemini image generation
- **State Version Control**: Atomic state versioning prevents frontend-backend synchronization issues
- **WebSocket Reconnection Recovery**: Automatic message history restoration after connection loss
- **Structured Error Tracking**: Per-agent error tracking with detailed information display
- **Parallel Analysis**: Pathology, Bioinformatics, and Biochemistry agents now run in parallel
- **Figure Review Loop**: PI Agent reviews and approves publication-quality figures before inclusion

## License

MIT License
