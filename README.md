# ScAI - Single Cell Sequencing Analysis Driven by AI

**ScAI** is a multi-agent AI system for comprehensive biomedical research data analysis. It brings together pathology image analysis, single-cell sequencing data, biochemical indicators, and literature research into a unified, iterative workflow — powered by seven specialized AI agents that collaborate like a research team.

**Version**: 0.14.4

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
cd scai
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

### v0.14.4 (2026-03-17)

- **Project renamed**: InteSc → ScAI (Single Cell Sequencing Analysis Driven by AI)
- **Directory restructure**: Data directory `intesc_data/` → `data/`, dependency directory `intesc_dep/` → `dep/`, code directory `intesc/` → `scai/`
- **Environment variable renamed**: `INTESC_DATA_DIR` → `SCAI_DATA_DIR`
- **Configurable startup**: `start.sh` now accepts `--host`, `--port`, and `--backport` arguments (defaults: 127.0.0.1, 3000, 8000)
- **GitHub repo renamed**: `minzhangcheng/intesc` → `minzhangcheng/scai`

### v0.14.3 (2026-03-17)

- **JPG figure output**: Bioinformatics figures now saved as JPG (quality 85, optimized) instead of PNG for web display, reducing file size; companion PDF still generated for publication
- **Larger plot fonts**: Default matplotlib font sizes increased (title 16pt, axes labels 14pt, tick labels 12pt) for better readability in reports and publications
- **Updated analysis library APIs**: decoupler v2.x API guidance (dc.op.collectri/progeny + dc.mt.ulm); gseapy organism must be lowercase; defensive empty-DataFrame checks in per-cluster loops; Harmony batch correction dimension-mismatch prevention; `importlib` added to blocked operations list
- **Supervisor report redesign**: Figures now embedded inline via `[FIGURE: filename]` markers rather than appended in a separate appendix; monitoring stats bar removed; report framed as a scientific document; user interaction history (cluster annotations, research direction choices) and downloadable notebook added to appendix
- **Painter agent reliability**: Image generation and revision now retry up to 3 times with exponential backoff; non-retriable client errors (4xx) are skipped immediately
- **Frontend JPG support**: Figure gallery download button updated PNG→JPG; graphic abstract and message stream auto-detect PNG vs JPEG from base64 header for correct MIME type
- **Bug fixes**: Race condition in WebSocket message restoration (no longer overwrites live messages); chat panel stale closure fix for message list; 100MB streaming size limit for generic file uploads; stop analysis now handles GENERATING_REPORT state; path traversal validation added to HTML report serve endpoints; catch-all report file endpoint added

### v0.14.2 (2026-03-16)

- **Bioinformatics robustness improvements**: Automatic duplicate index repair after concat/subset operations; corrected step budget calculation on checkpoint resume; round mismatch validation prevents cross-round state contamination
- **Smarter cell annotation**: Cluster annotation results are cached to avoid redundant Deep Research calls; annotation workflow now requires observing marker output before calling the annotation tool
- **Graphic abstract token management**: Input token estimation and auto-trimming for the Gemini image generation model, preventing context overflow errors
- **Connection stability**: WebSocket heartbeat keeps connections alive through proxies; concurrency fix prevents race conditions on connection close
- **Bug fixes**: Exploration report dialog timing fix (no longer blocked between Phase 1 completion and direction selection); PI_DECIDE resume correctly extends round limit when needed

### v0.14.1 (2026-03-15)

- **Fixed: Round feedback not waking workflow**: Submitting text guidance or files between rounds now immediately resumes the workflow instead of waiting for the countdown to expire
- **Fixed: Extra round silently skipped**: A stale convergence flag could cause an extra round (triggered by user input) to terminate immediately — now correctly cleared at round start
- **Fixed: Pause during annotation confirmation**: Pausing analysis while reviewing cluster annotations now correctly preserves the paused state
- **Fixed: Single-direction extra round extension**: Single-direction sessions incorrectly blocked the extra round when the user provided input on the last round
- **Fixed: Exploration report failure silently swallowed**: If exploration report generation fails, an error message is now shown so users can still select a research direction

### v0.14.0 (2026-03-15)

- **Score methodology transparency**: Bioinformatics Agent now registers how each score was computed (method, gene list, source) and surfaces this in the final report's methodology section
- **Jupyter sandbox security**: File writes outside the session directory are blocked at runtime; kernel initializes with correct working directory, figure output path, and subdirectories
- **Kernel failure detection**: If the Jupyter kernel fails to restart, subsequent code executions immediately return a clear error instead of hanging
- **Richer annotation dialog**: Cell annotation confirmation now shows alternative cell types and low-confidence reasons for each cluster; key markers increased to 10 per cluster; annotation context HTML report downloadable during review
- **Report improvements**: Report narrative organized by round with Exploration/Innovation phase labels; publication figures reliably included in the figure map
- **Innovation phase**: User text guidance between rounds is merged as supplementary tasks rather than overriding the direction-driven plan; direction queue state persisted immediately after pop to survive crashes
- **Direction workflow fixes**: Resume without a chosen direction defaults to PI selection; multi-direction queue can be aborted mid-run; cell type name overrides sanitized before code injection

### v0.13.4 (2026-03-13)

- **Fixed: Pause/resume deadlock** where resuming could leave the UI stuck with no visible "waiting for input" indicator
- **Fixed: Spurious extra round** after multi-direction correlation analysis completed
- **Fixed: Empty task lists in round 3+** when PI planning output lacked expected prefix tags
- **Performance**: Supervisor Agent LLM evaluation now sampled (errors, figures, every 5th step) to reduce API cost; bioinformatics message history capped at 80 entries

### v0.13.3 (2026-03-12)

- **Annotation context panel**: Cell annotation confirmation dialog now shows a structured Analysis Context sidebar with experimental design, biochemistry/pathology findings, and recent literature — giving biological context when reviewing cluster assignments
- **Exploration Report link**: Results panel shows a "View Exploration Report" button linking to the Phase 1 standalone HTML report
- **Report narrative context**: Generated report sections now reference Phase 1 exploration directions and user-chosen directions

### v0.13.2 (2026-03-12)

- **Fixed: Crash on first checkpoint** when the session states directory did not yet exist

### v0.13.1 (2026-03-12)

- **Multi-direction research**: Select multiple research directions from the Exploration Report — each runs as a dedicated innovation round, followed by an automatic multi-direction correlation analysis synthesizing findings across all directions
- **Direction candidate scoring**: Each AI-suggested direction now shows Innovation, Novelty, and Data Reliability scores (High/Medium/Low) with brief explanations
- **Redesigned direction selection dialog**: Multi-select with per-direction notes, custom direction entry, score badges, and countdown auto-selection; link to standalone Exploration Report HTML
- **Exploration Report HTML export**: Standalone HTML exploration report generated at end of Phase 1, accessible via API endpoint
- **Max rounds removed from sidebar**: Round count is now set automatically based on the number of selected directions

### v0.13.0 (2026-03-12)

- **Two-phase analysis workflow**: Round 1 is now a dedicated Exploration Phase for unbiased full-landscape discovery. After Round 1, an Exploration Report summarizes findings and offers research direction candidates. Users select a direction (or let PI decide) before entering the Innovation Phase (Round 2+) for focused hypothesis-driven analysis
- **Cell annotation confirmation**: Bioinformatics Agent pauses after cluster annotation so users can review, correct, or confirm each cluster's cell type and confidence level before analysis continues

### v0.12.4 (2026-03-11)

- **Session-centric data layout**: All per-session files (workspace, reports, uploads, states, logs) now co-located under `data/sessions/{session_id}/`. Existing sessions automatically resolved via legacy path fallback; migration script provided

### v0.12.3 (2026-03-10)

- **Report rendering**: Markdown headers and bullet lists now render correctly as HTML in generated reports; report detail view uses ReactMarkdown; appendix section added to report UI; Supervisor Agent HTML report includes full appendix

### v0.12.2 (2026-03-10)

- **Pause/Stop controls**: Added Pause and Stop buttons to the analysis feedback dialog; Stop can optionally generate a summary report before terminating
- **Publication-quality figures**: Bioinformatics Agent now enforces figure quality standards (axis labels, font sizes, legend placement, composition barplot rules) and uses publication-style matplotlib themes
- **Figure revision**: Auto-retry on runtime errors during figure code revision; progressive review leniency from revision 2+ to avoid infinite loops
- **UX**: "Skip" button renamed to "Continue" in the round feedback dialog

### v0.12.1 (2026-03-10)

- **Report generation resumption**: Interrupted reports now resume from the last completed chapter instead of restarting from scratch
- **Reliability fixes**: Session save race condition; session eviction race; completed-session WebSocket handling; memory growth limits for real-time messages; graceful figure revision failure handling
- **Supervisor Agent**: Customizable system prompt via Prompts panel

### v0.12.0 (2026-03-10)

- **Supervisor Agent**: Real-time monitoring of each bioinformatics analysis step — detects issues, identifies biological checkpoints, and escalates critical problems to the PI Agent
- **Stability fixes**: LLM retry logic; settings sync error reporting; round feedback dialog deduplication; WebSocket version conflict recovery; pre-analysis settings sync parallelized
- **UI**: Expand/collapse buttons for long message content

### v0.11.1 (2026-03-09)

- **Bug fixes**: Duplicate publication figures; session cleanup leaving orphaned files; literature model config; hardcoded model names in report generation and chat; LLM retry order for rate limits

### v0.11.0 (2026-03-09)

- **Execution order**: Biochemistry now runs first so its findings inform pathology and bioinformatics agents
- **Dynamic PI planning**: PI Agent refines each downstream agent's tasks based on already-completed results within the same round, building a coherent chain of thought across agents
- **Cross-agent context**: Biochemistry findings injected into pathology and bioinformatics prompts; literature topics dynamically refined from all completed agent results

### v0.10.3 (2026-03-09)

- **Analysis log**: Every bioinformatics step (code, PI consultation, literature search, annotation, figures) recorded with reasoning and timestamps; reports include a full chronological log timeline with inline figures
- **Figure/table interpretations**: Scientific interpretation narrative added to publication figures and tables in generated reports
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
