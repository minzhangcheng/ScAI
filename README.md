# InteSc - Integrated and Intelligent Single Cell Sequencing Analysis

**InteSc** (Integrated and Intelligent Single Cell Sequencing Analysis) is a multi-agent collaboration system based on LangGraph for comprehensive analysis of biomedical research data. The system integrates pathology section analysis, high-throughput sequencing data, biochemical indicators, and literature research, completing comprehensive phenotype analysis through multi-round iterative collaboration.

## Core Features

### Five Specialized Agent Collaboration

- **PI Agent**: Project coordinator, manages overall workflow and makes decisions
- **Pathology Agent**: Analyzes histological sections (supports H&E, Sirius Red, Prussian Blue, and other staining types)
- **Bioinformatics Agent**: Performs high-throughput sequencing data analysis (scRNA-seq, snRNA-seq, bulk RNA-seq)
- **Biochemistry Agent**: Interprets biochemical experimental results
- **Literature Agent**: Provides literature support (using Google Deep Research)

### Iterative Analysis Workflow

- Multi-round collaboration, progressively deepening analysis
- Automatic conflict detection and resolution
- Convergence detection for automatic stopping

### Local Code Execution

- Uses local Jupyter kernel for code execution (supports large memory analysis)
- Each execution is automatically saved as Jupyter Notebook (.ipynb format)
- Supports scanpy and other bioinformatics tools
- Reproducible code (fixed random seed 42)

### State Persistence

- Pause/resume analysis at any time
- Uses dill to save complete Python runtime environment (including all variables, AnnData objects, etc.)
- Resumes without re-executing code, directly loading all variables
- Notebook history saved as .ipynb files

### Detailed Report Generation

The system supports generating comprehensive reports with 9 chapters and 40+ analysis items:

1. **Quality Control (QC)**: QC violin plots, scatter plots, gene expression distribution, statistics table
2. **Cell Clustering**: UMAP clustering, grouping, sample distribution, cluster dendrogram
3. **Cell Type Annotation**: Marker gene heatmap, dotplot, feature plots
4. **Cell Proportion Analysis**: Stacked bar charts, grouped bar charts, heatmaps
5. **Differential Expression Analysis**: Volcano plots, heatmaps, DE gene lists
6. **Pathway Enrichment Analysis**: GO/KEGG enrichment, GSEA analysis
7. **Cell-Cell Communication Analysis**: Communication heatmaps, ligand-receptor analysis (built-in 100+ ligand-receptor pairs database)
8. **Trajectory Analysis**: PAGA graphs, trajectory analysis
9. **Targeted Analysis**: Custom analysis based on research objectives

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
│               (Zustand + Tailwind CSS)                      │
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
│       │    ┌─────────────┐      │                           │
│       └───►│  Literature │◄─────┘                           │
│            │    Agent    │                                  │
│            └─────────────┘                                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  OpenRouter  │  │Local Jupyter │  │    Google    │      │
│  │  (LLM API)   │  │   Kernel     │  │ Deep Research│      │
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

Configure the following API keys in the web interface:

- **OpenRouter API Key**: For calling Gemini and Claude models
  - Get it at: https://openrouter.ai/keys

- **Google AI API Key**: For Gemini Deep Research (optional)
  - Get it at: https://aistudio.google.com/app/apikey

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

### 4. Start Analysis

Click "Start Analysis", the system will:

1. PI creates analysis plan
2. Each Agent executes tasks in parallel
3. PI synthesizes results and decides next steps
4. Repeat until convergence (default max 3 rounds)

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
│   │   ├── pi_agent.py         # PI Agent
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent
│   │   ├── biochemistry_agent.py    # Biochemistry Agent
│   │   └── literature_agent.py      # Literature Agent
│   ├── models/
│   │   └── schemas.py          # Pydantic data models
│   ├── utils/
│   │   ├── llm_client.py       # OpenAI compatible API client
│   │   ├── sandbox.py          # Jupyter code execution environment
│   │   ├── image_utils.py      # Pathology image processing
│   │   └── llm_logger.py       # LLM interaction logging
│   ├── static/                 # Frontend static files
│   ├── config.py               # Configuration management
│   ├── workflow.py             # LangGraph workflow
│   ├── report_generator.py     # Detailed report generator
│   └── main.py                 # FastAPI service
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── stores/             # Zustand state management
│   │   ├── hooks/              # API hooks
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

- **Main Models**: Gemini 3 Pro, Claude Opus 4.5 (via OpenRouter)
- **Literature Research**: Google Gemini Deep Research

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

## License

MIT License
