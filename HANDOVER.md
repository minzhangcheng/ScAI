# InteSc Project Handover Document

This document is intended for developers and operations personnel, providing technical details, architecture design, development guides, and operations instructions.

## Project Overview

**InteSc** (Integrated and Intelligent Single Cell Sequencing Analysis) is a multi-agent collaboration system for biomedical data analysis. Through the coordination of PI Agent, Bioinformatics Agent, Pathology Agent, Biochemistry Agent, and Literature Agent, it automates comprehensive analysis of single-cell sequencing data.

## Technical Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Port 3000)                  │
│              Zustand State + Tailwind CSS                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API + WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                FastAPI Backend (Port 8000)                  │
├─────────────────────────────────────────────────────────────┤
│                  LangGraph Workflow Engine                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ PI Agent ──► Pathology Agent ──► Bioinformatics Agent │ │
│  │     │                                    │             │ │
│  │     └────────► Biochemistry Agent ◄──────┘             │ │
│  │                      │                                 │ │
│  │                Literature Agent                        │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  OpenRouter │  │   Jupyter   │  │   Google Gemini     │ │
│  │   LLM API   │  │   Kernel    │  │   Deep Research     │ │
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
│   ├── main.py                 # FastAPI entry point
│   │                           # - REST API endpoints
│   │                           # - WebSocket real-time communication
│   │                           # - Session management
│   │                           # - Report generation task scheduling
│   │
│   ├── workflow.py             # LangGraph workflow
│   │                           # - Multi-Agent coordination
│   │                           # - Iterative analysis logic
│   │                           # - Convergence detection
│   │                           # - State callbacks
│   │
│   ├── report_generator.py     # Report generator
│   │                           # - 9 chapters, 40+ analysis items
│   │                           # - Jupyter code execution
│   │                           # - Chart/table generation
│   │
│   ├── config.py               # Configuration management
│   │                           # - APIConfig: LLM keys
│   │                           # - ModelConfig: Model selection
│   │                           # - AgentPrompts: System prompts
│   │                           # - SystemConfig: System parameters
│   │                           # - Data directory path definitions
│   │
│   ├── agents/
│   │   ├── base.py             # Agent base class
│   │   ├── pi_agent.py         # PI Agent
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── biochemistry_agent.py    # Biochemistry Agent
│   │   └── literature_agent.py # Literature Agent
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic data models
│   │
│   ├── utils/
│   │   ├── llm_client.py       # LLM client
│   │   ├── sandbox.py          # Jupyter execution environment
│   │   ├── image_utils.py      # Image processing
│   │   └── llm_logger.py       # LLM logging
│   │
│   └── static/                 # Static files (backup)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main application component
│   │   ├── components/         # React components
│   │   ├── stores/             # Zustand state
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
├── sessions/                   # Session JSON files
├── uploads/                    # Uploaded data files
├── reports/                    # Generated reports
├── states/                     # Jupyter state files (.ipynb, .pkl)
├── workspace/                  # Jupyter working directory
└── logs/                       # LLM interaction logs

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

Multi-Agent coordination system based on LangGraph:

```python
class AnalysisWorkflow:
    def __init__(self, state_callback, message_callback):
        # state_callback: Save state after each round
        # message_callback: Real-time message push

    async def run(self, state: AnalysisState, session_id: str):
        # Iteratively execute analysis rounds
        # Each round: PI planning → Agent execution → PI synthesis → Convergence check
```

**Workflow:**

1. **PI Planning Phase**: PI Agent analyzes current state, assigns tasks to specialized Agents
2. **Parallel Execution Phase**: Pathology/Bioinformatics/Biochemistry Agents work in parallel
3. **Literature Support**: Literature Agent responds to queries from other Agents
4. **PI Synthesis Phase**: PI Agent synthesizes all findings, detects conflicts, decides next steps
5. **Convergence Detection**: Stops after 2 consecutive rounds with no new findings (configurable)

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
def save_session_to_disk(session_id: str, state: AnalysisState):
    """Save complete state to disk"""

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

### 3. Jupyter Code Execution (sandbox.py)

```python
class LocalJupyterSandbox:
    async def start(self):
        """Start Jupyter kernel"""

    async def execute(self, code: str) -> CodeExecutionResult:
        """Execute code, return output and charts"""

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

### 4. Report Generator (report_generator.py)

**9 Chapters:**

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

**Generation Flow:**

```python
class DetailedReportGenerator:
    async def generate_report(self, state: AnalysisState):
        # 1. Initialize Jupyter Sandbox
        await self.sandbox.start()

        # 2. Load data
        await self._setup_environment(state)

        # 3. PI plans report structure
        report_plan = await self._plan_report(state, conclusions, language)

        # 4. Generate each section
        for section_plan in report_plan["sections"]:
            section = await self._generate_section(...)

        # 5. Generate abstract and conclusions
        abstract = await self._generate_abstract(...)
        final_conclusions = await self._generate_final_conclusions(...)
```

### 5. LLM Client (llm_client.py)

```python
class OpenAICompatibleClient:
    def __init__(self, api_key, base_url, model):
        # Supports OpenRouter, OpenAI, and other compatible APIs

    async def chat(self, messages, tools=None) -> str:
        # Single conversation

    async def chat_stream(self, messages, tools=None):
        # Streaming response
```

**Retry Logic:**

- Max retries: 3
- Exponential backoff: 1s, 2s, 4s
- Supports function calling (tools)

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | Get all session list |
| `/api/session` | POST | Create new session |
| `/api/session/{id}` | GET | Get session status |
| `/api/session/{id}/load` | GET | Load complete session data |
| `/api/session/{id}/delete` | DELETE | Delete session |
| `/api/session/{id}/upload-pathology` | POST | Upload pathology images |
| `/api/session/{id}/upload-seq` | POST | Upload sequencing data |
| `/api/session/{id}/upload-biochemistry` | POST | Upload biochemistry data |
| `/api/session/{id}/detailed-report` | GET | Get generated report |
| `/api/session/{id}/generate-detailed-report` | POST | Start report generation |
| `/api/session/{id}/report/download-all-figures` | POST | Download all figures |
| `/api/session/{id}/report/download-all-tables` | POST | Download all tables |
| `/api/session/{id}/report/download-full` | POST | Download complete report |
| `/api/config` | GET/POST | Get/save configuration |
| `/api/file` | GET | Get file content |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{session_id}` | Analysis real-time messages |
| `/ws/report/{session_id}` | Report generation progress |

**WebSocket Message Types:**

```typescript
interface WebSocketMessage {
  type: 'status' | 'progress' | 'message' | 'result' | 'error' | 'figure' | 'table';
  data: any;
}
```

## Data Models

### AnalysisState (Core State)

```python
class AnalysisState(BaseModel):
    session_id: str
    status: TaskStatus  # pending, running, completed, failed, paused
    initial_instruction: str

    # Data paths
    control_seq_path: Optional[str]
    treatment_seq_path: Optional[str]
    pathology_images: List[PathologyImage]
    biochemistry_results: List[BiochemistryResult]

    # Analysis results
    round_results: List[RoundResult]
    messages: List[AgentMessage]
    final_summary: Optional[str]

    # Configuration
    custom_prompts: Optional[AgentPrompts]
    custom_models: Optional[ModelConfig]
    language: str = "en"
```

### RoundResult (Single Round Result)

```python
class RoundResult(BaseModel):
    round_number: int
    pathology_findings: Optional[str]
    bioinformatics_findings: Optional[str]
    biochemistry_findings: Optional[str]
    literature_findings: Optional[str]
    pi_synthesis: str
    next_steps: List[str]
    conflicts: List[str]
    timestamp: datetime
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
├── sessions/               # Session state
├── uploads/                # User uploaded data
├── reports/                # Generated reports
├── states/                 # Jupyter state
├── workspace/              # Jupyter working directory
└── logs/                   # LLM interaction logs

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
| `tile_size` | 1024 | Pathology image tile size |
| `tile_overlap` | 64 | Tile overlap pixels |

### Common Issues

**1. Session Load Failure**

Check if `sessions/{id}.json` file exists and is properly formatted.

**2. Report Generation Failure**

- Check if data file paths are correct
- Check backend logs for specific errors
- Confirm Jupyter kernel starts normally

**3. WebSocket Disconnection**

- Frontend implements auto-reconnect
- Check network connection and proxy settings

**4. Code Execution Failure**

- Check `states/{id}.ipynb` for execution history
- Confirm required Python packages are installed

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.100 | Web framework |
| langgraph | >=0.0.20 | Workflow orchestration |
| openai | >=1.0 | LLM API |
| pydantic | >=2.0 | Data models |
| jupyter_client | >=8.0 | Code execution |
| dill | >=0.3 | State serialization |

### Bioinformatics Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scanpy | >=1.9 | Single-cell analysis core |
| anndata | >=0.9 | Data structures |
| liana | >=1.0 | Cell communication analysis |
| scvelo | >=0.3 | RNA velocity |
| cellrank | >=2.0 | Cell fate analysis |
| decoupler | >=1.5 | Pathway activity |
| gseapy | >=1.1 | GO/KEGG enrichment |

## Contact

For issues, please contact the project maintainers.

---

*Document updated: 2026-01-29*
