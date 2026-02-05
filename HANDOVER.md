# InteSc Project Handover Document

This document is intended for developers and operations personnel, providing technical details, architecture design, development guides, and operations instructions.

**Version**: 0.3.0
**Last Updated**: 2026-02-04

## Project Overview

**InteSc** (Integrated and Intelligent Single Cell Sequencing Analysis) is a multi-agent collaboration system for biomedical data analysis. Through the coordination of six specialized agents (PI, Bioinformatics, Pathology, Biochemistry, Literature, and Painter), it automates comprehensive analysis of single-cell sequencing data with parallel execution, state version control, and structured error tracking.

## Technical Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Port 3000)                  │
│         Zustand State + Tailwind CSS + Recharts             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │  Data   │ │ Prompts │ │Analysis │ │ Results │ │Report │ │
│  │  Panel  │ │  Panel  │ │+ Errors │ │  Panel  │ │ Panel │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API + WebSocket (state_sync, parallel_errors)
┌──────────────────────────▼──────────────────────────────────┐
│                FastAPI Backend (Port 8000)                  │
├─────────────────────────────────────────────────────────────┤
│                  LangGraph Workflow Engine                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    PI Agent                            │ │
│  │                       │                                │ │
│  │        ┌──────────────┼──────────────┐ (parallel)      │ │
│  │        ▼              ▼              ▼                 │ │
│  │   Pathology    Bioinformatics   Biochemistry           │ │
│  │     Agent          Agent          Agent                │ │
│  │        │              │              │                 │ │
│  │        └──────────────┼──────────────┘                 │ │
│  │                       ▼                                │ │
│  │               Literature Agent                         │ │
│  │                       │                                │ │
│  │                       ▼                                │ │
│  │                Painter Agent (graphic abstract)        │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  OpenRouter │  │   Jupyter   │  │   Google APIs       │ │
│  │   LLM API   │  │ Kernel+dill │  │ (Research + Imagen) │ │
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
│   │   ├── pi_agent.py         # PI Agent (coordinator + figure reviewer)
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent (autonomous)
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── biochemistry_agent.py    # Biochemistry Agent
│   │   ├── literature_agent.py # Literature Agent
│   │   └── painter_agent.py    # Painter Agent (graphic abstract)
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic data models
│   │                           # - AgentError: Structured error tracking
│   │                           # - state_version: Version control
│   │                           # - WebSocketMessage: New types (state_sync, parallel_errors)
│   │
│   ├── utils/
│   │   ├── llm_client.py       # LLM client
│   │   ├── sandbox.py          # Jupyter execution environment
│   │   ├── image_utils.py      # Image processing
│   │   ├── upload.py           # Chunked upload manager (10GB support)
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

Multi-Agent coordination system based on LangGraph with parallel execution and error recovery:

```python
class SingleCellAnalysisWorkflow:
    def __init__(self, config, session_id, message_callback, state_callback):
        # message_callback: Real-time message push (WebSocket)
        # state_callback: Save state after each round

    async def run(self, state: AnalysisState):
        # Iteratively execute analysis rounds with parallel branches
        # Each round: PI planning → Parallel analysis → Aggregation → Literature → PI synthesis
```

**Workflow Graph Structure:**

```
initialize_round ◄────────────────────────┐
      │                                   │
      ▼                                   │
  pi_planning                             │
      │                                   │
      ├───┬───┐ (Fan-out: parallel)       │
      ▼   ▼   ▼                           │
 pathology  bioinformatics  biochemistry  │
      │   │   │                           │
      └───┴───┘ (Fan-in)                  │
      ▼                                   │
  aggregate_results  ←── error tracking   │
      │                                   │
      ▼                                   │
  literature_research                     │
      │                                   │
      ▼                                   │
  pi_synthesis                            │
      │                                   │
  ┌───┴───┐                               │
  │       │                               │
(continue) (stop)                         │
  │       │                               │
  │       ▼                               │
  │  generate_final_report                │
  │       │                               │
  │       ▼                               │
  │  generate_graphic_abstract            │
  │       │                               │
  │       ▼                               │
  │  generate_detailed_report             │
  │       │                               │
  │       ▼                               │
  │      END                              │
  │                                       │
  └───► check_convergence ────────────────┘
```

**Key Features:**

1. **Parallel Execution**: Pathology/Bioinformatics/Biochemistry Agents run in parallel (Fan-out/Fan-in pattern)
2. **Retry Mechanism**: `@with_retry(max_retries=3)` decorator with exponential backoff
3. **Structured Error Tracking**: Errors captured in `AgentError` model with agent, type, message, details
4. **State Version Control**: `_send_state_sync()` method ensures atomic version updates
5. **Figure Review Loop**: Bioinformatics Agent's figures are reviewed by PI before approval
6. **Convergence Detection**: Stops after 2 consecutive rounds with no new findings

**New Message Types:**

| Type | Purpose |
|------|---------|
| `state_sync` | Version control and state synchronization |
| `parallel_errors` | Structured error list from parallel execution |
| `figure_review` | PI reviewing generated figures |
| `graphic_abstract` | Painter Agent output |

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
  id: string;           // Unique message ID (for deduplication)
  type: MessageType;    // See table below
  agent?: AgentType;    // Source agent (if applicable)
  content: any;         // Message payload
  timestamp: string;    // ISO8601 timestamp
}

// Message Types
type MessageType =
  // System messages
  | 'status'            // Status update
  | 'state_sync'        // Version control sync (NEW)
  | 'parallel_errors'   // Structured error list (NEW)
  | 'completed'         // Analysis complete
  // Analysis process
  | 'message'           // General message
  | 'progress'          // Progress update
  | 'result'            // Analysis result
  | 'error'             // Error message
  // Results
  | 'round_summary'     // Round summary
  | 'integrated_report' // Integrated report
  | 'graphic_abstract'  // Graphic abstract (NEW)
  | 'detailed_report'   // Detailed report
  // Figure review
  | 'figure_review'     // PI reviewing figure (NEW)
  | 'figure_approved'   // Figure approved (NEW)
  | 'figure_revision'   // Revision requested (NEW)
```

**State Sync Message:**

```json
{
  "type": "state_sync",
  "content": {
    "change_type": "round_complete|status_change|error_added",
    "version": 5,
    "status": "running",
    "current_round": 2,
    "timestamp": "2026-02-04T10:30:00Z",
    "unresolved_errors": 0
  }
}
```

**Parallel Errors Message:**

```json
{
  "type": "parallel_errors",
  "content": {
    "errors": [
      {
        "id": "uuid",
        "agent": "pathology",
        "error_type": "execution_error",
        "message": "Image analysis failed",
        "details": "...",
        "round": 1
      }
    ],
    "total_errors": 1,
    "completed_agents": ["bioinformatics", "biochemistry"]
  }
}
```

## Data Models

### AnalysisState (Core State)

```python
class AnalysisState(BaseModel):
    id: str
    name: str
    status: TaskStatus  # pending, running, completed, failed, paused, waiting_input
    initial_instruction: str

    # Version control (NEW in v0.3.0)
    state_version: int = 0
    last_sync_timestamp: datetime

    # Data paths
    control_seq_path: Optional[str]
    treatment_seq_path: Optional[str]
    control_loom_path: Optional[str]      # For RNA velocity
    treatment_loom_path: Optional[str]
    pathology_images: List[PathologyImage]
    biochemistry_results: List[BiochemistryResult]
    biochemistry_files: List[BiochemistryFile]

    # Analysis results
    round_results: List[RoundResult]
    messages: List[AgentMessage]
    code_history: List[CodeExecution]
    final_summary: Optional[str]

    # Publication outputs
    publication_figures: List[PublicationFigure]
    publication_tables: List[PublicationTable]
    graphic_abstract: Optional[GraphicAbstract]  # NEW
    integrated_report: Optional[str]

    # Error tracking (NEW in v0.3.0)
    agent_errors: List[AgentError]

    # Configuration
    custom_prompts: Dict[str, str]
    custom_models: Dict[str, str]
    custom_temperatures: Dict[str, float]

    # Methods
    def increment_version(self)
    def add_error(agent, error_type, message, details) -> AgentError
    def get_unresolved_errors() -> List[AgentError]
    def resolve_error(error_id, resolution)
```

### AgentError (NEW in v0.3.0)

```python
class AgentError(BaseModel):
    id: str                      # Unique ID
    agent: AgentType             # Which agent failed
    error_type: str              # execution_error, timeout, data_missing, llm_error
    message: str                 # Error message (truncated to 200 chars)
    details: Optional[str]       # Full error details
    round_number: int            # Which round
    timestamp: datetime
    resolved: bool = False       # Resolution status
    resolution: Optional[str]    # How it was resolved
```

### RoundResult (Single Round Result)

```python
class RoundResult(BaseModel):
    round_number: int
    pathology_findings: List[str]
    bioinformatics_findings: List[str]
    biochemistry_findings: List[str]
    literature_findings: List[str]
    pi_summary: str
    next_steps: List[str]
    conflicts: List[Dict[str, Any]]
    new_findings: bool = True    # For convergence detection
    timestamp: datetime
```

### GraphicAbstract (NEW in v0.3.0)

```python
class GraphicAbstract(BaseModel):
    id: str
    prompt: str           # Generation prompt
    image_base64: str     # Base64 encoded image
    created_at: datetime
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

- Frontend implements auto-reconnect with exponential backoff (1s, 2s, 4s... max 30s)
- Message history is automatically restored on reconnection
- Check network connection and proxy settings

**4. Code Execution Failure**

- Check `states/{id}.ipynb` for execution history
- Confirm required Python packages are installed

**5. State Version Conflicts (NEW)**

- If frontend shows outdated data, check `state_version` in session JSON
- Frontend detects conflicts when received version < current version
- Resolution: Trigger full session reload via `loadSession()`

**6. Parallel Analysis Errors (NEW)**

- Errors are tracked per-agent in `agent_errors` list
- Check `AnalysisPanel` for error display
- Errors persist until manually resolved
- Debug via: `sessions/{id}.json` → `agent_errors` array

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

## Key Code Locations

### Backend

| Feature | File | Line Numbers |
|---------|------|--------------|
| Workflow graph construction | `workflow.py` | 290-356 |
| Parallel execution (Fan-out/Fan-in) | `workflow.py` | 310-322 |
| Parallel results aggregation | `workflow.py` | 499-579 |
| State sync method | `workflow.py` | 371-400 |
| Retry decorator | `workflow.py` | 40-115 |
| AgentError model | `schemas.py` | 38-49 |
| Error management methods | `schemas.py` | 257-286 |
| WebSocket message types | `schemas.py` | 339-362 |
| Painter Agent | `painter_agent.py` | Full file |

### Frontend

| Feature | File | Line Numbers |
|---------|------|--------------|
| Global state definition | `appStore.ts` | 164-239 |
| Version control state | `appStore.ts` | 229-231, 330-331 |
| Error tracking state | `appStore.ts` | 232-238, 335-355 |
| WebSocket hook | `useApi.ts` | 640-918 |
| state_sync handling | `useApi.ts` | 770-787 |
| parallel_errors handling | `useApi.ts` | 790-807 |
| Reconnection recovery | `useApi.ts` | 700-760 |
| Error display component | `AnalysisPanel.tsx` | 484-538 |
| Agent status with errors | `AnalysisPanel.tsx` | 68-92 |

## Contact

For issues, please contact the project maintainers.

---

*Document updated: 2026-02-04*
*Version: 0.3.0*
