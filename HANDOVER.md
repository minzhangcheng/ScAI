# InteSc Project Handover Document

This document is intended for developers and operations personnel, providing technical details, architecture design, development guides, and operations instructions.

**Version**: 0.7.2
**Last Updated**: 2026-02-11

## Project Overview

**InteSc** (Integrated and Intelligent Single Cell Sequencing Analysis) is a multi-agent collaboration system for biomedical data analysis. Through the coordination of six specialized agents (PI, Bioinformatics, Pathology, Biochemistry, Literature, and Painter), it automates comprehensive analysis of single-cell sequencing data with parallel execution, state version control, structured error tracking, and optimized real-time communication.

## Technical Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Port 3000)                  │
│         Zustand State + Tailwind CSS + Recharts             │
│  ┌──────┐ ┌───────┐ ┌────────┐ ┌───────┐ ┌──────┐ ┌──────┐ │
│  │ Data │ │Prompts│ │Analysis│ │Results│ │Report│ │ Chat │ │
│  │Panel │ │ Panel │ │+Errors │ │ Panel │ │Panel │ │Panel │ │
│  └──────┘ └───────┘ └────────┘ └───────┘ └──────┘ └──────┘ │
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
│   ├── main.py                 # FastAPI entry point (~3617 lines)
│   │                           # - REST API endpoints
│   │                           # - WebSocket real-time communication
│   │                           # - Session management & persistence
│   │                           # - Chunked file upload (10GB support)
│   │                           # - Report generation task scheduling
│   │                           # - Base64 stripping for message optimization
│   │                           # - PI Agent chat (figure display + generation)
│   │                           # - Session export/import (tar.gz/zip/JSON)
│   │
│   ├── workflow.py             # LangGraph workflow (~1685 lines)
│   │                           # - Multi-Agent coordination
│   │                           # - Parallel execution (Fan-out/Fan-in)
│   │                           # - Human-in-the-loop (wait_for_user_input)
│   │                           # - Retry with exponential backoff
│   │                           # - Convergence detection
│   │                           # - State version sync callbacks
│   │                           # - File upload handling during pause
│   │                           # - Round context enrichment (previous findings + code history)
│   │                           # - annotate_clusters_callback for literature-based annotation
│   │
│   ├── report_generator.py     # Report generator (~2691 lines)
│   │                           # - 9 chapters, 40+ analysis items
│   │                           # - Jupyter code execution
│   │                           # - Chart/table generation (PNG/PDF/CSV)
│   │                           # - Cluster annotation table extraction
│   │
│   ├── regenerate_report_html.py # Report HTML regeneration utility
│   │
│   ├── config.py               # Configuration management (~288 lines)
│   │                           # - APIConfig: LLM keys + proxy (google_api_key moved to literature_config)
│   │                           # - ModelConfig: Model selection per agent
│   │                           # - AgentTemperatures: Per-agent temperatures
│   │                           # - AgentPrompts: System prompts (English)
│   │                           # - SystemConfig: System parameters
│   │                           # - Data directory path definitions
│   │                           # - Literature agent defaults (mode, max_search_rounds)
│   │
│   ├── agents/
│   │   ├── base.py             # Agent base class
│   │   ├── pi_agent.py         # PI Agent (coordinator + figure reviewer)
│   │   ├── bioinformatics_agent.py  # Bioinformatics Agent (autonomous)
│   │   ├── pathology_agent.py  # Pathology Agent
│   │   ├── biochemistry_agent.py    # Biochemistry Agent (3-step v2)
│   │   ├── literature_agent.py # Literature Agent (2 research modes + cluster annotation, ~940 lines)
│   │   └── painter_agent.py    # Painter Agent (graphic abstract)
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic data models (~388 lines)
│   │                           # - AnalysisState: Core state with version control + literature_config + chat_messages
│   │                           # - AgentError: Structured error tracking
│   │                           # - BiochemistryResult/File: Biochemistry data models
│   │                           # - PublicationFigure/Table: Publication outputs
│   │                           # - GraphicAbstract: Painter agent output
│   │                           # - BioinformaticsCheckpoint: Step-level recovery
│   │                           # - WebSocketMessage: annotation_result type added
│   │
│   ├── utils/
│   │   ├── llm_client.py       # LLM client (~861 lines, OpenRouter + Google DR)
│   │   ├── sandbox.py          # Jupyter execution environment (~768 lines)
│   │   ├── image_utils.py      # Image processing
│   │   ├── upload.py           # Chunked upload manager (~398 lines)
│   │   └── llm_logger.py       # LLM logging
│   │
│   └── static/                 # Static files (backup)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main application component (+ Chat tab)
│   │   ├── components/         # React components
│   │   │   ├── ChatPanel.tsx   # PI Agent free-form chat (~466 lines)
│   │   │   └── ...             # Other panels (Analysis, Data, Prompts, Results, Report)
│   │   ├── stores/             # Zustand state (+ chatMessages, chatLoading)
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
      ▼                                   │
  wait_for_user_input  ◄── 2min timer     │
      │                    (frontend)     │
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
7. **Human-in-the-Loop**: `wait_for_user_input` node blocks between rounds via `asyncio.Event`. Frontend shows `RoundFeedbackDialog` with 2-minute wall-clock countdown (works correctly in background tabs). Users can provide text guidance, upload files (targeted to specific agents), or skip. Backend has 10-minute safety timeout. User input or file uploads on the last round extends `max_rounds` by 1.
8. **File Upload During Pause**: Users can upload pathology images while workflow is in `waiting_input` or `completed` state. Files are tracked in `pending_uploaded_files` and processed in the next round.
9. **Round Context Enrichment**: `_build_previous_findings_context()` and `_build_code_history_summary()` methods provide each agent with previous round findings and code execution history, preventing redundant analyses in multi-round workflows.
10. **Literature-based Cluster Annotation**: `_annotate_clusters_callback()` routes bioinformatics agent's `annotate_clusters` tool calls to the literature agent for efficient batch cluster annotation.

**WebSocket Message Types:**

| Type | Purpose |
|------|---------|
| `status` | Status updates |
| `state_sync` | Version control and state synchronization |
| `parallel_errors` | Structured error list from parallel execution |
| `message` | General agent messages |
| `progress` | Progress updates |
| `result` | Analysis results |
| `error` | Error messages |
| `round_summary` | End-of-round summary |
| `integrated_report` | Complete analysis report |
| `graphic_abstract` | Painter Agent output |
| `detailed_report` | Detailed report output |
| `figure_review` | PI reviewing generated figures |
| `figure_approved` | Figure approved |
| `figure_revision` | Revision requested |
| `completed` | Analysis complete (includes round_results + final_summary on reconnect) |
| `message_history` | Batch message restore (reconnect to completed session) |
| `human_input_requested` | Workflow waiting for user input (round feedback) |
| `human_input_received` | User input received, continuing workflow |
| `annotation_result` | Literature-based cluster annotation result |
| `data_request` | Agent requesting additional data |

### 1.1 Biochemistry Agent (v2 - 3-Step Workflow)

Since v0.5.1, the biochemistry agent uses a 3-step analysis pipeline:

1. **Step 1**: Process structured indicators and data files (CSV/XLSX)
2. **Step 2**: Full visual analysis of images (comprehensive prompt, not strict JSON extraction)
3. **Step 3**: Summarize findings, draw conclusions, and suggest follow-up experiments

This approach is more robust against vision API inconsistencies (see Known Issues).

### 1.2 Literature Agent (2 Research Modes + Cluster Annotation, ~940 lines)

The Literature Agent supports 2 research modes with automatic fallback:

| Mode | API | Model/Endpoint | Max Runtime |
|------|-----|----------------|-------------|
| `gemini_deep_research` | Google Interactions API | `deep-research-pro-preview-12-2025` | 30 min |
| `pubmed_llm` (default) | NCBI E-utilities + LLM | PubMed search + any LLM model | ~2 min |

**Fallback chain**: Configured mode → PubMed + LLM → Quick search (LLM knowledge only)

**Multi-round search**: Both modes support iterative refinement (configurable 1-20 rounds). After each round, LLM evaluates whether results are sufficient and generates refined queries if needed.

**Per-session configuration**: Literature mode, Google API key, PubMed email, and max search rounds are stored in `AnalysisState.literature_config` (not global `APIConfig`). The `GoogleDeepResearchClient` is created lazily inside the agent only when a Google API key is provided.

**Key methods:**
- `_multi_round_gemini()`: Google Interactions API (start + poll, 5s interval)
- `_multi_round_pubmed()`: E-Search → E-Fetch → XML parsing → LLM synthesis
- `_evaluate_search_results()` / `_evaluate_pubmed_results()`: LLM judges sufficiency
- `annotate_clusters()`: Two-phase cluster annotation — single literature search + LLM matching for all clusters at once. Called by bioinformatics agent via `annotate_clusters_callback`

### 1.3 PI Agent Chat (main.py, Chat Endpoints)

Free-form conversational interface with the PI Agent, separate from the workflow analysis:

**Endpoint**: `POST /api/session/{session_id}/chat`

**Features:**
- Accepts text message + file uploads with per-file agent targeting (PI, Pathology, Bioinformatics, Biochemistry)
- Builds analysis context from session state (round results, figures, literature reports) for informed responses
- **Figure Display**: PI references existing figures via `[[FIGURE:name]]` markers — matched against on-disk figure inventory (workspace + reports directories)
- **Figure Generation**: PI requests new figures via `[[GENERATE_FIGURE:description]]` — system spins up Jupyter sandbox, generates code via LLM, executes it, captures PNG/PDF output
- File routing: Uploaded images → `pathology_images` or `biochemistry_files` based on target agent
- Chat history stored in `AnalysisState.chat_messages` (last 20 messages used as LLM context)
- Vision API used when images are attached

**Key functions:**
- `_build_chat_context()`: Builds analysis summary + scans disk for available PNG figures
- `_generate_figures_in_chat()`: LLM writes code → sandbox executes → captures figures (neutralizes `plt.show/close`)
- `chat_with_pi()`: Main endpoint handler — file saving, context building, LLM call, figure post-processing

### 1.4 Session Export/Import (main.py)

**Export** (`POST /api/session/{session_id}/export`):
- Creates tar.gz archive on disk (temp file, not in memory) containing:
  - `manifest.json` with session_id, version, data_base_dir
  - `sessions/{id}.json` — complete state
  - `states/{id}*` — Jupyter notebooks (excludes .pkl for size)
  - `uploads/{id}/` — all uploaded files
  - `workspace/{id}/` — Jupyter working directory
  - `reports/{id}/` — generated reports
- Returns `FileResponse` with background cleanup of temp file
- Safe filename generation from session name

**Import** (`POST /api/session/import`):
- Supports tar.gz (streamed to disk first), zip, and plain JSON formats
- Path remapping: replaces archived `data_base_dir` with current installation's `DATA_BASE_DIR`
- Session ID validation (no path traversal)
- Extracts files to correct directories, loads state into memory

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

### 4. Report Generator (report_generator.py, ~2691 lines)

**9 Chapters + Cluster Annotation Table:**

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

**Cluster Annotation Extraction**: After loading adata in `_auto_load_processed_data()`, runs sandbox code to extract `leiden → cell_type` mapping from `obs` columns. Stored in `self.cluster_annotations` dict, rendered as HTML table in report, and included in LLM prompts so narrative uses cell type names instead of cluster numbers.

### 5. LLM Client (llm_client.py, ~861 lines)

**Three client classes:**

```python
class OpenAICompatibleClient:
    """Primary LLM client for OpenRouter/OpenAI-compatible APIs"""
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    async def chat(self, model, messages, temperature=0.7, max_tokens=4096,
                   stream=False, timeout=None) -> LLMResponse:
        # Standard chat with per-request timeout override
    async def chat_with_tools(self, model, messages, tools, ...) -> LLMResponse:
        # Function calling support
    async def chat_stream(self, model, messages, ...) -> AsyncGenerator:
        # Streaming response
    async def chat_with_vision(self, model, text_prompt, images, ...) -> LLMResponse:
        # Vision/multimodal support
    def _extract_content_from_message(self, message, response=None) -> str:
        # Robust content extraction from various response formats

class GoogleDeepResearchClient:
    """Client for Google Gemini Deep Research via Interactions API"""
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    async def start_research(self, query) -> str:
        # Start background research task, returns interaction_id
    async def poll_research(self, interaction_id, poll_interval=5.0, max_wait=1800.0):
        # Poll until completed/failed (30min max)
    async def research(self, query) -> Dict:
        # Convenience: start + poll, returns {report, sources, thinking}

class LLMManager:
    """Manager wrapper, provides .llm / .openrouter aliases"""
    # Note: no longer creates GoogleDeepResearchClient; that is now created
    # lazily inside LiteratureAgent when google_api_key is provided
```

**Key Features:**

- `_model_supports_temperature()`: Module-level function that skips temperature for reasoning models (`o1`/`o3`/`o4` series); these models use `max_completion_tokens` instead of `max_tokens`
- `_extract_content_from_message()`: Robust content extraction handling multiple formats — string content, list content (multimodal), None content with model_dump fallback, reasoning model responses (`reasoning`/`reasoning_content` fields), refusal detection, and debug logging for empty content with non-zero completion tokens
- `retry_with_backoff()`: Max 3 retries with exponential backoff (2x delay), rate limit handling
- Per-request `timeout` override: Deep research models pass `timeout=660.0` without changing client default (120s)

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
| `/api/session/{id}/literature-config` | POST | Save literature agent configuration (mode, keys, rounds) |
| `/api/session/{id}/resume` | POST | Resume analysis (paused/completed/waiting_input) |
| `/api/session/{id}/chat` | POST | Chat with PI Agent (supports file uploads with per-file agent targeting) |
| `/api/session/{id}/chat` | GET | Get chat history for a session |
| `/api/session/{id}/export` | POST | Export complete session as tar.gz archive |
| `/api/session/import` | POST | Import session from tar.gz/zip/JSON archive |
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
  | 'state_sync'        // Version control sync
  | 'parallel_errors'   // Structured error list
  | 'completed'         // Analysis complete (with round_results on reconnect)
  | 'message_history'   // Batch message restore (reconnect)
  // Analysis process
  | 'message'           // General message
  | 'progress'          // Progress update
  | 'result'            // Analysis result
  | 'error'             // Error message
  // Results
  | 'round_summary'     // Round summary
  | 'integrated_report' // Integrated report
  | 'graphic_abstract'  // Graphic abstract
  | 'detailed_report'   // Detailed report
  // Figure review
  | 'figure_review'     // PI reviewing figure
  | 'figure_approved'   // Figure approved
  | 'figure_revision'   // Revision requested
  // Cluster annotation
  | 'annotation_result' // Literature-based cluster annotation result
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
    status: TaskStatus  # pending, running, completed, failed, paused, waiting_input (new in v0.6.0)
    initial_instruction: str

    # Version control
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
    graphic_abstract: Optional[GraphicAbstract]
    integrated_report: Optional[str]

    # Error tracking
    agent_errors: List[AgentError]

    # Human-in-the-loop
    pending_human_input: Optional[str]  # User message waiting to be processed
    pending_uploaded_files: List[Dict]  # Files uploaded during pause
    max_rounds: int = 0                 # Dynamic max rounds (0 = use config default)

    # Literature config (per-session, replaces global google_api_key)
    literature_config: Dict[str, Any]   # mode (gemini_deep_research|pubmed_llm), google_api_key, pubmed_email, max_search_rounds

    # Chat messages (PI Agent free-form chat, separate from workflow)
    chat_messages: List[Dict[str, Any]] = []  # role, content, timestamp, files, figures

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

### AgentError

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

### GraphicAbstract

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
| `sandbox_timeout` | 1200 | Code execution timeout (seconds) |
| `image_tile_size` | 1024 | Pathology image tile size |
| `image_overlap` | 64 | Tile overlap pixels |
| `blank_threshold` | 0.95 | Threshold for detecting blank tiles |

**Literature Agent Defaults:**

| Config | Default | Description |
|--------|---------|-------------|
| `literature_mode` | `pubmed_llm` | Research mode (`gemini_deep_research` or `pubmed_llm`) |
| `max_search_rounds` | 5 | Max iterative search rounds (1-20) |

**LLM Model Defaults:**

| Agent | Default Model |
|-------|--------------|
| PI | `google/gemini-3-pro-preview` |
| Pathology | `google/gemini-3-pro-preview` |
| Bioinformatics | `anthropic/claude-opus-4.6` |
| Biochemistry | `google/gemini-3-pro-preview` |
| Painter | `google/gemini-3-pro-image-preview` |
| Literature | `google/gemini-3-pro-preview` (chat); Deep Research via mode config |

**Agent Temperatures:**

| Agent | Default | Rationale |
|-------|---------|-----------|
| PI | 0.5 | Balanced creativity for synthesis |
| Pathology | 0.2 | Objective, consistent analysis |
| Bioinformatics | 0.1 | Highly deterministic code generation |
| Biochemistry | 0.2 | Objective interpretation |
| Literature | 0.6 | Broader exploration |
| Painter | 0.7 | Creative image generation |

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

**5. State Version Conflicts**

- If frontend shows outdated data, check `state_version` in session JSON
- Frontend detects conflicts when received version < current version
- Resolution: Trigger full session reload via `loadSession()`

**6. Parallel Analysis Errors**

- Errors are tracked per-agent in `agent_errors` list
- Check `AnalysisPanel` for error display
- Errors persist until manually resolved
- Debug via: `sessions/{id}.json` → `agent_errors` array

**7. Vision API Empty Responses**

- Gemini via OpenRouter may return empty content with high token count
- Affects pathology and biochemistry image analysis
- Biochemistry Agent v2 mitigates this with comprehensive prompts
- Check `intesc_data/logs/error.log` for details

**8. Stale Session After Server Restart**

- If browser shows blank/error state after server restart, the session may no longer exist on server
- v0.6.1+ auto-detects this and clears stale session from localStorage
- Manual fix: Clear localStorage in browser DevTools → Application → Local Storage → delete `intesc-storage`

**9. Deep Research Timeout**

- OpenRouter deep research models can take up to 10 minutes per request
- v0.6.1+ uses per-request `timeout=660s` for deep research calls (vs default 120s)
- If timeout errors persist, check network/proxy connection and OpenRouter service status

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.109.0 | Web framework |
| langgraph | >=0.0.40 | Workflow orchestration |
| openai | >=1.12.0 | LLM API (OpenAI-compatible) |
| pydantic | >=2.5.0 | Data models |
| jupyter_client | 8.6.0 | Code execution |
| dill | >=0.3.7 | State serialization |

### Bioinformatics Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scanpy | >=1.9.6 | Single-cell analysis core |
| anndata | >=0.10.3 | Data structures |
| muon | >=0.1.5 | Multi-omics data |
| liana | >=1.0.0 | Cell communication analysis |
| scvelo | >=0.3.1 | RNA velocity |
| cellrank | >=2.0.0 | Cell fate analysis |
| decoupler | >=1.5.0 | Pathway activity |
| gseapy | >=1.1.0 | GO/KEGG enrichment |
| celltypist | >=1.6.0 | Cell annotation |
| scvi-tools | >=1.0.0 | Deep learning models |
| pySCENIC | >=0.12.0 | Gene regulation |

### Frontend Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| react | ^18.2.0 | UI framework |
| typescript | ^5.3.3 | Type safety |
| zustand | ^4.5.0 | State management |
| vite | ^5.0.12 | Build tool |
| tailwindcss | ^3.4.1 | Styling |
| recharts | ^2.10.4 | Charts |
| react-markdown | ^9.0.1 | Markdown rendering |
| react-syntax-highlighter | ^15.5.0 | Code highlighting |
| react-dropzone | ^14.2.3 | File upload |

## Key Code Locations

### Backend

| Feature | File | Approximate Size |
|---------|------|-----------------|
| FastAPI server, session mgmt, chat & export/import | `main.py` | ~3617 lines |
| Report generator (9 chapters) | `report_generator.py` | ~2691 lines |
| Workflow graph & orchestration | `workflow.py` | ~1685 lines |
| Bioinformatics Agent (autonomous) | `bioinformatics_agent.py` | ~1496 lines |
| PI Agent (coordinator + reviewer) | `pi_agent.py` | ~1330 lines |
| Literature Agent (2 modes + annotation) | `literature_agent.py` | ~940 lines |
| LLM client (OpenRouter + Google DR) | `llm_client.py` | ~861 lines |
| Jupyter sandbox | `sandbox.py` | ~768 lines |
| Biochemistry Agent (3-step v2) | `biochemistry_agent.py` | ~749 lines |
| Pathology Agent | `pathology_agent.py` | ~450 lines |
| Chunked upload manager | `upload.py` | ~398 lines |
| Painter Agent (graphic abstract) | `painter_agent.py` | ~326 lines |
| Configuration & prompts | `config.py` | ~288 lines |
| Agent base class | `base.py` | ~246 lines |

### Frontend

| Feature | File | Approximate Size |
|---------|------|-----------------|
| API hooks + WebSocket | `useApi.ts` | ~1172 lines |
| Data upload interface | `DataPanel.tsx` | ~965 lines |
| Analysis control + error display | `AnalysisPanel.tsx` | ~797 lines |
| Report generation & download | `ReportPanel.tsx` | ~789 lines |
| Real-time message streaming | `MessageStream.tsx` | ~778 lines |
| Session mgmt + export/import | `Sidebar.tsx` | ~598 lines |
| Agent prompt + literature config | `PromptsPanel.tsx` | ~509 lines |
| Round feedback dialog | `RoundFeedbackDialog.tsx` | ~479 lines |
| Zustand global state | `appStore.ts` | ~470 lines |
| PI Agent chat | `ChatPanel.tsx` | ~466 lines |
| Main application | `App.tsx` | ~309 lines |
| Round results display | `ResultsPanel.tsx` | ~265 lines |
| API + literature configuration | `ConfigPanel.tsx` | ~250 lines |

## Known Issues

### 1. Vision API Empty Content

**Symptom**: Gemini via OpenRouter sometimes returns empty content despite high completion token count (e.g., 4093 tokens but content=None).

**Affected Agents**: Pathology, Biochemistry (image analysis steps)

**Root Cause**: Likely OpenRouter relay or response format mismatch with Gemini vision API.

**Workaround**: Biochemistry Agent v2 uses comprehensive prompts instead of strict JSON extraction for more robustness. Retry mechanisms with exponential backoff help mitigate intermittent failures.

### 2. Large Memory Usage

Single-cell datasets can be very large. The local Jupyter kernel may require significant memory (8GB+) for larger datasets. dill serialization of the full Python environment can also produce large `.pkl` files.

### 3. localStorage Overflow (Fixed in v0.6.1)

**Symptom**: Page refresh fails to load session; browser console shows `QuotaExceededError`.

**Root Cause**: Zustand persist stores state to localStorage (5MB limit). Large sessions with many base64 images overflow.

**Fix**: Base64 data is now stripped from persisted messages and session objects in `appStore.ts` partialize function. API response also strips base64 via `_strip_message_base64()` in `main.py`.

### 4. Stale Session 404 Loop (Fixed in v0.6.1)

**Symptom**: After server restart or session cleanup, frontend enters infinite 404 + WebSocket reconnect loop.

**Root Cause**: Browser localStorage has a session ID that no longer exists on server. Every API call returns 404, WebSocket gets "Session not found" error, reconnect fires, repeat.

**Fix**: Added stale session detection in 3 places: Sidebar auto-reload, WebSocket `onmessage`, and WebSocket `onopen` reconnect. On 404 detection, clears session from Zustand state.

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.7.2 | 2026-02-11 | Bug fixes for chat file handling, sandbox message filtering, import path validation |
| 0.7.1 | 2026-02-11 | ChatPanel UX improvements (per-file agent selector, resizable textarea), Sidebar export/import UX polish |
| 0.7.0 | 2026-02-11 | Chat tab (PI Agent free-form chat with figure display/generation), session export/import (tar.gz), session browser |
| 0.6.3 | 2026-02-11 | Literature-based cluster annotation tool, round context enrichment, per-session literature config, ConfigPanel redesign, LLM reasoning model support, timer wall-clock fix |
| 0.6.2 | 2026-02-10 | Literature Agent rewrite (2 modes), annotate_clusters method, cluster annotations in report, session recovery, localStorage overflow fix, read mode |
| 0.6.1 | 2026-02-10 | Bug fixes (partialize null safety, session not found, report dict keys) |
| 0.6.0 | 2026-02-09 | Human-in-the-loop round feedback dialog, workflow wait_for_user_input node, cell annotation + report fixes |
| 0.5.5 | 2026-02-09 | WebSocket reconnect with full session data, report loading retry, annotation workflow improvements |
| 0.5.4 | 2026-02-09 | Bug fixes (bioinformatics, session handling, sandbox, UI) |
| 0.5.3 | 2026-02-09 | Report generation and bioinformatics enhancements |
| 0.5.1 | 2026-02-08 | Biochemistry Agent v2 (3-step workflow), major refactoring |
| 0.5.0 | 2026-02-06 | Real-time message optimization |
| 0.4.1 | 2026-02-05 | Painter agent improvements, bug fixes |
| 0.4.0 | 2026-02-05 | Report generation rebuilt |
| 0.3.0 | 2026-02-04 | Parallel execution, Painter Agent, state versioning, error tracking |
| 0.2.x | 2026-01 | Report features development |
| 0.1.x | 2026-01 | Initial functional version |

## Contact

For issues, please contact the project maintainers.

---

*Document updated: 2026-02-11*
*Version: 0.7.2*
