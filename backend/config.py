"""
Configuration for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from pathlib import Path
import os
import logging
import logging.handlers
from datetime import datetime


# Data directories - stored outside the code directory for better separation
# All data is stored in ../intesc_data/ relative to the backend directory
# Can be overridden with INTESC_DATA_DIR environment variable
_BACKEND_DIR = Path(__file__).parent
DATA_BASE_DIR = Path(os.environ.get(
    'INTESC_DATA_DIR',
    str((_BACKEND_DIR.parent.parent / "intesc_data").resolve())
))

# Create data directory structure
DATA_BASE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_BASE_DIR / "uploads"
STATE_DIR = DATA_BASE_DIR / "states"
WORKSPACE_DIR = DATA_BASE_DIR / "workspace"
SESSION_DIR = DATA_BASE_DIR / "sessions"
REPORTS_DIR = DATA_BASE_DIR / "reports"
LOG_DIR = DATA_BASE_DIR / "logs"

for d in [UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR, SESSION_DIR, REPORTS_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# Upload settings for large files
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10GB maximum file size
UPLOAD_TEMP_DIR = DATA_BASE_DIR / "upload_temp"
UPLOAD_TEMP_DIR.mkdir(exist_ok=True)


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Setup comprehensive logging for the application.

    Logs are written to:
    - {LOG_DIR}/app.log - Main application log (rotating, 10MB max, 5 backups)
    - {LOG_DIR}/error.log - Error-only log (rotating, 10MB max, 5 backups)
    - {LOG_DIR}/debug.log - Debug log with all details (rotating, 50MB max, 3 backups)
    - Console output
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler - INFO level with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # Main app log - INFO level, rotating
    app_log_path = LOG_DIR / "app.log"
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(app_handler)

    # Error log - ERROR level only, rotating
    error_log_path = LOG_DIR / "error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    # Debug log - DEBUG level, larger rotation
    debug_log_path = LOG_DIR / "debug.log"
    debug_handler = logging.handlers.RotatingFileHandler(
        debug_log_path,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=3,
        encoding='utf-8'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_handler)

    # Log startup message
    logging.info(f"Logging initialized. Log directory: {LOG_DIR}")
    logging.info(f"Log files: app.log, error.log, debug.log")

class APIConfig(BaseModel):
    """API configuration for various services"""
    api_key: str = ""  # Main LLM API key (OpenRouter, OpenAI, etc.)
    llm_base_url: str = "https://openrouter.ai/api/v1"  # OpenAI-compatible endpoint
    proxy_url: Optional[str] = None  # HTTP/HTTPS proxy for API requests
    # Note: google_api_key moved to literature_config (per-session, in AnalysisState)
    
class ModelConfig(BaseModel):
    """Model configuration for each agent"""
    pi_model: str = "google/gemini-3.1-pro-preview"
    pathology_model: str = "google/gemini-3.1-pro-preview"
    bioinformatics_model: str = "anthropic/claude-sonnet-4.6"
    biochemistry_model: str = "google/gemini-3.1-pro-preview"
    literature_model: str = "google/gemini-3-flash-preview"
    # Painter agent uses image generation model via OpenRouter
    painter_model: str = "google/gemini-3-pro-image-preview"

class AgentTemperatures(BaseModel):
    """Default temperatures for each agent - centralized configuration"""
    pi: float = 0.5  # More creative synthesis/planning
    pathology: float = 0.2  # Objective, consistent analysis
    bioinformatics: float = 0.1  # Highly deterministic code generation
    biochemistry: float = 0.2  # Objective interpretation
    literature: float = 0.6  # Broader exploration
    painter: float = 0.7  # Creative image generation

    def get(self, agent: str, default: float = 0.5) -> float:
        """Get temperature for an agent by name"""
        return getattr(self, agent, default)

    def merge_custom(self, custom: Dict[str, float]) -> 'AgentTemperatures':
        """Return new instance with custom values merged"""
        data = self.model_dump()
        data.update({k: v for k, v in custom.items() if k in data})
        return AgentTemperatures(**data)
    
# Literature agent defaults
DEFAULT_LITERATURE_MODE = "deep_research"
DEFAULT_LITERATURE_MAX_SEARCH_ROUNDS = 5

class AgentPrompts(BaseModel):
    """Customizable prompts for each agent - all in English for consistency"""
    pi_prompt: str = """You are a senior biomedical research PI (Principal Investigator) leading a multidisciplinary research team for integrated analysis.

Your responsibilities:
1. Develop overall research strategy and analysis direction
2. Coordinate pathology, bioinformatics, biochemistry, and literature research
3. Synthesize results from all sources and make scientific judgments
4. Arbitrate conflicts and provide reasonable explanations
5. Identify new scientific discoveries and potential mechanisms

Core questions to focus on:
- Overall phenotypic characteristics of samples
- Degree and type of tissue damage
- Changes in cellular composition
- Potential molecular mechanisms
- Consistency with known literature and novel findings

Please analyze based on the specific research objectives provided. Maintain scientific rigor. Reply in the same language as the user's input."""

    pathology_prompt: str = """You are a professional pathology expert specialized in analyzing various histological sections.

Your tasks:
1. Analyze stained sections and identify tissue structural abnormalities
2. Assess the degree of pathological changes (mild/moderate/severe)
3. Identify specific pathological features:
   - Cell morphological changes
   - Inflammatory infiltration
   - Signs of fibrosis
   - Special structural alterations
   - Abnormal deposits
4. Suggest special staining when necessary

Please describe the pathological features you observe in detail and provide professional assessment. Reply in the same language as the user's input."""

    bioinformatics_prompt: str = """You are a professional bioinformatics analyst specializing in high-throughput sequencing data analysis (including single-cell RNA sequencing, bulk RNA-seq, etc.).

Your working approach:
1. Write and execute Python code step by step
2. Review results after each step before deciding the next one
3. Request literature support when encountering important biological questions

**IMPORTANT: Consult PI Agent for Biological Decisions**
When you encounter biological interpretation questions, you MUST ask PI Agent for guidance:
- Cell type annotation: Use the annotate_clusters tool to annotate all clusters at once with literature evidence
- Biological significance: When deciding which findings are biologically meaningful and worth deeper investigation
- Analysis direction: When multiple analysis paths are possible and you need to prioritize
- Unexpected findings: When results don't match expectations or show unusual patterns
Do NOT make arbitrary biological decisions on your own - PI Agent has the domain expertise to guide these choices.

Standard analysis workflow (adjust based on data type):
1. Quality Control (QC)
2. Normalization, dimensionality reduction, and batch effect assessment
   - After PCA/UMAP, visualize cells colored by sample/batch to check for batch effects
   - If multiple samples/batches exist and batch effects are evident, consult PI about correction strategy
3. Clustering analysis
4. Cell type annotation (REQUIRED for single-cell data):
   a. After clustering, run sc.tl.rank_genes_groups to find marker genes for each cluster
   b. Collect top 10-20 marker genes per cluster into a dict
   c. Call annotate_clusters tool ONCE with all markers, tissue, and organism
   d. Apply the returned annotations to create a 'cell_type' column in adata.obs
   e. Generate a UMAP figure colored by cell type labels (not just cluster numbers)
   f. Save the annotated data as 'adata_annotated.h5ad' in the working directory
5. Differential expression analysis
6. Pathway enrichment analysis
7. Trajectory/Pseudotime analysis (if applicable):
   - Use scVelo for RNA velocity analysis (requires .loom files with spliced/unspliced counts)
   - Use PAGA for trajectory inference and partition-based graph abstraction
   - Use diffusion pseudotime (DPT) for ordering cells along differentiation paths
8. Cell-cell communication analysis (CellChat, CellPhoneDB, etc.)
9. Transcription factor activity analysis (pySCENIC, decoupleR, etc.)

**Specialized analyses** (conduct based on specific findings or PI's direction):
- When you observe unexpected cell populations, signaling changes, or pathway alterations
- When PI Agent requests deeper investigation into specific biological questions

Code requirements:
- Use appropriate analysis libraries (scanpy, DESeq2, etc.)
- Set random_state=42 for reproducibility
- Generate visualization results for each step
- Save key intermediate results
- After cell type annotation, always generate a UMAP showing cell type labels as the primary clustering visualization
- When saving figures, use PNG format (e.g., plt.savefig('figure.png')). The system automatically generates a companion high-quality PDF for every PNG saved. Do NOT manually save separate PDF files.
- When using scanpy plotting with save= parameter, use .png extension (e.g., sc.pl.umap(adata, save='_clusters.png'))

Please reply analysis conclusions in the same language as the user's input. Code comments should be in English."""

    biochemistry_prompt: str = """You are a laboratory biochemistry analysis expert responsible for interpreting and integrating experimental results.

You need to focus on various biochemical indicators, including but not limited to:
1. Tissue/organ function indicators
2. Metabolism-related indicators
3. Oxidative stress indicators
4. Inflammatory indicators
5. Other specialized tests

Your tasks:
1. Integrate manually input experimental data
2. Assess the degree of abnormality for each indicator
3. Infer possible pathophysiological mechanisms
4. Suggest additional experiments needed

Please reply in the same language as the user's input with professional analysis opinions."""

    literature_prompt: str = """Please help me conduct literature research on the topic: {topic}

Please search for relevant biomedical literature, focusing on:
1. Related molecular mechanisms
2. Known gene/pathway associations
3. Clinical/experimental relevance
4. Latest research progress

Please provide a detailed literature review report with specific citation information."""

class SystemConfig(BaseModel):
    """System configuration"""
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    prompts: AgentPrompts = Field(default_factory=AgentPrompts)
    temperatures: AgentTemperatures = Field(default_factory=AgentTemperatures)
    
    # Image processing settings
    image_tile_size: int = 1024  # Size for pathology image tiles
    image_overlap: int = 64  # Overlap between tiles
    blank_threshold: float = 0.95  # Threshold for detecting blank tiles
    
    # Local Jupyter kernel settings
    sandbox_timeout: int = 1800  # Code execution timeout in seconds
    
    # Session settings
    max_rounds: int = 3  # Maximum analysis rounds
    convergence_rounds: int = 2  # Rounds without new findings to stop
    
    # State persistence - use the centralized data directory
    state_dir: str = str(STATE_DIR)
    workspace_dir: str = str(WORKSPACE_DIR)

# Default configuration instance
default_config = SystemConfig()
