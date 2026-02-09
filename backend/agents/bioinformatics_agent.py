"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import json
import logging
import base64
import io
import uuid
import re
from pathlib import Path
from datetime import datetime

import aiofiles

from .base import BaseAgent, AgentConfig

# Configure logger
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages step-level checkpoints for bioinformatics analysis.
    Enables recovery from interruptions during the 80-step analysis loop.
    """

    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N steps

    def __init__(self, session_id: str, state_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            session_id: Session ID for this analysis
            state_dir: Directory to store checkpoint files
        """
        self.session_id = session_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.state_dir / f"{session_id}_bioinfo_checkpoint.json"

    async def save(self, checkpoint: Dict[str, Any]):
        """
        Save checkpoint to disk atomically.

        Args:
            checkpoint: Checkpoint data to save
        """
        checkpoint['updated_at'] = datetime.now().isoformat()

        # Atomic write: write to temp file then rename
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(checkpoint, indent=2, default=str))
            tmp_path.replace(self.checkpoint_path)
            logger.info(f"[{self.session_id}] Checkpoint saved at step {checkpoint.get('step_number', 0)}")
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to save checkpoint: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    async def load(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from disk.

        Returns:
            Checkpoint data or None if not found
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            async with aiofiles.open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to load checkpoint: {e}")
            return None

    def clear(self):
        """Clear checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"[{self.session_id}] Checkpoint cleared")

    def should_checkpoint(self, step: int) -> bool:
        """
        Check if we should save a checkpoint at this step.

        Args:
            step: Current step number

        Returns:
            True if checkpoint should be saved
        """
        return step > 0 and step % self.CHECKPOINT_INTERVAL == 0
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, CodeExecution,
    PublicationFigure, FigureReviewStatus
)
from ..utils.llm_client import OpenAICompatibleClient, ToolCall
from ..utils.sandbox import LocalJupyterSandbox, CodeExecutionResult


# Define the tools for bioinformatics agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Generate and execute Python code for single-cell analysis. Use for technical steps like data loading, QC, normalization, clustering, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this code does"
                    },
                    "step_type": {
                        "type": "string",
                        "enum": ["technical", "biological"],
                        "description": "Whether this is a technical step (can proceed autonomously) or requires biological interpretation"
                    }
                },
                "required": ["code", "description", "step_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_pi",
            "description": "Request guidance from PI Agent on biological decisions like: cell type annotation, analysis direction, which genes/pathways to analyze, whether to do trajectory analysis, result interpretation",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The biological question or decision that needs PI guidance"
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context from current analysis (e.g., cluster markers, current findings)"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Possible options or choices if applicable"
                    }
                },
                "required": ["question", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "Search literature for biological information like marker genes, cell type signatures, pathway information",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for in literature"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": "Report key findings and conclusions from the analysis. Use this when you have completed a major analysis phase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key findings"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary of the analysis results"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggestions for next steps or further analysis"
                    }
                },
                "required": ["findings", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_publication_figure",
            "description": "Generate a publication-quality figure with PNG thumbnail and high-res PDF. The figure will be automatically reviewed by the PI Agent. Use this for important figures that should be included in the final report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python plotting code using matplotlib, seaborn, or scanpy. Should produce a single figure."
                    },
                    "figure_name": {
                        "type": "string",
                        "description": "Name for the figure file (English, no spaces, e.g., 'umap_cell_types')"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the figure shows (will be used for figure legend)"
                    },
                    "figure_type": {
                        "type": "string",
                        "enum": ["umap", "heatmap", "violin", "dotplot", "barplot", "trajectory", "volcano", "other"],
                        "description": "Type of figure being generated"
                    }
                },
                "required": ["code", "figure_name", "description", "figure_type"]
            }
        }
    }
]


class BioinformaticsAgent(BaseAgent):
    """
    Bioinformatics Agent with Autonomous Mode

    Can autonomously execute technical steps:
    - Data loading, QC, filtering
    - Normalization, HVG selection
    - Dimensionality reduction, clustering
    - Basic visualization

    Requires PI consultation for biological decisions:
    - Cell type annotation
    - Result interpretation
    - Analysis direction choices
    - Gene/pathway selection
    - Trajectory analysis decisions
    """

    # Dangerous code patterns that should not be executed
    DANGEROUS_PATTERNS = [
        # System command execution
        (r'\bos\.system\b', 'os.system calls'),
        (r'\bsubprocess\b', 'subprocess module'),
        (r'\bos\.popen\b', 'os.popen calls'),
        (r'\bos\.spawn\w*\b', 'os.spawn* calls'),
        (r'\bos\.exec\w*\b', 'os.exec* calls'),
        (r'\bos\.fork\b', 'os.fork calls'),
        (r'\bpty\b', 'pty module'),

        # Code execution
        (r'\beval\s*\(', 'eval() function'),
        (r'\bexec\s*\(', 'exec() function'),
        (r'\b__import__\b', '__import__ function'),
        (r'\bcompile\s*\(', 'compile() function'),
        (r'\bimportlib\b', 'importlib module'),

        # Dangerous attribute access
        (r'\b__builtins__\b', '__builtins__ access'),
        (r'\b__class__\s*\.\s*__bases__\b', '__class__.__bases__ access'),
        (r'\b__class__\s*\.\s*__mro__\b', '__class__.__mro__ access'),
        (r'\b__subclasses__\s*\(', '__subclasses__ access'),
        (r'\bglobals\s*\(\s*\)', 'globals() function'),
        (r'\blocals\s*\(\s*\)', 'locals() function'),
        (r'\bgetattr\s*\([^,]+,\s*["\']__', 'getattr with dunder'),
        (r'\bsetattr\s*\([^,]+,\s*["\']__', 'setattr with dunder'),
        (r'\bdelattr\s*\([^,]+,\s*["\']__', 'delattr with dunder'),
        (r'\bvars\s*\(\s*\)', 'vars() function'),

        # File system operations (destructive)
        (r'\bopen\s*\([^)]*["\'][wa]', 'file write operations outside safe directories'),
        (r'\brm\s+-rf\b', 'rm -rf command'),
        (r'\bshutil\.rmtree\b', 'shutil.rmtree'),
        (r'\bos\.remove\b', 'os.remove'),
        (r'\bos\.unlink\b', 'os.unlink'),
        (r'\bos\.rmdir\b', 'os.rmdir'),
        (r'\bos\.makedirs\b', 'os.makedirs (use Path.mkdir instead)'),
        (r'\bos\.rename\b', 'os.rename'),
        (r'\bos\.replace\b', 'os.replace'),
        (r'\bshutil\.move\b', 'shutil.move'),
        (r'\bshutil\.copy\b', 'shutil.copy (use explicit paths)'),

        # Network access
        (r'\bsocket\b', 'socket module'),
        (r'\brequests\.(?:get|post|put|delete|patch|head|options)\b', 'HTTP requests'),
        (r'\burllib\b', 'urllib module'),
        (r'\bhttpx\b', 'httpx module'),
        (r'\baiohttp\b', 'aiohttp module'),

        # Low-level / dangerous modules
        (r'\bctypes\b', 'ctypes module'),
        (r'\bmultiprocessing\b', 'multiprocessing module'),
        (r'\bpickle\.loads?\b', 'pickle.load/loads (arbitrary code execution)'),
        (r'\bcPickle\b', 'cPickle module'),
        (r'\bmarshal\b', 'marshal module'),
        (r'\bshelve\b', 'shelve module'),

        # Environment manipulation (os.environ allowed for setting runtime flags like NUMBA_DISABLE_JIT)
        (r'\bos\.chdir\b', 'os.chdir (changing working directory)'),
        (r'\bos\.chroot\b', 'os.chroot'),
        (r'\bsys\.path\s*\.(?:append|insert|extend)\b', 'sys.path modification'),
    ]
    
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        sandbox: LocalJupyterSandbox,
        literature_callback: Optional[Callable[[str], Awaitable[str]]] = None,
        pi_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        config: Optional[AgentConfig] = None
    ):
        super().__init__(
            agent_type=AgentType.BIOINFORMATICS,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config
        )
        self.sandbox = sandbox
        self.literature_callback = literature_callback
        self.pi_callback = pi_callback
        self.message_callback = message_callback
        self.figure_review_callback = figure_review_callback

        # Track execution history
        self.code_history: List[CodeExecution] = []
        self.last_result: Optional[CodeExecutionResult] = None
        self.analysis_context: Dict[str, Any] = {}

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate code for security before execution.

        Args:
            code: The Python code to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                return False, f"Dangerous pattern detected: {description}"
        return True, ""

    async def _notify(self, message_type: str, content: str):
        """Send notification via callback"""
        if self.message_callback:
            await self.message_callback(message_type, content)
    
    def _build_notebook_context(self) -> str:
        """Build context string from notebook history"""
        if not self.code_history:
            return "Notebook is empty. This is the first code cell."
        
        context_parts = ["=== Notebook History (last 10 cells) ===\n"]
        
        recent_history = self.code_history[-10:]
        start_idx = len(self.code_history) - len(recent_history)
        
        for i, execution in enumerate(recent_history):
            context_parts.append(f"--- Cell {start_idx + i + 1} ---")
            context_parts.append(f"```python\n{execution.code}\n```")
            
            if execution.output:
                output = execution.output[:800]
                if len(execution.output) > 800:
                    output += "\n... (truncated)"
                context_parts.append(f"Output:\n{output}")
            
            if execution.error:
                context_parts.append(f"Error: {execution.error}")
            
            if execution.figures:
                context_parts.append(f"[Generated {len(execution.figures)} figure(s)]")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_data_context(self, state: AnalysisState) -> str:
        """Build context about available data"""
        data_parts = ["=== Available Data ===\n"]
        
        has_data = False
        
        if state.control_seq_path:
            has_data = True
            data_parts.append(f"Control scRNA-seq: {state.control_seq_path}")
            data_parts.append(f"  Load: adata_control = sc.read_h5ad('{state.control_seq_path}')")
            
        if state.treatment_seq_path:
            has_data = True
            data_parts.append(f"Treatment scRNA-seq: {state.treatment_seq_path}")
            data_parts.append(f"  Load: adata_treatment = sc.read_h5ad('{state.treatment_seq_path}')")
            
        if state.control_loom_path:
            has_data = True
            data_parts.append(f"Control Loom (velocity): {state.control_loom_path}")
            
        if state.treatment_loom_path:
            has_data = True
            data_parts.append(f"Treatment Loom (velocity): {state.treatment_loom_path}")
        
        if not has_data:
            data_parts.append("⚠️ No sequencing data uploaded")
        
        return "\n".join(data_parts)
    
    def _build_analysis_context(self) -> str:
        """Build context about current analysis state"""
        if not self.analysis_context:
            return ""
        
        parts = ["=== Current Analysis State ==="]
        for key, value in self.analysis_context.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value[:10])}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    async def run_autonomous_analysis(
        self,
        state: AnalysisState,
        initial_instruction: str,
        max_steps: int = 80,
        figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        resume_from_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Run analysis with autonomous technical steps, PI consultation, and step-level checkpointing.

        Args:
            state: Current analysis state
            initial_instruction: The instruction to execute
            max_steps: Maximum number of steps (default 80 to accommodate figure review loops)
            figure_review_callback: Optional callback for PI to review generated figures
            resume_from_checkpoint: Whether to attempt resuming from checkpoint (default True)

        Returns:
            Dict with analysis results
        """
        # Use provided callback or fall back to instance callback
        review_callback = figure_review_callback or self.figure_review_callback

        # Initialize checkpoint manager for step-level recovery
        if hasattr(self.sandbox, 'state_dir') and self.sandbox.state_dir:
            state_dir = str(self.sandbox.state_dir)
        else:
            # Use sandbox working directory as base for states
            state_dir = str(Path(self.sandbox.working_dir) / "states")
            Path(state_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_mgr = CheckpointManager(session_id=state.id, state_dir=state_dir)

        # Try to resume from checkpoint if enabled
        checkpoint = None
        checkpoint_valid = False
        if resume_from_checkpoint:
            try:
                checkpoint = await checkpoint_mgr.load()
                if checkpoint and not checkpoint.get('completed', False):
                    # Validate checkpoint has required fields with correct types
                    validation_errors = []

                    # Check step_number is a valid integer
                    step_num = checkpoint.get('step_number')
                    if not isinstance(step_num, int) or step_num < 0:
                        validation_errors.append(f"step_number invalid: {step_num}")

                    # Check messages is a non-empty list
                    messages = checkpoint.get('messages')
                    if not isinstance(messages, list) or len(messages) < 2:
                        validation_errors.append(f"messages invalid: expected list with >= 2 items")

                    # Check findings is a list
                    findings = checkpoint.get('findings')
                    if not isinstance(findings, list):
                        validation_errors.append(f"findings invalid: expected list")

                    # Check figures is a list
                    figures = checkpoint.get('figures')
                    if not isinstance(figures, list):
                        validation_errors.append(f"figures invalid: expected list")

                    if not validation_errors:
                        checkpoint_valid = True
                        logger.info(f"[Bioinformatics] Resuming from checkpoint at step {checkpoint['step_number']}")
                        await self._notify("status", f"Resuming analysis from step {checkpoint['step_number']}...")
                    else:
                        logger.warning(f"[Bioinformatics] Checkpoint validation failed: {validation_errors}, starting fresh")
                        checkpoint_mgr.clear()
                        checkpoint = None
            except Exception as e:
                logger.error(f"[Bioinformatics] Failed to load checkpoint: {e}, starting fresh")
                checkpoint_mgr.clear()
                checkpoint = None
                await self._notify("status", "Previous checkpoint corrupted, starting fresh analysis...")

        logger.info(f"[Bioinformatics] Starting autonomous analysis with max_steps={max_steps}")
        # Start sandbox if needed
        if self.sandbox.km is None:
            await self._notify("status", "Starting analysis environment...")
            start_result = await self.sandbox.start()
            logger.info("[Bioinformatics] Sandbox started")
            if start_result.get("restore_failed"):
                logger.warning("[Bioinformatics] Previous session restore failed, starting with fresh environment")
                await self._notify("status", "Note: Previous session could not be restored, starting fresh...")
        
        # Build initial prompt
        system_prompt = f"""{self.system_prompt}

You are conducting single-cell RNA-seq analysis. You have access to tools:
1. execute_code - Run Python code (scanpy, anndata, numpy, pandas, matplotlib available)
2. consult_pi - Ask PI for biological decisions
3. search_literature - Search for marker genes, pathways, biological knowledge
4. report_findings - Report conclusions when analysis phase is complete

WORKFLOW GUIDELINES:
- Technical steps can be done autonomously: data loading, QC, normalization, PCA, clustering, UMAP, basic DE
- For BIOLOGICAL DECISIONS, you MUST use consult_pi:
  * Cell type annotation (what are these clusters?)
  * Which clusters show interesting changes?
  * What genes or pathways to analyze?
  * Whether to do trajectory/velocity analysis?
  * Interpretation of results
  * Which comparisons matter biologically?

{self._build_data_context(state)}

{self._build_notebook_context()}

{self._build_analysis_context()}

**CRITICAL: DATA PREPROCESSING (MUST DO BEFORE PCA)**
Before running sc.tl.pca() or any dimensionality reduction:
1. Check for NaN: print(f"NaN in X: {{np.isnan(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X).sum()}}")
2. If NaN exists, fix it:
   - sc.pp.filter_genes(adata, min_cells=3)
   - adata.X = np.nan_to_num(adata.X, nan=0.0) if dense
   - adata.X.data = np.nan_to_num(adata.X.data, nan=0.0) if sparse
3. NEVER skip this step - PCA will fail with NaN values!

**CRITICAL: REPRODUCIBILITY IS MANDATORY**
- ALWAYS use random_state=42 or seed=42 in EVERY function that accepts it
- Examples: sc.tl.umap(..., random_state=42), sc.tl.leiden(..., random_state=42), 
  sc.tl.pca(..., random_state=42), train_test_split(..., random_state=42)
- NEVER omit the random_state parameter in stochastic functions
- Use n_jobs=1 when determinism is critical
- Print key metrics and results
- Generate informative visualizations
- Use report_findings when completing an analysis phase
"""

        # Initialize or restore state from checkpoint
        if checkpoint and checkpoint_valid:
            messages = checkpoint.get('messages', [])
            # Ensure messages have valid structure (at minimum system and user message)
            if len(messages) < 2:
                logger.warning("[Bioinformatics] Checkpoint messages invalid, using fresh messages")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
                ]
            all_findings = checkpoint.get('findings', [])
            all_figures = checkpoint.get('figures', [])
            step_count = checkpoint.get('step_number', 0)
            self.analysis_context = checkpoint.get('analysis_context', {})
            logger.info(f"[Bioinformatics] Restored state: step={step_count}, findings={len(all_findings)}, figures={len(all_figures)}")
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
            ]
            all_findings = []
            all_figures = []
            step_count = 0

        completed = False
        text_response_count = 0
        MAX_TEXT_RESPONSES = 2

        while step_count < max_steps and not completed:
            step_count += 1

            # Save checkpoint periodically
            if checkpoint_mgr.should_checkpoint(step_count):
                checkpoint_data = {
                    'session_id': state.id,
                    'step_number': step_count,
                    'total_steps': max_steps,
                    'messages': messages,
                    'findings': all_findings,
                    'figures': all_figures,
                    'code_history': [ce.model_dump() if hasattr(ce, 'model_dump') else ce.__dict__
                                     for ce in self.code_history],
                    'analysis_context': self.analysis_context,
                    'completed': False
                }
                await checkpoint_mgr.save(checkpoint_data)
                # Also save sandbox session for Python environment recovery
                await self.sandbox.save_session()
            
            self._set_llm_context()
            
            try:
                response = await self.llm_client.chat_with_tools(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=self.config.temperature,  # Use configured temperature
                    max_tokens=4096
                )
            except Exception as e:
                await self._notify("error", f"LLM call failed: {str(e)}")
                break
            
            if not response.has_tool_calls:
                if response.content:
                    await self._notify("response", response.content)
                    messages.append({"role": "assistant", "content": response.content})

                text_response_count += 1
                if text_response_count >= MAX_TEXT_RESPONSES:
                    logger.info(f"[Bioinformatics] {MAX_TEXT_RESPONSES} text-only responses, exiting loop")
                    break

                # Ask the LLM to continue with tools instead of text
                messages.append({
                    "role": "user",
                    "content": "Please continue the analysis. If you have completed your analysis, "
                               "call the report_findings tool with your key findings. "
                               "Do NOT respond with text only - you must use a tool."
                })
                continue
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                args = tool_call.arguments
                
                if tool_name == "execute_code":
                    code = args.get("code", "")
                    description = args.get("description", "")
                    step_type = args.get("step_type", "technical")

                    # Validate code for security before execution
                    is_safe, security_error = self._validate_code(code)
                    if not is_safe:
                        error_msg = f"Code validation failed: {security_error}"
                        logger.warning(f"[Bioinformatics] {error_msg}")
                        await self._notify("error", error_msg)
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": json.dumps(args)}
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Execution blocked: {error_msg}. Please rewrite the code without using dangerous operations."
                        })
                        continue

                    await self._notify("status", f"[{step_type.upper()}] {description}")
                    await self._notify("code", code)

                    # === Wrap code to auto-save figures in PNG + PDF ===
                    figures_dir = Path(self.sandbox.working_dir) / "figures"
                    figures_dir_repr = repr(str(figures_dir))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    wrapped_code = f'''
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

# Set PDF vectorized fonts
plt.rcParams.update({{
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}})

# Create figures directory
_figures_dir = Path({figures_dir_repr})
_figures_dir.mkdir(exist_ok=True)

# Execute user code
{code}

# Detect and save all active figures
_fig_nums = plt.get_fignums()
for _fig_num in _fig_nums:
    _fig = plt.figure(_fig_num)
    _fig_id = f"fig_{timestamp}_{{_fig_num}}"
    _png_path = _figures_dir / f"{{_fig_id}}.png"
    _pdf_path = _figures_dir / f"{{_fig_id}}.pdf"

    # Save PNG (for preview and web report)
    _fig.savefig(_png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')

    # Save PDF (vector format for publication and editing)
    _fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')

    # Display figure to trigger capture via display_data message
    display(_fig)

    print(f"[Auto-saved] PNG: {{_png_path}}")
    print(f"[Auto-saved] PDF: {{_pdf_path}}")

# Close all figures to free memory
plt.close('all')
'''

                    result = await self.sandbox.execute(wrapped_code)
                    
                    code_exec = CodeExecution(
                        code=code,
                        output=result.output,
                        error=result.error,
                        figures=result.figures,
                        execution_time=0.0
                    )
                    self.code_history.append(code_exec)
                    self.last_result = result
                    
                    if result.output:
                        await self._notify("result", result.output[:2000])
                    for fig in result.figures:
                        await self._notify("figure", fig)
                        all_figures.append(fig)
                    if result.error:
                        await self._notify("error", result.error)
                    
                    tool_result = f"Execution {'successful' if not result.error else 'failed'}.\n"
                    if result.output:
                        tool_result += f"Output:\n{result.output[:1500]}\n"
                    if result.error:
                        tool_result += f"Error: {result.error}\n"
                    if result.figures:
                        tool_result += f"Generated {len(result.figures)} figure(s).\n"
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    
                elif tool_name == "consult_pi":
                    question = args.get("question", "")
                    context = args.get("context", "")
                    options = args.get("options", [])
                    
                    await self._notify("status", f"Consulting PI: {question[:100]}...")
                    await self._notify("pi_question", json.dumps({
                        "question": question,
                        "context": context,
                        "options": options
                    }))
                    
                    pi_response = ""
                    if self.pi_callback:
                        pi_response = await self.pi_callback(question, context)
                        await self._notify("pi_response", pi_response)
                    else:
                        pi_response = "PI guidance not available. Proceed with standard practices."
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"PI Response: {pi_response}"
                    })
                    
                elif tool_name == "search_literature":
                    topic = args.get("topic", "")
                    
                    await self._notify("status", f"Searching literature: {topic[:50]}...")
                    
                    lit_result = ""
                    if self.literature_callback:
                        lit_result = await self.literature_callback(topic)
                        await self._notify("literature", lit_result[:1000])
                    else:
                        lit_result = "Literature search not available."
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Literature findings: {lit_result[:2000]}"
                    })
                    
                elif tool_name == "report_findings":
                    findings = args.get("findings", [])
                    summary = args.get("summary", "")
                    suggestions = args.get("suggestions", [])

                    all_findings.extend(findings)

                    await self._notify("findings", json.dumps({
                        "findings": findings,
                        "summary": summary,
                        "suggestions": suggestions
                    }))

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Findings recorded. Continue or wait for instructions."
                    })

                    completed = True
                    break

                elif tool_name == "generate_publication_figure":
                    code = args.get("code", "")
                    figure_name = args.get("figure_name", f"figure_{len(state.publication_figures) + 1}")
                    description = args.get("description", "")
                    figure_type = args.get("figure_type", "other")

                    # Track figure generation attempts to prevent infinite retry loops
                    if "_figure_attempts" not in self.analysis_context:
                        self.analysis_context["_figure_attempts"] = {}
                    fig_attempts = self.analysis_context["_figure_attempts"]
                    fig_attempts[figure_name] = fig_attempts.get(figure_name, 0) + 1

                    if fig_attempts[figure_name] > 3:
                        logger.warning(f"[Bioinformatics] Skipping figure {figure_name}: exceeded 3 attempts")
                        tool_result = (
                            f"Figure {figure_name} has been skipped after {fig_attempts[figure_name]-1} failed attempts. "
                            "Please move on to the next analysis step or generate a different figure."
                        )
                    else:
                        await self._notify("status", f"Generating publication figure: {figure_name} (attempt {fig_attempts[figure_name]})")

                        # Generate the figure with PI review
                        tool_result = await self._handle_generate_publication_figure(
                            code=code,
                            figure_name=figure_name,
                            description=description,
                            figure_type=figure_type,
                            state=state,
                            review_callback=review_callback
                        )

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

        status = "completed" if completed else "max_steps_reached"

        # Fallback: extract findings from last assistant text if report_findings was never called
        if not all_findings and not completed:
            logger.warning("[Bioinformatics] No findings reported, attempting fallback extraction...")
            last_assistant_content = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_assistant_content = msg["content"]
                    break
            if last_assistant_content:
                all_findings.append(last_assistant_content[:3000])
                status = "completed"
                logger.info("[Bioinformatics] Extracted findings from last text response")

        logger.info(f"[Bioinformatics] Analysis {status}: {step_count} steps executed, {len(all_findings)} findings, {len(all_figures)} figures")

        # Clear checkpoint only on successful completion (preserve for resume on max_steps_reached)
        if completed or status == "completed":
            checkpoint_mgr.clear()

        return {
            "status": status,
            "findings": all_findings,
            "figures": all_figures,
            "steps_executed": step_count,
            "code_history": self.code_history,
            "success": True
        }

    async def execute_step(self, state: AnalysisState, instruction: str) -> Dict[str, Any]:
        """Execute a single analysis step (legacy interface)"""
        return await self.run_autonomous_analysis(state, instruction, max_steps=5)
    
    async def process(self, state: AnalysisState, instruction: str, **kwargs) -> Dict[str, Any]:
        """Main entry point for workflow"""
        # Reset per-run state to avoid pollution from previous rounds
        self.code_history = []
        self.analysis_context = {}
        self.last_result = None

        if not state.control_seq_path and not state.treatment_seq_path:
            return {
                "status": "waiting_input",
                "message": "Please upload snRNA-seq data file",
                "findings": [],
                "success": False
            }
        
        result = await self.run_autonomous_analysis(
            state, 
            instruction,
            max_steps=kwargs.get("max_steps", 30)
        )
        
        return {
            "status": result["status"],
            "findings": result["findings"],
            "figures": result["figures"],
            "code_execution": self.code_history[-1] if self.code_history else None,
            "success": result["success"],
            "steps_executed": result["steps_executed"]
        }
    
    def get_all_code(self) -> str:
        """Get all executed code"""
        return "\n\n".join([
            f"# Cell {i+1}\n{ce.code}" 
            for i, ce in enumerate(self.code_history)
        ])
    
    def get_execution_summary(self) -> str:
        """Get summary of executions"""
        if not self.code_history:
            return "No code executed yet."
        
        errors = sum(1 for ce in self.code_history if ce.error)
        figures = sum(len(ce.figures) for ce in self.code_history)
        
        return f"Cells: {len(self.code_history)}, Success: {len(self.code_history) - errors}, Errors: {errors}, Figures: {figures}"
    
    async def save_state(self):
        """Save current state"""
        if self.sandbox:
            await self.sandbox.save_session()

    async def restore_state(self) -> dict:
        """Restore state

        Returns:
            dict with sandbox start status including session_restored and restore_failed flags
        """
        return await self.sandbox.start()

    # ============================================
    # Publication Figure Generation
    # ============================================

    async def _handle_generate_publication_figure(
        self,
        code: str,
        figure_name: str,
        description: str,
        figure_type: str,
        state: AnalysisState,
        review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None
    ) -> str:
        """
        Handle generate_publication_figure tool call.
        Generates the figure, saves in dual format, and optionally gets PI review.

        Args:
            code: Python plotting code
            figure_name: Name for the figure file
            description: Figure description
            figure_type: Type of figure
            state: Analysis state to store the figure
            review_callback: Optional callback for PI review

        Returns:
            Tool result message
        """
        logger.info(f"[Bioinformatics] Generating publication figure: {figure_name}")

        # Validate code before execution
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return f"Error: Code validation failed - {error_msg}"

        # Ensure figures directory exists
        figures_dir = Path(self.sandbox.working_dir) / "figures"

        # Clean up figure name
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in figure_name)
        png_path = figures_dir / f"{safe_name}.png"
        pdf_path = figures_dir / f"{safe_name}.pdf"

        # Wrap code to save both PNG and PDF (use repr() for paths to prevent injection)
        figures_dir_repr = repr(str(figures_dir))
        png_path_repr = repr(str(png_path))
        pdf_path_repr = repr(str(pdf_path))
        wrapped_code = f'''
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality parameters
plt.rcParams.update({{
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,  # Ensure fonts are vectorized in PDF
    'ps.fonttype': 42,
}})

# Create figures directory
_figures_dir = {figures_dir_repr}
Path(_figures_dir).mkdir(exist_ok=True)

# Execute the plotting code
{code}

# Get current figure
fig = plt.gcf()

_png_path = {png_path_repr}
_pdf_path = {pdf_path_repr}

# Save PNG thumbnail (lower DPI for preview)
fig.savefig(_png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved PNG: {{_png_path}}")

# Save high-res PDF
fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Saved PDF: {{_pdf_path}}")

plt.close()
'''

        # Execute the code
        result = await self.sandbox.execute(wrapped_code)

        if result.error:
            error_msg = f"Figure generation failed: {result.error}"
            logger.error(f"[Bioinformatics] {error_msg}")
            await self._notify("error", error_msg)
            return error_msg

        # Check if files were created
        if not png_path.exists():
            error_msg = f"PNG file was not created: {png_path}"
            logger.error(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Read PNG and compress to base64
        png_b64 = self._compress_image_to_base64(png_path, max_width=800)

        # Create PublicationFigure object
        figure = PublicationFigure(
            id=str(uuid.uuid4()),
            name=safe_name,
            description=description,
            figure_type=figure_type,
            png_path=str(png_path),
            pdf_path=str(pdf_path),
            png_base64=png_b64,
            review_status=FigureReviewStatus.PENDING
        )

        # Send figure to frontend for display
        await self._notify("figure", png_b64)

        # If review callback provided, get PI review
        # Allow up to 3 revisions (4 reviews total: initial + 3 revisions)
        if review_callback:
            revision_count = 0
            current_code = code

            while revision_count <= 3:
                logger.info(f"[Bioinformatics] Submitting figure for PI review (attempt {revision_count + 1})")
                await self._notify("status", f"PI reviewing figure: {figure_name}...")

                review = await review_callback(figure)

                if review.get('approved', False):
                    figure.review_status = FigureReviewStatus.APPROVED
                    state.publication_figures.append(figure)
                    logger.info(f"[Bioinformatics] Figure {figure_name} approved by PI")
                    await self._notify("status", f"Figure {figure_name} approved by PI")
                    return f"Figure {figure_name} has been approved by PI and saved.\nQuality score: {review.get('quality_score', 'N/A')}/10\nFeedback: {review.get('feedback', 'Approved')}"

                # PI requested revision
                feedback = review.get('feedback', '')
                suggestions = review.get('suggestions', [])
                must_fix = review.get('must_fix', [])

                figure.pi_feedback = feedback
                revision_count += 1
                figure.revision_count = revision_count

                if revision_count <= 3:
                    # Try to revise the figure
                    await self._notify("status", f"Revising figure based on PI feedback...")
                    logger.info(f"[Bioinformatics] Revising figure {figure_name}: {feedback}")

                    # Generate revised code
                    revised_code = await self._revise_figure_code(
                        current_code, feedback, suggestions, must_fix, figure_type
                    )

                    # Validate revised code before execution
                    is_valid, error_msg = self._validate_code(revised_code)
                    if not is_valid:
                        logger.warning(f"[Bioinformatics] Revised code failed validation: {error_msg}")
                        break

                    # Re-execute with revised code (rebuild wrapper instead of str.replace to avoid corruption)
                    revised_wrapped = f'''
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({{
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}})

_figures_dir = {figures_dir_repr}
Path(_figures_dir).mkdir(exist_ok=True)

{revised_code}

fig = plt.gcf()

_png_path = {png_path_repr}
_pdf_path = {pdf_path_repr}

fig.savefig(_png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.close()
'''
                    result = await self.sandbox.execute(revised_wrapped)

                    if not result.error and png_path.exists():
                        # Update figure with new image
                        figure.png_base64 = self._compress_image_to_base64(png_path, max_width=800)
                        await self._notify("figure", figure.png_base64)
                        current_code = revised_code
                    else:
                        logger.warning(f"[Bioinformatics] Revision failed: {result.error}")
                        break  # Don't review the same unchanged figure again

            # After 3 attempts, save with revision_requested status
            figure.review_status = FigureReviewStatus.REVISION_REQUESTED
            state.publication_figures.append(figure)
            logger.warning(f"[Bioinformatics] Figure {figure_name} not fully approved after {revision_count} revisions")
            return f"Figure {figure_name} saved after {revision_count} revision attempts. Status: needs further revision.\nLast feedback: {figure.pi_feedback}"

        # No review callback, just save the figure
        state.publication_figures.append(figure)
        return f"Figure {figure_name} has been generated and saved.\nPNG: {png_path}\nPDF: {pdf_path}"

    def _compress_image_to_base64(self, image_path: Path, max_width: int = 800) -> str:
        """
        Compress an image and convert to base64 for efficient transmission.

        Args:
            image_path: Path to the image file
            max_width: Maximum width for the compressed image

        Returns:
            Base64 encoded JPEG string
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                # Calculate resize ratio
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    # Handle PIL version compatibility
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS  # For older PIL versions
                    img = img.resize(new_size, resample_method)

                # Convert to RGB if necessary (handle RGBA, P modes)
                if img.mode in ('RGBA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        # Safely get alpha channel - check split() returns enough channels
                        channels = img.split()
                        if len(channels) >= 4:
                            background.paste(img, mask=channels[3])
                        else:
                            background.paste(img)
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            logger.error(f"[Bioinformatics] Failed to compress image: {e}")
            # Fallback: read raw file and encode
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    async def _revise_figure_code(
        self,
        original_code: str,
        feedback: str,
        suggestions: List[str],
        must_fix: List[str],
        figure_type: str
    ) -> str:
        """
        Generate revised plotting code based on PI feedback.

        Args:
            original_code: The original plotting code
            feedback: Overall feedback from PI
            suggestions: List of improvement suggestions
            must_fix: List of critical issues that must be fixed
            figure_type: Type of figure

        Returns:
            Revised Python plotting code
        """
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        must_fix_text = "\n".join(f"- {m}" for m in must_fix) if must_fix else "None"

        prompt = f"""Please revise the following plotting code based on PI feedback.

**Original Code:**
```python
{original_code}
```

**Figure Type:** {figure_type}

**PI Feedback:** {feedback}

**Improvement Suggestions:**
{suggestions_text}

**Critical Issues (MUST FIX):**
{must_fix_text}

**Requirements:**
1. Fix all critical issues listed above
2. Address as many suggestions as possible
3. Maintain the same overall figure structure
4. Keep using the same data variables (assume they exist)
5. Return ONLY the revised Python code, no explanations

**Common fixes for publication figures:**
- Add/improve axis labels and titles
- Adjust font sizes (plt.rcParams['font.size'] = 12)
- Add legends with proper placement
- Use colorblind-friendly palettes
- Add scale bars or annotations
- Improve color contrast
- Remove unnecessary elements

Return only the Python code:"""

        self._set_llm_context()
        # Use slightly lower temperature for code revision (needs precision)
        revision_temp = min(self.config.temperature, 0.2)
        response = await self.llm_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=revision_temp,
            max_tokens=2048
        )

        response_text = response.content if response and response.content else None
        if not response_text:
            logger.warning("[Bioinformatics] Empty response from LLM for code revision")
            return original_code  # Return unchanged code

        # Extract code from response with safe bounds checking
        code = response_text
        try:
            if "```python" in response_text:
                parts = response_text.split("```python")
                if len(parts) > 1:
                    code_parts = parts[1].split("```")
                    code = code_parts[0] if code_parts else parts[1]
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) > 1:
                    code = parts[1]
        except (IndexError, AttributeError) as e:
            logger.warning(f"Failed to extract code from response: {e}, using raw response")
            code = response_text

        return code.strip()
