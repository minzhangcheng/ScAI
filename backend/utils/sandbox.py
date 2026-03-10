"""
Local Jupyter Kernel Executor with State Persistence

Uses dill.dump_session/load_session to save and restore the complete Python environment.
Supports pause/resume by saving and restoring the full session state.
"""

import os
import re
import json
import base64
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from jupyter_client import KernelManager

logger = logging.getLogger(__name__)


# Dangerous code patterns that should not be executed in the sandbox.
# Enforced at the sandbox level so ALL callers (BioinformaticsAgent,
# report_generator, chat sandbox, etc.) are protected.
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

    # Environment manipulation
    (r'\bos\.chdir\b', 'os.chdir (changing working directory)'),
    (r'\bos\.chroot\b', 'os.chroot'),
    (r'\bsys\.path\s*\.(?:append|insert|extend)\b', 'sys.path modification'),
]


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Validate code for security before execution.

    Args:
        code: The Python code to validate

    Returns:
        Tuple of (is_safe, error_message). is_safe is True if the code
        passes all security checks.
    """
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            # Allow subprocess when ALL usages are pip install commands
            if 'subprocess' in description:
                subprocess_lines = [
                    line.strip() for line in code.splitlines()
                    if re.search(r'\bsubprocess\b', line)
                ]
                pip_pattern = r'subprocess\.\w+\(\s*\[?\s*["\'](?:pip|pip3)["\'].*install'
                import_pattern = r'^(?:import|from)\s+subprocess'
                non_import_lines = [
                    line for line in subprocess_lines
                    if not re.match(import_pattern, line)
                ]
                if not non_import_lines or all(re.search(pip_pattern, line) for line in non_import_lines):
                    continue
            return False, f"Dangerous pattern detected: {description}"
    return True, ""


def extract_code_from_markdown(text: str, language: str = "python") -> str:
    """
    Extract code from a markdown-formatted LLM response.

    Uses a state-machine approach to correctly handle:
    - Fenced code blocks with language tag (```python ... ```)
    - Fenced code blocks without language tag (``` ... ```)
    - Multiple code blocks (returns the first matching one)
    - Nested backticks inside code blocks

    Args:
        text: The LLM response text that may contain markdown code blocks
        language: The language tag to look for (default "python")

    Returns:
        The extracted code string, or the original text stripped if no
        code block is found.
    """
    if not text:
        return ""

    fence_with_lang = f"```{language}"
    fence_plain = "```"

    # Try language-tagged fence first, then plain fence
    for fence_start in (fence_with_lang, fence_plain):
        pos = text.find(fence_start)
        if pos == -1:
            continue

        # Move past the opening fence line
        code_start = text.find("\n", pos)
        if code_start == -1:
            continue
        code_start += 1  # skip the newline

        # Find closing fence: scan for ``` at the start of a line
        # (handles nested backticks by requiring ``` on its own line)
        search_pos = code_start
        while search_pos < len(text):
            line_end = text.find("\n", search_pos)
            if line_end == -1:
                line_end = len(text)
            line = text[search_pos:line_end].strip()
            if line == fence_plain:
                return text[code_start:search_pos].strip()
            search_pos = line_end + 1

        # No closing fence found — take everything after the opening fence
        return text[code_start:].strip()

    # No code block found — return text as-is
    return text.strip()


class CodeExecutionResult:
    """Result of code execution"""
    def __init__(
        self,
        success: bool,
        output: str = "",
        error: Optional[str] = None,
        figures: Optional[List[str]] = None,  # Base64 encoded images
        execution_count: int = 0
    ):
        self.success = success
        self.output = output
        self.error = error
        self.figures = figures or []
        self.execution_count = execution_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "figures": self.figures,
            "execution_count": self.execution_count
        }


class NotebookState:
    """Represents the state of a Jupyter notebook session"""
    def __init__(self, session_id: str, state_dir: str = "states"):
        self.session_id = session_id
        self.state_dir = Path(state_dir).absolute()  # Use absolute path
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.cells: List[Dict[str, Any]] = []  # List of {code, outputs, execution_count}
        self.execution_count: int = 0
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()
    
    @property
    def notebook_path(self) -> Path:
        return self.state_dir / f"{self.session_id}.ipynb"
    
    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.session_id}_state.json"
    
    @property
    def session_path(self) -> Path:
        """Path for dill session dump"""
        return self.state_dir / f"{self.session_id}_session.pkl"
    
    def add_cell(self, code: str, outputs: List[Dict], execution_count: int):
        """Add an executed cell to the history"""
        self.cells.append({
            "code": code,
            "outputs": outputs,
            "execution_count": execution_count,
            "executed_at": datetime.now().isoformat()
        })
        self.execution_count = execution_count
        self.updated_at = datetime.now().isoformat()
    
    def save_notebook(self):
        """Save as Jupyter notebook format"""
        nb = new_notebook()
        for cell in self.cells:
            code_cell = new_code_cell(source=cell["code"])
            code_cell.execution_count = cell["execution_count"]
            
            # Convert outputs to notebook format
            cell_outputs = []
            for out in cell["outputs"]:
                if out["type"] == "stream":
                    cell_outputs.append(new_output(
                        output_type="stream",
                        name=out.get("name", "stdout"),
                        text=out["text"]
                    ))
                elif out["type"] == "execute_result":
                    cell_outputs.append(new_output(
                        output_type="execute_result",
                        data=out.get("data", {}),
                        execution_count=cell["execution_count"]
                    ))
                elif out["type"] == "display_data":
                    cell_outputs.append(new_output(
                        output_type="display_data",
                        data=out.get("data", {})
                    ))
                elif out["type"] == "error":
                    cell_outputs.append(new_output(
                        output_type="error",
                        ename=out.get("ename", "Error"),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", [])
                    ))
            
            code_cell.outputs = cell_outputs
            nb.cells.append(code_cell)
        
        # Write notebook file
        with open(self.notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        logger.info(f"Saved notebook: {self.notebook_path}")
    
    def save_metadata(self):
        """Save state metadata"""
        state_data = {
            "session_id": self.session_id,
            "execution_count": self.execution_count,
            "cell_count": len(self.cells),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_session": self.session_path.exists()
        }
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)
    
    def save(self):
        """Save notebook and metadata"""
        self.save_notebook()
        self.save_metadata()
    
    def load(self) -> bool:
        """Load notebook state from disk"""
        if not self.notebook_path.exists():
            return False
        
        try:
            # Load notebook
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            self.cells = []
            for cell in nb.cells:
                if cell.cell_type == "code":
                    outputs = []
                    for out in cell.outputs:
                        if out.output_type == "stream":
                            outputs.append({
                                "type": "stream",
                                "name": out.get("name", "stdout"),
                                "text": out.text
                            })
                        elif out.output_type == "execute_result":
                            outputs.append({
                                "type": "execute_result",
                                "data": dict(out.data)
                            })
                        elif out.output_type == "display_data":
                            outputs.append({
                                "type": "display_data",
                                "data": dict(out.data)
                            })
                        elif out.output_type == "error":
                            outputs.append({
                                "type": "error",
                                "ename": out.ename,
                                "evalue": out.evalue,
                                "traceback": out.traceback
                            })
                    
                    self.cells.append({
                        "code": cell.source,
                        "outputs": outputs,
                        "execution_count": cell.execution_count or 0,
                        "executed_at": None
                    })
            
            # Load state metadata
            if self.state_path.exists():
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                self.execution_count = state_data.get("execution_count", 0)
                self.created_at = state_data.get("created_at", self.created_at)
                self.updated_at = state_data.get("updated_at", self.updated_at)
            
            logger.info(f"Loaded notebook state: {len(self.cells)} cells")
            return True
        except Exception as e:
            logger.error(f"Failed to load notebook state: {e}")
            return False
    
    def has_session_dump(self) -> bool:
        """Check if a session dump exists"""
        return self.session_path.exists()


class LocalJupyterSandbox:
    """
    Local Jupyter kernel executor with full session persistence using dill.
    
    Features:
    - Executes code in a local Jupyter kernel
    - Saves complete session state using dill.dump_session()
    - Restores session state using dill.load_session()
    - Saves notebook history as .ipynb file
    """
    
    def __init__(
        self,
        session_id: str,
        state_dir: str = "states",
        working_dir: str = "workspace",
        timeout: int = 1200
    ):
        self.session_id = session_id
        self.state_dir = state_dir
        self.working_dir = Path(working_dir) / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        self.km: Optional[KernelManager] = None
        self.kc = None
        self.notebook_state = NotebookState(session_id, state_dir)
        self._initialized = False
        self._session_restored = False
        self._session_restore_failed = False  # Track if session restore failed
    
    async def start(self) -> dict:
        """Start the Jupyter kernel

        Returns:
            dict with keys:
                - started: bool - whether kernel was started (False if already running)
                - session_restored: bool - whether previous session was restored
                - restore_failed: bool - whether restore was attempted but failed
        """
        if self.km is not None:
            return {
                "started": False,
                "session_restored": self._session_restored,
                "restore_failed": self._session_restore_failed
            }

        logger.info(f"Starting Jupyter kernel for session {self.session_id}")

        # Create kernel manager
        self.km = KernelManager(kernel_name='python3')
        self.km.start_kernel(cwd=str(self.working_dir))

        # Get client
        self.kc = self.km.client()
        self.kc.start_channels()

        # Wait for kernel to be ready
        try:
            await asyncio.wait_for(
                self._wait_for_ready(),
                timeout=30
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Kernel failed to start within timeout")

        # Initialize with common imports and settings
        await self._initialize_kernel()

        # Try to restore previous session
        await self._restore_session()

        return {
            "started": True,
            "session_restored": self._session_restored,
            "restore_failed": self._session_restore_failed
        }
    
    async def _wait_for_ready(self):
        """Wait for kernel to be ready"""
        while True:
            try:
                self.kc.kernel_info()
                msg = await asyncio.wait_for(
                    asyncio.to_thread(self.kc.get_shell_msg),
                    timeout=5
                )
                if msg['msg_type'] == 'kernel_info_reply':
                    break
            except Exception:
                # Kernel not ready yet, wait and retry
                await asyncio.sleep(0.5)
    
    async def _initialize_kernel(self):
        """Initialize kernel with common setup"""
        if self._initialized:
            return
        
        init_code = """
# === MANDATORY: Set ALL random seeds for reproducibility ===
import random
import os

SEED = 42

# Python built-in random
random.seed(SEED)

# Environment variable for hash seed
os.environ['PYTHONHASHSEED'] = str(SEED)

# NumPy
import numpy as np
np.random.seed(SEED)

# Try to set PyTorch seeds (if available)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
except ImportError:
    pass

# Try to set TensorFlow seeds (if available)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
except ImportError:
    pass

# Initialize environment
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import dill for session persistence
import dill

# Set matplotlib defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['pdf.fonttype'] = 42   # Vectorized fonts for editable PDFs
plt.rcParams['ps.fonttype'] = 42

# === Monkey-patch Figure.savefig to auto-generate PDF alongside PNG ===
import matplotlib.figure as _mpl_figure
_original_savefig = _mpl_figure.Figure.savefig

def _savefig_with_pdf(self, fname, *args, **kwargs):
    # Wrapper: after saving PNG, auto-generate a companion high-res PDF.
    _original_savefig(self, fname, *args, **kwargs)
    try:
        _fname_str = str(fname)
        if _fname_str.lower().endswith('.png'):
            _pdf_path = _fname_str[:-4] + '.pdf'
            _pdf_kwargs = dict(kwargs)
            _pdf_kwargs['format'] = 'pdf'
            _pdf_kwargs['dpi'] = 300
            if 'bbox_inches' not in _pdf_kwargs:
                _pdf_kwargs['bbox_inches'] = 'tight'
            _original_savefig(self, _pdf_path, *args, **_pdf_kwargs)
    except Exception as _pdf_err:
        print(f"[Auto-PDF] Warning: {_pdf_err}")

_mpl_figure.Figure.savefig = _savefig_with_pdf

# Try importing common bioinformatics packages
try:
    import scanpy as sc
    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=100, frameon=False)
    # Set scanpy random state
    sc.settings.seed = SEED
except ImportError:
    pass

try:
    import anndata as ad
except ImportError:
    pass

try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    pass

# Helper function to ensure reproducibility in function calls
def get_seed():
    return 42

# Helper function to check and fix NaN values in AnnData
def check_and_fix_nan(adata, fix=True):
    '''
    Check for NaN values in adata.X and optionally fix them.
    MUST be called before PCA or any dimensionality reduction!
    
    Args:
        adata: AnnData object
        fix: If True, replace NaN with 0
    
    Returns:
        Number of NaN values found
    '''
    import scipy.sparse as sp
    
    if sp.issparse(adata.X):
        nan_count = np.isnan(adata.X.data).sum()
        if fix and nan_count > 0:
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Fixed {nan_count} NaN values in sparse matrix")
    else:
        nan_count = np.isnan(adata.X).sum()
        if fix and nan_count > 0:
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Fixed {nan_count} NaN values in dense matrix")
    
    if nan_count == 0:
        print("No NaN values found - data is clean")
    
    return nan_count

print(f"Environment initialized with SEED={SEED} for reproducibility")
print("Helper functions available: get_seed(), check_and_fix_nan(adata)")
"""
        result = await self.execute(init_code, save_to_history=False, skip_safety_check=True)
        self._initialized = True
        logger.info("Kernel initialized")
    
    async def _restore_session(self):
        """Restore session from dill dump if available"""
        # First load notebook metadata
        self.notebook_state.load()

        if not self.notebook_state.has_session_dump():
            logger.info("No session dump to restore")
            return

        # Use absolute path with proper escaping for Python string
        session_path = str(self.notebook_state.session_path.absolute())
        escaped_path = repr(session_path)  # Properly escape special characters
        logger.info(f"Restoring session from {session_path}")

        # Load saved variables
        restore_code = f"""
import dill

try:
    with open({escaped_path}, 'rb') as f:
        _session_data = dill.load(f)

    # Restore variables to global namespace
    for _var_name, _obj in _session_data.items():
        globals()[_var_name] = _obj

    print(f"Session restored: {{len(_session_data)}} variables")
    print(f"Restored: {{list(_session_data.keys())}}")
except Exception as e:
    print(f"Note: Could not restore session ({{e}}), starting fresh")
"""
        result = await self.execute(restore_code, save_to_history=False, skip_safety_check=True)

        if result.success and "Session restored:" in (result.output or ""):
            self._session_restored = True
            self._session_restore_failed = False
            logger.info("Session restored successfully")
        else:
            self._session_restore_failed = True
            if result.success:
                logger.warning(f"Session restore code ran but restore failed. Output: {result.output}")
            else:
                logger.warning(f"Session restore failed: {result.error}. Starting with fresh session.")
    
    async def save_session(self):
        """Save the current session state using dill - only save data variables"""
        # Use absolute path to avoid working directory issues
        session_path = str(self.notebook_state.session_path.absolute())
        escaped_path = repr(session_path)  # Properly escape special characters
        logger.info(f"Saving session to {session_path}")

        # Only save important data variables, avoid non-serializable objects like zmq
        save_code = f"""
import dill
import os

# Ensure directory exists
os.makedirs(os.path.dirname({escaped_path}), exist_ok=True)

# List of variable names to save (data objects that can be serialized)
_vars_to_save = ['adata', 'adata_control', 'adata_treatment', 'adata_merged',
                 'ldata', 'ldata_control', 'ldata_treatment',
                 'results', 'markers', 'deg_results', 'pathway_results',
                 'cell_types', 'cluster_markers', 'differential_genes']

# Collect variables that exist and can be pickled
_session_data = {{}}
for _var_name in _vars_to_save:
    if _var_name in dir():
        try:
            _obj = eval(_var_name)
            # Test if it can be pickled
            dill.dumps(_obj)
            _session_data[_var_name] = _obj
        except Exception:
            pass  # Skip variables that can't be pickled

# Also save any AnnData objects we might have missed
import anndata as ad
for _var_name, _obj in list(globals().items()):
    if isinstance(_obj, ad.AnnData) and _var_name not in _session_data:
        try:
            dill.dumps(_obj)
            _session_data[_var_name] = _obj
        except Exception:
            pass

# Save the collected variables
with open({escaped_path}, 'wb') as f:
    dill.dump(_session_data, f)

print(f"Session saved: {{len(_session_data)}} variables")
"""
        result = await self.execute(save_code, save_to_history=False, skip_safety_check=True)

        if result.success:
            logger.info("Session saved successfully")
        else:
            logger.warning(f"Failed to save session: {result.error}")
        
        return result.success
    
    async def execute(
        self,
        code: str,
        save_to_history: bool = True,
        skip_safety_check: bool = False
    ) -> CodeExecutionResult:
        """Execute code in the kernel.

        Args:
            code: Python code to execute
            save_to_history: Whether to save successful execution to notebook history
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
        """
        # Validate code safety before execution (unless explicitly skipped for internal code)
        if not skip_safety_check:
            is_safe, error_msg = validate_code_safety(code)
            if not is_safe:
                logger.warning(f"Code blocked by safety check for session {self.session_id}: {error_msg}")
                return CodeExecutionResult(
                    success=False,
                    error=f"Code execution blocked: {error_msg}"
                )

        if self.kc is None:
            await self.start()
        
        outputs = []
        figures = []
        output_text = []
        error_msg = None
        execution_count = 0
        
        try:
            # Send execute request
            msg_id = self.kc.execute(code)

            # Collect outputs — only process messages belonging to THIS execution (matched by msg_id)
            while True:
                try:
                    msg = await asyncio.wait_for(
                        asyncio.to_thread(self.kc.get_iopub_msg),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    error_msg = f"Execution timed out after {self.timeout} seconds"
                    # Interrupt the kernel to stop the running code
                    if self.km:
                        try:
                            self.km.interrupt_kernel()
                            logger.warning(f"Kernel interrupted after timeout for session {self.session_id}")
                            # Wait briefly for kernel to acknowledge interrupt
                            await asyncio.sleep(0.5)
                            # Drain any remaining messages to clear the queue
                            try:
                                while True:
                                    drain_msg = await asyncio.wait_for(
                                        asyncio.to_thread(self.kc.get_iopub_msg),
                                        timeout=1.0
                                    )
                                    if drain_msg['msg_type'] == 'status' and drain_msg['content']['execution_state'] == 'idle':
                                        break
                            except asyncio.TimeoutError:
                                # Kernel may be in bad state, restart it
                                logger.warning(f"Kernel did not return to idle after interrupt for session {self.session_id}")
                                try:
                                    await asyncio.to_thread(self.km.restart_kernel)

                                    # Rebuild client connection
                                    if self.kc is not None:
                                        try:
                                            self.kc.stop_channels()
                                        except Exception:
                                            pass
                                    self.kc = self.km.client()
                                    self.kc.start_channels()

                                    # Wait for kernel to be ready
                                    await asyncio.wait_for(self._wait_for_ready(), timeout=30)

                                    # Re-initialize and restore session
                                    self._initialized = False
                                    await self._initialize_kernel()
                                    await self._restore_session()

                                    logger.info(f"Kernel restarted and session restored for session {self.session_id}")
                                except Exception as restart_error:
                                    logger.error(f"Failed to restart kernel: {restart_error}")
                        except Exception as interrupt_error:
                            logger.error(f"Failed to interrupt kernel: {interrupt_error}")
                    break

                # Skip messages from other execution requests (e.g. leftover from session restore)
                parent_msg_id = msg.get('parent_header', {}).get('msg_id', '')
                if parent_msg_id and parent_msg_id != msg_id:
                    continue

                msg_type = msg['msg_type']
                content = msg['content']

                if msg_type == 'status':
                    if content['execution_state'] == 'idle':
                        break

                elif msg_type == 'execute_input':
                    execution_count = content.get('execution_count', 0)

                elif msg_type == 'stream':
                    text = content.get('text', '')
                    output_text.append(text)
                    outputs.append({
                        "type": "stream",
                        "name": content.get('name', 'stdout'),
                        "text": text
                    })

                elif msg_type == 'execute_result':
                    data = content.get('data', {})
                    if 'text/plain' in data:
                        output_text.append(data['text/plain'])
                    outputs.append({
                        "type": "execute_result",
                        "data": data
                    })

                elif msg_type == 'display_data':
                    data = content.get('data', {})
                    # Extract images
                    if 'image/png' in data:
                        figures.append(data['image/png'])
                    outputs.append({
                        "type": "display_data",
                        "data": data
                    })

                elif msg_type == 'error':
                    error_msg = '\n'.join(content.get('traceback', []))
                    outputs.append({
                        "type": "error",
                        "ename": content.get('ename', 'Error'),
                        "evalue": content.get('evalue', ''),
                        "traceback": content.get('traceback', [])
                    })
            
            # Save to history
            if save_to_history and not error_msg:
                self.notebook_state.add_cell(code, outputs, execution_count)
                self.notebook_state.save()
            
            return CodeExecutionResult(
                success=error_msg is None,
                output='\n'.join(output_text),
                error=error_msg,
                figures=figures,
                execution_count=execution_count
            )
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return CodeExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def execute_and_get_figures(self, code: str) -> Tuple[str, List[str]]:
        """Execute code and return output and figures"""
        result = await self.execute(code)
        return result.output, result.figures
    
    def upload_file(self, local_path: str, filename: Optional[str] = None) -> str:
        """Copy a file to the sandbox working directory"""
        import shutil
        src = Path(local_path)
        dst = self.working_dir / (filename or src.name)
        shutil.copy(src, dst)
        logger.info(f"Uploaded file to sandbox: {dst}")
        return str(dst)
    
    def save_state(self):
        """Save current state synchronously - only saves notebook, not session.

        Note: This method only reliably saves the notebook state. Session state
        (Python variables) may not be saved if called from sync context.
        Use save_state_async() for complete state persistence.
        """
        self.notebook_state.save()
        # Try to run in existing event loop, or create one if needed
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - cannot safely run async code
            # Log warning and skip session save (notebook is already saved)
            logger.warning(
                "save_state() called from async context - session variables may not be saved. "
                "Use save_state_async() for guaranteed persistence."
            )
        except RuntimeError:
            # No running loop - safe to create a new one
            try:
                asyncio.run(self.save_session())
            except Exception as e:
                logger.error(f"Failed to save session state: {e}")
    
    async def save_state_async(self):
        """Save current state asynchronously"""
        self.notebook_state.save()
        await self.save_session()
    
    async def stop(self):
        """Stop the kernel and save state. Always shuts down kernel even if save fails."""
        # Try to save session before stopping, but don't let failure prevent kernel cleanup
        try:
            await self.save_session()
            self.notebook_state.save()
        except Exception as e:
            logger.error(f"Failed to save session during stop for {self.session_id}: {e}")
        finally:
            if self.kc is not None:
                try:
                    self.kc.stop_channels()
                except Exception as e:
                    logger.error(f"Error stopping kernel channels for {self.session_id}: {e}")
                self.kc = None

            if self.km is not None:
                try:
                    self.km.shutdown_kernel(now=True)
                except Exception as e:
                    logger.error(f"Error shutting down kernel for {self.session_id}: {e}")
                self.km = None
                self._initialized = False

            logger.info(f"Kernel stopped for session {self.session_id}")
    
    async def restart(self):
        """Restart kernel and restore state"""
        await self.stop()
        await self.start()
    
    def get_notebook_path(self) -> str:
        """Get path to the saved notebook"""
        return str(self.notebook_state.notebook_path)
    
    def get_session_path(self) -> str:
        """Get path to the session dump file"""
        return str(self.notebook_state.session_path)
    
    def get_all_state_files(self) -> List[str]:
        """Get all state-related files for export"""
        files = []
        if self.notebook_state.notebook_path.exists():
            files.append(str(self.notebook_state.notebook_path))
        if self.notebook_state.state_path.exists():
            files.append(str(self.notebook_state.state_path))
        if self.notebook_state.session_path.exists():
            files.append(str(self.notebook_state.session_path))
        return files
    
    def get_code_history(self) -> List[Dict[str, Any]]:
        """Get all executed code cells"""
        return self.notebook_state.cells

    def was_session_restored(self) -> bool:
        """Check if session was successfully restored from previous state"""
        return self._session_restored

    def did_session_restore_fail(self) -> bool:
        """Check if session restore was attempted but failed"""
        return self._session_restore_failed


# Convenience function to create sandbox
async def create_sandbox(
    session_id: str,
    state_dir: str = "states",
    working_dir: str = "workspace"
) -> LocalJupyterSandbox:
    """Create and start a local Jupyter sandbox"""
    sandbox = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=state_dir,
        working_dir=working_dir
    )
    await sandbox.start()
    return sandbox
