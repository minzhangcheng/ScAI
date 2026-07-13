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
import time
import hmac
import hashlib
import logging
from queue import Empty as QueueEmpty
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from jupyter_client import KernelManager

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256
from backend.config._config import DATA_BASE_DIR

# Cap on a single iopub poll so the wall-clock deadline is re-checked at least
# this often even while the kernel keeps streaming output (tqdm/scanpy verbose).
IOPUB_POLL_CAP = 30.0


# ---------------------------------------------------------------------------
# Session pkl HMAC integrity helpers (host-side only)
# ---------------------------------------------------------------------------

def _sig_path(pkl: Path, integrity_dir: Path) -> Path:
    return integrity_dir / (pkl.name + ".sig")


def _pkl_hmac(data: bytes) -> str:
    return sign_hmac_sha256(data)


def _pkl_hmac_stream(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hmac.new(get_integrity_secret().encode(), b"", hashlib.sha256)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_pkl_signature(pkl: Path, integrity_dir: Path) -> None:
    integrity_dir.mkdir(parents=True, exist_ok=True)
    _sig_path(pkl, integrity_dir).write_text(_pkl_hmac_stream(pkl))


def _verify_pkl_integrity(pkl: Path, integrity_dir: Path) -> bool:
    sig = _sig_path(pkl, integrity_dir)
    if not pkl.exists() or not sig.exists():
        return False
    return hmac.compare_digest(
        sig.read_text().strip(), _pkl_hmac_stream(pkl)
    )


def _integrity_dir(sid: str) -> Path:
    return DATA_BASE_DIR / "integrity" / sid


def _path_is_within(path: str, root: str) -> bool:
    resolved = str(Path(path).resolve())
    root_resolved = str(Path(root).resolve()).rstrip(os.sep)
    return resolved == root_resolved or resolved.startswith(root_resolved + os.sep)


def compute_remaining_timeout(deadline: float, now: float) -> float:
    """Seconds left until the wall-clock deadline; <= 0 means expired."""
    return deadline - now


logger = logging.getLogger(__name__)


def _is_kernel_protocol_error(exc: Exception) -> bool:
    """True for ZMQ/jupyter session-key corruption that makes the kernel
    permanently unusable until restarted (vs. a normal user code error)."""
    if not isinstance(exc, (ValueError, KeyError)):
        return False
    s = str(exc).lower()
    return (
        "invalid signature" in s
        or "<ids|msg>" in s
        or "is not in list" in s
    )


# Dangerous code patterns that should not be executed in the sandbox.
# Enforced at the sandbox level so ALL callers (BioinformaticsAgent,
# report_generator, chat sandbox, etc.) are protected.
# NOTE: regex is defense-in-depth only; runtime _sandboxed_open + write
# allowlist are the real boundary.
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
    (r'''['"](eval|exec|__import__)['"]''', 'string-referenced dangerous builtin'),

    # Dangerous attribute access
    (r'\b__builtins__\b', '__builtins__ access'),
    (r'\b__class__\s*\.\s*__bases__\b', '__class__.__bases__ access'),
    (r'\b__class__\s*\.\s*__mro__\b', '__class__.__mro__ access'),
    (r'\b__subclasses__\s*\(', '__subclasses__ access'),
    # globals() and getattr-with-dunder are commonly used by scanpy/anndata
    # and pose no additional risk in an already-isolated Jupyter kernel.
    # locals() is read-only (same as globals()) and commonly used in data-science code.
    (r'\bsetattr\s*\([^,]+,\s*["\']__', 'setattr with dunder'),
    (r'\bdelattr\s*\([^,]+,\s*["\']__', 'delattr with dunder'),
    (r'\bvars\s*\(\s*\)', 'vars() function'),

    # Reflection-based sandbox bypasses
    # getattr on dangerous modules (os, sys, subprocess, etc.) can retrieve dangerous funcs dynamically
    (r'\bgetattr\s*\(\s*(?:os|sys|subprocess|shutil|builtins|__builtins__)\b', 'getattr on dangerous module'),
    # __dict__ subscript access can bypass attribute-level checks
    (r'\b__dict__\s*\[', '__dict__ subscript access'),
    # __globals__ gives access to the global namespace of any function
    (r'\b__globals__\b', '__globals__ access'),
    # __class__.__init__ chain is used to reach __globals__ via method objects
    (r'\b__class__\s*\.\s*__init__\b', '__class__.__init__ chain'),
    # type.__subclasses__() enumerates all subclasses to find exploitable ones
    (r'\btype\s*\.\s*__subclasses__', 'type.__subclasses__ access'),

    # File system operations (destructive)
    # Note: open() writes are checked contextually in validate_code_safety()
    # rather than blanket-blocked, because the runtime I/O sandbox (_sandboxed_open)
    # already enforces path safety.  The static check only flags writes that do NOT
    # reference known safe path variables.
    # Note: This static check can be bypassed via variable indirection (e.g. mode='w'; open(f, mode)).
    # The runtime I/O sandbox (_sandboxed_open) is the primary defense; this is a best-effort early check.
    (r'\bopen\s*\([^)]*,\s*["\'][wa]', 'file write operations outside safe directories'),
    (r'\brm\s+-rf\b', 'rm -rf command'),
    (r'\bshutil\.rmtree\b', 'shutil.rmtree'),
    (r'\bos\.remove\b', 'os.remove'),
    (r'\bos\.unlink\b', 'os.unlink'),
    (r'\bos\.rmdir\b', 'os.rmdir'),
    # os.makedirs is allowed — directory creation is non-destructive (idempotent with exist_ok=True)
    # and LLM-generated bioinformatics code commonly uses it for cache/output directories.
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


def _is_safe_os_system(code: str) -> bool:
    """No os.system calls are allowed in notebook execution."""
    return False


def _is_safe_subprocess(code: str) -> bool:
    """Allow importing subprocess, but block subprocess execution calls."""
    lines = [l.strip() for l in code.splitlines() if re.search(r'\bsubprocess\b', l)]
    import_pattern = r'^(?:import|from)\s+subprocess'
    non_import = [l for l in lines if not re.match(import_pattern, l)]
    return not non_import


def _is_safe_importlib(code: str) -> bool:
    """Check if importlib usage is limited to safe patterns (import_module, find_spec, reload)."""
    lines = [l.strip() for l in code.splitlines() if re.search(r'\bimportlib\b', l)]
    safe_import = r'^import\s+importlib\s*$'
    safe_from = r'^from\s+importlib(?:\.util)?\s+import\s+'
    safe_use = r'importlib\.(?:import_module|util\.find_spec|reload)\s*\('
    return all(
        re.match(safe_import, l) or re.match(safe_from, l) or re.search(safe_use, l)
        for l in lines
    )


def _is_safe_file_write(code: str) -> bool:
    """Check if file write calls reference known safe path variables."""
    lines = [
        l.strip() for l in code.splitlines()
        if re.search(r'\bopen\s*\(', l) and re.search(r',\s*["\'][wa]', l)
    ]
    safe_ref = re.compile(r'_?WORKSPACE_DIR|working_dir|figures|tables|/tmp', re.IGNORECASE)
    return all(safe_ref.search(l) for l in lines)


# Maps pattern description keywords to their safety check functions
_EXCEPTION_CHECKS = {
    'os.system':   _is_safe_os_system,
    'subprocess':  _is_safe_subprocess,
    'importlib':   _is_safe_importlib,
    'file write':  _is_safe_file_write,
}


_DANGEROUS_AST_FUNCS = frozenset({
    'system', 'popen', 'exec', 'eval', 'execfile', 'spawn', 'fork',
    'remove', 'unlink', 'rmdir', 'rmtree', 'rename', 'replace',
})

_DANGEROUS_AST_ATTRS = frozenset({
    '__globals__', '__init__', '__subclasses__', '__bases__', '__mro__',
    '__builtins__',
})

_DANGEROUS_AST_MODULES = frozenset({
    'os', 'sys', 'subprocess', 'shutil', 'builtins',
})

_SUBPROCESS_EXEC_FUNCS = frozenset({
    "run",
    "call",
    "check_call",
    "check_output",
    "Popen",
})


def _validate_code_safety_ast(code: str) -> Tuple[bool, str]:
    """AST-based validation layer that catches reflection-based bypasses
    which regex patterns may miss (e.g. string concatenation, variable indirection).

    Returns (True, "") if safe, (False, reason) if dangerous pattern detected.
    Falls back to (True, "") on parse errors so this is purely additive.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # Cannot parse — let regex-only check decide

    module_aliases: dict[str, str] = {}
    blocked_direct_calls: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"subprocess", "os"}:
                    module_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                for alias in node.names:
                    if alias.name in _SUBPROCESS_EXEC_FUNCS:
                        blocked_direct_calls[alias.asname or alias.name] = f"subprocess.{alias.name}"
            elif node.module == "os":
                for alias in node.names:
                    if alias.name == "system":
                        blocked_direct_calls[alias.asname or alias.name] = "os.system"

    def _module_alias_from_expr(expr) -> Optional[str]:
        if isinstance(expr, ast.Name):
            return module_aliases.get(expr.id)
        return None

    def _blocked_call_from_expr(expr) -> Optional[str]:
        if isinstance(expr, ast.Name):
            return blocked_direct_calls.get(expr.id)
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            module_name = module_aliases.get(expr.value.id)
            if module_name == "subprocess" and expr.attr in _SUBPROCESS_EXEC_FUNCS:
                return f"subprocess.{expr.attr}"
            if module_name == "os" and expr.attr == "system":
                return "os.system"
        return None

    def _propagate_alias_to_target(target, value) -> bool:
        changed_alias = False
        if isinstance(target, ast.Name):
            module_name = _module_alias_from_expr(value)
            if module_name and module_aliases.get(target.id) != module_name:
                module_aliases[target.id] = module_name
                changed_alias = True

            blocked_reason = _blocked_call_from_expr(value)
            if blocked_reason and blocked_direct_calls.get(target.id) != blocked_reason:
                blocked_direct_calls[target.id] = blocked_reason
                changed_alias = True
            return changed_alias

        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            for target_item, value_item in zip(target.elts, value.elts):
                changed_alias = _propagate_alias_to_target(target_item, value_item) or changed_alias
        return changed_alias

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    changed = _propagate_alias_to_target(target, node.value) or changed
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                changed = _propagate_alias_to_target(node.target, node.value) or changed

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_direct_calls:
                return False, f"Call to blocked dependency/system function '{blocked_direct_calls[node.func.id]}'"

            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name = module_aliases.get(node.func.value.id)
                if module_name == "subprocess" and node.func.attr in _SUBPROCESS_EXEC_FUNCS:
                    return False, f"Call to blocked subprocess function 'subprocess.{node.func.attr}'"
                if module_name == "os" and node.func.attr == "system":
                    return False, "Call to blocked os.system"

        # Detect getattr(os, 'system') / getattr(os, dangerous_string)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr':
            if len(node.args) >= 2:
                target = node.args[0]
                attr_arg = node.args[1]
                # Check if target is a dangerous module
                if isinstance(target, ast.Name):
                    target_module = module_aliases.get(target.id, target.id)
                    if target_module in _DANGEROUS_AST_MODULES:
                        return False, f"getattr() on dangerous module '{target_module}'"
                # Check if attr argument is a dangerous function name
                if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                    if attr_arg.value in _DANGEROUS_AST_FUNCS:
                        return False, f"getattr() retrieving dangerous function '{attr_arg.value}'"

        # Detect attribute chains containing __globals__, __init__.__globals__, etc.
        if isinstance(node, ast.Attribute):
            if node.attr in ('__globals__', '__builtins__'):
                return False, f"Access to '{node.attr}' attribute"
            # Detect __class__.__init__ chain (used to reach __globals__)
            if node.attr == '__init__' and isinstance(node.value, ast.Attribute) and node.value.attr == '__class__':
                return False, "Access to __class__.__init__ chain"

        # Detect __dict__[...] subscript access
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == '__dict__':
                return False, "__dict__ subscript access"

    return True, ""


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Validate code for security before execution.

    Uses a two-layer approach:
    1. Regex pattern matching against DANGEROUS_PATTERNS (fast, broad)
    2. AST-based analysis to catch reflection/indirection bypasses

    Args:
        code: The Python code to validate

    Returns:
        Tuple of (is_safe, error_message). is_safe is True if the code
        passes all security checks.
    """
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            # Check if the matched pattern has a known safe-usage exception
            if any(key in description and check(code) for key, check in _EXCEPTION_CHECKS.items()):
                continue
            logger.warning(
                "Security: blocked dangerous code pattern [%s] — snippet: %.200s",
                description, code.replace('\n', ' '),
            )
            return False, f"Dangerous pattern detected: {description}"

    # Second layer: AST-based validation for reflection/indirection bypasses
    ast_safe, ast_reason = _validate_code_safety_ast(code)
    if not ast_safe:
        logger.warning(
            "Security: AST analysis blocked code [%s] — snippet: %.200s",
            ast_reason, code.replace('\n', ' '),
        )
        return False, f"Dangerous pattern detected (AST): {ast_reason}"

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
            logger.exception("Failed to load notebook state")
            return False
    
    def has_session_dump(self) -> bool:
        """Check if a session dump exists"""
        return self.session_path.exists()


# Template for unwinding any stale SCAI no-op stubs from plt.close/plt.show
# and installing fresh ones.  Embedded into f-string code templates via
# {_TPL_NOOP_INSTALL}.  The string must NOT contain { or } to be f-string safe.
_TPL_NOOP_INSTALL = (
    "# Temporarily disable plt.close/plt.show to keep figures alive.\n"
    "# Unwind any leftover SCAI no-op stubs from prior runs before installing fresh ones.\n"
    "_unwrap_close_guard = 0\n"
    "while getattr(plt.close, '_scai_noop', False) and _unwrap_close_guard < 64:\n"
    "    plt.close = getattr(plt.close, '_scai_prev', plt.close)\n"
    "    _unwrap_close_guard += 1\n"
    "_unwrap_show_guard = 0\n"
    "while getattr(plt.show, '_scai_noop', False) and _unwrap_show_guard < 64:\n"
    "    plt.show = getattr(plt.show, '_scai_prev', plt.show)\n"
    "    _unwrap_show_guard += 1\n"
    "# Defensive baseline restore for legacy sessions with unmarked stubs.\n"
    "try:\n"
    "    if '_original_plt_close' in globals() and plt.close is not _original_plt_close \\\n"
    "            and not getattr(plt.close, '_scai_noop', False):\n"
    "        plt.close = _original_plt_close\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    if '_capturing_show' in globals() and plt.show is not _capturing_show \\\n"
    "            and not getattr(plt.show, '_scai_noop', False):\n"
    "        plt.show = _capturing_show\n"
    "except Exception:\n"
    "    pass\n"
    "_orig_close = plt.close\n"
    "_orig_show = plt.show\n"
    "def _noop_close(*a, **kw): return None\n"
    "_noop_close._scai_noop = True\n"
    "_noop_close._scai_prev = _orig_close\n"
    "def _noop_show(*a, **kw): return None\n"
    "_noop_show._scai_noop = True\n"
    "_noop_show._scai_prev = _orig_show\n"
    "plt.close = _noop_close\n"
    "plt.show = _noop_show\n"
)


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
        import tempfile as _tmpfile_mod
        self._session_tmp = Path(_tmpfile_mod.gettempdir()) / f"scai-{self.session_id}"
        self._session_tmp.mkdir(parents=True, exist_ok=True)
        self.state_dir = state_dir
        # Caller passes session-specific working dir (e.g. sessions/{id}/workspace)
        self.working_dir = Path(working_dir).absolute()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        self.km: Optional[KernelManager] = None
        self.kc = None
        self.notebook_state = NotebookState(session_id, state_dir)
        self._initialized = False
        self.runtime_revision: int = 0
        self.saved_runtime_revision: int = 0
        self._session_restored = False
        self._session_restore_failed = False  # Track if session restore failed
        self._kernel_dead = False  # Set when kernel restart fails
        self._restarting = False  # reentrancy guard for _restart_kernel
        self._pending_save_tasks: set = set()
        # Snapshot V2 runtime dependency tracking (created/loaded user globals).
        self._known_user_globals: set[str] = set()
        self._last_execution_diff: Dict[str, Any] = {
            "created": set(),
            "loaded": set(),
            "removed": set(),
        }
        self.runtime_dependency_tracker = None  # set by agent / capture pipeline
        self._init_exec_lock()

    def _init_exec_lock(self):
        """Single lock serializing every coroutine that drives the kernel client
        sockets (execute + save_session). Two coroutines reading the same ZMQ
        socket concurrently triggers libzmq abort (see src/msg.cpp warning)."""
        self._exec_lock = asyncio.Lock()

    def _allowed_write_roots(self) -> List[str]:
        """Return the list of write-root paths rendered into the kernel init code,
        so host-side tests can assert that a path is NOT writable by the sandbox."""
        session_root_abs = str(self.working_dir.absolute().parent)
        return [session_root_abs, str(self._session_tmp)]

    async def start(self) -> dict:
        """Start the Jupyter kernel

        Returns:
            dict with keys:
                - started: bool - whether kernel was started (False if already running)
                - session_restored: bool - whether previous session was restored
                - restore_failed: bool - whether restore was attempted but failed
        """
        # Snapshot V2: unresolved recovery blocks kernel startup entirely.
        from backend.services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=self.session_id)

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

        # A freshly started+initialized kernel is alive regardless of any prior
        # dead state; reset before the best-effort restore below.
        self._kernel_dead = False

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
                msg = await asyncio.to_thread(self.kc.get_shell_msg, timeout=5)
                if msg['msg_type'] == 'kernel_info_reply':
                    break
            except QueueEmpty:
                # ZMQ poll timed out — kernel not ready yet
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Kernel not ready yet: {e}")
                await asyncio.sleep(0.5)
    
    async def _initialize_kernel(self):
        """Initialize kernel with common setup"""
        if self._initialized:
            return
        
        # NOTE: This is a plain string (not f-string) because the code contains
        # dict literals ({...}), inner f-strings, and other braces that conflict
        # with f-string parsing.  The only dynamic value (session_id) is injected
        # via str.replace() at the end.
        init_code = """
# === MANDATORY: Set ALL random seeds for reproducibility ===
import random
import os
from pathlib import Path

SEED = 42

# === Isolate Numba cache per session to prevent concurrent cache corruption ===
_numba_cache = Path('/tmp/numba_cache___SESSION_ID__')
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(_numba_cache)
# Prevent Numba from spawning extra threads that compete for cache locks
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

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

# Set matplotlib defaults for publication-quality figures
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,   # Vectorized fonts for editable PDFs
    'ps.fonttype': 42,
})

# Capture the true matplotlib.pyplot.close/show baselines before any wrapper
# installs stubs over them (e.g. _TPL_NOOP_INSTALL).  Placed before the
# Combined Monkey-patch below to avoid the _WORKSPACE_DIR NameError that
# makes lines 781+ dead code in the kernel.
_original_plt_close = plt.close
_original_plt_show = plt.show

# === Combined Monkey-patch: Figure.savefig with PNG->JPG + path redirect ===
import matplotlib.figure as _mpl_figure
_original_savefig = _mpl_figure.Figure.savefig

_FIGURES_DIR = _WORKSPACE_DIR / 'figures'
_WORKSPACE_DIR_RESOLVED = _WORKSPACE_DIR.resolve()

def _combined_savefig(self, fname, *args, **kwargs):
    '''Combined patch: PNG->JPG conversion + workspace root redirect.'''
    # Step 1: Handle PNG->JPG conversion (if fname is a path string)
    _fname_str = str(fname) if isinstance(fname, (str, os.PathLike)) else None
    _needs_jpg_convert = _fname_str and _fname_str.lower().endswith('.png')

    if _needs_jpg_convert:
        try:
            _base, _ = os.path.splitext(_fname_str)
            _jpg_path = _base + '.jpg'
            # Modify kwargs only; let the final return handle saving
            kwargs = dict(kwargs)
            kwargs['format'] = 'jpeg'
            kwargs.setdefault('facecolor', 'white')
            if 'pil_kwargs' not in kwargs:
                kwargs['pil_kwargs'] = {}
            kwargs['pil_kwargs'].setdefault('quality', 85)
            kwargs['pil_kwargs'].setdefault('optimize', True)
            print(f"[Auto-convert] {os.path.basename(_fname_str)} -> {os.path.basename(_jpg_path)}")
            # Only update fname, do NOT save here — let final return handle it
            fname = _jpg_path
        except Exception as _err:
            # Fallback: save as original PNG if JPEG conversion fails
            print(f"[Auto-convert] Warning: {_err}, saving as PNG")

    # Step 2: Redirect workspace root -> figures/ (only for path strings)
    _redirect_failed = False
    if isinstance(fname, (str, os.PathLike)):
        try:
            _p = Path(str(fname)).resolve()
            if _p.parent == _WORKSPACE_DIR_RESOLVED:
                _new = _FIGURES_DIR / _p.name
                print(f"[savefig-redirect] {_p.name} -> figures/")
                fname = str(_new)
            elif not str(_p).startswith(str(_WORKSPACE_DIR_RESOLVED)):
                # Path resolved outside the workspace root — log for diagnostics (3c)
                print(f"[savefig-redirect] skip: '{fname}' -> resolved to {_p} (workspace is {_WORKSPACE_DIR_RESOLVED}, cwd is {os.getcwd()})")
        except Exception as _redir_err:
            print(f"[savefig-redirect] failed: {_redir_err}, saving to original location")
            _redirect_failed = True

    try:
        return _original_savefig(self, fname, *args, **kwargs)
    except Exception as _save_err:
        # Last resort: if redirect caused issues, retry with original fname
        if not _redirect_failed:
            raise
        print(f"[savefig] redirect recovery failed: {_save_err}, trying original path")
        _orig_path_str = str(fname)
        if str(_FIGURES_DIR) in _orig_path_str:
            _relative = _orig_path_str.replace(str(_FIGURES_DIR), "", 1)
            _recovery = str(_WORKSPACE_DIR / _relative.lstrip("/"))
        else:
            _recovery = _orig_path_str
        return _original_savefig(self, _recovery, *args, **kwargs)

_mpl_figure.Figure.savefig = _combined_savefig

# Capture the true matplotlib.pyplot.close baseline before any wrapper installs
# stubs over it. Used by per-execution wrappers to defensively force-restore
# when a previous run left a stale no-op behind (e.g. legacy unmarked stubs
# from sessions started before the unwind-on-install fix).
_original_plt_close = plt.close

# === Monkey-patch plt.show() to capture figures as display_data ===
# With Agg backend, plt.show() is a no-op — figures are silently lost.
# This patch saves all open figures as inline PNG images via IPython.display,
# so the Jupyter kernel emits display_data messages that get captured.
import io as _io
_original_plt_show = plt.show  # Baseline before any SCAI patches (symmetric with _original_plt_close)

def _capturing_show(*args, **kwargs):
    '''Capture all open figures as inline images, then close them.'''
    import matplotlib.pyplot as _plt
    _fignums = _plt.get_fignums()
    if not _fignums:
        return
    try:
        from IPython.display import display, Image as IPImage
    except ImportError:
        _original_plt_show(*args, **kwargs)
        return
    for _fig_num in _fignums:
        _fig = _plt.figure(_fig_num)
        _buf = _io.BytesIO()
        _original_savefig(_fig, _buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        _buf.seek(0)
        display(IPImage(data=_buf.read(), format='png'))
        _buf.close()
    _plt.close('all')

plt.show = _capturing_show  # Baseline capturing_show; referenced by defensive restore in _TPL_NOOP_INSTALL

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
    sns.set_theme(style="ticks", font_scale=1.2)
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
""".replace('__SESSION_ID__', self.session_id)
        result = await self._execute_unlocked(init_code, save_to_history=False, skip_safety_check=True)
        if result.output:
            logger.debug("Kernel init stdout: %s", result.output[:1000])

        # Set working directory paths and install I/O sandbox (requires f-string with self.working_dir)
        working_dir_abs = str(self.working_dir.absolute())
        state_dir_abs = str(self.notebook_state.state_dir)
        # Session root is the parent of workspace (e.g. sessions/{id}/)
        session_root_abs = str(self.working_dir.absolute().parent)
        path_setup_code = f"""
# === Set working directory paths ===
import os
from pathlib import Path

# Redirect tempfile to per-session subdir
os.environ['TMPDIR'] = {repr(str(self._session_tmp))}
os.environ['TEMP'] = {repr(str(self._session_tmp))}
os.environ['TMP'] = {repr(str(self._session_tmp))}

_WORKSPACE_DIR = Path({repr(working_dir_abs)})
os.chdir(_WORKSPACE_DIR)

# Create standard subdirectories
(_WORKSPACE_DIR / 'figures').mkdir(exist_ok=True)
(_WORKSPACE_DIR / 'tables').mkdir(exist_ok=True)

# Set scanpy figdir to absolute path (prevent relative path issues)
try:
    import scanpy as sc
    sc.settings.figdir = str(_WORKSPACE_DIR / 'figures')
except Exception as _sc_err:
    print(f"[sandbox] scanpy figdir setup skipped: {{_sc_err}}")

# === Runtime I/O sandbox: intercept writes outside allowed directories ===
import builtins as _builtins
import pathlib as _pathlib

_ORIGINAL_OPEN = _builtins.open
_ORIGINAL_MKDIR = _pathlib.Path.mkdir

# Directories where write operations are allowed
_ALLOWED_WRITE_ROOTS = [
    {repr(session_root_abs)},   # Session root (workspace/, states/, uploads/, reports/)
    {repr(str(self._session_tmp))},   # Per-session temporary files
]

# System paths (pip installs, font cache, etc.) — always allowed
_SYSTEM_WRITE_PREFIXES = [
    '/usr/', '/opt/', '/var/',
    os.path.expanduser('~/.cache'),
    os.path.expanduser('~/.local'),
    os.path.expanduser('~/.config'),
]

def _is_write_allowed(filepath):
    '''Check if writing to this path is allowed.'''
    try:
        resolved = str(Path(filepath).resolve())
    except Exception as e:
        print(f"[sandbox] Path resolution failed for {{filepath!r}}: {{e}}")
        return False  # Can't resolve -> deny by default for safety
    # Check allowed roots
    for root in _ALLOWED_WRITE_ROOTS:
        r = root.rstrip('/')
        if resolved == r or resolved.startswith(r + '/'):
            return True
    # Check system paths
    for prefix in _SYSTEM_WRITE_PREFIXES:
        p = prefix.rstrip('/')
        if resolved == p or resolved.startswith(p + '/'):
            return True
    return False

def _sandboxed_open(file, mode='r', *args, **kwargs):
    '''Sandboxed open() that blocks writes outside allowed directories.'''
    # Integer file descriptors are already-open resources (e.g. the inter-process
    # pipes multiprocessing/dask use under GRNBoost2). The access decision was made
    # when they were opened, and they are not path-resolvable — only path-like
    # targets are subject to the write check.
    if isinstance(file, int):
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)
    if isinstance(mode, str) and any(m in mode for m in ('w', 'a', 'x')):
        if not _is_write_allowed(file):
            raise PermissionError(
                f"\\n[Sandbox] Write blocked: {{file}}\\n"
                f"Files must be saved within the session workspace.\\n"
                f"  - CSV/Excel -> {{_WORKSPACE_DIR}}/tables/\\n"
                f"  - H5AD      -> {{_WORKSPACE_DIR}}/\\n"
                f"  - Figures are saved automatically by the system."
            )
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)

def _sandboxed_mkdir(self, *args, **kwargs):
    '''Sandboxed Path.mkdir() that blocks directory creation outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] mkdir blocked: {{self}}\\n"
            f"Directories must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_MKDIR(self, *args, **kwargs)

_ORIGINAL_WRITE_TEXT = _pathlib.Path.write_text
_ORIGINAL_WRITE_BYTES = _pathlib.Path.write_bytes

def _sandboxed_write_text(self, *args, **kwargs):
    '''Sandboxed Path.write_text() that blocks writes outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Write blocked: {{self}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_WRITE_TEXT(self, *args, **kwargs)

def _sandboxed_write_bytes(self, *args, **kwargs):
    '''Sandboxed Path.write_bytes() that blocks writes outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Write blocked: {{self}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_WRITE_BYTES(self, *args, **kwargs)

_builtins.open = _sandboxed_open
_pathlib.Path.mkdir = _sandboxed_mkdir
_pathlib.Path.write_text = _sandboxed_write_text
_pathlib.Path.write_bytes = _sandboxed_write_bytes

# === Sandbox file/directory deletion ===
_ORIGINAL_OS_REMOVE = os.remove
_ORIGINAL_OS_UNLINK = os.unlink
_ORIGINAL_OS_RMDIR = os.rmdir
_ORIGINAL_PATH_UNLINK = _pathlib.Path.unlink
_ORIGINAL_PATH_RMDIR = _pathlib.Path.rmdir

def _sandboxed_os_remove(path, *args, **kwargs):
    '''Sandboxed os.remove() that blocks deletes outside allowed directories.'''
    if not _is_write_allowed(path):
        raise PermissionError(
            f"\\n[Sandbox] Delete blocked: {{path}}\\n"
            f"Deletes must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_REMOVE(path, *args, **kwargs)

def _sandboxed_os_unlink(path, *args, **kwargs):
    '''Sandboxed os.unlink() that blocks deletes outside allowed directories.'''
    if not _is_write_allowed(path):
        raise PermissionError(
            f"\\n[Sandbox] Delete blocked: {{path}}\\n"
            f"Deletes must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_UNLINK(path, *args, **kwargs)

def _sandboxed_os_rmdir(path, *args, **kwargs):
    '''Sandboxed os.rmdir() that blocks rmdir outside allowed directories.'''
    if not _is_write_allowed(path):
        raise PermissionError(
            f"\\n[Sandbox] rmdir blocked: {{path}}\\n"
            f"Directory removal must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_RMDIR(path, *args, **kwargs)

def _sandboxed_path_unlink(self, *args, **kwargs):
    '''Sandboxed Path.unlink() that blocks deletes outside allowed directories.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Delete blocked: {{self}}\\n"
            f"Deletes must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_PATH_UNLINK(self, *args, **kwargs)

def _sandboxed_path_rmdir(self, *args, **kwargs):
    '''Sandboxed Path.rmdir() that blocks rmdir outside allowed directories.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] rmdir blocked: {{self}}\\n"
            f"Directory removal must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_PATH_RMDIR(self, *args, **kwargs)

os.remove = _sandboxed_os_remove
os.unlink = _sandboxed_os_unlink
os.rmdir = _sandboxed_os_rmdir
_pathlib.Path.unlink = _sandboxed_path_unlink
_pathlib.Path.rmdir = _sandboxed_path_rmdir

print(f"Working directory set to: {{os.getcwd()}}")
print(f"I/O sandbox active: writes restricted to session directory")
"""
        path_result = await self._execute_unlocked(path_setup_code, save_to_history=False, skip_safety_check=True)
        if path_result.output:
            logger.debug("Kernel path-setup stdout: %s", path_result.output[:1000])

        self._initialized = True
        logger.info("Kernel initialized")
    
    async def _restore_session(self):
        """Restore session from per-variable streaming dill dump if available.

        Format produced by `save_session`: a sequence of `dill.dump((name, obj), f)`
        frames in a single uncompressed .pkl file. Read by looping `dill.load`
        until EOFError. No compression handling — snapshots are decompressed
        out-of-kernel before being placed at this path.
        """
        # Recover any interrupted runtime-pair apply FIRST — before the
        # has_session_dump() early-return — so a crashed apply that left no live
        # pkl cannot leave a journal lingering until a later start (where it could
        # otherwise interfere with a subsequently saved pkl).
        from .runtime_apply import (
            recover_runtime_apply_journal, verify_live_pkl_or_raise, RuntimeApplyError,
        )
        recover_runtime_apply_journal(self.session_id)

        # Snapshot V2: unresolved APPLYING journals block kernel rehydrate
        from backend.services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=self.session_id)

        self.notebook_state.load()

        if not self.notebook_state.has_session_dump():
            logger.info("No session dump to restore")
            return

        # A present pkl with a missing/invalid HMAC is a hard error, not a silent
        # skip: continuing would rehydrate a kernel inconsistent with the state
        # that referenced this pkl. Raising lets the caller surface a typed
        # runtime-failure state (Stage 1 safety contract).
        try:
            verify_live_pkl_or_raise(self.session_id)
        except RuntimeApplyError:
            # Symmetric with the dill-restore-failure path below: mark the failure
            # so start()'s "restore_failed" result reflects it, then abort startup.
            self._session_restore_failed = True
            logger.error(
                "[%s] live pkl present but HMAC invalid; aborting session restore",
                self.session_id,
            )
            raise

        session_path = str(self.notebook_state.session_path.absolute())
        escaped_path = repr(session_path)
        logger.info(f"Restoring session from {session_path}")

        restore_code = f"""
import dill

_restored_names = []
try:
    with open({escaped_path}, 'rb') as _f:
        while True:
            try:
                _name, _obj = dill.load(_f)
            except EOFError:
                break
            globals()[_name] = _obj
            _restored_names.append(_name)

    print(f"Session restored: {{len(_restored_names)}} variables")
    print(f"Restored: {{_restored_names}}")
except Exception as e:
    print(f"__SCAI_RESTORE_ERROR__: {{type(e).__name__}}: {{e}}")
    raise
"""
        result = await self._execute_unlocked(restore_code, save_to_history=False, skip_safety_check=True)

        if result.success and "Session restored:" in (result.output or ""):
            self._session_restored = True
            self._session_restore_failed = False
            logger.info("Session restored successfully")
        else:
            # P1-11: dill restore failure leaves the kernel without previously-saved
            # variables while the notebook metadata has already been loaded. Continuing
            # would yield silent NameErrors downstream. Surface the failure loudly and
            # abort start-up so the workflow can react (retry / fresh kernel / restore
            # from earlier snapshot).
            self._session_restore_failed = True
            detail = result.error or result.output or "unknown error"
            logger.error(
                f"[{self.session_id}] dill session restore failed: {detail}"
            )
            raise RuntimeError(
                f"Sandbox restore failed for session {self.session_id}: {detail}. "
                f"Previous kernel variable state is unavailable. Start a fresh kernel "
                f"or restore from an earlier snapshot."
            )
    
    async def save_session(self):
        """Save the current session state by streaming dill per-variable.

        Each saved variable is written as one independent `dill.dump((name, obj), f)`
        frame. Variables that cannot be pickled are skipped via
        `file.tell()/seek()/truncate()` rollback so the file stays parseable.

        No compression is applied here — compression happens out-of-kernel during
        snapshot creation. This keeps the kernel's peak memory low (a single
        variable's dill chunk buffer instead of multiple full pickle byte strings
        held simultaneously, which previously OOM-killed the kernel and crashed
        the parent backend via libzmq fq.cpp abort).
        """
        async with self._exec_lock:
            session_path = str(self.notebook_state.session_path.absolute())
            escaped_path = repr(session_path)
            logger.info(f"Saving session to {session_path}")

            save_code = """\
import dill
import os
try:
    import anndata as ad
except ImportError:
    ad = None

_vars_to_save = ['adata', 'adata_control', 'adata_treatment', 'adata_merged',
                 'ldata', 'ldata_control', 'ldata_treatment',
                 'results', 'markers', 'deg_results', 'pathway_results',
                 'cell_types', 'cluster_markers', 'differential_genes']

# Collect explicit list, then sweep globals() for any other AnnData we missed.
_to_dump = []  # list of (name, obj)
_seen = set()
for _var_name in _vars_to_save:
    if _var_name in dir():
        _to_dump.append((_var_name, globals()[_var_name]))
        _seen.add(_var_name)
if ad is not None:
    for _var_name, _obj in list(globals().items()):
        if isinstance(_obj, ad.AnnData) and _var_name not in _seen:
            _to_dump.append((_var_name, _obj))
            _seen.add(_var_name)

_skipped_vars = []  # P1-11: report what could not be persisted

# Write to a temp file then atomically replace, so a mid-write crash can't
# corrupt the only good pkl.
_tmp_path = """ + escaped_path + """ + '.tmp'
_saved_count = 0
try:
    with open(_tmp_path, 'wb') as _f:
        for _var_name, _obj in _to_dump:
            _checkpoint = _f.tell()
            try:
                # Stream one (name, obj) frame; dill writes incrementally so peak
                # memory is bounded by this single object's pickling, not the sum.
                dill.dump((_var_name, _obj), _f)
                _saved_count += 1
            except Exception as _e:
                # Roll back the half-written frame so subsequent loads stay valid.
                _f.seek(_checkpoint)
                _f.truncate()
                _skipped_vars.append((
                    _var_name,
                    type(_obj).__name__,
                    type(_e).__name__,
                    str(_e)[:200],
                ))
    os.replace(_tmp_path, """ + escaped_path + """)
except Exception:
    try:
        os.remove(_tmp_path)
    except OSError:
        pass
    raise

print(f"Session saved: {_saved_count} variables")
if _skipped_vars:
    for _sv in _skipped_vars:
        print(f"__SCAI_SAVE_SKIPPED__\\t{_sv[0]}\\t{_sv[1]}\\t{_sv[2]}\\t{_sv[3]}")
    print(f"__SCAI_SAVE_SKIPPED_COUNT__: {len(_skipped_vars)}")
"""
            result = await self._execute_unlocked(save_code, save_to_history=False, skip_safety_check=True)

            if result.success:
                skipped = []
                for line in (result.output or "").splitlines():
                    if line.startswith("__SCAI_SAVE_SKIPPED__\t"):
                        parts = line.split("\t")
                        if len(parts) >= 5:
                            skipped.append({
                                "name": parts[1],
                                "type": parts[2],
                                "exc": parts[3],
                                "msg": parts[4],
                            })
                if skipped:
                    for sv in skipped:
                        logger.warning(
                            f"[{self.session_id}] save_session: skipping unpicklable "
                            f"variable '{sv['name']}' ({sv['type']}): "
                            f"{sv['exc']}: {sv['msg']}"
                        )
                    logger.warning(
                        f"[{self.session_id}] save_session: {len(skipped)} variables were "
                        f"not persisted and will be unavailable on restore: "
                        f"{[sv['name'] for sv in skipped]}"
                    )
                    self._last_save_skipped = skipped
                else:
                    self._last_save_skipped = []
                logger.info("Session saved successfully")

                # Write HMAC signature (host-side, outside sandbox-writable roots)
                try:
                    pkl_path = Path(self.notebook_state.session_path)
                    if pkl_path.exists():
                        _write_pkl_signature(
                            pkl_path, _integrity_dir(self.session_id)
                        )
                except Exception:
                    logger.exception("Failed to write pkl integrity signature")
            else:
                logger.warning(f"Failed to save session: {result.error}")

            return result.success

    async def _restart_kernel(self) -> None:
        """Restart the kernel, then best-effort restore the session.

        Kernel-level failure (restart/ready/init) marks the sandbox dead. A
        restore failure (missing/corrupt dill dump) leaves the kernel fully
        usable, so it must NOT set _kernel_dead. Reentrancy is guarded because
        _initialize_kernel runs code through execute(), which can itself
        trigger a restart on a protocol error.
        """
        if self._restarting:
            logger.warning(
                f"[{self.session_id}] _restart_kernel reentered; skipping nested restart"
            )
            return
        self._restarting = True
        try:
            try:
                await asyncio.to_thread(self.km.restart_kernel)

                # Rebuild client connection.  Brief sleep to let any
                # in-flight ZMQ reader threads finish their current poll
                # cycle before we close the channels underneath them.
                await asyncio.sleep(0.3)

                if self.kc is not None:
                    try:
                        self.kc.stop_channels()
                    except Exception as e:
                        logger.debug(f"Failed to stop kernel channels during recovery: {e}")
                self.kc = self.km.client()
                self.kc.start_channels()

                # Drain stale messages that may have arrived on the new
                # channels before the kernel was fully initialised (e.g.
                # kernel-info broadcasts).  These carry the new session key
                # so they won't trigger "Invalid Signature", but leaving
                # them in the queue can confuse the first execute().
                try:
                    for _ in range(10):
                        await asyncio.to_thread(self.kc.get_iopub_msg, timeout=0.1)
                except QueueEmpty:
                    pass  # queue drained

                # Wait for kernel to be ready
                await asyncio.wait_for(self._wait_for_ready(), timeout=30)

                # Re-initialize the kernel
                self._initialized = False
                await self._initialize_kernel()
            except Exception as restart_error:
                logger.error(f"Failed to restart kernel: {restart_error}")
                self._kernel_dead = True
                return

            # Kernel is healthy again.
            self._kernel_dead = False

            # Restore is best-effort: a corrupt dump must not brick a live kernel.
            try:
                await self._restore_session()
            except Exception as restore_error:
                logger.error(
                    f"[{self.session_id}] Session restore failed after kernel restart "
                    f"(kernel remains usable): {restore_error}"
                )

            logger.info(f"Kernel restarted for session {self.session_id}")
        finally:
            self._restarting = False

    def mark_runtime_committed(self, runtime_revision: int | None = None) -> None:
        """Called after a snapshot capture transaction commits the runtime generation.

        Sets ``saved_runtime_revision`` to *runtime_revision* (or the current
        ``runtime_revision`` if omitted), marking the point at which the last
        successful commit occurred.
        """
        if runtime_revision is None:
            runtime_revision = self.runtime_revision
        self.saved_runtime_revision = runtime_revision


    def get_runtime_dependency_names(self) -> set[str]:
        """Return accumulated required runtime names from the dependency tracker."""
        tracker = self.runtime_dependency_tracker
        if tracker is None:
            return set()
        return set(tracker.required_outputs)

    def record_declared_outputs(self, names) -> None:
        """Merge explicit persistent_outputs into the dependency tracker."""
        if not names:
            return
        if self.runtime_dependency_tracker is None:
            from backend.services.snapshot_v2.runtime_registry import (
                RuntimeDependencyTracker,
            )
            self.runtime_dependency_tracker = RuntimeDependencyTracker()
        self.runtime_dependency_tracker.merge_declared(set(names))

    async def _snapshot_user_globals(self) -> set[str]:
        """Return non-underscore user global names currently in the kernel."""
        code = (
            "import json as _jg\n"
            "_names = [n for n in list(globals().keys()) if not n.startswith('_')]\n"
            "print('__USER_GLOBALS__:' + _jg.dumps(sorted(_names)))\n"
        )
        result = await self._execute_unlocked(
            code, save_to_history=False, skip_safety_check=True
        )
        if not result.success:
            return set(self._known_user_globals)
        for line in (result.output or "").splitlines():
            if line.startswith("__USER_GLOBALS__:"):
                raw = line[len("__USER_GLOBALS__:"):].strip()
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return set(str(x) for x in parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
        return set(self._known_user_globals)

    def _record_globals_diff(
        self,
        before: set[str],
        after: set[str],
        declared: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Update tracker from a created/loaded user-global diff.

        - created: names present after but not before
        - loaded: names present both before and after (treated as read/used)
        - removed: names present before but not after
        """
        created = after - before
        removed = before - after
        loaded = before & after
        self._last_execution_diff = {
            "created": set(created),
            "loaded": set(loaded),
            "removed": set(removed),
        }
        self._known_user_globals = set(after)
        if self.runtime_dependency_tracker is None:
            from backend.services.snapshot_v2.runtime_registry import (
                RuntimeDependencyTracker,
            )
            self.runtime_dependency_tracker = RuntimeDependencyTracker()
        # Only promote loaded names that were previously created by a cell.
        # Passing the full intersection would promote every pre-existing name.
        # `removed` names (del'd from the namespace) are pruned from the required
        # set so a declared-then-deleted transient (e.g. a loompy connection)
        # cannot permanently break snapshot capture.
        self.runtime_dependency_tracker.record_execution(
            created=created,
            loaded=loaded,
            declared=set(declared or set()),
            removed=removed,
        )
        return self._last_execution_diff

    async def execute(
        self,
        code: str,
        save_to_history: bool = True,
        skip_safety_check: bool = False,
        declared_outputs: Optional[List[str]] = None,
        track_globals: bool = False,
    ) -> CodeExecutionResult:
        """Execute code in the kernel.

        Args:
            code: Python code to execute
            save_to_history: Whether to save successful execution to notebook history
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
            declared_outputs: Optional persistent_outputs declared by the agent tool call.
            track_globals: When True (and save_to_history), record created/loaded
                user-global diffs into the runtime dependency tracker.
        """
        declared = set(declared_outputs or [])
        before: set[str] = set()
        should_track = bool(track_globals or declared) and save_to_history
        async with self._exec_lock:
            if should_track:
                before = await self._snapshot_user_globals()
            result = await self._execute_unlocked(
                code, save_to_history=save_to_history, skip_safety_check=skip_safety_check
            )
            if result.success and should_track:
                after = await self._snapshot_user_globals()
                self._record_globals_diff(before, after, declared=declared)
        if result.success and save_to_history:
            self.runtime_revision += 1
            if declared:
                self.record_declared_outputs(declared)
        return result

    async def _execute_unlocked(
        self,
        code: str,
        save_to_history: bool = True,
        skip_safety_check: bool = False
    ) -> CodeExecutionResult:
        """Execute code in the kernel (unlocked — caller must hold _exec_lock).

        Args:
            code: Python code to execute
            save_to_history: Whether to save successful execution to notebook history
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
        """
        # Check if kernel is dead (failed restart)
        if self._kernel_dead:
            return CodeExecutionResult(
                success=False,
                output="",
                error="Kernel is unavailable (restart failed). Please restart the analysis session.",
                figures=[]
            )

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

        # Guard against silent kernel death (e.g. OOM kill, libzmq abort).
        # km.is_alive() checks whether the kernel subprocess is still running.
        # Without this check, kc.execute() on a dead kernel triggers ZMQ channel
        # corruption that calls abort() inside libzmq, killing the entire process.
        if self.km is not None and not self.km.is_alive():
            logger.warning(
                f"[{self.session_id}] Kernel process is dead before execute; "
                f"attempting restart"
            )
            await self._restart_kernel()
            if self._kernel_dead:
                return CodeExecutionResult(
                    success=False,
                    output="",
                    error="Kernel died unexpectedly and restart failed. Please restart the analysis session.",
                    figures=[]
                )

        outputs = []
        figures = []
        output_text = []
        error_msg = None
        execution_count = 0
        
        try:
            # Send execute request
            msg_id = self.kc.execute(code)

            # Collect outputs — only process messages belonging to THIS execution (matched by msg_id).
            # Timeout is wall-clock: the deadline is fixed before the loop so a task that
            # keeps streaming output (tqdm/scanpy verbose) cannot reset it indefinitely.
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = compute_remaining_timeout(deadline, time.monotonic())
                msg = None
                if remaining > 0:
                    try:
                        # Pass timeout directly to get_iopub_msg so the thread
                        # returns after the ZMQ-level poll expires.  Previously
                        # asyncio.wait_for(asyncio.to_thread(get_iopub_msg)) was
                        # used, but that only cancels the *coroutine* — the
                        # underlying thread stays blocked on ZMQ recv forever.
                        # Each timeout leaked an orphaned reader thread; multiple
                        # threads reading the same ZMQ socket is unsafe and
                        # eventually triggers libzmq abort (src/msg.cpp:387).
                        msg = await asyncio.to_thread(
                            self.kc.get_iopub_msg,
                            timeout=min(remaining, IOPUB_POLL_CAP)
                        )
                    except QueueEmpty:
                        # ZMQ poll timed out — no message arrived within the
                        # poll cap.  Re-check the wall-clock deadline and
                        # kernel liveness before looping or declaring timeout.
                        if self.km and not self.km.is_alive():
                            logger.warning(
                                f"[{self.session_id}] Kernel died during execution; "
                                f"aborting output collection"
                            )
                            await self._restart_kernel()
                            if self._kernel_dead:
                                return CodeExecutionResult(
                                    success=False,
                                    output="",
                                    error="Kernel died during execution and restart failed.",
                                    figures=[]
                                )
                            # Kernel restarted successfully; the old execution is gone,
                            # return an error so the caller can retry.
                            return CodeExecutionResult(
                                success=False,
                                output="",
                                error="Kernel died during execution. Restarted successfully, but the execution was lost.",
                                figures=[]
                            )
                        if compute_remaining_timeout(deadline, time.monotonic()) > 0:
                            continue
                        remaining = 0  # fall through to timeout handling below

                if remaining <= 0:
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
                                    drain_msg = await asyncio.to_thread(
                                        self.kc.get_iopub_msg, timeout=1.0
                                    )
                                    if drain_msg['msg_type'] == 'status' and drain_msg['content']['execution_state'] == 'idle':
                                        break
                            except QueueEmpty:
                                # Kernel may be in bad state, restart it
                                logger.warning(f"Kernel did not return to idle after interrupt for session {self.session_id}")
                                await self._restart_kernel()
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
            logger.exception("Execution error")
            if _is_kernel_protocol_error(e):
                logger.warning(
                    f"Kernel protocol corruption detected for session {self.session_id}; "
                    f"restarting kernel"
                )
                await self._restart_kernel()
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
            # 在已有事件循环中调度异步保存
            task = loop.create_task(self.save_session())
            self._pending_save_tasks.add(task)
            task.add_done_callback(self._pending_save_tasks.discard)
        except RuntimeError:
            # No running loop - safe to create a new one
            try:
                asyncio.run(self.save_session())
            except Exception as e:
                logger.exception("Failed to save session state")
    
    async def save_state_async(self):
        """Save current state asynchronously"""
        self.notebook_state.save()
        await self.save_session()
    
    async def stop(self):
        """Stop the kernel and save state. Always shuts down kernel even if save fails."""
        # Try to save session before stopping, but don't let failure prevent kernel cleanup
        try:
            await self.await_pending_saves()
            await self.save_session()
            self.notebook_state.save()
        except Exception as e:
            logger.exception(f"Failed to save session during stop for {self.session_id}")
        finally:
            if self.kc is not None:
                try:
                    self.kc.stop_channels()
                except Exception as e:
                    logger.exception(f"Error stopping kernel channels for {self.session_id}")
                self.kc = None

            if self.km is not None:
                try:
                    self.km.shutdown_kernel(now=True)
                except Exception as e:
                    logger.exception(f"Error shutting down kernel for {self.session_id}")
                self.km = None
                self._initialized = False

            logger.info(f"Kernel stopped for session {self.session_id}")
    
    async def await_pending_saves(self):
        """Wait for all pending save tasks to complete."""
        while self._pending_save_tasks:
            tasks = list(self._pending_save_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

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
