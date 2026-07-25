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
    (r'\b__import__\s*\(', '__import__ function'),
    # Block the dangerous built-in, but allow ordinary attribute calls such as
    # ``re.compile(...)``.  The previous word-boundary expression matched both.
    (r'(?<![\w.])compile\s*\(', 'compile() function'),
    (r'\bimportlib\b', 'importlib module'),

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
    """Allow only the small, non-executing importlib surface analyses need."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    allowed_modules = {"importlib", "importlib.util", "importlib.metadata"}
    allowed_from = {
        "importlib": {"import_module", "reload"},
        "importlib.util": {"find_spec"},
        "importlib.metadata": {
            "version", "PackageNotFoundError", "packages_distributions",
        },
    }
    allowed_calls = {
        "importlib.import_module", "importlib.reload",
        "importlib.util.find_spec", "importlib.metadata.version",
        "importlib.metadata.packages_distributions",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("importlib") and alias.name not in allowed_modules
                   for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("importlib"):
            if node.module not in allowed_from:
                return False
            if any(alias.name not in allowed_from[node.module] for alias in node.names):
                return False
        elif isinstance(node, ast.Call):
            parts = []
            expr = node.func
            while isinstance(expr, ast.Attribute):
                parts.append(expr.attr)
                expr = expr.value
            if isinstance(expr, ast.Name) and expr.id == "importlib":
                dotted = ".".join([expr.id, *reversed(parts)])
                if dotted not in allowed_calls:
                    return False
    return True


def _is_safe_file_write(code: str) -> bool:
    """Reject static ``open(..., write-mode)`` calls on unsafe *literal* paths.

    Aliased or computed paths are allowed here and re-enforced at runtime by
    the kernel's ``_sandboxed_open`` write allowlist (``_ALLOWED_WRITE_ROOTS``),
    which is the authoritative boundary for anything too dynamic to prove
    statically. Keeping this check a cheap early-reject for obvious-bad
    literals avoids a parallel path-provenance prover that would have to grow
    with every new idiom (BinOp/JoinedStr/Call chains, fixpoint alias
    propagation) to track a boundary the runtime already enforces.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    def _literal_is_safe(value: str) -> bool:
        normalized = value.replace("\\", "/").lower()
        return (
            normalized.startswith("/tmp/scai-")
            or "/workspace/" in normalized
            or "/figures/" in normalized
            or "/tables/" in normalized
            or "/annotation_validation/" in normalized
            or normalized.startswith((
                "figures/", "tables/", "annotation_validation/",
                "./figures/", "./tables/", "./annotation_validation/",
            ))
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "open" or not node.args:
            continue
        mode = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        if not (isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+"))):
            continue
        path_arg = node.args[0]
        # Only literal paths can be ruled on statically; everything else is
        # resolved and gated by _sandboxed_open at execution time.
        if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
            if not _literal_is_safe(path_arg.value):
                return False
    return True


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
    dangerous_string_aliases: dict[str, str] = {}
    namespace_aliases: set[str] = set()

    def _constant_string(expr) -> Optional[str]:
        """Resolve only statically known strings; ordinary string data is safe."""
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Name):
            return dangerous_string_aliases.get(expr.id)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left = _constant_string(expr.left)
            right = _constant_string(expr.right)
            if left is not None and right is not None:
                return left + right
        return None

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
            if expr.id in {"eval", "exec", "__import__", "compile"}:
                return expr.id
            return blocked_direct_calls.get(expr.id)
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            module_name = module_aliases.get(expr.value.id)
            if module_name == "subprocess" and expr.attr in _SUBPROCESS_EXEC_FUNCS:
                return f"subprocess.{expr.attr}"
            if module_name == "os" and expr.attr == "system":
                return "os.system"
        return None

    def _is_namespace_expr(expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in namespace_aliases
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            if expr.func.id in {"globals", "locals"} and not expr.args:
                return True
            if expr.func.id == "vars" and len(expr.args) == 1:
                target = expr.args[0]
                if isinstance(target, ast.Name):
                    target_name = module_aliases.get(target.id, target.id)
                    return target_name in {"builtins", "__builtins__"}
        return False

    def _subscript_key(expr) -> Optional[str]:
        if not isinstance(expr, ast.Subscript):
            return None
        return _constant_string(expr.slice)

    def _dangerous_lookup_from_expr(expr) -> Optional[str]:
        if isinstance(expr, ast.Subscript) and _is_namespace_expr(expr.value):
            key = _subscript_key(expr)
            if key in {"eval", "exec", "__import__", "compile"}:
                return key
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "getattr"
            and len(expr.args) >= 2
        ):
            target = expr.args[0]
            attr = _constant_string(expr.args[1])
            if isinstance(target, ast.Name):
                target_name = module_aliases.get(target.id, target.id)
                if (
                    target_name in {"builtins", "__builtins__"}
                    and attr in {"eval", "exec", "__import__", "compile"}
                ):
                    return attr
        return None

    def _propagate_alias_to_target(target, value) -> bool:
        changed_alias = False
        if isinstance(target, ast.Name):
            module_name = _module_alias_from_expr(value)
            if module_name and module_aliases.get(target.id) != module_name:
                module_aliases[target.id] = module_name
                changed_alias = True

            blocked_reason = _blocked_call_from_expr(value)
            blocked_reason = blocked_reason or _dangerous_lookup_from_expr(value)
            if blocked_reason and blocked_direct_calls.get(target.id) != blocked_reason:
                blocked_direct_calls[target.id] = blocked_reason
                changed_alias = True

            string_value = _constant_string(value)
            if (
                string_value in {"eval", "exec", "__import__", "compile"}
                and dangerous_string_aliases.get(target.id) != string_value
            ):
                dangerous_string_aliases[target.id] = string_value
                changed_alias = True

            if _is_namespace_expr(value) and target.id not in namespace_aliases:
                namespace_aliases.add(target.id)
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

            dangerous_lookup = _dangerous_lookup_from_expr(node.func)
            if dangerous_lookup:
                return False, f"Call to dynamically retrieved dangerous builtin '{dangerous_lookup}'"

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
                # A dangerous-looking string is data unless this call retrieves
                # it from a dangerous module/namespace.  This avoids rejecting
                # normal column names, metric labels and parameter dictionaries.

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
        from backend.utils.live_paths import live_notebook_path

        return live_notebook_path(self.state_dir, self.session_id)
    
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


