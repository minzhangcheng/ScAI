"""Code-safety validation and protocol helpers for the sandbox facade."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Tuple

def compute_remaining_timeout(deadline: float, now: float) -> float:
    """Seconds left until the wall-clock deadline; <= 0 means expired."""
    return deadline - now


async def _to_thread_drained(func, /, *args, **kwargs):
    """Run a blocking resource call without orphaning its worker on cancel.

    The owned executor keeps both cancellation and loop shutdown from orphaning
    a worker that still touches the same socket, client, or process manager.
    """
    from backend.utils.asyncio_tasks import run_in_owned_thread

    return await run_in_owned_thread(func, *args, **kwargs)


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
    (r'\bos\.posix_spawnp?\b', 'os.posix_spawn* calls'),
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
    (r'\bscai_kernel_trust\b', 'reserved kernel trust module'),

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
    (
        r'\bopen\s*\([^)]*,\s*["\'][^"\']*[wax+]',
        'file write operations outside safe directories',
    ),
    (
        r'\.open\s*\(\s*["\'][^"\']*[wax+]',
        'file write operations outside safe directories',
    ),
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
    """Block the subprocess surface, including import-only cross-cell aliases."""
    return False


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


def _literal_is_safe(value: str) -> bool:
    """Whether a literal path names an existing sandbox output area."""
    normalized = value.replace("\\", "/").lower()
    return (
        normalized.startswith("/tmp/scai-")
        or "/workspace/" in normalized
        or "/figures/" in normalized
        or "/tables/" in normalized
        or "/annotation_validation/" in normalized
        or normalized.startswith(
            (
                "figures/",
                "tables/",
                "annotation_validation/",
                "./figures/",
                "./tables/",
                "./annotation_validation/",
            )
        )
    )


def _literal_path(expr):
    """Return the literal path named by an AST expression, if any."""
    import ast

    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, (ast.Name, ast.Attribute))
        and getattr(expr.func, "id", getattr(expr.func, "attr", None)) == "Path"
        and expr.args
        and isinstance(expr.args[0], ast.Constant)
        and isinstance(expr.args[0].value, str)
    ):
        return expr.args[0].value
    return None


def _is_safe_file_removal(code: str, operation: str) -> bool:
    """Validate one direct ``os`` deletion API and its unique path target."""
    import ast
    import re

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    lines = code.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    def _offset(lineno: int, col: int) -> int:
        if lineno <= 0 or lineno > len(line_offsets):
            return -1
        return line_offsets[lineno - 1] + col

    direct_spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != operation:
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ):
            # Unrelated receivers (for example ``values.remove`` or
            # ``Path.unlink``) keep their prior treatment. A textual spelling
            # such as ``foo.os.remove`` is rejected by the span check below.
            continue

        start = _offset(node.func.lineno, node.func.col_offset)
        end = _offset(node.func.end_lineno, node.func.end_col_offset)
        if start < 0 or end < start:
            return False
        direct_spans.append((start, end))

        path_keywords = [keyword for keyword in node.keywords if keyword.arg == "path"]
        if (
            not isinstance(node.func.value, ast.Name)
            or node.func.value.id != "os"
            or len(path_keywords) > 1
            or len(node.args) > 1
            or (node.args and path_keywords)
            or any(keyword.arg is None for keyword in node.keywords)
        ):
            return False
        if node.args:
            target = node.args[0]
        elif path_keywords:
            target = path_keywords[0].value
        else:
            return False
        if isinstance(target, ast.Starred):
            return False
        literal = _literal_path(target)
        if literal is not None and not _literal_is_safe(literal):
            return False

    pattern = re.compile(rf"\bos\.{re.escape(operation)}\b")
    for match in pattern.finditer(code):
        if not any(start <= match.start() < end for start, end in direct_spans):
            return False
    return True


def _is_safe_os_remove(code: str) -> bool:
    return _is_safe_file_removal(code, "remove")


def _is_safe_os_unlink(code: str) -> bool:
    return _is_safe_file_removal(code, "unlink")


def _is_safe_os_rmdir(code: str) -> bool:
    return _is_safe_file_removal(code, "rmdir")


def _is_safe_file_write(code: str) -> bool:
    """Reject static file-open calls in write modes on unsafe literal paths.

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

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        path_arg = None
        positional_mode_index = None
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            if node.args:
                path_arg = node.args[0]
                positional_mode_index = 1
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Name)
                and receiver.id in {"builtins", "io"}
                and node.args
            ):
                path_arg = node.args[0]
                positional_mode_index = 1
            else:
                path_arg = receiver
                positional_mode_index = 0
        if path_arg is None:
            continue

        mode = None
        if (
            positional_mode_index is not None
            and len(node.args) > positional_mode_index
            and isinstance(node.args[positional_mode_index], ast.Constant)
        ):
            mode = node.args[positional_mode_index].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        if not (isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+"))):
            continue
        # Only literal paths can be ruled on statically; everything else is
        # resolved and gated by the process-level boundary at execution time.
        literal_path = _literal_path(path_arg)
        if literal_path is not None and not _literal_is_safe(literal_path):
            return False
    return True


# Maps pattern description keywords to their safety check functions
_EXCEPTION_CHECKS = {
    'os.system':   _is_safe_os_system,
    'subprocess':  _is_safe_subprocess,
    'importlib':   _is_safe_importlib,
    'file write':  _is_safe_file_write,
    'os.remove':   _is_safe_os_remove,
    'os.unlink':   _is_safe_os_unlink,
    'os.rmdir':    _is_safe_os_rmdir,
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
    'os', 'sys', 'subprocess', 'shutil', 'builtins', 'scai_kernel_trust',
})

_SUBPROCESS_EXEC_FUNCS = frozenset({
    "run",
    "call",
    "check_call",
    "check_output",
    "Popen",
})

_OS_PROCESS_EXEC_FUNCS = frozenset({
    "system",
    "popen",
    "fork",
    "forkpty",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "posix_spawn",
    "posix_spawnp",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
})


# Namespace members that hand back live objects rather than their names.
# ``keys()`` and bare iteration yield strings, so they stay available for the
# "what exists right now" question analysis code legitimately asks.
_NAMESPACE_VALUE_ACCESSORS = frozenset({
    "items",
    "values",
    "copy",
})


_CALL_SITE_PATTERN_KEYS = frozenset({
    "eval() function",
    "exec() function",
    "__import__ function",
    "compile() function",
})


def _mask_prose(code: str) -> str:
    """Blank comments and string literals without changing source offsets."""
    import ast
    import io
    import tokenize

    try:
        ast.parse(code)
    except SyntaxError:
        return code

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return code

    lines = [list(line) for line in code.splitlines(keepends=True)]
    for token in tokens:
        if token.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            index = row - 1
            if not 0 <= index < len(lines):
                continue
            line = lines[index]
            lo = start_col if row == start_row else 0
            hi = end_col if row == end_row else len(line)
            for col in range(lo, min(hi, len(line))):
                if line[col] not in "\r\n":
                    line[col] = " "
    return "".join("".join(line) for line in lines)


def _match_context(code: str, match: "re.Match[str]", window: int = 90) -> str:
    """Return the original-source line and nearby context for a match."""
    start = max(0, match.start() - window)
    end = min(len(code), match.end() + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(code) else ""
    context = code[start:end].replace("\n", " ")
    line = code.count("\n", 0, match.start()) + 1
    return f"line {line}: {prefix}{context}{suffix}"


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
                        return False, (
                            "Import of blocked subprocess function "
                            f"'subprocess.{alias.name}'"
                        )
            elif node.module == "os":
                for alias in node.names:
                    if alias.name in _OS_PROCESS_EXEC_FUNCS:
                        return False, (
                            "Import of blocked process function "
                            f"'os.{alias.name}'"
                        )

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
            if module_name == "os" and expr.attr in _OS_PROCESS_EXEC_FUNCS:
                return f"os.{expr.attr}"
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

    def _captures_namespace_values(expr) -> Optional[str]:
        """Name the syntax that materializes namespace *values*, else None."""
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            if (
                expr.func.attr in _NAMESPACE_VALUE_ACCESSORS
                and _is_namespace_expr(expr.func.value)
            ):
                return f"globals().{expr.func.attr}()"
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "dict"
            and any(_is_namespace_expr(arg) for arg in expr.args)
        ):
            return "dict(globals())"
        if isinstance(expr, ast.Dict) and any(
            key is None and _is_namespace_expr(value)
            for key, value in zip(expr.keys, expr.values)
        ):
            return "{**globals()}"
        return None

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
        if isinstance(node, ast.Name) and node.id.startswith("_scai_"):
            return False, "Access to reserved ScAI kernel namespace"

        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"eval", "exec", "__import__", "compile"}
            ):
                return False, f"Call to blocked builtin '{node.func.id}'"
            if isinstance(node.func, ast.Name) and node.func.id in blocked_direct_calls:
                return False, f"Call to blocked dependency/system function '{blocked_direct_calls[node.func.id]}'"

            dangerous_lookup = _dangerous_lookup_from_expr(node.func)
            if dangerous_lookup:
                return False, f"Call to dynamically retrieved dangerous builtin '{dangerous_lookup}'"

            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name = module_aliases.get(node.func.value.id)
                if module_name == "subprocess" and node.func.attr in _SUBPROCESS_EXEC_FUNCS:
                    return False, f"Call to blocked subprocess function 'subprocess.{node.func.attr}'"
                if module_name == "os" and node.func.attr in _OS_PROCESS_EXEC_FUNCS:
                    return False, f"Call to blocked os.{node.func.attr}"

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

        if isinstance(node, ast.Subscript) and _is_namespace_expr(node.value):
            key = _subscript_key(node)
            if isinstance(key, str) and key.startswith("_scai_"):
                return False, "Access to reserved ScAI kernel namespace"

        # Binding the whole namespace to one name -- the observed form was
        # `namespace_items = list(globals().items())` -- puts every live object
        # behind a single user global. The runtime dependency tracker then
        # promotes that name into its required set, so one unpicklable value
        # anywhere in the namespace (an sqlite3 connection, an open handle)
        # fails every later save on a name the analysis never meant to keep.
        # Only *binding* is rejected: iterating or printing captures nothing.
        bound_value = None
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.NamedExpr)):
            bound_value = node.value
        elif isinstance(node, ast.AnnAssign):
            bound_value = node.value
        if bound_value is not None:
            for bound_node in ast.walk(bound_value):
                captured = _captures_namespace_values(bound_node)
                if captured:
                    return False, (
                        "Bulk capture of the kernel namespace via "
                        f"{captured}. Bind the variables you need by name; "
                        "globals().keys() lists what exists."
                    )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "setdefault"}
            and _is_namespace_expr(node.func.value)
        ):
            candidates = list(node.args)
            candidates.extend(keyword.value for keyword in node.keywords)
            if node.func.attr == "setdefault" and node.args:
                key = _constant_string(node.args[0])
                if isinstance(key, str) and key.startswith("_scai_"):
                    return False, "Access to reserved ScAI kernel namespace"
            for candidate in candidates:
                if not isinstance(candidate, ast.Dict):
                    continue
                for key_node in candidate.keys:
                    key = _constant_string(key_node)
                    if isinstance(key, str) and key.startswith("_scai_"):
                        return False, "Access to reserved ScAI kernel namespace"

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
    masked: Optional[str] = None
    for pattern, description in DANGEROUS_PATTERNS:
        match = re.search(pattern, code)
        if match:
            if description in _CALL_SITE_PATTERN_KEYS:
                if masked is None:
                    masked = _mask_prose(code)
                match = re.search(pattern, masked)
                if match is None:
                    continue
            # Check if the matched pattern has a known safe-usage exception
            if any(key in description and check(code) for key, check in _EXCEPTION_CHECKS.items()):
                continue
            logger.warning(
                "Security: blocked dangerous code pattern [%s] at offset %d — %s",
                description,
                match.start(),
                _match_context(code, match),
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
