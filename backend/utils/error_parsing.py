"""Shared utilities for parsing Python error tracebacks."""
import re
from collections.abc import Mapping
from os import PathLike
from typing import Any, FrozenSet, Optional

from .secret_redaction import redact_secrets

# Error types that indicate API misuse (worth looking up docs)
API_ERROR_TYPES: FrozenSet[str] = frozenset({
    "AttributeError", "TypeError", "ImportError", "ModuleNotFoundError",
})

# Error types caused by LLM code-generation bugs (static hint, no API doc lookup)
CODE_GEN_ERROR_TYPES: FrozenSet[str] = frozenset({
    "SyntaxError", "IndentationError", "TabError",
})

# Error types that indicate purely technical failures (no PI guidance needed)
TECHNICAL_ERROR_TYPES: FrozenSet[str] = frozenset({
    "KeyError", "TypeError", "IndexError", "ValueError",
    "ModuleNotFoundError", "AttributeError", "NameError",
    "SyntaxError", "ImportError", "FileNotFoundError",
    "IndentationError", "TabError",
    "ZeroDivisionError", "RuntimeError", "OSError",
    "IOError", "PermissionError", "StopIteration",
    "UnicodeDecodeError", "UnicodeEncodeError",
    "RecursionError", "OverflowError", "NotImplementedError",
    "UnboundLocalError",
    # Host-side objective tool-contract/resource-ceiling failures.
    "InvalidToolArgumentsError", "RepeatedExecutionError",
    # Library-specific technical errors
    "InvalidIndexError", "AnnDataReadError", "SparseDataError",
    # Network and transport failures. Absent from this set, a failed remote call
    # is classified as nothing at all: it reaches neither the documentation
    # lookup nor the swallowed-error detector, so the round records a capability
    # as unavailable on the strength of one transient response.
    "ConnectionError", "ConnectionResetError", "ConnectionRefusedError",
    "ConnectionAbortedError", "TimeoutError", "SSLError", "HTTPError",
    "URLError", "RemoteDisconnected", "IncompleteRead", "ChunkedEncodingError",
    "ReadTimeout", "ConnectTimeout", "RequestException", "ProtocolError",
    "JSONDecodeError",
})

# Regex to strip ANSI escape sequences from Jupyter tracebacks.
# Covers CSI (all ECMA-48 parameter/intermediate bytes), OSC terminated by
# either BEL or ST, and charset designations.
_ANSI_RE = re.compile(
    r'\x1b\[[0-?]*[ -/]*[@-~]'  # CSI ... final byte
    r'|\x1b\](?:[^\x07\x1b]|\x1b(?!\\))*(?:\x07|\x1b\\)'  # OSC ... BEL/ST
    r'|\x1b[()][AB012]'         # charset designations
)

_SECRET_VALUE = "<redacted-secret>"
_CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _is_secret_mapping_key(key: Any) -> bool:
    """Return whether a mapping key semantically names a credential value."""
    if not isinstance(key, str):
        return False
    separated = _CAMEL_CASE_BOUNDARY_RE.sub("_", key.strip())
    tokens = [
        token
        for token in re.sub(r"[^A-Za-z0-9]+", "_", separated).lower().split("_")
        if token
    ]
    if not tokens:
        return False
    compact = "".join(tokens)
    if compact.endswith("apikey"):
        return True
    return tokens[-1] in {"token", "password", "passwd", "secret"}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub('', text)


def extract_error_type(error_text: str) -> str:
    """Extract the error type (e.g. 'AttributeError') from a traceback string."""
    error_text = strip_ansi(error_text)
    for line in reversed(error_text.strip().split('\n')):
        line = line.strip()
        if line and ':' in line and not line.startswith(('File ', 'Traceback')):
            err_type = line.split(':')[0].strip()
            if '.' in err_type:
                err_type = err_type.rsplit('.', 1)[-1]
            # Basic validation: error types are single words starting with uppercase
            if err_type and ' ' not in err_type and err_type[0].isupper():
                return err_type
    return ""


_EXCEPTION_LINE_RE = re.compile(
    r"^(?:[\w.]+\.)?[A-Z][A-Za-z0-9_]*(?:Error|Exception|Interrupt|Warning)\b(?:\s*:|$)"
)
_CHAIN_LINE_RE = re.compile(
    r"^(?:During handling of the above exception|The above exception was the direct cause)",
    re.IGNORECASE,
)
_PRIVATE_HOME_PATH_RE = re.compile(
    r"/(?:home|Users)/[^\s\"'<>:,)\]}]+"
    r"|[A-Za-z]:\\Users\\[^\s\"'<>:,)\]}]+"
)
_WORKSPACE_PATH_RE = re.compile(r"/workspace/[^\s\"'<>:,)\]}]+")
_SESSION_PATH_RE = re.compile(
    r"/(?:data/)?sessions/[^\s\"'<>:,)\]}]+"
)
_LOCAL_PRIVATE_PATH_RE = re.compile(
    r"/(?:root|tmp|private|mnt|opt|var/tmp|data)/[^\s\"'<>:,)\]}]+"
)
_USER_FRAME_RE = re.compile(
    r"(?:Cell In\[|<ipython-input|<string>|/workspace/|/sessions/|"
    r"<workspace-path>|<session-path>|<private-path>)"
)

DIAGNOSTIC_TRUNCATION_MARKER = "[... diagnostic truncated ...]"


def _redact_private_paths(text: str) -> str:
    """Remove host-, workspace-, and session-specific absolute paths."""
    rendered = _SESSION_PATH_RE.sub("<session-path>", text)
    rendered = _WORKSPACE_PATH_RE.sub("<workspace-path>", rendered)
    rendered = _PRIVATE_HOME_PATH_RE.sub("<private-path>", rendered)
    return _LOCAL_PRIVATE_PATH_RE.sub("<private-path>", rendered)


def compact_traceback(error_text: str, max_chars: int = 3000) -> str:
    """Return a bounded, redacted traceback that retains its real exception.

    Jupyter tracebacks can contain thousands of characters of renderer and
    cleanup frames before the final exception. This representation keeps a
    small header, user-code frames, exception-chain separators, and the final
    traceback tail. Recognizable credentials are redacted before any slice is
    persisted or sent to another model.
    """
    if not error_text:
        return ""
    limit = max(256, int(max_chars))
    cleaned = redact_secrets(
        _redact_private_paths(strip_ansi(str(error_text)))
    ).replace("\r\n", "\n")
    lines = cleaned.splitlines()
    if not lines:
        return ""

    exception_indexes = [
        index for index, line in enumerate(lines)
        if _EXCEPTION_LINE_RE.match(line.strip())
    ]
    has_prior_exception = len(exception_indexes) > 1

    selected: set[int] = set(range(min(5, len(lines))))
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _CHAIN_LINE_RE.match(stripped) or _EXCEPTION_LINE_RE.match(stripped):
            selected.add(index)
        if _USER_FRAME_RE.search(line):
            selected.update(range(max(0, index - 1), min(len(lines), index + 3)))

    tail_start = max(0, len(lines) - 14)
    selected.update(range(tail_start, len(lines)))

    rendered_lines: list[str] = []
    # ``-1`` means index zero is contiguous with the start. The old ``-2``
    # inserted an "omitted" marker before every one-line message and even
    # before dictionary keys when the recursive sanitizer reused this helper.
    previous = -1
    for index in sorted(selected):
        line = lines[index]
        if has_prior_exception and "_scai_cleanup_plt" in line:
            continue
        if index > previous + 1:
            rendered_lines.append("[... traceback frames omitted ...]")
        rendered_lines.append(line)
        previous = index
    rendered = "\n".join(rendered_lines).strip()
    if len(rendered) <= limit:
        return rendered

    # Preserve the final exception and chain tail even under a very small
    # consumer budget. The leading slice remains useful for execution context.
    marker = "\n[... compact traceback truncated ...]\n"
    tail_budget = min(max(limit // 2, 160), limit - len(marker) - 80)
    head_budget = max(40, limit - len(marker) - tail_budget)
    return rendered[:head_budget].rstrip() + marker + rendered[-tail_budget:].lstrip()


def bounded_diagnostic(value: Any, limit: int = 500) -> str:
    """Redact and compact a diagnostic before applying a hard character budget.

    Unlike ordinary identifier fields, diagnostics need their terminal
    exception to remain actionable when a consumer persists only a small
    summary.  Compact first so the final traceback tail is available, then
    retain both context and that tail when the consumer budget requires a cut.
    """
    try:
        budget = int(limit)
    except (TypeError, ValueError):
        return ""
    if budget <= 0 or value is None:
        return ""
    raw = str(value)
    if not raw:
        return ""

    compacted = " ".join(
        compact_traceback(raw, max_chars=max(3000, budget * 8)).split()
    )
    if len(compacted) <= budget:
        return compacted
    if budget <= len(DIAGNOSTIC_TRUNCATION_MARKER):
        return DIAGNOSTIC_TRUNCATION_MARKER[:budget]

    remaining = budget - len(DIAGNOSTIC_TRUNCATION_MARKER)
    head_budget = remaining // 2
    tail_budget = remaining - head_budget
    return (
        compacted[:head_budget].rstrip()
        + DIAGNOSTIC_TRUNCATION_MARKER
        + compacted[-tail_budget:].lstrip()
    )


def sanitize_error_payload(value: Any, max_chars: int = 3000) -> Any:
    """Recursively redact and bound every string in an error payload.

    Snapshot errors cross several persistence and transport boundaries as
    nested dictionaries. Sanitizing only the top-level message leaves raw
    tracebacks, credentials, and private paths in ``details``. This helper is
    deliberately shape-preserving for JSON-like values and cycle-safe for
    defensive use at final serialization boundaries.
    """
    seen: set[int] = set()

    def sanitize(item: Any, depth: int) -> Any:
        if isinstance(item, str):
            return compact_traceback(item, max_chars=max_chars)
        if isinstance(item, (bytes, bytearray, memoryview, PathLike, Exception)):
            return compact_traceback(str(item), max_chars=max_chars)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if depth >= 16:
            return "<error-payload-depth-limit>"

        identity = id(item)
        if identity in seen:
            return "<recursive-error-payload>"

        if isinstance(item, Mapping):
            seen.add(identity)
            try:
                sanitized_mapping = {}
                for key, nested in item.items():
                    safe_key = (
                        compact_traceback(key, max_chars=256)
                        if isinstance(key, str)
                        else key
                    )
                    sanitized_mapping[safe_key] = (
                        _SECRET_VALUE
                        if _is_secret_mapping_key(key)
                        else sanitize(nested, depth + 1)
                    )
                return sanitized_mapping
            finally:
                seen.remove(identity)
        if isinstance(item, list):
            seen.add(identity)
            try:
                return [sanitize(nested, depth + 1) for nested in item]
            finally:
                seen.remove(identity)
        if isinstance(item, tuple):
            seen.add(identity)
            try:
                return tuple(sanitize(nested, depth + 1) for nested in item)
            finally:
                seen.remove(identity)
        if isinstance(item, (set, frozenset)):
            seen.add(identity)
            try:
                return [sanitize(nested, depth + 1) for nested in item]
            finally:
                seen.remove(identity)
        return item

    return sanitize(value, 0)


_SKIP_DECISION_RE = re.compile(
    r"skip[-_ ](?:velocity|trajectory|analysis|step)"
    r"|['\"]decision['\"]\s*:\s*['\"]skip"
    r"|\bskipping\b",
    re.IGNORECASE,
)


def detect_swallowed_error(output) -> Optional[str]:
    """Detect a technical error that agent code caught (printed) and then chose
    to skip, instead of letting it surface as a hard execution error.

    Returns the technical error type (e.g. 'AttributeError') when BOTH hold:
      1. the output contains an abandon/skip decision keyword, AND
      2. a known TECHNICAL_ERROR_TYPES token appears in repr ('Foo(') or
         str ('Foo:') form.
    Otherwise returns None. Requiring (1) keeps legit data-driven skips and
    transient errors the agent later fixes from triggering a false positive.
    """
    if not output:
        return None
    text = strip_ansi(output)
    if not _SKIP_DECISION_RE.search(text):
        return None
    best_idx = None
    best_type = None
    for etype in TECHNICAL_ERROR_TYPES:
        m = re.search(rf"\b{re.escape(etype)}\s*[(:]", text)
        if m and (best_idx is None or m.start() < best_idx):
            best_idx = m.start()
            best_type = etype
    return best_type
