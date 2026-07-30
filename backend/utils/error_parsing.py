"""Shared utilities for parsing Python error tracebacks."""
import re
from typing import FrozenSet, Optional

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
# Covers CSI (any final byte), OSC (BEL-terminated), and charset designations.
_ANSI_RE = re.compile(
    r'\x1b\[[0-9;]*[a-zA-Z]'   # CSI ... final letter
    r'|\x1b\][^\x07]*\x07'      # OSC ... BEL
    r'|\x1b[()][AB012]'         # charset designations
)


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
