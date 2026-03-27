"""Shared utilities for parsing Python error tracebacks."""
import re
from typing import FrozenSet

# Error types that indicate API misuse (worth looking up docs)
API_ERROR_TYPES: FrozenSet[str] = frozenset({
    "AttributeError", "TypeError", "ImportError", "ModuleNotFoundError",
})

# Regex to strip ANSI escape sequences (e.g. \x1b[31m) from Jupyter tracebacks
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


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
