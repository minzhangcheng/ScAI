"""Shared utilities for parsing Python error tracebacks."""
from typing import FrozenSet

# Error types that indicate API misuse (worth looking up docs)
API_ERROR_TYPES: FrozenSet[str] = frozenset({
    "AttributeError", "TypeError", "ImportError", "ModuleNotFoundError",
})


def extract_error_type(error_text: str) -> str:
    """Extract the error type (e.g. 'AttributeError') from a traceback string."""
    for line in reversed(error_text.strip().split('\n')):
        line = line.strip()
        if line and ':' in line and not line.startswith(('File ', 'Traceback')):
            err_type = line.split(':')[0].strip()
            if '.' in err_type:
                err_type = err_type.rsplit('.', 1)[-1]
            return err_type
    return ""
