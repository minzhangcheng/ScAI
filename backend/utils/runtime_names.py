"""Validated runtime variable names, and the kernel's own name rules."""

from __future__ import annotations

import re

_RUNTIME_GLOBAL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def validate_runtime_global_name(name: str) -> str:
    """Return a safe kernel global name or raise ValueError.

    Runtime names are used both as Python globals and legacy h5ad sidecar
    filenames.  Keeping the accepted set to normal identifiers avoids path
    traversal, hidden files, ambiguous stems, and names the sandbox cannot
    assign back into globals safely.
    """
    if not isinstance(name, str):
        raise ValueError("runtime variable name must be a string")
    if not _RUNTIME_GLOBAL_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid runtime variable name: {name!r}")
    return name


#: Names IPython puts in the user namespace itself. Excluded from the private
#: coverage record because that record is for names a user might go looking
#: for, and ``_i7`` is not one of them.
SHELL_HISTORY_NAMES = frozenset({
    "_", "__", "___", "_i", "_ii", "_iii", "_ih", "_oh", "_dh", "_sh",
    "_exit_code",
})

#: Kernel source, spliced into the save script. It lives here rather than in
#: the script literal for one reason: ``runtime_save.py`` sits exactly on the
#: 1,000-line file limit, and several source-text guards pin the script's
#: existing text to that file, so it can neither grow nor be split. New script
#: text therefore goes where it can be read on its own -- which is also where
#: the other name rules already are.
#:
#: What it computes: ``_scai_snapshot``, the namespace as it stood before the
#: save script defined anything of its own, and ``_excluded_private``, the
#: user globals dropped for having a private name. Spec 4.4 -- a name absent
#: from the dump must not also be absent from every record of the dump, and
#: this exclusion used to report through neither channel.
PRIVATE_EXCLUSION_SCRIPT = """_scai_shell_names = frozenset(""" + repr(
    sorted(SHELL_HISTORY_NAMES)
) + """)


def _scai_is_shell_noise(_scai_n):
    if _scai_n.startswith('__') or _scai_n.startswith('_scai_'):
        return True
    if _scai_n in _scai_shell_names:
        return True
    _scai_r = _scai_n[1:]
    return _scai_r.isdigit() or (_scai_r[:1] == 'i' and _scai_r[1:].isdigit())


# Taken before `_live` and every later name exists, so the script cannot report
# itself as a user's private value.
_scai_snapshot = list(globals().keys())
_excluded_private = [
    {"name": _scai_n, "type": type(globals()[_scai_n]).__name__}
    for _scai_n in _scai_snapshot
    if _scai_n.startswith('_') and not _scai_is_shell_noise(_scai_n)
]
"""
