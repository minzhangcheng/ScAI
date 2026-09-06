"""Runtime value kinds the persistence layer excludes by type.

Spec 4.2 keeps volatile-type exclusion, so this tuple outlives Snapshot V2 and
lives here rather than in that package. Its contents and order are behaviour,
not style: the generated save script embeds
``repr(VOLATILE_RUNTIME_TYPE_PREFIXES)`` directly into kernel code.
"""
from __future__ import annotations

# Handles and connections are the dangerous case: dill often "succeeds" on them
# by recording a path or fd, then reopens it on restore, so a pickling probe
# cannot see the problem and a later restore on another machine aborts. They are
# excluded by type both when the save runs and when a name is considered for
# promotion, so the two decisions cannot drift apart.
VOLATILE_RUNTIME_TYPE_PREFIXES = (
    "_io.", "io.", "socket.", "_socket.", "ssl.",
    "_thread.", "threading.", "multiprocessing.", "concurrent.", "asyncio.",
    "sqlite3.", "h5py.", "tables.", "loompy.", "zarr.",
    "subprocess.", "tempfile.",
)
