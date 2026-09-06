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


#: Pickler refusals that are a property of the **value**, not of the machine:
#: the same computation retried produces the same refusal. Spec 4.4 makes these
#: coverage -- the value is skipped, recorded with a reason, and the generation
#: still commits.
#:
#: Everything else -- OSError, MemoryError, a lifecycle error -- is the machine
#: failing, where a retry could genuinely reach a different outcome. Recording
#: one of those as coverage commits the generation as ``current``, which clears
#: ``failed_variables`` and ``error_summary``, leaving the only trace in a
#: manifest field that has no reader.
STRUCTURAL_PICKLE_ERRORS: frozenset[str] = frozenset({
    "TypeError",
    "PicklingError",
    "NotImplementedError",
    "AttributeError",
})
