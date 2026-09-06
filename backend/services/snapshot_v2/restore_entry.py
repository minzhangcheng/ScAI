"""Deprecated location. Snapshot V2 is deleted in A2-5; import from
``backend.workflow.restore_entry`` instead."""

from backend.workflow.restore_entry import (  # noqa: F401
    RestoreEntryPlan,
    build_restore_entry_plan,
)
