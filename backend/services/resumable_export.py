"""Freeze, flush, verify — what ``include_runtime=true`` has to mean. Spec 5.4.

An export that packs whatever pkl happens to be on disk ships a runtime that
can be many cells old, beside a state file and a checkpoint captured later. The
archive then imports cleanly and cannot continue: ``import_state`` compares the
archived pairing against what the archive actually carries and retires both the
runtime and the checkpoint when they disagree, so "the import succeeded" is not
evidence of anything.

Three files have to describe one moment: the state, the runtime dump and the
checkpoint copy. This module produces that moment, or refuses.

**Refusing is the design.** Falling back to the previous committed generation
would ship a *new* state beside an *old* runtime, which is exactly the
incoherence spec 5.1 names — and this design does not retain the matching older
state and checkpoint to pair the old runtime with.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResumableExportUnavailable(RuntimeError):
    """The export cannot produce a resumable archive right now.

    Carries a reason the caller can render. Never a 500: the session is intact,
    the previous generation is intact, and a results-only export still works.
    """

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


def live_sandbox(session_id: str) -> Optional[Any]:
    """The sandbox currently owning *session_id*'s kernel, if any.

    A workflow's own sandbox first: it is the one executing cells. The chat
    sandbox is the fallback for a session whose analysis has finished but whose
    kernel is still up for chat-driven figures.
    """
    from backend.state import chat_sandboxes, workflows

    workflow = workflows.get(session_id)
    sandbox = getattr(workflow, "sandbox", None)
    if sandbox is not None and getattr(sandbox, "km", None) is not None:
        return sandbox
    sandbox = chat_sandboxes.get(session_id)
    if sandbox is not None and getattr(sandbox, "km", None) is not None:
        return sandbox
    return None


def _pairing(session_id: str, state) -> Optional[dict]:
    from backend.utils.runtime_pairing import capture_pairing

    try:
        return capture_pairing(session_id, state)
    except Exception:
        logger.warning(
            "[%s] Could not capture the export pairing", session_id, exc_info=True,
        )
        return None


async def flush_for_resumable_export(session_id: str, state) -> Optional[dict]:
    """Flush the live kernel and return the pairing the archive must record.

    Returns ``None`` when there is no live kernel: an archive built from a
    session whose kernel is already down carries the last committed generation,
    which is coherent with the state on disk beside it. There is nothing to
    flush and nothing to refuse.

    Raises :class:`ResumableExportUnavailable` when a kernel is live and its
    generation could not be committed.
    """
    sandbox = live_sandbox(session_id)
    if sandbox is None:
        return _pairing(session_id, state)

    try:
        committed = await sandbox.save_session()
    except Exception as exc:
        raise ResumableExportUnavailable(
            "flush_failed",
            "The runtime could not be saved, so a resumable archive would "
            "contain a kernel older than the session state beside it. Export "
            "without runtime to keep the results, or retry once the session "
            "is idle.",
        ) from exc
    if committed is not True:
        raise ResumableExportUnavailable(
            "flush_failed",
            "The runtime could not be saved, so a resumable archive would "
            "contain a kernel older than the session state beside it. Export "
            "without runtime to keep the results, or retry once the session "
            "is idle.",
        )

    verify_committed_generation(session_id)
    return _pairing(session_id, state)


def verify_committed_generation(session_id: str) -> None:
    """Refuse an archive whose committed generation does not verify. Spec 4.4.

    A dump that fails its own HMAC is not a generation; shipping it produces an
    archive that imports and then fails to rehydrate on the destination, where
    there is nothing left to fall back to.
    """
    from backend.utils.runtime_apply import RuntimeApplyError, verify_live_pkl_or_raise

    try:
        verify_live_pkl_or_raise(session_id)
    except RuntimeApplyError as exc:
        raise ResumableExportUnavailable(
            "generation_unverified",
            "The saved runtime failed its integrity check, so it cannot be "
            "packed into a portable archive. Export without runtime to keep "
            "the results.",
        ) from exc


def assert_boundary_held(session_id: str, state, before: Optional[dict]) -> None:
    """Refuse an archive assembled across a moving execution boundary.

    The pairing is ``(state_version, checkpoint_identity)``. If either moved
    between the flush and the end of the capture, the three files no longer
    describe one moment and the destination's own check would retire the
    runtime -- silently, after reporting a successful import. Refusing here is
    the same verdict, delivered where the user can act on it.
    """
    after = _pairing(session_id, state)
    if before == after:
        return
    raise ResumableExportUnavailable(
        "boundary_moved",
        "The analysis advanced while the archive was being built, so its "
        "runtime, state and checkpoint would not describe the same moment. "
        "Retry the export once the session is idle.",
    )
