"""Measure replicate counts before anything plans against them.

Runs once, in round 1, between initialize_round and pi_planning. It has to be
before pi_planning because build_experimental_design_summary is what the PI
reads, and that string carrying a declared "3 sample(s)" is what put an
unsatisfiable acceptance condition into a plan step.

Failure here is not fatal. A verification pass that becomes a new single point
of failure repeats the defect 0.39.40 removed, where a check nothing could
satisfy sent one stage back through planning five times.
"""
import logging
from typing import TYPE_CHECKING, Any

from ..retry import with_retry
from ...constants import PROFILED_OBJECTS_KEY
from ...models.schemas import AgentType
from .modality_nodes import append_modality_sample_counts

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)



def should_profile(analysis_state: Any) -> bool:
    """Round 1 only, and only when no SEQUENCING record exists yet.

    The second clause is what makes a restore idempotent: a resumed session
    that already profiled sequencing must not pay for the pass twice. It has
    to check the sequencing section specifically, not "any record": pathology
    and biochemistry register their own counts into the same list. In graph
    order they run AFTER data_profiling, so this is not normally reachable
    with their records already present -- but a restore-and-resume can land
    between their pass and this one, with their records already in the
    restored state, and that resume must not read them as "sequencing
    already profiled" and skip this pass permanently.
    """
    if int(getattr(analysis_state, "current_round", 0) or 0) != 1:
        return False
    records = getattr(analysis_state, "detected_sample_counts", None) or []
    return not any(getattr(r, "section", None) == "sequencing" for r in records)


class ProfilingNodeMixin:
    """The data_profiling graph node. Composed into AgentNodesMixin."""

    @with_retry("data_profiling")
    async def _data_profiling(self, state: WorkflowState) -> WorkflowState:
        analysis_state = state["analysis_state"]

        if not should_profile(analysis_state):
            logger.info("[%s] data_profiling: skipped (round %s, %d existing record(s))",
                        self.session_id,
                        getattr(analysis_state, "current_round", None),
                        len(getattr(analysis_state, "detected_sample_counts", None) or []))
            return state

        await self._send_message(
            "status", "Inspecting uploaded data to determine replicate counts...",
            AgentType.BIOINFORMATICS,
        )

        try:
            result = await self.bioinformatics_agent.profile_data(analysis_state)
        except Exception as exc:                          # noqa: BLE001
            # This catch runs before @with_retry's own except clauses ever see
            # this call, so its retry loop never fires here -- deliberate,
            # since this pass must never be able to end a session. profile_data
            # already absorbs its own transient failures by returning rather
            # than raising, so this branch is for a genuine defect, not a
            # transient LLM error.
            logger.warning("[%s] data_profiling failed, continuing without "
                           "detected counts: %s", self.session_id, exc)
            await self._send_message(
                "status",
                "Replicate-count detection failed; continuing with the declared "
                "sample counts.",
                AgentType.BIOINFORMATICS,
            )
            return state

        records = list(result.get("records") or [])
        # Section-scoped, like the other two producers (see
        # append_modality_sample_counts). A plain overwrite here was dormant
        # while should_profile treated any existing record as "already
        # profiled" -- nothing could reach this line with a non-empty list.
        # Narrowing that guard to sequencing made a restore path reachable
        # where pathology/biochemistry records already exist on the state
        # before this pass runs, and neither section re-registers its own:
        # both early-return on a cache hit before ever calling
        # append_modality_sample_counts. A wholesale overwrite here would
        # silently drop them for the round.
        append_modality_sample_counts(analysis_state, "sequencing", records)

        sandbox_state = analysis_state.sandbox_state or {}
        sandbox_state[PROFILED_OBJECTS_KEY] = list(result.get("objects") or [])
        analysis_state.sandbox_state = sandbox_state

        # profile_data's own contract: it never raises for a modelling or
        # execution problem, it returns this shape instead. That must not be
        # silently indistinguishable from "nothing to profile" -- whatever
        # records it did register above are kept, not discarded, but the
        # degraded run still needs to be logged and surfaced to the user.
        if result.get("status") != "completed":
            logger.warning("[%s] data_profiling: pass returned status=%r, "
                           "keeping %d record(s): %s",
                           self.session_id, result.get("status"), len(records),
                           result.get("summary") or "")
            await self._send_message(
                "status",
                "Replicate-count detection did not complete; continuing with "
                "the declared sample counts.",
                AgentType.BIOINFORMATICS,
            )
            return state

        conflicts = [r for r in records if getattr(r, "conflicts", False)]
        logger.info("[%s] data_profiling: %d record(s), %d conflict(s), objects=%s",
                    self.session_id, len(records), len(conflicts),
                    sandbox_state[PROFILED_OBJECTS_KEY])

        if conflicts:
            detail = "; ".join(
                f"{r.group_name}/{r.data_type}: detected {r.detected_count}, "
                f"declared {r.declared_count}" for r in conflicts[:6]
            )
            await self._send_message(
                "status",
                f"Detected replicate counts differ from the declared design "
                f"({detail}). The detected counts will be used.",
                AgentType.BIOINFORMATICS,
            )

        return state
