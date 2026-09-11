"""Expression provenance and artifact invalidation bookkeeping."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .ast import mutates_expression_representation
from .constants import _MARKER


def strip_expression_contract_markers(output: str | None) -> str | None:
    """Remove trusted provenance marker lines from user-visible output."""
    if output is None:
        return None
    return "\n".join(
        line for line in output.splitlines() if not line.startswith(_MARKER)
    )


def parse_expression_contracts(output: str | None) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for line in (output or "").splitlines():
        if line.startswith(_MARKER):
            try:
                value = json.loads(line[len(_MARKER):])
                if isinstance(value, list):
                    contracts.extend(
                        item for item in value if isinstance(item, dict)
                    )
            except (TypeError, json.JSONDecodeError):
                continue
    return contracts


def record_expression_contract(state: Any, contracts: list[dict[str, Any]]) -> None:
    # Contract markers are emitted into the kernel stream before the agent can
    # strip them from its tool response, so their payload is also present in a
    # successfully persisted notebook cell.  Keep only provenance metadata;
    # raw extrema (emitted by the first implementation) are neither required
    # for later policy decisions nor appropriate for a durable execution log.
    safe_contracts = [
        {
            key: contract.get(key)
            for key in ("name", "layer", "shape", "scale", "dtype_class")
            if key in contract
        }
        for contract in contracts
        if isinstance(contract, dict)
    ]
    if not safe_contracts:
        return
    sandbox_state = state.sandbox_state or {}
    sandbox_state["expression_contract_v1"] = {
        "round_number": int(state.current_round or 0),
        "analysis_generation": int(sandbox_state.get("analysis_generation", 0) or 0),
        "inputs": safe_contracts,
    }
    state.sandbox_state = sandbox_state


_REPRESENTATION_WARNING = (
    "Warning: the canonical expression representation changed after this artifact "
    "was generated; the artifact remains included and should be interpreted against "
    "its recorded analysis generation."
)


def _record_detected_representation_change(state: Any) -> None:
    """Advance provenance without changing artifact report/gallery eligibility."""
    sandbox_state = state.sandbox_state or {}
    generation = int(sandbox_state.get("analysis_generation", 0) or 0) + 1
    sandbox_state["analysis_generation"] = generation
    sandbox_state.pop("expression_contract_v1", None)
    state.sandbox_state = sandbox_state
    current_round = int(getattr(state, "current_round", 0) or 0)
    current_lineage = getattr(state, "restore_branch_id", None) or "main"
    for collection_name in ("publication_figures", "publication_tables"):
        for artifact in getattr(state, collection_name, None) or []:
            if int(getattr(artifact, "round_number", 0) or 0) not in (0, current_round):
                continue
            if (getattr(artifact, "lineage_id", None) or "main") != current_lineage:
                continue
            provenance = dict(getattr(artifact, "expression_provenance", None) or {})
            provenance["representation_change_warning"] = _REPRESENTATION_WARNING
            artifact.expression_provenance = provenance
            description = str(getattr(artifact, "description", "") or "")
            if _REPRESENTATION_WARNING not in description:
                artifact.description = (description + "\n\n" + _REPRESENTATION_WARNING).strip()


def record_representation_change(state: Any, code: str) -> int:
    """Bump provenance while retaining all existing artifacts."""
    if not mutates_expression_representation(code):
        return 0
    _record_detected_representation_change(state)
    return 0


def record_executed_representation_change(
    state: Any,
    code: str,
    working_dir: str | Path,
    before: dict[str, int],
    *,
    execution_started: bool,
    representation_mutated: bool | None = None,
) -> tuple[int, set[str], str]:
    """Record representation drift after a dispatched, non-transactional cell.

    A Jupyter cell can mutate ``adata.X`` and then fail.  A failed result is
    therefore not evidence that the canonical representation stayed intact.
    Provenance generation still advances, but previously generated artifacts
    remain eligible for report and gallery use with an attached warning.
    """
    detected = (
        mutates_expression_representation(code)
        if representation_mutated is None else representation_mutated
    )
    if not execution_started or not detected:
        return 0, set(), ""
    if representation_mutated is None:
        record_representation_change(state, code)
    else:
        _record_detected_representation_change(state)
    return (
        0,
        set(),
        "[Provenance] The expression representation changed. Existing figures "
        "and tables remain eligible and carry an analysis-generation warning.",
    )


def workspace_artifact_versions(working_dir: str | Path) -> dict[str, int]:
    """Snapshot derived-file mtimes before a representation change."""
    root = Path(working_dir)
    versions: dict[str, int] = {}
    for directory in (root, root / "figures", root / "tables"):
        if not directory.exists():
            continue
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.csv"):
            for path in directory.glob(suffix):
                versions[path.stem] = max(
                    versions.get(path.stem, 0), path.stat().st_mtime_ns
                )
    return versions


#: Where ``record_round_start`` parks the stamp. Lives in ``sandbox_state``
#: rather than on a dedicated field so the mechanism needs no schema change and
#: travels with the rest of this module's provenance record.
ROUND_START_KEY = "round_started_ns_v1"


def record_round_start(state: Any) -> int:
    """Stamp the wall-clock start of the round in progress and return it.

    Idempotent per round: both stages of a round re-enter the node that calls
    this, and a stage-2 representation change invalidates stage-1 artifacts too,
    so the round keeps the stamp it was given when it opened.
    """
    sandbox_state = state.sandbox_state or {}
    current = int(getattr(state, "current_round", 0) or 0)
    stamp = sandbox_state.get(ROUND_START_KEY)
    if isinstance(stamp, dict) and int(stamp.get("round", -1) or -1) == current:
        started = int(stamp.get("started_ns", 0) or 0)
        if started > 0:
            return started
    started = time.time_ns()
    sandbox_state[ROUND_START_KEY] = {"round": current, "started_ns": started}
    state.sandbox_state = sandbox_state
    return started


def round_start_ns(state: Any) -> int:
    """Wall-clock start of the round in progress, in ns.

    Read from the stamp ``record_round_start`` writes when the round opens. The
    analysis log cannot answer this while the round is running and never could:
    the bioinformatics agent keeps its steps in a local list and they are merged
    into ``state.analysis_log`` -- and stamped with ``round_number`` -- only
    after the round's execution has finished. So on every main pass the
    current-round lookup found nothing and the function returned
    ``time.time_ns()``, a bound nothing predates, which made
    ``unchanged_artifact_stems`` return the empty set: no figure was ever
    withheld, while ``describe_invalidated_artifacts`` still told the agent that
    N of them had been.

    The remaining fallbacks are ordered so an unknown start includes artifacts
    rather than excluding them. Withholding too much is visible to the user and
    the agent can regenerate; withholding nothing is silent and ships a figure
    that no longer matches its data.
    """
    current = int(getattr(state, "current_round", 0) or 0)
    sandbox_state = getattr(state, "sandbox_state", None) or {}
    stamp = sandbox_state.get(ROUND_START_KEY) if isinstance(sandbox_state, dict) else None
    if isinstance(stamp, dict) and int(stamp.get("round", -1) or -1) == current:
        started = int(stamp.get("started_ns", 0) or 0)
        if started > 0:
            return started

    steps = list(getattr(state, "analysis_log", None) or [])
    starts = [
        step.timestamp
        for step in steps
        if int(getattr(step, "round_number", 0) or 0) == current
        and getattr(step, "timestamp", None) is not None
    ]
    if starts:
        return int(min(starts).timestamp() * 1_000_000_000)

    # No stamp (a session that predates it, or a node that never opened the
    # round) and nothing logged for this round yet. The last step of an earlier
    # round is the boundary this one started after: it still excludes finished
    # rounds' artifacts, which is what the round scoping exists for.
    ends = [
        step.timestamp
        for step in steps
        if 0 < int(getattr(step, "round_number", 0) or 0) < current
        and getattr(step, "timestamp", None) is not None
    ]
    if ends:
        return int(max(ends).timestamp() * 1_000_000_000)

    # Nothing to bound with at all -- the first round of a session, where every
    # artifact on disk does belong to the round in progress.
    return 0


def unchanged_artifact_stems(
    working_dir: str | Path, before: dict[str, int], since_ns: int = 0,
) -> set[str]:
    """Return prior artifacts not overwritten by the correcting execution.

    ``since_ns`` bounds the answer to artifacts produced during the round that is
    mutating, matching the rule ``invalidate_current_artifacts`` already applies
    to the model-level record (``artifact_round not in (0, round_number)``). The
    file-stem path never got that scoping, so one mutating cell in round N
    blacklisted every figure from rounds 1..N-1 for the rest of the session --
    and since the set is a monotonic union with a single narrow subtraction, the
    only way back out was regenerating that exact figure. Measured across 27 real
    sessions it removed 76% of all gallery candidates.

    Defaulted rather than required: this function is re-exported by nine
    ``backend.agents.bioinformatics`` modules.
    """
    after = workspace_artifact_versions(working_dir)
    return {
        stem for stem, version in before.items()
        if after.get(stem) == version and version >= since_ns
    }


_MAX_NAMED_STALE_ARTIFACTS = 12


def describe_invalidated_artifacts(stems: set[str], model_count: int) -> str:
    """One line telling the agent which of its own artifacts just went stale.

    Invalidation was silent: seven consumers read the record as "hide this from
    the report" and none as "regenerate this", and nothing in any prompt, tool
    result, or context block ever mentioned it. The producer could therefore
    never repair the deficit it had just caused. This is that missing signal --
    a statement of fact routed back to the party able to act on it, not an
    instruction, since whether a figure is still worth having is the agent's
    call.
    """
    if not stems and model_count <= 0:
        return ""
    names = sorted(stems)
    shown = ", ".join(names[:_MAX_NAMED_STALE_ARTIFACTS])
    if len(names) > _MAX_NAMED_STALE_ARTIFACTS:
        shown += f", and {len(names) - _MAX_NAMED_STALE_ARTIFACTS} more"
    total = len(names) or model_count
    return (
        f"[Provenance] This step changed the expression representation, so {total} "
        f"artifact(s) produced earlier in this round no longer match the data they "
        f"describe and have been withheld from the report"
        + (f": {shown}." if shown else ".")
        + " Regenerate any of them you want reported."
    )


def record_invalidated_file_stems(state: Any, stems: set[str]) -> None:
    if not stems:
        return
    state.invalidated_artifact_stems = sorted(
        set(state.invalidated_artifact_stems or []) | set(stems)
    )
