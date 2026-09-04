"""Turn measurements shown in a biochemistry image into indicator records.

Images were read into prose and stopped there. That prose routinely carries
every number a figure prints and the group label attached to each one, and none
of it reached ``state.biochemistry_indicators`` -- the structured sink that group
comparison, status classification and mechanism adjudication all read from. A
study whose biochemistry arrived as a figure therefore had, everywhere
downstream, no biochemistry at all: the values survived only as narrative quoted
from the vision text, while the gates correctly reported that nothing could be
adjudicated.

Kept out of the agent module, which is at its size limit.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

from ...models.schemas import AnalysisState, DataFile
from ...utils.llm_client import UpstreamExhaustedError
from .prompts import image_measurement_extraction_prompt
from .records import clean_optional_text, reconcile_provenance

logger = logging.getLogger(__name__)

_MARKER = "measurement_extraction"

# Every indicator ever minted is rendered into step 1's classification
# prompt, which does not truncate. How many rows one figure may contribute
# is therefore bounded here rather than left to whatever the model emits.
_MAX_ROWS_PER_IMAGE = 60


def analysis_fingerprint(bio_file: DataFile, analysis: str) -> str:
    """Identify the image version these measurements were read from.

    Prefers the digest the vision step already computed; falls back to the text
    so the once-only guard still holds when the file could not be hashed.
    """
    extracted = bio_file.extracted_data or {}
    return str(extracted.get("input_sha256") or "") or hashlib.sha256(
        analysis.encode("utf-8", "replace")
    ).hexdigest()


def carry_marker_across_vision(bio_file: DataFile, marker: Optional[Dict[str, Any]]) -> None:
    """Put the once-only marker back after the vision step rewrote its home.

    ``_analyze_single_image`` owns ``extracted_data`` and replaces it wholesale
    on every non-terminal analysis, so an image the vision step grades anything
    other than usable is re-read each round -- and would have had its marker
    erased each round, re-minting the same measurements until they looked like
    replicates. The marker is keyed by image digest, so restoring it is safe:
    a genuinely changed image produces a different fingerprint and mints again.
    """
    if marker is None or (bio_file.extracted_data or {}).get(_MARKER):
        return
    extracted = dict(bio_file.extracted_data or {})
    extracted[_MARKER] = marker
    bio_file.extracted_data = extracted


def already_minted(bio_file: DataFile, fingerprint: str) -> bool:
    prior = (bio_file.extracted_data or {}).get(_MARKER) or {}
    return bool(fingerprint) and prior.get("input_sha256") == fingerprint


def mark_minted(bio_file: DataFile, fingerprint: str, indicator_ids: List[str]) -> None:
    """Record what this image version produced.

    ``process`` runs once per round. Without this, one figure's measurements
    would be appended again every round, and the duplicates would read
    downstream as replicates the experiment never had.
    """
    extracted = dict(bio_file.extracted_data or {})
    extracted[_MARKER] = {
        "input_sha256": fingerprint,
        "indicator_ids": list(indicator_ids),
        "count": len(indicator_ids),
    }
    bio_file.extracted_data = extracted


def provenance_notes(filename: str, row: Dict[str, Any]) -> List[str]:
    """Say where the number came from and how firmly the figure states it.

    A value estimated against an axis and a value printed as text are not the
    same evidence. The significance marker beside a panel is the test the authors
    actually ran; deriving one from two extracted group means would invent a
    statistic, so it is carried across verbatim instead.
    """
    panel = clean_optional_text(row.get("panel"))
    origin = f"value read from image {filename}"
    if panel:
        origin += f", panel {panel}"
    notes = [origin]
    if str(row.get("readout") or "").strip().lower() == "chart_estimate":
        notes.append("chart estimate read against an axis, not a printed number")
    annotation = clean_optional_text(row.get("statistical_annotation"))
    if annotation:
        notes.append(f"figure annotates {annotation}")
    # Kept as prose, never as BiochemistryIndicator.sample_number -- see
    # register_rows for why a replicate label read off a panel must not become
    # the study's replicate count.
    replicate = clean_optional_text(row.get("sample"))
    if replicate:
        notes.append(f"figure labels this reading {replicate!r}")
    return notes


def extraction_is_readable(payload: Any) -> bool:
    """Whether the extraction response can be read at all.

    A readable answer carries ``results`` as a list; an EMPTY one is a real
    answer about a qualitative panel, and the image is finished with. Anything
    else means the response could not be read, and the two must not be
    conflated: the once-only marker is permanent, so recording "nothing to
    mint" for a response that was merely unparseable silences that image for
    the rest of the session, including after the operator switches models.

    The unreadable case is not hypothetical. ``_safe_parse_json`` returns the
    FIRST balanced object it finds, so a model that answers with a bare JSON
    array — observed behaviour on at least one configured vision model — yields
    a single measurement row carrying no ``results`` key at all.
    """
    return isinstance(payload, dict) and isinstance(payload.get("results"), list)


def usable_rows(payload: Any) -> List[Dict[str, Any]]:
    """Keep the rows that carry a name and a real number, bounded."""
    rows = payload.get("results") if isinstance(payload, dict) else None
    kept: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        indicator = clean_optional_text(row.get("indicator"))
        if not indicator:
            continue
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            logger.warning(
                "[BiochemistryImage] non-numeric value for %r: %r",
                indicator, row.get("value"),
            )
            continue
        if not math.isfinite(value):
            logger.warning("[BiochemistryImage] non-finite value for %r", indicator)
            continue
        kept.append({**row, "indicator": indicator, "value": value})
        if len(kept) >= _MAX_ROWS_PER_IMAGE:
            logger.warning(
                "[BiochemistryImage] capping this image at %d measurement(s)",
                _MAX_ROWS_PER_IMAGE,
            )
            break
    return kept


async def register_rows(agent, bio_file: DataFile, rows: List[Dict[str, Any]],
                        state: AnalysisState) -> List[str]:
    """Send each row through the same reconcile/add path a data file uses.

    ``reconcile_provenance`` already accepts a group label parsed from a file
    body when the upload declared none, and records that it did so. That is
    exactly the image case: the label can only live inside the picture — and
    for an image it is authoritative, because one figure carries every group it
    plots and so can be uploaded under only one of them. Left to the ordinary
    declared-wins rule, a Control-vs-treated chart uploaded under either arm
    mints both arms' bars under that one group and the comparison runs against
    itself.

    Replicate identity is deliberately NOT taken from the figure.
    ``biochemistry_sample_counts`` derives a group's detected n from the
    distinct ``sample_number`` values on its indicators and reports it as
    authoritative over the declared design, so one replicate label read off a
    panel of group means would restate the study as n=1. What the figure says
    about replicates is kept in ``provenance_notes`` instead.

    All or nothing: ``add_result`` commits to ``state`` as it goes, so a row
    that raises part way would leave the earlier rows behind with no once-only
    marker, and the next round would mint the whole figure again on top of them
    — duplicates that read downstream as replicates the experiment never had.
    """
    start = len(state.biochemistry_indicators)
    minted: List[str] = []
    try:
        for row in rows:
            group_id, sample_number, label, notes = reconcile_provenance(
                file_group_id=bio_file.group_id,
                file_sample_number=bio_file.sample_number,
                row_group_label=row.get("group"),
                row_sample_value=None,
                state=state,
                row_label_is_authoritative=True,
            )
            record = await agent.add_result(
                indicator=row["indicator"],
                value=row["value"],
                unit=clean_optional_text(row.get("unit")) or "",
                state=state,
                reference_range=clean_optional_text(row.get("reference_range")),
                group_id=group_id,
                sample_number=sample_number,
                source_group_label=label,
                provenance_notes=provenance_notes(bio_file.filename, row) + notes,
            )
            minted.append(record.id)
    except BaseException:
        del state.biochemistry_indicators[start:]
        raise
    return minted


async def extract_image_measurements(agent, state: AnalysisState,
                                     image_files: List[DataFile],
                                     instruction: str) -> Dict[str, Optional[str]]:
    """Mint indicator records from the measurements each image prints.

    Returns ``{file_path: analysis-or-None}`` so the image step that follows
    reuses this vision pass rather than paying for it twice.
    """
    analyses: Dict[str, Optional[str]] = {}
    for bio_file in image_files:
        prior_marker = (bio_file.extracted_data or {}).get(_MARKER)
        try:
            analysis = await agent._analyze_single_image(bio_file, instruction)
        finally:
            # The vision step also replaces ``extracted_data`` on the path where
            # it RAISES: an exhausted upstream writes {"error": ...} and re-raises
            # so the workflow's LLM-failure gate can offer Retry, and that retry
            # re-enters this node. Restoring the marker unconditionally is what
            # stops the retry from minting the same figure a second time.
            carry_marker_across_vision(bio_file, prior_marker)
        analyses[bio_file.file_path] = analysis
        if not analysis:
            continue

        fingerprint = analysis_fingerprint(bio_file, analysis)
        if already_minted(bio_file, fingerprint):
            logger.info(
                "[BiochemistryImage] measurements already registered for %s",
                bio_file.filename,
            )
            continue

        try:
            response = await agent.chat(
                image_measurement_extraction_prompt(bio_file.filename, analysis),
                temperature=0.1,
            )
            payload = agent._safe_parse_json(
                response, agent_label="BiochemistryAgent"
            )
            if not extraction_is_readable(payload):
                logger.warning(
                    "[BiochemistryImage] unreadable extraction response for %s; "
                    "leaving the image unmarked so a later round can retry",
                    bio_file.filename,
                )
                continue
            rows = usable_rows(payload)
            minted = await register_rows(agent, bio_file, rows, state)
        except UpstreamExhaustedError:
            raise
        except Exception:
            logger.exception(
                "[BiochemistryImage] failed to extract measurements from %s",
                bio_file.filename,
            )
            continue

        mark_minted(bio_file, fingerprint, minted)
        logger.info(
            "[BiochemistryImage] registered %d measurement(s) from %s",
            len(minted), bio_file.filename,
        )
    return analyses
