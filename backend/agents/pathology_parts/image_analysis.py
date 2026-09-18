"""Pathology image preparation, tile analysis, and aggregation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _severity_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rubric_name(block: Any) -> str:
    """The rubric's short identity, lowercased. Never guessed from prose."""
    if not isinstance(block, dict):
        return ""
    return str(block.get("severity_rubric_name") or "").strip().casefold()


def _severity_observations(overview: Any, synthesis: Any) -> List[str]:
    """Describe how a reported severity fails to hang together.

    A severity number only means something with the scale it was graded on,
    so a score without a maximum, a score outside its own range, or a score
    with no stated rubric is worth flagging. These are OBSERVATIONS carried
    alongside the result for the PI to weigh -- nothing here rejects an
    image, blocks a report, or edits the model's answer.
    """
    notes: List[str] = []
    for label, block in (("overview", overview), ("synthesis", synthesis)):
        if not isinstance(block, dict):
            continue
        score = _severity_number(block.get("severity_score"))
        maximum = _severity_number(block.get("severity_max"))
        basis = str(block.get("severity_basis") or "").strip()
        if score is None and maximum is None and not basis:
            continue
        if score is not None and maximum is None:
            notes.append(
                f"{label}: severity {score:g} reported without a maximum, so the "
                "scale it was graded on is unknown"
            )
        if score is None and maximum is not None:
            notes.append(f"{label}: a severity maximum was given without a score")
        if score is not None and maximum is not None and not 0 <= score <= maximum:
            notes.append(
                f"{label}: severity {score:g} lies outside its own stated range "
                f"0-{maximum:g}"
            )
        if score is not None and not basis:
            notes.append(
                f"{label}: severity {score:g} reported without naming the rubric it "
                "was graded against, so it is not comparable to any other image"
            )
    # Compare rubric IDENTITY -- the short name and the range -- not the prose.
    # Two descriptions of the same scale differ in wording constantly; comparing
    # the explanations made every rephrasing look like a second rubric, and
    # fuzzy-matching free text would only trade one wrong answer for another.
    o_name = _rubric_name(overview)
    s_name = _rubric_name(synthesis)
    o_max = _severity_number((overview or {}).get("severity_max")) if isinstance(overview, dict) else None
    s_max = _severity_number((synthesis or {}).get("severity_max")) if isinstance(synthesis, dict) else None
    o_scored = _severity_number((overview or {}).get("severity_score")) is not None
    s_scored = _severity_number((synthesis or {}).get("severity_score")) is not None
    if o_name and s_name and o_name != s_name:
        notes.append(
            f"overview graded on {o_name!r} and synthesis on {s_name!r}; the two "
            "severities are on different rubrics and are not comparable"
        )
    elif o_name and s_name and o_max is not None and s_max is not None and o_max != s_max:
        notes.append(
            f"both layers name the rubric {o_name!r} but grade out of {o_max:g} and "
            f"{s_max:g}; the scores are on different ranges"
        )
    for label, scored, name in (("overview", o_scored, o_name), ("synthesis", s_scored, s_name)):
        if scored and not name:
            notes.append(
                f"{label}: rubric identity unavailable -- a score was given with no "
                "short rubric name, so it cannot be matched to any other grading"
            )
    return notes

logger = __import__("logging").getLogger(__name__)


async def _analyze_tile_batches(
    agent,
    image_info,
    tiles_to_analyze,
    *,
    batch_size: int,
    bindings,
):
    logger = bindings.logger
    tile_findings = []
    invalid_batches = []
    num_batches = (len(tiles_to_analyze) + batch_size - 1) // batch_size
    logger.info(
        f"[PathologyAgent] Analyzing {len(tiles_to_analyze)} tiles in {num_batches} batches"
    )
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tiles_to_analyze))
        batch_tiles = tiles_to_analyze[start_idx:end_idx]

        logger.info(f"[PathologyAgent] Processing batch {batch_idx + 1}/{num_batches} ({len(batch_tiles)} tiles)")

        # Describe the response contract without an example finding. In
        # production, the former "Tile N findings..." example was echoed
        # verbatim by the model and silently accepted as evidence.
        batch_prompt = f"""
For EACH tile in order, report only features that are directly visible in that
tile and interpretable for the stated stain. Use observation-first language:
describe the visible morphology or stain pattern, its distribution/localization,
and its approximate extent when defensible. Do not force every tile through a
fixed disease checklist, infer an etiology, or reuse a generic category label
such as "Cell morphology changes" as the finding itself.

Return one JSON object with a "tile_findings" array containing one object per tile in image order.
Each object must contain:
- "tile_index": the zero-based integer index for that tile
- "findings": a concrete, non-empty observation from that tile

Never copy field descriptions, use ellipses, or return placeholders such as "Tile N findings".
Do not repeat an identical abnormal sentence for different tiles; describe the
tile-specific evidence. Identical explicit normal findings are allowed.
If no abnormality is visible in a tile, use the explicit finding "No abnormality detected."

Stain: {image_info.data_type}
Total tiles in this section: {len(tiles_to_analyze)}

## Current Batch
You are analyzing {len(batch_tiles)} regions (tiles {start_idx + 1} to {end_idx}).
Return EXACTLY {len(batch_tiles)} objects, with tile_index beginning at {start_idx}.
"""

        # Dynamic timeout: 30s per image on top of the default floor
        batch_timeout = max(600.0, 120.0 + 30.0 * len(batch_tiles))
        normalized_batch: List[Dict[str, Any]] = []
        last_validation_errors: List[str] = []
        for validation_attempt in range(2):
            validation_prompt = batch_prompt
            if validation_attempt:
                validation_prompt += (
                    "\nYour previous response failed validation: "
                    + "; ".join(last_validation_errors[:5])
                    + "\nReturn a corrected batch that follows every constraint exactly."
                )
            batch_result = await agent._call_llm_with_hitl(
                lambda bt=validation_prompt, imgs=batch_tiles, to=batch_timeout: agent.llm_client.chat_with_vision(
                    model=agent.model,
                    text_prompt=bt,
                    images=imgs,
                    temperature=0.3,
                    request_timeout=to,
                    system_prompt=agent.system_prompt,
                    prefer_stream=True,
                ),
                image_label=image_info.filename,
                step_label=f"tile batch {batch_idx + 1}/{num_batches}",
            )

            batch_content = batch_result.content if batch_result else ""
            batch_data = bindings._coerce_tile_batch(
                agent._safe_parse_json(
                    batch_content,
                    default={"raw_response": batch_content},
                    agent_label="PathologyAgent",
                ),
                batch_content,
            )
            normalized_batch, last_validation_errors = bindings._validate_tile_batch(
                batch_data,
                start_idx=start_idx,
                expected_count=len(batch_tiles),
                prior_findings=tile_findings,
            )
            if not last_validation_errors:
                tile_findings.extend(normalized_batch)
                break
            logger.warning(
                "[PathologyAgent] Invalid tile batch %s/%s (validation attempt %s/2): %s",
                batch_idx + 1,
                num_batches,
                validation_attempt + 1,
                "; ".join(last_validation_errors[:5]),
            )

        if last_validation_errors:
            invalid_batches.append({
                "batch_index": batch_idx,
                "tile_start": start_idx,
                "tile_end": end_idx - 1,
                "attempts": 2,
                "errors": last_validation_errors,
            })


    return tile_findings, invalid_batches


async def analyze_image(
    agent,
    image_info: Any,
    instruction: str,
    *,
    bindings,
) -> Dict[str, Any]:
    """
    Analyze a single pathology image

    Args:
        image_info: DataFile object (section=pathology)
        instruction: Analysis instruction

    Returns:
        Analysis results
    """
    os = bindings.os
    logger = bindings.logger
    load_image = getattr(bindings, "load_image")
    process_pathology_image = getattr(bindings, "process_pathology_image")
    create_overview_image = getattr(bindings, "create_overview_image")
    json = bindings.json
    _ImageSkipRequested = bindings._ImageSkipRequested
    agent.refresh_runtime_config_if_needed()
    # Set LLM context before vision calls (client is shared between agents)
    agent._set_llm_context()

    # Batch size for tile analysis — smaller batches are more reliable
    # and avoid vision API timeouts on large multimodal requests
    BATCH_SIZE = 5

    # Check if file exists
    if not os.path.exists(image_info.file_path):
        logger.error(f"[PathologyAgent] Image file not found: {image_info.file_path}")
        return {
            "image_id": image_info.id,
            "image_type": image_info.data_type,
            "status": "failed",
            "tile_validation": {"status": "failed"},
            "error": f"Image file not found: {image_info.file_path}",
            "findings": []
        }

    try:
        logger.info(f"[PathologyAgent] Processing image: {image_info.file_path}")

        # Load image once and reuse for both tiling and overview
        img = load_image(image_info.file_path)

        # Process image into tiles
        tiles = process_pathology_image(
            image_info.file_path,
            tile_size=agent.tile_size,
            max_tiles=agent.max_tiles,
            preloaded_image=img
        )

        logger.info(f"[PathologyAgent] Generated {len(tiles)} tiles from image")

        if not tiles:
            return {
                "image_id": image_info.id,
                "image_type": image_info.data_type,
                "status": "error",
                "tile_validation": {
                    "status": "error",
                    "expected_tiles": 0,
                    "valid_tiles": 0,
                    "coverage_ratio": 0.0,
                    "invalid_batches": [],
                },
                "error": "No valid tiles generated from image (image may be too small or blank)",
                "findings": []
            }

        # Create overview from already-loaded image
        overview = create_overview_image(img)

        logger.info(f"[PathologyAgent] Starting overview analysis...")

        # Analyze overview first
        overview_prompt = f"""
Please first describe the overall features of the section:
1. What is the overall tissue architecture, and does it depart from the expected architecture
   for this tissue and stain? Say what you are comparing against
2. Are there regional lesions or abnormalities, and are they focal or diffuse?
3. Overall severity — ONLY if a defensible scale applies here

Severity is a graded judgement against a rubric, and a rubric belongs to one tissue and one
lesion class (they are not interchangeable across tissues, stains, or lesion types). If you can
name the scale you are applying and it fits this tissue, stain and lesion, grade on it, report
the score WITH its maximum, and say in severity_basis which scale it is and what each level
means. If you cannot name such a scale, set severity_score AND severity_max to null and say why
in severity_basis. Do NOT fall back to a generic normal/mild/moderate/severe ladder: relabelling
an ungraded impression as a tier makes it look comparable across images and studies when it is
not, which is the same defect as an unanchored number.

Please reply in JSON format:
{{
    "overall_structure": "Overall structure description",
    "regional_changes": "Regional changes description",
    "severity_rubric_name": "<short canonical name of the scale, e.g. \"Ishak fibrosis\", or null>",
    "severity_score": <int, or null when no named scale applies>,
    "severity_max": <int, or null>,
    "severity_basis": "what the scale measures and what each level means, or why none applies",
    "areas_of_interest": ["Area 1 requiring detailed observation", ...]
}}

This is an overview of a tissue section stained with {image_info.data_type}.

Analysis instruction: {instruction}
"""

        overview_result = await agent._call_llm_with_hitl(
            lambda: agent.llm_client.chat_with_vision(
                model=agent.model,
                text_prompt=overview_prompt,
                images=[overview],
                temperature=0.3,
                request_timeout=600.0,  # 10 min for single overview image
                system_prompt=agent.system_prompt,
                prefer_stream=True,
            ),
            image_label=image_info.filename,
            step_label="overview analysis",
        )

        # Parse overview result (handle None)
        overview_content = overview_result.content if overview_result else ""
        overview_data = agent._safe_parse_json(overview_content, default={"raw_response": overview_content}, agent_label="PathologyAgent")
        logger.info(f"[PathologyAgent] Overview analysis complete")

        # Analyze tiles serially in bounded batches.
        tiles_to_analyze = tiles[:agent.max_tiles]
        tile_findings, invalid_batches = await _analyze_tile_batches(
            agent,
            image_info,
            tiles_to_analyze,
            batch_size=BATCH_SIZE,
            bindings=bindings,
        )

        expected_tiles = len(tiles_to_analyze)
        valid_tiles = len(tile_findings)
        coverage_ratio = valid_tiles / expected_tiles if expected_tiles else 0.0
        tile_validation = {
            "status": (
                "completed" if valid_tiles == expected_tiles
                else "partial" if valid_tiles
                else "error"
            ),
            "expected_tiles": expected_tiles,
            "valid_tiles": valid_tiles,
            "coverage_ratio": coverage_ratio,
            "invalid_batches": invalid_batches,
        }

        if not tile_findings:
            logger.error(
                "[PathologyAgent] No valid tile batches for image %s after validation retries",
                image_info.id,
            )
            return {
                "image_id": image_info.id,
                "image_type": image_info.data_type,
                "overview": overview_data,
                "tile_count": expected_tiles,
                "tile_findings": [],
                "tile_validation": tile_validation,
                "status": "error",
                "error": "No valid tile findings after validation retries",
                "findings": [],
            }

        logger.info(
            "[PathologyAgent] Tile batches complete: %s/%s valid; synthesizing findings",
            valid_tiles,
            expected_tiles,
        )

        # Combine findings
        combined_prompt = f"""
Based on the following regional analysis results, please summarize the pathological findings for the entire section:

Please summarize:
1. Main pathological findings (list format)
2. Severity, graded only against a scale you can name and defend for THIS tissue, stain and
   lesion class. Report the score together with its maximum and what the scale measures, so the
   number can be read. If no such scale applies, set severity_score to null and say why in
   severity_basis — do not emit an unanchored 0-4 number, which looks quantitative and is not
   comparable across images, stains, or studies
3. Directions for further analysis
4. Suggested special staining

Reply in JSON format:
{{
    "findings": ["finding 1", "finding 2", ...],
    "severity_rubric_name": "<the SAME short scale name the overview used, copied verbatim; or a new name if you are deliberately grading on a different scale, which you must say in severity_basis>",
    "severity_score": <int or null>,
    "severity_max": <int or null>,
    "severity_basis": "what the scale measures and what each level means, or why none applies",
    "further_analysis": ["direction 1", ...],
    "suggested_staining": ["staining type 1", ...]
}}

## Overview Analysis
{json.dumps(overview_data, ensure_ascii=False, indent=2)}

## Regional Analysis
{json.dumps(tile_findings, ensure_ascii=False, indent=2)}

## Regional Coverage
{valid_tiles}/{expected_tiles} tiles passed validation. Do not infer findings for omitted tiles.
"""

        combined_result = await agent._call_llm_with_hitl(
            lambda: agent.chat_without_history(combined_prompt, temperature=0.3),
            image_label=image_info.filename,
            step_label="findings synthesis",
        )
        final_data = agent._safe_parse_json(combined_result, default={"raw_response": combined_result}, agent_label="PathologyAgent")

        logger.info(f"[PathologyAgent] Image analysis complete: {len(final_data.get('findings', []))} findings")

        # Overview and synthesis now grade on the SAME rubric-bound shape
        # (score + max + basis, or null). There is no generic tier ladder left to
        # project onto a canonical 0-4, and no projection to do: a severity that is
        # not anchored to a named scale stays absent rather than becoming a number.
        if isinstance(overview_data, dict):
            overview_data.setdefault("severity_score", None)
            overview_data.setdefault("severity_max", None)
            overview_data.setdefault("severity_basis", "")
            overview_data.setdefault("severity_rubric_name", None)

        # Read the two severity reports and describe any way they do not hang
        # together. This deliberately does NOT reject, clamp, or drop a value:
        # the PI decides what a questionable grade is worth, and a host that
        # silently discarded it would hide the very thing worth looking at.
        severity_observations = _severity_observations(overview_data, final_data)
        if severity_observations:
            logger.info(
                "[PathologyAgent] Severity reporting observations for %s: %s",
                image_info.filename,
                "; ".join(severity_observations),
            )

        return {
            "image_id": image_info.id,
            "image_type": image_info.data_type,
            "overview": overview_data,
            "tile_count": expected_tiles,
            "tile_findings": tile_findings,
            "tile_validation": tile_validation,
            "status": tile_validation["status"],
            "findings": final_data.get("findings", []),
            "severity_score": final_data.get("severity_score"),
            "severity_max": final_data.get("severity_max"),
            "severity_basis": final_data.get("severity_basis", ""),
            "severity_rubric_name": final_data.get("severity_rubric_name"),
            "severity_observations": severity_observations,
            "further_analysis": final_data.get("further_analysis", []),
            "suggested_staining": final_data.get("suggested_staining", [])
        }

    except _ImageSkipRequested:
        raise
    except Exception as e:
        # User termination must reach the workflow boundary. This check is
        # intentionally before the generic per-image error conversion.
        from ...workflow.helpers import WorkflowTerminatedError
        if isinstance(e, WorkflowTerminatedError):
            raise
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"[PathologyAgent] Failed to analyze image {image_info.id}: {error_detail}")
        return {
            "image_id": image_info.id if image_info else "unknown",
            "image_type": getattr(image_info, "data_type", "") if image_info else "",
            "status": "failed",
            "tile_validation": {"status": "failed"},
            "error": str(e),
            "findings": []
        }
