"""Pathology image preparation, tile analysis, and aggregation."""
from __future__ import annotations

from typing import Any, Dict, List

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
You are analyzing {len(batch_tiles)} regions (tiles {start_idx + 1} to {end_idx}) from a {image_info.data_type} stained tissue section.
Total tiles in this section: {len(tiles_to_analyze)}

For EACH tile in order, report only features that are directly visible in that
tile and interpretable for the stated stain. Use observation-first language:
describe the visible morphology or stain pattern, its distribution/localization,
and its approximate extent when defensible. Do not force every tile through a
fixed disease checklist, infer an etiology, or reuse a generic category label
such as "Cell morphology changes" as the finding itself.

Return one JSON object with a "tile_findings" array containing EXACTLY {len(batch_tiles)} objects in image order.
Each object must contain:
- "tile_index": the zero-based integer index for that tile, beginning at {start_idx}
- "findings": a concrete, non-empty observation from that tile

Never copy field descriptions, use ellipses, or return placeholders such as "Tile N findings".
Do not repeat an identical abnormal sentence for different tiles; describe the
tile-specific evidence. Identical explicit normal findings are allowed.
If no abnormality is visible in a tile, use the explicit finding "No abnormality detected."
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
    severity_label_to_score = bindings.severity_label_to_score
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
This is an overview of a tissue section stained with {image_info.data_type}.

Analysis instruction: {instruction}

Please first describe the overall features of the section:
1. Is the overall tissue structure normal?
2. Are there obvious regional lesions or abnormalities?
3. Overall severity assessment (if applicable)

Please reply in JSON format:
{{
    "overall_structure": "Overall structure description",
    "regional_changes": "Regional changes description",
    "severity": "one of: normal | mild | moderate | marked | severe (will be mapped to a 0-4 score)",
    "areas_of_interest": ["Area 1 requiring detailed observation", ...]
}}
"""

        overview_result = await agent._call_llm_with_hitl(
            lambda: agent.llm_client.chat_with_vision(
                model=agent.model,
                text_prompt=overview_prompt,
                images=[overview],
                temperature=0.3,
                request_timeout=600.0,  # 10 min for single overview image
                system_prompt=agent.system_prompt,
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

## Overview Analysis
{json.dumps(overview_data, ensure_ascii=False, indent=2)}

## Regional Analysis
{json.dumps(tile_findings, ensure_ascii=False, indent=2)}

## Regional Coverage
{valid_tiles}/{expected_tiles} tiles passed validation. Do not infer findings for omitted tiles.

Please summarize:
1. Main pathological findings (list format)
2. Severity score (0-4)
3. Directions for further analysis
4. Suggested special staining

Reply in JSON format:
{{
    "findings": ["finding 1", "finding 2", ...],
    "severity_score": 0-4,
    "further_analysis": ["direction 1", ...],
    "suggested_staining": ["staining type 1", ...]
}}
"""

        combined_result = await agent._call_llm_with_hitl(
            lambda: agent.chat_without_history(combined_prompt, temperature=0.3),
            image_label=image_info.filename,
            step_label="findings synthesis",
        )
        final_data = agent._safe_parse_json(combined_result, default={"raw_response": combined_result}, agent_label="PathologyAgent")

        logger.info(f"[PathologyAgent] Image analysis complete: {len(final_data.get('findings', []))} findings")

        # Map the overview-level textual severity onto the canonical 0-4
        # numeric scale so downstream consumers have a single, consistent
        # scalar. The detailed tile-level severity_score is emitted separately
        # below (final_data). overview_data keeps the raw LLM "severity" text.
        if isinstance(overview_data, dict) and "severity_score" not in overview_data:
            overview_data["severity_score"] = severity_label_to_score(
                overview_data.get("severity", "")
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
            "severity_score": final_data.get("severity_score", 0),
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
