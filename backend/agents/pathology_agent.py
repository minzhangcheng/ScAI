"""
Pathology Agent - Analyzes histology images
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
import json
import logging
import os
import re

from .base import BaseAgent
from ..constants import LlmFailureAction
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, DataFile
)
from ..utils.formatting import safe_parse_json_array
from ..utils.image_utils import (
    process_pathology_image, create_overview_image, load_image
)
from ..utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ..utils.settings_provider import AgentSettingsProvider

logger = logging.getLogger(__name__)


_TILE_FINDING_PLACEHOLDER_RE = re.compile(
    r"^tile\s*(?:\d+|n)?(?:\s*[:\-])?\s*findings?\s*(?:\.{3}|…)?$",
    re.IGNORECASE,
)
_GENERIC_TILE_FINDINGS = frozenset({
    "analysis failed",
    "findings",
    "n/a",
    "no findings",
    "none",
    "not available",
    "unknown",
})
_PROMPT_CATEGORY_PREFIX_RE = re.compile(
    r"^(?:category|feature|observation|finding)\s*(?:\d+|n)?\s*[:\-]",
    re.IGNORECASE,
)


def _finding_key(value: str) -> str:
    """Normalize one observation for exact semantic-template deduplication."""
    return re.sub(r"\s+", " ", value.casefold()).strip().rstrip(".!")


def _coerce_tile_batch(batch_data: Any, raw_response: str) -> Any:
    """Recover a batch the model returned as a bare JSON array.

    safe_parse_json scans for the first '{', so a top-level array comes back as
    its own first element — a one-tile dict with no "tile_findings" key. That
    reads downstream as a schema violation and discards the whole batch, even
    though the model produced every finding asked for. Vision models differ on
    whether they wrap the array in the documented object; both shapes are
    unambiguous, so accept either.
    """
    if isinstance(batch_data, list):
        return {"tile_findings": batch_data}
    if isinstance(batch_data, dict) and "tile_findings" in batch_data:
        return batch_data
    entries = safe_parse_json_array(raw_response)
    if entries:
        return {"tile_findings": entries}
    return batch_data


def _validate_tile_batch(
    batch_data: Any,
    *,
    start_idx: int,
    expected_count: int,
    prior_findings: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Validate one model-produced tile batch and assign canonical indices.

    A batch is atomic: any malformed or placeholder entry invalidates the full
    batch. Model-provided indices are deliberately ignored because the prompt
    uses human-facing one-based tile labels while internal indices are zero-based.
    """
    errors: List[str] = []
    if isinstance(batch_data, list):
        batch_data = {"tile_findings": batch_data}
    if not isinstance(batch_data, dict):
        return [], ["response must be a JSON object"]

    entries = batch_data.get("tile_findings")
    if not isinstance(entries, list):
        return [], ["tile_findings must be an array"]
    if len(entries) != expected_count:
        return [], [
            f"tile_findings count {len(entries)} does not match expected {expected_count}"
        ]

    normalized: List[Dict[str, Any]] = []
    prior_keys = {
        _finding_key(str(item.get("findings", "")))
        for item in (prior_findings or [])
        if isinstance(item, dict) and item.get("findings")
    }
    batch_keys: set[str] = set()
    explicit_normal_key = _finding_key("No abnormality detected.")
    for offset, entry in enumerate(entries):
        label = start_idx + offset + 1
        if not isinstance(entry, dict):
            errors.append(f"tile {label} entry must be an object")
            continue
        finding = entry.get("findings")
        if not isinstance(finding, str):
            errors.append(f"tile {label} findings must be a string")
            continue
        finding = finding.strip()
        folded = _finding_key(finding)
        if not finding:
            errors.append(f"tile {label} findings are empty")
        elif len(finding) < 12:
            errors.append(f"tile {label} findings are too short")
        elif folded in _GENERIC_TILE_FINDINGS:
            errors.append(f"tile {label} findings are generic")
        elif _TILE_FINDING_PLACEHOLDER_RE.fullmatch(finding):
            errors.append(f"tile {label} findings repeat the schema placeholder")
        elif _PROMPT_CATEGORY_PREFIX_RE.match(finding):
            errors.append(
                f"tile {label} findings copy a prompt category instead of an observation"
            )
        elif (
            folded != explicit_normal_key
            and (folded in prior_keys or folded in batch_keys)
        ):
            errors.append(
                f"tile {label} repeats an identical abnormal finding from another tile"
            )
        else:
            normalized.append({
                **entry,
                "tile_index": start_idx + offset,
                "findings": finding,
            })
            batch_keys.add(folded)

    if errors:
        return [], errors
    return normalized, []


def _summarize_status(detailed_results, all_findings) -> str:
    """Summarize image outcomes without hiding degraded modality coverage."""
    if not detailed_results:
        return "error"
    any_success = any(not r.get("error") and r.get("findings") for r in detailed_results)
    if not any_success:
        return "error"
    if any(
        r.get("error")
        or r.get("skipped")
        or str(r.get("status") or "").lower() in {"partial", "error", "failed"}
        for r in detailed_results
    ):
        return "partial"
    return "completed"


def severity_label_to_score(label: str) -> int:
    """Map textual severity tiers to the canonical 0-4 numeric scale.
    normal=0, mild=1, moderate=2, marked=3, severe=4. Unknown -> 0."""
    return {"normal": 0, "mild": 1, "moderate": 2, "marked": 3, "severe": 4}.get(
        (label or "").strip().lower(), 0
    )


class _ImageSkipRequested(Exception):
    """Internal sentinel: user chose to skip the current pathology image at the
    LLM-failure HITL gate. Caught by process() to skip only this image."""
    def __init__(self, image_label: str, last_error: Optional[BaseException] = None):
        self.image_label = image_label
        self.last_error = last_error
        super().__init__(f"Pathology image skipped by user: {image_label}")


class PathologyAgent(BaseAgent):
    """
    Pathology Agent
    
    Responsibilities:
    - Analyze histology/pathology tissue sections
    - Identify pathological features based on tissue type and staining
    - Suggest additional staining when needed
    - Quantify pathological changes
    """
    
    def __init__(
        self,
        settings_provider: 'AgentSettingsProvider',
        tile_size: int = 1024,
        max_tiles: int = 50,
    ):
        super().__init__(
            agent_type=AgentType.PATHOLOGY,
            settings_provider=settings_provider,
        )
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        # HITL callback invoked when an LLM vision/chat call is exhausted
        # (UpstreamExhaustedError). Signature:
        #   async (error: UpstreamExhaustedError, image_label: str) -> LlmFailureAction
        # When unset, the agent preserves the legacy behaviour of skipping the
        # failed image silently (handled by process()).
        self.llm_intervention_callback: Optional[
            Callable[[UpstreamExhaustedError, str], Awaitable[LlmFailureAction]]
        ] = None
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process pathology analysis instruction"""
        self.refresh_runtime_config_if_needed()
        # Check if there are images to analyze
        images_to_analyze = kwargs.get("images", state.get_files_by_section("pathology"))
        
        logger.info(f"[PathologyAgent] Processing with {len(images_to_analyze) if images_to_analyze else 0} images")
        
        if not images_to_analyze:
            return {
                "status": "waiting_input",
                "message": "Please upload pathology slide images for analysis",
                "findings": []
            }
        
        # Analyze each image
        all_findings = []
        detailed_results = []
        
        for i, img_info in enumerate(images_to_analyze):
            logger.info(f"[PathologyAgent] Analyzing image {i+1}/{len(images_to_analyze)}: {img_info.filename}")
            
            try:
                result = await self._analyze_image(img_info, instruction)
                all_findings.extend(result.get("findings", []))
                detailed_results.append(result)
                logger.info(f"[PathologyAgent] Image {i+1} analysis complete: {len(result.get('findings', []))} findings")
            except _ImageSkipRequested as skip:
                # User chose SKIP at the LLM-failure HITL gate: drop just this
                # image and continue with the rest. The user was already notified
                # via the intervention dialog, so this is no longer a silent skip.
                logger.warning(
                    f"[PathologyAgent] Image {img_info.filename} skipped by user "
                    f"after LLM failure"
                )
                detailed_results.append({
                    "image_id": img_info.id,
                    "error": f"Skipped by user after LLM failure: {skip.last_error}",
                    "skipped": True,
                    "findings": []
                })
            except Exception as e:
                # WorkflowTerminatedError (user chose TERMINATE) is a subclass of
                # RuntimeError — let it propagate so the workflow aborts.
                from ..workflow.helpers import WorkflowTerminatedError
                if isinstance(e, WorkflowTerminatedError):
                    raise
                logger.exception(f"[PathologyAgent] Failed to analyze image {img_info.filename}")
                detailed_results.append({
                    "image_id": img_info.id,
                    "error": str(e),
                    "findings": []
                })
        
        # Never ask the synthesizer to interpret skipped/failed images. Partial
        # image results remain eligible because their valid tile subset is useful
        # and is accompanied by explicit coverage metadata.
        synthesis_results = [
            result for result in detailed_results if result.get("findings")
        ]
        if synthesis_results:
            try:
                synthesis = await self._synthesize_findings(
                    instruction, synthesis_results, state
                )
            except Exception as e:
                logger.exception("[PathologyAgent] Failed to synthesize findings")
                synthesis = {
                    "summary": "Failed to synthesize findings",
                    "error": str(e)
                }
        else:
            synthesis = {"summary": "No images were successfully analyzed"}
        
        status = _summarize_status(detailed_results, all_findings)
        result = {
            "status": status,
            "findings": all_findings,
            "detailed_results": detailed_results,
            "synthesis": synthesis,
            "suggested_staining": synthesis.get("suggested_staining", []),
            "images_analyzed": len(images_to_analyze),
        }
        if status == "error":
            errors = [r.get("error") for r in detailed_results if r.get("error")]
            result["error"] = "; ".join(e for e in errors if e) or "All pathology image analyses failed"
        return result
    
    async def _call_llm_with_hitl(
        self,
        call_factory: Callable[[], Awaitable[Any]],
        image_label: str,
        step_label: str,
    ) -> Any:
        """Run a single LLM vision/chat call with LLM-failure HITL.

        On UpstreamExhaustedError (the LLM client already exhausted its own
        retries), pause and ask the user what to do:
          - RETRY     → re-attempt only this one LLM call
          - SKIP      → skip the whole current image (raise _ImageSkipRequested)
          - TERMINATE → abort the analysis (raise WorkflowTerminatedError)

        When no intervention callback is wired, the error propagates unchanged so
        the per-image handler in process() skips the image (legacy behaviour).
        """
        self.refresh_runtime_config_if_needed()
        while True:
            try:
                return await call_factory()
            except UpstreamExhaustedError as e:
                if self.llm_intervention_callback is None:
                    # No HITL wired — let process() handle as a per-image failure.
                    raise
                logger.warning(
                    f"[PathologyAgent] LLM exhausted at '{step_label}' for "
                    f"{image_label}; awaiting user intervention"
                )
                action = await self.llm_intervention_callback(e, image_label)
                if action == LlmFailureAction.RETRY:
                    logger.info(
                        f"[PathologyAgent] User chose RETRY for '{step_label}' "
                        f"({image_label})"
                    )
                    continue
                if action == LlmFailureAction.TERMINATE:
                    # Lazy import avoids circular import (workflow imports agents).
                    from ..workflow.helpers import WorkflowTerminatedError
                    raise WorkflowTerminatedError(
                        f"User terminated analysis at pathology '{step_label}' "
                        f"after LLM failure"
                    ) from e
                # SKIP (or any other action) → skip just this image
                logger.info(
                    f"[PathologyAgent] User chose SKIP for {image_label} "
                    f"(at '{step_label}')"
                )
                raise _ImageSkipRequested(image_label, last_error=e) from e

    async def _analyze_image(
        self,
        image_info: DataFile,
        instruction: str
    ) -> Dict[str, Any]:
        """
        Analyze a single pathology image

        Args:
            image_info: DataFile object (section=pathology)
            instruction: Analysis instruction

        Returns:
            Analysis results
        """
        self.refresh_runtime_config_if_needed()
        # Set LLM context before vision calls (client is shared between agents)
        self._set_llm_context()
        
        # Batch size for tile analysis — smaller batches are more reliable
        # and avoid vision API timeouts on large multimodal requests
        BATCH_SIZE = 5
        
        # Check if file exists
        if not os.path.exists(image_info.file_path):
            logger.error(f"[PathologyAgent] Image file not found: {image_info.file_path}")
            return {
                "image_id": image_info.id,
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
                tile_size=self.tile_size,
                max_tiles=self.max_tiles,
                preloaded_image=img
            )

            logger.info(f"[PathologyAgent] Generated {len(tiles)} tiles from image")

            if not tiles:
                return {
                    "image_id": image_info.id,
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
            
            overview_result = await self._call_llm_with_hitl(
                lambda: self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=overview_prompt,
                    images=[overview],
                    temperature=0.3,
                    request_timeout=600.0,  # 10 min for single overview image
                    system_prompt=self.system_prompt,
                ),
                image_label=image_info.filename,
                step_label="overview analysis",
            )
            
            # Parse overview result (handle None)
            overview_content = overview_result.content if overview_result else ""
            overview_data = self._safe_parse_json(overview_content, default={"raw_response": overview_content}, agent_label="PathologyAgent")
            logger.info(f"[PathologyAgent] Overview analysis complete")
            
            # Analyze tiles in batches
            tile_findings = []
            invalid_batches = []
            tiles_to_analyze = tiles[:self.max_tiles]
            num_batches = (len(tiles_to_analyze) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"[PathologyAgent] Analyzing {len(tiles_to_analyze)} tiles in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(tiles_to_analyze))
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
                    batch_result = await self._call_llm_with_hitl(
                        lambda bt=validation_prompt, imgs=batch_tiles, to=batch_timeout: self.llm_client.chat_with_vision(
                            model=self.model,
                            text_prompt=bt,
                            images=imgs,
                            temperature=0.3,
                            request_timeout=to,
                            system_prompt=self.system_prompt,
                        ),
                        image_label=image_info.filename,
                        step_label=f"tile batch {batch_idx + 1}/{num_batches}",
                    )

                    batch_content = batch_result.content if batch_result else ""
                    batch_data = _coerce_tile_batch(
                        self._safe_parse_json(
                            batch_content,
                            default={"raw_response": batch_content},
                            agent_label="PathologyAgent",
                        ),
                        batch_content,
                    )
                    normalized_batch, last_validation_errors = _validate_tile_batch(
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
            
            combined_result = await self._call_llm_with_hitl(
                lambda: self.chat_without_history(combined_prompt, temperature=0.3),
                image_label=image_info.filename,
                step_label="findings synthesis",
            )
            final_data = self._safe_parse_json(combined_result, default={"raw_response": combined_result}, agent_label="PathologyAgent")
            
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
            from ..workflow.helpers import WorkflowTerminatedError
            if isinstance(e, WorkflowTerminatedError):
                raise
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"[PathologyAgent] Failed to analyze image {image_info.id}: {error_detail}")
            return {
                "image_id": image_info.id if image_info else "unknown",
                "error": str(e),
                "findings": []
            }
    
    async def _synthesize_findings(
        self,
        instruction: str,
        results: List[Dict[str, Any]],
        state: AnalysisState
    ) -> Dict[str, Any]:
        """Synthesize findings from multiple images"""
        
        # Group by image type
        by_type = {}
        for result in results:
            img_type = result.get("image_type", "unknown")
            if img_type not in by_type:
                by_type[img_type] = []
            by_type[img_type].append(result)
        
        prompt = f"""
Based on the analysis results from various staining types, please provide a comprehensive assessment:

## Original Instruction
{instruction}

## Results by Staining Type
{json.dumps(by_type, ensure_ascii=False, indent=2)}

Please provide comprehensive analysis and reply in JSON format:
{{
    "summary": "Comprehensive pathological diagnosis summary",
    "key_findings": ["key finding 1", ...],
    "grading": {{
        "<dimension_name>": {{
            "score": <int>,
            "max": <int>,
            "scale_description": "<what the scale measures and what each level means>"
        }}
        // Define grading dimensions from the stated stain and the observations in
        // THIS dataset. Each dimension must name a visible, measurable feature and
        // explain its scale; do not import a disease- or tissue-specific checklist.
    }},
    "suggested_staining": ["suggested additional special staining"],
    "questions_for_bioinformatics": ["questions requiring bioinformatics assistance"],
    "consistency_with_literature": "consistency assessment with literature"
}}
"""
        
        response = await self.chat_without_history(prompt, temperature=0.3)
        return self._safe_parse_json(response, default={"raw_response": response}, agent_label="PathologyAgent")
    
    async def targeted_analysis(
        self,
        image_info: DataFile,
        specific_question: str
    ) -> Dict[str, Any]:
        """
        Perform targeted analysis based on specific question

        Used when PI requests specific pathology verification
        """
        try:
            self.refresh_runtime_config_if_needed()
            # Set LLM context before vision calls (client is shared between agents)
            self._set_llm_context()

            # Validate file exists
            if not os.path.exists(image_info.file_path):
                logger.warning(f"[PathologyAgent] Image not found: {image_info.file_path}")
                return {
                    "question": specific_question,
                    "answer": "Image file not found for targeted analysis",
                    "image_id": image_info.id,
                    "error": "file_not_found"
                }

            prompt = f"""
Perform pathology slide analysis for the following specific question:

Question: {specific_question}

Please carefully observe the slide and answer:
1. Whether relevant features are observed
2. Quantitative/semi-quantitative assessment (if applicable)
3. Specific morphological description
4. Other findings related to the question

Please reply in structured format.
"""

            # Process image
            tiles = process_pathology_image(
                image_info.file_path,
                tile_size=self.tile_size,
                max_tiles=10  # Fewer tiles for targeted analysis
            )

            if not tiles:
                return {
                    "question": specific_question,
                    "answer": "Image appears to be blank or too small for analysis",
                    "image_id": image_info.id
                }

            result = await self._call_llm_with_hitl(
                lambda: self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=prompt,
                    images=tiles,
                    temperature=0.3,
                    system_prompt=self.system_prompt,
                ),
                image_label=image_info.id,
                step_label="targeted",
            )

            return {
                "question": specific_question,
                "answer": result.content if result and result.content else "Analysis failed to generate a response",
                "image_id": image_info.id
            }
        except _ImageSkipRequested as skip:
            logger.warning(
                "[PathologyAgent] Targeted analysis for %s skipped by user",
                image_info.filename,
            )
            return {
                "question": specific_question,
                "answer": "Targeted pathology analysis skipped by user",
                "image_id": image_info.id,
                "status": "skipped",
                "skipped": True,
                "last_error": str(skip.last_error) if skip.last_error else "",
            }
        except Exception as e:
            from ..workflow.helpers import WorkflowTerminatedError
            if isinstance(e, WorkflowTerminatedError):
                raise
            logger.exception("[PathologyAgent] Targeted analysis failed")
            return {
                "question": specific_question,
                "answer": f"Targeted analysis failed: {str(e)}",
                "image_id": image_info.id,
                "error": str(e)
            }
