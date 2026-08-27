"""
Pathology Agent - Analyzes histology images
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
import json
import logging
import os
import re
import sys

from .pathology_parts import image_analysis

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
                    "image_type": getattr(img_info, "data_type", "") or "",
                    "status": "skipped",
                    "tile_validation": {"status": "skipped"},
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
                    "image_type": getattr(img_info, "data_type", "") or "",
                    "status": "failed",
                    "tile_validation": {"status": "failed"},
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
        """Analyze a single pathology image."""
        return await image_analysis.analyze_image(
            self,
            image_info,
            instruction,
            bindings=sys.modules[__name__],
        )

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
