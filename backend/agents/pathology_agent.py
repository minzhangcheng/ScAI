"""
Pathology Agent - Analyzes histology images
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
import json
import logging
import os

from .base import BaseAgent
from ..constants import LlmFailureAction
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, DataFile
)
from ..utils.image_utils import (
    process_pathology_image, create_overview_image, load_image
)
from ..utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ..utils.settings_provider import AgentSettingsProvider

logger = logging.getLogger(__name__)


def _summarize_status(detailed_results, all_findings) -> str:
    """'error' if every image failed (no findings and all carry an error);
    otherwise 'completed'."""
    if not detailed_results:
        return "error"
    any_success = any(not r.get("error") and r.get("findings") for r in detailed_results)
    if not any_success and not all_findings:
        return "error"
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
        
        # Synthesize findings if we have results
        if detailed_results:
            try:
                synthesis = await self._synthesize_findings(
                    instruction, detailed_results, state
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
                    request_timeout=180.0  # 3 min for single overview image
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
            tiles_to_analyze = tiles[:self.max_tiles]
            num_batches = (len(tiles_to_analyze) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"[PathologyAgent] Analyzing {len(tiles_to_analyze)} tiles in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(tiles_to_analyze))
                batch_tiles = tiles_to_analyze[start_idx:end_idx]
                
                logger.info(f"[PathologyAgent] Processing batch {batch_idx + 1}/{num_batches} ({len(batch_tiles)} tiles)")
                
                # Create batch prompt
                batch_prompt = f"""
You are analyzing {len(batch_tiles)} regions (tiles {start_idx + 1} to {end_idx}) from a {image_info.data_type} stained tissue section.
Total tiles in this section: {len(tiles_to_analyze)}

For EACH tile (in order), identify pathological features based on the tissue type and staining:
1. Cell morphology changes (size, shape, nuclear abnormalities)
2. Inflammatory cell infiltration
3. Signs of fibrosis or tissue remodeling
4. Vascular or structural changes
5. Other abnormalities specific to this tissue type

Reply in JSON format with an array of findings for each tile:
{{
    "tile_findings": [
        {{"tile_index": {start_idx}, "findings": "Tile {start_idx + 1} findings..."}},
        {{"tile_index": {start_idx + 1}, "findings": "Tile {start_idx + 2} findings..."}},
        ...
    ]
}}

Be concise but thorough for each tile.
"""
                
                # Dynamic timeout: 120s base + 30s per image
                batch_timeout = 120.0 + 30.0 * len(batch_tiles)
                batch_result = await self._call_llm_with_hitl(
                    lambda bt=batch_prompt, imgs=batch_tiles, to=batch_timeout: self.llm_client.chat_with_vision(
                        model=self.model,
                        text_prompt=bt,
                        images=imgs,
                        temperature=0.3,
                        request_timeout=to
                    ),
                    image_label=image_info.filename,
                    step_label=f"tile batch {batch_idx + 1}/{num_batches}",
                )
                
                # Parse batch results (handle None)
                batch_content = batch_result.content if batch_result else ""
                try:
                    batch_data = self._safe_parse_json(batch_content, default={"raw_response": batch_content}, agent_label="PathologyAgent")
                    if "tile_findings" in batch_data:
                        tile_findings.extend(batch_data["tile_findings"])
                    else:
                        # Fallback: treat entire response as one finding
                        tile_findings.append({
                            "tile_index": start_idx,
                            "findings": batch_content or "No findings"
                        })
                except Exception as e:
                    # If parsing fails, log warning and store error info
                    logger.warning(f"[PathologyAgent] Failed to parse batch {batch_idx} response: {e}")
                    tile_findings.append({
                        "tile_index": start_idx,
                        "findings": batch_content[:500] if batch_content else "Analysis failed",
                        "parse_error": str(e),
                        "is_error": True
                    })
            
            logger.info(f"[PathologyAgent] All batches complete, synthesizing findings...")
            
            # Combine findings
            combined_prompt = f"""
Based on the following regional analysis results, please summarize the pathological findings for the entire section:

## Overview Analysis
{json.dumps(overview_data, ensure_ascii=False, indent=2)}

## Regional Analysis
{json.dumps(tile_findings, ensure_ascii=False, indent=2)}

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
                lambda: self.chat(combined_prompt, temperature=0.3),
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
                "tile_count": len(tiles),
                "findings": final_data.get("findings", []),
                "severity_score": final_data.get("severity_score", 0),
                "further_analysis": final_data.get("further_analysis", []),
                "suggested_staining": final_data.get("suggested_staining", [])
            }
            
        except Exception as e:
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
        // Define grading dimensions appropriate for THIS tissue type and the
        // pathology observed. Examples (DO NOT use as-is — derive from data):
        //   liver/NASH:   inflammation, fibrosis, steatosis, hepatocellular injury
        //   brain/AD:     amyloid burden, tau pathology, neuronal loss
        //   tumor:        tumor grade, mitotic count, necrosis extent, invasion depth
        //   kidney/AKI:   tubular injury, interstitial inflammation, fibrosis
        //   lung/IPF:     UIP pattern severity, fibroblastic foci density, honeycombing
    }},
    "suggested_staining": ["suggested additional special staining"],
    "questions_for_bioinformatics": ["questions requiring bioinformatics assistance"],
    "consistency_with_literature": "consistency assessment with literature"
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)
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

            result = await self.llm_client.chat_with_vision(
                model=self.model,
                text_prompt=prompt,
                images=tiles,
                temperature=0.3
            )

            return {
                "question": specific_question,
                "answer": result.content if result and result.content else "Analysis failed to generate a response",
                "image_id": image_info.id
            }
        except Exception as e:
            logger.exception("[PathologyAgent] Targeted analysis failed")
            return {
                "question": specific_question,
                "answer": f"Targeted analysis failed: {str(e)}",
                "image_id": image_info.id,
                "error": str(e)
            }
