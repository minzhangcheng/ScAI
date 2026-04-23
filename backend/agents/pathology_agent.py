"""
Pathology Agent - Analyzes histology images
"""
from typing import Dict, Any, List, Optional
import json
import os
import re
import logging

from .base import BaseAgent

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, DataFile
)
from ..utils.llm_client import OpenAICompatibleClient
from ..utils.image_utils import (
    process_pathology_image, create_overview_image, load_image
)


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
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        tile_size: int = 1024,
        max_tiles: int = 50,
        config: Optional['AgentConfig'] = None
    ):
        from .base import AgentConfig
        super().__init__(
            agent_type=AgentType.PATHOLOGY,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config or AgentConfig()
        )
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process pathology analysis instruction"""
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
            except Exception as e:
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
        
        return {
            "status": "completed",
            "findings": all_findings,
            "detailed_results": detailed_results,
            "synthesis": synthesis,
            "suggested_staining": synthesis.get("suggested_staining", []),
            "images_analyzed": len(images_to_analyze)
        }
    
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
    "severity": "mild/moderate/severe/normal",
    "areas_of_interest": ["Area 1 requiring detailed observation", ...]
}}
"""
            
            overview_result = await self.llm_client.chat_with_vision(
                model=self.model,
                text_prompt=overview_prompt,
                images=[overview],
                temperature=0.3,
                request_timeout=180.0  # 3 min for single overview image
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
                batch_result = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=batch_prompt,
                    images=batch_tiles,
                    temperature=0.3,
                    request_timeout=batch_timeout
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
            
            combined_result = await self.chat(combined_prompt, temperature=0.3)
            final_data = self._safe_parse_json(combined_result, default={"raw_response": combined_result}, agent_label="PathologyAgent")
            
            logger.info(f"[PathologyAgent] Image analysis complete: {len(final_data.get('findings', []))} findings")
            
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
    "pathological_grade": {{
        "inflammation": 0-4,
        "fibrosis": 0-4,
        "steatosis": 0-4,
        "injury": 0-4
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
