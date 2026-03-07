"""
Biochemistry Agent - Handles biochemical experiment results

Three-step analysis pipeline:
  Step 1: Process structured indicators and data files (skip if none)
  Step 2: Process images with full visual analysis (skip if none)
  Step 3: Summarize findings from steps 1 & 2, provide conclusions and experiment suggestions
"""
from typing import Dict, Any, List, Optional
import json
import math
import re
import os
import logging

from .base import BaseAgent
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, BiochemistryIndicator, DataFile
)
from ..utils.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class BiochemistryAgent(BaseAgent):
    """
    Biochemistry Agent

    Responsibilities:
    - Integrate user-uploaded biochemistry data
    - Interpret biochemical indicators dynamically based on project context
    - Suggest additional experiments
    - Correlate with pathology and bioinformatics findings

    Note: This agent does NOT use hardcoded reference ranges. Instead, it relies on
    the LLM to dynamically interpret biochemistry indicators based on the specific
    research context (species, tissue type, experimental conditions, etc.).
    """

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        config: Optional['AgentConfig'] = None
    ):
        from .base import AgentConfig
        super().__init__(
            agent_type=AgentType.BIOCHEMISTRY,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config or AgentConfig()
        )

    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process biochemistry analysis in three steps:

        Step 1: Process structured indicators + data files (skip if none)
        Step 2: Analyze images with vision model (skip if none)
        Step 3: Summarize results, provide conclusions and experiment suggestions
        """

        bio_files = state.get_files_by_section("biochemistry")

        # Separate image files from data files
        image_files = [f for f in bio_files if f.data_type == 'image']
        data_files = [f for f in bio_files if f.data_type != 'image']

        # ====================================================================
        # Step 1: Process structured indicators and data files
        # ====================================================================
        step1_result = await self._step1_process_indicators(
            state, data_files, instruction
        )

        # ====================================================================
        # Step 2: Process images with full visual analysis
        # ====================================================================
        step2_result = await self._step2_process_images(
            image_files, instruction
        )

        # ====================================================================
        # Step 3: Summarize and provide conclusions + experiment suggestions
        # ====================================================================
        current_round_context = kwargs.get("current_round_context")
        final_result = await self._step3_summarize(
            step1_result, step2_result, instruction, state,
            current_round_context=current_round_context
        )

        return final_result

    # ------------------------------------------------------------------
    # Step 1: Process indicators
    # ------------------------------------------------------------------

    async def _step1_process_indicators(
        self,
        state: AnalysisState,
        data_files: List[DataFile],
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        """Step 1: Process structured indicators and data files.

        - Extract values from unprocessed data files (CSV/Excel/TXT)
        - Analyze all structured indicators
        - Returns None if there is nothing to analyze
        """

        # Extract data from unprocessed data files first
        for bio_file in data_files:
            if bio_file.extracted_data is None:
                try:
                    extracted = await self._extract_from_data_file(bio_file)
                    bio_file.extracted_data = extracted

                    if extracted and "results" in extracted:
                        for r in extracted["results"]:
                            try:
                                value = float(r.get("value", 0))
                                if not math.isfinite(value):
                                    logger.warning(
                                        f"Invalid value for {r.get('indicator')}: {r.get('value')}, using 0"
                                    )
                                    value = 0.0
                                await self.add_result(
                                    indicator=r.get("indicator", "Unknown"),
                                    value=value,
                                    unit=r.get("unit", ""),
                                    state=state,
                                )
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Failed to convert value for {r.get('indicator')}: "
                                    f"{r.get('value')} - {e}"
                                )
                except Exception as e:
                    logger.error(f"Failed to process data file {bio_file.filename}: {e}")
                    bio_file.extracted_data = {"error": str(e)}

        # Check if we have any indicators to analyze
        results = state.biochemistry_indicators
        if not results:
            logger.info("[BiochemistryAgent] Step 1: No indicators to analyze, skipping")
            return None

        logger.info(f"[BiochemistryAgent] Step 1: Analyzing {len(results)} indicators")

        # Build results summary
        results_text = []
        for r in results:
            results_text.append(f"- {r.indicator}: {r.value} {r.unit}")

        prompt = f"""Analyze the following biochemistry indicators in the context of the research objective.

## Biochemistry Results:
{chr(10).join(results_text)}

## Research Context / Analysis Objective:
{instruction}

**IMPORTANT**: You must determine which indicators are abnormal based on:
1. The specific research context (species, tissue type, experimental model)
2. Standard reference ranges for the relevant species/tissue
3. The expected changes given the experimental conditions
4. Relative changes between treatment and control groups (if applicable)

Please analyze and provide:
1. Which indicators are abnormal (high or low) and their significance
2. Correlations between indicators
3. Possible pathophysiological mechanisms
4. Suggested supplementary experiments

Reply in JSON format:
{{
    "abnormal_markers": [
        {{"indicator": "name", "value": number, "status": "high/low/normal", "significance": "brief explanation"}},
        ...
    ],
    "findings": ["finding 1", "finding 2", ...],
    "mechanisms": ["possible mechanism 1", ...],
    "suggested_experiments": ["experiment 1", ...]
}}
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._parse_json_response(response)

        return {
            "findings": data.get("findings", []),
            "abnormal_markers": data.get("abnormal_markers", []),
            "mechanisms": data.get("mechanisms", []),
            "suggested_experiments": data.get("suggested_experiments", []),
        }

    # ------------------------------------------------------------------
    # Step 2: Process images
    # ------------------------------------------------------------------

    async def _step2_process_images(
        self,
        image_files: List[DataFile],
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        """Step 2: Analyze images with full visual analysis.

        Instead of extracting only JSON numbers, performs a comprehensive
        visual analysis of each image (chart type, trends, values, significance).
        Returns None if there are no images to analyze.
        """

        if not image_files:
            logger.info("[BiochemistryAgent] Step 2: No images to analyze, skipping")
            return None

        logger.info(
            f"[BiochemistryAgent] Step 2: Analyzing {len(image_files)} image(s)"
        )

        image_analyses: List[str] = []

        for bio_file in image_files:
            analysis = await self._analyze_single_image(bio_file, instruction)
            if analysis:
                image_analyses.append(
                    f"### Image: {bio_file.filename}\n{analysis}"
                )

        if not image_analyses:
            logger.warning(
                "[BiochemistryAgent] Step 2: All image analyses returned empty"
            )
            return None

        # Consolidate all image analyses into structured findings
        combined_text = "\n\n".join(image_analyses)

        prompt = f"""Based on the following visual analysis of biochemistry experiment images,
please consolidate the findings into a structured summary.

## Image Analysis Results:
{combined_text}

## Research Context:
{instruction}

Synthesize the image analysis into a structured JSON response:
{{
    "image_findings": ["key finding from images 1", "key finding 2", ...],
    "observed_values": [
        {{"indicator": "name", "value_description": "description of observed value or trend", "significance": "explanation"}},
        ...
    ],
    "image_mechanisms": ["mechanism suggested by image data 1", ...],
    "image_suggested_experiments": ["suggested experiment based on image findings 1", ...]
}}

Focus on:
1. Key differences between experimental groups visible in the images
2. Statistical significance markers if present
3. Trends and patterns across multiple images
4. Any abnormal or noteworthy observations
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._parse_json_response(response)

        return {
            "image_findings": data.get("image_findings", []),
            "observed_values": data.get("observed_values", []),
            "image_mechanisms": data.get("image_mechanisms", []),
            "image_suggested_experiments": data.get("image_suggested_experiments", []),
            "raw_analyses": combined_text,
        }

    async def _analyze_single_image(
        self, bio_file: DataFile, instruction: str
    ) -> Optional[str]:
        """Analyze a single biochemistry image using vision model.

        Performs a full visual analysis rather than strict JSON extraction,
        which is more robust against vision API format issues.
        """

        if not os.path.exists(bio_file.file_path):
            logger.warning(f"[BiochemistryAgent] Image file not found: {bio_file.file_path}")
            return None

        from ..utils.image_utils import load_image, create_overview_image

        try:
            img = load_image(bio_file.file_path)
            image_data = create_overview_image(img, max_size=1536)
            logger.info(
                f"[BiochemistryAgent] Preprocessed image {bio_file.filename}: "
                f"original {img.size}, resized for API"
            )
        except Exception as e:
            logger.error(
                f"[BiochemistryAgent] Failed to load/process image "
                f"{bio_file.file_path}: {e}"
            )
            return None

        self._set_llm_context()

        prompt = f"""You are a biochemistry expert. Carefully analyze this experimental result image.

## Research Context:
{instruction}

## Analysis Requirements:
1. **Image type**: Describe the type of visualization (bar chart, western blot, table, gel image, scatter plot, etc.)
2. **Experimental groups**: Identify ALL experimental groups and conditions shown
3. **Measured indicators**: List all biochemistry indicators / measurements shown
4. **Specific values**: Extract ALL specific numerical values visible (indicator name, value, unit)
5. **Statistical significance**: Note any significance markers (*, **, ***, p-values, error bars)
6. **Group comparisons**: Describe key differences and trends between groups
7. **Expert interpretation**: Provide your interpretation of what these results mean

Be thorough and extract EVERY piece of information visible in the image.
If you see bar charts, describe the height/value of each bar for each group.
If you see tables, extract all rows and columns.
"""

        # Try vision analysis with retry
        for attempt in range(2):
            try:
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=prompt,
                    images=[image_data],
                    temperature=0.2,
                    max_tokens=32768,
                )

                content = response.content if response else ""

                if content and content.strip():
                    logger.info(
                        f"[BiochemistryAgent] Successfully analyzed image "
                        f"{bio_file.filename} (attempt {attempt + 1})"
                    )
                    bio_file.extracted_data = {
                        "analysis": content,
                        "type": "visual_analysis",
                    }
                    return content

                logger.warning(
                    f"[BiochemistryAgent] Vision returned empty for "
                    f"{bio_file.filename} (attempt {attempt + 1})"
                )

            except Exception as e:
                logger.error(
                    f"[BiochemistryAgent] Vision analysis failed for "
                    f"{bio_file.filename} (attempt {attempt + 1}): {e}"
                )

        # All attempts failed
        bio_file.extracted_data = {"error": "Vision API returned empty content after retries"}
        logger.error(
            f"[BiochemistryAgent] All vision attempts failed for {bio_file.filename}"
        )
        return None

    # ------------------------------------------------------------------
    # Step 3: Summarize and conclude
    # ------------------------------------------------------------------

    async def _step3_summarize(
        self,
        step1_result: Optional[Dict[str, Any]],
        step2_result: Optional[Dict[str, Any]],
        instruction: str,
        state: AnalysisState,
        current_round_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Step 3: Summarize results from steps 1 & 2, provide conclusions
        and experiment suggestions.

        - If both steps produced results: combine and synthesize
        - If only one produced results: summarize that
        - If neither produced results: provide general analysis and suggestions
        """

        has_indicators = step1_result is not None
        has_images = step2_result is not None

        # Get cross-agent context (prefer current round data over previous round)
        other_context = self._get_other_agent_context(state, current_round_context)

        # ----- Case 1: Neither step produced results -----
        if not has_indicators and not has_images:
            logger.info(
                "[BiochemistryAgent] Step 3: No data from steps 1 & 2, "
                "providing general analysis"
            )
            return await self._general_analysis_no_data(instruction, other_context, state)

        # ----- Build summary prompt based on available data -----
        sections = []

        if has_indicators:
            sections.append(f"""## Indicator Analysis Results (Step 1):
- Findings: {json.dumps(step1_result.get('findings', []), ensure_ascii=False)}
- Abnormal markers: {json.dumps(step1_result.get('abnormal_markers', []), ensure_ascii=False)}
- Possible mechanisms: {json.dumps(step1_result.get('mechanisms', []), ensure_ascii=False)}
- Suggested experiments: {json.dumps(step1_result.get('suggested_experiments', []), ensure_ascii=False)}""")

        if has_images:
            sections.append(f"""## Image Analysis Results (Step 2):
- Image findings: {json.dumps(step2_result.get('image_findings', []), ensure_ascii=False)}
- Observed values: {json.dumps(step2_result.get('observed_values', []), ensure_ascii=False)}
- Image-based mechanisms: {json.dumps(step2_result.get('image_mechanisms', []), ensure_ascii=False)}
- Image-based suggested experiments: {json.dumps(step2_result.get('image_suggested_experiments', []), ensure_ascii=False)}""")

        data_sections = "\n\n".join(sections)

        prompt = f"""You are a biochemistry expert. Synthesize ALL available analysis results
into a comprehensive conclusion.

{data_sections}

## Research Context:
{instruction}

## Other Agent Context:
{other_context if other_context else "None available"}

Please provide a comprehensive summary in JSON format:
{{
    "findings": ["integrated finding 1", "integrated finding 2", ...],
    "abnormal_markers": [
        {{"indicator": "name", "value": "value or description", "status": "high/low/normal/abnormal", "significance": "explanation"}},
        ...
    ],
    "interpretation": "2-3 paragraph comprehensive interpretation covering: (1) organ/tissue function status, (2) consistency with other findings, (3) contribution to overall phenotype",
    "suggested_experiments": ["prioritized experiment suggestion 1", ...],
    "conclusions": ["key conclusion 1", "key conclusion 2", ...]
}}

Requirements:
1. Integrate findings from ALL available data sources (indicators and/or images)
2. Highlight correlations and consistencies between different data types
3. Identify any contradictions or gaps
4. Prioritize experiment suggestions by importance
5. Ensure conclusions are evidence-based and actionable
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._parse_json_response(response)

        # Build the final result
        all_findings = data.get("findings", [])
        all_markers = data.get("abnormal_markers", [])
        all_experiments = data.get("suggested_experiments", [])
        interpretation = data.get("interpretation", "")

        # Count processed files
        all_files = state.get_files_by_section("biochemistry")
        processed_count = len([f for f in all_files if f.extracted_data])

        return {
            "status": "completed",
            "findings": all_findings,
            "abnormal_markers": all_markers,
            "interpretation": interpretation,
            "suggested_experiments": all_experiments,
            "conclusions": data.get("conclusions", []),
            "processed_files": processed_count,
            "steps_completed": {
                "indicators_analyzed": has_indicators,
                "images_analyzed": has_images,
            },
        }

    async def _general_analysis_no_data(
        self,
        instruction: str,
        other_context: str,
        state: AnalysisState,
    ) -> Dict[str, Any]:
        """When no indicators or images are available, provide general analysis
        and experiment suggestions based on the research context."""

        prompt = f"""You are a biochemistry expert. No biochemistry experimental data
(neither indicators nor images) has been provided yet.

## Research Context:
{instruction}

## Other Agent Context:
{other_context if other_context else "None available"}

Based on the research context and any available findings from other agents,
please provide:
1. What biochemistry experiments SHOULD be performed for this research
2. Which specific indicators are most relevant and why
3. Recommended experimental design (groups, sample sizes, time points)
4. Priority ranking of suggested experiments

Reply in JSON format:
{{
    "findings": ["contextual observation 1", ...],
    "suggested_experiments": [
        "highest priority experiment with specific indicators",
        ...
    ],
    "recommended_indicators": [
        {{"indicator": "name", "reason": "why this indicator is relevant"}},
        ...
    ],
    "experimental_design_notes": "brief notes on recommended experimental design"
}}
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._parse_json_response(response)

        return {
            "status": "no_data",
            "message": "No biochemistry data provided. General analysis and recommendations generated based on research context.",
            "findings": data.get("findings", []),
            "abnormal_markers": [],
            "interpretation": data.get("experimental_design_notes", ""),
            "suggested_experiments": data.get("suggested_experiments", self._suggest_initial_experiments()),
            "recommended_indicators": data.get("recommended_indicators", []),
            "conclusions": [],
            "processed_files": 0,
            "steps_completed": {
                "indicators_analyzed": False,
                "images_analyzed": False,
            },
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_other_agent_context(
        self,
        state: AnalysisState,
        current_round_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Gather context from other agents' results.

        Prefers current round findings (passed from workflow) over previous
        round results stored in state.round_results.
        """
        context_parts = []
        has_path_context = False
        has_bio_context = False

        # Prefer current round context when available
        if current_round_context:
            path_findings = current_round_context.get("pathology_findings", [])
            bio_findings = current_round_context.get("bioinformatics_findings", [])
            if path_findings:
                context_parts.append(
                    f"Pathology findings (current round): {', '.join(path_findings)}"
                )
                has_path_context = True
            if bio_findings:
                context_parts.append(
                    f"Bioinformatics findings (current round): {', '.join(bio_findings)}"
                )
                has_bio_context = True

        # Supplement missing fields from previous round (not just when context_parts is empty)
        if state.round_results:
            last_round = state.round_results[-1]
            if not has_path_context and last_round.pathology_findings:
                context_parts.append(
                    f"Pathology findings (previous round): {', '.join(last_round.pathology_findings)}"
                )
            if not has_bio_context and last_round.bioinformatics_findings:
                context_parts.append(
                    f"Bioinformatics findings (previous round): {', '.join(last_round.bioinformatics_findings)}"
                )

        return "\n".join(context_parts)

    async def _extract_from_data_file(self, bio_file: DataFile) -> Dict[str, Any]:
        """Extract biochemistry data from a data file (CSV, Excel, TXT)"""

        if not os.path.exists(bio_file.file_path):
            return {"error": f"File not found: {bio_file.file_path}"}

        filename_lower = bio_file.filename.lower()
        file_content = ""

        try:
            if filename_lower.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(bio_file.file_path)
                file_content = df.to_string()
            elif filename_lower.endswith(('.xlsx', '.xls')):
                import pandas as pd
                df = pd.read_excel(bio_file.file_path)
                file_content = df.to_string()
            else:
                with open(bio_file.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()

            return await self._parse_file_content(file_content, bio_file.filename)

        except Exception as e:
            logger.error(f"Failed to read file {bio_file.filename}: {e}")
            return {"error": str(e), "results": []}

    async def _parse_file_content(self, content: str, filename: str) -> Dict[str, Any]:
        """Parse file content using LLM to extract biochemistry values"""

        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated)"

        prompt = f"""Analyze this biochemistry data file and extract ALL numerical values.

Filename: {filename}

Content:
{content}

For EACH biochemistry indicator found, extract:
1. Indicator name (standardize to common abbreviations like ALT, AST, TG, etc.)
2. Numerical value
3. Unit

Reply ONLY in JSON format:
{{
    "results": [
        {{"indicator": "ALT", "value": 45.2, "unit": "U/L"}},
        {{"indicator": "AST", "value": 89.5, "unit": "U/L"}},
        ...
    ],
    "notes": "Any additional observations"
}}

Extract EVERY biochemistry-related numerical value. Ignore metadata like dates or sample IDs.
"""

        response = await self.chat(prompt, temperature=0.1)
        result = self._parse_json_response(response)
        logger.info(f"Extracted {len(result.get('results', []))} values from file {filename}")
        return result

    async def add_result(
        self,
        indicator: str,
        value: float,
        unit: str,
        state: AnalysisState
    ) -> BiochemistryIndicator:
        """Add a single biochemistry indicator.

        Note: Interpretation is NOT done here with hardcoded reference ranges.
        Instead, the LLM will interpret all results contextually in step 1.
        """
        result = BiochemistryIndicator(
            indicator=indicator,
            value=value,
            unit=unit,
            reference_range=None,
            interpretation=None,
        )

        state.biochemistry_indicators.append(result)
        return result

    async def parse_results_from_text(
        self,
        text: str,
        state: AnalysisState
    ) -> List[BiochemistryIndicator]:
        """Parse biochemistry results from free-form text input"""

        prompt = f"""
Please extract biochemistry indicator data from the following text:

{text}

Please return in JSON format as follows:
{{
    "results": [
        {{"indicator": "indicator name", "value": number, "unit": "unit"}},
        ...
    ]
}}

Notes:
1. Identify all mentioned biochemistry indicators
2. Convert values to floating point numbers
3. Standardize unit names
"""

        response = await self.chat(prompt, temperature=0.2)

        results = []
        try:
            data = self._safe_parse_json(response)
            if data:
                for item in data.get("results", []):
                    indicator = item.get("indicator")
                    value = item.get("value")
                    unit = item.get("unit", "")

                    if indicator is None or value is None:
                        logger.warning(f"Skipping incomplete result item: {item}")
                        continue

                    try:
                        value = float(value)
                        if not math.isfinite(value):
                            logger.warning(f"Invalid value for {indicator}: {item.get('value')}, skipping")
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert value for {indicator}: {value} - {e}")
                        continue

                    result = await self.add_result(
                        indicator,
                        value,
                        unit,
                        state
                    )
                    results.append(result)
        except Exception as e:
            logger.warning(f"Failed to parse results from response: {e}")

        return results

    async def correlate_with_pathology(
        self,
        pathology_findings: List[str],
        state: AnalysisState
    ) -> Dict[str, Any]:
        """Correlate biochemistry results with pathology findings"""

        prompt = f"""
Pathology findings:
{json.dumps(pathology_findings, ensure_ascii=False)}

Biochemistry results:
{json.dumps([{"indicator": r.indicator, "value": r.value, "interpretation": r.interpretation}
             for r in state.biochemistry_indicators], ensure_ascii=False)}

Please analyze the consistency between pathology and biochemistry findings:
1. Which findings corroborate each other
2. Which findings are contradictory
3. How to explain these correlations or contradictions

Reply in JSON format.
"""

        response = await self.chat(prompt, temperature=0.3)
        return self._parse_json_response(response)

    def _suggest_initial_experiments(self) -> List[str]:
        """Suggest initial experiments when no data is available - generic suggestions"""
        return [
            "Basic tissue function tests relevant to the organ/tissue being studied",
            "Metabolic indicators: glucose, lactate, lipid profiles",
            "Oxidative stress markers: GSH, SOD, MDA, ROS levels",
            "Inflammatory markers: IL-6, TNF-alpha, IL-1beta",
            "Cell death markers: LDH, caspase activity (if applicable)",
            "Tissue-specific functional markers based on the research context",
        ]

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response using safe balanced bracket matching from base class"""
        data = self._safe_parse_json(response)
        if "raw_response" in data:
            logger.warning(
                f"[BiochemistryAgent] JSON parse failed, raw response: {response[:200]}..."
            )
        return data
