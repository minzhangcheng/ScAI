"""
Biochemistry Agent - Handles biochemical experiment results
"""
from typing import Dict, Any, List, Optional
import json
import re
import os
import base64
import logging

from .base import BaseAgent
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, BiochemistryResult, BiochemistryFile
)
from ..utils.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class BiochemistryAgent(BaseAgent):
    """
    Biochemistry Agent
    
    Responsibilities:
    - Integrate user-uploaded biochemistry data
    - Interpret biochemical indicators
    - Suggest additional experiments
    - Correlate with pathology and bioinformatics findings
    """
    
    # Reference ranges for common biochemistry markers (examples - actual interpretation should consider context)
    # The agent should dynamically interpret results based on the project context and tissue type
    REFERENCE_RANGES = {
        # General metabolic markers
        "Glucose": {"min": 70, "max": 110, "unit": "mg/dL", "name": "Blood Glucose"},
        "Lactate": {"min": 0.5, "max": 2.2, "unit": "mmol/L", "name": "Lactate"},
        # Tissue function markers (context-dependent)
        "ALT": {"min": 20, "max": 60, "unit": "U/L", "name": "Alanine Aminotransferase"},
        "AST": {"min": 50, "max": 150, "unit": "U/L", "name": "Aspartate Aminotransferase"},
        "ALP": {"min": 40, "max": 150, "unit": "U/L", "name": "Alkaline Phosphatase"},
        # Oxidative stress markers
        "GSH": {"min": 5, "max": 15, "unit": "μmol/g tissue", "name": "Glutathione"},
        "SOD": {"min": 100, "max": 300, "unit": "U/mg protein", "name": "Superoxide Dismutase"},
        "MDA": {"min": 0.5, "max": 2.0, "unit": "nmol/mg protein", "name": "Malondialdehyde"},
        # Inflammatory markers
        "TNF-α": {"min": 0, "max": 50, "unit": "pg/mL", "name": "Tumor Necrosis Factor Alpha"},
        "IL-6": {"min": 0, "max": 10, "unit": "pg/mL", "name": "Interleukin-6"},
        "IL-1β": {"min": 0, "max": 20, "unit": "pg/mL", "name": "Interleukin-1 Beta"},
        # Note: Reference ranges vary by species, tissue, and experimental conditions
        # The LLM should interpret results in context of the specific project
    }
    
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
        """Process biochemistry analysis instruction"""
        
        # Get biochemistry results from state
        results = state.biochemistry_results
        files = state.biochemistry_files
        
        # First, process any uploaded files/images that haven't been extracted yet
        if files:
            for bio_file in files:
                if bio_file.extracted_data is None:
                    try:
                        extracted = await self._process_file(bio_file)
                        bio_file.extracted_data = extracted
                        
                        # Add extracted results to state
                        if extracted and "results" in extracted:
                            for r in extracted["results"]:
                                try:
                                    value = float(r.get("value", 0))
                                    # Validate the value is a real number
                                    if not (value == value and value not in (float('inf'), float('-inf'))):
                                        logger.warning(f"Invalid value for {r.get('indicator')}: {r.get('value')}, using 0")
                                        value = 0.0
                                    await self.add_result(
                                        indicator=r.get("indicator", "Unknown"),
                                        value=value,
                                        unit=r.get("unit", ""),
                                        state=state
                                    )
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Failed to convert value for {r.get('indicator')}: {r.get('value')} - {e}")
                    except Exception as e:
                        logger.error(f"Failed to process biochemistry file {bio_file.filename}: {e}")
                        bio_file.extracted_data = {"error": str(e)}
        
        # Re-get results after processing files
        results = state.biochemistry_results
        
        if not results and not files:
            return {
                "status": "waiting_input",
                "message": "Please enter biochemistry results (values table, image, or file)",
                "findings": [],
                "suggested_experiments": self._suggest_initial_experiments()
            }
        
        # Analyze all results
        analysis = await self._analyze_results(results, instruction)
        
        # Generate interpretation
        interpretation = await self._generate_interpretation(analysis, state)
        
        return {
            "status": "completed",
            "findings": analysis["findings"],
            "abnormal_markers": analysis["abnormal_markers"],
            "interpretation": interpretation,
            "suggested_experiments": analysis["suggested_experiments"],
            "processed_files": len([f for f in files if f.extracted_data]) if files else 0
        }
    
    async def _process_file(self, bio_file: BiochemistryFile) -> Dict[str, Any]:
        """Process an uploaded biochemistry file (image or data file)"""
        
        if bio_file.file_type == 'image':
            return await self._extract_from_image(bio_file)
        else:
            return await self._extract_from_data_file(bio_file)
    
    async def _extract_from_image(self, bio_file: BiochemistryFile) -> Dict[str, Any]:
        """Extract biochemistry data from an image using vision model"""

        if not os.path.exists(bio_file.file_path):
            return {"error": f"File not found: {bio_file.file_path}"}

        # Use image_utils to load and preprocess image (same as PathologyAgent)
        # This ensures the image is properly resized to avoid API issues with large images
        from ..utils.image_utils import load_image, create_overview_image

        try:
            # Load image with PIL and create a properly sized version
            img = load_image(bio_file.file_path)
            # Use create_overview_image to resize (max 1536px for better text readability) and encode to base64
            image_data = create_overview_image(img, max_size=1536)
            logger.info(f"[BiochemistryAgent] Preprocessed image {bio_file.filename}: original {img.size}, resized for API")
        except Exception as e:
            logger.error(f"[BiochemistryAgent] Failed to load/process image {bio_file.file_path}: {e}")
            return {"error": f"Image processing failed: {str(e)}", "results": []}

        # Set LLM context
        self._set_llm_context()
        
        prompt = """Analyze this biochemistry results image and extract ALL numerical values.

For EACH indicator found, extract:
1. Indicator name (standardize to common abbreviations like ALT, AST, TG, etc.)
2. Numerical value
3. Unit

Reply ONLY in JSON format:
{
    "results": [
        {"indicator": "ALT", "value": 45.2, "unit": "U/L"},
        {"indicator": "AST", "value": 89.5, "unit": "U/L"},
        ...
    ],
    "notes": "Any additional observations about the image"
}

Be thorough - extract EVERY numerical value visible in the image.
If a value has a range, extract the single measured value.
"""
        
        try:
            response = await self.llm_client.chat_with_vision(
                model=self.model,
                text_prompt=prompt,
                images=[image_data],
                temperature=0.1
            )
            
            # Parse JSON from response
            result = self._parse_json_response(response.content)
            logger.info(f"Extracted {len(result.get('results', []))} values from image {bio_file.filename}")
            return result
            
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return {"error": str(e), "results": []}
    
    async def _extract_from_data_file(self, bio_file: BiochemistryFile) -> Dict[str, Any]:
        """Extract biochemistry data from a data file (CSV, Excel, TXT)"""
        
        if not os.path.exists(bio_file.file_path):
            return {"error": f"File not found: {bio_file.file_path}"}
        
        filename_lower = bio_file.filename.lower()
        file_content = ""
        
        try:
            # Read file based on type
            if filename_lower.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(bio_file.file_path)
                file_content = df.to_string()
            elif filename_lower.endswith(('.xlsx', '.xls')):
                import pandas as pd
                df = pd.read_excel(bio_file.file_path)
                file_content = df.to_string()
            else:
                # Read as text
                with open(bio_file.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            
            # Use LLM to extract structured data
            return await self._parse_file_content(file_content, bio_file.filename)
            
        except Exception as e:
            logger.error(f"Failed to read file {bio_file.filename}: {e}")
            return {"error": str(e), "results": []}
    
    async def _parse_file_content(self, content: str, filename: str) -> Dict[str, Any]:
        """Parse file content using LLM to extract biochemistry values"""
        
        # Truncate if too long
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
    ) -> BiochemistryResult:
        """Add a single biochemistry result"""
        
        # Check if we have reference range
        ref_range = self.REFERENCE_RANGES.get(indicator.upper())
        
        interpretation = None
        if ref_range:
            if value < ref_range["min"]:
                interpretation = "below normal range"
            elif value > ref_range["max"]:
                interpretation = "above normal range"
            else:
                interpretation = "within normal range"
        
        result = BiochemistryResult(
            indicator=indicator,
            value=value,
            unit=unit,
            reference_range=f"{ref_range['min']}-{ref_range['max']} {ref_range['unit']}" if ref_range else None,
            interpretation=interpretation
        )
        
        state.biochemistry_results.append(result)
        return result
    
    async def parse_results_from_text(
        self,
        text: str,
        state: AnalysisState
    ) -> List[BiochemistryResult]:
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
        
        # Parse response
        results = []
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                for item in data.get("results", []):
                    # Safe access with .get() to avoid KeyError
                    indicator = item.get("indicator")
                    value = item.get("value")
                    unit = item.get("unit", "")

                    if indicator is None or value is None:
                        logger.warning(f"Skipping incomplete result item: {item}")
                        continue

                    try:
                        value = float(value)
                        if not (value == value and value not in (float('inf'), float('-inf'))):
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
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
        
        return results
    
    async def _analyze_results(
        self,
        results: List[BiochemistryResult],
        instruction: str
    ) -> Dict[str, Any]:
        """Analyze all biochemistry results"""
        
        # Build results summary
        results_text = []
        abnormal_markers = []
        
        for r in results:
            status = ""
            ref_info = self.REFERENCE_RANGES.get(r.indicator.upper(), {})
            
            if ref_info:
                if r.value < ref_info.get("min", 0):
                    status = "↓ Low"
                    abnormal_markers.append({
                        "indicator": r.indicator,
                        "value": r.value,
                        "status": "low",
                        "reference": f"{ref_info['min']}-{ref_info['max']}"
                    })
                elif r.value > ref_info.get("max", float('inf')):
                    status = "↑ High"
                    abnormal_markers.append({
                        "indicator": r.indicator,
                        "value": r.value,
                        "status": "high",
                        "reference": f"{ref_info['min']}-{ref_info['max']}"
                    })
                else:
                    status = "Normal"
            
            name = ref_info.get("name", r.indicator)
            results_text.append(f"- {name} ({r.indicator}): {r.value} {r.unit} {status}")
        
        prompt = f"""
Analyze the following mouse biochemistry indicators:

{chr(10).join(results_text)}

Current analysis objective: {instruction}

Please analyze:
1. Main abnormal indicators and their significance
2. Correlations between indicators
3. Possible pathophysiological mechanisms
4. Suggested supplementary experiments

Reply in JSON format:
{{
    "findings": ["finding 1", "finding 2", ...],
    "mechanisms": ["possible mechanism 1", ...],
    "suggested_experiments": ["experiment 1", ...]
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use the safer JSON extraction
        data = self._parse_json_response(response)

        return {
            "findings": data.get("findings", []),
            "abnormal_markers": abnormal_markers,
            "mechanisms": data.get("mechanisms", []),
            "suggested_experiments": data.get("suggested_experiments", [])
        }
    
    async def _generate_interpretation(
        self,
        analysis: Dict[str, Any],
        state: AnalysisState
    ) -> str:
        """Generate comprehensive interpretation of biochemistry results"""
        
        # Get context from other analyses
        context = ""
        if state.round_results:
            last_round = state.round_results[-1]
            if last_round.pathology_findings:
                context += f"\nPathology findings: {', '.join(last_round.pathology_findings)}"
            if last_round.bioinformatics_findings:
                context += f"\nBioinformatics findings: {', '.join(last_round.bioinformatics_findings)}"
        
        prompt = f"""
Based on biochemistry analysis results and other analysis context, generate a comprehensive interpretation:

## Biochemistry Findings
{json.dumps(analysis['findings'], ensure_ascii=False, indent=2)}

## Abnormal Markers
{json.dumps(analysis['abnormal_markers'], ensure_ascii=False, indent=2)}

## Other Analysis Context
{context if context else "None available"}

Please generate a concise comprehensive interpretation (2-3 paragraphs), focusing on:
1. Tissue/organ function status reflected by biochemistry indicators
2. Consistency with pathology/bioinformatics findings
3. Contribution to overall phenotype
"""
        
        return await self.chat(prompt, temperature=0.3)
    
    def _suggest_initial_experiments(self) -> List[str]:
        """Suggest initial experiments when no data is available - generic suggestions"""
        return [
            "Basic tissue function tests relevant to the organ/tissue being studied",
            "Metabolic indicators: glucose, lactate, lipid profiles",
            "Oxidative stress markers: GSH, SOD, MDA, ROS levels",
            "Inflammatory markers: IL-6, TNF-α, IL-1β",
            "Cell death markers: LDH, caspase activity (if applicable)",
            "Tissue-specific functional markers based on the research context"
        ]
    
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
             for r in state.biochemistry_results], ensure_ascii=False)}

Please analyze the consistency between pathology and biochemistry findings:
1. Which findings corroborate each other
2. Which findings are contradictory
3. How to explain these correlations or contradictions

Reply in JSON format.
"""
        
        response = await self.chat(prompt, temperature=0.3)
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response using non-greedy extraction"""
        # Handle None or empty response
        if not response:
            return {"raw_response": "", "error": "Empty response from LLM"}

        try:
            # Try to find balanced JSON object (handles nested objects)
            # Find the first { and then count brackets to find matching }
            start = response.find('{')
            if start == -1:
                return {"raw_response": response}

            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if depth == 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
        return {"raw_response": response}
