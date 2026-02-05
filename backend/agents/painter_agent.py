"""
Painter Agent - Generates graphic abstracts using Google Gemini image generation
"""
from typing import Dict, Any, Optional
import httpx
import logging
import uuid
from datetime import datetime

from ..models.schemas import AgentType, AnalysisState, GraphicAbstract

logger = logging.getLogger(__name__)


class PainterAgent:
    """
    Painter Agent using Google Gemini's image generation capabilities.

    Responsibilities:
    - Generate graphic abstracts summarizing research findings
    - Create visual representations of key discoveries

    Note: This agent does not inherit from BaseAgent as it uses a different
    API (Google's image generation) instead of the standard LLM client.
    """

    def __init__(
        self,
        google_api_key: str,
        model: str = "gemini-3-pro-image-preview",
        proxy_url: Optional[str] = None,
        session_id: str = "default"
    ):
        """
        Initialize the Painter Agent.

        Args:
            google_api_key: Google API key for Gemini
            model: Model to use for image generation
            proxy_url: Optional HTTP/HTTPS proxy URL
            session_id: Session ID for logging
        """
        self.google_api_key = google_api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.session_id = session_id
        self.agent_type = AgentType.PAINTER
        self.proxy_url = proxy_url

        # Lazy-initialized HTTP client to avoid resource leaks
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(f"[Painter] Initialized with model={model}")

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialized HTTP client with proxy support"""
        if self._client is None:
            transport = None
            if self.proxy_url:
                transport = httpx.AsyncHTTPTransport(proxy=self.proxy_url)
            self._client = httpx.AsyncClient(
                transport=transport,
                timeout=180.0  # Image generation can take time
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def set_session(self, session_id: str):
        """Update session ID for logging"""
        self.session_id = session_id

    async def generate_graphic_abstract(
        self,
        state: AnalysisState,
        summary: str
    ) -> GraphicAbstract:
        """
        Generate a graphic abstract based on analysis findings.

        Args:
            state: Analysis state with all findings
            summary: PI's summary prompt for the graphic abstract

        Returns:
            GraphicAbstract with generated image
        """
        logger.info(f"[{self.session_id}][Painter] Generating graphic abstract")

        # Build comprehensive prompt for image generation
        prompt = self._build_graphic_prompt(state, summary)

        # Call Gemini image generation API
        image_base64 = await self._generate_image(prompt)

        graphic_abstract = GraphicAbstract(
            prompt=prompt,
            image_base64=image_base64
        )

        logger.info(f"[{self.session_id}][Painter] Graphic abstract generated successfully")
        return graphic_abstract

    def _build_graphic_prompt(self, state: AnalysisState, summary: str) -> str:
        """
        Build a detailed prompt for graphic abstract generation.

        Args:
            state: Analysis state
            summary: PI's summary of findings

        Returns:
            Formatted prompt string
        """
        # Extract key findings from all rounds
        findings = []
        for round_result in state.round_results:
            findings.extend(round_result.bioinformatics_findings[:3])
            findings.extend(round_result.pathology_findings[:2])

        findings_text = "\n".join(f"- {f}" for f in findings[:8]) if findings else "No specific findings available"

        prompt = f"""Create a professional scientific graphic abstract for a biomedical research paper.

Research Topic: {state.name}
Research Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research analysis'}

Key Findings Summary:
{summary}

Main Discoveries:
{findings_text}

Style Requirements:
- Professional scientific illustration style suitable for journal publication
- Clean, modern design with clear visual hierarchy
- Use appropriate scientific symbols and cell/tissue representations
- Include visual representations of:
  * Key cell types or tissues involved
  * Molecular pathways or mechanisms
  * Treatment effects or comparisons
  * Key biomarkers or genes
- Color scheme: Use professional, publication-appropriate colors
- Layout: Clear flow from left to right or top to bottom showing the research narrative
- Avoid excessive text - focus on visual communication
- Include simple labels for key elements
- Resolution and quality suitable for journal publication

The graphic abstract should tell the story of the research at a glance, highlighting the main findings and conclusions.
"""
        return prompt

    async def _generate_image(self, prompt: str) -> str:
        """
        Call Gemini image generation API.

        Args:
            prompt: The prompt for image generation

        Returns:
            Base64 encoded image string

        Raises:
            ValueError: If google_api_key is not configured
        """
        # Validate API key before making request
        if not self.google_api_key or not self.google_api_key.strip():
            error_msg = "Google API key is not configured. Please set the Google API key in the configuration to enable graphic abstract generation."
            logger.error(f"[{self.session_id}][Painter] {error_msg}")
            raise ValueError(error_msg)

        url = f"{self.base_url}/models/{self.model}:generateContent"

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }

        logger.debug(f"[{self.session_id}][Painter] Calling Gemini API: {url}")

        try:
            response = await self.client.post(
                url,
                headers={"x-goog-api-key": self.google_api_key},
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # Extract image from response
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if "inlineData" in part:
                    image_data = part["inlineData"]
                    # Return the base64 data (may include mime type prefix)
                    return image_data.get("data", "")

            # If no inline data found, log the response structure for debugging
            logger.warning(f"[{self.session_id}][Painter] Response structure: {data.keys()}")
            raise ValueError("No image data found in response")

        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.session_id}][Painter] HTTP error: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Image generation HTTP error: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"[{self.session_id}][Painter] Error generating image: {e}")
            raise ValueError(f"Image generation failed: {e}") from e

    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for processing.

        Args:
            state: Current analysis state
            instruction: Summary/instruction for graphic abstract generation

        Returns:
            Dict with graphic_abstract and success status
        """
        try:
            graphic = await self.generate_graphic_abstract(state, instruction)
            return {
                "graphic_abstract": graphic,
                "success": True
            }
        except Exception as e:
            logger.error(f"[{self.session_id}][Painter] Process failed: {e}")
            return {
                "graphic_abstract": None,
                "success": False,
                "error": str(e)
            }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures client is closed"""
        await self.close()
