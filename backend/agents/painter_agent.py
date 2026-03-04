"""
Painter Agent - Generates graphic abstracts using LLM image generation via OpenRouter

This agent uses the user-configured LLM hosting service (default: OpenRouter) to call
image generation models like Google Gemini with modalities=["image", "text"].
"""
from typing import Dict, Any, Optional, List
import logging
import base64
import json

from .base import BaseAgent, AgentConfig
from ..models.schemas import AgentType, AnalysisState, GraphicAbstract
from ..utils.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class PainterAgent(BaseAgent):
    """
    Painter Agent using OpenRouter (or other OpenAI-compatible API) for image generation.

    Responsibilities:
    - Generate graphic abstracts summarizing research findings
    - Create visual representations of key discoveries

    This agent inherits from BaseAgent and uses the same LLM client as other agents,
    ensuring it respects the user's configured LLM hosting service (OpenRouter by default).
    """

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str = "google/gemini-3-pro-image-preview",
        system_prompt: str = "",
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize the Painter Agent.

        Args:
            llm_client: OpenAI-compatible client (OpenRouter, etc.)
            model: Model to use for image generation (e.g., google/gemini-3-pro-image-preview)
            system_prompt: System prompt (optional for image generation)
            config: Agent configuration
        """
        super().__init__(
            agent_type=AgentType.PAINTER,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt or "You are a professional scientific illustrator.",
            config=config or AgentConfig(temperature=0.7)
        )

        logger.info(f"[Painter] Initialized with model={model}")

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

        # Call image generation via OpenRouter
        image_base64 = await self._generate_image(prompt)

        graphic_abstract = GraphicAbstract(
            prompt=prompt,
            image_base64=image_base64
        )

        logger.info(f"[{self.session_id}][Painter] Graphic abstract generated successfully")
        return graphic_abstract

    async def revise_graphic_abstract(
        self,
        state: AnalysisState,
        current_image_base64: str,
        original_prompt: str,
        feedback: str,
        suggestions: List[str],
        must_fix: List[str],
        revision_history: List[Dict[str, Any]]
    ) -> str:
        """
        Revise a graphic abstract based on PI feedback.

        Sends the current image + revision instructions to the image generation model,
        which can see the existing image and produce an improved version.

        Args:
            state: Analysis state
            current_image_base64: Current graphic abstract image (base64)
            original_prompt: Original generation prompt
            feedback: Current round's feedback text
            suggestions: Current round's suggestions
            must_fix: Current round's critical issues
            revision_history: All previous rounds' feedback

        Returns:
            Base64 encoded revised image

        Raises:
            ValueError: If revision fails
        """
        self._set_llm_context()
        logger.info(f"[{self.session_id}][Painter] Revising graphic abstract (revision {len(revision_history)})")

        # Build revision prompt with history to prevent regression
        history_text = ""
        if len(revision_history) > 1:
            # Show all previous rounds (excluding current) so model avoids regression
            for entry in revision_history[:-1]:
                round_num = entry.get("round", "?")
                hist_feedback = entry.get("feedback", "")
                hist_suggestions = entry.get("suggestions", [])
                history_text += f"\n--- Previous Round {round_num} feedback ---\n"
                history_text += f"Feedback: {hist_feedback}\n"
                if hist_suggestions:
                    history_text += f"Suggestions: {'; '.join(hist_suggestions)}\n"
            history_text += "\nIMPORTANT: Ensure that issues from previous rounds remain fixed.\n"

        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        must_fix_text = "\n".join(f"- {m}" for m in must_fix) if must_fix else "None"

        revision_prompt = f"""Revise this graphic abstract based on the following feedback.

**Original Design Intent:**
{original_prompt[:1000]}

{history_text}

**Current Round Feedback:**
{feedback}

**Suggestions to address:**
{suggestions_text}

**Critical issues that MUST be fixed:**
{must_fix_text}

**Instructions:**
- Look at the attached current graphic abstract image carefully
- Address ALL the feedback and critical issues listed above
- Maintain the overall scientific narrative and research story
- Keep the professional, publication-quality style
- Improve the visual communication based on the feedback
- Generate an improved version of the graphic abstract
"""

        try:
            # Build multimodal message: current image + revision instructions
            # Gemini supports image-in + image-out
            from ..utils.llm_client import _detect_image_mime_type
            mime_type = _detect_image_mime_type(current_image_base64)

            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{current_image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": revision_prompt
                }
            ]

            response = await self.llm_client.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.config.temperature,
                max_tokens=4096,
                extra_body={
                    "modalities": ["image", "text"]
                }
            )
            new_image = self._extract_image_from_response(response)
            logger.info(f"[{self.session_id}][Painter] Graphic abstract revision successful")
            return new_image

        except Exception as e:
            logger.error(f"[{self.session_id}][Painter] Error revising graphic abstract: {e}")
            raise ValueError(f"Graphic abstract revision failed: {e}") from e

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

    def _extract_image_from_response(self, response) -> str:
        """
        Extract base64 image from API response.

        Handles multiple response formats from different providers.

        Args:
            response: Raw API response

        Returns:
            Base64 encoded image string

        Raises:
            ValueError: If no image found in response
        """
        if response and response.choices and len(response.choices) > 0:
            message = response.choices[0].message

            # Method 1: Check for images field directly (OpenRouter format)
            if hasattr(message, 'images') and message.images:
                logger.debug(f"[{self.session_id}][Painter] Found images field in response")
                for img in message.images:
                    if isinstance(img, dict):
                        url = img.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            return url.split(",", 1)[1] if "," in url else url

            # Method 2: Check model_dump for images (Pydantic model with extra='allow')
            if hasattr(message, 'model_dump'):
                try:
                    msg_data = message.model_dump(mode='python')
                except TypeError:
                    msg_data = message.model_dump()
                logger.debug(f"[{self.session_id}][Painter] model_dump keys: {list(msg_data.keys())}")

                if 'images' in msg_data and msg_data['images']:
                    logger.debug(f"[{self.session_id}][Painter] Found images in model_dump")
                    for img in msg_data['images']:
                        if isinstance(img, dict):
                            url = img.get("image_url", {}).get("url", "")
                            if url.startswith("data:image"):
                                return url.split(",", 1)[1] if "," in url else url

            # Method 2b: Check response.model_dump() for images at choice level
            if hasattr(response, 'model_dump'):
                try:
                    resp_data = response.model_dump(mode='python')
                except TypeError:
                    resp_data = response.model_dump()
                logger.debug(f"[{self.session_id}][Painter] response model_dump keys: {list(resp_data.keys())}")

                if 'choices' in resp_data and resp_data['choices']:
                    choice_msg = resp_data['choices'][0].get('message', {})
                    if 'images' in choice_msg and choice_msg['images']:
                        logger.debug(f"[{self.session_id}][Painter] Found images in response model_dump")
                        for img in choice_msg['images']:
                            if isinstance(img, dict):
                                url = img.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    return url.split(",", 1)[1] if "," in url else url

            # Method 3: Check content for image data
            if message and message.content:
                content = message.content

                if isinstance(content, str):
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "image" in data:
                            return data["image"]
                        if isinstance(data, dict) and "images" in data:
                            for img in data["images"]:
                                if isinstance(img, dict):
                                    url = img.get("image_url", {}).get("url", "")
                                    if url.startswith("data:image"):
                                        return url.split(",", 1)[1] if "," in url else url
                    except json.JSONDecodeError:
                        pass

                    if content.startswith("data:image"):
                        if "," in content:
                            return content.split(",", 1)[1]
                        return content

                    if len(content) > 100 and not content.startswith("{"):
                        try:
                            base64.b64decode(content[:100])
                            return content
                        except Exception:
                            pass

                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if "inlineData" in part:
                                return part["inlineData"].get("data", "")
                            if part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    return url.split(",", 1)[1] if "," in url else url

            # Log the response structure for debugging
            msg_attrs = {}
            if message:
                for attr in ['content', 'images', 'role', 'refusal']:
                    if hasattr(message, attr):
                        val = getattr(message, attr)
                        msg_attrs[attr] = str(val)[:100] if val else None
                if hasattr(message, 'model_dump'):
                    msg_attrs['model_dump_keys'] = list(message.model_dump().keys())

            logger.warning(
                f"[{self.session_id}][Painter] Could not extract image from response. "
                f"Message attributes: {msg_attrs}"
            )

            if message and message.content and isinstance(message.content, str):
                raise ValueError(
                    f"Model returned text instead of image. "
                    f"Response preview: {message.content[:300]}"
                )

        raise ValueError("No image data found in response")

    async def _generate_image(self, prompt: str) -> str:
        """
        Call image generation API via OpenRouter.

        Args:
            prompt: The prompt for image generation

        Returns:
            Base64 encoded image string

        Raises:
            ValueError: If image generation fails
        """
        self._set_llm_context()

        logger.debug(f"[{self.session_id}][Painter] Calling image generation via OpenRouter")

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=4096,
                extra_body={
                    "modalities": ["image", "text"]
                }
            )
            return self._extract_image_from_response(response)

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
