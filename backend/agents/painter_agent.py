"""
Painter Agent - Generates graphic abstracts using LLM image generation

This agent uses the user-configured LLM hosting service to call image generation models.
It selects the protocols that are compatible with the configured provider:

  1. Images API — POST /v1/images/generations (text prompt) or /v1/images/edits
     (when an input image is provided, i.e. revision)
  2. Chat completions with modalities=["image", "text"]
  3. DashScope native async task API

Permanent failures disable only the affected protocol for the current operation;
transient failures may be retried.  Revision is best-effort because a usable input
image already exists, so it has a small wall-clock budget rather than repeating five
full 180-second cascades.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import base64
import json
import asyncio
import re

import httpx

from .base import BaseAgent, AgentConfig
from ..models.schemas import AgentType, AnalysisState, GraphicAbstract
from ..utils.llm_client import OpenAICompatibleClient
from ..utils.settings_provider import AgentSettingsProvider

import sys
from .painter_parts import prompts as painter_prompts
from .painter_parts import protocols as painter_protocols
from .painter_parts import responses as painter_responses
from .painter_parts.protocols import (
    _image_error_disposition,
    _is_fatal_image_error,
    _is_response_format_compat_error,
)

logger = logging.getLogger(__name__)

# Model limits for google/gemini-3-pro-image-preview
_TOTAL_CONTEXT_TOKENS = 65_536
_SAFE_INPUT_TOKEN_LIMIT = 30_000  # Conservative: 64K - 32.8K ≈ 31.2K, use 30K

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com"
_GOOGLE_AI_STUDIO_BASE_URL = "generativelanguage.googleapis.com"
_ZHIPU_BASE_URL = "open.bigmodel.cn"

# Protocol cascade tuning (see module docstring).
_MAX_RETRY_ROUNDS = 5          # full-cascade rounds before giving up
_REVISION_MAX_RETRY_ROUNDS = 2
_REVISION_TOTAL_TIMEOUT = 720.0
_INTER_PROTOCOL_DELAY = 1.0    # seconds to wait when switching protocols within a round
_PROTOCOL_TIMEOUT = 180.0      # per-attempt timeout for Images API / Chat protocols


@dataclass(frozen=True)
class _PainterRuntimeSnapshot:
    """Configuration and transport frozen for one complete image operation."""

    llm_client: OpenAICompatibleClient
    model: str
    system_prompt: str
    temperature: float
    has_vision: bool


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: len/3 for mixed CJK+English (conservative)."""
    return max(1, len(text) // 3)








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
        settings_provider: 'AgentSettingsProvider',
    ):
        """
        Initialize the Painter Agent.

        Args:
            settings_provider: Settings provider for hot-reloadable agent settings
        """
        super().__init__(
            agent_type=AgentType.PAINTER,
            settings_provider=settings_provider,
        )

        logger.info(f"[Painter] Initialized with model={self.model}")

    def _capture_runtime(self) -> _PainterRuntimeSnapshot:
        """Capture one coherent runtime before an operation reaches its first await."""
        llm_client = self.llm_client
        llm_client.set_context(self.session_id, self.agent_type.value)
        return _PainterRuntimeSnapshot(
            llm_client=llm_client,
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            has_vision=self.has_vision,
        )

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
        self.refresh_runtime_config_if_needed()
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
        self.refresh_runtime_config_if_needed()
        runtime = self._capture_runtime()
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

        # Validate text input token count (excludes image tokens which are counted separately)
        text_input = (runtime.system_prompt or "") + revision_prompt
        est_tokens = _estimate_tokens(text_input)
        # Image tokens add overhead (~258 tokens for a 1024px image), leave extra margin
        est_tokens += 300
        logger.debug(
            f"[{self.session_id}][Painter] Revision input token estimate: {est_tokens} "
            f"(limit: {_SAFE_INPUT_TOKEN_LIMIT})"
        )
        if est_tokens > _SAFE_INPUT_TOKEN_LIMIT:
            excess_chars = (est_tokens - _SAFE_INPUT_TOKEN_LIMIT) * 3 + 300
            logger.warning(
                f"[{self.session_id}][Painter] Revision input exceeds safe limit, "
                f"trimming {excess_chars} chars"
            )
            revision_prompt = revision_prompt[:len(revision_prompt) - excess_chars] + \
                "\n... [trimmed to fit token limit]"

        # Text-only variant for the DashScope native path, which cannot see the
        # current image and must regenerate from the revised instructions alone.
        text_only_prompt = revision_prompt.replace(
            "- Look at the attached current graphic abstract image carefully\n", ""
        )

        try:
            # The scope protects the frozen transport across every retry, poll,
            # and result download, even if runtime settings change concurrently.
            async with runtime.llm_client.direct_request_scope():
                # Revision has an input image, so the Images protocol uses /v1/images/edits.
                protocols = self._build_protocols(
                    image_run=lambda: self._images_edit_attempt(
                        revision_prompt,
                        current_image_base64,
                        runtime=runtime,
                    ),
                    chat_run=lambda: self._chat_image_attempt(
                        revision_prompt,
                        input_image_b64=current_image_base64,
                        runtime=runtime,
                    ),
                    dashscope_run=lambda: self._generate_image_dashscope(
                        text_only_prompt,
                        runtime=runtime,
                    ),
                    runtime=runtime,
                )
                new_image = await self._run_protocol_cascade(
                    protocols,
                    op="revision",
                    max_rounds=_REVISION_MAX_RETRY_ROUNDS,
                    total_timeout=_REVISION_TOTAL_TIMEOUT,
                )
            logger.info(f"[{self.session_id}][Painter] Graphic abstract revision successful")
            return new_image
        except Exception as e:
            logger.exception(f"[{self.session_id}][Painter] Error revising graphic abstract")
            raise ValueError(f"Graphic abstract revision failed: {e}") from e

    def _build_graphic_prompt(self, state: AnalysisState, summary: str) -> str:
        painter_prompts.bind_facade(sys.modules[__name__])
        return painter_prompts._build_graphic_prompt(self, state, summary)

    def _extract_image_from_response(self, response) -> str:
        painter_responses.bind_facade(sys.modules[__name__])
        return painter_responses._extract_image_from_response(self, response)

    async def _download_image_as_base64(
        self, url: str, *, runtime: _PainterRuntimeSnapshot
    ) -> str:
        painter_responses.bind_facade(sys.modules[__name__])
        return await painter_responses._download_image_as_base64(
            self, url, runtime=runtime
        )

    async def _generate_image_dashscope(
        self, prompt: str, *, runtime: _PainterRuntimeSnapshot
    ) -> str:
        painter_protocols.bind_facade(sys.modules[__name__])
        return await painter_protocols._generate_image_dashscope(
            self, prompt, runtime=runtime
        )

    async def _generate_image(self, prompt: str) -> str:
        painter_protocols.bind_facade(sys.modules[__name__])
        return await painter_protocols._generate_image(self, prompt)

    def _protocol_order(
        self, runtime: Optional[_PainterRuntimeSnapshot] = None
    ) -> List[str]:
        painter_protocols.bind_facade(sys.modules[__name__])
        return painter_protocols._protocol_order(self, runtime)

    def _build_protocols(
        self, *, image_run, chat_run, dashscope_run,
        runtime: Optional[_PainterRuntimeSnapshot] = None,
    ) -> List[Dict[str, Any]]:
        painter_protocols.bind_facade(sys.modules[__name__])
        return painter_protocols._build_protocols(
            self, image_run=image_run, chat_run=chat_run,
            dashscope_run=dashscope_run, runtime=runtime,
        )

    async def _run_protocol_cascade(
        self, protocols: List[Dict[str, Any]], *, op: str = "generation",
        max_rounds: int = _MAX_RETRY_ROUNDS,
        total_timeout: Optional[float] = None,
    ) -> str:
        painter_protocols.bind_facade(sys.modules[__name__])
        return await painter_protocols._run_protocol_cascade(
            self, protocols, op=op, max_rounds=max_rounds,
            total_timeout=total_timeout,
        )

    async def _chat_image_attempt(
        self, prompt: str, *, runtime: _PainterRuntimeSnapshot,
        input_image_b64: Optional[str] = None,
    ) -> str:
        painter_protocols.bind_facade(sys.modules[__name__])
        return await painter_protocols._chat_image_attempt(
            self, prompt, runtime=runtime, input_image_b64=input_image_b64
        )

    async def _images_edit_attempt(
        self, prompt: str, input_image_b64: str, *,
        runtime: _PainterRuntimeSnapshot,
    ) -> str:
        painter_protocols.bind_facade(sys.modules[__name__])
        return await painter_protocols._images_edit_attempt(
            self, prompt, input_image_b64, runtime=runtime
        )

    async def _images_api_attempt(
        self, prompt: str, *, runtime: _PainterRuntimeSnapshot,
        is_zhipu: bool = False,
    ) -> str:
        painter_protocols.bind_facade(sys.modules[__name__])
        return await painter_protocols._images_api_attempt(
            self, prompt, runtime=runtime, is_zhipu=is_zhipu
        )

    async def _extract_image_from_images_response(
        self, response, *, runtime: _PainterRuntimeSnapshot
    ) -> str:
        painter_responses.bind_facade(sys.modules[__name__])
        return await painter_responses._extract_image_from_images_response(
            self, response, runtime=runtime
        )

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
            logger.exception(f"[{self.session_id}][Painter] Process failed")
            return {
                "graphic_abstract": None,
                "success": False,
                "error": str(e)
            }
