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

logger = logging.getLogger(__name__)

# Model limits for google/gemini-3-pro-image-preview
_TOTAL_CONTEXT_TOKENS = 65_536
_MAX_OUTPUT_TOKENS = 32_800
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


def _is_response_format_compat_error(exc: Exception) -> bool:
    """Whether retrying the same endpoint without response_format can help."""
    err = str(exc)
    # Only malformed/unsupported-parameter responses justify repeating the
    # same endpoint. Auth, permission, missing endpoint, size, and rate-limit
    # errors must return to the protocol cascade instead.
    for code in ("400", "422"):
        if f"Error code: {code}" in err or f"status_code={code}" in err:
            return True
    return False


def _is_fatal_image_error(exc: Exception) -> bool:
    """Return whether the entire image operation must abort immediately."""
    return _image_error_disposition(exc) == "abort"


def _image_error_disposition(exc: Exception) -> str:
    """Classify an attempt as ``abort``, ``disable`` or ``retry``.

    Authentication and endpoint/capability errors are deterministic for the
    affected protocol, but another protocol may still work.  Content-policy
    failures apply to the requested operation and abort the whole cascade.
    """
    err = str(exc).lower()
    for kw in (
        "content_filter", "content policy", "content_policy",
        "safety system", "moderation",
    ):
        if kw in err:
            return "abort"

    status_match = re.search(r"(?:error code:|status_code\s*=|http/\S+\s+)?\s*([45]\d\d)\b", err)
    status = int(status_match.group(1)) if status_match else None
    permanent_auth = any(kw in err for kw in (
        "invalid api key", "incorrect api key", "authentication failed",
        "authentication_error", "unauthorized api key", "401 unauthorized",
    ))
    if permanent_auth or status in {400, 401, 403, 404, 405, 410, 415, 422}:
        return "disable"
    return "retry"


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

        # Budget: ~30K input tokens total. Fixed template ≈ 500 tokens, leave rest for summary.
        # Cap summary at ~25K tokens (~75K chars) as safety net — PI should already keep it short.
        max_summary_chars = 75_000
        if len(summary) > max_summary_chars:
            logger.warning(
                f"[{self.session_id}][Painter] Summary too long ({len(summary)} chars), "
                f"truncating to {max_summary_chars} chars"
            )
            summary = summary[:max_summary_chars] + "\n... [truncated due to token limit]"

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
            if message is None:
                raise ValueError("API response contained no message")

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

                    # Method 3b: Extract from markdown image syntax
                    # Some models (e.g. Gemini) return: ![image](data:image/...;base64,XXXXX)
                    md_match = re.search(
                        r'!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)\)',
                        content,
                    )
                    if md_match:
                        logger.info(
                            f"[{self.session_id}][Painter] Extracted image from markdown syntax"
                        )
                        return md_match.group(1).strip()

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
                            if "inline_data" in part:
                                return part["inline_data"].get("data", "")
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

    async def _download_image_as_base64(
        self,
        url: str,
        *,
        runtime: _PainterRuntimeSnapshot,
    ) -> str:
        """Download image from URL and return as base64 string.

        Used for providers (e.g. SiliconFlow) that return temporary image URLs
        instead of base64-encoded data.  Reuses the llm_client's httpx client
        so proxy and timeout settings are inherited.
        """
        # Reuse the shared httpx client (inherits proxy config)
        resp = await runtime.llm_client.http_client.get(url, timeout=60.0)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')

    async def _generate_image_dashscope(
        self,
        prompt: str,
        *,
        runtime: _PainterRuntimeSnapshot,
    ) -> str:
        """Generate image using DashScope native async task API.

        DashScope's OpenAI compatibility mode does not support image generation.
        This method calls the native DashScope task API directly:
        1. POST /api/v1/services/aigc/text2image/image-synthesis  → task_id
        2. Poll  GET /api/v1/tasks/{task_id}                      → until SUCCEEDED
        3. Download the result URL and return as base64.
        """
        api_key = runtime.llm_client.api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        payload = {
            "model": runtime.model,
            "input": {"prompt": prompt},
            "parameters": {"size": "1024*1024", "n": 1},
        }
        create_url = f"{_DASHSCOPE_BASE_URL}/api/v1/services/aigc/text2image/image-synthesis"

        logger.info(
            f"[{self.session_id}][Painter] DashScope: submitting image task "
            f"(model={runtime.model})"
        )
        resp = await runtime.llm_client.http_client.post(
            create_url, json=payload, headers=headers, timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()

        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise ValueError(f"DashScope image generation: no task_id in response: {data}")
        logger.info(f"[{self.session_id}][Painter] DashScope: task submitted, id={task_id}")

        poll_url = f"{_DASHSCOPE_BASE_URL}/api/v1/tasks/{task_id}"
        for attempt in range(60):  # up to 5 minutes (60 × 5 s)
            await asyncio.sleep(5)
            poll_resp = await runtime.llm_client.http_client.get(
                poll_url, headers=headers, timeout=30.0
            )
            poll_resp.raise_for_status()
            status_data = poll_resp.json()

            task_status = status_data.get("output", {}).get("task_status", "")
            logger.debug(
                f"[{self.session_id}][Painter] DashScope task {task_id}: status={task_status} "
                f"(attempt {attempt + 1}/60)"
            )

            if task_status == "SUCCEEDED":
                results = status_data.get("output", {}).get("results", [])
                if results:
                    url = results[0].get("url") if isinstance(results[0], dict) else None
                    if url:
                        logger.info(
                            f"[{self.session_id}][Painter] DashScope task succeeded, downloading image"
                        )
                        return await self._download_image_as_base64(url, runtime=runtime)
                raise ValueError(
                    f"DashScope image generation succeeded but no result URL found: {status_data}"
                )
            elif task_status in ("FAILED", "CANCELED"):
                raise ValueError(
                    f"DashScope image generation task {task_status}: {status_data}"
                )

        raise ValueError(
            f"DashScope image generation timed out after 5 minutes (task_id={task_id})"
        )

    async def _generate_image(self, prompt: str) -> str:
        """
        Call image generation API with multi-provider compatibility.

        Supports OpenRouter, OpenAI, Google AI Studio, n1n.ai, and SiliconFlow
        via a cascade of strategies:
        1. DashScope native async task API (when base_url contains dashscope.aliyuncs.com)
        2. Chat completions with modalities (OpenRouter, OpenAI, n1n.ai)
        3. Images API with b64_json (OpenAI, Google AI Studio, n1n.ai)
        4. Images API with URL response (SiliconFlow)

        Args:
            prompt: The prompt for image generation

        Returns:
            Base64 encoded image string

        Raises:
            ValueError: If image generation fails
        """
        self.refresh_runtime_config_if_needed()
        runtime = self._capture_runtime()

        # Validate input token count before calling API
        system_text = runtime.system_prompt or ""
        total_input = system_text + prompt
        est_tokens = _estimate_tokens(total_input)
        logger.debug(
            f"[{self.session_id}][Painter] Input token estimate: {est_tokens} "
            f"(limit: {_SAFE_INPUT_TOKEN_LIMIT})"
        )
        if est_tokens > _SAFE_INPUT_TOKEN_LIMIT:
            # Truncate prompt to fit within budget, preserving the tail (style instructions)
            excess_chars = (est_tokens - _SAFE_INPUT_TOKEN_LIMIT) * 3 + 300  # +300 for margin
            logger.warning(
                f"[{self.session_id}][Painter] Input exceeds safe limit "
                f"({est_tokens} > {_SAFE_INPUT_TOKEN_LIMIT} tokens), "
                f"trimming {excess_chars} chars from prompt"
            )
            prompt = prompt[:len(prompt) - excess_chars] + "\n... [trimmed to fit token limit]"

        logger.debug(f"[{self.session_id}][Painter] Calling image generation")

        _is_zhipu = _ZHIPU_BASE_URL in (runtime.llm_client.base_url or "")

        try:
            async with runtime.llm_client.direct_request_scope():
                # No input image, so the Images protocol uses /v1/images/generations.
                protocols = self._build_protocols(
                    image_run=lambda: self._images_api_attempt(
                        prompt,
                        runtime=runtime,
                        is_zhipu=_is_zhipu,
                    ),
                    chat_run=lambda: self._chat_image_attempt(prompt, runtime=runtime),
                    dashscope_run=lambda: self._generate_image_dashscope(
                        prompt,
                        runtime=runtime,
                    ),
                    runtime=runtime,
                )
                return await self._run_protocol_cascade(protocols, op="generation")
        except ValueError:
            raise
        except Exception as e:
            logger.exception(f"[{self.session_id}][Painter] Error generating image")
            raise ValueError(f"Image generation failed: {e}") from e

    def _protocol_order(
        self,
        runtime: Optional[_PainterRuntimeSnapshot] = None,
    ) -> List[str]:
        """Return protocols eligible for the configured provider, in order.

        - DashScope: use only its native task API.
        - OpenRouter: prefer chat, then its OpenAI-compatible Images API.
        - Everything else (New API, OpenAI, Google AI Studio, Zhipu, SiliconFlow,
          ...): use Images API then chat.  Never send a non-DashScope model/key
          to the DashScope native endpoint.
        """
        llm_client = runtime.llm_client if runtime is not None else self.llm_client
        base = (llm_client.base_url or "").lower()
        if _DASHSCOPE_BASE_URL in base:
            return ["dashscope"]
        if "openrouter" in base:
            return ["chat", "images"]
        return ["images", "chat"]

    def _build_protocols(
        self,
        *,
        image_run,
        chat_run,
        dashscope_run,
        runtime: Optional[_PainterRuntimeSnapshot] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble the ordered protocol list consumed by ``_run_protocol_cascade``.

        Each entry carries a human-readable ``name``, the async ``run`` callable,
        and a per-attempt ``timeout`` (``None`` = no wrapper, used for the DashScope
        native task which legitimately polls for minutes).
        """
        runners = {
            "images": {"name": "images_api", "run": image_run, "timeout": _PROTOCOL_TIMEOUT},
            "chat": {"name": "chat_modalities", "run": chat_run, "timeout": _PROTOCOL_TIMEOUT},
            "dashscope": {"name": "dashscope_native", "run": dashscope_run, "timeout": None},
        }
        return [runners[p] for p in self._protocol_order(runtime)]

    async def _run_protocol_cascade(
        self,
        protocols: List[Dict[str, Any]],
        *,
        op: str = "generation",
        max_rounds: int = _MAX_RETRY_ROUNDS,
        total_timeout: Optional[float] = None,
    ) -> str:
        """Try each protocol in order; on any non-fatal failure move to the next.

        Within a round, protocols are separated by ``_INTER_PROTOCOL_DELAY``. After
        a full round fails, back off (2/4/8/16s) and retry the whole cascade, up to
        ``max_rounds`` rounds. Deterministic auth/capability errors disable that
        protocol; operation-wide content-policy failures abort immediately. Each
        protocol carrying a ``timeout`` is bounded by it and by the optional total
        deadline.

        Returns the first successfully parsed base64 image; raises ValueError if all
        protocols fail across all rounds.
        """
        errors: Dict[str, str] = {}
        disabled: set[str] = set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + total_timeout if total_timeout is not None else None

        def _remaining() -> Optional[float]:
            if deadline is None:
                return None
            return max(0.0, deadline - loop.time())

        for round_idx in range(max_rounds):
            if deadline is not None and (_remaining() or 0.0) <= 0:
                break
            if round_idx > 0:
                backoff = 2 ** round_idx  # 2, 4, 8, 16 before rounds 1..4
                logger.warning(
                    f"[{self.session_id}][Painter] All protocols failed round {round_idx} "
                    f"({op}); retrying whole cascade in {backoff}s"
                )
                remaining = _remaining()
                if remaining is not None and remaining <= backoff:
                    break
                await asyncio.sleep(backoff)

            attempted = 0
            for proto in protocols:
                name = proto["name"]
                if name in disabled:
                    continue
                if attempted > 0:
                    remaining = _remaining()
                    if remaining is not None and remaining <= _INTER_PROTOCOL_DELAY:
                        break
                    await asyncio.sleep(_INTER_PROTOCOL_DELAY)
                attempted += 1
                timeout = proto.get("timeout")
                remaining = _remaining()
                if remaining is not None:
                    if remaining <= 0:
                        break
                    timeout = min(float(timeout), remaining) if timeout else remaining
                try:
                    if timeout:
                        return await asyncio.wait_for(proto["run"](), timeout=timeout)
                    return await proto["run"]()
                except asyncio.TimeoutError:
                    errors[name] = f"timeout after {timeout}s"
                    logger.warning(
                        f"[{self.session_id}][Painter] protocol {name} ({op}) round "
                        f"{round_idx + 1} timed out after {timeout}s, switching"
                    )
                except Exception as e:  # noqa: BLE001 - any parse/HTTP error switches protocol
                    disposition = _image_error_disposition(e)
                    if disposition == "abort":
                        logger.warning(
                            f"[{self.session_id}][Painter] fatal error on {name} ({op}), "
                            f"aborting cascade: {e}"
                        )
                        raise ValueError(
                            f"Image {op} aborted on {name} (unrecoverable): {e}"
                        ) from e
                    errors[name] = str(e)[:200]
                    if disposition == "disable":
                        disabled.add(name)
                    logger.warning(
                        f"[{self.session_id}][Painter] protocol {name} ({op}) round "
                        f"{round_idx + 1} failed: {e}, "
                        f"{'disabling protocol' if disposition == 'disable' else 'switching'}"
                    )

            if all(proto["name"] in disabled for proto in protocols):
                break

        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "no protocols attempted"
        raise ValueError(
            f"Image {op} failed after at most {max_rounds} rounds across eligible protocols. {detail}"
        )

    async def _chat_image_attempt(
        self,
        prompt: str,
        *,
        runtime: _PainterRuntimeSnapshot,
        input_image_b64: Optional[str] = None,
    ) -> str:
        """One attempt at image generation via chat completions modalities.

        When ``input_image_b64`` is provided (revision) and vision is enabled, the
        current image is attached so the model can see and modify it; otherwise the
        prompt is sent text-only (the "look at the attached image" line is stripped).
        """
        if input_image_b64 is not None and runtime.has_vision:
            from ..utils.llm_client import _detect_image_mime_type
            mime_type = _detect_image_mime_type(input_image_b64)
            user_content: Any = [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{input_image_b64}"}},
                {"type": "text", "text": prompt},
            ]
        elif input_image_b64 is not None:
            # Revision without vision: text-only, drop the image reference line.
            user_content = prompt.replace(
                "- Look at the attached current graphic abstract image carefully\n", ""
            )
        else:
            user_content = prompt

        response = await runtime.llm_client.client.chat.completions.create(
            model=runtime.model,
            messages=[
                {"role": "system", "content": runtime.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=runtime.temperature,
            max_tokens=4096,
            extra_body={"modalities": ["image", "text"]},
        )
        return self._extract_image_from_response(response)

    async def _images_edit_attempt(
        self,
        prompt: str,
        input_image_b64: str,
        *,
        runtime: _PainterRuntimeSnapshot,
    ) -> str:
        """One attempt at image revision via the Images edits API (/v1/images/edits).

        Sends the current image plus the revision prompt. Falls back to omitting
        ``response_format`` for providers that reject it (e.g. gpt-image-1, which
        always returns b64). Raises on any failure so the cascade can switch.
        """
        from ..utils.llm_client import _detect_image_mime_type
        img_bytes = base64.b64decode(input_image_b64)
        mime = _detect_image_mime_type(input_image_b64)
        ext = "jpg" if "jpeg" in mime else ("webp" if "webp" in mime else "png")
        image_file = (f"image.{ext}", img_bytes, mime)

        try:
            response = await runtime.llm_client.client.images.edit(
                model=runtime.model,
                image=image_file,
                prompt=prompt,
                n=1,
                response_format="b64_json",
            )
        except Exception as fmt_err:
            # Only repeat the same endpoint for an unsupported response_format.
            # A generic 403/404 must return to the cascade so chat/native
            # revision protocols still get a chance.
            if not _is_response_format_compat_error(fmt_err):
                raise
            logger.info(
                f"[{self.session_id}][Painter] Images edits with b64_json failed ({fmt_err}), "
                f"retrying without response_format"
            )
            response = await runtime.llm_client.client.images.edit(
                model=runtime.model,
                image=image_file,
                prompt=prompt,
                n=1,
            )
        return await self._extract_image_from_images_response(response, runtime=runtime)

    async def _images_api_attempt(
        self,
        prompt: str,
        *,
        runtime: _PainterRuntimeSnapshot,
        is_zhipu: bool = False,
    ) -> str:
        """One attempt at the Images API (/v1/images/generations).

        Handles the b64_json -> URL response-format fallback for providers that
        do not support the ``response_format`` parameter (e.g. SiliconFlow), and
        extracts the image from both standard (``data``) and non-standard
        (``images``) response shapes.

        Args:
            prompt: The image generation prompt.
            is_zhipu: Whether the provider is Zhipu AI (pre-computed by caller).

        Raises on any failure so the caller can decide whether to retry.
        """
        # Zhipu AI does not support response_format="b64_json" parameter;
        # skip it and use the default URL response format.
        if is_zhipu:
            logger.info(
                f"[{self.session_id}][Painter] Zhipu AI detected, "
                f"using default URL response format"
            )
            response = await runtime.llm_client.client.images.generate(
                model=runtime.model,
                prompt=prompt,
            )
        else:
            # Try with b64_json first (OpenAI, Google AI Studio, n1n.ai)
            # then fall back without it (SiliconFlow only supports URL responses)
            try:
                response = await runtime.llm_client.client.images.generate(
                    model=runtime.model,
                    prompt=prompt,
                    n=1,
                    response_format="b64_json",
                )
            except Exception as b64_err:
                # Only fall back for client errors (400/422 = unsupported param);
                # propagate transient/auth errors immediately
                if not _is_response_format_compat_error(b64_err):
                    raise
                logger.info(
                    f"[{self.session_id}][Painter] Images API with b64_json failed ({b64_err}), "
                    f"trying without response_format"
                )
                response = await runtime.llm_client.client.images.generate(
                    model=runtime.model,
                    prompt=prompt,
                    n=1,
                )

        return await self._extract_image_from_images_response(response, runtime=runtime)

    async def _extract_image_from_images_response(
        self,
        response,
        *,
        runtime: _PainterRuntimeSnapshot,
    ) -> str:
        """Extract a base64 image from an Images API response (generations or edits).

        Handles standard ``data[0].b64_json`` / ``data[0].url`` shapes plus the
        non-standard SiliconFlow ``images`` array, surfacing provider error /
        content_filter fields. Raises ValueError if no image can be extracted.
        """
        # Debug: log raw response for diagnosis
        raw = None
        if hasattr(response, 'model_dump'):
            raw = response.model_dump()
            logger.debug(f"[{self.session_id}][Painter] Images API raw response keys: {list(raw.keys())}")
            logger.debug(f"[{self.session_id}][Painter] Images API raw response: {raw}")
        else:
            logger.debug(f"[{self.session_id}][Painter] Images API response type: {type(response)}")

        # Extract image from response
        if getattr(response, "data", None):
            img_item = response.data[0]
            # Prefer b64_json (OpenAI, Google AI Studio)
            if hasattr(img_item, 'b64_json') and img_item.b64_json:
                return img_item.b64_json
            # Fall back to URL and download (SiliconFlow returns temporary URLs)
            if hasattr(img_item, 'url') and img_item.url:
                logger.info(f"[{self.session_id}][Painter] Downloading image from URL")
                return await self._download_image_as_base64(
                    img_item.url,
                    runtime=runtime,
                )

        # SiliconFlow non-standard: 'images' array
        # Zhipu / standard: 'data' array
        if raw is not None:
            # Check error field first (Zhipu may return 500 with data=None)
            if 'error' in raw and raw['error']:
                error_info = raw['error']
                error_msg = error_info.get('message', str(error_info)) if isinstance(error_info, dict) else str(error_info)
                logger.warning(f"[{self.session_id}][Painter] Images API returned error: {error_info}")
                raise ValueError(f"Images API error: {error_msg}")

            # Check content_filter for 审核拦截
            if 'content_filter' in raw and raw['content_filter']:
                logger.warning(f"[{self.session_id}][Painter] Content filter triggered: {raw['content_filter']}")

            # SiliconFlow non-standard: 'images' array
            if 'images' in raw and raw['images']:
                for img in raw['images']:
                    url = img.get('url', '') if isinstance(img, dict) else ''
                    if url:
                        logger.info(f"[{self.session_id}][Painter] Downloading image from non-standard response")
                        return await self._download_image_as_base64(url, runtime=runtime)

            # Zhipu / standard: 'data' array
            if 'data' in raw and raw['data']:
                img_data = raw['data'][0]
                if isinstance(img_data, dict):
                    url = img_data.get('url', '')
                    if url:
                        logger.info(f"[{self.session_id}][Painter] Downloading image from standard response (data[0].url)")
                        return await self._download_image_as_base64(url, runtime=runtime)

        raise ValueError("Images API returned no image data")

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
