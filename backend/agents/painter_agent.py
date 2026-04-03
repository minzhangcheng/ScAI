"""
Painter Agent - Generates graphic abstracts using LLM image generation

This agent uses the user-configured LLM hosting service to call image generation models.
Supports multiple platforms via dual-path strategy:
1. DashScope native async task API (when base_url contains dashscope.aliyuncs.com)
2. Chat completions with modalities=["image", "text"] (OpenRouter)
3. Images API (/v1/images/generations) fallback (OpenAI, Google AI Studio, SiliconFlow, etc.)
"""
from typing import Dict, Any, Optional, List
import logging
import base64
import json
import asyncio
import re

import httpx

from .base import BaseAgent, AgentConfig
from ..models.schemas import AgentType, AnalysisState, GraphicAbstract
from ..utils.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

# Model limits for google/gemini-3-pro-image-preview
_TOTAL_CONTEXT_TOKENS = 65_536
_MAX_OUTPUT_TOKENS = 32_800
_SAFE_INPUT_TOKEN_LIMIT = 30_000  # Conservative: 64K - 32.8K ≈ 31.2K, use 30K

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com"
_GOOGLE_AI_STUDIO_BASE_URL = "generativelanguage.googleapis.com"
_ZHIPU_BASE_URL = "open.bigmodel.cn"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: len/3 for mixed CJK+English (conservative)."""
    return max(1, len(text) // 3)


def _is_client_error(exc: Exception) -> bool:
    """Check if an exception is a non-retriable client error (4xx)."""
    err = str(exc)
    # OpenAI/OpenRouter client wraps HTTP status in error messages
    for code in ("400", "401", "403", "404", "413", "422", "429"):
        if f"Error code: {code}" in err or f"status_code={code}" in err:
            # 429 (rate limit) IS retriable, skip it
            if code == "429":
                return False
            return True
    return False


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

        # Validate text input token count (excludes image tokens which are counted separately)
        text_input = (self.system_prompt or "") + revision_prompt
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

        try:
            # DashScope native API: OpenAI compat mode does not support image models.
            # Re-generate with revised text prompt via the native task API.
            if _DASHSCOPE_BASE_URL in self.llm_client.base_url:
                logger.info(f"[{self.session_id}][Painter] DashScope: revising via native text2image API")
                text_only_prompt = revision_prompt.replace(
                    "- Look at the attached current graphic abstract image carefully\n", ""
                )
                return await self._generate_image_dashscope(text_only_prompt)

            # Build multimodal message: current image + revision instructions
            if self.has_vision:
                # Vision enabled: send current image so model can see and modify it
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
            else:
                # No vision: only send text, model regenerates based on feedback alone
                logger.info(f"[{self.session_id}][Painter] Vision disabled, revising with text-only prompt")
                # Remove reference to attached image from prompt
                text_only_prompt = revision_prompt.replace(
                    "- Look at the attached current graphic abstract image carefully\n", ""
                )
                user_content = [{"type": "text", "text": text_only_prompt}]

            last_error = None
            for attempt in range(3):
                try:
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
                    last_error = e
                    # Don't retry client errors (4xx) — they'll always fail
                    if _is_client_error(e):
                        break
                    if attempt < 2:
                        delay = 2 ** attempt
                        logger.warning(
                            f"[{self.session_id}][Painter] Revision attempt {attempt + 1} failed: {e}, "
                            f"retrying in {delay}s"
                        )
                        await asyncio.sleep(delay)
            raise last_error

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

    async def _download_image_as_base64(self, url: str) -> str:
        """Download image from URL and return as base64 string.

        Used for providers (e.g. SiliconFlow) that return temporary image URLs
        instead of base64-encoded data.  Reuses the llm_client's httpx client
        so proxy and timeout settings are inherited.
        """
        # Reuse the shared httpx client (inherits proxy config)
        resp = await self.llm_client.http_client.get(url, timeout=60.0)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')

    async def _generate_image_dashscope(self, prompt: str) -> str:
        """Generate image using DashScope native async task API.

        DashScope's OpenAI compatibility mode does not support image generation.
        This method calls the native DashScope task API directly:
        1. POST /api/v1/services/aigc/text2image/image-synthesis  → task_id
        2. Poll  GET /api/v1/tasks/{task_id}                      → until SUCCEEDED
        3. Download the result URL and return as base64.
        """
        api_key = self.llm_client.api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        payload = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": {"size": "1024*1024", "n": 1},
        }
        create_url = f"{_DASHSCOPE_BASE_URL}/api/v1/services/aigc/text2image/image-synthesis"

        logger.info(f"[{self.session_id}][Painter] DashScope: submitting image task (model={self.model})")
        resp = await self.llm_client.http_client.post(
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
            poll_resp = await self.llm_client.http_client.get(
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
                        return await self._download_image_as_base64(url)
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
        self._set_llm_context()

        # Validate input token count before calling API
        system_text = self.system_prompt or ""
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

        # Strategy 0: DashScope native async task API
        # DashScope's OpenAI compatibility mode does not expose image generation endpoints.
        if _DASHSCOPE_BASE_URL in self.llm_client.base_url:
            return await self._generate_image_dashscope(prompt)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Google AI Studio and Zhipu do not support modalities in chat completions;
        # skip Strategy 1 and go directly to Images API (Strategy 2).
        _is_google_ai_studio = _GOOGLE_AI_STUDIO_BASE_URL in self.llm_client.base_url
        _is_zhipu = _ZHIPU_BASE_URL in self.llm_client.base_url
        _skip_chat_modalities = _is_google_ai_studio or _is_zhipu

        try:
            # Strategy 1: Try chat completions with modalities (works on OpenRouter)
            chat_error = None
            if not _skip_chat_modalities:
                last_error = None
                for attempt in range(3):
                    try:
                        response = await self.llm_client.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.config.temperature,
                            extra_body={
                                "modalities": ["image", "text"]
                            }
                        )
                        return self._extract_image_from_response(response)
                    except Exception as e:
                        last_error = e
                        # Don't retry client errors (4xx) — they'll always fail
                        if _is_client_error(e):
                            break
                        if attempt < 2:
                            delay = 2 ** attempt
                            logger.warning(
                                f"[{self.session_id}][Painter] Chat generation attempt {attempt + 1} failed: {e}, "
                                f"retrying in {delay}s"
                            )
                            await asyncio.sleep(delay)
                chat_error = last_error
            else:
                provider = "Google AI Studio" if _is_google_ai_studio else "Zhipu"
                logger.info(
                    f"[{self.session_id}][Painter] {provider} detected, "
                    f"skipping chat modalities strategy"
                )

            # Strategy 2: Try Images API (/v1/images/generations)
            # Works on OpenAI, Google AI Studio, n1n.ai, SiliconFlow, etc.
            logger.info(
                f"[{self.session_id}][Painter] Chat completions image gen "
                f"{'skipped' if _skip_chat_modalities else f'failed ({chat_error})'}, "
                f"trying Images API fallback"
            )
            try:
                # Zhipu AI does not support response_format="b64_json" parameter;
                # skip it and use the default URL response format.
                if _is_zhipu:
                    logger.info(
                        f"[{self.session_id}][Painter] Zhipu AI detected, "
                        f"using default URL response format"
                    )
                    response = await self.llm_client.client.images.generate(
                        model=self.model,
                        prompt=prompt,
                    )
                else:
                    # Try with b64_json first (OpenAI, Google AI Studio, n1n.ai)
                    # then fall back without it (SiliconFlow only supports URL responses)
                    try:
                        response = await self.llm_client.client.images.generate(
                            model=self.model,
                            prompt=prompt,
                            n=1,
                            response_format="b64_json",
                        )
                    except Exception as b64_err:
                        # Only fall back for client errors (400/422 = unsupported param);
                        # propagate transient/auth errors immediately
                        if not _is_client_error(b64_err):
                            raise
                        logger.info(
                            f"[{self.session_id}][Painter] Images API with b64_json failed ({b64_err}), "
                            f"trying without response_format"
                        )
                        response = await self.llm_client.client.images.generate(
                            model=self.model,
                            prompt=prompt,
                            n=1,
                        )

                # Debug: log raw response for diagnosis
                raw = None
                if hasattr(response, 'model_dump'):
                    raw = response.model_dump()
                    logger.info(f"[{self.session_id}][Painter] Images API raw response keys: {list(raw.keys())}")
                    logger.debug(f"[{self.session_id}][Painter] Images API raw response: {raw}")
                else:
                    logger.debug(f"[{self.session_id}][Painter] Images API response type: {type(response)}, attrs: {dir(response)}")

                # Extract image from response
                if response.data and len(response.data) > 0:
                    img_item = response.data[0]
                    # Prefer b64_json (OpenAI, Google AI Studio)
                    if hasattr(img_item, 'b64_json') and img_item.b64_json:
                        return img_item.b64_json
                    # Fall back to URL and download (SiliconFlow returns temporary URLs)
                    if hasattr(img_item, 'url') and img_item.url:
                        logger.info(f"[{self.session_id}][Painter] Downloading image from URL")
                        return await self._download_image_as_base64(img_item.url)

                # SiliconFlow non-standard: 'images' array
                # Zhipu / standard: 'data' array
                if raw is not None:

                    # Check content_filter for 审核拦截
                    if 'content_filter' in raw and raw['content_filter']:
                        logger.warning(f"[{self.session_id}][Painter] Content filter triggered: {raw['content_filter']}")

                    # SiliconFlow non-standard: 'images' array
                    if 'images' in raw and raw['images']:
                        for img in raw['images']:
                            url = img.get('url', '') if isinstance(img, dict) else ''
                            if url:
                                logger.info(f"[{self.session_id}][Painter] Downloading image from non-standard response")
                                return await self._download_image_as_base64(url)

                    # Zhipu / standard: 'data' array
                    if 'data' in raw and raw['data']:
                        img_data = raw['data'][0]
                        if isinstance(img_data, dict):
                            url = img_data.get('url', '')
                            if url:
                                logger.info(f"[{self.session_id}][Painter] Downloading image from standard response (data[0].url)")
                                return await self._download_image_as_base64(url)

                raise ValueError("Images API returned no image data")
            except Exception as images_error:
                logger.warning(
                    f"[{self.session_id}][Painter] Images API also failed: {images_error}"
                )
                # Raise the original chat error as primary, mention images API failure
                raise ValueError(
                    f"Image generation failed. "
                    f"Chat completions: {chat_error}. "
                    f"Images API: {images_error}"
                ) from chat_error

        except ValueError:
            raise
        except Exception as e:
            logger.exception(f"[{self.session_id}][Painter] Error generating image")
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
            logger.exception(f"[{self.session_id}][Painter] Process failed")
            return {
                "graphic_abstract": None,
                "success": False,
                "error": str(e)
            }
