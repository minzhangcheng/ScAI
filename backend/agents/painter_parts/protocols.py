"""Painter provider protocol selection, retries, and attempt adapters."""
from __future__ import annotations

import re

from ...utils.llm_client_parts.retry import llm_timeout_seconds

_MAX_RETRY_ROUNDS = 5


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

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
        create_url, json=payload, headers=headers, timeout=llm_timeout_seconds()
    )
    resp.raise_for_status()
    data = resp.json()

    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise ValueError(f"DashScope image generation: no task_id in response: {data}")
    logger.info(f"[{self.session_id}][Painter] DashScope: task submitted, id={task_id}")

    poll_url = f"{_DASHSCOPE_BASE_URL}/api/v1/tasks/{task_id}"
    for attempt in range(120):  # At least ten minutes for generation (120 × 5 s).
        await asyncio.sleep(5)
        poll_resp = await runtime.llm_client.http_client.get(
            poll_url, headers=headers, timeout=30.0
        )
        poll_resp.raise_for_status()
        status_data = poll_resp.json()

        task_status = status_data.get("output", {}).get("task_status", "")
        logger.debug(
            f"[{self.session_id}][Painter] DashScope task {task_id}: status={task_status} "
            f"(attempt {attempt + 1}/120)"
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
        f"DashScope image generation timed out after 10 minutes (task_id={task_id})"
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
    protocol carrying a ``timeout`` receives at least 600 seconds. The optional
    total deadline controls whether another attempt starts; it never truncates
    a started generation to a smaller leftover budget.

    Returns the first successfully parsed base64 image; raises ValueError if all
    protocols fail across all rounds.
    """
    errors: Dict[str, str] = {}
    disabled: set[str] = set()
    loop = asyncio.get_running_loop()
    deadline = (
        loop.time() + llm_timeout_seconds(total_timeout)
        if total_timeout is not None else None
    )

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
            if timeout is not None:
                timeout = llm_timeout_seconds(timeout)
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
        from ...utils.llm_client import _detect_image_mime_type
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
    from ...utils.llm_client import _detect_image_mime_type
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
