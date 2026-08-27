"""Painter response image extraction and download helpers."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

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
