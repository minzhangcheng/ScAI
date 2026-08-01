# Google Deep Research client and its background-poll settings.
# Moved verbatim out of backend/utils/llm_client.py.
"""
LLM client utilities using OpenAI SDK for OpenAI-compatible APIs
Supports OpenRouter, OpenAI, and other compatible endpoints
"""
import base64
import html
import httpx
import inspect
import asyncio
import json
import logging
import re
import time
from uuid import uuid4
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError, InternalServerError, BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError

from ..llm_logger import get_llm_logger, LLMInteraction

logger = logging.getLogger(__name__)


# Background-mode polling for deep-research Responses calls. These models run for
# several minutes; a blocking responses.create() keeps the connection idle long
# enough for an upstream gateway (e.g. Cloudflare in front of api.n1n.ai) to sever
# it with HTTP 524 while the job keeps running server-side. Submitting with
# background=True returns immediately, then we poll responses.retrieve() with short
# requests that never trip the gateway's idle timeout.
_DR_BACKGROUND_POLL_INTERVAL = 5.0     # seconds between retrieve() polls

_DR_BACKGROUND_MAX_WAIT = 1800.0       # 30 min ceiling for a single DR task

_DR_BACKGROUND_MAX_POLL_ERRORS = 5     # consecutive transient poll failures before giving up

_DR_BACKGROUND_TERMINAL_FAILED = ("failed", "cancelled", "canceled", "incomplete", "expired")

class GoogleDeepResearchClient:
    """Client for Google Gemini Deep Research via Interactions API"""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    DEFAULT_MODEL = "deep-research-preview-04-2026"
    API_REVISION = "2026-05-20"
    GROUNDING_REDIRECT_MARKER = "grounding-api-redirect"

    def __init__(self, api_key: str, proxy_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self._default_headers = {
            "x-goog-api-key": self.api_key,
            "Api-Revision": self.API_REVISION,
        }
        # httpx will also read from HTTP_PROXY/HTTPS_PROXY env vars if trust_env=True (default)
        if proxy_url:
            self.client = httpx.AsyncClient(
                timeout=600.0,
                proxy=proxy_url
            )
        else:
            self.client = httpx.AsyncClient(timeout=600.0)  # Long timeout for research
        self._closed: bool = False

    def __del__(self):
        if not getattr(self, '_closed', True):
            logger.warning(
                f"[DeepResearch] GoogleDeepResearchClient leaked without close()"
            )

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.client.aclose()

    def _check_response(self, response: httpx.Response, context: str) -> None:
        """Raise HTTPStatusError for 4xx/5xx responses with logging.

        Matches raise_for_status() behavior (raises on 4xx/5xx only, lets 3xx
        pass through) while adding structured warning logs for debugging.
        """
        if response.is_error:
            body = response.text
            logger.warning(
                f"[DeepResearch] {context} {response.status_code} from {response.url} "
                f"| response_body={body[:2000]}"
            )
            raise httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase} from {response.url}: {body[:1000]}",
                request=response.request,
                response=response,
            )

    async def start_research(
        self,
        query: str,
        background: bool = True
    ) -> str:
        """
        Start a Deep Research task

        Args:
            query: Research query
            background: Whether to run in background

        Returns:
            Interaction ID
        """
        url = f"{self.BASE_URL}/interactions"

        payload = {
            "agent": self.model,
            "input": query,
            "background": background,
            "agent_config": {
                "type": "deep-research",
                "thinking_summaries": "auto",
                # "off" disables agent-generated charts/images. We only consume
                # the textual report, so suppressing inline images keeps the
                # response light and the parser simple.
                "visualization": "off",
                "collaborative_planning": False,
            },
        }

        response = await self.client.post(
            url,
            headers=self._default_headers,
            json=payload,
        )
        self._check_response(response, f"start_research model={self.model} api_revision={self.API_REVISION}")

        data = response.json()
        # Handle both "id" field and "name" resource path format
        return data.get("id") or data.get("name", "").split("/")[-1]
    
    async def poll_research(
        self,
        interaction_id: str,
        poll_interval: float = 5.0,
        max_wait: float = 1800.0  # 30 minutes max
    ) -> Dict[str, Any]:
        """
        Poll for research results
        
        Args:
            interaction_id: Interaction ID from start_research
            poll_interval: Seconds between polls
            max_wait: Maximum seconds to wait
            
        Returns:
            Research results
        """
        url = f"{self.BASE_URL}/interactions/{interaction_id}"
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            try:
                response = await self.client.get(
                    url,
                    headers=self._default_headers,
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                logger.warning(
                    f"[DeepResearch] Poll transient error, will retry: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                await asyncio.sleep(poll_interval)
                continue
            self._check_response(response, f"poll_research interaction_id={interaction_id}")
            
            data = response.json()
            
            # Check if completed (handle both "status" and legacy "state", case-insensitive)
            status = data.get("status", data.get("state", "")).lower()
            if status == "completed":
                return self._extract_research_result(data)
            elif status == "failed":
                raise Exception(f"Research failed: {data.get('error', 'Unknown error')}")
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError("Research timed out")
    
    def _extract_research_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research result from Interactions API response.

        Current schema (Api-Revision 2026-05-20, agent deep-research-preview-04-2026):
        the interaction holds a `steps[]` timeline. The final report lives in
        `model_output` steps as `content[]` blocks of `{"type": "text", "text",
        "annotations": [...]}`; planning summaries live in `thought` steps as a
        `summary[]` array. Citations are `url_citation` annotations on the text
        blocks. Legacy `output_text`/`outputs[]`/`output` paths are kept as
        defensive fallbacks for older revisions.
        """
        result = {
            "report": "",
            "sources": [],
            "thinking": []
        }
        seen_uris = set()

        def _add_source(uri: str, title: str = "") -> None:
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                result["sources"].append({"title": title, "uri": uri})

        # Current schema: steps[] timeline
        for step in data.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type", "")

            if step_type == "thought":
                # Planning / reasoning summaries: summary is a list of text parts.
                for part in step.get("summary", []) or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                        result["thinking"].append({"type": step_type, "text": part["text"]})

            elif step_type == "model_output":
                # Final report text + inline citations live in content[] text blocks.
                for block in step.get("content", []) or []:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    if isinstance(block.get("text"), str):
                        result["report"] += block["text"]
                    for ann in block.get("annotations", []) or []:
                        if isinstance(ann, dict) and ann.get("type") == "url_citation":
                            _add_source(ann.get("url", ""), ann.get("title", ""))

        # Legacy fallback: top-level output_text
        if not result["report"]:
            output_text = data.get("output_text")
            if isinstance(output_text, str) and output_text:
                result["report"] = output_text

        # Legacy fallback: steps[].text / steps[].parts[].text + groundingMetadata
        if not result["report"]:
            for step in data.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                text = step.get("text")
                if isinstance(text, str) and text:
                    result["report"] += text
                for part in step.get("parts", []) or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        result["report"] += part["text"]
                grounding = step.get("groundingMetadata") or step.get("grounding_metadata") or {}
                if grounding:
                    self._extract_sources(grounding, result["sources"])

        # Legacy fallback: outputs[]
        if not result["report"]:
            for output_item in data.get("outputs", []) or []:
                if not isinstance(output_item, dict):
                    continue
                if "text" in output_item:
                    result["report"] += output_item["text"]
                elif "parts" in output_item:
                    for part in output_item["parts"]:
                        if isinstance(part, dict) and "text" in part:
                            result["report"] += part["text"]
                grounding = output_item.get("groundingMetadata", {})
                self._extract_sources(grounding, result["sources"])

        # Legacy fallback: single output.parts[].text
        if not result["report"]:
            output = data.get("output", {})
            if isinstance(output, dict):
                for part in output.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        result["report"] += part["text"]
                grounding = output.get("groundingMetadata", {})
                self._extract_sources(grounding, result["sources"])

        # Top-level groundingMetadata (any schema)
        top_grounding = data.get("groundingMetadata") or data.get("grounding_metadata") or {}
        if top_grounding:
            self._extract_sources(top_grounding, result["sources"])

        return result

    @staticmethod
    def _extract_sources(grounding: Dict[str, Any], sources: List[Dict[str, str]]):
        """Extract source URLs from grounding metadata"""
        seen_uris = {s.get("uri") for s in sources}
        for chunk in grounding.get("groundingChunks", []):
            if "web" in chunk:
                uri = chunk["web"].get("uri", "")
                if uri and uri not in seen_uris:
                    sources.append({
                        "title": chunk["web"].get("title", ""),
                        "uri": uri
                    })
                    seen_uris.add(uri)

    async def _resolve_redirect(self, url: str) -> str:
        """Resolve a Vertex grounding-api-redirect URL to the real article URL.

        Reads the 302 ``Location`` header without following it further. Returns
        the original ``url`` unchanged for non-grounding URLs or on any failure
        (the redirect links are time-sensitive, so a miss must not break the
        literature flow). Relative ``Location`` values are joined against the
        request URL to produce an absolute URL.
        """
        if not url or self.GROUNDING_REDIRECT_MARKER not in url:
            return url
        try:
            resp = await self.client.get(
                url, follow_redirects=False, timeout=15.0
            )
            if resp.is_redirect:
                location = resp.headers.get("location")
                if location:
                    # Handle relative Location headers (e.g. "/article/123").
                    if not location.startswith(("http://", "https://")):
                        location = str(httpx.URL(resp.url).join(location))
                    return location
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            logger.warning(f"[DeepResearch] redirect resolve failed for {url[:80]}: {e}")
        return url

    _TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

    async def _fetch_title(self, url: str) -> str:
        """Fetch ``url`` and return its HTML ``<title>`` text, or '' on failure.

        Only parses ``text/html`` responses; collapses internal whitespace and
        unescapes HTML entities. A browser-ish User-Agent avoids trivial bot
        blocks. Any error yields '' so the caller can fall back.
        """
        try:
            resp = await self.client.get(
                url,
                follow_redirects=True,
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ScAI-LiteratureBot/1.0)"},
            )
            if "html" not in resp.headers.get("content-type", "").lower():
                return ""
            match = self._TITLE_RE.search(resp.text)
            if match:
                title = html.unescape(match.group(1))
                return re.sub(r"\s+", " ", title).strip()
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            logger.warning(f"[DeepResearch] title fetch failed for {url[:80]}: {e}")
        return ""

    async def _enrich_source(self, source: Dict[str, str]) -> Dict[str, str]:
        """Resolve one source's redirect URL and backfill its title if empty."""
        real_url = await self._resolve_redirect(source.get("uri", ""))
        title = source.get("title") or ""
        if not title and real_url:
            title = await self._fetch_title(real_url)
        return {"title": title, "uri": real_url}

    async def enrich_sources(
        self, sources: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Resolve redirects + backfill titles for all sources, concurrently.

        Order is preserved so the report's inline ``[cite: N]`` markers stay
        aligned with the References list. Network failures degrade to the
        original URL / empty title per-source (handled in ``_enrich_source``).
        Concurrency is capped to avoid triggering rate limits on academic
        sites (nih.gov, researchgate.net, etc.).
        """
        if not sources:
            return sources
        sem = asyncio.Semaphore(5)

        async def _bounded_enrich(s):
            async with sem:
                return await self._enrich_source(s)

        return list(await asyncio.gather(*(_bounded_enrich(s) for s in sources)))

    async def research(self, query: str) -> Dict[str, Any]:
        """
        Convenience method to start research and wait for results
        
        Args:
            query: Research query
            
        Returns:
            Research results with report and sources
        """
        interaction_id = await self.start_research(query)
        result = await self.poll_research(interaction_id)
        result["sources"] = await self.enrich_sources(result.get("sources", []))
        return result
