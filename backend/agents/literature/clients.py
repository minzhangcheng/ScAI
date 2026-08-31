"""Deep-research client lifecycle, fallback chat, and model-label resolution.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Dict, Any, Optional, Tuple
import asyncio
import logging

import httpx
from openai import APIConnectionError as OpenAIConnectionError


logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ...utils.llm_client import OpenAICompatibleClient, GoogleDeepResearchClient
from openai import RateLimitError

from .shared import (  # noqa: F401
    _FALLBACK_EXCEPTIONS,
    _MAX_MAIN_RETRIES,
    _RETRY_DELAY_BASE,
    plausibility_terms_for_tissue,
    _gene_case_variants,
    _sanitize_synonyms,
    status_from_judge,
    build_focused_query,
)


class ClientsMixin:
    """Deep-research client lifecycle, fallback chat, and model-label resolution."""

    async def _chat_with_fallback(self, prompt: str, temperature: float = 0.3) -> Tuple[str, bool]:
        """Returns (response, used_fallback)."""
        # 主模型重试 N 次
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_MAIN_RETRIES):
            try:
                return await self.chat_without_history(prompt, temperature=temperature), False
            except _FALLBACK_EXCEPTIONS as e:
                last_err = e
                if isinstance(e, RateLimitError):
                    # RateLimitError 已由内层 retry_with_backoff 做了充分退避，外层继续重试无意义
                    logger.warning(
                        f"[Literature] LLM call hit RateLimitError on attempt {attempt + 1}/{_MAX_MAIN_RETRIES}, skipping to fallback"
                    )
                    break  # 跳出主模型重试循环，走 fallback
                if attempt < _MAX_MAIN_RETRIES - 1:
                    delay = _RETRY_DELAY_BASE * (attempt + 1)
                    logger.warning(
                        f"[Literature] LLM call failed (attempt {attempt + 1}/{_MAX_MAIN_RETRIES}): "
                        f"{type(e).__name__}: {str(e)[:100]}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # 最后一次失败（且非 RateLimitError），记录日志
                    logger.warning(
                        f"[Literature] LLM call failed final attempt ({attempt + 1}/{_MAX_MAIN_RETRIES}): "
                        f"{type(e).__name__}: {str(e)[:100]}"
                    )

        # 所有主模型重试都失败，尝试 fallback model
        if self.fallback_llm_client and self.fallback_model:
            logger.warning(
                f"[Literature] Main model failed after {_MAX_MAIN_RETRIES} attempts, "
                f"trying fallback model: {self.fallback_model}"
            )
            try:
                resp = await self._budgeted_text_call(
                    self.fallback_llm_client.chat,
                    model=self.fallback_model,
                    messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                    temperature=temperature,
                    prefer_stream=True,
                )
                if not resp.content:
                    logger.warning("[Literature] Fallback model returned empty content")
                    raise last_err or ValueError("Fallback model returned empty content")
                return resp.content, True
            except Exception as fallback_err:
                # fallback 也失败时，记录 fallback 的真实错误
                logger.error(
                    f"[Literature] Fallback model also failed: "
                    f"{type(fallback_err).__name__}: {str(fallback_err)[:150]}"
                )
                raise fallback_err from last_err

        # 无 fallback 时抛出主模型的最后错误
        raise last_err or ValueError("LLM call failed")

    def _close_stale_client(self, client) -> None:
        if client is None or not hasattr(client, "close"):
            return

        # Keep a strong ownership record until close succeeds. Runtime config
        # refresh is synchronous, so the actual close may run in a background
        # task; a transient failure must remain retryable during workflow close.
        self._retired_runtime_clients[id(client)] = client
        self._schedule_retired_runtime_client_close(client)

    def _lease_runtime_client(self, client) -> None:
        if client is None:
            return
        provider_lease = getattr(self._provider, "lease_client", None)
        if provider_lease:
            provider_lease(client)
        client_id = id(client)
        self._runtime_client_use_counts[client_id] = (
            self._runtime_client_use_counts.get(client_id, 0) + 1
        )

    def _release_runtime_client(self, client) -> None:
        if client is None:
            return
        client_id = id(client)
        provider_release = getattr(self._provider, "release_client", None)
        if provider_release:
            provider_release(client)
        remaining = self._runtime_client_use_counts.get(client_id, 0) - 1
        if remaining > 0:
            self._runtime_client_use_counts[client_id] = remaining
            return
        self._runtime_client_use_counts.pop(client_id, None)
        if self._retired_runtime_clients.get(client_id) is client:
            self._schedule_retired_runtime_client_close(client)

    def _schedule_retired_runtime_client_close(self, client) -> None:
        client_id = id(client)
        if self._runtime_client_use_counts.get(client_id, 0) > 0:
            return
        if client_id in self._client_close_task_ids:
            return

        async def _close():
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"[Literature] Failed to close stale runtime client: {e}")
            else:
                if self._retired_runtime_clients.get(id(client)) is client:
                    self._retired_runtime_clients.pop(id(client), None)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_close())
        else:
            task = loop.create_task(_close())
            self._client_close_tasks.add(task)
            self._client_close_task_ids.add(client_id)

            def _discard(done_task):
                self._client_close_tasks.discard(done_task)
                self._client_close_task_ids.discard(client_id)

            task.add_done_callback(_discard)

    def _deep_research_client_signature(self) -> Tuple[str, str]:
        return (
            self.deep_research_llm_base_url or "",
            self.deep_research_api_key or "",
        )

    async def _deep_research_via_chat(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        research_brief: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deep research via OpenRouter chat API (OpenAI o4, Perplexity sonar, etc.).

        Makes a single long-running chat call with high max_tokens and timeout.
        The deep research model handles web search and synthesis internally.
        """
        self.refresh_runtime_config_if_needed()
        # Freeze one coherent request configuration before the first await.
        # A retry must not combine an old endpoint/key client with a model or
        # system prompt that changed while the request was in flight.
        request_model = self.deep_research_model
        request_system_prompt = self.system_prompt
        if not request_model:
            raise ValueError("deep_research_model not configured for deep_research mode")

        if research_brief:
            body = research_brief
        else:
            body = (
                "Conduct a literature review on the following topic.\n\n"
                f"Research topic: {topic}"
            )

        prompt = f"{body}\n\n{self._DEEP_RESEARCH_CITATION_FOOTER}"

        logger.info(
            f"[Literature] Deep Research via {request_model}: "
            f"{topic[:200]}"
        )

        self._set_llm_context()

        # Use separate deep research LLM client if configured
        dr_client = self._deep_research_llm_client or self.llm_client
        self._lease_runtime_client(dr_client)

        # The LLM client disables retries for timeout>300s (to avoid re-running
        # expensive long requests). Retry transient *connection* errors here.
        _max_conn_retries = 2
        _last_err: Optional[Exception] = None
        try:
            for _attempt in range(_max_conn_retries + 1):
                try:
                    response = await self._budgeted_text_call(
                        dr_client.chat,
                        model=request_model,
                        messages=[
                            {"role": "system", "content": request_system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=None,
                        timeout=1800.0,
                        tools=[{"type": "web_search_preview"}],
                    )
                    break
                except (OpenAIConnectionError, httpx.ConnectError, httpx.ConnectTimeout, OSError) as e:
                    _last_err = e
                    if _attempt < _max_conn_retries:
                        _delay = 5 * (_attempt + 1)
                        logger.warning(
                            f"[Literature] Deep research connection error "
                            f"(attempt {_attempt + 1}/{_max_conn_retries + 1}): "
                            f"{type(e).__name__}: {str(e)[:150]}. Retrying in {_delay}s..."
                        )
                        await asyncio.sleep(_delay)
                    else:
                        raise
            else:
                raise _last_err  # type: ignore[misc]
        finally:
            self._release_runtime_client(dr_client)

        report_text = response.content or ""
        if not report_text:
            raise ValueError("Deep research returned empty response")

        logger.info(
            f"[Literature] Deep Research completed: "
            f"{len(report_text)} chars"
        )

        references = self._select_references(response, report_text)
        return await self._finalize_result(topic, report_text, references, state, 1)

    def _make_deep_research_client(self) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient(
            api_key=self.deep_research_api_key,
            base_url=self.deep_research_llm_base_url,
            proxy_url=self.proxy_url,
            session_id=self.session_id,
            agent_name=self.agent_type.value,
        )
        apply_retry_callback = getattr(self._provider, "apply_retry_callback", None)
        if apply_retry_callback:
            apply_retry_callback(client)
        return client

    def _make_google_client(self) -> GoogleDeepResearchClient:
        return GoogleDeepResearchClient(
            self._google_api_key,
            proxy_url=self.proxy_url,
        )

    async def _multi_round_gemini(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        research_brief: Optional[str] = None
    ) -> Dict[str, Any]:
        """Single Gemini Deep Research call via Google Interactions API.

        Deep Research handles multi-round search/synthesis internally on Google's
        servers, so we only need one call per topic. Gap coverage (if any) is
        handled at the workflow level by falling back to PubMed+LLM.
        """
        if not self._deep_research:
            raise ValueError("Google API Key not configured for Gemini Deep Research")

        query = research_brief if research_brief else topic
        logger.info(f"[Literature] Gemini Deep Research: {query[:200]}")

        deep_research_client = self._deep_research
        self._lease_runtime_client(deep_research_client)
        try:
            result = await deep_research_client.research(query)
        finally:
            self._release_runtime_client(deep_research_client)
        report_text = result.get("report", "")
        sources = result.get("sources", [])

        references = [
            {"title": s.get("title", ""), "url": s.get("uri", "")}
            for s in sources
        ]

        return await self._finalize_result(topic, report_text, references, state, 1)

    def _sync_literature_config_from_provider(self) -> None:
        if self._runtime_clients_closing:
            raise RuntimeError("literature runtime clients are closing")
        lit_config = self._provider.get_literature_config() or {}

        new_mode = lit_config.get("mode", self.literature_mode)
        new_model = lit_config.get("deep_research_model", self.deep_research_model)
        new_base_url = lit_config.get("deep_research_llm_base_url", self.deep_research_llm_base_url)
        new_api_key = lit_config.get("deep_research_api_key", self.deep_research_api_key)
        new_google_api_key = lit_config.get("google_api_key", self._google_api_key)
        new_pubmed_email = lit_config.get("pubmed_email", self.pubmed_email)
        new_max_rounds = lit_config.get("max_search_rounds", self.max_search_rounds)

        old_client_signature = self._deep_research_client_signature()

        self.literature_mode = new_mode
        self.deep_research_model = new_model
        self.deep_research_llm_base_url = new_base_url
        self.deep_research_api_key = new_api_key
        self.pubmed_email = new_pubmed_email
        self.max_search_rounds = max(1, min(50, int(new_max_rounds or 20)))

        new_client_signature = self._deep_research_client_signature()
        if new_client_signature != old_client_signature:
            old_client = self._deep_research_llm_client
            self._deep_research_llm_client = None
            self._close_stale_client(old_client)
            if self.deep_research_llm_base_url and self.deep_research_api_key:
                self._deep_research_llm_client = self._make_deep_research_client()

        if new_google_api_key != self._google_api_key:
            old_client = self._deep_research
            self._deep_research = None
            self._close_stale_client(old_client)
            self._google_api_key = new_google_api_key
            if new_google_api_key:
                self._deep_research = self._make_google_client()

    async def close_runtime_clients(self) -> None:
        """Close literature-owned clients that are not owned by AgentSettingsProvider."""
        async with self._runtime_client_cleanup_lock:
            if self._runtime_clients_closed:
                return
            self._runtime_clients_closing = True

            for client in (
                self._deep_research_llm_client,
                self._deep_research,
            ):
                if client is not None and hasattr(client, "close"):
                    self._retired_runtime_clients[id(client)] = client
            self._deep_research_llm_client = None
            self._deep_research = None

            while True:
                if self._client_close_tasks:
                    pending_tasks = list(self._client_close_tasks)
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    self._client_close_tasks.difference_update(pending_tasks)

                failures: list[Exception] = []
                in_use = 0
                for client_id, client in list(
                    self._retired_runtime_clients.items()
                ):
                    if self._runtime_client_use_counts.get(client_id, 0) > 0:
                        in_use += 1
                        continue
                    # A lease may have been released while another close was
                    # awaited. Let its tracked task finish before this cleanup
                    # retries the same client.
                    if client_id in self._client_close_task_ids:
                        continue
                    try:
                        await client.close()
                    except Exception as exc:
                        failures.append(exc)
                        logger.debug(
                            "[Literature] Failed to close runtime client: %s", exc
                        )
                    else:
                        if self._retired_runtime_clients.get(client_id) is client:
                            self._retired_runtime_clients.pop(client_id, None)

                if failures:
                    raise RuntimeError(
                        f"failed to close {len(failures)} literature runtime client(s)"
                    ) from failures[0]
                if in_use:
                    raise RuntimeError(
                        f"{in_use} literature runtime client(s) still in use"
                    )
                if self._client_close_tasks:
                    continue
                if self._retired_runtime_clients:
                    # A tracked task callback can run just after the task set
                    # was sampled. Yield once, then drain the retained client.
                    await asyncio.sleep(0)
                    continue

                self._runtime_clients_closed = True
                return

    @property
    def fallback_llm_client(self) -> OpenAICompatibleClient:
        return self._provider.get_llm_client("pi")

    @property
    def fallback_model(self) -> str:
        return self._provider.get_model("pi")

    @property
    def is_deep_research_available(self) -> bool:
        """Check if deep research mode is available and properly configured."""
        if self.literature_mode not in ("gemini_deep_research", "deep_research"):
            return False
        return self._deep_research is not None or bool(self.deep_research_model)

    def on_runtime_config_changed(
        self,
        previous_revision: Optional[int],
        current_revision: int,
    ) -> None:
        self._sync_literature_config_from_provider()
        self._annotation_cache.clear()

    def resolve_model_label(self, search_mode: str) -> str:
        """Map a ``search_mode`` string (from _do_research) to a display label
        for the final report's Literature row.

        - gemini_deep_research -> "Google Deep Research"
        - deep_research        -> the configured DR model (or lit model if unset)
        - everything else (pubmed_llm, quick_search, "(fallback from ...)") ->
          the literature synthesis model (matches the legacy report value)
        """
        mode = (search_mode or "").strip()
        if mode == "gemini_deep_research":
            return "Google Deep Research"
        if mode == "deep_research":
            return self.deep_research_model or self.model
        return self.model
