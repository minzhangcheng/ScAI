"""Top-level research drivers and search-result evaluation.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Callable, Dict, Any, List, Optional
import asyncio
import json
import logging
import time


logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ...utils.llm_client import UpstreamExhaustedError

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


def literature_technical_failure(result: Dict[str, Any]) -> Dict[str, Any]:
    """Project explicit technical failures without judging scientific sufficiency."""
    search_meta = result.get("search_meta") or {}
    if not (
        result.get("source_failure")
        or result.get("source_processing_error")
        or search_meta.get("evaluation_failed")
        or result.get("technical_failures")
    ):
        return {}
    return {
        "status": result.get("research_status") or result.get("status") or "partial",
        **{
            key: result[key]
            for key in (
                "source_failure", "source_processing_error", "source_provenance",
                "search_meta", "technical_failures",
            )
            if key in result
        },
    }


def literature_provenance_disclosure(result: Dict[str, Any]) -> str:
    """Keep source and execution facts visible in text-only PI projections."""
    metadata = {
        key: str(result[key])[:1000]
        for key in (
            "research_status", "status", "search_mode", "source_provenance",
            "source_processing_error",
        )
        if key in result
    }
    if result.get("source_failure"):
        failure = result["source_failure"]
        metadata["source_failure"] = (
            {key: str(failure[key])[:1000] for key in ("mode", "reason") if key in failure}
            if isinstance(failure, dict) else str(failure)[:1000]
        )
    if result.get("search_meta"):
        metadata["search_meta"] = {
            key: result["search_meta"][key]
            for key in ("stop_reason", "evaluation_failed", "evidence_strength")
            if key in result["search_meta"]
        }
    if not metadata:
        return ""
    return "Literature research provenance:\n" + json.dumps(metadata, ensure_ascii=False)


class ResearchMixin:
    """Top-level research drivers and search-result evaluation."""

    async def _do_research(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        pubmed_max_results: Optional[int] = None,
        current_round_findings: Optional[List[str]] = None,
        mode_override: Optional[str] = None,
        research_brief: Optional[str] = None,
        *,
        pi_plan_fn: Optional[Callable] = None,
        pi_review_fn: Optional[Callable] = None,
        research_context: str = "",
    ) -> Dict[str, Any]:
        """
        Route to configured research mode with fallback chain.

        Fallback: deep_research/gemini → pubmed → quick_search

        Args:
            topic: Research topic
            state: Optional analysis state for context
            pubmed_max_results: Override max articles per PubMed search round (default 20)
            current_round_findings: Findings from current round's other agents
            mode_override: Override self.literature_mode for this call
        """
        mode = mode_override if mode_override is not None else self.literature_mode
        actual_mode = mode  # Track which mode actually produced results
        context_parts = [str(research_context or "").strip()]
        if research_brief:
            context_parts.append(f"Research brief:\n{str(research_brief)[:8000]}")
        if current_round_findings:
            context_parts.append(
                "Current findings:\n" + "\n".join(
                    str(item)[:1000] for item in current_round_findings[:10]
                )
            )
        research_context = "\n\n".join(part for part in context_parts if part)[:16000]

        async def run_pubmed() -> Dict[str, Any]:
            if not (pi_plan_fn and pi_review_fn):
                return await self._multi_round_pubmed(
                    topic, state, pubmed_max_results=pubmed_max_results,
                    current_round_findings=current_round_findings,
                    research_context=research_context,
                )
            budget = float(getattr(self, "search_wall_clock_seconds", 600) or 600)
            plan = await self._await_literature_llm(
                pi_plan_fn(
                    topic=topic,
                    context=research_context,
                    state=state,
                    mode="pubmed",
                ),
                timeout_seconds=budget,
            )
            # Planning has its own LLM budget; retrieval starts after it settles.
            deadline = time.monotonic() + budget
            primary_value = plan.get("primary_query") if isinstance(plan, dict) else None
            primary = primary_value.strip() if isinstance(primary_value, str) else ""
            raw_alternatives = plan.get("alternative_queries") if isinstance(plan, dict) else []
            alternatives = [
                value.strip() for value in (raw_alternatives or [])
                if isinstance(value, str) and value.strip()
            ][:2]
            if not primary:
                raise ValueError("PI literature search plan contained no primary query")

            async def evaluate(articles, search_history):
                nonlocal deadline
                review_started = time.monotonic()
                try:
                    return await pi_review_fn(
                        topic=topic,
                        context=research_context,
                        articles=articles,
                        search_history=search_history,
                        remaining_budget_seconds=max(0.0, deadline - time.monotonic()),
                        state=state,
                        mode="pubmed",
                    )
                finally:
                    deadline += max(0.0, time.monotonic() - review_started)

            return await self._multi_round_pubmed(
                topic, state, pubmed_max_results=pubmed_max_results,
                current_round_findings=current_round_findings,
                initial_query=primary,
                evaluate_fn=evaluate,
                seed_queries=alternatives,
                research_context=research_context,
                planned_queries=[primary, *alternatives],
                deadline_override=deadline,
            )

        try:
            if mode == "deep_research":
                result = await self._deep_research_via_chat(topic, state, research_brief=research_brief)
            elif mode == "gemini_deep_research":
                result = await self._multi_round_gemini(topic, state, research_brief=research_brief)
            elif mode == "pubmed_llm":
                result = await run_pubmed()
            else:
                logger.warning(f"[Literature] Unknown mode: {mode}, using quick search")
                actual_mode = "quick_search"
                result = await self._quick_search(topic, state)
            result["search_mode"] = actual_mode
            return result
        except UpstreamExhaustedError:
            # The LLM upstream is genuinely exhausted (all retries failed).
            # Do NOT silently degrade to quick_search — every fallback mode
            # also needs the same LLM, so degrading just yields a low-quality
            # result while hiding the failure. Propagate so the workflow's
            # LLM-failure HITL gate lets the user choose retry/skip/stop.
            raise
        except Exception as e:
            logger.warning(f"[Literature] Mode {mode} failed: {e}")
            final_failure = e
            failed_source_mode = mode

            # Fallback: deep_research or gemini → try pubmed
            if mode in ("gemini_deep_research", "deep_research"):
                logger.info("[Literature] Falling back to PubMed + LLM mode")
                try:
                    result = await run_pubmed()
                    result["search_mode"] = f"pubmed_llm (fallback from {mode})"
                    return result
                except UpstreamExhaustedError:
                    raise
                except Exception as e2:
                    logger.warning(f"[Literature] PubMed fallback also failed: {e2}")
                    final_failure = e2
                    failed_source_mode = f"pubmed_llm (fallback from {mode})"

            # Final fallback: quick search (LLM knowledge only)
            logger.info("[Literature] Falling back to quick search (LLM knowledge only)")
            result = await self._quick_search(topic, state)
            result["search_mode"] = f"quick_search (fallback from {mode})"
            result["source_provenance"] = "knowledge_only"
            result["status"] = "partial"
            result["source_failure"] = {
                "mode": str(failed_source_mode),
                "reason": f"{type(final_failure).__name__}: {final_failure}"[:1000],
            }
            return result

    async def _evaluate_search_results(
        self,
        topic: str,
        search_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate whether to keep searching (Gemini/OpenRouter modes).

        Kept on the same contract as the PubMed evaluator: "stop searching" and
        "the evidence supports a conclusion" are separate judgements, and a reply
        we cannot read is neither.
        """
        history_summary = json.dumps(search_history, ensure_ascii=False, indent=2)

        prompt = f"""Evaluate whether the search results so far are sufficient for a comprehensive
literature review on the topic:

Topic: {topic}

Search history:
{history_summary[:8000]}

Respond in JSON format:
{{
    "search_exhausted": true or false,
    "evidence_strength": "strong | moderate | sparse | contradictory | insufficient",
    "reasoning": "why the search is or is not exhausted, and how strong the evidence is",
    "refined_query": "suggested refined search query if the search is not exhausted",
    "missing_aspects": ["aspects not yet covered"]
}}"""

        response = await self.chat_without_history(prompt, temperature=0.3)
        result = self._safe_parse_json(response)
        if not result or "search_exhausted" not in result:
            return {
                "search_exhausted": False,
                "evidence_strength": "unassessed",
                "evaluation_failed": True,
                "reasoning": (
                    "The search evaluation could not be read. This says nothing about "
                    "whether the retrieved literature supports any conclusion."
                ),
            }
        return result

    async def integration_literature_search(
        self, state, integration_report: str, *, pi_plan_fn=None, pi_review_fn=None
    ) -> str:
        balanced_framing = (
            "For each contestable claim below, deliberately search for credible contradicting, "
            "qualifying, null, or non-replicating evidence as well as supporting work. Weight "
            "and report what you find by methodological quality, directness, and how well the "
            "model, tissue and species apply here — not by how many papers sit on each side, "
            "and not by giving the two equal space regardless of quality. A well-established "
            "descriptive fact needs no manufactured counterpoint. Do not assemble a one-sided "
            "case, and state the conditions under which each line of evidence applies."
        )
        topic = (
            f"Integrative synthesis for {state.name}: {state.initial_instruction}. "
            f"{balanced_framing}"
        )
        # The brief, not the topic, is what a Deep Research provider receives when
        # both are supplied, so the framing has to travel inside the brief itself.
        result = await self._do_research(
            topic=topic,
            state=state,
            research_brief=(
                f"{balanced_framing}\n\n"
                f"Findings to assess:\n{integration_report[:2000]}"
            ),
            research_context=(
                f"Project question: {state.initial_instruction}\n\n"
                f"Current integration findings:\n{integration_report[:6000]}"
            ),
            pi_plan_fn=pi_plan_fn,
            pi_review_fn=pi_review_fn,
        )
        if isinstance(result, dict):
            findings = result.get("findings") or []
            summary = result.get("summary") or ""
            evidence = (summary + "\n" + "\n".join(str(f) for f in findings)).strip()
            if not evidence:
                evidence = str(result.get("report") or "")
            return "\n\n".join(
                part for part in (literature_provenance_disclosure(result), evidence) if part
            )
        return str(result)
