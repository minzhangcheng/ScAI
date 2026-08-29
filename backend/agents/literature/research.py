"""Top-level research drivers and search-result evaluation.

Moved verbatim out of backend/agents/literature_agent.py; composed into LiteratureAgent.
"""
from typing import Dict, Any, List, Optional
import json
import logging


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


class ResearchMixin:
    """Top-level research drivers and search-result evaluation."""

    async def _do_research(
        self,
        topic: str,
        state: Optional[AnalysisState] = None,
        pubmed_max_results: Optional[int] = None,
        current_round_findings: Optional[List[str]] = None,
        mode_override: Optional[str] = None,
        research_brief: Optional[str] = None
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

        try:
            if mode == "deep_research":
                result = await self._deep_research_via_chat(topic, state, research_brief=research_brief)
            elif mode == "gemini_deep_research":
                result = await self._multi_round_gemini(topic, state, research_brief=research_brief)
            elif mode == "pubmed_llm":
                result = await self._multi_round_pubmed(
                    topic, state, pubmed_max_results=pubmed_max_results,
                    current_round_findings=current_round_findings
                )
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

            # Fallback: deep_research or gemini → try pubmed
            if mode in ("gemini_deep_research", "deep_research"):
                logger.info("[Literature] Falling back to PubMed + LLM mode")
                try:
                    result = await self._multi_round_pubmed(
                        topic, state, pubmed_max_results=pubmed_max_results,
                        current_round_findings=current_round_findings
                    )
                    result["search_mode"] = f"pubmed_llm (fallback from {mode})"
                    return result
                except UpstreamExhaustedError:
                    raise
                except Exception as e2:
                    logger.warning(f"[Literature] PubMed fallback also failed: {e2}")

            # Final fallback: quick search (LLM knowledge only)
            logger.info("[Literature] Falling back to quick search (LLM knowledge only)")
            result = await self._quick_search(topic, state)
            result["search_mode"] = f"quick_search (fallback from {mode})"
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

    async def integration_literature_search(self, state, integration_report: str) -> str:
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
        )
        if isinstance(result, dict):
            findings = result.get("findings") or []
            summary = result.get("summary") or ""
            return (summary + "\n" + "\n".join(str(f) for f in findings)).strip()
        return str(result)
