# Literature research node, round briefs, and deep-research coverage evaluation.
# Moved verbatim out of backend/workflow/agents.py; composed into AgentNodesMixin.
"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from ...utils.llm_client import UpstreamExhaustedError
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...agents.bioinformatics_agent import count_saved_figures
from ...agents.literature.pubmed import PUBMED_QUERY_POLICY_VERSION
from ..helpers import _get_task_text, _has_stage1_keyword, _normalize_tasks, cache_hit, cache_invalidate, cache_store, decide_bioinfo_post_execution, prepend_species, safe_pi_brief, STAGE1_KEYWORDS
from ...agents.literature.research import literature_technical_failure
from ...utils.blob_store import BLOB_REF_PREFIX
from ..retry import with_retry
from ...constants import LlmFailureAction
from ..helpers import WorkflowTerminatedError, WorkflowSupersededError

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

from .shared import (  # noqa: F401
    asyncio,
    hashlib,
    json,
    logging,
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    AgentType,
    AnalysisPhase,
    AnalysisStep,
    MessageRole,
    TaskStatus,
    UpstreamExhaustedError,
    MECHANISM_EVIDENCE_POLICY,
    count_saved_figures,
    _get_task_text,
    _has_stage1_keyword,
    _normalize_tasks,
    cache_hit,
    cache_invalidate,
    cache_store,
    decide_bioinfo_post_execution,
    prepend_species,
    safe_pi_brief,
    STAGE1_KEYWORDS,
    BLOB_REF_PREFIX,
    with_retry,
    _reset_pi_instructions_for_stage2,
    _stable_code_hash,
    _STAGE1_INSTRUCTION,
    _MECHANISM_CAPTURE_INSTRUCTION,
    _build_exploration_instruction,
    _build_innovation_instruction,
    _build_stage2_instruction,
    build_phase_instruction,
)


logger = logging.getLogger(__name__)
_EXTRACTION_RECOVERY_KEY = "literature_extraction_resume_v1"


class LiteratureNodeMixin:
    """Literature research node, round briefs, and deep-research coverage evaluation."""

    def _literature_extraction_identity(self, state) -> str:
        analysis = state["analysis_state"]
        instructions = state.get("pi_instructions") or {}
        inputs = {
            "round": analysis.current_round,
            "lineage": analysis.restore_branch_id,
            "plan_id": getattr(analysis.current_plan, "id", None),
            "node_retry_id": state.get("node_retry_id"),
            "instruction": analysis.initial_instruction,
            "species": analysis.species,
            "direction": analysis.current_round_direction,
            "pi_instructions": {
                key: instructions.get(key)
                for key in ("literature_topics", "deferred_literature_topics", "overview", "deferred_literature_overview")
            },
            "results": {
                key: {
                    field: (state.get(key) or {}).get(field)
                    for field in ("findings", "status", "evidence_execution_ids")
                }
                for key in ("bioinformatics_result", "biochemistry_result", "pathology_result")
            },
        }
        return hashlib.sha256(json.dumps(
            inputs, sort_keys=True, ensure_ascii=False, default=str,
        ).encode()).hexdigest()

    async def _save_literature_extraction_state(self, analysis_state):
        from ...services.session_store import async_save_session

        try:
            state_callback = getattr(self, "state_callback", None)
            if state_callback is not None:
                # Publish through the active workflow's ownership and cache boundary.
                saved = await state_callback(analysis_state, force=True)
            else:
                saved = await async_save_session(analysis_state.id, analysis_state, force=True)
            if saved is False:
                logger.warning("[%s] Literature extraction checkpoint was not acknowledged", self.session_id)
        except (WorkflowTerminatedError, WorkflowSupersededError):
            raise
        except Exception:
            logger.warning("[%s] Literature extraction checkpoint save failed", self.session_id, exc_info=True)

    def _literature_model_label(self, search_modes) -> str:
        """Resolve a list of literature search_modes into deduplicated,
        comma-joined display labels for the report (e.g. "Google Deep Research")."""
        labels = []
        for mode in search_modes or []:
            label = self.literature_agent.resolve_model_label(mode)
            if label and label not in labels:
                labels.append(label)
        return ", ".join(labels)

    async def _generate_round_brief(self, topic, current_round_findings, state):
        """Ask PI for a research-round Deep Research brief (None on failure/empty)."""
        return await safe_pi_brief(
            lambda: self.pi_agent.generate_round_research_brief(
                topic=topic,
                current_round_findings=current_round_findings,
                state=state,
            ),
            self.session_id,
            "round",
        )

    def _safe_literature_fallback_topic(self, topic: str, species: str = "") -> str:
        """Focused topic safe to send directly to deep research when PI brief is unavailable."""
        return prepend_species(topic, species)

    def _round_brief_topic_context(
        self,
        topic: str,
        species: str,
        previous_lit_context: str,
    ) -> str:
        """Richer PI-only context for authoring a round research brief."""
        enriched = topic
        if previous_lit_context:
            enriched = (
                f"{topic}\n{previous_lit_context}\n"
                f"Focus on aspects NOT already covered above."
            )
        return prepend_species(enriched, species)

    @with_retry("literature_research")
    async def _literature_research(self, state: WorkflowState) -> WorkflowState:
        """Literature agent researches relevant topics.

        Deep Research mode: merges all topics into one comprehensive query and
        makes a single DR call. If some topics are insufficiently covered,
        falls back to PubMed+LLM for the gaps.

        PubMed mode: searches each topic individually (up to 3).
        """
        analysis_state = state["analysis_state"]

        skip = await self._require_core_agent(state, "literature", "literature_research")
        if skip is not None:
            return skip

        extraction_identity = self._literature_extraction_identity(state)
        recovery = (analysis_state.sandbox_state or {}).get(_EXTRACTION_RECOVERY_KEY)
        if isinstance(recovery, dict) and recovery.get("identity") == extraction_identity:
            return await self._finish_literature_extraction(state, recovery)
        if isinstance(recovery, dict):
            # The older round cache has no input identity; it must not override
            # an explicit change to the frozen extraction's research inputs.
            cache_invalidate(analysis_state, AgentType.LITERATURE)
            analysis_state.sandbox_state.pop(_EXTRACTION_RECOVERY_KEY, None)

        cached = cache_hit(analysis_state, AgentType.LITERATURE)
        if cached is not None and (
            literature_technical_failure(cached)
            or any(
                str(mode).startswith("quick_search (fallback from ")
                for mode in (cached.get("search_modes") or [])
            )
        ):
            cache_invalidate(analysis_state, AgentType.LITERATURE)
            cached = None
        if cached is not None:
            cached_uses_pubmed = any(
                "pubmed" in str(mode).lower()
                for mode in (cached.get("search_modes") or [])
            )
            if (
                cached_uses_pubmed
                and cached.get("query_policy_version") != PUBMED_QUERY_POLICY_VERSION
            ):
                logger.info(
                    f"[{self.session_id}] stale literature query-policy cache; "
                    "re-executing literature research"
                )
                cache_invalidate(analysis_state, AgentType.LITERATURE)
                cached = None
        if cached is not None:
            logger.info(f"[{self.session_id}] literature cache hit, skipping re-execution")
            # Restore the actually-used model label so the report stays accurate
            # even when literature execution is served entirely from cache.
            if not analysis_state.literature_actual_model:
                label = self._literature_model_label(cached.get("search_modes", []))
                if label:
                    analysis_state.literature_actual_model = label
            await self._send_message("result", cached, AgentType.LITERATURE)
            return {
                **state,
                "analysis_state": analysis_state,
                "literature_result": cached,
                "failed_node": None,
            }

        instructions = state["pi_instructions"]

        topics = _normalize_tasks(
            instructions.get("literature_topics")
            or instructions.get("deferred_literature_topics")
            or []
        )
        logger.info(f"[{self.session_id}] Starting literature research with topics: {topics}")

        # Dynamic PI planning: refine literature topics based on all completed results
        overview = (
            instructions.get("overview")
            or instructions.get("deferred_literature_overview")
            or ""
        )
        completed = self._collect_current_round_results(state)
        if overview and topics:
            if completed:
                await self._send_message("status", "PI is refining literature topics...", AgentType.PI)
                topics = await self.pi_agent.plan_agent_tasks(
                    agent_name="literature",
                    original_tasks=topics,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        if not topics and completed:
            await self._send_message(
                "status", "PI is defining focused literature topics...", AgentType.PI
            )
            topics = await self.pi_agent.plan_agent_tasks(
                agent_name="literature",
                original_tasks=[],
                completed_results=completed,
                overall_strategy=overview,
                research_objective=analysis_state.initial_instruction,
                state=analysis_state,
            )

        no_topic_reason = ""
        if not topics:
            no_topic_reason = (
                "No focused literature topic was planned, so no literature "
                "search query was executed."
            )
            logger.warning(
                f"[{self.session_id}] No focused literature topics were planned; "
                "skipping literature search execution"
            )

        if topics:
            await self._send_message(
                "status", "Literature research in progress...", AgentType.LITERATURE
            )

        # Collect current round findings from other agents for literature context
        current_round_findings: List[str] = []
        for result_key in ("biochemistry_result", "pathology_result", "bioinformatics_result"):
            result = state.get(result_key)
            if result and result.get("findings"):
                for f in result["findings"][:3]:
                    short = f[:100] if isinstance(f, str) else str(f)[:100]
                    current_round_findings.append(short)

        # Enrich topics with previous findings for round 2+
        previous_lit_context = self._build_previous_findings_context(analysis_state, "literature")
        search_context = (
            f"Project question: {analysis_state.initial_instruction}\n"
            f"Round overview: {overview}\n"
            f"Current findings: {'; '.join(current_round_findings)}"
        )[:12000]
        pi_search_kwargs = {
            "pi_plan_fn": self.pi_agent.plan_literature_search,
            "pi_review_fn": self.pi_agent.review_literature_search,
            "research_context": search_context,
        }

        # Check if Deep Research mode is available (merges topics into one call)
        is_deep_research = self.literature_agent.is_deep_research_available

        limited_topics = topics[:5]

        if is_deep_research and limited_topics:
            lit_mode = getattr(self.literature_agent, "literature_mode", "")
            if lit_mode == "gemini_deep_research":
                # Gemini Deep Research self-decomposes internally; skip the extra
                # PI planning call and always merge into one comprehensive query.
                logger.info(
                    f"[{self.session_id}] Gemini Deep Research: skipping "
                    f"decomposition, merging {len(limited_topics)} topics"
                )
                dr_plan = {"decompose": False, "tasks": limited_topics}
            else:
                # Ask PI whether to split into multiple independent DR sub-tasks.
                dr_plan = await self.pi_agent.plan_deep_research_tasks(
                    topics=limited_topics,
                    completed_results=self._collect_current_round_results(state),
                    overall_strategy=instructions.get("overview", ""),
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

            if dr_plan.get("decompose") and len(dr_plan.get("tasks", [])) > 1:
                # === Multi-task DR: one serial DR call per sub-task ===
                # plan_deep_research_tasks already caps the list at MAX_DR_TASKS.
                dr_tasks = dr_plan["tasks"]
                logger.info(
                    f"[{self.session_id}] Deep Research decomposed into "
                    f"{len(dr_tasks)} sub-tasks"
                )
                results = []
                for i, dr_task in enumerate(dr_tasks, 1):
                    await self._send_message(
                        "status",
                        f"Deep Research sub-task {i}/{len(dr_tasks)} in progress...",
                        AgentType.LITERATURE,
                    )
                    fallback_topic = self._safe_literature_fallback_topic(
                        dr_task, analysis_state.species
                    )
                    brief_topic = self._round_brief_topic_context(
                        dr_task, analysis_state.species, previous_lit_context
                    )
                    sub_brief = await self._generate_round_brief(
                        brief_topic, current_round_findings, analysis_state
                    )
                    r = await self.literature_agent.process(
                        analysis_state, fallback_topic, use_deep_research=True,
                        current_round_findings=current_round_findings,
                        research_brief=sub_brief,
                        **pi_search_kwargs,
                    )
                    results.append(r)
            else:
                # === Legacy merged DR: one call + PubMed gap fill ===
                merged_base_topic = "Please conduct a comprehensive literature review covering the following aspects:\n"
                for i, t in enumerate(limited_topics, 1):
                    merged_base_topic += f"{i}. {t}\n"

                merged_fallback_topic = self._safe_literature_fallback_topic(
                    merged_base_topic, analysis_state.species
                )
                merged_brief_topic = self._round_brief_topic_context(
                    merged_base_topic, analysis_state.species, previous_lit_context
                )

                logger.info(f"[{self.session_id}] Deep Research mode: merging {len(limited_topics)} topics into one query")
                merged_brief = await self._generate_round_brief(
                    merged_brief_topic, current_round_findings, analysis_state
                )
                dr_result = await self.literature_agent.process(
                    analysis_state, merged_fallback_topic, use_deep_research=True,
                    current_round_findings=current_round_findings,
                    research_brief=merged_brief,
                    **pi_search_kwargs,
                )
                results = [dr_result]

                # Evaluate coverage — fallback to PubMed for gaps
                # (also runs for a single topic so under-covered DR output can be supplemented)
                if limited_topics:
                    gaps = await self._evaluate_dr_coverage(
                        limited_topics, dr_result, analysis_state
                    )
                    if gaps:
                        logger.info(f"[{self.session_id}] Deep Research gaps detected, PubMed fallback for: {gaps}")
                        await self._send_message("status", f"Supplementing {len(gaps)} topic(s) via PubMed...", AgentType.LITERATURE)
                        for gap_topic in gaps:
                            gap_result = await self.literature_agent.process(
                                analysis_state, gap_topic, use_deep_research=True,
                                current_round_findings=current_round_findings,
                                mode_override="pubmed_llm",
                                **pi_search_kwargs,
                            )
                            results.append(gap_result)
        else:
            # === PubMed mode: per-topic search ===
            results = []
            for topic in limited_topics:
                result = await self.literature_agent.process(
                    analysis_state,
                    topic,
                    # True routes to _do_research() for full PubMed pipeline,
                    # not the external Deep Research API (guarded by is_deep_research_available).
                    use_deep_research=True,
                    current_round_findings=current_round_findings,
                    **pi_search_kwargs,
                )
                results.append(result)

        # Combine results
        combined_result = {
            "topics": topics,
            "findings": [],
            "reports": [],
            "search_modes": [],
            "source_provenance": [],
            "reference_count": 0,
            "retained_source_report_ids": [],
            "technical_failures": [],
        }

        for r in results:
            combined_result["findings"].extend(r.get("findings", []))
            if r.get("report"):
                combined_result["reports"].append(r["report"])
            if r.get("search_mode"):
                combined_result["search_modes"].append(r["search_mode"])
            provenance = r.get("source_provenance") or r.get("note")
            if provenance:
                combined_result["source_provenance"].append(str(provenance))
            combined_result["reference_count"] += len(r.get("references") or [])
            combined_result["retained_source_report_ids"].extend(
                str(value) for value in (r.get("retained_source_report_ids") or [])
            )
            technical_failure = literature_technical_failure(r)
            if technical_failure:
                combined_result["technical_failures"].append(technical_failure)

        if no_topic_reason:
            combined_result.update({
                "status": "no_focused_topic",
                "reason": no_topic_reason,
            })
        elif combined_result["technical_failures"]:
            combined_result["status"] = "partial"
        elif any(
            "pubmed" in str(mode).lower()
            for mode in combined_result["search_modes"]
        ):
            combined_result["query_policy_version"] = PUBMED_QUERY_POLICY_VERSION

        label = self._literature_model_label(combined_result["search_modes"])
        if label:
            analysis_state.literature_actual_model = label

        logger.info(f"[{self.session_id}] Literature research completed: {len(combined_result['findings'])} findings from {len(topics)} topics")

        # Non-critical: extract generic mechanism signals co-occurring with
        # observed data, persisted onto LiteratureReport for the cross-modal audit.
        # Gate on whether this round actually produced reports. Disabled/skipped is
        # handled by the _require_core_agent early return above; no-topic rounds have
        # an explicit status but no report IDs and therefore short-circuit here.
        report_ids = [r.get("report_id") for r in results if r.get("report_id")]
        recovery = {
            "identity": extraction_identity,
            "phase": "pending",
            "report_ids": report_ids,
            "observed_context": None,
            "result": deepcopy(combined_result),
        }
        if report_ids:
            analysis_state.sandbox_state[_EXTRACTION_RECOVERY_KEY] = recovery
            await self._save_literature_extraction_state(analysis_state)
        return await self._finish_literature_extraction(state, recovery)

    async def _finish_literature_extraction(self, state, recovery):
        """Resume only extraction of the frozen, already-paid search results."""
        analysis_state = state["analysis_state"]
        combined_result = recovery["result"]
        report_ids = recovery["report_ids"]
        completed = recovery.get("phase") == "completed"
        if report_ids and not completed:
            while True:
                try:
                    if recovery.get("observed_context") is None:
                        recovery["observed_context"] = self._build_observed_context_for_literature(state)
                        await self._save_literature_extraction_state(analysis_state)
                    await self._persist_literature_mechanism_signals(
                        analysis_state,
                        self.literature_agent,
                        report_ids,
                        recovery["observed_context"],
                    )
                    break
                except UpstreamExhaustedError as e:
                    action = await self._wait_for_llm_intervention(
                        e, "literature_research", analysis_state,
                    )
                    if action == LlmFailureAction.RETRY:
                        continue
                    if action == LlmFailureAction.TERMINATE:
                        raise WorkflowTerminatedError(
                            "User terminated literature mechanism extraction"
                        ) from e
                    combined_result["source_provenance"].append(
                        "Mechanism extraction was skipped after an upstream failure; "
                        "completed literature searches and findings were retained."
                    )
                    break
                except (WorkflowTerminatedError, WorkflowSupersededError):
                    raise
                except Exception as e:
                    logger.warning(f"[{self.session_id}] mechanism signal persistence skipped: {e}")
                    combined_result["source_provenance"].append(
                        "Mechanism extraction was unavailable after a technical failure; "
                        f"completed literature searches were retained: {str(e)[:500]}"
                    )
                    break

        if not combined_result["technical_failures"]:
            cache_store(analysis_state, AgentType.LITERATURE, combined_result)
        if not completed:
            analysis_state.add_message(
                AgentType.LITERATURE,
                MessageRole.ASSISTANT,
                json.dumps({"findings": combined_result["findings"]}, ensure_ascii=False),
            )
            recovery["phase"] = "completed"
            if report_ids:
                analysis_state.sandbox_state[_EXTRACTION_RECOVERY_KEY] = recovery
                await self._save_literature_extraction_state(analysis_state)
        await self._send_message("result", combined_result, AgentType.LITERATURE)
        return {
            **state,
            "analysis_state": analysis_state,
            "literature_result": combined_result
        }

    async def _persist_literature_mechanism_signals(
        self, analysis_state, literature_agent, report_ids, observed_context: str,
    ):
        """Extract generic mechanism signals from the current round's literature
        reports and persist them onto the LiteratureReport models. Upstream
        failures reach the extraction-only intervention loop."""
        reports = [
            r for r in analysis_state.literature_reports if r.id in set(report_ids)
        ] if report_ids else []
        if not reports:
            return
        # Combine current-round report bodies + references for one extraction pass.
        # Skip reports whose body is an unresolved blob token (offloaded to blob store).
        usable_reports = []
        for r in reports:
            body = r.report or ""
            if body.startswith(BLOB_REF_PREFIX):
                logger.warning(
                    f"[{self.session_id}] literature report {r.id} body is an unresolved "
                    f"blob token; skipping mechanism extraction for it"
                )
                continue
            usable_reports.append(r)
        if not usable_reports:
            return
        combined_text = "\n\n".join((r.report or "") for r in usable_reports)
        combined_refs = []
        for r in usable_reports:
            combined_refs.extend(r.references or [])
        try:
            signals = await literature_agent.extract_mechanism_signals(
                report_text=combined_text,
                references=combined_refs,
                observed_context=observed_context,
            )
        except UpstreamExhaustedError:
            raise  # let the node gate decide; see the DR coverage path below
        except Exception as e:
            logger.warning(f"[{self.session_id}] literature mechanism extraction failed: {e}")
            return
        if not signals:
            return
        # E1: attach the round-level aggregate signals to EVERY current-round report.
        # A round may produce multiple LiteratureReports (multi-topic / PubMed gap-fill);
        # _build_literature_audit_context (Task 7) iterates ALL reports for signals, so
        # attaching to only the last one would silently drop them on reload. This is a
        # deliberate round-level aggregate (one extraction over combined text → shared
        # signal list on each report of the round), not per-topic attribution.
        for rep in reports:
            rep.mechanism_signals = signals
        logger.info(
            f"[{self.session_id}] persisted {len(signals)} literature mechanism signals "
            f"to {len(reports)} report(s)"
        )

    def _build_observed_context_for_literature(self, workflow_state) -> str:
        analysis_state = workflow_state["analysis_state"]
        parts = []
        # (1) Prefer the CURRENT round's freshly-computed agent results from WorkflowState
        for label, key in [
            ("Biochemistry", "biochemistry_result"),
            ("Bioinformatics", "bioinformatics_result"),
            ("Pathology", "pathology_result"),
        ]:
            result = workflow_state.get(key) or {}
            vals = result.get("findings") or []
            if vals:
                parts.append(f"{label}: " + "; ".join(str(v) for v in vals[:8]))
        # (2) Fallback to the prior round's findings if no current-round result present
        if not parts and analysis_state.round_results:
            last = analysis_state.round_results[-1]
            for label, vals in [
                ("Biochemistry", last.biochemistry_findings),
                ("Bioinformatics", last.bioinformatics_findings),
                ("Pathology", last.pathology_findings),
            ]:
                if vals:
                    parts.append(f"{label}: " + "; ".join(str(v) for v in vals[:8]))
        # (3) Manual biochemistry indicators are always current observed data
        if analysis_state.biochemistry_indicators:
            inds = "; ".join(
                f"{i.indicator}={i.value}{i.unit or ''}" for i in analysis_state.biochemistry_indicators[:20]
            )
            parts.append(f"Biochemistry indicators: {inds}")
        # (4) Data-driven mechanism programs from bioinformatics analysis.
        # Computed here (after the (2) fallback gate has already evaluated `not parts`)
        # but inserted at the FRONT of the output so this high-value mechanism signal
        # survives the trailing [:4000] truncation in data-heavy runs — it is exactly
        # what extract_mechanism_signals() maps to literature.
        programs = getattr(analysis_state, "data_driven_mechanism_programs", []) or []
        if programs:
            recent = programs[-8:]
            prog_lines = ["Bioinformatics data-driven mechanism programs:"]
            for p in recent:
                prog_lines.append(p.to_prompt_text(style="detailed", max_observations=3, max_genes=15, max_pathways=8, max_alternatives=3, max_missing=3))
            parts.insert(0, "\n".join(prog_lines))
        return ("\n".join(parts))[:4000]

    async def _evaluate_dr_coverage(
        self, topics: List[str], dr_result: Dict, analysis_state=None
    ) -> List[str]:
        """Evaluate which topics were NOT sufficiently covered by Deep Research.

        Returns a list of gap topic strings for PubMed+LLM fallback.
        """
        report = dr_result.get("report", "")
        if not report:
            return topics  # No report at all — fall back on everything

        try:
            return await self.pi_agent.review_deep_research_coverage(
                topics=topics,
                report=report,
                state=analysis_state,
            )
        except UpstreamExhaustedError:
            raise
        except Exception as e:
            logger.warning(f"[{self.session_id}] DR coverage evaluation failed: {e}")
            return topics
