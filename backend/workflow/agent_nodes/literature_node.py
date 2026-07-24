# Literature research node, round briefs, and deep-research coverage evaluation.
# Moved verbatim out of backend/workflow/agents.py; composed into AgentNodesMixin.
"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from ...utils.llm_client import UpstreamExhaustedError
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...agents.bioinformatics_agent import count_saved_figures
from ..helpers import _get_task_text, _has_stage1_keyword, _normalize_tasks, cache_hit, cache_invalidate, cache_store, decide_bioinfo_post_execution, prepend_species, safe_pi_brief, STAGE1_KEYWORDS
from ...utils.blob_store import BLOB_REF_PREFIX
from ..retry import with_retry

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


class LiteratureNodeMixin:
    """Literature research node, round briefs, and deep-research coverage evaluation."""

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

        cached = cache_hit(analysis_state, AgentType.LITERATURE)
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

        topics = _normalize_tasks(instructions.get("literature_topics", []))
        logger.info(f"[{self.session_id}] Starting literature research with topics: {topics}")

        # Dynamic PI planning: refine literature topics based on all completed results
        overview = instructions.get("overview", "")
        if overview and topics:
            completed = self._collect_current_round_results(state)
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

        if not topics:
            # Default topics based on other findings
            path_findings = state.get("pathology_result", {}).get("findings", []) if state.get("pathology_result") else []
            bio_findings = state.get("bioinformatics_result", {}).get("findings", []) if state.get("bioinformatics_result") else []
            biochem_findings = state.get("biochemistry_result", {}).get("findings", []) if state.get("biochemistry_result") else []

            all_source_findings = path_findings + bio_findings + biochem_findings
            if all_source_findings:
                # Use truncated findings as search keywords (full findings are too long for search queries)
                short_findings = [
                    (f[:80] if isinstance(f, str) else str(f)[:80])
                    for f in all_source_findings[:3]
                ]
                objective = (analysis_state.initial_instruction or "").strip()
                objective_prefix = f"{objective[:120]} — " if objective else ""
                topics = [f"{objective_prefix}single cell sequencing {' '.join(short_findings)}"]
            else:
                # No findings from other agents — build topic from research objective and treatment groups
                objective = analysis_state.initial_instruction or ""
                group_names = [g.name for g in (analysis_state.treatment_groups or []) if g.name]
                if objective:
                    topics = [f"single cell sequencing {objective[:120]}"]
                elif group_names:
                    topics = [f"single cell sequencing {' '.join(group_names[:3])}"]
                else:
                    logger.warning(f"[{self.session_id}] No findings or objective for literature search, skipping")
                    topics = []

        await self._send_message("status", "Literature research in progress...", AgentType.LITERATURE)

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
                )
                results = [dr_result]

                # Evaluate coverage — fallback to PubMed for gaps
                # (also runs for a single topic so under-covered DR output can be supplemented)
                if limited_topics:
                    gaps = await self._evaluate_dr_coverage(limited_topics, dr_result)
                    if gaps:
                        logger.info(f"[{self.session_id}] Deep Research gaps detected, PubMed fallback for: {gaps}")
                        await self._send_message("status", f"Supplementing {len(gaps)} topic(s) via PubMed...", AgentType.LITERATURE)
                        for gap_topic in gaps:
                            enriched = gap_topic
                            if previous_lit_context:
                                enriched = f"{gap_topic}\n{previous_lit_context}\nFocus on aspects NOT already covered."
                            enriched = prepend_species(enriched, analysis_state.species)
                            gap_result = await self.literature_agent.process(
                                analysis_state, enriched, use_deep_research=True,
                                current_round_findings=current_round_findings,
                                mode_override="pubmed_llm",
                            )
                            results.append(gap_result)
        else:
            # === PubMed mode: per-topic search ===
            results = []
            for topic in limited_topics:
                enriched_topic = topic
                if previous_lit_context:
                    enriched_topic = f"{topic}\n{previous_lit_context}\nFocus on aspects NOT already covered above."
                enriched_topic = prepend_species(enriched_topic, analysis_state.species)
                result = await self.literature_agent.process(
                    analysis_state,
                    enriched_topic,
                    # True routes to _do_research() for full PubMed pipeline,
                    # not the external Deep Research API (guarded by is_deep_research_available).
                    use_deep_research=True,
                    current_round_findings=current_round_findings
                )
                results.append(result)

        # Combine results
        combined_result = {
            "topics": topics,
            "findings": [],
            "reports": [],
            "search_modes": [],
        }

        for r in results:
            combined_result["findings"].extend(r.get("findings", []))
            if r.get("report"):
                combined_result["reports"].append(r["report"])
            if r.get("search_mode"):
                combined_result["search_modes"].append(r["search_mode"])

        label = self._literature_model_label(combined_result["search_modes"])
        if label:
            analysis_state.literature_actual_model = label

        logger.info(f"[{self.session_id}] Literature research completed: {len(combined_result['findings'])} findings from {len(topics)} topics")

        await self._send_message(
            "result",
            combined_result,
            AgentType.LITERATURE
        )

        analysis_state.add_message(
            AgentType.LITERATURE,
            MessageRole.ASSISTANT,
            json.dumps({"findings": combined_result["findings"]}, ensure_ascii=False)
        )

        # Non-critical: extract generic mechanism signals co-occurring with
        # observed data, persisted onto LiteratureReport for the cross-modal audit.
        # GUARD (B3): combined_result has NO "status" key (verified — it only holds
        # topics/findings/reports/search_modes), so gate on whether this round actually
        # produced reports. Disabled/skipped is already handled by the _require_core_agent
        # early-return above; no_data rounds produce no report_ids and short-circuit here.
        report_ids = [r.get("report_id") for r in results if r.get("report_id")]
        if report_ids:
            try:
                observed_context = self._build_observed_context_for_literature(state)
                await self._persist_literature_mechanism_signals(
                    analysis_state,
                    self.literature_agent,
                    report_ids,
                    observed_context,
                )
            except Exception as e:
                logger.warning(f"[{self.session_id}] mechanism signal persistence skipped: {e}")

        cache_store(analysis_state, AgentType.LITERATURE, combined_result)
        return {
            **state,
            "analysis_state": analysis_state,
            "literature_result": combined_result
        }

    async def _persist_literature_mechanism_signals(
        self, analysis_state, literature_agent, report_ids, observed_context: str,
    ):
        """Extract generic mechanism signals from the current round's literature
        reports and persist them onto the LiteratureReport models. Non-critical:
        never raises, never changes literature_result status."""
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

    async def _evaluate_dr_coverage(self, topics: List[str], dr_result: Dict) -> List[str]:
        """Evaluate which topics were NOT sufficiently covered by Deep Research.

        Returns a list of gap topic strings for PubMed+LLM fallback.
        """
        report = dr_result.get("report", "")
        if not report:
            return topics  # No report at all — fall back on everything

        topics_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
        prompt = f"""The following Deep Research report was generated:

{report[:3000]}

The original research topics were:
{topics_list}

Which topics are NOT sufficiently covered in the report? Return a JSON array of topic strings that need additional research. If all topics are well covered, return an empty array [].

Return ONLY the JSON array, nothing else."""

        try:
            response = await self.literature_agent.chat_without_history(prompt, temperature=0.2)
            # Strip markdown code fences if present (e.g. ```json ... ```)
            cleaned = response.strip()
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            gaps = json.loads(cleaned)
            if isinstance(gaps, list):
                return [g for g in gaps if isinstance(g, str)]
        except UpstreamExhaustedError:
            raise
        except Exception as e:
            logger.warning(f"[{self.session_id}] DR coverage evaluation failed: {e}")
            return topics
