"""Guardrails, dependency validation, and final selection of research directions.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List
import json
import logging

from ...utils.direction_payload import (
    CANDIDATE_PROMPT_FIELDS,
    budgeted_prompt_block,
    investigated_direction_texts,
)
from ...utils.formatting import as_list

# The selection step is the only place a direction can now be turned down, so
# its candidate list is never shortened by dropping entries.
SELECTION_CANDIDATE_BUDGET = 20000

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState

from .shared import (  # noqa: F401
    _CONTROL_SIGNAL_RESULTS,
    _BRIEF_PREFERRED_TYPES,
    _BRIEF_QC_MARKERS,
    _DIRECTION_CANDIDATE_LIST_FIELDS,
    _normalize_supporting_data,
    _normalize_direction_candidate,
    _normalize_integration_scored_candidates,
    _bounded_brief_fallback,
    _format_database_priors_block,
    _build_default_bio_tasks,
)


def _dict_items(value):
    """Keep only the object-shaped entries of a selection list.

    ``selected_directions`` being a list was checked; its ELEMENTS being objects
    was not, so a reply like ``["something"]`` passed validation and then raised
    on the first ``.get()`` several call frames away.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _selection_is_readable(data) -> bool:
    """Whether a selection response can be read.

    ``["bad"]`` is NOT a readable empty selection. Filtering it down to [] turned
    a malformed reply into the scientific statement "no direction is worth a
    round", which is exactly the conflation this whole path exists to prevent.
    An actually empty list is readable, and means none was selected.
    """
    if not isinstance(data, dict):
        return False
    selected = data.get("selected_directions")
    if not isinstance(selected, list):
        return False
    return not selected or any(isinstance(item, dict) for item in selected)


class DirectionSelectionMixin:
    """Guardrails, dependency validation, and final selection of research directions."""

    def _validate_direction_dependency_graph(self, graph: Dict[str, List[str]]) -> None:
        """Validate direction dependency graph for cycles using DFS."""
        visiting = set()
        visited = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"Cycle detected in direction dependency graph at {node}")
            if node in visited:
                return
            visiting.add(node)
            for dep in graph.get(node, []) or []:
                visit(dep)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)

    @staticmethod
    def _acyclic_direction_dependency_graph(
        graph: Dict[str, List[str]],
    ) -> tuple[Dict[str, List[str]], List[str]]:
        """Drop only cycle-closing edges while preserving all other edges."""
        repaired: Dict[str, List[str]] = {}
        dropped: List[str] = []
        on_stack: set[str] = set()
        finished: set[str] = set()

        def visit(node: str) -> None:
            if node in finished:
                return
            on_stack.add(node)
            kept: List[str] = []
            for dependency in as_list(graph.get(node)):
                if dependency in on_stack:
                    dropped.append(f"{node} -> {dependency}")
                    continue
                visit(dependency)
                kept.append(dependency)
            if node in graph:
                repaired[node] = kept
            on_stack.discard(node)
            finished.add(node)

        for node in graph:
            visit(node)
        return repaired, dropped

    def _sanitize_direction_dependency_graph(self, report: Any) -> List[str]:
        graph = getattr(report, "direction_dependency_graph", None) or {}
        if not graph:
            return []
        repaired, dropped = self._acyclic_direction_dependency_graph(graph)
        if not dropped:
            return []
        report.direction_dependency_graph = repaired
        notes = list(getattr(report, "direction_dependency_repairs", None) or [])
        for edge in dropped:
            note = (
                f"Prerequisite dependency {edge} was dropped because it closed a cycle; "
                "the directions themselves are unchanged."
            )
            if note not in notes:
                notes.append(note)
        report.direction_dependency_repairs = notes
        logger.warning("[PI] Repaired cyclic direction dependency graph: %s", dropped)
        return dropped

    @staticmethod
    def _candidate_has_specific_core_observation(candidate: Dict[str, Any]) -> bool:
        text = str(candidate.get("core_observation") or "").strip()
        if len(text) < 25:
            return False
        generic = (
            "interesting pattern", "interesting patterns", "data suggests",
            "data suggest", "further analysis", "needs investigation"
        )
        lowered = text.lower()
        if any(g in lowered for g in generic) and not any(ch.isdigit() for ch in text):
            return False
        markers = (
            "cluster", "log2", "fold", "p=", "padj", "rho", "correlation",
            "score", "cell", "gene", "pathway", "module", "pmid"
        )
        return any(m in lowered for m in markers) or any(ch.isdigit() for ch in text)

    @staticmethod
    def _append_unique_flag(candidate: Dict[str, Any], flag: str) -> None:
        flags = candidate.get("risk_flags")
        if not isinstance(flags, list):
            flags = []
        if flag not in flags:
            flags.append(flag)
        candidate["risk_flags"] = flags

    @staticmethod
    def _select_directions(candidates: list) -> list:
        return [d.get("direction", "") for d in candidates if isinstance(d, dict) and d.get("direction")]

    @classmethod
    def _apply_direction_guardrails(
        cls,
        scored: List[Dict[str, Any]],
        raw_candidates: List[Dict[str, Any]],
        literature_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence_by_id = {
            str(e.get("direction_id", "")): e
            for e in literature_evidence or []
            if isinstance(e, dict)
        }
        for candidate in scored:
            if not isinstance(candidate, dict):
                continue
            if not cls._candidate_has_specific_core_observation(candidate):
                candidate["recommendation"] = "needs_more_data"
                cls._append_unique_flag(candidate, "generic_core_observation")
            # A "prioritize" recommendation is downgraded if any quality
            # dimension is disqualifying. Add a (field, bad_value) tuple to extend.
            downgrade_conditions = (
                ("data_reliability", "low"),
                ("artifact_risk", "high"),
                ("data_sufficiency", "insufficient"),
            )
            if candidate.get("recommendation") == "prioritize" and any(
                str(candidate.get(field, "")).lower() == bad_value
                for field, bad_value in downgrade_conditions
            ):
                candidate["recommendation"] = "needs_more_data"
            if candidate.get("contradicts_stated_premise") is True:
                cls._append_unique_flag(candidate, "contradicts_premise")
            ev = evidence_by_id.get(str(candidate.get("evidence_ref") or candidate.get("direction_id") or ""), {})
            novelty_status = str(ev.get("novelty_status", "")).lower()
            if novelty_status in {"unknown_due_to_timeout", "unknown_due_to_failure"}:
                candidate["literature_novelty"] = "unknown"
                if candidate.get("recommendation") == "prioritize":
                    candidate["recommendation"] = "needs_more_data"
                cls._append_unique_flag(
                    candidate,
                    "literature_evidence_timeout" if novelty_status == "unknown_due_to_timeout" else "literature_evidence_failed",
                )
        return scored

    @staticmethod
    def _critical_direction_fields_missing(candidate: Dict[str, Any]) -> List[str]:
        placeholders = {"...", "tbd", "todo", "n/a", "na", "none", "not specified"}

        def meaningful(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip()) and value.strip().lower() not in placeholders
            if isinstance(value, dict):
                return any(meaningful(item) for item in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(meaningful(item) for item in value)
            return True

        missing = []
        for key in (
            "core_observation", "alternative_hypotheses", "artifact_risks",
            "contradictory_or_missing_evidence", "validation_plan", "recommendation",
        ):
            value = candidate.get(key)
            if not meaningful(value):
                missing.append(key)
        alternatives = as_list(candidate.get("alternative_hypotheses"))
        if "alternative_hypotheses" not in missing and sum(
            meaningful(item) for item in alternatives
        ) < 2:
            missing.append("alternative_hypotheses")
        validation_plan = candidate.get("validation_plan")
        if isinstance(validation_plan, dict) and not all(
            meaningful(validation_plan.get(key))
            for key in ("next_analysis", "wet_lab_validation", "decision_criteria")
        ) and "validation_plan" not in missing:
            missing.append("validation_plan")
        return missing

    async def select_best_directions(
        self,
        state: AnalysisState
    ) -> Dict[str, Any]:
        """
        PI autonomously selects the best research directions from exploration report candidates.
        Used when user clicks 'Let PI Decide' or countdown expires.

        Returns:
            DirectionSelectionDecision-shaped dict with selected_directions, rejected_directions, selection_rationale.
        """
        logger.info(f"[PI] Selecting best research directions")
        self.reset_history()

        candidates = []
        if state.exploration_report and state.exploration_report.research_direction_candidates:
            candidates = state.exploration_report.research_direction_candidates

        if not candidates:
            # No candidate to choose from is not a reason to invent one. An
            # innovation round needs a direction that came out of the data;
            # manufacturing "Continue analysis based on Round 1 findings" sends the
            # whole round chasing a placeholder.
            return {
                "selected_directions": [],
                "rejected_directions": [],
                "decision_outcome": "none_selected",
                "selection_rationale": (
                    "Exploration produced no research direction candidates, so there is "
                    "nothing to investigate in an innovation round; proceeding to the "
                    "final report on the Round 1 results."
                ),
            }

        exploration_context = ""
        dependency_graph_section = ""
        if state.exploration_report:
            exploration_context = f"""
Cluster Overview: {state.exploration_report.cluster_overview[:500]}
Key Pathway Changes: {json.dumps(state.exploration_report.key_pathway_changes[:5], ensure_ascii=False)}
Strongest Heterogeneity: {json.dumps(state.exploration_report.strongest_heterogeneity[:5], ensure_ascii=False)}
Data Quality Warnings: {json.dumps(state.exploration_report.data_quality_warnings[:3], ensure_ascii=False)}
PI Direction Assessment: {state.exploration_report.pi_direction_assessment[:500]}
"""
            if state.exploration_report.direction_dependency_graph:
                dependency_graph_section = f"""
## Direction Dependency Graph
{json.dumps(state.exploration_report.direction_dependency_graph, ensure_ascii=False, indent=2)}

Select directions that respect dependency ordering: dependent directions should only be selected if their prerequisites are also selected.
"""

        prompt = f"""You are the PI selecting the best research direction(s) for the innovation phase.

## Exploration Report
{exploration_context}
{dependency_graph_section}
## Candidates
{budgeted_prompt_block(candidates, CANDIDATE_PROMPT_FIELDS, SELECTION_CANDIDATE_BUDGET)}

Select the direction(s) — usually 1 to 2 — that together provide the most informative outcomes while respecting prerequisite dependencies. If no candidate clears the reliability bar, it is a valid and useful answer to select none and say what would have to be measured or resolved first; do not promote a weak candidate merely to fill the slot.

SELECTION CRITERIA:
1. **Reliability** is not merely a gate, it is a weight. A more surprising finding needs MORE
   supporting evidence, not less. In single-cell data the most surprising signals are
   disproportionately artifacts — ambient RNA, doublets, batch, a cluster driven by a handful of
   cells — so ranking on surprise alone systematically selects for artifacts.
2. **Innovation + Novelty** raise priority only after reliability and artifact risk have been
   accounted for. Discount any novelty claim by the candidate's artifact_risk and
   data_sufficiency; a high-novelty, high-artifact-risk candidate ranks below a solid one.
3. Selected directions should ideally be complementary (not redundant), and selected_directions must be listed in priority order, most important first
4. Consider the dependency graph: if a direction has prerequisites, ensure its prerequisites are also selected or the direction is independently viable
5. For each rejected direction, provide an explanation

Return JSON:
{{
    "selected_directions": [
        {{
            "direction": "The exact direction text from candidates",
            "notes": "Brief PI notes on why this was selected and what to focus on"
        }},
        {{
            "direction": "The exact direction text from candidates",
            "notes": "Brief PI notes on why this was selected and what to focus on"
        }}
    ],
    "rejected_directions": [
        {{
            "direction": "The exact direction text from rejected candidates",
            "reason_not_selected": "Why this direction was not selected",
            "follow_up_value": "high|medium|low",
            "dependency_note": "Whether it has unmet prerequisites or is suitable as follow-up"
        }}
    ],
    "selection_rationale": "Why these directions were chosen over others"
}}
"""

        # An empty selection is a SCIENTIFIC answer ("nothing here clears the
        # reliability bar") and must be told apart from a response we failed to
        # read. Only the latter is a malfunction. Conflating them made the prompt's
        # "it is valid to select none" unreachable: the PI returned [], recovery
        # overwrote it, and a low-quality direction entered an innovation round.
        data = {}
        response_readable = False
        for attempt in range(2):
            response = await self.chat(prompt, temperature=0.3)
            data = self._safe_parse_json(response, default={})
            response_readable = _selection_is_readable(data)
            if response_readable:
                break
            logger.warning(
                "[PI] Direction-selection response unreadable (attempt %s/2)", attempt + 1
            )

        selected = _dict_items(data.get("selected_directions", [])) if response_readable else []
        rejected = data.get("rejected_directions", []) if isinstance(data, dict) else []
        rationale = data.get("selection_rationale", "") if isinstance(data, dict) else ""

        if not response_readable:
            # Choosing a research direction is the PI's call. When the reply cannot
            # be read even on retry, the honest outcome is to select none and say
            # so -- the run finalizes on Round 1. Having the host pick "the two
            # highest-priority candidates" would silently substitute a mechanical
            # ranking for a scientific judgement, and the report could not tell the
            # difference afterwards.
            logger.error(
                "[PI] Direction selection unreadable after retry; selecting no direction"
            )
            selected = []
            rejected = []
            rationale = (
                "DIRECTION SELECTION UNAVAILABLE: the PI's selection response could not be "
                "read after a retry, so no innovation direction was selected and the run "
                "finalizes on the Round 1 results. This is a tooling failure, not a "
                "scientific conclusion about the candidates; the scored candidates are "
                "retained in the exploration report for review."
            )
        elif not selected and not rationale:
            rationale = (
                "No candidate cleared the reliability bar; no innovation direction was "
                "selected."
            )

        # Validate dependency graph if present
        dep_graph = state.exploration_report.direction_dependency_graph if state.exploration_report else {}
        if dep_graph and state.exploration_report:
            self._sanitize_direction_dependency_graph(state.exploration_report)

        # Prerequisite closure (multi-hop): keep adding prerequisites until none are missing.
        text_by_id = {c.get("direction_id"): c.get("direction", "") for c in candidates if isinstance(c, dict)}
        id_by_text = {c.get("direction", ""): c.get("direction_id") for c in candidates if isinstance(c, dict)}
        cand_by_id = {c.get("direction_id"): c for c in candidates if isinstance(c, dict)}
        selected_texts = {d.get("direction", "") for d in selected}
        changed = True
        while changed:
            changed = False
            for d in list(selected):
                dir_id = id_by_text.get(d.get("direction", ""))
                cand = cand_by_id.get(dir_id)
                if not cand:
                    continue
                for prereq_id in as_list(cand.get("prerequisite_dependencies")):
                    prereq_text = text_by_id.get(prereq_id)
                    if prereq_text and prereq_text not in selected_texts:
                        # Tagged, not just described: a per-gate direction cap
                        # slices the tail of this list, which is exactly where
                        # promoted prerequisites land. Without a field to read,
                        # the cap commits the dependent and defers the thing it
                        # depends on.
                        selected.append({"direction": prereq_text,
                                         "notes": f"Added to satisfy prerequisite of '{d.get('direction', '')}'.",
                                         "prerequisite_for": d.get("direction", "")})
                        selected_texts.add(prereq_text)
                        changed = True

        # A direction promoted into selected (e.g. as a prerequisite) must not also appear as rejected.
        rejected = [r for r in rejected
                    if isinstance(r, dict) and r.get("direction", "") not in selected_texts]

        logger.info(f"[PI] Selected {len(selected)} directions: {[d.get('direction', '')[:50] for d in selected]}")
        return {
            "selected_directions": selected,
            "rejected_directions": rejected,
            "selection_rationale": rationale,
            "decision_outcome": (
                "selected" if selected
                else ("none_selected" if response_readable else "tooling_failure")
            ),
        }

    async def select_integration_directions(
        self,
        state: "AnalysisState",
        scored_candidates: List[Dict[str, Any]],
        integration_report: str,
    ) -> Dict[str, Any]:
        """PI selects directions from integration candidates after cross-direction synthesis.

        Unlike select_best_directions (which forces at least 1 direction), this method
        allows returning zero selected_directions when none are sufficiently novel.

        Returns:
            DirectionSelectionDecision-shaped dict with selected_directions,
            rejected_directions, selection_rationale. selected_directions may be empty.
        """
        logger.info("[PI] Selecting integration directions")
        self.reset_history()

        if not scored_candidates:
            return {
                "selected_directions": [],
                "rejected_directions": [],
                "selection_rationale": "No integration candidates available; proceeding to final report.",
            }

        # Not user_chosen_directions: that field holds only the most recent
        # batch, so every earlier cycle's choices would be invisible here and an
        # already-investigated direction could come back as "new".
        already_chosen = investigated_direction_texts(state)
        rounds_block = []
        for rr in (state.round_results or []):
            if getattr(rr, "phase", None) != "innovation":
                continue
            rounds_block.append(
                f"### Round {rr.round_number} — direction: {getattr(rr, 'direction', 'N/A')}\n"
                f"- Bioinformatics: {json.dumps(getattr(rr, 'bioinformatics_findings', [])[:8], ensure_ascii=False)}\n"
                f"- Pathology: {json.dumps(getattr(rr, 'pathology_findings', [])[:3], ensure_ascii=False)}\n"
                f"- Biochemistry: {json.dumps(getattr(rr, 'biochemistry_findings', [])[:3], ensure_ascii=False)}\n"
                f"- Literature: {json.dumps(getattr(rr, 'literature_findings', [])[:3], ensure_ascii=False)}\n"
                f"- Summary: {(getattr(rr, 'pi_summary', '') or '')[:600]}"
            )

        prompt = f"""You are the PI deciding whether to pursue new research directions after cross-direction integration analysis.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}
- Species: {state.species or "Not specified"}
{state.build_experimental_design_summary()}

## Cross-Direction Integration Report
{integration_report[:4000]}

## Innovation Rounds (cumulative)
{chr(10).join(rounds_block) if rounds_block else "No innovation rounds yet."}

## Integration Direction Candidates (ranked by the scoring step, not filtered by it)
{budgeted_prompt_block(scored_candidates, CANDIDATE_PROMPT_FIELDS, SELECTION_CANDIDATE_BUDGET)}

A candidate carrying "carried_from_gate" was proposed at an earlier gate and never run; one carrying "previously_declined" was turned down there, with the reason given. Neither is a verdict you have to keep: the results that have arrived since are exactly what could change the answer. A "risk_flags" entry of "unscored_by_gate" means the scoring step did not return that candidate and it reached you unranked, which says nothing about its merit.

## Already-Investigated Directions (DO NOT repeat)
{json.dumps(already_chosen, ensure_ascii=False)}

Evaluate each candidate direction. Select those that are:
1. Genuinely NEW relative to the already-investigated directions listed above
2. Answerable by analyses a further round could actually run on the data this project has
3. Likely to yield meaningful new biological insight

On criterion 2, separate two kinds of thin evidence, because only one of them is a reason to stop. A direction this project's own results already argue against is not worth a round. A direction that is merely unmeasured is the opposite case: no round has tested it, and running it is what would produce the evidence. Never reject a direction for lacking evidence that only running it could produce. Prior-art volume is not a verdict either; a well-studied mechanism can still be the unanswered question in this project. Judge a direction on whether this project's data can speak to it, and rank one that addresses the stated research objective head-on above one that does not.

IMPORTANT: An EMPTY selected_directions list is the right answer when every candidate repeats work already done, is contradicted by this project's own results, or cannot be advanced with the data available. It is not the right answer merely because the candidates are unproven -- being unproven is the reason to run a round, not to skip one.

List selected_directions in priority order, most important first: when a run has a per-gate direction cap, that order is what decides which ones run now.

Return JSON:
{{
    "selected_directions": [
        {{
            "direction": "The exact direction text from candidates",
            "notes": "Brief PI notes on why this was selected and what to focus on"
        }}
    ],
    "rejected_directions": [
        {{
            "direction": "The direction text from rejected candidates",
            "reason_not_selected": "Why this direction was not selected",
            "follow_up_value": "high|medium|low",
            "dependency_note": ""
        }}
    ],
    "selection_rationale": "Why these directions were chosen, or why none were selected"
}}"""

        data = {}
        response_readable = False
        for attempt in range(2):
            response = await self.chat(prompt, temperature=0.3)
            data = self._safe_parse_json(response, default={})
            response_readable = _selection_is_readable(data)
            if response_readable:
                break
            logger.warning(
                "[PI] Integration-selection response unreadable (attempt %s/2)", attempt + 1
            )

        if not response_readable:
            # Both outcomes end the study, but they are not the same statement and
            # the report must not conflate them: "nothing novel enough remains" is a
            # scientific conclusion, "we could not read the reply" is a malfunction.
            logger.error(
                "[PI] Integration selection unreadable after retry; selecting no direction"
            )
            return {
                "selected_directions": [],
                "rejected_directions": [],
                "decision_outcome": "tooling_failure",
                "selection_rationale": (
                    "INTEGRATION SELECTION UNAVAILABLE: the PI's selection response could not "
                    "be read after a retry, so no further direction was started and the run "
                    "proceeds to the final report. This is a tooling failure, not a finding "
                    "that no novel direction remains; the scored candidates are retained for "
                    "review."
                ),
            }

        selected = _dict_items(data.get("selected_directions", []))
        rejected = data.get("rejected_directions", [])
        rationale = data.get("selection_rationale", "")

        # Unlike select_best_directions, we do NOT force at least 1 direction.
        # Empty selected_directions is a valid outcome meaning "proceed to report".
        if not rationale:
            if selected:
                rationale = "Selected based on novelty and evidence strength."
            else:
                rationale = "No sufficiently novel and evidence-supported direction found; proceeding to final report."

        return {
            "selected_directions": selected,
            "rejected_directions": rejected,
            "selection_rationale": rationale,
            "decision_outcome": "selected" if selected else "none_selected",
        }
