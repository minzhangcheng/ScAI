"""Guardrails, dependency validation, and final selection of research directions.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List
import json
import logging

from ...utils.formatting import as_list

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
            return {
                "selected_directions": [{"direction": "Continue analysis based on Round 1 findings", "notes": ""}],
                "rejected_directions": [],
                "selection_rationale": "No candidates available; using default direction.",
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
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Select the best direction(s) — usually 1 to 2 — that together provide the most comprehensive and impactful outcomes while respecting prerequisite dependencies.

SELECTION CRITERIA:
1. **Reliability** is the baseline: low-reliability directions should be avoided
2. **Innovation + Novelty** together determine priority
3. Selected directions should ideally be complementary (not redundant)
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

        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        selected = data.get("selected_directions", [])
        rejected = data.get("rejected_directions", [])
        rationale = data.get("selection_rationale", "")

        if not selected or len(selected) < 1:
            # Fallback: pick first 2 candidates
            selected = [
                {"direction": c.get("direction", ""), "notes": ""}
                for c in candidates[:2]
            ]
            # Build rejected list from remaining candidates
            selected_dirs_set = {d.get("direction", "") for d in selected}
            rejected = [
                {"direction": c.get("direction", ""), "reason_not_selected": "Not selected in fallback",
                 "follow_up_value": "low", "dependency_note": ""}
                for c in candidates
                if c.get("direction", "") not in selected_dirs_set
            ]
            if not rationale:
                rationale = "Fallback: selected first available candidates."

        # Ensure at least 1 direction
        if not selected:
            selected = [{"direction": "Continue analysis based on Round 1 findings", "notes": ""}]

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
                        selected.append({"direction": prereq_text,
                                         "notes": f"Added to satisfy prerequisite of '{d.get('direction', '')}'."})
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

        already_chosen = [d.get("direction", "") for d in (state.user_chosen_directions or [])]
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

## Integration Direction Candidates (scored by novelty and evidence)
{json.dumps(scored_candidates, ensure_ascii=False, indent=2)}

## Already-Investigated Directions (DO NOT repeat)
{json.dumps(already_chosen, ensure_ascii=False)}

Evaluate each candidate direction. Select ONLY those that are:
1. Genuinely NOVEL relative to already-investigated directions
2. Well-supported by concrete evidence from the integration analysis
3. Likely to yield meaningful new biological insights

IMPORTANT: If none of the candidates are sufficiently novel and evidence-supported, return an EMPTY selected_directions list. It is better to proceed to the final report than to pursue a direction that lacks solid evidence.

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

        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        selected = data.get("selected_directions", [])
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
        }
