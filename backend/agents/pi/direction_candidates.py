"""Generation and evidence-scoring of research-direction candidates.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List, Optional
import json
import logging

from ...utils.direction_payload import (
    CANDIDATE_PROMPT_FIELDS,
    EVIDENCE_PROMPT_FIELDS,
    budgeted_prompt_block,
    investigated_direction_texts,
)
from ...utils.formatting import safe_parse_json_array
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...utils.bioinformatics_evidence import render_evidence_briefs

# How much of the scoring prompt the candidate list and the novelty records may
# use. Both blocks give every entry a slot rather than cutting the list short.
SCORER_CANDIDATE_BUDGET = 16000
SCORER_EVIDENCE_BUDGET = 6000

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import (
    AnalysisState,
    RoundResult,
    ExplorationReport,
    DirectionSearchTask,
    DirectionPIReview,
)

from .shared import (  # noqa: F401
    _CONTROL_SIGNAL_RESULTS,
    _BRIEF_PREFERRED_TYPES,
    _BRIEF_QC_MARKERS,
    _DIRECTION_CANDIDATE_LIST_FIELDS,
    _normalize_supporting_data,
    _normalize_direction_candidate,
    _normalize_integration_scored_candidates,
    _restore_unscored_candidates,
    _bounded_brief_fallback,
    _format_database_priors_block,
    _build_default_bio_tasks,
)


def _candidates_are_usable(data) -> bool:
    """Whether a candidate response can be read at all.

    A missing key, a non-list, or a non-empty list with no object in it are all
    the same kind of answer: we could not read it. An EMPTY list is different --
    that is the PI saying there is nothing worth investigating.
    """
    if not isinstance(data, dict):
        return False
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return False
    return not candidates or any(isinstance(c, dict) for c in candidates)


class DirectionCandidatesMixin:
    """Generation and evidence-scoring of research-direction candidates."""

    async def plan_direction_literature_search(
        self,
        state: AnalysisState,
        candidates: List[Dict[str, Any]],
        prior_records: Optional[List[Dict[str, Any]]] = None,
        *,
        phase: str = "exploration",
    ) -> Dict[str, Any]:
        """Ask the PI for literal, candidate-specific PubMed search tasks.

        The host validates the returned task records later.  This method only
        supplies the scientific plan; it does not compile queries or add scope
        terms.  A dictionary result keeps planning diagnostics visible to the
        shared Literature coordinator without changing the ordinary topic path.
        """
        self.refresh_runtime_config_if_needed()
        candidate_ids = [str(c.get("direction_id") or "").strip() for c in candidates]
        candidate_payload = [
            {
                "direction_id": c.get("direction_id", ""),
                "direction": c.get("direction", ""),
                "core_observation": c.get("core_observation", ""),
                "main_hypothesis": c.get("main_hypothesis", ""),
                "alternative_hypotheses": c.get("alternative_hypotheses", []),
                "key_genes": c.get("key_genes", []),
                "key_cell_types": c.get("key_cell_types", []),
                "key_pathways": c.get("key_pathways", []),
                "organism": c.get("organism", ""),
                "tissue": c.get("tissue", ""),
                "experimental_context": c.get("experimental_context", ""),
            }
            for c in candidates
            if isinstance(c, dict)
        ]
        prompt = f"""You are the PI designing a bounded PubMed search for research-direction candidates.

The Literature agent will execute exactly the literal concept groups you approve. You own the
evidence question, concept relationships, explicit required scope, alternative explanations,
and the limits of what a matching paper can establish. Do not infer aliases or add organism,
tissue, perturbation, disease, or model terms unless you explicitly author them as terms.
Alternatives in one concept group are OR; separate groups are AND. Exclusions are explicit
PI-authored NOT groups with a rationale. Return a task list for every candidate; an empty task
list for a candidate is a planning failure and must be disclosed, not replaced by a template.

Phase: {phase}
Project objective: {state.initial_instruction}
Candidates:
{json.dumps(candidate_payload, ensure_ascii=False)}
Prior direction-search records (technical context only):
{json.dumps(prior_records or [], ensure_ascii=False)[:12000]}

Each task must have this shape:
{{"task_id":"<unique>","direction_id":"<candidate id>","evidence_question":"...",
"purpose":"...","concept_groups":[{{"group_id":"g1","terms":[{{"text":"literal term","field":"tiab"}}]}}],
"exclusions":[{{"exclusion_id":"x1","terms":[{{"text":"literal term","field":"tiab"}}],"rationale":"..."}}],
"scope":{{"required_group_ids":["g1"],"required_constraints":{{}},"unrestricted_dimensions":[],"rationale":"..."}},
"mechanism_link":"...","can_establish":["..."],"cannot_establish":["..."]}}
Use only field tags all, tiab, title, author, mesh, majr, pt, sh, subh, nm. Do not put Boolean
operators, brackets, parentheses, quotes, or PubMed syntax inside term text.
Use at most six initial tasks per candidate. Split separate mechanism links or alternative
explanations into separate tasks when that makes the evidence boundary clearer; one paper or
one task addressing a link does not establish a complete mechanism chain. Keep each term literal.
Return JSON only: {{"tasks":[...],"planning_rationale":"..."}}."""
        response = await self.chat_without_history(prompt, temperature=0.2)
        parsed = self._safe_parse_json(response, default={})
        raw_tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        valid_tasks: List[Dict[str, Any]] = []
        planning_errors: Dict[str, str] = {}
        seen_tasks: set[str] = set()
        allowed_ids = set(candidate_ids)
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue
            try:
                task = DirectionSearchTask.model_validate(raw)
                if task.direction_id not in allowed_ids:
                    raise ValueError(f"unknown candidate id {task.direction_id!r}")
                if task.task_id in seen_tasks:
                    raise ValueError(f"duplicate task id {task.task_id!r}")
                seen_tasks.add(task.task_id)
                valid_tasks.append(task.model_dump(mode="json"))
            except Exception as exc:
                direction_id = str(raw.get("direction_id") or "")
                planning_errors[direction_id] = str(exc)[:240]
        for direction_id in candidate_ids:
            if direction_id and not any(t.get("direction_id") == direction_id for t in valid_tasks):
                planning_errors.setdefault(direction_id, "PI returned no valid search task")
        return {
            "policy_version": "direction_search_v1",
            "plan_id": str(parsed.get("plan_id") or "") if isinstance(parsed, dict) else "",
            "tasks": valid_tasks,
            "planning_errors": planning_errors,
            "planning_rationale": str(parsed.get("planning_rationale") or "") if isinstance(parsed, dict) else "",
            "plan_status": "completed" if valid_tasks and not planning_errors else "partial",
        }

    async def review_direction_literature_search(
        self,
        state: AnalysisState,
        packet: Dict[str, Any] | List[Dict[str, Any]],
        *,
        phase: str = "exploration",
    ) -> Dict[str, Any]:
        """Review retained execution/extraction evidence and choose finish/follow-up."""
        self.refresh_runtime_config_if_needed()
        packet_payload = packet if isinstance(packet, dict) else {"candidates": packet}
        # Keep the packet bounded per candidate and disclose omitted details in
        # structured fields.  Never slice the serialized tail: that silently
        # removes the last candidates and their zero-result tasks.
        packet_for_prompt = dict(packet_payload)
        packet_for_prompt["candidates"] = [
            dict(item) for item in (packet_payload.get("candidates") or [])
            if isinstance(item, dict)
        ]
        packet_for_prompt["article_index"] = list(
            packet_payload.get("article_index") or []
        )
        packet_for_prompt["requested_article_details"] = list(
            packet_payload.get("requested_article_details") or []
        )
        packet_text = json.dumps(packet_for_prompt, ensure_ascii=False)
        prompt = f"""You are the PI reviewing a bounded, PI-directed literature search for research directions.

Every candidate packet is present, including zero-result, failed, timeout, and unexecuted tasks.
Counts describe retrieval only. Raw article identities, source passages, study context, and
applicability limits are the evidence you must weigh. A zero-result or a Literature provisional
label does not establish novelty. Keep scientific classifications and uncertainty explicit.
A source_passage marked unverified is not an exact quotation: inspect its retained abstract in
requested_article_details and its limitation before weighing the finding. Do not discard the
article or infer that its scientific observation is false solely from quotation mismatch.
You may request at most one follow-up task batch per candidate, and only with an evidence gap
and scope-change rationale. Named PMID detail requests must use retained article IDs. When
requested_article_details are present, use those raw sources and either finish or return an
explicit follow-up task; do not repeat a detail request that the packet already fulfilled.

Phase: {phase}
Packet:
{packet_text}

Return JSON only with one review per candidate:
{{"reviews":[{{"direction_id":"...","action":"finish|search_more",
"requested_detail_pmids":[],"follow_up_tasks":[{{"task_id":"<new globally unique id>",
"direction_id":"<same candidate id>","evidence_question":"...","purpose":"...",
"concept_groups":[{{"group_id":"g1","terms":[{{"text":"literal term","field":"tiab"}}]}}],
"exclusions":[],"scope":{{"required_group_ids":["g1"],"required_constraints":{{}},
"unrestricted_dimensions":[],"rationale":"..."}},"mechanism_link":"...",
"can_establish":["..."],"cannot_establish":["..."]}}],"rationale":"...","uncertainty":[],
"cited_article_ids":[],"final_scientific_classifications":{{"literature_novelty":"unknown",
"support_level":"unknown","contradiction_level":"unknown","prior_art_level":"unknown"}},
"assessment_status":"assessed|uncertain|unavailable|unassessed"}}]}}.
Candidate and PMID IDs must be present in the packet. Every follow_up_tasks entry must use the
same direction_id as its review and a new task_id not present anywhere in the packet. Return finish
with unknown classifications when evidence is insufficient or technically unavailable."""
        response = await self.chat_without_history(prompt, temperature=0.2)
        parsed = self._safe_parse_json(response, default={})
        raw_reviews = parsed.get("reviews", []) if isinstance(parsed, dict) else []
        if not isinstance(raw_reviews, list):
            raw_reviews = []
        allowed_ids = {
            str(item.get("direction_id") or "")
            for item in (packet_payload.get("candidates") or [])
            if isinstance(item, dict)
        }
        allowed_pmids = {
            str(item.get("pmid") or "")
            for item in (packet_payload.get("article_index") or [])
            if isinstance(item, dict)
        }
        existing_task_ids = {
            str(task.get("task_id") or "")
            for candidate in (packet_payload.get("candidates") or [])
            if isinstance(candidate, dict)
            for task in (
                list(candidate.get("planned_tasks") or [])
                + list(candidate.get("tasks") or [])
            )
            if isinstance(task, dict)
        }
        candidate_pmids = {
            str(candidate.get("direction_id") or ""): {
                str(pmid) for pmid in (candidate.get("article_ids") or [])
            }
            for candidate in (packet_payload.get("candidates") or [])
            if isinstance(candidate, dict)
        }
        reviews: List[Dict[str, Any]] = []
        review_errors: Dict[str, str] = {}
        seen: set[str] = set()
        seen_follow_up_task_ids: set[str] = set()
        for raw in raw_reviews:
            if not isinstance(raw, dict):
                continue
            try:
                review = DirectionPIReview.model_validate(raw)
                if allowed_ids and review.direction_id not in allowed_ids:
                    raise ValueError(f"unknown candidate id {review.direction_id!r}")
                if review.direction_id in seen:
                    raise ValueError(f"duplicate review for {review.direction_id!r}")
                invalid_pmids = set(review.requested_detail_pmids) | set(review.cited_article_ids)
                invalid_pmids -= allowed_pmids & candidate_pmids.get(review.direction_id, set())
                if invalid_pmids:
                    raise ValueError(f"unknown article ids: {sorted(invalid_pmids)}")
                for task in review.follow_up_tasks:
                    if task.direction_id != review.direction_id:
                        raise ValueError(
                            f"follow-up task {task.task_id!r} must use review direction_id "
                            f"{review.direction_id!r}"
                        )
                    if (
                        task.task_id in existing_task_ids
                        or task.task_id in seen_follow_up_task_ids
                    ):
                        raise ValueError(
                            f"follow-up task id {task.task_id!r} is not globally unique"
                        )
                    seen_follow_up_task_ids.add(task.task_id)
                seen.add(review.direction_id)
                reviews.append(review.model_dump(mode="json"))
            except Exception as exc:
                review_errors[str(raw.get("direction_id") or "")] = str(exc)[:240]
        for direction_id in sorted(allowed_ids - seen):
            review_errors.setdefault(direction_id, "PI review unavailable for candidate")
        return {"reviews": reviews, "review_errors": review_errors}

    async def generate_raw_direction_candidates(
        self,
        state: AnalysisState,
        round_result: RoundResult,
    ) -> List[Dict[str, Any]]:
        """
        Generate raw research direction candidates without final scoring.
        Called by generate_exploration_report as a separate substep.
        """
        logger.info(f"[PI] Generating raw direction candidates")
        self.reset_history()

        ann_summary = self._format_cluster_annotation_summary(state)

        mech_programs = getattr(state, "data_driven_mechanism_programs", []) or []
        mech_block = ""
        if mech_programs:
            lines = ["## Data-Driven Mechanism Programs",
                     "Use these programs as candidate evidence, not as conclusions.",
                     "Do not overclaim named mechanisms unless supported by data and literature.",
                     "Prefer directions grounded in concrete observations and cross-modal support.",
                     "For every mechanism-derived direction, include alternative hypotheses and missing evidence.",
                     ""]
            for p in mech_programs[-8:]:
                lines.append(f"- {p.to_prompt_text(style='direction', max_observations=2, max_genes=12, max_missing=2)}")
            mech_block = "\n".join(lines)

        prompt = f"""Based on the Round 1 exploration results, propose raw research direction candidates.

Propose as many as the results actually support — usually a handful. If Round 1 produced
nothing that would justify a further round of investigation, return an EMPTY candidates
list and say why in no_candidate_reason. That is a legitimate and useful answer: a
manufactured direction consumes a whole round chasing something the data never suggested.
Do not pad the list to reach a count.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}
- Species: {state.species or "Not specified"}
{state.build_experimental_design_summary()}

## Round 1 Results
**Bioinformatics findings**: {json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False)}
**Pathology findings**: {json.dumps(round_result.pathology_findings[:5], ensure_ascii=False)}
**Biochemistry findings**: {json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False)}
**Literature findings**: {json.dumps(round_result.literature_findings[:5], ensure_ascii=False)}
**PI Summary**: {round_result.pi_summary[:1000]}
{render_evidence_briefs(round_result.bioinformatics_finding_records)}

## Cell Annotations
{ann_summary}

{mech_block}

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

For each candidate, provide ALL of the following fields:
- direction_id: Unique identifier like "dir_001"
- direction: A specific, actionable research question
- core_observation: Concrete Round 1 data observation motivating the candidate. Must cite specific data elements such as cluster ID, cell type, gene, pathway, fold-change, p-value, correlation, module score, or sample-level observation.
- main_hypothesis: Primary testable hypothesis.
- alternative_hypotheses: Array of at least two strings unless the data clearly excludes alternatives. Draw the alternatives from THIS study's measured entities and design. Consider at least: a different molecular driver of the same observation, a different biological process class, the reverse causal direction, an unmodelled covariate or composition shift, and technical artifacts (ambient signal, doublets, batch, depth). Never return a single string.
- artifact_risks: Array of strings describing technical or interpretive risks such as batch effect, doublets, low-quality cluster, annotation uncertainty, ambient RNA, dropout, or confounded design. Never return a single string.
- rationale: Why this direction is interesting based on the data
- supporting_data: Array of objects with string fields source, evidence, and strength (high|medium|low). Never return a scalar.
- key_genes: Array of relevant gene strings. Never return a single string.
- key_cell_types: Array of relevant cell-type strings. Never return a single string.
- key_pathways: Array of relevant pathway strings. Never return a single string.
- key_uncertainties: Array of strings describing what remains uncertain or needs validation.
- suggested_literature_topics: Array of natural-language topics for literature search.
- suggested_validation_analysis: Array of computational follow-up analysis strings.
- suggested_wet_lab_validation: Array of wet-lab validation experiment strings.
- decision_criteria: Array of strings describing results that would support or weaken the hypothesis.
- biology_expectedness: one of "expected", "partially_expected", "unexpected", "unknown"
- functional_impact_hypothesis: Brief hypothesis about functional/clinical impact
- prerequisite_dependencies: Array of direction_id strings that should be investigated first (empty array if none)
- organism: {state.species or "Not specified"}
- tissue: Tissue context from the experiment
- experimental_context: The experimental design context

Do not produce generic core_observation text like "the data suggest interesting patterns". The core_observation must name the specific data element that triggered the direction.

IMPORTANT: Do NOT assign final innovation, literature novelty, data reliability, or priority scores. Those will be evaluated in a separate step with literature evidence.

Return a JSON object with a single key "candidates" containing the list of candidate objects:
{{
    "candidates": [
        {{
            "direction_id": "dir_001",
            "direction": "...",
            "hypothesis": "...",
            ...
        }}
    ]
}}
"""
        self.last_candidate_generation_error = None
        data = {}
        for attempt in range(2):
            response = await self.chat(prompt, temperature=0.3)
            data = self._safe_parse_json(response, default={})
            # A list holding no usable object is as unreadable as no list at all,
            # so it has to be judged HERE -- checking it after the retry loop meant
            # {"candidates": ["bad"]} was declared a structural failure without ever
            # being retried.
            if _candidates_are_usable(data):
                break
            logger.warning(
                "[PI] Direction-candidate response unreadable (attempt %s/2)", attempt + 1
            )
        # "The PI proposed no direction" and "we could not read the PI's reply" are
        # different facts and only the second is a malfunction. Conflating them
        # meant an honest empty answer was overwritten with a fabricated
        # "Continue analysis based on Round 1 findings" candidate, which then flowed
        # through scoring and selection as though the PI had proposed it.
        response_readable = _candidates_are_usable(data)
        candidates = data.get("candidates", []) if response_readable else []
        if not candidates and response_readable:
            logger.info(
                "[PI] No research direction candidates proposed: %s",
                (data.get("no_candidate_reason") or "no reason given")
                if isinstance(data, dict) else "no reason given",
            )
            return []
        if not candidates:
            logger.error(
                "[PI] Direction-candidate response could not be parsed; returning no "
                "candidates and recording the failure out-of-band"
            )
            self.last_candidate_generation_error = (
                "The PI's Round 1 direction-candidate response could not be read. No "
                "candidates were produced by this attempt. This is a tooling failure, not "
                "a finding that the data supports no direction."
            )
            return []
        return [
            _normalize_direction_candidate(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]

    async def generate_integration_direction_candidates(
        self,
        state: "AnalysisState",
        cumulative_rounds: List["RoundResult"],
        integration_report: str,
        new_literature: str,
    ) -> List[Dict[str, Any]]:
        """NEW direction candidates from the CUMULATIVE integration of all
        innovation rounds (round 2..current). Excludes already-chosen directions
        (dedup). Output field schema matches generate_raw_direction_candidates."""
        self.reset_history()
        # Not user_chosen_directions: that field holds only the most recent
        # batch, so every earlier cycle's choices would be invisible here and an
        # already-investigated direction could come back as "new".
        already_chosen = investigated_direction_texts(state)
        rounds_block = []
        for rr in cumulative_rounds:
            rounds_block.append(
                f"### Round {rr.round_number} — direction: {rr.direction or 'N/A'}\n"
                f"- Bioinformatics: {json.dumps(rr.bioinformatics_findings[:8], ensure_ascii=False)}\n"
                f"- Pathology: {json.dumps(rr.pathology_findings[:3], ensure_ascii=False)}\n"
                f"- Biochemistry: {json.dumps(rr.biochemistry_findings[:3], ensure_ascii=False)}\n"
                f"- Literature: {json.dumps(rr.literature_findings[:3], ensure_ascii=False)}\n"
                f"- Summary: {(rr.pi_summary or '')[:600]}"
            )
        prompt = f"""Based on the CUMULATIVE integration of ALL innovation rounds so far, propose NEW research direction candidates that go BEYOND what has already been investigated.

Propose as many as the integrated results support, and none if they support none:
returning an EMPTY candidates list means the rounds so far have answered what they
set out to answer, which is a legitimate outcome and ends the study cleanly. Do not
pad the list to reach a count.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}
- Species: {state.species or "Not specified"}
{state.build_experimental_design_summary()}

## Cross-Direction Integration Analysis
{integration_report[:4000]}

## All Innovation Rounds (cumulative, round 2..current)
{chr(10).join(rounds_block)}
{render_evidence_briefs([record for result in cumulative_rounds for record in result.bioinformatics_finding_records])}

## Newly Retrieved Literature (integration-facing search)
{new_literature[:3000]}

## Already-Investigated Directions (DO NOT repeat or trivially rephrase)
{json.dumps(already_chosen, ensure_ascii=False)}

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

Only propose directions genuinely NEW relative to the already-investigated list and grounded in concrete cumulative observations. If nothing sufficiently novel and evidence-supported exists, return an empty list [].

For each candidate, provide ALL fields with these exact JSON types:
- direction_id, direction, core_observation, main_hypothesis, rationale,
  functional_impact_hypothesis, organism, tissue, experimental_context: strings
- alternative_hypotheses, artifact_risks, key_genes, key_cell_types,
  key_pathways, key_uncertainties, suggested_literature_topics,
  suggested_validation_analysis, suggested_wet_lab_validation,
  decision_criteria: arrays of strings (never a single string)
- supporting_data: an array of objects, each with string fields
  source, evidence, and strength (high|medium|low); never a single string
- biology_expectedness: expected|partially_expected|unexpected|unknown
- prerequisite_dependencies: array of direction_id strings

Example field shapes:
{{"alternative_hypotheses": ["..."], "artifact_risks": ["..."],
  "supporting_data": [{{"source": "round 2 bioinformatics",
  "evidence": "...", "strength": "high"}}],
  "suggested_validation_analysis": ["..."],
  "suggested_wet_lab_validation": ["..."], "decision_criteria": ["..."]}}

Return a JSON array. Return [] if no sufficiently novel direction exists."""
        # This prompt contracts for a top-level JSON array; the dict parser drops
        # valid arrays. No `default` is passed on purpose: None then means "could
        # not parse" and [] means "the PI proposed none", which default=[] merged
        # into one indistinguishable answer.
        self.last_candidate_generation_error = None
        data = None
        for attempt in range(2):
            response = await self.chat(prompt, temperature=0.3)
            data = safe_parse_json_array(response)
            if isinstance(data, list) and (
                not data or any(isinstance(c, dict) for c in data)
            ):
                break
            logger.warning(
                "[PI] Integration candidate response unreadable (attempt %s/2)", attempt + 1
            )
        if not isinstance(data, list):
            logger.error("[PI] Integration candidate generation unreadable after retry")
            self.last_candidate_generation_error = (
                "The PI's integration direction-candidate response could not be read after a "
                "retry, so no candidates were produced by this attempt. This is a tooling "
                "failure, not a finding that no new direction remains."
            )
            return []
        usable = [candidate for candidate in data if isinstance(candidate, dict)]
        if data and not usable:
            self.last_candidate_generation_error = (
                "The PI's integration direction-candidate response contained no usable "
                "candidate objects. This is a structural failure, not a finding that no new "
                "direction remains."
            )
            return []
        return [_normalize_direction_candidate(candidate) for candidate in usable]

    async def score_integration_directions_with_evidence(
        self,
        state: "AnalysisState",
        raw_candidates: List[Dict[str, Any]],
        literature_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score integration-round candidates with literature novelty evidence.
        Returns a scored candidate LIST (not an ExplorationReport). literature_evidence
        is the list returned by LiteratureAgent.evaluate_direction_novelty_candidates."""
        if not raw_candidates:
            return []
        self.reset_history()
        candidate_count = len(raw_candidates)
        pi_directed_evidence = any(
            isinstance(item, dict) and item.get("policy_version")
            for item in (literature_evidence or [])
        )
        policy_scoring_note = (
            "\n## PI-directed evidence contract\n"
            "For records with policy_version, treat counts, returned article counts, and executed "
            "task categories as retrieval measurements only. Do not derive novelty, support, "
            "contradiction, completion, recommendation, or candidate retention from counts. "
            "Use explicit PI assessment, cited PMIDs, source passages, applicability limits, "
            "and technical limitations. Keep literature_novelty and related fields unknown when "
            "the PI did not explicitly assess them; every candidate remains eligible for scoring.\n"
            if pi_directed_evidence else ""
        )
        prompt = f"""You are the PI scoring NEW research direction candidates that emerged from cross-direction integration.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}

## Direction Candidates ({candidate_count} in total, every one of them listed)
{budgeted_prompt_block(raw_candidates, CANDIDATE_PROMPT_FIELDS, SCORER_CANDIDATE_BUDGET)}

## Literature Novelty Evidence (per candidate)
{budgeted_prompt_block(literature_evidence, EVIDENCE_PROMPT_FIELDS, SCORER_EVIDENCE_BUDGET)}
{policy_scoring_note}

Score every one of the {candidate_count} candidates. Return all {candidate_count}; this step ranks, it does not filter. Selecting which directions to pursue happens in a later step that reads your ranking, and a candidate missing from your reply cannot be reconsidered there.

For each candidate return its direction_id and ONLY the fields below. Every other field it
already carries is merged back onto it automatically, so restating them spends the response
budget this step needs to finish:
- novelty_score: 0-10 (higher = more novel vs existing literature)
- evidence_strength: "strong" | "moderate" | "weak"
- confidence: "high" | "medium" | "low"
- report_recommendation: "headline" | "secondary" | "future_direction" | "do_not_emphasize"
- priority_rank: integer, 1 = highest, each candidate a distinct rank

Two things to weigh, neither of which lowers a candidate below another on its own:
- Prior-art volume. A heavily published mechanism means the field is active, not that the question is answered for this project. Let it inform novelty_score. When the novelty record is unknown, timed out, or absent, prior-art volume is the only signal left and it settles nothing.
- Absence of observed supporting data. These candidates are directions no round has run, so by construction none of them has observed support yet. Producing that support is what running the direction is for. Record the shortfall in evidence_strength and confidence.

Rank a candidate that speaks directly to the stated research objective above one that does not.

Return a JSON array of all {candidate_count} candidates, sorted by priority_rank ascending."""
        data: List[Any] = []
        readable = False
        for attempt in range(2):
            response = await self.chat(prompt, temperature=0.3)
            # This prompt contracts for a top-level JSON array; the dict parser drops valid arrays.
            data = safe_parse_json_array(response, default=[])
            readable = isinstance(data, list) and bool(data)
            if readable:
                break
            logger.warning(
                "[PI] Integration-scoring response unreadable (attempt %s/2)", attempt + 1
            )
        if not readable:
            # Every sibling call on this path retries and then says which of the two
            # things happened. Returning [] here would hand the integration node an
            # empty candidate list, which it reports as "no sufficiently novel
            # direction remains" -- a scientific conclusion drawn from a parse error.
            self.last_candidate_scoring_error = (
                "DIRECTION SCORING UNAVAILABLE: the PI's scoring response could not be "
                "read after a retry, so the candidates carry no ranking. This is a "
                "tooling failure, not a judgement about the candidates; all of them are "
                "passed on unranked."
            )
            logger.error(
                "[PI] Integration scoring unreadable after retry; passing candidates on unranked"
            )
            return _restore_unscored_candidates([], raw_candidates)
        self.last_candidate_scoring_error = None
        scored = _normalize_integration_scored_candidates(data, raw_candidates)
        # Ranking is not a licence to remove. An omission here is unrecoverable --
        # nothing downstream rereads raw_candidates -- so it is repaired rather
        # than read as a verdict.
        return _restore_unscored_candidates(scored, raw_candidates)

    async def score_directions_with_evidence(
        self,
        state: AnalysisState,
        round_result: RoundResult,
        raw_candidates: List[Dict[str, Any]],
        literature_evidence: List[Dict[str, Any]],
    ) -> ExplorationReport:
        """
        Score raw direction candidates with literature evidence to produce a complete ExplorationReport.
        """
        logger.info(f"[PI] Scoring {len(raw_candidates)} direction candidates with evidence")
        raw_candidates = [
            _normalize_direction_candidate(candidate)
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        ]
        self.reset_history()

        ann_summary = self._format_cluster_annotation_summary(state)
        pi_directed_evidence = any(
            isinstance(item, dict) and item.get("policy_version")
            for item in (literature_evidence or [])
        )
        policy_scoring_note = (
            "\n## PI-directed evidence contract\n"
            "For records with policy_version, counts and executed task categories are technical "
            "retrieval measurements only. Do not derive novelty, support, contradiction, "
            "completion, recommendation, or candidate retention from counts. Read the explicit "
            "PI assessment, cited article IDs, source passages, applicability limits, and "
            "technical limitations. Keep scientific summary fields unknown unless PI assessed "
            "them; every candidate must reach scoring.\n"
            if pi_directed_evidence else ""
        )

        prompt = f"""You are the PI scoring research direction candidates with literature evidence.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}

## Round 1 Results
**Bioinformatics findings**: {json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False)}
**Pathology findings**: {json.dumps(round_result.pathology_findings[:5], ensure_ascii=False)}
**Biochemistry findings**: {json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False)}
**Literature findings**: {json.dumps(round_result.literature_findings[:5], ensure_ascii=False)}
**PI Summary**: {round_result.pi_summary[:1000]}

## Cell Annotations
{ann_summary}

## Raw Direction Candidates
{json.dumps(raw_candidates, ensure_ascii=False, indent=2)}

## Literature Evidence
{budgeted_prompt_block(literature_evidence, EVIDENCE_PROMPT_FIELDS, SCORER_EVIDENCE_BUDGET) if literature_evidence else "No literature evidence provided."}
{policy_scoring_note}

### PI SCORING RULES

Apply ALL of the following scoring rules:

1. **Data reliability is the baseline**: If data_reliability=low, this direction cannot be a top recommendation unless the user explicitly chooses it.
2. **Failed literature confidence forces unknown novelty**: If literature_confidence=failed, literature_novelty must be "unknown".
3. **Speculative combinations**: High literature_novelty with low data_reliability is speculative and must not be the default recommendation; add a risk flag.
4. **Strong combination dominates**: a candidate that is high on data_reliability AND high on data_novelty AND high on mechanistic_importance ranks above one that is high on only some of them. Novelty never substitutes for reliability.
5. **Complementarity**: when recommending multiple directions, they must be complementary (e.g., one describes network architecture, one tests causal origin), not redundant.
6. **Independence**: functional_impact must be assessed separately from mechanistic_importance.
7. **Independence**: biology_expectedness must be assessed separately from literature_novelty and data_novelty.
8. **Prerequisites**: if a direction has prerequisites, its prerequisite direction must also be selected or explicitly noted as already satisfied by prior results.
9. **Composite priority (single definition)**: overall_priority is derived from ALL of the scoring dimensions below, with data_reliability, data_sufficiency and artifact_risk acting as limiting factors — a candidate cannot be high priority while any of those is unfavourable, however novel or impactful it looks. When dimensions conflict, give explicit reasoning. This rule is the only definition of overall_priority; no other rule overrides it.
10. **Conflict reconciliation**: when literature_novelty and biology_expectedness conflict, reconcile the discrepancy explicitly in literature_support_summary, data_support_summary, or risk_flags.
11. **Negative-validation caveat**: when an evidence record has negative_validation_confidence=medium, any literature_novelty=high must be explicitly justified in literature_support_summary because at least one evidence axis is incomplete.
12. **Read the evidence, not only its count**: key_evidence names the articles behind direct_evidence_count and contradictory_evidence_count. Retrieval is keyword-based, so an article can be counted as direct while addressing a different question. When the named titles do not bear on this candidate, say so in literature_support_summary and do not let the count alone move literature_novelty in either direction.

### SCORING DIMENSIONS

For each candidate, score ALL of these dimensions:

1. **data_reliability**: "high" / "medium" / "low" — Based on sample sizes, effect sizes, statistical significance, and data quality.
2. **literature_novelty**: "high" / "medium" / "low" / "unknown" — How well-studied is this area? Use "unknown" when the PI assessment remains scientifically uncertain or unavailable, including bounded zero-result searches and technical failures.
3. **biology_expectedness**: "expected" / "partially_expected" / "unexpected" / "unknown" — Given known pathways/mechanisms, is the finding expected, partly expected, or unexpected? This is PI reasoning, not PubMed coverage.
4. **data_novelty**: "high" / "medium" / "low" — How unexpected or surprising is this finding in the data?
5. **mechanistic_importance**: "high" / "medium" / "low" — How central is the proposed mechanism to the phenotype/disease?
6. **functional_impact**: "high" / "medium" / "low" — What is the likely functional or clinical impact?
7. **data_sufficiency**: "sufficient" / "partial" / "insufficient" — Is there enough data to support this direction?
8. **methodological_gap**: "none" / "minor" / "major" — What methodological gaps exist?

Also derive:
- **overall_priority**: "high" / "medium" / "low" — derived as scoring rule 9 specifies.
- **risk_flags**: List of specific risk flags (e.g., "cross_species_extrapolation", "low_statistical_power").
- **prerequisite_dependencies**: Which other direction_ids must be completed first.
- **evidence_ref**: Must match the direction_id from literature evidence.

Scoring rubric:

biological_plausibility:
- high: mechanism is consistent with established biology and direct pathway/literature evidence.
- medium: biologically possible but depends on indirect or analogous evidence.
- low: contradicts known biology, lacks meaningful support, or requires a novel pathway with no precedent.

validation_feasibility:
- high: routine assays such as qPCR, Western, IF/IHC, ELISA, standard flow cytometry, or standard perturbation.
- medium: specialized equipment or expertise such as ICP-MS, Seahorse, FACS sorting, spatial transcriptomics, advanced imaging, or animal generation.
- low: technically prohibitive, unavailable reagents/models, or cannot distinguish competing hypotheses.

If direction evidence contains novelty_status=known_area, literature_novelty should usually be low or medium unless novelty comes from a new model, species, tissue, cell type, perturbation, or validation strategy.
If direction evidence contains novelty_status=adjacent_work_only, the search retrieved related work but nothing that studies this question directly. Related work is context, not an answer: neither the volume of it nor the absence of a direct hit settles the question, and literature_support_summary must say which of the two the record actually shows.
If direction evidence contains novelty_status=unknown_due_to_timeout or unknown_due_to_failure, literature_novelty must be unknown and recommendation must be conservative.
Contradictory evidence must never be discounted for being contradictory, and must never be left out of the summaries. Weight every piece of evidence — supporting and contradictory alike — by the same criteria: study design and statistical power, how directly it addresses this claim, and how well its model, tissue, and species apply here. One well-powered direct contradiction outweighs several indirect or correlational supports, and the reverse also holds; do not average them as if the categories were symmetric.

Finally, produce the **direction_dependency_graph** — a mapping from each direction_id to a list of direction_ids it depends on (from prerequisite_dependencies).

Return a complete structured report JSON:
{{
    "cluster_overview": "...",
    "data_quality_warnings": [...],
    "strongest_heterogeneity": [...],
    "key_pathway_changes": [...],
    "abnormal_markers": [...],
    "low_confidence_annotations": [...],
    "pi_direction_assessment": "...",
    "research_direction_candidates": [
        {{
            "direction_id": "dir_001",
            "evidence_ref": "dir_001",
            "contradicts_stated_premise": false,
            "validation_plan": {{"decision_criteria": ["..."]}},
            "biological_plausibility": "high|medium|low",
            "validation_feasibility": "high|medium|low",
            "artifact_risk": "high|medium|low",
            "recommendation": "prioritize|secondary|needs_more_data|avoid",
            "pi_review_summary": "...",
            "data_support_summary": "...",
            "literature_support_summary": "...",
            "data_reliability": "high|medium|low",
            "literature_novelty": "high|medium|low|unknown",
            "biology_expectedness": "expected|partially_expected|unexpected|unknown",
            "data_novelty": "high|medium|low",
            "mechanistic_importance": "high|medium|low",
            "functional_impact": "high|medium|low",
            "data_sufficiency": "sufficient|partial|insufficient",
            "methodological_gap": "none|minor|major",
            "overall_priority": "high|medium|low",
            "risk_flags": ["..."],
            "prerequisite_dependencies": ["dir_id", "..."]
        }}
    ],
    "direction_dependency_graph": {{"dir_001": ["dir_002"], ...}},
    "direction_selection_summary": {{}}
}}

Return one entry for every candidate you were given, keyed by direction_id.

Return ONLY the fields shown above. Everything else a candidate already carries -- direction,
core_observation, main_hypothesis, alternative_hypotheses, artifact_risks, supporting_data,
contradictory_or_missing_evidence, rationale, key_genes, key_pathways, key_cell_types and the
suggested_* lists -- is merged onto your entry automatically, and so are the raw candidate list
and the evidence matrix. Under validation_plan give only decision_criteria; the analysis and
wet-lab entries are carried over. Restating any of it buys nothing and costs the one thing this
reply is short of: a scoring response long enough to restate its own input does not finish
inside the request deadline, and a scoring step that does not finish loses the whole round's
literature evidence with it.
"""
        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        # Build ExplorationReport from parsed data. Use from_llm_dict so a single
        # type-mismatched field (e.g. the LLM emitting a string for the dict-typed
        # direction_selection_summary) is dropped rather than raising a
        # ValidationError that would abort Report-A generation and silently skip
        # the exploration->innovation transition + direction-selection gate.
        report = ExplorationReport.from_llm_dict(data)

        # Normalize: research_direction_candidates must hold the FINAL scored candidates.
        scored = data.get("research_direction_candidates") or []
        # Recover if the LLM placed scored candidates under raw_direction_candidates by mistake.
        if not any(isinstance(c, dict) and c.get("overall_priority") for c in scored):
            recovered = [c for c in (data.get("raw_direction_candidates") or [])
                         if isinstance(c, dict) and c.get("overall_priority")]
            if recovered:
                scored = recovered
        # Merge score-only or shape-unstable responses onto canonical raw
        # candidates before any downstream validation or rendering.
        scored = _normalize_integration_scored_candidates(scored, raw_candidates)

        # Guarantee direction_id + evidence_ref on every scored candidate.
        raw_by_index = {i: rc for i, rc in enumerate(raw_candidates)}
        for i, c in enumerate(scored):
            if not isinstance(c, dict):
                continue
            if not c.get("direction_id"):
                c["direction_id"] = (raw_by_index.get(i, {}) or {}).get("direction_id", f"dir_{i+1:03d}")
            if not c.get("evidence_ref"):
                c["evidence_ref"] = c["direction_id"]
        # Build round context for soft retry of missing critical fields.
        round_context = "\n".join([
            "Bioinformatics findings: " + json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False),
            "Pathology findings: " + json.dumps(round_result.pathology_findings[:5], ensure_ascii=False),
            "Biochemistry findings: " + json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False),
            "Literature findings: " + json.dumps(round_result.literature_findings[:5], ensure_ascii=False),
            "PI summary: " + (round_result.pi_summary or "")[:1000],
        ])
        for i, candidate in enumerate(scored):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("overall_priority") != "high" and candidate.get("recommendation") != "prioritize":
                continue
            missing = self._critical_direction_fields_missing(candidate)
            if not missing:
                continue
            try:
                scored[i] = await self._soft_retry_direction_candidate(candidate, round_context, missing)
            except Exception as retry_err:
                self._append_unique_flag(candidate, "critical_field_retry_failed")
                candidate.setdefault("limitations", []).append(f"Critical field retry failed: {str(retry_err)[:120]}")
        scored = _normalize_integration_scored_candidates(scored, raw_candidates)
        scored = self._apply_direction_guardrails(scored, raw_candidates, literature_evidence)
        report.research_direction_candidates = scored

        # Preserve the raw candidate content in its canonical normalized shape.
        report.raw_direction_candidates = raw_candidates
        # Evidence matrix always preserved.
        if not report.direction_literature_evidence:
            report.direction_literature_evidence = literature_evidence

        # Keep direction candidates available while removing only cycle-closing edges.
        if report.direction_dependency_graph:
            self._sanitize_direction_dependency_graph(report)

        return report

    async def _soft_retry_direction_candidate(
        self,
        candidate: Dict[str, Any],
        round_context: str,
        missing_fields: List[str],
    ) -> Dict[str, Any]:
        prompt = f"""Repair one research direction candidate by filling only missing critical fields.

Round context:
{round_context[:3000]}

Candidate JSON:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Missing fields: {', '.join(missing_fields)}

Return ONLY the complete candidate JSON object. Preserve existing fields unless they directly conflict with the repaired fields. Required repaired fields:
- core_observation: specific Round 1 data element, not generic prose
- alternative_hypotheses: list of plausible alternatives
- artifact_risks: list of technical or interpretive risks
- contradictory_or_missing_evidence: list of missing or conflicting evidence
- validation_plan: object with next_analysis, wet_lab_validation, decision_criteria lists
- recommendation: prioritize|secondary|needs_more_data|avoid
"""
        response = await self.chat_without_history(prompt, temperature=0.2)
        repaired = self._safe_parse_json(response, default={})
        if isinstance(repaired, dict) and repaired.get("direction"):
            merged = dict(candidate)
            merged.update(repaired)
            return merged
        return candidate
