"""A selected direction must be able to reach its own exploration payload."""
import inspect

from backend.models.schemas import AnalysisState
from backend.models.schema_parts.outputs import ExplorationReport
from backend.utils import direction_payload
from backend.utils.direction_payload import (
    build_direction_context_block,
    candidate_for_state,
    direction_entities,
    direction_requirements,
    find_direction_candidate,
    normalize_direction_text,
)

DIRECTION = "Determine whether condition X drives program Y"

CANDIDATE = {
    "direction": DIRECTION,
    "main_hypothesis": "Perturbation X activates program Y",
    "key_uncertainties": [
        "Whether the state corresponds to modeA, modeB, or reversible remodeling.",
    ],
    "alternative_hypotheses": ["The signature is secondary to injury."],
    "suggested_validation_analysis": [
        "Score nuclei for modeA, modeB, and modeC gene sets separately.",
    ],
    "validation_plan": {
        "next_analysis": [
            "Score nuclei for modeA, modeB, and modeC gene sets separately.",
            "Infer regulator activity for factorQ.",
        ],
        "decision_criteria": ["Support if modeA markers rise."],
    },
    "key_genes": ["Geneaa1", "Genebb2"],
    "key_pathways": ["Pathway alpha"],
    "key_cell_types": ["Celltype one"],
}


def test_normalization_is_case_and_whitespace_insensitive():
    assert normalize_direction_text("  Foo   BAR ") == "foo bar"
    assert normalize_direction_text(None) == ""


def test_exact_and_containment_matching():
    pool = [CANDIDATE]
    assert find_direction_candidate(pool, CANDIDATE["direction"]) is CANDIDATE
    assert find_direction_candidate(pool, CANDIDATE["direction"].upper()) is CANDIDATE
    edited = CANDIDATE["direction"] + " in the pilot cohort"
    assert find_direction_candidate(pool, edited) is CANDIDATE
    assert find_direction_candidate(pool, "an unrelated direction entirely") is None
    assert find_direction_candidate(pool, "") is None


def test_requirements_are_deduplicated_and_ordered():
    requirements = direction_requirements(CANDIDATE)
    assert requirements[0].startswith("Whether the state corresponds")
    # The same analysis appears in suggested_validation_analysis and in
    # validation_plan.next_analysis; it must be carried once.
    assert sum("modeC gene sets separately" in r for r in requirements) == 1
    assert "Infer regulator activity for factorQ." in requirements
    assert direction_requirements(None) == []
    assert direction_requirements({}) == []


def test_entities_collect_genes_pathways_and_cell_types():
    entities = direction_entities(CANDIDATE)
    assert entities == ["Geneaa1", "Genebb2", "Pathway alpha", "Celltype one"]


def test_context_block_carries_open_questions_and_entities():
    block = build_direction_context_block(CANDIDATE)
    assert "Main hypothesis" in block
    assert "modeA" in block
    assert "factorQ" in block
    assert "Geneaa1" in block
    assert build_direction_context_block(None) == ""
    assert build_direction_context_block({"direction": "bare"}) == ""


def test_candidate_resolves_from_state_including_raw_candidates():
    state = AnalysisState(user_id="u")
    assert candidate_for_state(state) is None

    state.exploration_report = ExplorationReport(raw_direction_candidates=[CANDIDATE])
    state.current_round_direction = CANDIDATE["direction"]
    assert candidate_for_state(state) == CANDIDATE

    state.exploration_report = ExplorationReport(research_direction_candidates=[CANDIDATE])
    assert candidate_for_state(state) == CANDIDATE


# --- the exact link a selection gate already recorded ----------------------

def test_recorded_candidate_index_beats_text_matching():
    """A gate stores which candidate a selection came from; use it.

    Two candidates can be worded similarly enough that a containment match
    binds the wrong one, and the payload that results steers a plan and a
    user-facing coverage warning without anything saying it was mismatched.
    """
    from backend.models.schema_parts.session import (
        DirectionSelectionRecord,
        SelectionEntry,
    )
    from backend.utils.direction_payload import candidate_from_selection_records

    # The active direction matches neither candidate exactly, so text matching
    # falls through to containment — and the shorter candidate swallows it.
    active = DIRECTION + " restricted to compartment Z, pilot cohort only"
    decoy = {"direction": DIRECTION, "key_genes": ["Wrongaa1"]}
    real = {"direction": DIRECTION + " restricted to compartment Z",
            "key_genes": ["Rightbb2"]}

    state = AnalysisState(user_id="u")
    state.exploration_report = ExplorationReport(research_direction_candidates=[decoy, real])
    state.current_round_direction = active
    state.direction_selection_records = [
        DirectionSelectionRecord(
            candidates=[decoy, real],
            selections=[SelectionEntry(direction=active, candidate_index=1)],
        )
    ]

    # Pydantic copies the frozen candidate list, so compare by value.
    assert candidate_from_selection_records(state, active) == real
    assert candidate_for_state(state)["key_genes"] == ["Rightbb2"]
    # Without the record, containment binds the wrong candidate silently.
    state.direction_selection_records = []
    assert candidate_for_state(state)["key_genes"] == ["Wrongaa1"]


def test_recorded_lookup_ignores_records_it_cannot_trust():
    from backend.models.schema_parts.session import (
        DirectionSelectionRecord,
        SelectionEntry,
    )
    from backend.utils.direction_payload import candidate_from_selection_records

    state = AnalysisState(user_id="u")
    state.direction_selection_records = [
        # No index recorded — this is what the PI-decide path and a
        # user-edited direction both produce, so the caller must fall back.
        DirectionSelectionRecord(
            candidates=[CANDIDATE],
            selections=[SelectionEntry(direction=DIRECTION, candidate_index=None)],
        ),
        # Index out of range for its own frozen candidate list.
        DirectionSelectionRecord(
            candidates=[CANDIDATE],
            selections=[SelectionEntry(direction=DIRECTION, candidate_index=7)],
        ),
    ]
    assert candidate_from_selection_records(state, DIRECTION) is None
    assert candidate_from_selection_records(state, "") is None


# --- fallback for directions that never had a candidate --------------------

def _round_result():
    from backend.models.schema_parts.analysis import DataDrivenMechanismProgram
    from backend.models.schema_parts.outputs import RoundResult

    return RoundResult(
        round_number=4,
        next_steps=[
            "[Bioinformatics] Score the remodelling axis against modeA and modeB panels.",
            "[Pathology] Stain matched sections for modeA-associated damage.",
            "[Biochemistry] Measure metalX in liver homogenate.",
            "Compare rankings against public gammareference datasets.",
        ],
        data_driven_mechanism_programs=[
            DataDrivenMechanismProgram(
                label="High-axis remodelling state",
                supporting_genes=["Geneaa1", "Genebb2"],
                supporting_pathways=["Pathway alpha"],
                missing_evidence=["Independent biological replicates."],
                alternative_explanations=["The state may reflect capture bias."],
            )
        ],
    )


def test_fallback_requirements_are_scoped_to_what_a_plan_can_execute():
    """A plan cannot run a stain, so checking one is a permanent false positive."""
    from backend.utils.direction_payload import context_from_round_result

    ctx = context_from_round_result(_round_result())
    assert ctx.source == "previous_round"
    assert ctx.requirements == [
        "Score the remodelling axis against modeA and modeB panels.",
        "Compare rankings against public gammareference datasets.",
    ]
    joined = " ".join(ctx.notes)
    assert "[Pathology]" in joined and "[Biochemistry]" in joined
    # Missing evidence informs the planner but is never held against the plan.
    assert "Independent biological replicates." in ctx.notes
    assert all("replicates" not in r for r in ctx.requirements)


def test_fallback_entities_come_from_the_previous_round_programs():
    from backend.utils.direction_payload import context_from_round_result

    ctx = context_from_round_result(_round_result())
    assert ctx.entities == [
        "High-axis remodelling state", "Geneaa1", "Genebb2", "Pathway alpha",
    ]
    assert context_from_round_result(None).source == "none"


def test_resolution_prefers_the_candidate_then_falls_back():
    from backend.utils.direction_payload import resolve_direction_context

    state = AnalysisState(user_id="u")
    state.current_round_direction = CANDIDATE["direction"]
    state.exploration_report = ExplorationReport(research_direction_candidates=[CANDIDATE])
    assert resolve_direction_context(state).source == "exploration_candidate"

    # A later round pursues a direction the exploration report never raised.
    state.current_round_direction = "An entirely different later-round question"
    state.round_results = [_round_result()]
    ctx = resolve_direction_context(state)
    assert ctx.source == "previous_round"
    assert ctx.requirements

    bare = AnalysisState(user_id="u")
    assert resolve_direction_context(bare).source == "none"


def test_resolution_skips_an_empty_intervening_round():
    """An integration round yields no next steps and no programs.

    Keying on the immediate predecessor therefore goes blank for every
    direction round that follows one, which is most of a long run.
    """
    from backend.models.schema_parts.outputs import RoundResult
    from backend.utils.direction_payload import resolve_direction_context

    state = AnalysisState(user_id="u")
    state.current_round_direction = "A later-round question with no candidate"
    state.round_results = [_round_result(), RoundResult(round_number=5)]
    ctx = resolve_direction_context(state)
    assert ctx.source == "previous_round"
    assert "Score the remodelling axis against modeA and modeB panels." in ctx.requirements


def test_context_block_labels_where_the_payload_came_from():
    from backend.utils.direction_payload import (
        build_direction_context_block,
        context_from_round_result,
    )

    block = build_direction_context_block(context_from_round_result(_round_result()))
    assert "carried from the previous round's results" in block
    assert "Score the remodelling axis" in block
    assert "Geneaa1" in block
    assert "capture bias" in block
    # A raw candidate dict is still accepted.
    assert "exploration round that raised this direction" in build_direction_context_block(CANDIDATE)


def test_module_carries_no_domain_vocabulary():
    source = inspect.getsource(direction_payload).lower()
    for domain_term in (
        "ferroptos", "apoptos", "cuproptos", "necroptos", "nrf2", "hepatocyte",
        "liver", "mouse", "copper",
    ):
        assert domain_term not in source
