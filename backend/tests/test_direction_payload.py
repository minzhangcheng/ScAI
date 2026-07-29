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

CANDIDATE = {
    "direction": "Determine whether condition X drives program Y",
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


def test_module_carries_no_domain_vocabulary():
    source = inspect.getsource(direction_payload).lower()
    for domain_term in (
        "ferroptos", "apoptos", "cuproptos", "necroptos", "nrf2", "hepatocyte",
        "liver", "mouse", "copper",
    ):
        assert domain_term not in source
