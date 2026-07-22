from backend.workflow.agents import (
    _build_stage2_instruction, _build_exploration_instruction, build_phase_instruction,
)
from backend.utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

_MECHANISM_INSTRUCTION = "register_mechanism_programs"


def test_stage2_instruction_mentions_register_tool():
    instr = _build_stage2_instruction(has_bulk=True, has_scrna=True)
    assert _MECHANISM_INSTRUCTION in instr
    assert "data-driven mechanism program" in instr.lower()
    assert "no_program_reason" in instr


def test_exploration_instruction_mentions_register_tool():
    instr = _build_exploration_instruction(has_bulk=True, has_scrna=True)
    assert _MECHANISM_INSTRUCTION in instr
    assert "no_program_reason" in instr


def test_full_exploration_route_carries_instruction():
    # planning_stage="full" + exploration routes through _build_exploration_instruction
    instr = build_phase_instruction(
        planning_stage="full", analysis_phase="exploration",
        has_bulk=False, has_scrna=True, direction="",
    )
    assert _MECHANISM_INSTRUCTION in instr


def test_instructions_have_no_hardcoded_mechanisms():
    for instr in (
        _build_stage2_instruction(has_bulk=False, has_scrna=True),
        _build_exploration_instruction(has_bulk=False, has_scrna=True),
    ):
        low = instr.lower()
        for banned in ("ferroptosis", "apoptosis", "cuproptosis", "nrf2"):
            assert banned not in low


def test_instructions_balance_unbiased_discovery_with_explicit_naming():
    for instr in (
        _build_stage2_instruction(has_bulk=False, has_scrna=True),
        _build_exploration_instruction(has_bulk=False, has_scrna=True),
    ):
        assert MECHANISM_EVIDENCE_POLICY in instr
        assert "name it explicitly" in instr
        assert "prespecified hypotheses" in instr
        assert "causal mechanism execution" in instr
