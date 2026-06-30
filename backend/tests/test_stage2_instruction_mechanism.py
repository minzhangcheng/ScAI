from backend.workflow.agents import (
    _build_stage2_instruction, _build_exploration_instruction, build_phase_instruction,
)

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
