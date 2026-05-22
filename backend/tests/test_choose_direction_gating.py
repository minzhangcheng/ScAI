"""P1-5: choose-direction must only accept calls at the post-exploration gate."""
import re
import backend.routes.interaction as ix_mod


def test_choose_direction_checks_for_direction_gate():
    """P1-5: choose-direction must verify session is at post-exploration gate.

    After P1-9 session-lock refactor, the body lives in `_choose_direction_locked`.
    Search both the public handler and the locked helper.
    """
    src = open(ix_mod.__file__).read()
    fn = re.search(r"async def _choose_direction_locked\b.*?(?=\n(?:async )?def |\Z)", src, re.DOTALL)
    if not fn:
        fn = re.search(r"async def choose_direction\b.*?(?=\n(?:async )?def |\Z)", src, re.DOTALL)
    assert fn
    body = fn.group(0)
    # Must check current_round AND analysis_phase, not just status
    assert "current_round" in body, "P1-5: must check current_round"
    assert "INNOVATION" in body or "analysis_phase" in body, \
        "P1-5: must check analysis_phase"
    # Must reject with 400 if not at gate
    assert "400" in body and (
        ("direction" in body.lower() and "gate" in body.lower())
        or "direction selection" in body.lower()
    ), "P1-5: must reject non-gate calls with 400"
