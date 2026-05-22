import re
import pytest

import backend.workflow.synthesis as syn_mod


@pytest.mark.asyncio
async def test_signal_between_synthesis_and_wait_is_preserved():
    """P0-6: _wait_for_user_input must not call _continue_event.clear().

    The clear() in _pi_synthesis (before WAITING_INPUT status is announced)
    is the only legitimate one. A second clear inside _wait_for_user_input
    would discard signals that arrive during the LangGraph checkpoint I/O
    window between the two nodes.
    """
    source = open(syn_mod.__file__).read()

    wait_fn = re.search(
        r"async def _wait_for_user_input\b.*?(?=\n    async def |\Z)",
        source,
        re.DOTALL,
    )
    assert wait_fn, "could not locate _wait_for_user_input"
    wait_body = wait_fn.group(0)

    assert "_continue_event.clear()" not in wait_body, (
        "P0-6: _wait_for_user_input must not call _continue_event.clear()"
    )
