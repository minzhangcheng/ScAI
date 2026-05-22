"""P1-1: directions_queue commit-on-next-round-entry."""
import re
import backend.workflow.planning as planning_mod
import backend.workflow.initialize as init_mod


def test_pi_planning_does_not_pop_directions_queue():
    """P1-1: _pi_planning must only peek the queue, not pop. Pop happens
    in _initialize_round on next round entry."""
    src = open(planning_mod.__file__).read()
    fn = re.search(r"async def _pi_planning\b.*?(?=\n    (?:async )?def |\Z)", src, re.DOTALL)
    assert fn
    body = fn.group(0)
    assert "directions_queue.pop(" not in body, \
        "P1-1: _pi_planning must not pop directions_queue (deferred to _initialize_round)"


def test_initialize_round_commits_direction():
    """P1-1: _initialize_round must commit the previously peeked direction by
    popping directions_queue[0] when it matches current_round_direction."""
    src = open(init_mod.__file__).read()
    fn = re.search(r"async def _initialize_round\b.*?(?=\n    (?:async )?def |\Z)", src, re.DOTALL)
    assert fn
    body = fn.group(0)
    assert "directions_queue.pop" in body or "directions_queue[1:]" in body, \
        "P1-1: _initialize_round must commit (pop) directions_queue on entry"
    assert "current_round_direction" in body, \
        "P1-1: _initialize_round must check current_round_direction"
