from backend.workflow.annotation_validation import compute_validation_attempt_id


def _markers():
    return {"0": ["Alb", "Slc2a2"], "1": ["Clec4f", "Adgre1"]}


def test_attempt_id_is_stable():
    a = compute_validation_attempt_id(1, "liver", "mouse", "leiden", _markers())
    b = compute_validation_attempt_id(1, "liver", "mouse", "leiden", _markers())
    assert a == b
    assert isinstance(a, str) and len(a) == 64


def test_attempt_id_stable_under_marker_reordering():
    m1 = {"0": ["Alb", "Slc2a2"], "1": ["Clec4f"]}
    m2 = {"1": ["Clec4f"], "0": ["Slc2a2", "Alb"]}
    assert compute_validation_attempt_id(1, "liver", "mouse", "leiden", m1) == \
           compute_validation_attempt_id(1, "liver", "mouse", "leiden", m2)


def test_attempt_id_changes_with_inputs():
    base = compute_validation_attempt_id(1, "liver", "mouse", "leiden", _markers())
    assert compute_validation_attempt_id(2, "liver", "mouse", "leiden", _markers()) != base
    assert compute_validation_attempt_id(1, "kidney", "mouse", "leiden", _markers()) != base
    assert compute_validation_attempt_id(1, "liver", "human", "leiden", _markers()) != base
    assert compute_validation_attempt_id(1, "liver", "mouse", "louvain", _markers()) != base
    assert compute_validation_attempt_id(1, "liver", "mouse", "leiden", {"0": ["Alb"]}) != base
