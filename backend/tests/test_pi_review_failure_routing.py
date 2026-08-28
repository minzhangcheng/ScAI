import re
import inspect
import backend.workflow.planning as planning_mod
from backend.workflow.planning_parts import dialogue


def test_pi_review_failure_exhaustion_sets_should_continue_false():
    """Static check: the failure branch in planning_dialogue must set
    should_continue=False to prevent revise loops."""
    body = inspect.getsource(dialogue.run_planning_dialogue_rounds)
    assert 'should_continue' in body and 'False' in body, \
        "P0-7: failure exhaustion path must set should_continue=False"
    # Locate the PI review failure-exhaustion branch (after the
    # "PI review failed after ... attempts" message).
    marker = body.find("PI review failed after")
    assert marker != -1, "could not locate PI review failure branch"
    # Find the next "status": "failed" after that marker.
    failed_idx = body.find('"status": "failed"', marker)
    assert failed_idx != -1
    neighborhood = body[max(0, failed_idx - 300):failed_idx + 400]
    assert 'should_continue' in neighborhood and 'False' in neighborhood, \
        "P0-7: PI-review exhaustion path must set should_continue=False"


def test_bioinfo_plan_generation_failure_sets_should_continue_false():
    """Static check: the bioinfo plan generation failure branch in
    planning_dialogue must also set should_continue=False (P0-7 sibling)."""
    body = inspect.getsource(dialogue.run_planning_dialogue_rounds)
    # Locate the bioinfo plan generation failure branch.
    marker = body.find("Bioinformatics plan generation failed")
    assert marker != -1, "could not locate bioinfo plan generation failure branch"
    failed_idx = body.find('"status": "failed"', marker)
    assert failed_idx != -1
    neighborhood = body[max(0, failed_idx - 300):failed_idx + 400]
    assert 'should_continue' in neighborhood and 'False' in neighborhood, \
        "P0-7: bioinfo plan generation exhaustion path must set should_continue=False"
