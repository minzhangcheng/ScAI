from backend.models.schemas import AnalysisPlan, PlanStep
from backend.workflow.planning import plans_materially_equivalent


def _plan(*, title="Normalize and cluster", method="scanpy", extra_step=False):
    steps = [
        PlanStep(
            id=1,
            title=title,
            method=method,
            input="adata",
            expected_output="clusters",
            rationale="characterize structure",
        )
    ]
    if extra_step:
        steps.append(PlanStep(id=2, title="Differential analysis", method="rank genes"))
    return AnalysisPlan(
        phase="exploration",
        stage="full",
        overview="Run a domain-neutral exploratory analysis",
        steps=steps,
        key_decision_points=["review cluster stability"],
    )


def test_identical_plan_is_stagnant():
    plan = _plan()
    assert plans_materially_equivalent(plan, plan.model_copy(deep=True))


def test_added_step_is_material_change():
    assert not plans_materially_equivalent(_plan(), _plan(extra_step=True))


def test_changed_method_is_material_change():
    assert not plans_materially_equivalent(_plan(), _plan(method="alternative pipeline"))
