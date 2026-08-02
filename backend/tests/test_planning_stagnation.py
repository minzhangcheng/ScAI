from backend.models.schemas import AnalysisPlan, FigureRequirement, PlanStep
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


def test_changed_figure_contract_is_material_change():
    previous = _plan()
    current = _plan()
    current.steps[0].figure_requirements = [
        FigureRequirement(figure_type="heatmap", description="Result overview")
    ]
    assert not plans_materially_equivalent(previous, current)


def test_changed_text_only_exception_is_material_change():
    previous = _plan()
    current = _plan()
    current.steps[0].allow_text_only = True
    assert not plans_materially_equivalent(previous, current)


def test_changed_text_only_reason_is_material_change():
    previous = _plan()
    previous.steps[0].allow_text_only = True
    previous.steps[0].text_only_reason = "Narrative synthesis"
    current = previous.model_copy(deep=True)
    current.steps[0].text_only_reason = "Workflow integration checkpoint"
    assert not plans_materially_equivalent(previous, current)
