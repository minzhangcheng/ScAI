"""Adoption of workspace renders into the plan figure contract.

Spec: docs/superpowers/specs/2026-08-27-figure-adoption-design.md

The executor saves result figures with plt.savefig inside execute_code. Those
files never become PublicationFigure records, and _matches reads
``figure.plan_step_id`` -- never prose -- so the contract reports them missing
however plainly they sit on disk. Measured on session 3d675631: 37 jpg files,
one generate_publication_figure call all round, 16 requirements at 0/1 eligible,
every one of them present in workspace/figures.

Adoption closes that by constructing the record. A PI call decides which
candidate answers which requirement; this module validates that decision and
writes the fields the contract actually reads.
"""
import pytest

from backend.models.schema_parts.enums import FigureReviewStatus
from backend.services.figure_adoption import (
    AdoptionCandidate,
    FigureBinding,
    adopt_bindings,
    build_adoption_candidates,
    validate_bindings,
)
from backend.services.figure_contract import RequiredFigure
from backend.services.report.artifacts import (
    LAYOUT_UNVERIFIED_KIND,
    archival_render_available,
    is_report_eligible,
    layout_was_measured,
)


# --------------------------------------------------------------- fixtures

def _render(path, *, pdf=True):
    """A real image big enough for image_file_sha256's floors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    rng = np.random.default_rng(0)
    ax.imshow(rng.random((64, 64)))
    fig.savefig(path, format="jpg")
    if pdf:
        fig.savefig(str(path).replace(".jpg", ".pdf"), format="pdf")
    plt.close(fig)
    return path


@pytest.fixture
def workspace(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    for stem in ("volcano_primary", "heatmap_primary",
                 "draft_render", "draft_render_final"):
        _render(figures / f"{stem}.jpg")
    return tmp_path


def _state(tmp_path, *, plan_id="plan-1", round_number=1):
    from backend.models.schemas import AnalysisState

    state = AnalysisState(session_id="s", user_id="u", research_purpose="t")
    state.current_round = round_number
    state.publication_figures = []

    class _Plan:
        id = plan_id
        steps = []

    state.current_plan = _Plan()
    return state


def _req(step, figure_type, key=None):
    return RequiredFigure(
        key=key or f"plan.{step}.{figure_type}",
        figure_type=figure_type,
        description=f"{figure_type} for step {step}",
        minimum_count=1,
        plan_step_id=step,
        plan_id="plan-1",
        require_plan_link=True,
    )


def _cand(workspace, stem):
    jpg = workspace / "figures" / f"{stem}.jpg"
    return AdoptionCandidate(
        name=stem,
        path=str(jpg),
        pdf_path=str(jpg.with_suffix(".pdf")),
        provenance=f"execute_code step N: produced {stem}",
    )


# ----------------------------------------------------- binding validation

def test_valid_binding_survives(workspace):
    reqs = [_req(3, "volcano")]
    cands = [_cand(workspace, "volcano_primary")]
    out = validate_bindings(
        [FigureBinding(3, "volcano", "volcano_primary", "cap")], reqs, cands)
    assert len(out) == 1
    assert out[0].plan_step_id == 3


def test_unknown_filename_is_dropped(workspace):
    out = validate_bindings(
        [FigureBinding(3, "volcano", "not_on_disk", "cap")],
        [_req(3, "volcano")], [_cand(workspace, "volcano_primary")])
    assert out == []


def test_type_mismatch_is_dropped(workspace):
    """The PI naming a heatmap for a volcano requirement is a misbinding, and a
    figure filed under the wrong step is worse than a reported gap."""
    out = validate_bindings(
        [FigureBinding(3, "heatmap", "volcano_primary", "cap")],
        [_req(3, "volcano")], [_cand(workspace, "volcano_primary")])
    assert out == []


def test_unknown_step_is_dropped(workspace):
    out = validate_bindings(
        [FigureBinding(99, "volcano", "volcano_primary", "cap")],
        [_req(3, "volcano")], [_cand(workspace, "volcano_primary")])
    assert out == []


def test_one_file_cannot_answer_two_requirements(workspace):
    reqs = [_req(3, "volcano"), _req(4, "volcano")]
    cands = [_cand(workspace, "volcano_primary")]
    out = validate_bindings([
        FigureBinding(3, "volcano", "volcano_primary", "cap"),
        FigureBinding(4, "volcano", "volcano_primary", "cap"),
    ], reqs, cands)
    assert len(out) == 1, "the duplicate must not silently satisfy both steps"


def test_type_alias_is_normalized(workspace):
    """normalize_figure_type is the comparison, not string equality. The alias
    table is the platform's (bar_chart -> barplot); this module must not invent
    aliases of its own, or a binding would validate here and fail in _matches."""
    out = validate_bindings(
        [FigureBinding(3, "Bar Chart", "heatmap_primary", "cap")],
        [_req(3, "barplot")], [_cand(workspace, "heatmap_primary")])
    assert len(out) == 1


def test_unaliased_type_variant_is_not_invented(workspace):
    """"Volcano Plot" has no entry in _TYPE_ALIASES. Accepting it here would
    admit a binding the contract's own _matches would then reject."""
    out = validate_bindings(
        [FigureBinding(3, "Volcano Plot", "volcano_primary", "cap")],
        [_req(3, "volcano")], [_cand(workspace, "volcano_primary")])
    assert out == []


# --------------------------------------------------------- record fields

def test_adopted_record_carries_the_fields_the_contract_reads(workspace, tmp_path):
    state = _state(tmp_path)
    adopt_bindings(
        state,
        [FigureBinding(3, "volcano", "volcano_primary", "A caption")],
        [_cand(workspace, "volcano_primary")],
    )
    assert len(state.publication_figures) == 1
    fig = state.publication_figures[0]
    assert fig.plan_step_id == 3
    assert fig.plan_id == "plan-1"
    assert fig.figure_type == "volcano"
    assert fig.purpose == "result"
    assert fig.diagnostic is False
    assert fig.producer_status == "published"
    assert fig.is_current is True
    assert fig.round_number == 1
    assert fig.description == "A caption"


def test_adopted_record_records_layout_as_unmeasured(workspace, tmp_path):
    """The in-kernel linter needs a live matplotlib figure. An empty list is what
    a measured-clean figure carries, so writing it here would claim a
    measurement that never happened."""
    state = _state(tmp_path)
    adopt_bindings(state, [FigureBinding(3, "volcano", "volcano_primary", "c")],
                   [_cand(workspace, "volcano_primary")])
    fig = state.publication_figures[0]
    assert fig.layout_violations, "must not be an empty (== measured clean) list"
    assert fig.layout_violations[0].get("kind") == LAYOUT_UNVERIFIED_KIND
    assert layout_was_measured(fig) is False


def test_adopted_record_carries_reservations_not_a_clean_approval(workspace, tmp_path):
    """is_report_eligible admits a non-diagnostic record only at APPROVED or
    ACCEPTED_WITH_RESERVATIONS, so PI inspection is the eligibility
    precondition, not a nicety. The reservation is literal and checkable: the
    layout was never measured."""
    from backend.services.figure_adoption import ADOPTION_RESERVATION

    state = _state(tmp_path)
    adopt_bindings(state, [FigureBinding(3, "volcano", "volcano_primary", "c")],
                   [_cand(workspace, "volcano_primary")])
    fig = state.publication_figures[0]
    assert fig.review_status == FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
    assert fig.pi_feedback == ADOPTION_RESERVATION
    assert "unverified" in fig.pi_feedback


def test_unreviewed_adoption_stays_out_of_the_contract(workspace, tmp_path):
    """A caller that did not show the images to the PI must pass PENDING, and
    the record must then be inert -- present for the gallery, invisible to the
    contract. This is the lever that keeps clause 5 honest."""
    state = _state(tmp_path)
    adopt_bindings(state, [FigureBinding(3, "volcano", "volcano_primary", "c")],
                   [_cand(workspace, "volcano_primary")],
                   review_status=FigureReviewStatus.PENDING)
    fig = state.publication_figures[0]
    assert is_report_eligible(fig) is False


def test_adoption_never_claims_to_be_a_platform_diagnostic(workspace, tmp_path):
    """Setting diagnostic=True would dodge the review check, but it is false
    twice: the platform did not render these, and that branch requires a
    measured layout an adopted record by construction lacks."""
    state = _state(tmp_path)
    adopt_bindings(state, [FigureBinding(3, "volcano", "volcano_primary", "c")],
                   [_cand(workspace, "volcano_primary")])
    assert state.publication_figures[0].diagnostic is False


def test_adopted_record_binds_to_its_bytes(workspace, tmp_path):
    state = _state(tmp_path)
    adopt_bindings(state, [FigureBinding(3, "volcano", "volcano_primary", "c")],
                   [_cand(workspace, "volcano_primary")])
    fig = state.publication_figures[0]
    assert fig.render_sha256, "render digest missing"
    assert fig.pdf_sha256, "pdf digest missing"
    assert fig.preview_sha256, "preview digest missing"


def test_adopted_record_is_report_eligible(workspace, tmp_path):
    state = _state(tmp_path)
    adopt_bindings(state, [FigureBinding(3, "volcano", "volcano_primary", "c")],
                   [_cand(workspace, "volcano_primary")])
    fig = state.publication_figures[0]
    assert is_report_eligible(fig) is True
    assert archival_render_available(fig) is True


# -------------------------------------------------------- contract closure

def test_adoption_closes_the_requirement(workspace, tmp_path):
    """End to end against the real audit: missing before, satisfied after."""
    from backend.services.figure_contract import audit_figure_contract

    state = _state(tmp_path)
    step = type("S", (), {
        "id": 3, "title": "t", "method": "m", "status": "pending",
        "figure_requirements": [
            type("R", (), {"figure_type": "volcano", "description": "d",
                           "minimum_count": 1, "platform_baseline": ""})(),
        ],
    })()
    state.current_plan.steps = [step]

    before = audit_figure_contract(state, plan=state.current_plan,
                                   planning_stage="stage2")
    assert not before.passed, "fixture must start with the requirement unmet"

    reqs = [m.requirement for m in before.missing]
    bindings = validate_bindings(
        [FigureBinding(3, "volcano", "volcano_primary", "c")],
        reqs, [_cand(workspace, "volcano_primary")])
    adopt_bindings(state, bindings, [_cand(workspace, "volcano_primary")])

    after = audit_figure_contract(state, plan=state.current_plan,
                                  planning_stage="stage2")
    assert after.passed, f"still missing: {after.summary()['missing']}"


# --------------------------------------------------------- best effort

@pytest.mark.parametrize("bindings", [[], None])
def test_no_bindings_leaves_state_untouched(workspace, tmp_path, bindings):
    state = _state(tmp_path)
    adopt_bindings(state, bindings, [_cand(workspace, "volcano_primary")])
    assert state.publication_figures == []


def test_missing_file_is_not_adopted(tmp_path):
    state = _state(tmp_path)
    ghost = AdoptionCandidate(name="ghost", path=str(tmp_path / "ghost.jpg"),
                              pdf_path=None, provenance="")
    adopt_bindings(state, [FigureBinding(3, "volcano", "ghost", "c")], [ghost])
    assert state.publication_figures == []


def test_adoption_is_idempotent(workspace, tmp_path):
    state = _state(tmp_path)
    binding = [FigureBinding(3, "volcano", "volcano_primary", "c")]
    cands = [_cand(workspace, "volcano_primary")]
    adopt_bindings(state, binding, cands)
    adopt_bindings(state, binding, cands)
    assert len(state.publication_figures) == 1


# ------------------------------------------------------ filename neutrality

def test_filenames_do_not_decide_precedence(workspace, tmp_path):
    """PublicationFigure.logical_figure_id: "Filenames are presentation details
    and must never be parsed for _v2/_final style suffixes". Measured session
    has four such pairs, and in one of them the _fixed member is the OLDER
    render. The PI picks; the code must not second-guess by name."""
    state = _state(tmp_path)
    cands = [_cand(workspace, "draft_render"),
             _cand(workspace, "draft_render_final")]
    out = validate_bindings(
        [FigureBinding(3, "volcano", "draft_render", "c")],
        [_req(3, "volcano")], cands)
    assert len(out) == 1
    adopt_bindings(state, out, cands)
    assert state.publication_figures[0].name == "draft_render", (
        "the PI chose draft_render; _final must not override it"
    )


def test_module_names_no_specific_biology():
    """CLAUDE.md: the platform stays domain-neutral."""
    import re

    import backend.services.figure_adoption as mod

    src = open(mod.__file__).read().lower()
    for term in ("ferroptos", "apoptos", "hepatocyte", "kupffer", "nrf2"):
        assert term not in src
    for word in ("liver", "mouse", "human", "cancer", "copper", "tumor"):
        assert not re.search(rf"\b{word}\b", src), f"{word!r} leaked into the module"


# ------------------------------------------------------ candidate assembly

def test_candidates_carry_execution_provenance(workspace, tmp_path, monkeypatch):
    """The PI needs to know which execution produced a file; that is the only
    signal that distinguishes a draft from the render that superseded it."""
    state = _state(tmp_path)
    state.analysis_log = [
        {"step_number": 7, "step_type": "execute_code",
         "code_description": "Produce the primary result view",
         "saved_figure_filenames": ["volcano_primary.jpg"]},
    ]
    cands = build_adoption_candidates(state, workspace)
    by_name = {c.name: c for c in cands}
    assert "volcano_primary" in by_name
    assert "Produce the primary result view" in by_name["volcano_primary"].provenance


# ------------------------------------------------------------- the wiring

def test_adoption_runs_before_the_workflow_audit():
    """Spec clause 1. The audit reads records and nothing else, so adoption that
    runs after it registers figures no requirement will ever consult -- the pass
    would be inert and its tests would still pass. Order is the whole point, and
    only the call site can carry that evidence."""
    import inspect

    import backend.workflow.agents as agents_mod

    src = inspect.getsource(agents_mod)
    adopt = src.find("await self._adopt_workspace_figures(")
    assert adopt != -1, "the adoption pass is not wired into the workflow"
    audit = src.find("figure_contract = audit_figure_contract(")
    assert audit != -1, "the workflow figure audit call site moved"
    assert adopt < audit, (
        "adoption must run before audit_figure_contract; after it, adopted "
        "records cannot satisfy any requirement"
    )


def test_bioinformatics_analysis_keeps_its_retry_decorator():
    """Inserting a helper directly above a decorated method silently rebinds the
    decorator to the helper. That happened while wiring this feature: three
    upstream-propagation tests failed because @with_retry("bioinformatics_analysis")
    had moved onto the new function."""
    import inspect
    import re

    import backend.workflow.agents as agents_mod

    src = inspect.getsource(agents_mod)
    assert re.search(
        r'@with_retry\("bioinformatics_analysis"\)\s*\n\s*async def _bioinformatics_analysis\b',
        src,
    ), "@with_retry no longer sits immediately above _bioinformatics_analysis"


def test_adoption_failure_cannot_break_the_node():
    """Spec clause 9: best effort. Every failure mode is inside one try/except
    that returns 0, so a discovery or PI fault leaves the audit exactly as it
    was rather than failing the analysis node."""
    import inspect

    from backend.workflow.agents import AgentNodesMixin

    body = inspect.getsource(AgentNodesMixin._adopt_workspace_figures)
    assert "except Exception" in body
    assert body.count("try:") >= 1
