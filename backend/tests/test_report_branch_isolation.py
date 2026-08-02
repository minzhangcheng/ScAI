from backend.services.report.paths import resolve_report_output_dir, resolve_report_target_lineage


def test_main_report_output_uses_reports_root(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.report.paths.get_session_reports", lambda session_id: tmp_path / session_id / "reports")
    assert resolve_report_output_dir("s", None) == tmp_path / "s" / "reports"


def test_branch_report_output_uses_safe_branch_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.report.paths.get_session_reports", lambda session_id: tmp_path / session_id / "reports")
    assert resolve_report_output_dir("s", "branch/A") == tmp_path / "s" / "reports" / "branches" / "branch_A"


def test_report_target_lineage_does_not_depend_on_current_session_branch():
    assert resolve_report_target_lineage(branch_id=None) == "main"
    assert resolve_report_target_lineage(branch_id="branch-a") == "branch-a"


def test_main_regenerate_target_ignores_current_restore_branch():
    current_session_state = {"restore_branch_id": "branch-a"}
    branch_id_from_request = None
    assert current_session_state["restore_branch_id"] == "branch-a"
    assert resolve_report_target_lineage(branch_id_from_request) == "main"


import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.models.schemas import (
    AnalysisPhase,
    AnalysisState,
    AnalysisStep,
    FigureReviewStatus,
    PublicationFigure,
    RoundResult,
    TaskStatus,
)
from backend.tests.split_facade import setattr_facade
from backend.workflow.reporting import ReportingMixin


def _mixed_lineage_state() -> AnalysisState:
    return AnalysisState(
        id="s",
        user_id="u",
        status=TaskStatus.RUNNING,
        analysis_phase=AnalysisPhase.INNOVATION,
        restore_branch_id="branch-a",
        analysis_log=[AnalysisStep(step_number=1, step_type="execute_code")],
        publication_figures=[
            PublicationFigure(
                id="main-figure",
                name="main_panel",
                description="main",
                figure_type="heatmap",
                png_path="/tmp/main.jpg",
                pdf_path="/tmp/main.pdf",
                lineage_id="main",
            ),
            PublicationFigure(
                id="branch-figure",
                name="branch_panel",
                description="branch",
                figure_type="heatmap",
                png_path="/tmp/branch.jpg",
                pdf_path="/tmp/branch.pdf",
                lineage_id="branch-a",
            ),
        ],
        round_results=[
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction A",
                pi_summary="main summary",
                bioinformatics_findings=["main finding"],
                evidence_status="substantive",
                lineage_id="main",
            ),
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction A",
                pi_summary="branch summary",
                bioinformatics_findings=["branch finding"],
                evidence_status="substantive",
                lineage_id="branch-a",
                branch_id="branch-a",
            ),
        ],
    )


def test_build_lineage_report_state_filters_round_results_for_supervisor_path():
    from backend.services.report.lineage import build_lineage_report_state

    state = _mixed_lineage_state()

    branch_state = build_lineage_report_state(state, "branch-a")
    main_state = build_lineage_report_state(state, "main")

    assert [r.pi_summary for r in branch_state.round_results] == ["branch summary"]
    assert [r.pi_summary for r in main_state.round_results] == ["main summary"]
    assert [f.name for f in branch_state.publication_figures] == ["branch_panel"]
    assert [f.name for f in main_state.publication_figures] == ["main_panel"]
    assert [r.pi_summary for r in state.round_results] == ["main summary", "branch summary"]


def test_branch_report_rejects_unowned_log_and_workspace_figures(tmp_path):
    from backend.agents.supervisor.report_html import ReportHtmlMixin
    from backend.services.report.lineage import build_lineage_report_state
    from backend.tests.figure_image_helpers import write_valid_image

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    branch_path = figures_dir / "branch_owned.png"
    main_path = figures_dir / "main_only.png"
    orphan_path = figures_dir / "legacy_unowned.png"
    scan_only_path = figures_dir / "scan_only_legacy.png"
    deep_unlogged_path = tmp_path / "a" / "b" / "c" / "d" / "deep_unlogged.png"
    deep_unlogged_path.parent.mkdir(parents=True)
    write_valid_image(branch_path, min_bytes=2500)
    write_valid_image(main_path, min_bytes=2500)
    write_valid_image(orphan_path, min_bytes=2500)
    write_valid_image(scan_only_path, min_bytes=2500)
    write_valid_image(deep_unlogged_path, min_bytes=2500)

    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_log=[AnalysisStep(
            step_number=1,
            step_type="execute_code",
            saved_figure_filenames=[
                branch_path.name,
                main_path.name,
                orphan_path.name,
            ],
        )],
        publication_figures=[
            PublicationFigure(
                id="main",
                name="main_only",
                description="main lineage",
                figure_type="heatmap",
                png_path=str(main_path),
                pdf_path=str(main_path.with_suffix(".pdf")),
                lineage_id="main",
                review_status=FigureReviewStatus.APPROVED,
            ),
            PublicationFigure(
                id="branch",
                name="branch_owned",
                description="branch lineage",
                figure_type="heatmap",
                png_path=str(branch_path),
                pdf_path=str(branch_path.with_suffix(".pdf")),
                lineage_id="branch-a",
                review_status=FigureReviewStatus.APPROVED,
            ),
        ],
    )

    branch_state = build_lineage_report_state(
        state, "branch-a", workspace_path=tmp_path
    )
    # Exercise the full Supervisor collector with the original mixed log. Its
    # direct saved-filename tier and its recursive workspace tier must both obey
    # the branch-local exclusion set carried by the report snapshot.
    figure_map = ReportHtmlMixin._collect_figure_paths(
        object(),
        [{"saved_figure_filenames": [
            *state.analysis_log[0].saved_figure_filenames,
            deep_unlogged_path.name,
        ]}],
        tmp_path,
        state=branch_state,
    )

    assert set(figure_map) == {branch_path.name}
    assert branch_state.analysis_log[0].saved_figure_filenames == [branch_path.name]
    assert {main_path.stem, orphan_path.stem}.issubset(
        set(branch_state.invalidated_artifact_stems)
    )
    # This depth exceeds bounded workspace discovery and the filename was not in
    # AnalysisState.analysis_log. Only the collector's strict lineage rule can
    # stop the Supervisor's independent record tier from finding it via rglob.
    assert deep_unlogged_path.stem not in branch_state.invalidated_artifact_stems

    # Exercise the recursive discovery guard without precomputing workspace
    # invalidations: an unregistered scan-only image is still not branch-owned.
    branch_without_scan = build_lineage_report_state(state, "branch-a")
    scan_map = ReportHtmlMixin._collect_figure_paths(
        object(), [], tmp_path, state=branch_without_scan
    )
    assert set(scan_map) == {branch_path.name}
    # Main keeps the pre-lineage legacy fallback for persisted sessions.
    main_state = build_lineage_report_state(state, "main", workspace_path=tmp_path)
    assert main_state.analysis_log[0].saved_figure_filenames == [
        branch_path.name,
        main_path.name,
        orphan_path.name,
    ]


def test_inherited_publication_figures_are_cloned_without_destroying_source(tmp_path):
    from backend.services.report.lineage import clone_publication_figures_for_branch

    original = _archivable_figure(
        tmp_path,
        "fig-main",
        name="shared_result",
        logical_id="logical-shared-result",
    )

    cloned = clone_publication_figures_for_branch(
        [original],
        branch_id="branch-a",
        source_lineage_id="main",
        workspace_path=tmp_path,
    )

    assert [(item.lineage_id, item.name) for item in cloned] == [
        ("main", "shared_result"),
        ("branch-a", "shared_result"),
    ]
    assert cloned[0].id == "fig-main"
    assert cloned[1].id != cloned[0].id


def test_branch_clone_archives_full_pair_before_same_name_redraw(tmp_path):
    import hashlib
    from pathlib import Path

    from backend.services.report.artifacts import (
        resolve_archival_pdf,
        resolve_archival_render,
    )
    from backend.services.report.lineage import clone_publication_figures_for_branch
    from backend.tests.figure_image_helpers import valid_pdf_bytes, write_valid_image

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    mutable_jpg = figures_dir / "shared_result.jpg"
    mutable_pdf = figures_dir / "shared_result.pdf"
    original_jpg = write_valid_image(
        mutable_jpg, color=(30, 80, 120), min_bytes=3000
    )
    original_pdf = valid_pdf_bytes(color=(30, 80, 120))
    mutable_pdf.write_bytes(original_pdf)
    original = PublicationFigure(
        id="fig-main",
        name="shared_result",
        description="shared ancestor",
        figure_type="heatmap",
        png_path=str(mutable_jpg),
        pdf_path=str(mutable_pdf),
        render_sha256=hashlib.sha256(original_jpg).hexdigest(),
        pdf_sha256=hashlib.sha256(original_pdf).hexdigest(),
        lineage_id="main",
        review_status=FigureReviewStatus.APPROVED,
    )

    cloned = clone_publication_figures_for_branch(
        [original],
        branch_id="branch-a",
        source_lineage_id="main",
        workspace_path=tmp_path,
    )

    source, inherited = cloned
    assert source.png_path == inherited.png_path
    assert source.pdf_path == inherited.pdf_path
    assert Path(source.png_path).is_relative_to(figures_dir / ".lineage")
    assert Path(source.pdf_path).is_relative_to(figures_dir / ".lineage")
    assert Path(source.png_path).read_bytes() == original_jpg
    assert Path(source.pdf_path).read_bytes() == original_pdf

    cloned = clone_publication_figures_for_branch(
        cloned,
        branch_id="branch-a",
        source_lineage_id="main",
        workspace_path=tmp_path,
    )
    assert len(cloned) == 2
    source, inherited = cloned
    assert inherited.png_path == source.png_path
    assert inherited.pdf_path == source.pdf_path

    # The restored branch is free to publish the same logical filename. Neither
    # inherited record may fall back to the mutable top-level pair afterward.
    write_valid_image(mutable_jpg, color=(150, 20, 20), min_bytes=3000)
    mutable_pdf.write_bytes(valid_pdf_bytes(color=(150, 20, 20)))
    for record in (source, inherited):
        assert resolve_archival_render(record, tmp_path).read_bytes() == original_jpg
        assert resolve_archival_pdf(record, tmp_path).read_bytes() == original_pdf


def _archivable_figure(
    workspace, record_id, *, name=None, lineage="main", logical_id=None, color=(30, 60, 90),
):
    import hashlib

    from backend.models.schemas import FigureReviewStatus, PublicationFigure
    from backend.tests.figure_image_helpers import valid_pdf_bytes, write_valid_image

    figure_name = name or record_id
    jpg = workspace / "figures" / f"{figure_name}.jpg"
    pdf = jpg.with_suffix(".pdf")
    jpg.parent.mkdir(parents=True, exist_ok=True)
    image_payload = write_valid_image(jpg, color=color, min_bytes=2200)
    pdf_payload = valid_pdf_bytes(color=color)
    pdf.write_bytes(pdf_payload)
    return PublicationFigure(
        id=record_id,
        name=figure_name,
        description=figure_name,
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        render_sha256=hashlib.sha256(image_payload).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf_payload).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        lineage_id=lineage,
        logical_figure_id=logical_id or f"logical-{record_id}",
    )


def test_branch_clone_requires_a_complete_frozen_render_pdf_pair(tmp_path):
    from backend.services.report.lineage import clone_publication_figures_for_branch

    source = _archivable_figure(tmp_path, "source")
    Path(source.pdf_path).unlink()
    original_paths = (source.png_path, source.pdf_path)

    with pytest.raises(RuntimeError):
        clone_publication_figures_for_branch(
            [source], branch_id="branch-a", workspace_path=tmp_path
        )

    assert (source.png_path, source.pdf_path) == original_paths


def test_lineage_archive_rejects_preexisting_symlink_object(tmp_path):
    import hashlib

    from backend.services.report.lineage import clone_publication_figures_for_branch

    source = _archivable_figure(tmp_path, "source")
    pair_digest = hashlib.sha256(
        f"{source.render_sha256}:{source.pdf_sha256}".encode("ascii")
    ).hexdigest()
    pair_dir = tmp_path / "figures" / ".lineage" / pair_digest
    pair_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside_payload = Path(source.png_path).read_bytes()
    outside.write_bytes(outside_payload)
    (pair_dir / "render.jpg").symlink_to(outside)
    original_paths = (source.png_path, source.pdf_path)

    with pytest.raises(RuntimeError):
        clone_publication_figures_for_branch(
            [source], branch_id="branch-a", workspace_path=tmp_path
        )

    assert outside.read_bytes() == outside_payload
    assert (source.png_path, source.pdf_path) == original_paths


def test_lineage_archive_rejects_symlinked_root_without_external_writes(tmp_path):
    from backend.services.report.lineage import clone_publication_figures_for_branch

    source = _archivable_figure(tmp_path, "source")
    outside = tmp_path / "outside-lineage"
    outside.mkdir()
    (tmp_path / "figures" / ".lineage").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(RuntimeError):
        clone_publication_figures_for_branch(
            [source], branch_id="branch-a", workspace_path=tmp_path
        )

    assert list(outside.iterdir()) == []


def test_branch_clone_id_collision_rejects_unrelated_target_record(tmp_path):
    from backend.services.report.lineage import clone_publication_figures_for_branch

    source = _archivable_figure(
        tmp_path, "source", logical_id="logical-source"
    )
    collider = _archivable_figure(
        tmp_path,
        "source:branch-a",
        name="unrelated",
        lineage="branch-a",
        logical_id="logical-other",
    )
    collider_before = collider.model_dump()

    with pytest.raises(RuntimeError):
        clone_publication_figures_for_branch(
            [source, collider], branch_id="branch-a", workspace_path=tmp_path
        )

    assert collider.model_dump() == collider_before


def test_later_lineage_collision_does_not_mutate_an_earlier_source(tmp_path):
    import hashlib

    from backend.services.report.lineage import clone_publication_figures_for_branch

    first = _archivable_figure(tmp_path, "first", color=(20, 40, 60))
    second = _archivable_figure(tmp_path, "second", color=(200, 40, 20))
    first_before = first.model_dump()
    second_before = second.model_dump()
    pair_digest = hashlib.sha256(
        f"{second.render_sha256}:{second.pdf_sha256}".encode("ascii")
    ).hexdigest()
    collision = (
        tmp_path / "figures" / ".lineage" / pair_digest / "render.jpg"
    )
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"unrelated archive object")

    with pytest.raises(RuntimeError):
        clone_publication_figures_for_branch(
            [first, second], branch_id="branch-a", workspace_path=tmp_path
        )

    assert first.model_dump() == first_before
    assert second.model_dump() == second_before


def test_second_source_archive_failure_leaves_all_records_unchanged(
    tmp_path, monkeypatch,
):
    from backend.services.report import lineage

    first = _archivable_figure(tmp_path, "first", color=(20, 40, 60))
    second = _archivable_figure(tmp_path, "second", color=(200, 40, 20))
    first_before = first.model_dump()
    second_before = second.model_dump()
    real_archive = lineage._archive_inherited_figure
    calls = 0

    def fail_second(figure, workspace_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second source archive failed")
        return real_archive(figure, workspace_path)

    monkeypatch.setattr(lineage, "_archive_inherited_figure", fail_second)
    with pytest.raises(OSError):
        lineage.clone_publication_figures_for_branch(
            [first, second], branch_id="branch-a", workspace_path=tmp_path
        )

    assert first.model_dump() == first_before
    assert second.model_dump() == second_before


def test_branch_invalidation_does_not_expire_source_lineage_figure():
    from backend.services.report.artifacts import invalidate_current_artifacts

    figures = [
        PublicationFigure(
            id=f"fig-{lineage}", name="shared", description=lineage,
            figure_type="heatmap", png_path="", pdf_path="",
            lineage_id=lineage, round_number=2,
        )
        for lineage in ("main", "branch-a")
    ]

    invalidated = invalidate_current_artifacts(
        figures,
        round_number=2,
        reason="branch representation changed",
        lineage_id="branch-a",
    )

    assert invalidated == 1
    assert figures[0].is_current is True
    assert figures[1].is_current is False


class _SettingsProvider:
    def get_llm_client(self, agent_name: str):
        return object()

    def get_model(self, agent_name: str):
        return f"{agent_name}-model"


class _Supervisor:
    def __init__(self):
        self.seen_state = None

    async def generate_detailed_report(self, **kwargs):
        self.seen_state = kwargs["analysis_state"]
        return {"id": "r1", "title": "Report", "sections": [], "figures": [], "tables": []}


class _ReportingWorkflow(ReportingMixin):
    def __init__(self):
        self.session_id = "s"
        self.supervisor_agent = _Supervisor()
        self._settings_provider = _SettingsProvider()
        self._vision_settings = {}
        self.sent_messages = []
        self.synced = []
        self.snapshots = []

    def ensure_agent_ready(self, agent_name: str):
        assert agent_name == "supervisor"

    async def _send_message(self, message_type, content, agent=None):
        self.sent_messages.append((message_type, content, agent))

    async def _send_state_sync(self, analysis_state, change_type):
        self.synced.append((analysis_state, change_type))

    async def _create_auto_snapshot(self, analysis_state, phase):
        self.snapshots.append((analysis_state, phase))

    def _set_status(self, analysis_state, status):
        analysis_state.status = status


@pytest.mark.asyncio
async def test_workflow_supervisor_report_receives_filtered_branch_state(tmp_path, monkeypatch):
    from backend.workflow import reporting as reporting_module

    state = _mixed_lineage_state()
    workflow = _ReportingWorkflow()

    setattr_facade(monkeypatch, "backend.workflow.reporting", "get_session_workspace", lambda session_id: tmp_path / "workspace")
    setattr_facade(
        monkeypatch,
        "backend.workflow.reporting",
        "resolve_report_output_dir",
        lambda session_id, branch_id: tmp_path / "reports" / (branch_id or "main"),
    )
    # Isolate post-report snapshot capture from real session storage paths.
    async def _fake_capture(analysis_state, report_generation_id):
        return SimpleNamespace(snapshot_id="snap-test", failure=None)

    monkeypatch.setattr(workflow, "_capture_post_report_snapshot", _fake_capture)

    result = await workflow._generate_detailed_report({"analysis_state": state})

    seen = workflow.supervisor_agent.seen_state
    assert [r.pi_summary for r in seen.round_results] == ["branch summary"]
    assert seen is not state
    assert result["analysis_state"] is state
    assert state.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_branch_report_figure_route_serves_branch_asset(tmp_path, monkeypatch):
    from backend.routes import reports

    reports_root = tmp_path / "s" / "reports"
    main_figures = reports_root / "figures"
    branch_figures = reports_root / "branches" / "branch-a" / "figures"
    main_figures.mkdir(parents=True)
    branch_figures.mkdir(parents=True)
    (main_figures / "plot.png").write_bytes(b"main")
    (branch_figures / "plot.png").write_bytes(b"branch")

    monkeypatch.setattr("backend.services.report.paths.get_session_reports", lambda session_id: reports_root)

    response = await reports.get_report_figure(
        session_id="s",
        filename="plot.png",
        branch_id="branch-a",
        state=AnalysisState(id="s", user_id="u"),
    )

    assert str(response.path) == str(branch_figures / "plot.png")


@pytest.mark.asyncio
async def test_branch_report_table_route_serves_branch_asset(tmp_path, monkeypatch):
    from backend.routes import reports

    reports_root = tmp_path / "s" / "reports"
    main_tables = reports_root / "tables"
    branch_tables = reports_root / "branches" / "branch-a" / "tables"
    main_tables.mkdir(parents=True)
    branch_tables.mkdir(parents=True)
    (main_tables / "table.csv").write_text("main\n", encoding="utf-8")
    (branch_tables / "table.csv").write_text("branch\n", encoding="utf-8")

    monkeypatch.setattr("backend.services.report.paths.get_session_reports", lambda session_id: reports_root)

    response = await reports.get_report_table(
        session_id="s",
        filename="table.csv",
        branch_id="branch-a",
        state=AnalysisState(id="s", user_id="u"),
    )

    assert str(response.path) == str(branch_tables / "table.csv")


@pytest.mark.asyncio
async def test_branch_report_preview_urls_include_branch_id(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from backend.routes import reports

    reports_root = tmp_path / "s" / "reports"
    branch_dir = reports_root / "branches" / "branch-a"
    (branch_dir / "figures").mkdir(parents=True)
    (branch_dir / "tables").mkdir(parents=True)
    (branch_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (branch_dir / "report.json").write_text("{}", encoding="utf-8")
    (branch_dir / "figures" / "plot.png").write_bytes(b"branch")
    (branch_dir / "tables" / "table.csv").write_text("branch\n", encoding="utf-8")

    async def _fake_owner(session_id: str, user_id: str):
        return AnalysisState(id=session_id, user_id=user_id)

    monkeypatch.setattr("backend.services.report.paths.get_session_reports", lambda session_id: reports_root)
    # Bind on every module of the reports split, not just the facade: the handler
    # reads the name from the reports_parts module it happens to live in.
    setattr_facade(monkeypatch, "backend.routes.reports", "require_session_ownership", _fake_owner)

    payload = await reports.preview_report(
        session_id="s",
        request=SimpleNamespace(state=SimpleNamespace(user_id="u")),
        branch_id="branch-a",
    )

    assert payload["figures"][0]["png_url"].endswith("/api/session/s/report/figure/plot.png?branch_id=branch-a")
    assert payload["tables"][0]["csv_url"].endswith("/api/session/s/report/table/table.csv?branch_id=branch-a")
    assert payload["html_url"].endswith("/api/session/s/report/html?branch_id=branch-a")
    assert payload["download_url"].endswith("/api/session/s/report/download-full?branch_id=branch-a")


@pytest.mark.asyncio
async def test_detailed_report_cache_is_lineage_scoped(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from backend.routes import reports
    from backend.state import detailed_reports

    detailed_reports.clear()

    reports_root = tmp_path / "s" / "reports"
    main_dir = reports_root
    branch_dir = reports_root / "branches" / "branch-a"
    main_dir.mkdir(parents=True)
    branch_dir.mkdir(parents=True)
    (main_dir / "report.json").write_text('{"which": "main"}', encoding="utf-8")
    (branch_dir / "report.json").write_text('{"which": "branch"}', encoding="utf-8")

    async def _fake_owner(session_id: str, user_id: str):
        return AnalysisState(id=session_id, user_id=user_id)

    monkeypatch.setattr("backend.services.report.paths.get_session_reports", lambda session_id: reports_root)
    # Bound across the whole reports split; see the note above.
    setattr_facade(monkeypatch, "backend.routes.reports", "require_session_ownership", _fake_owner)

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    branch_payload = await reports.get_detailed_report("s", request, branch_id="branch-a")
    main_payload = await reports.get_detailed_report("s", request, branch_id=None)

    assert branch_payload["report"]["which"] == "branch"
    assert main_payload["report"]["which"] == "main"
    assert detailed_reports["s"]["which"] == "main"
    assert detailed_reports["s:branch-a"]["which"] == "branch"


def test_clear_detailed_report_cache_removes_main_and_branch_entries():
    from backend.routes.reports import clear_detailed_report_cache
    from backend.state import detailed_reports

    detailed_reports.clear()
    detailed_reports["s"] = {"which": "main"}
    detailed_reports["s:branch-a"] = {"which": "branch-a"}
    detailed_reports["s:branch-b"] = {"which": "branch-b"}
    detailed_reports["sibling"] = {"which": "other-session"}

    clear_detailed_report_cache("s")

    assert "s" not in detailed_reports
    assert "s:branch-a" not in detailed_reports
    assert "s:branch-b" not in detailed_reports
    assert detailed_reports["sibling"] == {"which": "other-session"}
