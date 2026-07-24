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
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.models.schemas import AnalysisPhase, AnalysisState, AnalysisStep, RoundResult, TaskStatus
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
    assert [r.pi_summary for r in state.round_results] == ["main summary", "branch summary"]


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
