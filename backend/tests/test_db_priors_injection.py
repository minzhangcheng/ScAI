import json
import pytest
from backend.utils.sandbox import CodeExecutionResult
from backend.agents.bioinformatics_agent import BioinformaticsAgent


class _FakeSandbox:
    """模拟真实 sandbox.execute 返回 CodeExecutionResult（非 dict）。"""
    def __init__(self, stdout, *, success=True, error=None):
        self._stdout = stdout
        self._success = success
        self._error = error
        self.last_code = None

    async def execute(self, code, **kw):
        self.last_code = code
        return CodeExecutionResult(
            success=self._success, output=self._stdout, error=self._error)


def _agent(sandbox):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox
    a.session_id = "s1"
    return a


@pytest.mark.asyncio
async def test_compute_priors_parses_output(tmp_path, monkeypatch):
    priors = {"0": [{"cell_type": "pDC", "p_adj": 1e-4, "overlap": ["IRF8", "SIGLECH"],
                     "source": "PanglaoDB", "universe_source": "adata.var_names", "strong": True}]}
    sandbox = _FakeSandbox(json.dumps(priors))
    agent = _agent(sandbox)
    monkeypatch.setattr("backend.agents.bioinformatics_agent.marker_priors_write",
                        lambda organism, ws: tmp_path / "tables" / "marker_priors_input.tsv")
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "marker_priors_input.tsv").write_text("cell_type\tgene_symbol\tsource\n")
    out = await agent._compute_database_priors(
        "adata", "leiden", "rank_genes_groups", "mouse", tmp_path,
        cluster_markers={"0": ["Irf8", "Siglech"]})
    assert out.status == "used"
    assert out.priors == priors
    assert "compute_ora_priors(adata" in sandbox.last_code  # f-string 注入了 adata_var
    assert "_provided_cluster_markers" in sandbox.last_code
    assert "Irf8" in sandbox.last_code and "Siglech" in sandbox.last_code


@pytest.mark.asyncio
async def test_compute_priors_empty_when_species_unsupported(tmp_path, monkeypatch):
    agent = _agent(_FakeSandbox(""))
    monkeypatch.setattr("backend.agents.bioinformatics_agent.marker_priors_write",
                        lambda organism, ws: None)  # 不支持物种
    out = await agent._compute_database_priors("adata", "leiden", "rank_genes_groups", "rat", tmp_path)
    assert out.status == "skipped"
    assert out.priors == {}
    assert "rat" in out.detail


@pytest.mark.asyncio
async def test_compute_priors_empty_on_bad_output(tmp_path, monkeypatch):
    agent = _agent(_FakeSandbox("not-json-traceback"))
    monkeypatch.setattr("backend.agents.bioinformatics_agent.marker_priors_write",
                        lambda organism, ws: tmp_path / "x.tsv")
    (tmp_path / "x.tsv").write_text("cell_type\tgene_symbol\tsource\n")
    out = await agent._compute_database_priors("adata", "leiden", "rank_genes_groups", "mouse", tmp_path)
    assert out.status == "failed"
    assert out.priors == {}
    assert "no JSON" in out.detail


@pytest.mark.asyncio
async def test_compute_priors_surfaces_sandbox_failure(tmp_path, monkeypatch):
    sandbox = _FakeSandbox("", success=False, error="rank key missing")
    agent = _agent(sandbox)
    monkeypatch.setattr(
        "backend.agents.bioinformatics_agent.marker_priors_write",
        lambda organism, ws: tmp_path / "x.tsv")
    (tmp_path / "x.tsv").write_text("cell_type\tgene_symbol\tsource\n")

    out = await agent._compute_database_priors(
        "adata", "leiden", "rank_genes_groups", "mouse", tmp_path,
        cluster_markers={"0": ["Irf8", "Siglech"]})

    assert out.status == "failed"
    assert out.priors == {}
    assert "rank key missing" in out.detail
    assert "failed" in out.summary(1)
    assert "failed" in out.realtime_message(1)


@pytest.mark.asyncio
async def test_compute_priors_distinguishes_successful_empty_result(tmp_path, monkeypatch):
    agent = _agent(_FakeSandbox("{}"))
    monkeypatch.setattr(
        "backend.agents.bioinformatics_agent.marker_priors_write",
        lambda organism, ws: tmp_path / "x.tsv")
    (tmp_path / "x.tsv").write_text("cell_type\tgene_symbol\tsource\n")

    out = await agent._compute_database_priors(
        "adata", "leiden", "rank_genes_groups", "mouse", tmp_path,
        cluster_markers={"0": ["Irf8"]})

    assert out.status == "empty"
    assert out.priors == {}
    assert out.summary(1).startswith("empty:")
    assert "no eligible matches" in out.realtime_message(1)


def test_analysis_step_has_priors_summary_field():
    from backend.models.schemas import AnalysisStep
    step = AnalysisStep(step_number=1, step_type="annotate_clusters")
    assert hasattr(step, "annotation_database_priors_summary")
    step.annotation_database_priors_summary = "3/24 clusters got DB priors"
    assert step.annotation_database_priors_summary.startswith("3/24")


def test_database_prior_result_statuses_are_explicit_for_log_and_realtime():
    from backend.agents.bioinformatics_agent import DatabasePriorResult

    priors = {"0": [{
        "cell_type": "pDC", "p_adj": 1e-4, "strong": True,
        "overlap": ["IRF8", "SIGLECH"],
    }]}
    used = DatabasePriorResult(
        "used", priors, marker_source="provided_cluster_markers")
    skipped = DatabasePriorResult("skipped", {}, "user chose literature-only annotation")

    assert used.summary(2).startswith("used: 1/2 clusters got DB priors")
    assert "provided_cluster_markers" in used.summary(2)
    assert used.realtime_message(2).startswith("Marker database priors used:")
    assert skipped.summary(2).startswith("skipped:")
    assert "literature-only" in skipped.realtime_message(2)


def test_summarize_priors_counts_strong():
    from backend.services.marker_priors import summarize_database_priors
    priors = {"8": [{"cell_type": "pDC", "p_adj": 1e-4, "strong": True, "overlap": ["A", "B"]}],
              "5": [{"cell_type": "T cell", "p_adj": 0.2, "strong": False, "overlap": ["C", "D"]}]}
    s = summarize_database_priors(priors, total_clusters=24)
    assert "2/24" in s and "strong" in s and "pDC" in s
    assert summarize_database_priors({}, 24) is None


def test_callback_signature_has_database_priors():
    from backend.workflow.annotation import AnnotationMixin
    import inspect as _i
    sig = _i.signature(AnnotationMixin._annotate_clusters_callback)
    assert "database_priors" in sig.parameters
