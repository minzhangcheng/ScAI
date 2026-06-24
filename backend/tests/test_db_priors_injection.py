import json
import pytest
from backend.utils.sandbox import CodeExecutionResult
from backend.agents.bioinformatics_agent import BioinformaticsAgent


class _FakeSandbox:
    """模拟真实 sandbox.execute 返回 CodeExecutionResult（非 dict）。"""
    def __init__(self, stdout): self._stdout = stdout; self.last_code = None
    async def execute(self, code, **kw):
        self.last_code = code
        return CodeExecutionResult(success=True, output=self._stdout)


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
    out = await agent._compute_database_priors("adata", "leiden", "rank_genes_groups", "mouse", tmp_path)
    assert out == priors
    assert "compute_ora_priors(adata" in sandbox.last_code  # f-string 注入了 adata_var


@pytest.mark.asyncio
async def test_compute_priors_empty_when_species_unsupported(tmp_path, monkeypatch):
    agent = _agent(_FakeSandbox(""))
    monkeypatch.setattr("backend.agents.bioinformatics_agent.marker_priors_write",
                        lambda organism, ws: None)  # 不支持物种
    out = await agent._compute_database_priors("adata", "leiden", "rank_genes_groups", "rat", tmp_path)
    assert out == {}


@pytest.mark.asyncio
async def test_compute_priors_empty_on_bad_output(tmp_path, monkeypatch):
    agent = _agent(_FakeSandbox("not-json-traceback"))
    monkeypatch.setattr("backend.agents.bioinformatics_agent.marker_priors_write",
                        lambda organism, ws: tmp_path / "x.tsv")
    (tmp_path / "x.tsv").write_text("cell_type\tgene_symbol\tsource\n")
    out = await agent._compute_database_priors("adata", "leiden", "rank_genes_groups", "mouse", tmp_path)
    assert out == {}  # 解析失败静默降级


def test_analysis_step_has_priors_summary_field():
    from backend.models.schemas import AnalysisStep
    step = AnalysisStep(step_number=1, step_type="annotate_clusters")
    assert hasattr(step, "annotation_database_priors_summary")
    step.annotation_database_priors_summary = "3/24 clusters got DB priors"
    assert step.annotation_database_priors_summary.startswith("3/24")


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
