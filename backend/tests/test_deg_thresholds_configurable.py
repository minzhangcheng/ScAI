"""P2-11: Session-configurable DEG thresholds."""
from backend.models.schemas import AnalysisState
import backend.services.report.figures as fig_mod
import backend.agents.bioinformatics_agent as bio_mod
from backend.tests.split_facade import split_source


def test_analysis_state_has_deg_thresholds():
    """P2-11: AnalysisState must expose deg_thresholds with defaults."""
    s = AnalysisState(user_id="u1")
    assert hasattr(s, "deg_thresholds")
    assert isinstance(s.deg_thresholds, dict)
    assert s.deg_thresholds.get("logfc") == 1.0
    assert s.deg_thresholds.get("padj") == 0.05


def test_figures_reads_thresholds_from_state():
    """P2-11: figures.py must read deg_thresholds from analysis_state, not hardcode."""
    src = open(fig_mod.__file__).read()
    assert "deg_thresholds" in src, "P2-11: figures.py must reference deg_thresholds"


def test_bioinformatics_agent_reads_thresholds_from_state():
    src = split_source("backend.agents.bioinformatics_agent")
    assert "deg_thresholds" in src, "P2-11: bioinformatics_agent.py must reference deg_thresholds"
