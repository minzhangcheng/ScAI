"""_summarize_cluster_markers: full pass-through, 2-tuple return, no [:15] truncation."""
from backend.agents.pi_agent import PIAgent


def _markers(n):
    return {str(i): [f"Gene{i}_{j}" for j in range(10)] for i in range(n)}


def test_returns_two_tuple():
    out = PIAgent._summarize_cluster_markers(_markers(3))
    assert len(out) == 2
    n_clusters, markers_summary = out
    assert n_clusters == 3
    assert isinstance(markers_summary, str)


def test_includes_all_clusters_beyond_15():
    n_clusters, markers_summary = PIAgent._summarize_cluster_markers(_markers(22))
    assert n_clusters == 22
    for i in range(22):
        assert f"Cluster {i}:" in markers_summary


def test_caps_genes_per_cluster_at_8():
    n_clusters, markers_summary = PIAgent._summarize_cluster_markers(_markers(1))
    assert "Gene0_7" in markers_summary
    assert "Gene0_8" not in markers_summary
