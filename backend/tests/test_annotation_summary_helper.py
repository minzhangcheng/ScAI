from types import SimpleNamespace
from backend.agents.pi_agent import PIAgent


def _agent():
    return PIAgent.__new__(PIAgent)


def test_no_annotation_state():
    state = SimpleNamespace(annotation_state=None)
    assert _agent()._format_cluster_annotation_summary(state) == "No annotation data"


def test_formats_clusters():
    ca = SimpleNamespace(cluster_id="0", cell_type="T cell", confidence="high",
                         key_markers=[f"M{i}" for i in range(12)])
    state = SimpleNamespace(annotation_state=SimpleNamespace(cluster_annotations=[ca]))
    out = _agent()._format_cluster_annotation_summary(state)
    assert out == "- Cluster 0: T cell [confidence: high] markers: M0, M1, M2, M3, M4, M5, M6, M7, M8, M9"
