import inspect
import backend.workflow.graph as graph_mod
import backend.workflow.core as core_mod
from backend.workflow.integration import IntegrationMixin
from backend.constants import NODE_TO_AGENT, RESULT_KEYLESS_NODES


def test_core_composes_integration_mixin():
    assert issubclass(core_mod.SingleCellAnalysisWorkflow, IntegrationMixin)


def test_constants_registered():
    assert NODE_TO_AGENT.get("integration_synthesis") == "pi"
    assert "integration_synthesis" in RESULT_KEYLESS_NODES


def test_graph_registers_node_and_edges():
    src = inspect.getsource(graph_mod.GraphMixin._build_graph)
    assert 'add_node("integration_synthesis"' in src
    assert '"integrate": "integration_synthesis"' in src
    assert "_after_integration" in src
    assert '"gate": "wait_for_user_input"' in src
    assert '"stop": "generate_final_report"' in src
