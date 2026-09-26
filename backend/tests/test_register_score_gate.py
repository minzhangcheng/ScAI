"""register_score records provenance limitations and real panel size.

Missing provenance remains visible to PI review but cannot reject registration
or decide scientific acceptance.
"""
import inspect

from backend.agents.bioinformatics import executor_registration


def test_handler_records_invalid_provenance_without_rejecting_registration():
    block = inspect.getsource(executor_registration.handle_register_score)

    assert "parse_provenance" in block, "handler does not validate provenance"
    assert "prov_error" in block
    before_registration = block[:block.index("context.has_register_score = True")]
    assert "return ToolAction.CONTINUE" not in before_registration
    assert "register_score rejected" not in block
    assert "register-score-provenance" in block
    assert 'severity="advise"' in block
    assert "PI must decide" in block


def test_power_is_assessed_on_the_frozen_panel_not_the_reported_excerpt():
    """gene_list is documented as an excerpt; the frozen plan is the panel.

    Sizing power from gene_list understates every panel longer than the excerpt
    cap, which is exactly the case where the underpowered flag matters. The
    handler must take its counts and its hash from plan.panel_genes when a plan
    is cited, and use gene_list only for the drift comparison.
    """
    block = inspect.getsource(executor_registration.handle_register_score)
    assert "panel_members = list(plan.panel_genes)" in block
    assert "assess_panel_power(len(panel_members)" in block
    assert "panel_hash(panel_members)" in block
    assert "diff_panels(plan.panel_genes, gene_list)" in block, (
        "drift must still compare the frozen panel against what was reported"
    )
