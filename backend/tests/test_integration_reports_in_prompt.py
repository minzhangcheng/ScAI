import inspect
from backend.agents.pi_agent import PIAgent


def test_final_summary_prompt_includes_integration_reports():
    """generate_final_summary must feed integration_reports into the prompt
    so cross-direction integration insights appear in the final report."""
    src = inspect.getsource(PIAgent.generate_final_summary)
    assert "build_integration_block" in src, (
        "generate_final_summary must call build_integration_block"
    )
    assert "integration" in src.lower()


def test_integrated_report_prompt_includes_integration_reports():
    """generate_integrated_report must also feed integration_reports."""
    src = inspect.getsource(PIAgent.generate_integrated_report)
    assert "build_integration_block" in src, (
        "generate_integrated_report must call build_integration_block"
    )


def test_supervisor_report_prompt_includes_integration_reports():
    """SupervisorAgent._build_report_prompt must feed integration_reports."""
    from backend.agents.supervisor import report_prompt
    src = inspect.getsource(report_prompt._build_report_prompt)
    assert "build_integration_block" in src, (
        "_build_report_prompt must call build_integration_block"
    )


def test_supervisor_report_prompt_consolidates_evidence_limitations():
    from backend.agents.supervisor import report_prompt
    src = inspect.getsource(report_prompt._build_report_prompt)

    assert "build_quality_disclosures" in src
    assert "**Limitations** subsection" in src
    assert "Rounds marked partial/unverified" not in src
