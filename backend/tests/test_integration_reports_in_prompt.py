import inspect
from backend.agents.pi_agent import PIAgent


def test_final_summary_prompt_includes_integration_reports():
    """generate_final_summary must feed integration_reports into the prompt
    so cross-direction integration insights appear in the final report."""
    src = inspect.getsource(PIAgent.generate_final_summary)
    assert "integration_reports" in src, (
        "generate_final_summary must reference state.integration_reports"
    )
    assert "integration" in src.lower()


def test_integrated_report_prompt_includes_integration_reports():
    """generate_integrated_report must also feed integration_reports."""
    src = inspect.getsource(PIAgent.generate_integrated_report)
    assert "integration_reports" in src, (
        "generate_integrated_report must reference state.integration_reports"
    )


def test_supervisor_report_prompt_includes_integration_reports():
    """SupervisorAgent._build_report_prompt must feed integration_reports."""
    from backend.agents.supervisor_agent import SupervisorAgent
    src = inspect.getsource(SupervisorAgent._build_report_prompt)
    assert "integration_reports" in src, (
        "_build_report_prompt must reference state.integration_reports"
    )
