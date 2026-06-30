import asyncio
from backend.models.schemas import AnalysisState, LiteratureReport
from backend.workflow.core import SingleCellAnalysisWorkflow


class _FakeLitAgent:
    def __init__(self):
        self.called_with = None

    async def extract_mechanism_signals(self, report_text, references, observed_context):
        self.called_with = report_text
        return None  # non-critical path; return no signals


def test_blob_token_report_is_not_passed_as_text():
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "t"
    st = AnalysisState(user_id="u1")
    rep = LiteratureReport(topic="t", report="@scai-blob:deadbeef", references=[])
    st.literature_reports = [rep]
    agent = _FakeLitAgent()

    asyncio.run(wf._persist_literature_mechanism_signals(
        st, agent, [rep.id], observed_context="ctx",
    ))
    # Either extraction was skipped (not called) or the blob token was filtered out.
    assert agent.called_with is None or "@scai-blob:" not in (agent.called_with or "")
