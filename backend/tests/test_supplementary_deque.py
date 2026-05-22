"""P2-5: supplementary requests must be queued, not overwritten."""
import re
from collections import deque

import backend.workflow.core as core_mod


def test_supplementary_queue_is_deque():
    """P2-5: supplementary requests must be queued, not overwritten."""
    src = open(core_mod.__file__).read()
    # Field type / initialization should be a deque (or list with append/pop)
    assert "_supplementary_queue" in src or "deque" in src, \
        "P2-5: must use a queue/deque for supplementary requests"
    # No bare scalar assignment to _supplementary_request in signal_supplementary_analysis
    fn = re.search(
        r"def signal_supplementary_analysis\b.*?(?=\n    (?:async )?def |\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    # Should append, not overwrite
    assert "append" in body, \
        "P2-5: signal_supplementary_analysis must append (not overwrite)"


def test_two_consecutive_requests_preserved():
    """Two rapid requests should both be retained in order."""
    from backend.workflow.core import SingleCellAnalysisWorkflow

    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf._supplementary_queue = deque()
    # mimic asyncio events being available
    import asyncio
    wf._annotation_event = asyncio.Event()
    wf._continue_event = asyncio.Event()

    SingleCellAnalysisWorkflow.signal_supplementary_analysis(wf, "first", "annotation")
    SingleCellAnalysisWorkflow.signal_supplementary_analysis(wf, "second", "direction")

    assert len(wf._supplementary_queue) == 2
    assert wf._supplementary_queue[0] == ("first", "annotation")
    assert wf._supplementary_queue[1] == ("second", "direction")
