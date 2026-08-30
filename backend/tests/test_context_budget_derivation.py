"""Bioinformatics compaction thresholds follow the effective agent window."""

from backend.agents.bioinformatics_agent import BioinformaticsAgent


class _Stub:
    _EXECUTOR_COMPLETION_RESERVE = 32_768
    _COMPACT_THRESHOLD_FACTOR = 0.90
    _COMPACT_TARGET_RATIO = 0.70

    def __init__(self, context_window):
        self.context_window = context_window

    def _resolve_context_window(self):
        return self.context_window


def _threshold(window):
    return BioinformaticsAgent.COMPACT_THRESHOLD_TOKENS.fget(_Stub(window))


def test_default_thresholds_reproduce_the_existing_tuning():
    stub = _Stub(200_000)
    stub.COMPACT_THRESHOLD_TOKENS = _threshold(200_000)
    assert stub.COMPACT_THRESHOLD_TOKENS == 150_048
    assert BioinformaticsAgent.COMPACT_TARGET_TOKENS.fget(stub) == 105_033


def test_threshold_tracks_a_reconfigured_context_window():
    assert _threshold(400_000) == 330_048
    assert _threshold(400_000) > _threshold(200_000)
