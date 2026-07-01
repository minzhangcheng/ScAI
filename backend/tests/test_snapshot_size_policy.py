from backend.utils.snapshot import should_exclude_snapshot_file, should_include_runtime_state


def test_oversized_runtime_pickle_is_skipped():
    include, reason = should_include_runtime_state(size_bytes=3 * 1024 * 1024 * 1024, max_bytes=2 * 1024 * 1024 * 1024)
    assert include is False
    assert "exceeds" in reason


def test_runtime_pickle_under_limit_is_included():
    include, reason = should_include_runtime_state(size_bytes=1024, max_bytes=2048)
    assert include is True
    assert reason == ""


def test_core_dump_is_excluded_from_manifest():
    assert should_exclude_snapshot_file("core.123", ("core.*", "*.tmp", "*.lock")) is True
    assert should_exclude_snapshot_file("analysis.csv", ("core.*", "*.tmp", "*.lock")) is False
