import re
import backend.utils.snapshot as snap_mod


def test_no_snapshots_last_fallback():
    """P2-4: when no last_restored_snapshot_id, parent_id must NOT fall back
    to snapshots[-1]. None (new branch root) is acceptable."""
    src = open(snap_mod.__file__).read()
    fn = re.search(r"def compute_snapshot_context\b.*?(?=\n(?:async )?def |\Z)", src, re.DOTALL)
    if not fn:
        fn = re.search(r"parent_id\s*=.*?(?=\n)", src)
        assert fn is not None, "could not locate parent_id assignment"
        body = fn.group(0)
    else:
        body = fn.group(0)
    fallback = re.search(r"parent_id\s*=\s*[^=].*snapshots\s*\[\s*-\s*1\s*\]", body)
    assert fallback is None, "P2-4: snapshots[-1] fallback for parent_id must be removed"
