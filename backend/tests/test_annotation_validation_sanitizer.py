from backend.workflow.annotation_validation import sanitize_validation_result


def test_truncates_long_summary():
    out = sanitize_validation_result({"status": "completed", "summary": "x" * 10000,
                                      "cluster_results": {}, "artifacts": [], "warnings": []})
    assert len(out["summary"]) <= 4000


def test_flags_each_figure_missing_summary_as_partial():
    res = {"status": "completed", "summary": "ok",
           "cluster_results": {"5": {"support_level": "supported", "qc_flags": ["ok"]}},
           "artifacts": [{"type": "figure", "name": "good", "path": "g.jpg", "summary": "means: ..."},
                         {"type": "figure", "name": "bad", "path": "b.jpg"}],
           "warnings": []}
    out = sanitize_validation_result(res)
    assert out["status"] == "partial"
    assert any("bad" in w for w in out["warnings"])


def test_keeps_complete_result_completed():
    res = {"status": "completed", "summary": "ok",
           "cluster_results": {"5": {"support_level": "supported",
                                     "evidence_for": ["marker high"], "qc_flags": ["ok"]}},
           "artifacts": [{"type": "figure", "name": "f", "path": "f.jpg", "summary": "means: a>b"},
                         {"type": "table", "name": "t", "path": "t.csv"}],
           "warnings": []}
    out = sanitize_validation_result(res)
    assert out["status"] == "completed"


def test_caps_artifacts_count():
    res = {"status": "completed", "summary": "ok", "cluster_results": {},
           "artifacts": [{"type": "table", "name": str(i), "path": f"{i}.csv"} for i in range(100)],
           "warnings": []}
    out = sanitize_validation_result(res)
    assert len(out["artifacts"]) <= 30


def test_truncates_long_artifact_summary():
    res = {"status": "completed", "summary": "ok",
           "cluster_results": {"5": {"qc_flags": ["ok"]}},
           "artifacts": [{"type": "figure", "name": "f", "path": "f.jpg", "summary": "x" * 9000}],
           "warnings": []}
    out = sanitize_validation_result(res)
    assert len(out["artifacts"][0]["summary"]) <= 2100  # _MAX_FIELD (2000) + truncation marker
