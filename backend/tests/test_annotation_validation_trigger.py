from backend.workflow.annotation_validation import should_run_validation


def test_skips_when_all_high_confidence_and_no_triage():
    annotations = {"0": {"cell_type": "Hepatocyte", "confidence": "high",
                         "alternative_cell_types": [], "low_confidence_reason": ""}}
    assert should_run_validation(annotations, {}) is False


def test_triggers_on_low_or_medium_confidence():
    assert should_run_validation({"0": {"cell_type": "X", "confidence": "low"}}, {}) is True
    assert should_run_validation({"0": {"cell_type": "X", "confidence": "medium"}}, {}) is True


def test_triggers_on_alternative_cell_types():
    assert should_run_validation({"0": {"cell_type": "X", "confidence": "high",
                                        "alternative_cell_types": ["Y"]}}, {}) is True


def test_triggers_on_uncertainty_keyword():
    assert should_run_validation({"0": {"cell_type": "Ambiguous lineage", "confidence": "high"}}, {}) is True
    assert should_run_validation({"0": {"cell_type": "X", "confidence": "high",
                                        "low_confidence_reason": "possible doublet"}}, {}) is True


def test_triggers_on_risk_triage_buckets():
    assert should_run_validation({"0": {"confidence": "high"}},
                                 {"suspected_contamination": [{"cluster_id": "7"}]}) is True
    assert should_run_validation({"0": {"confidence": "high"}},
                                 {"rare_unresolved": [{"cluster_id": "5"}]}) is True
    # Unknown risk-ish bucket name still triggers when non-empty.
    assert should_run_validation({"0": {"confidence": "high"}},
                                 {"technical_qc_concern": [{"cluster_id": "9"}]}) is True


def test_covered_bucket_does_not_trigger():
    assert should_run_validation({"0": {"confidence": "high"}},
                                 {"covered": [{"cluster_id": "0"}, {"cluster_id": "1"}]}) is False
    assert should_run_validation({"0": {"confidence": "high"}},
                                 {"covered": [{"cluster_id": "0"}], "rare_unresolved": [],
                                  "suspected_contamination": [], "context_dependent_state": []}) is False


def test_empty_risk_buckets_do_not_trigger():
    assert should_run_validation({"0": {"confidence": "high"}},
                                 {"suspected_contamination": [], "rare_unresolved": []}) is False


def test_empty_annotations_does_not_trigger():
    assert should_run_validation({}, {}) is False
