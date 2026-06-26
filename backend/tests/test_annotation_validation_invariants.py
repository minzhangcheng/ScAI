from backend.workflow.annotation_validation import MAIN_OBJECT_SIGNATURE_KEYS, signatures_match


def _sig():
    return {
        "id": 140123, "n_obs": 18233, "n_vars": 32000,
        "obs_columns": ["leiden", "condition"], "cluster_counts": {"0": 3406, "1": 2222},
        "layers": ["counts"], "x_id": 9001, "x_sum": 12345.6, "x_nnz": 999000,
        "obs_hash": "abc", "var_hash": "def",
        "layers_fingerprint": {"counts": {"sum": 100.0, "nnz": 50}},
    }


def test_detects_inplace_layer_content_change():
    after = _sig(); after["layers_fingerprint"] = {"counts": {"sum": 0.0, "nnz": 50}}
    ok, changed = signatures_match(_sig(), after)
    assert ok is False and "layers_fingerprint" in changed


def test_signature_keys_documented():
    assert set(MAIN_OBJECT_SIGNATURE_KEYS) == set(_sig().keys())


def test_identical_signatures_match():
    ok, changed = signatures_match(_sig(), _sig())
    assert ok is True and changed == []


def test_obs_column_order_does_not_matter():
    after = _sig(); after["obs_columns"] = ["condition", "leiden"]
    assert signatures_match(_sig(), after)[0] is True


def test_detects_inplace_x_content_change_with_same_id():
    after = _sig(); after["x_sum"] = 0.0; after["x_nnz"] = 500000
    ok, changed = signatures_match(_sig(), after)
    assert ok is False and "x_sum" in changed


def test_detects_existing_obs_column_value_change():
    after = _sig(); after["obs_hash"] = "MUTATED"
    ok, changed = signatures_match(_sig(), after)
    assert ok is False and "obs_hash" in changed


def test_detects_n_obs_and_layer_change():
    after = _sig(); after["n_obs"] = 17000; after["layers"] = ["counts", "scaled"]
    ok, changed = signatures_match(_sig(), after)
    assert ok is False and "n_obs" in changed and "layers" in changed


def test_missing_after_signature_is_unsafe():
    assert signatures_match(_sig(), {})[0] is False
