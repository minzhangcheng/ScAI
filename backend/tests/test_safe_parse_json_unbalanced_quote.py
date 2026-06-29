import backend.utils.formatting as formatting


def _malformed_plan_response() -> str:
    return '''{
  "overview": "Stage 1 plan",
  "steps": [
    {"id": 1, "title": "Load data", "method": "scanpy.read_10x_h5", "depends_on": []},
    {"id": 27", "title": "Compute UMAP", "method": "scanpy.tl.umap", "depends_on": [1]}
  ],
  "expected_runtime": "~2 steps"
}'''


def test_safe_parse_json_unbalanced_quote_repair(monkeypatch):
    def fake_repair(candidate, return_objects=False):
        assert return_objects is True
        assert candidate.startswith("{")
        return {
            "overview": "Stage 1 plan",
            "steps": [
                {"id": 1, "title": "Load data", "method": "scanpy.read_10x_h5", "depends_on": []},
                {"id": 27, "title": "Compute UMAP", "method": "scanpy.tl.umap", "depends_on": [1]},
            ],
            "expected_runtime": "~2 steps",
        }

    monkeypatch.setattr(formatting, "_JSON_REPAIR_AVAILABLE", True)
    monkeypatch.setattr(formatting, "_repair_json", fake_repair)

    result = formatting.safe_parse_json(_malformed_plan_response(), default=None)

    assert result is not None
    assert len(result["steps"]) == 2
    assert result["steps"][1]["id"] == 27


def test_safe_parse_json_unbalanced_quote_repair_with_wrapping_text(monkeypatch):
    calls = []

    def fake_repair(candidate, return_objects=False):
        calls.append(candidate)
        if candidate.startswith("{"):
            raise ValueError("segment repair failed")
        return {
            "overview": "wrapped plan",
            "steps": [{"id": 1, "title": "QC", "method": "scanpy.pp.filter_cells"}],
        }

    monkeypatch.setattr(formatting, "_JSON_REPAIR_AVAILABLE", True)
    monkeypatch.setattr(formatting, "_repair_json", fake_repair)

    response = "Here is the plan:\n```json\n" + _malformed_plan_response() + "\n```"
    result = formatting.safe_parse_json(response, default=None)

    assert result is not None
    assert result["overview"] == "wrapped plan"
    assert len(result["steps"]) == 1
    assert len(calls) == 2
    assert calls[0].startswith("{")
    assert calls[1].startswith("Here is the plan")


def test_safe_parse_json_unbalanced_quote_no_repair(monkeypatch):
    monkeypatch.setattr(formatting, "_JSON_REPAIR_AVAILABLE", False)

    result = formatting.safe_parse_json(_malformed_plan_response(), default=None)

    assert result is None
