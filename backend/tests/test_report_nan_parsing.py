import json


def test_nan_in_string_value_preserved(tmp_path):
    from backend.routes.reports import _load_report_json

    p = tmp_path / "report.json"
    p.write_text('{"title": "NaN treatment study", "x": NaN}')
    data = _load_report_json(p)
    assert data["title"] == "NaN treatment study"
    assert data["x"] is None
