"""Provider cache telemetry aggregation and report rendering tests."""

import json

import pytest

from backend.services.report import regenerator


def _write_interactions(tmp_path, monkeypatch, interactions):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "llm.jsonl").write_text(
        "\n".join(json.dumps(item) for item in interactions),
        encoding="utf-8",
    )
    monkeypatch.setattr(regenerator, "get_session_logs", lambda _session_id: log_dir)


def test_agent_llm_stats_aggregate_cache_hits_before_calculating_rate(
    tmp_path, monkeypatch
):
    _write_interactions(tmp_path, monkeypatch, [
        {
            "agent": "pi",
            "model": "test-model",
            "temperature": 0.2,
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 100,
                "cached_tokens": 600,
            },
            "duration_ms": 100,
        },
        {
            "agent": "pi",
            "model": "test-model",
            "temperature": 0.2,
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 50,
                "cached_tokens": 0,
            },
            "duration_ms": 300,
        },
        {
            "agent": "literature",
            "model": "other-model",
            "usage": {"prompt_tokens": 200, "completion_tokens": 20},
        },
    ])

    stats = regenerator.get_agent_llm_stats("session-id")

    assert stats["pi"]["total_input_tokens"] == 1_500
    assert stats["pi"]["total_cached_input_tokens"] == 600
    assert stats["pi"]["input_cache_hit_rate"] == pytest.approx(0.4)
    assert stats["pi"]["avg_response_time_ms"] == pytest.approx(200)
    assert stats["literature"]["input_cache_hit_rate"] is None


def test_agent_llm_stats_requires_cache_metric_for_every_positive_input_call(
    tmp_path, monkeypatch
):
    _write_interactions(tmp_path, monkeypatch, [
        {
            "agent": "pi",
            "usage": {"prompt_tokens": 1_000, "cached_tokens": 600},
        },
        {"agent": "pi", "usage": {"prompt_tokens": 500}},
        {
            "agent": "pi",
            "usage": {"prompt_tokens": 100, "cached_tokens": 0},
        },
    ])

    stats = regenerator.get_agent_llm_stats("session-id")

    assert stats["pi"]["total_cached_input_tokens"] == 600
    assert stats["pi"]["input_cache_hit_rate"] is None


def test_agent_llm_stats_clamps_malformed_accumulated_cache_tokens(
    tmp_path, monkeypatch
):
    _write_interactions(tmp_path, monkeypatch, [
        {
            "agent": "pi",
            "usage": {"prompt_tokens": 100, "cached_tokens": 150},
        },
        {
            "agent": "pi",
            "usage": {"prompt_tokens": 100, "cached_tokens": -20},
        },
    ])

    stats = regenerator.get_agent_llm_stats("session-id")

    assert stats["pi"]["total_cached_input_tokens"] == 150
    assert stats["pi"]["input_cache_hit_rate"] == pytest.approx(0.75)


def test_agent_config_table_renders_zero_and_unknown_cache_rates():
    rendered = regenerator.render_agent_config_table_html(
        [
            ("PI", "pi", "model<&", 0.2),
            ("Literature", "literature", "model-b", 0.6),
        ],
        {
            "pi": {
                "call_count": 1,
                "total_input_tokens": 200,
                "total_output_tokens": 20,
                "input_cache_hit_rate": 0.0,
                "avg_response_time_ms": 50,
            },
            "literature": {
                "call_count": 1,
                "total_input_tokens": 200,
                "total_output_tokens": 20,
                "input_cache_hit_rate": None,
                "avg_response_time_ms": 0,
            },
        },
    )

    assert "200 <span" in rendered
    assert "(cache hit: 0.0%)" in rendered
    assert "(cache hit: not reported)" in rendered
    assert "model&lt;&amp;" in rendered
