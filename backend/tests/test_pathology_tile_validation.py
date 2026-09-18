"""Regression tests for pathology tile-batch validation and retry."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.agents.pathology_agent as mod
from backend.agents.pathology_agent import (
    PathologyAgent,
    _coerce_tile_batch,
    _validate_tile_batch,
)
from backend.models.schemas import AgentType, DataFile
from backend.tests.split_facade import setattr_facade


def _bare_agent() -> PathologyAgent:
    agent = object.__new__(PathologyAgent)
    agent.agent_type = AgentType.PATHOLOGY
    agent._provider = MagicMock()
    agent._provider.get_config_revision.return_value = 0
    agent._provider.get_model.return_value = "vision-model"
    agent._last_config_revision = 0
    agent.llm_intervention_callback = None
    agent.tile_size = 224
    agent.max_tiles = 2
    return agent


def test_tile_batch_rejects_wrong_count_and_placeholders():
    too_many = {
        "tile_findings": [
            {"tile_index": i, "findings": f"Concrete observation for region {i}."}
            for i in range(45)
        ]
    }
    normalized, errors = _validate_tile_batch(
        too_many, start_idx=0, expected_count=5
    )
    assert normalized == []
    assert "count 45 does not match expected 5" in errors[0]

    placeholders = {
        "tile_findings": [
            {"tile_index": 1, "findings": "Tile 1 findings..."},
            {"tile_index": 2, "findings": "Tile 2 findings..."},
        ]
    }
    normalized, errors = _validate_tile_batch(
        placeholders, start_idx=0, expected_count=2
    )
    assert normalized == []
    assert all("placeholder" in error for error in errors)


def test_tile_batch_assigns_canonical_indices_and_accepts_explicit_normal():
    batch = {
        "tile_findings": [
            {"tile_index": 100, "findings": "No abnormality detected."},
            {"tile_index": 101, "findings": "Mild stromal expansion is visible."},
        ]
    }
    normalized, errors = _validate_tile_batch(
        batch, start_idx=5, expected_count=2
    )
    assert errors == []
    assert [entry["tile_index"] for entry in normalized] == [5, 6]


def test_tile_batch_rejects_prompt_category_echo_and_repeated_abnormality():
    templated = {
        "tile_findings": [
            {
                "tile_index": 0,
                "findings": "Finding 1: enlarged irregular cells are present.",
            },
            {
                "tile_index": 1,
                "findings": "Dense collagen surrounds several vascular profiles.",
            },
        ]
    }
    normalized, errors = _validate_tile_batch(
        templated, start_idx=0, expected_count=2
    )
    assert normalized == []
    assert any("prompt category" in error for error in errors)

    repeated = {
        "tile_findings": [
            {
                "tile_index": 2,
                "findings": "Dense collagen surrounds several vascular profiles.",
            }
        ]
    }
    normalized, errors = _validate_tile_batch(
        repeated,
        start_idx=2,
        expected_count=1,
        prior_findings=[{
            "tile_index": 0,
            "findings": "Dense collagen surrounds several vascular profiles.",
        }],
    )
    assert normalized == []
    assert any("repeats an identical abnormal finding" in error for error in errors)


def test_tile_batch_allows_repeated_explicit_normal_findings():
    batch = {
        "tile_findings": [
            {"tile_index": 0, "findings": "No abnormality detected."},
            {"tile_index": 1, "findings": "No abnormality detected."},
        ]
    }
    normalized, errors = _validate_tile_batch(
        batch,
        start_idx=0,
        expected_count=2,
        prior_findings=[{"findings": "No abnormality detected."}],
    )
    assert errors == []
    assert len(normalized) == 2


def _patch_image_setup(monkeypatch, agent: PathologyAgent) -> DataFile:
    monkeypatch.setattr(agent, "refresh_runtime_config_if_needed", lambda: None)
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(mod.os.path, "exists", lambda path: True)
    setattr_facade(monkeypatch, mod.__name__, "load_image", lambda path: object())
    setattr_facade(
        monkeypatch,
        mod.__name__,
        "process_pathology_image",
        lambda *args, **kwargs: ["tile-a", "tile-b"],
    )
    setattr_facade(
        monkeypatch, mod.__name__, "create_overview_image", lambda image: "overview"
    )

    async def direct_hitl(call_factory, image_label, step_label):
        return await call_factory()

    monkeypatch.setattr(agent, "_call_llm_with_hitl", direct_hitl)
    return DataFile(
        id="img-1",
        filename="slide.tif",
        file_path="/fake/slide.tif",
        section="pathology",
        data_type="H&E",
    )


@pytest.mark.asyncio
async def test_invalid_tile_batch_retries_then_uses_only_validated_entries(monkeypatch):
    agent = _bare_agent()
    image = _patch_image_setup(monkeypatch, agent)
    prompts = []
    stream_preferences = []
    responses = [
        {"overall_structure": "Preserved", "severity": "normal"},
        {
            "tile_findings": [
                {"tile_index": 1, "findings": "Tile 1 findings..."},
                {"tile_index": 2, "findings": "Tile 2 findings..."},
            ]
        },
        {
            "tile_findings": [
                {"tile_index": 99, "findings": "No abnormality detected."},
                {"tile_index": 100, "findings": "Mild stromal expansion is visible."},
            ]
        },
    ]

    class FakeClient:
        async def chat_with_vision(self, **kwargs):
            prompts.append(kwargs["text_prompt"])
            stream_preferences.append(kwargs.get("prefer_stream"))
            return SimpleNamespace(content=json.dumps(responses.pop(0)))

    agent._provider.get_llm_client.return_value = FakeClient()

    async def synthesize(prompt, temperature=None):
        assert prompt.index("Reply in JSON format:") < prompt.index("## Overview Analysis")
        assert "Mild stromal expansion is visible." in prompt
        return json.dumps({
            "findings": ["Validated finding"],
            "severity_score": 1,
            "further_analysis": [],
            "suggested_staining": [],
        })

    monkeypatch.setattr(agent, "chat_without_history", synthesize)
    result = await agent._analyze_image(image, "Analyze")

    assert prompts[0].index("Please reply in JSON format:") < prompts[0].index("Analysis instruction:")
    assert len(prompts) == 3  # overview + invalid batch + corrected retry
    assert stream_preferences == [True, True, True]
    assert "previous response failed validation" in prompts[-1]
    assert [entry["tile_index"] for entry in result["tile_findings"]] == [0, 1]
    assert result["tile_validation"]["status"] == "completed"
    assert result["tile_validation"]["invalid_batches"] == []


@pytest.mark.asyncio
async def test_twice_invalid_tile_batch_is_excluded_and_marks_image_error(monkeypatch):
    agent = _bare_agent()
    image = _patch_image_setup(monkeypatch, agent)
    responses = [
        {"overall_structure": "Preserved", "severity": "normal"},
        {"tile_findings": [{"tile_index": 0, "findings": "Tile 1 findings..."}] * 2},
        {"tile_findings": [{"tile_index": 0, "findings": "No findings"}] * 2},
    ]

    class FakeClient:
        async def chat_with_vision(self, **kwargs):
            return SimpleNamespace(content=json.dumps(responses.pop(0)))

    agent._provider.get_llm_client.return_value = FakeClient()

    async def unexpected_synthesis(*args, **kwargs):
        raise AssertionError("invalid tile data must not reach synthesis")

    monkeypatch.setattr(agent, "chat_without_history", unexpected_synthesis)
    result = await agent._analyze_image(image, "Analyze")

    assert result["status"] == "error"
    assert result["findings"] == []
    assert result["tile_validation"]["valid_tiles"] == 0
    assert len(result["tile_validation"]["invalid_batches"]) == 1


def test_bare_array_batch_is_recovered_rather_than_discarded():
    """A top-level array is unambiguous; safe_parse_json alone truncates it.

    safe_parse_json scans for the first '{', which for a bare array is its own
    first element — one tile, no "tile_findings" key. Left uncorrected the whole
    batch is thrown away even though every requested finding is present.
    """
    entries = [
        {"tile_index": i, "findings": "No abnormality detected."}
        for i in range(5)
    ]
    raw = json.dumps(entries)
    truncated = json.loads(raw[raw.index("{"):raw.index("}") + 1])
    assert "tile_findings" not in truncated  # what safe_parse_json returns today

    coerced = _coerce_tile_batch(truncated, raw)
    normalized, errors = _validate_tile_batch(
        coerced, start_idx=0, expected_count=5
    )
    assert errors == []
    assert [entry["tile_index"] for entry in normalized] == [0, 1, 2, 3, 4]


def test_bare_array_recovered_from_markdown_fence_and_prose():
    entries = [
        {"tile_index": 0, "findings": "Dense collagen surrounds a portal tract."},
        {"tile_index": 1, "findings": "No abnormality detected."},
    ]
    raw = "Here are the findings:\n```json\n" + json.dumps(entries) + "\n```"
    truncated = {"tile_index": 0, "findings": entries[0]["findings"]}

    coerced = _coerce_tile_batch(truncated, raw)
    normalized, errors = _validate_tile_batch(
        coerced, start_idx=0, expected_count=2
    )
    assert errors == []
    assert len(normalized) == 2


def test_coerce_leaves_documented_object_untouched():
    batch = {"tile_findings": [{"tile_index": 0, "findings": "Mild steatosis."}]}
    assert _coerce_tile_batch(batch, json.dumps(batch)) is batch


def test_coerce_keeps_original_when_no_array_is_present():
    unparsable = {"raw_response": "the model refused"}
    assert _coerce_tile_batch(unparsable, "the model refused") is unparsable


def test_validate_accepts_a_list_directly():
    entries = [{"tile_index": 0, "findings": "No abnormality detected."}]
    normalized, errors = _validate_tile_batch(
        entries, start_idx=3, expected_count=1
    )
    assert errors == []
    assert normalized[0]["tile_index"] == 3


@pytest.mark.asyncio
async def test_bare_array_batch_validates_without_burning_a_retry(monkeypatch):
    agent = _bare_agent()
    image = _patch_image_setup(monkeypatch, agent)
    prompts = []
    responses = [
        json.dumps({"overall_structure": "Preserved", "severity": "normal"}),
        # Bare array — the shape Lingjian-1.5 emits in production.
        json.dumps([
            {"tile_index": 0, "findings": "No abnormality detected."},
            {"tile_index": 1, "findings": "Mild stromal expansion is visible."},
        ]),
    ]

    class FakeClient:
        async def chat_with_vision(self, **kwargs):
            prompts.append(kwargs["text_prompt"])
            return SimpleNamespace(content=responses.pop(0))

    agent._provider.get_llm_client.return_value = FakeClient()

    async def synthesize(prompt, temperature=None):
        return json.dumps({
            "findings": ["Validated finding"],
            "severity_score": 1,
            "further_analysis": [],
            "suggested_staining": [],
        })

    monkeypatch.setattr(agent, "chat_without_history", synthesize)
    result = await agent._analyze_image(image, "Analyze")

    assert len(prompts) == 2  # overview + one batch, no validation retry
    assert all("previous response failed validation" not in p for p in prompts)
    assert [entry["tile_index"] for entry in result["tile_findings"]] == [0, 1]
    assert result["tile_validation"]["status"] == "completed"
    assert result["tile_validation"]["invalid_batches"] == []


@pytest.mark.asyncio
async def test_tile_batch_prefix_survives_index_and_batch_size_changes(monkeypatch):
    from backend.agents.pathology_parts.image_analysis import _analyze_tile_batches

    agent = _bare_agent()
    image = _patch_image_setup(monkeypatch, agent)
    prompts = []
    images = []

    class FakeClient:
        async def chat_with_vision(self, **kwargs):
            prompts.append(kwargs["text_prompt"])
            images.append(kwargs["images"])
            return SimpleNamespace(content=json.dumps({"tile_findings": [
                {"tile_index": 0, "findings": "No abnormality detected."}
                for _ in kwargs["images"]
            ]}))

    agent._provider.get_llm_client.return_value = FakeClient()
    findings, invalid = await _analyze_tile_batches(
        agent, image, ["a", "b", "c"], batch_size=2, bindings=mod,
    )
    assert not invalid
    assert [f["tile_index"] for f in findings] == [0, 1, 2]
    assert images == [["a", "b"], ["c"]]
    assert prompts[0].split("## Current Batch")[0] == prompts[1].split("## Current Batch")[0]
    assert '"tile_findings"' in prompts[0].split("## Current Batch")[0]
    assert "Return EXACTLY 2 objects, with tile_index beginning at 0." in prompts[0]
    assert "Return EXACTLY 1 objects, with tile_index beginning at 2." in prompts[1]
