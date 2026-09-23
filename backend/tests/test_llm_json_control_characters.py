"""Literal newlines in model strings must not discard plans or tool arguments."""

import json
from unittest.mock import AsyncMock

import pytest

from backend.agents.base import BaseAgent
from backend.agents.pi.research_planning import ResearchPlanningMixin
from backend.agents.pi.synthesis import _synthesis_report_chapter
from backend.agents.supervisor.evaluation_support import _parse_supervisor_response
from backend.models.schemas import AnalysisState
from backend.routes.chat_parts.delegation import _try_parse_delegation
from backend.utils import formatting
from backend.utils.llm_client_parts.protocol import (
    _anthropic_messages_translate,
    _parse_responses_dump,
)
from backend.utils.llm_client_parts.shared import _recover_stringified_tool_call


def _model_json(value):
    return json.dumps(value).replace(r"\n", "\n").replace(r"\t", "\t")


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```"])
def test_delegation_preserves_multiline_instructions(wrapper):
    plan = {
        "delegation": True,
        "plan": "Review data\nGenerate figures",
        "delegates": [{
            "agent": "bioinformatics",
            "instruction": 'Inspect {existing} results.\n\tSave "publication" figures.',
        }],
        "synthesis_instruction": "Summarize results\nInclude figures",
    }
    assert _try_parse_delegation(wrapper.format(_model_json(plan))) == plan


@pytest.mark.parametrize("text", [
    "Please explain this result.",
    '{"delegation": false, "delegates": [{"agent": "bioinformatics"}]}',
    '{"delegation": true, "delegates": []}',
    '{"delegation": true, "delegates": [{"agent": "bioinformatics"}]',
])
def test_non_delegations_and_broken_structure_do_not_dispatch(text):
    assert _try_parse_delegation(text) is None


def test_normal_agent_parsers_preserve_controls_without_optional_repair(monkeypatch):
    monkeypatch.setattr(formatting, "_JSON_REPAIR_AVAILABLE", False)
    plan = {"instructions": 'Review {results}\n\tPreserve "labels"', "approved": True}
    raw = _model_json(plan)
    assert BaseAgent._safe_parse_json(None, raw) == plan
    assert formatting.safe_parse_json_array("[" + raw + "]") == [plan]
    with pytest.raises(json.JSONDecodeError):
        formatting.loads_llm_json(raw[:-1])
    # This tolerance is local to model output, not global JSON validation.
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)


@pytest.mark.asyncio
async def test_workflow_plan_does_not_fall_back_on_multiline_steps():
    pi = ResearchPlanningMixin()
    steps = ["Inspect existing data\nReuse verified results", "Generate figures"]
    pi.chat = AsyncMock(return_value=_model_json(steps))
    assert await pi.plan_bioinformatics_workflow(AnalysisState(user_id="u")) == steps


def test_separately_fenced_report_chapter_preserves_multiline_text():
    text = "First observation.\nSecond observation."
    chapter = {"title": "Results", "blocks": [{"text": text, "artifact_ids": []}]}
    response = '```json\n{"summary": "Retained summary"}\n```\n```json\n'
    response += _model_json(chapter) + "\n```"
    parsed = _synthesis_report_chapter({"summary": "Retained summary"}, response, [], 1)
    assert parsed.blocks[0].text == text


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Evaluation:\n{}"])
def test_supervisor_does_not_drop_multiline_observations(wrapper):
    verdict = {"severity": "warning", "message": "Legend overlaps.\nPlease revise."}
    parsed = _parse_supervisor_response(wrapper.format(_model_json(verdict)))
    assert parsed["severity"] == "warning"
    assert parsed["message"] == verdict["message"]
    assert not parsed.get("parse_failed")


def test_tool_protocols_preserve_multiline_code_and_indentation():
    args = {"code": "for i in range(2):\n\tprint(i)"}
    raw = _model_json(args)
    _, calls, _, _ = _parse_responses_dump({"output": [{
        "type": "function_call", "name": "execute_code", "arguments": raw,
    }]})
    assert calls[0].arguments == args
    assert calls[0].parse_error is None
    _, messages = _anthropic_messages_translate([{
        "role": "assistant", "tool_calls": [{
            "id": "call", "function": {"name": "execute_code", "arguments": raw},
        }],
    }])
    assert messages[0]["content"][0]["input"] == args
    assert _recover_stringified_tool_call(_model_json({
        "tool": "execute_code", "parameters": args,
    })) == ("execute_code", args)


def test_tool_protocol_still_reports_broken_json_structure():
    raw = '{"code": "print(1)"'
    _, calls, _, _ = _parse_responses_dump({"output": [{
        "type": "function_call", "name": "execute_code", "arguments": raw,
    }]})
    assert calls[0].parse_error
    assert calls[0].raw_arguments == raw
