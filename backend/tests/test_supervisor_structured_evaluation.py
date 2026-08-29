from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.supervisor.evaluation import (
    EvaluationMixin,
    _SUPERVISOR_EVALUATION_FIELDS,
    _SUPERVISOR_REQUIRED_FIELDS,
    _SUPERVISOR_EVALUATION_TOOL_NAME,
)
from backend.utils.llm_client_parts.responses import ToolCall


def _valid_verdict(**overrides):
    verdict = {
        "severity": "warning",
        "issues": ["The legend is unreadable"],
        "action": "warn",
        "message": "Increase the legend font size.",
        "redirect_instruction": "",
        "issue_type": "technical",
        "is_biological_checkpoint": False,
        "pi_question": "",
    }
    verdict.update(overrides)
    return verdict


def _tool_response(arguments=None, *, content=None, parse_error=None):
    calls = []
    if arguments is not None:
        calls.append(
            ToolCall(
                id="call-1",
                name=_SUPERVISOR_EVALUATION_TOOL_NAME,
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return SimpleNamespace(content=content, tool_calls=calls)


class _FakeClient:
    def __init__(self, responses, *, vision_content=""):
        self.responses = list(responses)
        self.calls = []
        self.vision_calls = []
        self.vision_content = vision_content

    async def chat_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def chat_with_vision(self, **kwargs):
        self.vision_calls.append(kwargs)
        return SimpleNamespace(content=self.vision_content)


class _SupervisorHarness(EvaluationMixin):
    model = "test-model"
    temperature = 0.1
    system_prompt = ""

    def __init__(self, client, *, has_vision=False):
        self.llm_client = client
        self.has_vision = has_vision
        self.context_calls = 0

    def _set_llm_context(self):
        self.context_calls += 1

    @staticmethod
    async def _budgeted_text_call(method, **kwargs):
        return await method(**kwargs)

    @staticmethod
    def _compress_base64_figure(value):
        return value


@pytest.mark.asyncio
async def test_structured_evaluation_retries_schema_failure_with_forced_tool():
    client = _FakeClient(
        [
            _tool_response(_valid_verdict(issues="not-a-list")),
            _tool_response(_valid_verdict()),
        ]
    )
    supervisor = _SupervisorHarness(client)

    result = await supervisor._request_structured_evaluation("system", "step")

    assert result == _valid_verdict()
    assert len(client.calls) == 2
    first = client.calls[0]
    assert first["tool_choice"]["function"]["name"] == _SUPERVISOR_EVALUATION_TOOL_NAME
    schema = first["tools"][0]["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == _SUPERVISOR_REQUIRED_FIELDS
    assert "previous response was rejected" in client.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_structured_evaluation_marks_exhausted_invalid_output_unparsed():
    invalid = _valid_verdict(is_biological_checkpoint="false")
    client = _FakeClient([_tool_response(invalid), _tool_response(invalid)])
    supervisor = _SupervisorHarness(client)

    result = await supervisor._request_structured_evaluation("system", "step")

    assert result["parse_failed"] is True
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_clean_verdict_casefolds_enums_and_defaults_optional_null_without_retry():
    client = _FakeClient([
        _tool_response(_valid_verdict(
            severity=" NONE ",
            issues=[],
            action=" NONE ",
            message="",
            issue_type=None,
            redirect_instruction=None,
            pi_question=None,
        )),
    ])
    supervisor = _SupervisorHarness(client)

    result = await supervisor._request_structured_evaluation("system", "step")

    assert result["severity"] == "none"
    assert result["action"] == "none"
    assert result["issue_type"] == "ambiguous"
    assert result["redirect_instruction"] == ""
    assert result["pi_question"] == ""
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_null_severity_and_blank_issue_type_with_issues_stay_invalid():
    invalid = _valid_verdict(severity=None, issue_type="", issues=["real issue"])
    client = _FakeClient([_tool_response(invalid), _tool_response(_valid_verdict())])
    supervisor = _SupervisorHarness(client)

    result = await supervisor._request_structured_evaluation("system", "step")

    assert result == _valid_verdict()
    assert len(client.calls) == 2
    assert "invalid severity" in client.calls[1]["messages"][-1]["content"]


def test_optional_fields_may_be_omitted_or_explicit_null():
    omitted, omitted_error = EvaluationMixin._validate_supervisor_result(
        {"severity": " NONE "}
    )
    explicit_null, null_error = EvaluationMixin._validate_supervisor_result({
        "severity": "info",
        "issues": None,
        "action": None,
        "message": None,
        "redirect_instruction": None,
        "issue_type": None,
        "is_biological_checkpoint": None,
        "pi_question": None,
    })

    assert omitted_error == ""
    assert omitted == {
        "severity": "none",
        "issues": [],
        "action": "none",
        "message": "",
        "redirect_instruction": "",
        "issue_type": "ambiguous",
        "is_biological_checkpoint": False,
        "pi_question": "",
    }
    assert null_error == ""
    assert explicit_null == {**omitted, "severity": "info"}


def test_parser_accepts_optional_omission_but_rejects_wrongly_typed_objects():
    import json

    partial = EvaluationMixin._parse_supervisor_response('{"severity":"none"}')
    assert partial["severity"] == "none"
    assert partial["issues"] == []
    assert "parse_failed" not in partial
    invalid = _valid_verdict(issues=[1])
    assert EvaluationMixin._parse_supervisor_response(json.dumps(invalid))[
        "parse_failed"
    ] is True


def test_parser_accepts_schema_valid_markdown_json():
    import json

    text = f"```json\n{json.dumps(_valid_verdict())}\n```"
    assert EvaluationMixin._parse_supervisor_response(text) == _valid_verdict()


@pytest.mark.asyncio
async def test_vision_observation_is_revalidated_by_structured_tool_call():
    client = _FakeClient(
        [_tool_response(_valid_verdict(severity="critical"))],
        vision_content="The plot is blank.",
    )
    supervisor = _SupervisorHarness(client, has_vision=True)
    step = SimpleNamespace(
        step_number=5,
        step_type="execute_code",
        agent_reasoning=None,
        code="plot()",
        code_output="ok",
        code_error=None,
        literature_topic=None,
        literature_result=None,
        annotation_result_summary=None,
        findings=[],
    )

    result = await supervisor._llm_evaluate(step, ["base64-image"])

    assert result["severity"] == "critical"
    assert client.vision_calls[0]["system_prompt"]
    structured_prompt = client.calls[0]["messages"][1]["content"]
    assert "The plot is blank." in structured_prompt
