"""P2-2: surface tool call JSON parse failures explicitly."""
import re
import backend.utils.llm_client as llm_mod


def test_toolcall_has_parse_error_fields():
    """P2-2: ToolCall must expose parse_error and raw_arguments."""
    src = open(llm_mod.__file__).read()
    m = re.search(r"class ToolCall\b.*?(?=\nclass |\n@dataclass|\Z)", src, re.DOTALL)
    assert m, "could not locate ToolCall"
    body = m.group(0)
    assert "parse_error" in body, "P2-2: ToolCall must have parse_error field"
    assert "raw_arguments" in body, "P2-2: ToolCall must have raw_arguments field"


def test_arguments_not_wrapped_in_raw():
    """P2-2: must not produce arguments={'raw': ...} on parse failure."""
    src = open(llm_mod.__file__).read()
    assert '"raw":' not in src and "{'raw'" not in src, \
        "P2-2: must not wrap failed parse in {'raw': ...} — set parse_error/raw_arguments instead"


def test_toolcall_constructed_with_parse_error_on_bad_json():
    """When JSON parsing fails, ToolCall should carry parse_error and raw_arguments."""
    tc = llm_mod.ToolCall(
        id="call_1",
        name="execute_code",
        arguments={},
        parse_error="Expecting value: line 1 column 1 (char 0)",
        raw_arguments="not json",
    )
    assert tc.parse_error is not None
    assert tc.raw_arguments == "not json"
    assert tc.arguments == {}


def test_toolcall_default_no_parse_error():
    """Normal ToolCall construction should leave parse_error/raw_arguments as None."""
    tc = llm_mod.ToolCall(id="call_2", name="foo", arguments={"x": 1})
    assert tc.parse_error is None
    assert tc.raw_arguments is None
