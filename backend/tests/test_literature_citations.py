from backend.utils.llm_client import LLMResponse
from backend.agents.literature_agent import LiteratureAgent


def test_citations_preferred_over_regex():
    # 当 LLMResponse 带结构化 citations 时，应优先于正则提取
    cites = [{"title": "Struct", "url": "http://struct"}]
    resp = LLMResponse(content="body text without [1] refs", model="m", usage={},
                       raw_response=None, citations=cites)
    refs = LiteratureAgent._select_references(resp, "body text without [1] refs")
    assert refs == cites


def test_select_references_fallback_uses_regex_when_no_citations():
    # citations 为空时应走正则提取而非崩溃
    resp = LLMResponse(content="x", model="m", usage={}, raw_response=None, citations=None)
    text = "[1] Some Paper Title https://example.com/p1"
    refs = LiteratureAgent._select_references(resp, text)
    assert refs == [{"title": "Some Paper Title", "url": "https://example.com/p1"}]
