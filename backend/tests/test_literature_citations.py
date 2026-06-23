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
    resp = LLMResponse(content="x", model="m", usage={}, raw_response=None, citations=None)
    text = "[1] Some Paper Title https://example.com/p1"
    refs = LiteratureAgent._select_references(resp, text)
    assert refs == [{"title": "Some Paper Title", "url": "https://example.com/p1"}]


def test_extract_inline_markdown_links():
    text = (
        "Kupffer cells express Clec4f "
        "([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC9154260/)). "
        "Endothelial markers include Stab2 "
        "([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6769646/))."
    )
    refs = LiteratureAgent._extract_references_from_text(text)
    urls = {r["url"] for r in refs}
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC9154260/" in urls
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC6769646/" in urls


def test_extract_ignores_image_markdown():
    text = (
        "Figure ![heatmap](https://example.com/fig.png) shows zonation "
        "([a](https://example.com/r1)) ([b](https://example.com/r2)) "
        "([c](https://example.com/r3))."
    )
    refs = LiteratureAgent._extract_references_from_text(text)
    urls = {r["url"] for r in refs}
    assert "https://example.com/fig.png" not in urls
    assert {"https://example.com/r1", "https://example.com/r2",
            "https://example.com/r3"} <= urls


def test_extract_strips_pmc_text_fragment_and_dedups():
    text = (
        "A ([x](https://pmc.ncbi.nlm.nih.gov/articles/PMC9154260/#:~:text=alpha)) "
        "and B ([x](https://pmc.ncbi.nlm.nih.gov/articles/PMC9154260/#:~:text=beta))."
    )
    refs = LiteratureAgent._extract_references_from_text(text)
    urls = [r["url"] for r in refs]
    assert urls.count("https://pmc.ncbi.nlm.nih.gov/articles/PMC9154260/") == 1


def test_extract_mixed_numbered_and_inline():
    text = (
        "Body cites Pax5 ([src](https://example.com/inline)).\n\n"
        "[1] Smith et al. (2023). Liver atlas. Nature. https://doi.org/10.1038/aaa\n"
        "[2] Doe et al. (2024). Kupffer cells. Cell. https://doi.org/10.1016/bbb"
    )
    refs = LiteratureAgent._extract_references_from_text(text)
    urls = {r["url"] for r in refs}
    assert "https://example.com/inline" in urls
    assert "https://doi.org/10.1038/aaa" in urls
    assert "https://doi.org/10.1016/bbb" in urls
