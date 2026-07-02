"""select_brief_figures must never dump every candidate on failure."""
import pytest

from backend.agents.pi_agent import _bounded_brief_fallback


def _cands(*specs):
    return [{"name": n, "type": t, "is_publication": p} for n, t, p in specs]


def test_fallback_prefers_publication_then_result_types():
    cands = _cands(
        ("qc_violin", "violin", False),
        ("pca_variance", "other", False),
        ("umap_cell_types", "umap", False),
        ("pub_key", "heatmap", True),
    )
    out = _bounded_brief_fallback(cands, limit=20)
    assert out[0] == "pub_key"
    assert "umap_cell_types" in out
    assert "qc_violin" not in out
    assert "pca_variance" not in out


def test_fallback_is_bounded():
    cands = _cands(*[(f"umap_{i}", "umap", False) for i in range(50)])
    out = _bounded_brief_fallback(cands, limit=20)
    assert len(out) == 20


def test_fallback_is_deterministic_and_preserves_input_order():
    cands = _cands(
        ("umap_b", "umap", False),
        ("umap_a", "umap", False),
        ("volcano_c", "volcano", False),
    )
    assert _bounded_brief_fallback(cands, limit=20) == ["umap_b", "umap_a", "volcano_c"]


def test_fallback_empty_when_only_diagnostics():
    cands = _cands(
        ("qc_metrics", "other", False),
        ("doublet_score", "other", False),
    )
    assert _bounded_brief_fallback(cands, limit=20) == []


def test_fallback_excludes_ambiguous_other_type():
    cands = _cands(*[(f"artifact_{i}", "other", False) for i in range(160)])
    assert _bounded_brief_fallback(cands, limit=20) == []


def test_fallback_returns_pub_even_when_rest_are_other():
    cands = _cands(
        ("pub_a", "other", True),
        ("scratch_1", "other", False),
        ("scratch_2", "other", False),
    )
    assert _bounded_brief_fallback(cands, limit=20) == ["pub_a"]


@pytest.mark.asyncio
async def test_select_brief_figures_llm_failure_uses_bounded_fallback(monkeypatch):
    from backend.agents.pi_agent import PIAgent

    agent = PIAgent.__new__(PIAgent)

    class _BoomClient:
        async def chat(self, *a, **k):
            raise RuntimeError("upstream down")

    monkeypatch.setattr(type(agent), "llm_client", property(lambda self: _BoomClient()))
    monkeypatch.setattr(type(agent), "model", property(lambda self: "test-model"))
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None, raising=False)

    class _State:
        initial_instruction = "objective"

    all_figs = [
        {"name": f"umap_{i}", "type": "umap", "is_publication": False}
        for i in range(50)
    ]
    result = await agent.select_brief_figures(_State(), all_figs)
    assert 0 < len(result) <= 20
    assert set(result).issubset({f["name"] for f in all_figs})


def _ok_response(payload):
    class _Resp:
        content = payload

    class _Client:
        async def chat(self, *a, **k):
            return _Resp()

    return _Client()


def _agent_with_response(monkeypatch, payload):
    from backend.agents.pi_agent import PIAgent

    agent = PIAgent.__new__(PIAgent)
    monkeypatch.setattr(type(agent), "llm_client",
                        property(lambda self: _ok_response(payload)))
    monkeypatch.setattr(type(agent), "model", property(lambda self: "test-model"))
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None, raising=False)
    return agent


class _State:
    initial_instruction = "objective"


@pytest.mark.asyncio
async def test_select_filters_names_not_in_candidates(monkeypatch):
    """Names the PI returns that are not real candidates must be dropped."""
    agent = _agent_with_response(monkeypatch, '["umap_real", "nonexistent"]')
    all_figs = [
        {"name": "umap_real", "type": "umap", "is_publication": False},
        {"name": "volcano_real", "type": "volcano", "is_publication": False},
    ]
    result = await agent.select_brief_figures(_State(), all_figs)
    assert result == ["umap_real"]


@pytest.mark.asyncio
async def test_select_all_invalid_names_falls_back(monkeypatch):
    """If every returned name is invalid, fall back to the bounded set rather
    than returning hallucinated names that would empty the Results gallery."""
    agent = _agent_with_response(monkeypatch, '["ghost_a", "ghost_b"]')
    all_figs = [
        {"name": "umap_real", "type": "umap", "is_publication": False},
    ]
    result = await agent.select_brief_figures(_State(), all_figs)
    # bounded fallback surfaces the real result-type figure, never the ghosts.
    assert result == ["umap_real"]
