"""Integration: nested candidates reach PI selection; failed PI selection is bounded."""
import types

import pytest

import backend.workflow.reporting as reporting_mod
from backend.tests.split_facade import setattr_facade


def _pub_fig(name):
    """Minimal publication-figure stand-in with the attributes reporting reads."""
    return types.SimpleNamespace(
        id=f"id-{name}",
        name=name,
        description="",
        figure_type="umap",
        png_base64="AAA",
        pdf_path=None,
        review_status=types.SimpleNamespace(value="approved"),
        revision_count=0,
    )


class _CaptureReporter(reporting_mod.ReportingMixin):
    """Minimal harness exercising only the figure-assembly block."""

    def __init__(self, session_id, discovered, publication_figures,
                 pi_return=None, pi_agent=None):
        self.session_id = session_id
        self._discovered = discovered
        self._pi_return = pi_return
        self.captured_selection_input = None
        self.sent = {}

        if pi_agent is not None:
            self.pi_agent = pi_agent
        else:
            class _PI:
                async def select_brief_figures(_self, state, all_figs):
                    self.captured_selection_input = all_figs
                    return self._pi_return

            self.pi_agent = _PI()

        class _State:
            initial_instruction = "obj"
        self.analysis_state = _State()
        self.analysis_state.publication_figures = publication_figures
        self.analysis_state.publication_tables = []

    async def _send_message(self, kind, payload, agent):
        self.sent[kind] = payload


@pytest.mark.asyncio
async def test_nested_candidates_reach_selection_and_cap(monkeypatch, tmp_path):
    discovered = [
        {"name": f"umap_{i}", "type": "umap",
         "path": str(tmp_path / f"umap_{i}.png"), "pdf_path": None,
         "mtime": float(i), "source": "r1_outputs/figures"}
        for i in range(250)
    ]
    setattr_facade(
        monkeypatch, "backend.workflow.reporting", "discover_workspace_figures", lambda sid: discovered
    )
    setattr_facade(
        monkeypatch, "backend.workflow.reporting", "compress_image_to_base64", lambda *a, **k: None
    )

    reporter = _CaptureReporter("sess-x", discovered, publication_figures=[],
                                pi_return=["umap_0", "umap_5"])
    await reporter._assemble_brief_figures(reporter.analysis_state, {"summary": "s"})

    assert len(reporter.captured_selection_input) == 200
    assert all("is_publication" in f for f in reporter.captured_selection_input)
    payload = reporter.sent["integrated_report"]
    names = {f["name"] for f in payload["figures"]}
    assert names == {"umap_0", "umap_5"}


@pytest.mark.asyncio
async def test_real_pi_llm_failure_does_not_dump_all(monkeypatch, tmp_path):
    """End-to-end: the real select_brief_figures LLM call fails, and the
    integrated_report payload must stay bounded -- not every discovered candidate."""
    discovered = [
        {"name": f"umap_{i}", "type": "umap",
         "path": str(tmp_path / f"umap_{i}.png"), "pdf_path": None,
         "mtime": float(i), "source": "r1_outputs/figures"}
        for i in range(160)
    ]
    setattr_facade(
        monkeypatch, "backend.workflow.reporting", "discover_workspace_figures", lambda sid: discovered
    )
    setattr_facade(
        monkeypatch, "backend.workflow.reporting", "compress_image_to_base64", lambda *a, **k: None
    )

    from backend.agents.pi_agent import PIAgent
    pi = PIAgent.__new__(PIAgent)

    class _BoomClient:
        async def chat(self, *a, **k):
            raise RuntimeError("upstream down")

    monkeypatch.setattr(type(pi), "llm_client", property(lambda self: _BoomClient()))
    monkeypatch.setattr(type(pi), "model", property(lambda self: "test-model"))
    monkeypatch.setattr(pi, "_set_llm_context", lambda: None, raising=False)

    reporter = _CaptureReporter("sess-y", discovered, publication_figures=[],
                                pi_agent=pi)
    await reporter._assemble_brief_figures(reporter.analysis_state, {"summary": "s"})

    payload = reporter.sent["integrated_report"]
    assert 0 < payload["total_figures"] <= 20
    assert payload["total_figures"] < len(discovered)


@pytest.mark.asyncio
async def test_invalid_selection_names_self_heal_to_publication(monkeypatch, tmp_path):
    """Defensive layer: if PI selection returns names that match no candidate,
    reporting must not emit an empty gallery -- it falls back to publication
    figures instead of trusting the bogus non-empty selection."""
    discovered = [
        {"name": "umap_real", "type": "umap",
         "path": str(tmp_path / "umap_real.png"), "pdf_path": None,
         "mtime": 1.0, "source": "r1_outputs/figures"},
    ]
    setattr_facade(
        monkeypatch, "backend.workflow.reporting", "discover_workspace_figures", lambda sid: discovered
    )
    setattr_facade(
        monkeypatch, "backend.workflow.reporting", "compress_image_to_base64", lambda *a, **k: None
    )

    reporter = _CaptureReporter(
        "sess-z",
        discovered,
        publication_figures=[_pub_fig("pub_one")],
        pi_return=["totally_hallucinated", "also_fake"],
    )
    await reporter._assemble_brief_figures(reporter.analysis_state, {"summary": "s"})

    payload = reporter.sent["integrated_report"]
    names = {f["name"] for f in payload["figures"]}
    # Bogus names dropped; publication fallback surfaces the real pub figure.
    assert names == {"pub_one"}
    assert payload["total_figures"] == 1
