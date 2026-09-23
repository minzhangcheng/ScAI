"""Expression-contract enforcement for PI-generated figure revisions."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from backend.models.schemas import AnalysisState
from backend.tests.test_figure_revision_gate import (
    CLEAN,
    _Agent,
    _FakeSandbox,
    _approved_prior_figure,
    _review,
    _reviews,
)


class _RecordingSandbox(_FakeSandbox):
    def __init__(self, working_dir: Path, script):
        super().__init__(working_dir, script)
        self.executed_code: list[str] = []
        self._frozen_pair: tuple[bytes, bytes] | None = None

    async def execute(self, code, **kwargs):
        self.executed_code.append(code)
        result = await super().execute(code, **kwargs)
        jpg_match = re.search(r"^\s*_jpg_path = (.+)$", code, re.MULTILINE)
        pdf_match = re.search(r"^\s*_pdf_path = (.+)$", code, re.MULTILINE)
        jpg_path = Path(ast.literal_eval(jpg_match.group(1)))
        pdf_path = Path(ast.literal_eval(pdf_match.group(1)))
        if self._frozen_pair is None:
            self._frozen_pair = (jpg_path.read_bytes(), pdf_path.read_bytes())
        else:
            jpg_path.write_bytes(self._frozen_pair[0])
            pdf_path.write_bytes(self._frozen_pair[1])
        return result


class _ExpressionRevisionAgent(_Agent):
    async def _revise_figure_code(self, *args, **kwargs):
        self.revision_calls.append({"requested": True})
        return (
            "sc.pp.log1p(adata)\n"
            "result = NMF().fit_transform(raw).sum()\n"
            "plot()"
        )


@pytest.mark.asyncio
async def test_revision_candidate_is_guarded_and_warns_old_generation(tmp_path):
    prior, _old_bytes = _approved_prior_figure(tmp_path)
    prior.round_number = 1
    sandbox = _RecordingSandbox(tmp_path, [CLEAN, CLEAN])
    agent = _ExpressionRevisionAgent(sandbox)
    state = AnalysisState(
        user_id="u",
        current_round=1,
        publication_figures=[prior],
    )

    result = await agent._handle_generate_publication_figure(
        code="plot()",
        figure_name="f",
        logical_figure_id="new-logical-result",
        description="candidate with a mutating PI revision",
        figure_type="heatmap",
        state=state,
        review_callback=_reviews(
            _review(approved=False, score=6),
            _review(approved=True, score=9),
        ),
    )

    assert "approved" in result
    assert len(sandbox.executed_code) == 2
    revised_wrapper = sandbox.executed_code[1]
    assert revised_wrapper.count("_scai_expression_contract_target(") >= 2
    assert "NMF().fit_transform(raw).sum()" not in revised_wrapper
    assert prior.is_current is True
    assert "representation_change_warning" in prior.expression_provenance
    assert state.sandbox_state["analysis_generation"] == 1
    revised = state.publication_figures[-1]
    assert revised.is_current is True
    assert revised.analysis_generation == 1
