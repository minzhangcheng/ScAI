import json
from types import SimpleNamespace

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AnalysisState, TreatmentGroup


class _FakeSandbox:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.codes = []

    async def execute(self, code, save_to_history=True, skip_safety_check=False):
        self.codes.append(code)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(success=True, output=output, error=None)


def _agent(sandbox):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox
    a.session_id = "s"
    return a


def _state():
    state = AnalysisState(user_id="test")
    state.treatment_groups = [
        TreatmentGroup(id="nash", name="NASH", order=2),
        TreatmentGroup(id="control", name="Control", order=1),
    ]
    return state


def _state_many_groups():
    state = AnalysisState(user_id="test")
    state.treatment_groups = [
        TreatmentGroup(id="g1", name="G1", order=1),
        TreatmentGroup(id="g2", name="G2", order=2),
        TreatmentGroup(id="g3", name="G3", order=3),
        TreatmentGroup(id="g4", name="G4", order=4),
        TreatmentGroup(id="g5", name="G5", order=5),
        TreatmentGroup(id="g6", name="G6", order=6),
    ]
    return state


def _metadata(condition_like=None, sample_like=None, batch_like=None):
    return json.dumps({
        "var": "adata",
        "cluster_key": "leiden",
        "condition_like": condition_like or [],
        "sample_like": sample_like or [],
        "batch_like": batch_like or [],
        "obs_summaries": {
            "condition": {"n_categories": 2, "top": ["NASH", "Control"]},
            "sample": {"n_categories": 4, "top": ["S1", "S2", "S3", "S4"]},
        },
    })


@pytest.mark.asyncio
async def test_compute_cluster_group_proportions_uses_condition_and_treatment_order():
    crosstab = json.dumps({
        "group_labels": ["Control", "NASH"],
        "proportions": {
            "0": {"Control": 40.0, "NASH": 10.0},
            "1": {"Control": 60.0, "NASH": 90.0},
        },
        "column_used": "condition",
        "dimension": "group",
    })
    sandbox = _FakeSandbox([_metadata(condition_like=["condition"]), crosstab])
    result = await _agent(sandbox).compute_cluster_group_proportions(_state())

    assert result["group_labels"] == ["Control", "NASH"]
    assert result["dimension"] == "group"
    assert result["column_used"] == "condition"
    assert result["proportions"]["0"] == {"Control": 40.0, "NASH": 10.0}
    assert "normalize='columns'" in sandbox.codes[1]
    assert "_group_col = 'condition'" in sandbox.codes[1]
    assert "_ordered_labels = ['Control', 'NASH']" in sandbox.codes[1]


@pytest.mark.asyncio
async def test_compute_cluster_group_proportions_prefers_matching_condition_column():
    crosstab = json.dumps({
        "group_labels": ["Control", "NASH"],
        "proportions": {"0": {"Control": 25.0, "NASH": 75.0}},
        "column_used": "treatment",
        "dimension": "group",
    })
    sandbox = _FakeSandbox([
        json.dumps({
            "var": "adata",
            "cluster_key": "leiden",
            "condition_like": ["disease", "treatment"],
            "sample_like": [],
            "batch_like": [],
            "obs_summaries": {
                "disease": {"n_categories": 2, "top": ["case", "control"]},
                "treatment": {"n_categories": 2, "top": ["NASH", "Control"]},
            },
        }),
        crosstab,
    ])
    result = await _agent(sandbox).compute_cluster_group_proportions(_state())

    assert result["group_labels"] == ["Control", "NASH"]
    assert result["dimension"] == "group"
    assert result["column_used"] == "treatment"
    assert "_group_col = 'treatment'" in sandbox.codes[1]
    assert "_ordered_labels = ['Control', 'NASH']" in sandbox.codes[1]


@pytest.mark.asyncio
async def test_compute_cluster_group_proportions_falls_back_to_overall_when_condition_labels_do_not_match_groups():
    overall = json.dumps({
        "group_labels": ["整体"],
        "proportions": {"0": {"整体": 100.0}},
        "column_used": None,
        "dimension": "overall",
    })
    sandbox = _FakeSandbox([
        json.dumps({
            "var": "adata",
            "cluster_key": "leiden",
            "condition_like": ["condition"],
            "sample_like": [],
            "batch_like": [],
            "obs_summaries": {
                "condition": {"n_categories": 2, "top": ["case", "ctrl"]},
            },
        }),
        overall,
    ])
    result = await _agent(sandbox).compute_cluster_group_proportions(_state())

    assert result["group_labels"] == ["整体"]
    assert result["dimension"] == "overall"
    assert result["column_used"] is None
    assert "_group_col = None" in sandbox.codes[1]


def test_select_proportion_grouping_partial_label_match_falls_back_to_overall():
    """A condition column where only some categories map to treatment groups
    (e.g. ["case", "Control"]) is disease/status metadata, not the experimental
    grouping, so it must fall back to overall rather than be picked as group."""
    tg = [
        TreatmentGroup(id="c", name="Control", order=1),
        TreatmentGroup(id="n", name="NASH", order=2),
    ]
    meta = {
        "condition_like": ["condition"],
        "obs_summaries": {"condition": {"n_categories": 2, "top": ["case", "Control"]}},
    }
    grouping = BioinformaticsAgent._select_proportion_grouping(meta, tg)
    assert grouping["dimension"] == "overall"
    assert grouping["column"] is None


@pytest.mark.asyncio
async def test_compute_cluster_group_proportions_appends_crosstab_columns_missing_from_truncated_top_labels():
    crosstab = json.dumps({
        "group_labels": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "proportions": {"0": {"G1": 10.0, "G2": 20.0, "G3": 30.0, "G4": 40.0, "G5": 50.0, "G6": 60.0}},
        "column_used": "condition",
        "dimension": "group",
    })
    sandbox = _FakeSandbox([
        json.dumps({
            "var": "adata",
            "cluster_key": "leiden",
            "condition_like": ["condition"],
            "sample_like": [],
            "batch_like": [],
            "obs_summaries": {
                "condition": {"n_categories": 6, "top": ["G1", "G2", "G3", "G4", "G5"]},
            },
        }),
        crosstab,
    ])
    result = await _agent(sandbox).compute_cluster_group_proportions(_state_many_groups())

    assert result["group_labels"] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert "_labels += sorted(_c for _c in _present if _c not in _labels)" in sandbox.codes[1]


@pytest.mark.asyncio
async def test_compute_cluster_group_proportions_falls_back_to_overall_without_condition():
    overall = json.dumps({
        "group_labels": ["整体"],
        "proportions": {"0": {"整体": 33.3}, "1": {"整体": 66.7}},
        "column_used": None,
        "dimension": "overall",
    })
    sandbox = _FakeSandbox([_metadata(condition_like=[], sample_like=["sample"]), overall])
    result = await _agent(sandbox).compute_cluster_group_proportions(_state())

    assert result["group_labels"] == ["整体"]
    assert result["dimension"] == "overall"
    assert result["column_used"] is None
    assert "_group_col = None" in sandbox.codes[1]
    assert "value_counts(normalize=True)" in sandbox.codes[1]
    assert "sample" not in sandbox.codes[1]


@pytest.mark.asyncio
async def test_compute_cluster_group_proportions_returns_empty_on_missing_sandbox_or_errors():
    assert await _agent(None).compute_cluster_group_proportions(_state()) == {
        "group_labels": [],
        "proportions": {},
        "column_used": None,
        "dimension": "group",
    }

    sandbox = _FakeSandbox([RuntimeError("boom")])
    assert await _agent(sandbox).compute_cluster_group_proportions(_state()) == {
        "group_labels": [],
        "proportions": {},
        "column_used": None,
        "dimension": "group",
    }
