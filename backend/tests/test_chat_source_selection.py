"""PI chooses Chat AnnData from live facts before any Bioinformatics code."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.routes.chat_parts.source_selection import (
    ChatSourceSelectionError,
    select_chat_source,
)


def _sandbox(*names):
    lock = asyncio.Lock()

    async def inspect():
        assert lock.locked()
        return {
            "canonical_present": "adata" in names,
            "candidates": [{"name": name, "n_obs": 10} for name in names],
        }

    return SimpleNamespace(
        _exec_lock=lock,
        inspect_checkpoint_adata_locked=AsyncMock(side_effect=inspect),
    )


def _decision(name, *, load=False):
    return {
        "uses_adata": True,
        "adata_source": {"mode": "current"},
        "active_adata_var": name,
        "load_registered_inputs": load,
        "rationale": "Matches the requested analysis",
    }


@pytest.mark.asyncio
async def test_pi_selects_exact_live_chat_object_without_loading():
    pi = SimpleNamespace(select_chat_adata_source=AsyncMock(
        return_value=_decision("hepatocytes")
    ))
    state = AnalysisState(user_id="owner")
    result = await select_chat_source(
        pi, _sandbox("adata", "hepatocytes"), state, "Analyze hepatocytes"
    )
    assert result["selection"] == {"mode": "current", "checkpoint_id": None,
                                   "live_binding_id": None}
    assert result["active_adata_var"] == "hepatocytes"
    assert result["candidate_names"] == ["adata", "hepatocytes"]
    assert pi.select_chat_adata_source.await_args.args[1]["candidates"][1]["name"] == "hepatocytes"


@pytest.mark.asyncio
async def test_ambiguous_or_invalid_chat_source_never_defaults_to_adata():
    pi = SimpleNamespace(select_chat_adata_source=AsyncMock(side_effect=[
        _decision(None), _decision("missing"),
    ]))
    with pytest.raises(ChatSourceSelectionError, match="valid AnnData source"):
        await select_chat_source(
            pi, _sandbox("adata", "subset"), AnalysisState(user_id="owner"),
            "Compare the relevant source",
        )
    assert pi.select_chat_adata_source.await_count == 2
    assert "technically invalid" in pi.select_chat_adata_source.await_args.args[4]


@pytest.mark.asyncio
async def test_missing_live_adata_requires_explicit_input_choice():
    pi = SimpleNamespace(select_chat_adata_source=AsyncMock(
        return_value=_decision(None)
    ))
    with pytest.raises(ChatSourceSelectionError, match="valid AnnData source"):
        await select_chat_source(
            pi, _sandbox(), AnalysisState(user_id="owner"),
            "Analyze uploaded cells",
        )


@pytest.mark.asyncio
async def test_pi_can_explicitly_choose_new_registered_inputs():
    pi = SimpleNamespace(select_chat_adata_source=AsyncMock(
        return_value=_decision(None, load=True)
    ))
    result = await select_chat_source(
        pi, _sandbox("adata"), AnalysisState(user_id="owner"),
        "Analyze newly uploaded data",
    )
    assert result["active_adata_var"] is None
    assert result["load_registered_inputs"] is True


@pytest.mark.asyncio
async def test_pi_can_select_non_adata_chat_task():
    pi = SimpleNamespace(select_chat_adata_source=AsyncMock(return_value={
        "uses_adata": False, "adata_source": None, "active_adata_var": None,
        "rationale": "This request uses an existing bulk table",
    }))
    result = await select_chat_source(
        pi, _sandbox("adata", "subset"), AnalysisState(user_id="owner"),
        "Plot the bulk table",
    )
    assert result["uses_adata"] is False
    assert result["active_adata_var"] is None
