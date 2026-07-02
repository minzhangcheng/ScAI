import asyncio
import pytest


@pytest.mark.asyncio
async def test_no_double_generation():
    from backend.routes.reports import _try_begin_generation, clear_generation_pending

    clear_generation_pending("s")
    a, b = await asyncio.gather(_try_begin_generation("s"), _try_begin_generation("s"))
    assert {a, b} == {True, False}


@pytest.mark.asyncio
async def test_sequential_generation_allowed():
    from backend.routes.reports import _try_begin_generation, clear_generation_pending

    clear_generation_pending("t")
    assert await _try_begin_generation("t") is True
    assert await _try_begin_generation("t") is False
    clear_generation_pending("t")
    assert await _try_begin_generation("t") is True
