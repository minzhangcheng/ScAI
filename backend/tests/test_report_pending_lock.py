import asyncio
import pytest


@pytest.mark.asyncio
async def test_no_double_generation():
    from backend.routes.reports import _try_begin_generation, clear_generation_pending

    clear_generation_pending("s")
    a, b = await asyncio.gather(_try_begin_generation("s"), _try_begin_generation("s"))
    assert sum(value is None for value in (a, b)) == 1
    assert sum(isinstance(value, str) for value in (a, b)) == 1


@pytest.mark.asyncio
async def test_sequential_generation_allowed():
    from backend.routes.reports import _try_begin_generation, clear_generation_pending

    clear_generation_pending("t")
    first = await _try_begin_generation("t")
    assert isinstance(first, str)
    assert await _try_begin_generation("t") is None
    clear_generation_pending("t", first)
    assert isinstance(await _try_begin_generation("t"), str)


@pytest.mark.asyncio
async def test_old_owner_cannot_clear_reused_session_marker():
    from backend.routes.reports import (
        _report_generation_pending,
        _try_begin_generation,
        clear_generation_pending,
    )

    clear_generation_pending("aba")
    old_token = await _try_begin_generation("aba")
    assert isinstance(old_token, str)

    # Lifecycle deletion clears the old generation without knowing its token;
    # a reused generation then installs a distinct owner.
    clear_generation_pending("aba")
    new_token = await _try_begin_generation("aba")
    assert isinstance(new_token, str) and new_token != old_token

    clear_generation_pending("aba", old_token)
    assert _report_generation_pending.get("aba") == new_token
    clear_generation_pending("aba", new_token)
