"""Kernel must recover from ZMQ signature corruption during normal execution,
not just from timeout-path drain failures."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.utils.sandbox import LocalJupyterSandbox, _is_kernel_protocol_error


def test_protocol_error_classifier():
    assert _is_kernel_protocol_error(ValueError("Invalid Signature: b'abc'"))
    assert _is_kernel_protocol_error(ValueError("b'<IDS|MSG>' is not in list"))
    assert not _is_kernel_protocol_error(ValueError("some unrelated error"))
    assert not _is_kernel_protocol_error(KeyError("x"))


@pytest.mark.asyncio
async def test_execute_restarts_kernel_on_signature_error(monkeypatch):
    sb = object.__new__(LocalJupyterSandbox)        # bypass heavy __init__
    sb.session_id = "s"
    sb._kernel_dead = False
    sb.km = MagicMock()
    sb.kc = MagicMock()
    sb.kc.execute = MagicMock(side_effect=ValueError("Invalid Signature: b'zz'"))

    restarted = {"n": 0}

    async def fake_restart():
        restarted["n"] += 1

    sb._restart_kernel = fake_restart

    result = await sb.execute("print(1)", skip_safety_check=True)
    assert result.success is False
    assert restarted["n"] == 1  # signature error triggered a restart attempt