"""safe_pi_brief 在 brief 为空时应记录日志并返回 None。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import logging
from backend.workflow.helpers import safe_pi_brief


def test_empty_brief_returns_none_and_logs(caplog):
    async def make():
        return "   "

    with caplog.at_level(logging.INFO):
        result = asyncio.run(safe_pi_brief(make, "sess1", "round"))
    assert result is None
    assert any("empty" in r.message.lower() for r in caplog.records)
