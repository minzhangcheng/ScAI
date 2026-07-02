"""Test that report_websockets has proper locking."""
import pytest


def test_report_ws_registry_has_lock():
    import backend.routes.websockets as ws
    assert hasattr(ws, "report_ws_lock")
