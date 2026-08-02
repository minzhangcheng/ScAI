"""Report WebSockets use the generation-aware serialized delivery registry."""


def test_report_ws_endpoint_uses_shared_delivery_registry():
    import backend.routes.websockets as ws

    assert hasattr(ws, "register_report_websocket")
    assert hasattr(ws, "unregister_websocket")
