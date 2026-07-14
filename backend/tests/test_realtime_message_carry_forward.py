"""Regression tests for realtime-message stream survival across state-object swaps.

Root cause of "realtime messages vanish on refresh": the running workflow replaces
sessions[session_id] with fresh AnalysisState objects (LangGraph checkpoint
round-trips), while the WebSocket message_callback historically appended to the
object captured at handler entry. The stream stranded on an orphaned object, so
/load and reconnect only ever served the messages that predated the first swap.

The fix appends to the *currently cached* object and carries the fuller stream
forward on each swap via _preserve_realtime_stream.
"""
from backend.models.schemas import AnalysisState, WebSocketMessage
from backend.routes.websockets import _preserve_realtime_stream


def _msg(content: str) -> WebSocketMessage:
    return WebSocketMessage(type="status", agent=None, content=content)


def test_carry_forward_moves_longer_stream_onto_incoming():
    prev = AnalysisState(user_id="u")
    prev.realtime_messages = [_msg("a"), _msg("b"), _msg("c")]
    incoming = AnalysisState(user_id="u")
    incoming.realtime_messages = [_msg("a")]  # stale graph copy

    _preserve_realtime_stream(prev, incoming)

    assert [m.content for m in incoming.realtime_messages] == ["a", "b", "c"]


def test_carry_forward_noop_when_same_object():
    state = AnalysisState(user_id="u")
    state.realtime_messages = [_msg("a"), _msg("b")]

    _preserve_realtime_stream(state, state)

    assert [m.content for m in state.realtime_messages] == ["a", "b"]


def test_carry_forward_noop_when_incoming_is_longer():
    prev = AnalysisState(user_id="u")
    prev.realtime_messages = [_msg("a")]
    incoming = AnalysisState(user_id="u")
    incoming.realtime_messages = [_msg("a"), _msg("b")]

    _preserve_realtime_stream(prev, incoming)

    assert [m.content for m in incoming.realtime_messages] == ["a", "b"]


def test_carry_forward_tolerates_missing_prev():
    incoming = AnalysisState(user_id="u")
    incoming.realtime_messages = [_msg("a")]

    _preserve_realtime_stream(None, incoming)

    assert [m.content for m in incoming.realtime_messages] == ["a"]


def test_swap_sequence_preserves_full_stream():
    """Simulate the real flow: append to the cached object, then swap to a fresh
    graph object each round, carrying the stream forward. Nothing should be lost.
    """
    cached = AnalysisState(user_id="u")
    cached.realtime_messages = []

    # Round 1: two messages land on the cached object.
    cached.realtime_messages.append(_msg("init"))
    cached.realtime_messages.append(_msg("round1-step1"))

    # state_callback fires with a fresh (stale) graph object.
    graph_obj_1 = AnalysisState(user_id="u")
    graph_obj_1.realtime_messages = [_msg("init")]  # seeded before appends
    _preserve_realtime_stream(cached, graph_obj_1)
    cached = graph_obj_1  # becomes canonical

    # Round 2: more messages append to the now-canonical object.
    cached.realtime_messages.append(_msg("round2-step1"))

    graph_obj_2 = AnalysisState(user_id="u")
    graph_obj_2.realtime_messages = [_msg("init")]
    _preserve_realtime_stream(cached, graph_obj_2)
    cached = graph_obj_2

    assert [m.content for m in cached.realtime_messages] == [
        "init", "round1-step1", "round2-step1",
    ]
