"""Regression tests for removing the output-token cap (and the deadline that
replaces it).

Both halves matter and neither substitutes for the other:

*   A thinking model spends its output budget on private reasoning first, so any
    cap we set is a cap on thinking.  When it runs out the provider returns
    ``status=incomplete`` with an empty answer channel and the whole generation
    is wasted -- observed 11 times in one run against a single model, while 60
    uncapped calls to that same model failed 0 times.
*   Removing the cap removes the only bound on how long one call can run: the
    per-event idle timeout never fires while a runaway chain keeps emitting
    deltas.  ``_StartedStream`` therefore carries a *total* deadline, matching
    the non-streaming path's request timeout.
"""
import asyncio
import time

import pytest

from backend.utils.llm_client_parts.adapters import _StartedStream, _start_stream
from backend.utils.llm_client_parts.errors import StreamInterruptedError


# ── half 1: no output cap reaches the wire ───────────────────────────────────

class TestNoOutputCapOnTheWire:
    def test_responses_adapter_omits_max_output_tokens(self):
        """``max_tokens=None`` must omit the field, not send a default."""
        from backend.utils.llm_client_parts.adapters import _ResponsesAdapter

        adapter = _ResponsesAdapter.__new__(_ResponsesAdapter)
        params = _ResponsesAdapter._build_params(
            adapter,
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=None,
        )
        assert "max_output_tokens" not in params

        capped = _ResponsesAdapter._build_params(
            adapter,
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=32768,
        )
        assert capped["max_output_tokens"] == 32768

    def test_agent_config_default_is_uncapped(self):
        from backend.agents.base import AgentConfig

        assert AgentConfig().max_tokens is None

    @pytest.mark.parametrize(
        "module_path",
        [
            "backend/agents/bioinformatics/figures.py",
            "backend/agents/bioinformatics/executor_loop.py",
            "backend/agents/bioinformatics/profiling.py",
            "backend/agents/supervisor/evaluation.py",
            "backend/agents/supervisor/figure_binding.py",
            "backend/agents/supervisor/diagrams.py",
            "backend/agents/pi/figures.py",
            "backend/agents/technique_agent.py",
            "backend/agents/biochemistry_agent.py",
            "backend/services/report/llm_calls.py",
            "backend/services/report/pi_review.py",
            "backend/services/report/planning.py",
            "backend/workflow/callbacks.py",
        ],
    )
    def test_no_output_cap_at_call_sites(self, module_path):
        """Every max_tokens these modules supply must be the no-cap sentinel.

        AST-level rather than a text scan, so a *named* constant holding a
        number is caught too -- the caps this change removed were written both
        ways (a bare 32768 and a REASONING_MODEL_MIN_TOKENS that was 4096), and
        a text scan for digits silently passes the second form.

        Source-level rather than behavioural because the defect is the value at
        the call site: any one of these can reintroduce a cap without a test
        that exercises it noticing.
        """
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        tree = ast.parse((repo_root / module_path).read_text())

        # A pass-through forwards a value chosen elsewhere rather than picking
        # a cap here: a parameter of the same name (its caller is covered by its
        # own entry in this table), or `<budgeted dispatch>.max_tokens`, which
        # is None unless the context window is too small to hold the reserve.
        allowed = {"NO_OUTPUT_TOKEN_CAP", "max_tokens"}

        def describe(node):
            if node is None:
                return None
            if isinstance(node, ast.Constant) and node.value is None:
                return None
            if isinstance(node, ast.Name) and node.id in allowed:
                return None
            if isinstance(node, ast.Attribute) and node.attr == "max_tokens":
                return None
            return ast.dump(node)

        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "max_tokens":
                bad = describe(node.value)
                if bad:
                    offenders.append((node.value.lineno, bad))
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "max_tokens":
                        bad = describe(value)
                        if bad:
                            offenders.append((value.lineno, bad))
            elif isinstance(node, ast.arguments):
                defaults = dict(
                    zip(
                        [a.arg for a in node.args][-len(node.defaults):] if node.defaults else [],
                        node.defaults,
                    )
                )
                defaults.update(
                    {a.arg: d for a, d in zip(node.kwonlyargs, node.kw_defaults)}
                )
                if "max_tokens" in defaults:
                    bad = describe(defaults["max_tokens"])
                    if bad:
                        offenders.append((defaults["max_tokens"].lineno, bad))

        assert offenders == [], (
            f"{module_path} supplies an output cap at {offenders}; "
            "use NO_OUTPUT_TOKEN_CAP"
        )


# ── half 2: the wall-clock deadline that replaces the cap ────────────────────

class _SlowIterator:
    """Yields forever, fast enough that the idle bound can never fire."""

    def __init__(self, interval):
        self.interval = interval
        self.count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self.interval)
        self.count += 1
        return f"event-{self.count}"


class _FakeStream:
    def __init__(self, iterator):
        self._iterator = iterator
        self.closed = False

    def __aiter__(self):
        return self._iterator

    async def close(self):
        self.closed = True


class TestStreamTotalDeadline:
    @pytest.mark.asyncio
    async def test_events_stop_at_the_total_deadline(self):
        """A chain that keeps streaming must still be cut off.

        The idle bound is 10x the total budget here, so only the total deadline
        can end this stream. Without it the loop never terminates.
        """
        started = _StartedStream(
            stream=_FakeStream(None),
            iterator=_SlowIterator(0.01),
            first_event="event-0",
            consumer=None,
            idle_timeout=5.0,
            label="test stream",
            deadline=time.monotonic() + 0.15,
            total_timeout=0.15,
        )

        seen = []
        with pytest.raises(StreamInterruptedError) as exc_info:
            async for event in started.events():
                seen.append(event)

        assert "deadline" in str(exc_info.value)
        # It really streamed before being cut off, rather than failing at once.
        assert len(seen) > 1

    @pytest.mark.asyncio
    async def test_stalled_stream_still_reports_its_cause(self):
        """A stall must keep reporting the underlying timeout, as before.

        The deadline and the idle bound are the same value in production (see
        _start_stream), so a stalled stream trips the deadline clamp rather than
        the idle allowance.  It must not lose the cause in the process -- the
        dispatch layer and its tests read the exception type out of this text.
        """
        started = _StartedStream(
            stream=_FakeStream(None),
            iterator=_SlowIterator(10.0),
            first_event="event-0",
            consumer=None,
            idle_timeout=0.05,
            label="test stream",
            deadline=time.monotonic() + 0.05,
            total_timeout=0.05,
        )

        with pytest.raises(StreamInterruptedError) as exc_info:
            async for _event in started.events():
                pass

        assert "TimeoutError" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_start_stream_arms_the_deadline(self):
        """The clock starts with the request, not with the first event."""
        stream = _FakeStream(_SlowIterator(0.0))
        before = time.monotonic()
        started = await _start_stream(
            stream, consumer=None, timeout=7.0, label="test stream"
        )
        after = time.monotonic()

        assert started.total_timeout == 7.0
        assert before + 7.0 <= started.deadline <= after + 7.0

    @pytest.mark.asyncio
    async def test_unbounded_stream_has_no_deadline(self):
        """A _StartedStream built without a deadline keeps the old behaviour."""
        started = _StartedStream(
            stream=_FakeStream(None),
            iterator=_SlowIterator(0.0),
            first_event="event-0",
            consumer=None,
            idle_timeout=5.0,
            label="test stream",
        )
        assert started.deadline is None
        assert started._remaining() is None
