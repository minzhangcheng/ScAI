"""The notebook is a record of what the kernel ran. Spec 3.10, spec 5.3.

Two things were wrong with it. A cell that raised was discarded at a single
`if`, so the `error` branch `save_notebook` has always carried was unreachable.
And the decision about what to record was spread across fifteen call sites with
no principle behind it, so mutating cells stayed out while the flag's name
suggested they had not mutated anything.

These drive the real `_execute_unlocked` against a real `NotebookState` and read
the `.ipynb` back off disk. A source-text assertion would pass against a gate
that is never reached.
"""
from __future__ import annotations

import json
import pathlib
from queue import Empty as QueueEmpty
from types import SimpleNamespace
import uuid

import pytest

from backend.utils.sandbox_parts import execution as execution_module
from backend.utils.sandbox_parts.notebook_state import NotebookState

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"


# --- a scripted kernel -------------------------------------------------------


def _msgs(*, count: int, fails: bool, text: str | None = None):
    """One execution's iopub traffic, in the order a kernel emits it."""
    out = [{"msg_type": "execute_input", "content": {"execution_count": count}}]
    if text is not None:
        out.append({"msg_type": "stream", "content": {"name": "stdout", "text": text}})
    if fails:
        out.append({
            "msg_type": "error",
            "content": {
                "ename": "ValueError",
                "evalue": "cell %d blew up" % count,
                "traceback": [
                    "Traceback (most recent call last):",
                    "  File \"<ipython-input-%d>\", line 1, in <module>" % count,
                    "ValueError: cell %d blew up" % count,
                ],
            },
        })
    out.append({"msg_type": "status", "content": {"execution_state": "idle"}})
    return out


class _Kernel:
    """Replays a queued script. `execute` is what a rejected cell must not reach."""

    def __init__(self):
        self.queued: list = []
        self.executed: list[str] = []

    def execute(self, code):
        self.executed.append(code)
        return "msg-%d" % len(self.executed)

    def get_iopub_msg(self, *, timeout):
        if not self.queued:
            raise QueueEmpty
        return self.queued.pop(0)


def _sandbox(tmp_path):
    state_dir = tmp_path / "states"
    notebook = NotebookState(
        session_id="notebook-fidelity-%s" % uuid.uuid4().hex[:8],
        state_dir=str(state_dir),
    )
    kernel = _Kernel()

    class _Sandbox(execution_module.ExecutionMixin):
        _kernel_dead = False
        kc = kernel
        km = SimpleNamespace(is_alive=lambda: True, interrupt_kernel=lambda: None)
        timeout = 30.0
        session_id = notebook.session_id
        notebook_state = notebook

        async def _restart_kernel(self):  # pragma: no cover - not reached here
            raise AssertionError("the kernel is alive in these tests")

    return _Sandbox(), kernel, notebook


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    async def direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(execution_module, "_to_thread_drained", direct)
    return _sandbox(tmp_path)


async def _run(box, kernel, code, msgs, **kwargs):
    kernel.queued.extend(msgs)
    return await box._execute_unlocked(code, skip_safety_check=True, **kwargs)


def _cells_on_disk(notebook) -> list[dict]:
    notebook.save_notebook()
    return json.loads(notebook.notebook_path.read_text())["cells"]


def _sources(notebook) -> list[str]:
    """nbformat stores `source` as a list of lines; join it back."""
    out = []
    for cell in _cells_on_disk(notebook):
        source = cell["source"]
        out.append("".join(source) if isinstance(source, list) else source)
    return out


# --- item 1: a failed cell is part of the record -----------------------------


@pytest.mark.asyncio
async def test_a_failed_cell_is_recorded_with_its_traceback(sandbox):
    box, kernel, notebook = sandbox

    result = await _run(box, kernel, "raise ValueError", _msgs(count=1, fails=True))

    assert result.success is False
    cells = _cells_on_disk(notebook)
    assert len(cells) == 1, "the cell the kernel ran was discarded"
    errors = [o for o in cells[0]["outputs"] if o["output_type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["ename"] == "ValueError"
    assert "cell 1 blew up" in errors[0]["evalue"]
    assert any("Traceback" in line for line in errors[0]["traceback"])


@pytest.mark.asyncio
async def test_a_failed_cell_keeps_its_position_between_its_neighbours(sandbox):
    """Ordering, not just presence.

    Appending the failure at the end -- or writing it to a side channel and
    merging later -- would satisfy "the cell is recorded" while making the
    notebook unreplayable, which is the only reason to record it.
    """
    box, kernel, notebook = sandbox

    await _run(box, kernel, "a = 1", _msgs(count=1, fails=False, text="ok"))
    await _run(box, kernel, "raise ValueError", _msgs(count=2, fails=True))
    await _run(box, kernel, "b = 2", _msgs(count=3, fails=False, text="ok"))

    assert _sources(notebook) == ["a = 1", "raise ValueError", "b = 2"]
    kinds = [{o["output_type"] for o in c["outputs"]} for c in _cells_on_disk(notebook)]
    assert kinds[1] == {"stream", "error"} or "error" in kinds[1]
    assert "error" not in kinds[0] and "error" not in kinds[2]


@pytest.mark.asyncio
async def test_a_cell_rejected_by_the_safety_check_is_not_recorded(sandbox):
    """Nothing reached the kernel, so there is nothing to replay.

    This is the one exclusion the unconditional gate depends on: the rejection
    returns before the record, so the gate never has to ask whether the code ran.
    """
    box, kernel, notebook = sandbox

    result = await box._execute_unlocked("shutil.rmtree('/')", skip_safety_check=False)

    assert result.success is False
    assert "blocked" in (result.error or "")
    assert kernel.executed == [], "the kernel saw code the safety check rejected"
    assert _cells_on_disk(notebook) == []


@pytest.mark.asyncio
async def test_a_read_only_probe_stays_out_of_the_record(sandbox):
    """The opt-out still works, and it is the only thing that excludes a cell."""
    box, kernel, notebook = sandbox

    await _run(box, kernel, "adata", _msgs(count=1, fails=False, text="AnnData"),
               records_history=False)
    await _run(box, kernel, "sc.pp.pca(adata)", _msgs(count=2, fails=False))

    assert _sources(notebook) == ["sc.pp.pca(adata)"]


# --- execution_count: the kernel's counter, not the notebook's index ---------


@pytest.mark.asyncio
async def test_recorded_counts_are_the_kernel_counts_and_strictly_increasing(sandbox):
    """Increasing, and deliberately not contiguous.

    A probe that legitimately stays out of the notebook still consumes a kernel
    count. Renumbering to close the gap would make the notebook disagree with
    the kernel transcript it exists to record, so the assertion is on the gap
    surviving.
    """
    box, kernel, notebook = sandbox

    await _run(box, kernel, "a = 1", _msgs(count=1, fails=False))
    await _run(box, kernel, "a", _msgs(count=2, fails=False), records_history=False)
    await _run(box, kernel, "raise ValueError", _msgs(count=3, fails=True))
    await _run(box, kernel, "b = 2", _msgs(count=4, fails=False))

    counts = [c["execution_count"] for c in _cells_on_disk(notebook)]
    assert counts == [1, 3, 4]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)


# --- item 3: the boundary decides, not a list --------------------------------


@pytest.mark.asyncio
async def test_a_resumed_context_carries_the_failures_a_live_one_shows(sandbox):
    """`get_code_history()` returns `notebook_state.cells` -- the same list.

    Revision 1 of the spec claimed the agent's context was a separate structure
    that this change could not move. It is not: `PathsMixin.get_code_history`
    returns the list `add_cell` appends to, and `executor_runtime
    .prepare_sandbox` seeds `agent.code_history` from it on a restored session.
    The live path already appends failed executions unconditionally, so what
    changes is that a *resumed* session stops silently dropping the failures a
    live one always showed. That convergence is the behaviour to pin.
    """
    box, kernel, notebook = sandbox

    await _run(box, kernel, "raise ValueError", _msgs(count=1, fails=True))

    from backend.utils.sandbox_parts.paths import PathsMixin

    history = PathsMixin.get_code_history(box)
    assert history is notebook.cells
    assert [c["code"] for c in history] == ["raise ValueError"]
    evalues = [
        o.get("evalue") for o in history[0]["outputs"] if o.get("type") == "error"
    ]
    assert evalues == ["cell 1 blew up"]


def test_no_call_site_passes_the_retired_save_to_history_flag():
    """`records_history` replaces `save_to_history`; it does not shadow it.

    A keyword argument fails at call time, not import time, so a survivor hides
    in whatever path the suite does not exercise. A blanket replace on this
    branch has already put a keyword argument on a function with no such
    parameter, and the resulting TypeError masked the real diagnosis.
    """
    offenders = []
    for root in ("backend", "clients", "scripts"):
        base = REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            # This guard is the one file that has to name the retired flag.
            if path.resolve() == pathlib.Path(__file__).resolve():
                continue
            if "save_to_history" in path.read_text(errors="ignore"):
                offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], offenders


def test_every_read_only_probe_opts_out_explicitly():
    """The classification, pinned.

    The invariant is the boundary -- everything is recorded unless it says
    otherwise -- and spec 5.3.3 audits the existing sites once. This is that
    audit's result, so a new opt-out has to be argued for here rather than
    added quietly.
    """
    expected = {
        "backend/agents/bioinformatics/executor_priors.py": 3,
        "backend/agents/bioinformatics/exploration_diagnostics.py": 1,
        "backend/agents/bioinformatics/introspection.py": 4,
        "backend/agents/bioinformatics/validation.py": 4,
        "backend/services/cohort_identity.py": 1,
        "backend/utils/sandbox.py": 1,
        "backend/utils/sandbox_parts/kernel.py": 2,
        "backend/utils/sandbox_parts/runtime_restore.py": 1,
        "backend/utils/sandbox_parts/runtime_save.py": 1,
        "backend/utils/sandbox_parts/state_tracking.py": 2,
    }
    observed = {}
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("backend/tests/"):
            continue
        count = path.read_text(errors="ignore").count("records_history=False")
        if count:
            observed[rel] = count

    assert observed == expected, observed


def test_the_mutating_cells_no_longer_opt_out():
    """This is the change, not the rename.

    Six executions that bound names or wrote figure files were invisible to the
    notebook, so a replay from the top hit a NameError on a variable a figure
    cell had bound. `executor_execute` is here because one of its two sites
    calls `obs_names_make_unique()` on every AnnData in globals -- it reads like
    a probe and is not one.
    """
    for rel in (
        "backend/agents/bioinformatics/executor_execute.py",
        "backend/agents/bioinformatics/figure_generation.py",
        "backend/agents/bioinformatics/figure_revision.py",
        "backend/agents/bioinformatics/innovation_baseline_figures.py",
        "backend/agents/bioinformatics/stage1_baseline_figures.py",
        "backend/routes/chat_parts/figures.py",
    ):
        assert "records_history=False" not in (REPO / rel).read_text(), rel
