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
async def test_a_read_only_probe_stays_out_of_the_record(sandbox):
    """The opt-out still works, and it is the only thing that excludes a cell."""
    box, kernel, notebook = sandbox

    await _run(box, kernel, "adata", _msgs(count=1, fails=False, text="AnnData"),
               records_history=False)
    await _run(box, kernel, "sc.pp.pca(adata)", _msgs(count=2, fails=False))

    assert _sources(notebook) == ["sc.pp.pca(adata)"]


# --- execution_count: the kernel's counter, not the notebook's index ---------


# --- item 3: the boundary decides, not a list --------------------------------


# --- the kernel bootstrap is part of the record (spec 5.3.2) ----------------


def _bootstrap(notebook, *, common="import scanpy\nSEED = 0", paths="os.chdir('/w')"):
    notebook.set_bootstrap_cells([
        {"code": common, "outputs": [
            {"type": "stream", "name": "stdout", "text": "Environment initialized"}],
         "execution_count": 1},
        {"code": paths, "outputs": [], "execution_count": 2},
    ])


@pytest.mark.asyncio
async def test_the_bootstrap_survives_a_save_and_load_round_trip(sandbox):
    """It has to come back as the bootstrap, not as an ordinary first cell.

    If `load` read it into `cells`, the next kernel start would prepend a second
    copy, and every resume would add another.
    """
    box, kernel, notebook = sandbox
    _bootstrap(notebook)
    await _run(box, kernel, "a = 1", _msgs(count=3, fails=False))
    notebook.save_notebook()

    reloaded = NotebookState(
        session_id=notebook.session_id, state_dir=str(notebook.state_dir)
    )
    assert reloaded.load() is True
    assert [c["code"] for c in reloaded.bootstrap_cells] == [
        "import scanpy\nSEED = 0", "os.chdir('/w')",
    ]
    assert [c["code"] for c in reloaded.cells] == ["a = 1"]
    assert _sources(reloaded) == _sources(notebook)


# --- what the executor is told to generate (spec 5.3.4, 5.3.5) --------------
#
# These assert the prompt text, which is all a test can assert. They do not
# prove the model complies; spec section 8 records that as an accepted risk,
# and it is why this batch's prompt changes are committed alone.
