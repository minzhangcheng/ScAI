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
        # One `execute()` call, `build_innovation_baseline_probe_code(...)`,
        # already carrying `mutates_namespace=False`. A2-9 classified this file
        # as mutation-capable from its name and removed the opt-out; the merge
        # with main surfaced the mistake.
        "backend/agents/bioinformatics/innovation_baseline_figures.py": 1,
        "backend/agents/bioinformatics/introspection.py": 4,
        "backend/agents/bioinformatics/validation.py": 4,
        "backend/services/cohort_identity.py": 2,
        "backend/utils/sandbox.py": 1,
        # Trusted checkpoint serialization, registry queries, source
        # application, and reconciliation are Host bookkeeping. They must not
        # be replayed as user-authored notebook cells.
        "backend/utils/sandbox_parts/adata_checkpoints.py": 6,
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
        "backend/agents/bioinformatics/stage1_baseline_figures.py",
        "backend/routes/chat_parts/figures.py",
    ):
        assert "records_history=False" not in (REPO / rel).read_text(), rel


# --- the kernel bootstrap is part of the record (spec 5.3.2) ----------------


def _bootstrap(notebook, *, common="import scanpy\nSEED = 0", paths="os.chdir('/w')"):
    notebook.set_bootstrap_cells([
        {"code": common, "outputs": [
            {"type": "stream", "name": "stdout", "text": "Environment initialized"}],
         "execution_count": 1},
        {"code": paths, "outputs": [], "execution_count": 2},
    ])


@pytest.mark.asyncio
async def test_the_notebook_starts_with_the_kernel_bootstrap(sandbox):
    """Cell 0 is the environment, not the first thing the analyst typed.

    Without it the notebook replays into whatever environment the reader
    happens to have: a different seed, a different working directory, none of
    the imports. The cells that follow all assume it.
    """
    box, kernel, notebook = sandbox
    _bootstrap(notebook)

    await _run(box, kernel, "sc.pp.pca(adata)", _msgs(count=3, fails=False))

    assert _sources(notebook)[0].startswith("import scanpy")
    assert _sources(notebook) == [
        "import scanpy\nSEED = 0", "os.chdir('/w')", "sc.pp.pca(adata)",
    ]


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


@pytest.mark.asyncio
async def test_a_kernel_restart_does_not_add_a_second_bootstrap(sandbox):
    """`set_bootstrap_cells` replaces. The environment is recorded once."""
    box, kernel, notebook = sandbox
    _bootstrap(notebook)
    await _run(box, kernel, "a = 1", _msgs(count=3, fails=False))
    _bootstrap(notebook)

    assert _sources(notebook) == [
        "import scanpy\nSEED = 0", "os.chdir('/w')", "a = 1",
    ]


@pytest.mark.asyncio
async def test_setting_the_bootstrap_does_not_write_the_notebook(sandbox):
    """The ordering trap, pinned.

    `_initialize_kernel` runs before `_restore_session` calls
    `notebook_state.load()`. A save here would put a two-cell notebook over a
    resumed session's history and restore would read that back.
    """
    box, kernel, notebook = sandbox
    await _run(box, kernel, "a = 1", _msgs(count=1, fails=False))
    notebook.save_notebook()
    before = notebook.notebook_path.read_bytes()

    _bootstrap(notebook)

    assert notebook.notebook_path.read_bytes() == before


@pytest.mark.asyncio
async def test_the_notebook_is_still_byte_stable_across_repeated_saves(sandbox):
    """Ids derive from `sha256(index:code)` so an unchanged notebook is
    unchanged on disk. Prepending the bootstrap shifts every id once -- it must
    not make each save look different from the last."""
    box, kernel, notebook = sandbox
    _bootstrap(notebook)
    await _run(box, kernel, "a = 1", _msgs(count=3, fails=False))
    await _run(box, kernel, "raise ValueError", _msgs(count=4, fails=True))

    notebook.save_notebook()
    first = notebook.notebook_path.read_bytes()
    notebook.save_notebook()
    assert notebook.notebook_path.read_bytes() == first


@pytest.mark.asyncio
async def test_an_upgraded_session_gains_a_bootstrap_rather_than_losing_one(sandbox):
    """A notebook written before this change has no bootstrap-marked cell.

    `load` must leave the live one in place instead of clearing it, or a
    resumed pre-upgrade session would stay without an environment forever.
    """
    box, kernel, notebook = sandbox
    await _run(box, kernel, "a = 1", _msgs(count=1, fails=False))
    notebook.save_notebook()

    resumed = NotebookState(
        session_id=notebook.session_id, state_dir=str(notebook.state_dir)
    )
    _bootstrap(resumed)          # this process's kernel start
    assert resumed.load() is True  # then restore reads the old notebook

    assert [c["code"] for c in resumed.bootstrap_cells] == [
        "import scanpy\nSEED = 0", "os.chdir('/w')",
    ]
    assert [c["code"] for c in resumed.cells] == ["a = 1"]


def test_the_bootstrap_executions_stay_out_of_the_ordinary_record_gate():
    """They are recorded, but not through `add_cell`.

    The gate saves, and `_initialize_kernel` runs before restore has read the
    notebook, so routing them through it is the overwrite this design avoids.
    """
    kernel_source = (BACKEND / "utils/sandbox_parts/kernel.py").read_text()
    assert kernel_source.count("records_history=False") == 2
    assert "set_bootstrap_cells([" in kernel_source


# --- what the executor is told to generate (spec 5.3.4, 5.3.5) --------------
#
# These assert the prompt text, which is all a test can assert. They do not
# prove the model complies; spec section 8 records that as an accepted risk,
# and it is why this batch's prompt changes are committed alone.


def _workspace_prompt():
    from backend.agents.bioinformatics.executor_prompt import (
        _executor_prompt_workspace,
    )

    return _executor_prompt_workspace()


def test_the_prompt_cannot_name_the_session_root_because_it_is_not_given_one():
    """The absolute path was in the instructions, so it ended up in the code.

    One real notebook held 630 absolute-path occurrences while the prompt
    already forbade them -- and the prompt was also handing the model the exact
    string to paste. A path a cell hard-codes cannot be replayed anywhere else,
    not even in the same installation's clean workspace.

    Asserted structurally rather than by passing a path and looking for it: the
    section takes no workspace argument now, so there is nothing to interpolate
    and no way to reintroduce one without changing the signature.
    """
    import inspect

    from backend.agents.bioinformatics.executor_prompt import (
        _executor_prompt_workspace,
        build_static_executor_prompt,
    )

    assert inspect.signature(_executor_prompt_workspace).parameters == {}
    assert "workspace" not in inspect.signature(build_static_executor_prompt).parameters
    assert "_WORKSPACE_DIR" in _workspace_prompt()


def test_the_prompt_points_at_the_notebook_cell_that_defines_the_root():
    """`_WORKSPACE_DIR` is only a portable root if its definition travels.

    It is set by the kernel bootstrap, which became notebook cells 0 and 1 in
    this batch. Before that a notebook using the variable was no more replayable
    than one hard-coding the path.
    """
    lowered = _workspace_prompt().lower()
    assert "first cell" in lowered or "opening cell" in lowered or "bootstrap" in lowered


def test_the_prompt_forbids_random_state_none():
    """"Never omit it" is satisfied by passing None, which is not determinism.

    `sc.tl.umap(adata, random_state=None)` has the parameter, reads as compliant
    to any check that looks for its presence, and reseeds from entropy on every
    run -- so the replay contract's embedding tolerance can never be met.
    """
    prompt = _workspace_prompt()
    assert "random_state=None" in prompt, "the failure mode is not named"
    assert "seed=None" in prompt


def test_the_prompt_still_requires_a_fixed_seed():
    """The existing instruction is not weakened by the new one."""
    prompt = _workspace_prompt()
    assert "random_state=42" in prompt
    assert "NEVER omit the random_state parameter" in prompt


def test_the_templates_emit_no_absolute_paths():
    """The prompt is advice; a template is code we ship.

    A template that hard-coded a path would put one in every notebook regardless
    of what the model was told.
    """
    import re

    for rel in (
        "backend/agents/bioinformatics/kernel_templates.py",
        "backend/agents/bioinformatics/kernel_template_sections.py",
    ):
        source = (REPO / rel).read_text()
        literals = re.findall(r"""['"](/(?:data|workspace|home|tmp)/[^'"]*)['"]""", source)
        assert literals == [], (rel, literals)
