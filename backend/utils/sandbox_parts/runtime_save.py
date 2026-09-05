"""Runtime save phases for LocalJupyterSandbox."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Any, Dict


@dataclass
class SaveContext:
    sandbox: Any
    restore_failed: bool
    revision: int
    expected_lifecycle_generation: int
    required_names: set[str]
    escaped_required: str
    escaped_path: str
    record_generation_failure: Any
    restore_previous_persistence: Any
    SessionGenerationError: type[BaseException]
    SessionGoneError: type[BaseException]
    assert_session_generation_active: Any
    lifecycle_file_lock: Any
    # Defaulted so a caller building a context by hand still produces valid
    # kernel code; _build_save_code substitutes the canonical tuple when empty.
    escaped_volatile_prefixes: str = ""


class RequiredValueDemoted(Exception):
    """A deterministically unpicklable required name was released mid-save.

    Raised only after the name has left the tracker's required set, so the
    retry that follows builds a different dump instead of repeating one that is
    guaranteed to fail again. ``names`` travels with it because attribution
    must survive the release: once a name is out of the required set, the
    failure record can no longer rediscover it from the tracker.
    """

    def __init__(self, names) -> None:
        self.names = set(names)
        super().__init__(", ".join(sorted(self.names)))


# Deterministic refusals from the pickler: the value's structure, not the
# machine's state, is what failed. Everything else (OSError, MemoryError, a
# lifecycle error) stays fail-closed, because there a retry could genuinely
# reach a different outcome.
_STRUCTURAL_PICKLE_ERRORS = frozenset({
    "TypeError",
    "PicklingError",
    "NotImplementedError",
    "AttributeError",
})

_MAX_DEMOTION_RETRIES = 1


async def save_session(sandbox, *, bindings):
    async with sandbox._exec_lock:
        retries_left = _MAX_DEMOTION_RETRIES
        demoted: set[str] = set()
        while True:
            context = await _prepare_save(sandbox, bindings=bindings)
            if isinstance(context, bool):
                if context:
                    await _publish_committed_projection(
                        sandbox, bindings=bindings
                    )
                return context
            save_code = _build_save_code(context)
            try:
                return await _execute_save(
                    context,
                    save_code,
                    bindings=bindings,
                    allow_demotion=retries_left > 0,
                    previously_demoted=demoted,
                )
            except RequiredValueDemoted as demotion:
                # The offending name is gone from the required set, so the next
                # pass is a different dump rather than the same one repeated.
                demoted |= demotion.names
                retries_left -= 1


async def _publish_committed_projection(sandbox, *, bindings) -> None:
    """Notify the workflow after the lifecycle lock and durable worker exit."""
    callback = getattr(sandbox, "runtime_persistence_commit_callback", None)
    if not callable(callback):
        return
    persistence = dict(
        getattr(sandbox.notebook_state, "runtime_persistence", {}) or {}
    )
    event = {
        "session_id": str(sandbox.session_id),
        "lifecycle_generation": int(
            persistence.get("lifecycle_generation")
            or getattr(sandbox, "_expected_lifecycle_generation", 0)
            or 0
        ),
        "current_revision": int(
            persistence.get("current_revision") or 0
        ),
        "saved_revision": int(persistence.get("saved_revision") or 0),
        "runtime_persistence": persistence,
    }
    try:
        outcome = callback(event)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        # The authoritative sidecar is already committed. Projection delivery
        # is retried naturally by every session read/save and must not turn a
        # successful runtime generation into a false persistence failure.
        bindings.logger.exception(
            "[%s] Runtime commit projection callback failed",
            sandbox.session_id,
        )


async def _prepare_save(sandbox, *, bindings):
    restore_failed = bool(getattr(sandbox, "_session_restore_failed", False))
    mark = getattr(sandbox, "_restore_failed_at_revision", None)
    revision = getattr(sandbox, "runtime_revision", 0)
    saved_revision = getattr(sandbox, "saved_runtime_revision", -1)
    previous_persistence = dict(
        getattr(sandbox.notebook_state, "runtime_persistence", {}) or {}
    )
    expected_lifecycle_generation = int(
        sandbox._expected_lifecycle_generation
    )

    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionGoneError,
        assert_session_generation_active,
        lifecycle_file_lock,
    )

    def _restore_previous_persistence() -> None:
        sandbox.saved_runtime_revision = int(
            previous_persistence.get("saved_revision") or 0
        )
        if hasattr(sandbox.notebook_state, "runtime_persistence"):
            sandbox.notebook_state.runtime_persistence = dict(
                previous_persistence
            )

    def _record_generation_failure(
        error_summary: str,
        *,
        failed_variables=None,
        budget_violations=None,
        top_offenders=None,
        recommended_action=None,
        requires_recompute=False,
        recompute_variables=None,
    ) -> bool:
        """Persist one bounded failure fact for the frozen generation."""
        _restore_previous_persistence()
        try:
            with lifecycle_file_lock(sandbox.session_id):
                assert_session_generation_active(
                    sandbox.session_id,
                    expected_generation=expected_lifecycle_generation,
                )
                updater = getattr(
                    sandbox.notebook_state,
                    "update_runtime_persistence",
                    None,
                )
                if callable(updater):
                    updater(
                        "failed",
                        current_revision=revision,
                        saved_revision=sandbox.saved_runtime_revision,
                        failed_variables=list(failed_variables or []),
                        error_summary=error_summary,
                        budget_violations=list(budget_violations or []),
                        top_offenders=list(top_offenders or []),
                        recommended_action=recommended_action,
                        requires_recompute=requires_recompute,
                        recompute_variables=list(
                            recompute_variables or []
                        ),
                    )
        except (SessionGenerationError, SessionGoneError) as lifecycle_error:
            _restore_previous_persistence()
            bindings.logger.warning(
                "[%s] Suppressed runtime failure fact for frozen "
                "lifecycle generation %d (%s)",
                sandbox.session_id,
                expected_lifecycle_generation,
                type(lifecycle_error).__name__,
            )
            return False
        return True

    has_committed_dump = getattr(sandbox.notebook_state, "has_session_dump", None)
    if (
        not restore_failed
        and revision == saved_revision
        and (
            getattr(sandbox.notebook_state, "runtime_persistence", {}) or {}
        ).get("commit_source") == "regular_save"
        and callable(has_committed_dump)
        and has_committed_dump()
    ):
        bindings.logger.debug(
            "[%s] Reusing integrity-validated runtime generation %d",
            sandbox.session_id,
            revision,
        )
        return True
    if restore_failed and (mark is None or revision <= mark):
        # Cheap short-circuit: nothing has run, so nothing can have
        # been rebuilt, and the kernel need not be touched to establish
        # that.  This decision is deliberately made under the lock.
        bindings.logger.error(
            "[%s] refusing to persist a kernel whose restore failed and "
            "which has executed nothing since; keeping the previous pkl",
            sandbox.session_id,
        )
        _record_generation_failure(
            "Runtime restore failed and no recomputation has completed.",
            failed_variables=sorted(sandbox.get_runtime_dependency_names()),
        )
        return False
    session_path = str(sandbox.notebook_state.session_path.absolute())
    escaped_path = repr(session_path)

    if restore_failed:
        rebuilt = await sandbox._rebuilt_since_restore_failure()
        if not rebuilt:
            bindings.logger.error(
                "[%s] refusing to persist a kernel whose restore failed: "
                "%d cell(s) have run since, but the namespace still holds "
                "nothing this kernel built, so the previous pkl stays the "
                "better evidence and is kept",
                sandbox.session_id,
                revision - mark,
            )
            _record_generation_failure(
                "Runtime restore failed and the live namespace has not "
                "rebuilt required continuation state.",
                failed_variables=sorted(sandbox.get_runtime_dependency_names()),
            )
            return False
        bindings.logger.warning(
            "[%s] restore failed earlier; the kernel has since rebuilt "
            "%d value(s) (%s), persisting the rebuilt namespace over the "
            "previous pkl",
            sandbox.session_id,
            len(rebuilt),
            ", ".join(sorted(rebuilt)[:8]),
        )
        # The latch is cleared only once the write has actually landed --
        # see the end of this method. Clearing it here would drop the
        # protection permanently the first time a save raised.

    from backend.utils.runtime_budget import (
        RuntimeBudgetExceeded,
        evaluate_runtime_budget,
    )

    budgets = dict(getattr(sandbox, "runtime_budgets", {}) or {})
    resource_metrics: Dict[str, Any] = {"top_offenders": []}

    def _record_budget_failure(violations, *, error_summary: str) -> None:
        serialized = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in violations
        ]
        _record_generation_failure(
            error_summary,
            budget_violations=serialized,
            top_offenders=resource_metrics.get("top_offenders") or [],
            recommended_action="reduce_runtime_footprint",
        )

    if budgets:
        resource_metrics = await sandbox._runtime_resource_metrics()
        required_metrics = {
            "anndata_total_bytes",
            "rss_bytes",
            "backed_anndata_total_bytes",
            "projected_materialized_rss_bytes",
        }
        missing_metrics = sorted(required_metrics - set(resource_metrics))
        if missing_metrics:
            measurement_failure = [{
                "resource": "measurement",
                "observed": "unavailable",
                "limit": ",".join(missing_metrics),
            }]
            _record_budget_failure(
                measurement_failure,
                error_summary=(
                    "Runtime resource measurement failed; persistence was "
                    "stopped before serialization."
                ),
            )
            raise RuntimeError(
                "Sandbox runtime resource measurement failed for session "
                f"{sandbox.session_id}: {', '.join(missing_metrics)}"
            )
        preflight_violations = evaluate_runtime_budget(
            resource_metrics, budgets
        )
        if preflight_violations:
            _record_budget_failure(
                preflight_violations,
                error_summary="Runtime resource budget exceeded before serialization.",
            )
            raise RuntimeBudgetExceeded(
                preflight_violations,
                resource_metrics.get("top_offenders"),
            )

    bindings.logger.info(f"Saving session to {session_path}")

    # Compatibility cleanup for dependency trackers written by older
    # execute() code. That code could mark a declared output as required
    # even when a swallowed exception meant the name never existed. Only
    # prune missing names that were never observed as created; genuinely
    # tracked cross-cell dependencies remain fail-closed below.
    live_names = await sandbox._snapshot_user_globals()
    stale_names = sandbox._prune_legacy_unobserved_required(live_names)
    if stale_names:
        bindings.logger.warning(
            "[%s] Pruned legacy required outputs never observed in the "
            "kernel: %s",
            sandbox.session_id,
            sorted(stale_names),
        )

    # Canonical names only matter when they actually exist; a canonical
    # name that is simply absent is not an error. Tracker-declared names
    # stay fail-closed whether or not they are live (see _missing_required).
    from backend.services.snapshot_v2.runtime_registry import (
        CANONICAL_RUNTIME_ARTIFACTS,
    )
    required_names = set(sandbox.get_runtime_dependency_names())
    required_names |= set(CANONICAL_RUNTIME_ARTIFACTS) & set(live_names)
    escaped_required = repr(sorted(required_names))
    from backend.services.snapshot_v2.runtime_registry import (
        VOLATILE_RUNTIME_TYPE_PREFIXES,
    )
    return SaveContext(
        sandbox=sandbox,
        restore_failed=restore_failed,
        revision=revision,
        expected_lifecycle_generation=expected_lifecycle_generation,
        required_names=required_names,
        escaped_required=escaped_required,
        escaped_path=escaped_path,
        escaped_volatile_prefixes=repr(VOLATILE_RUNTIME_TYPE_PREFIXES),
        record_generation_failure=_record_generation_failure,
        restore_previous_persistence=_restore_previous_persistence,
        SessionGenerationError=SessionGenerationError,
        SessionGoneError=SessionGoneError,
        assert_session_generation_active=assert_session_generation_active,
        lifecycle_file_lock=lifecycle_file_lock,
    )


def _build_save_code(context: SaveContext) -> str:
    from backend.services.snapshot_v2.runtime_registry import (
        VOLATILE_RUNTIME_TYPE_PREFIXES,
    )

    volatile_prefixes = (
        context.escaped_volatile_prefixes
        or repr(VOLATILE_RUNTIME_TYPE_PREFIXES)
    )
    save_code = """\
import dill
import io
import json as _scai_json
import os
import tempfile
import types
import zstandard as zstd

_required_names = set(""" + context.escaped_required + """)
# Names the host declared required, captured before the backed-AnnData pass
# below adds to the set. Only these are eligible for the in-memory
# persistability probe: an AnnData that cannot be pickled is authoritative
# analysis state, and dropping it silently is exactly the loss the required
# tier exists to prevent.
_scai_host_required = set(_required_names)
_scai_probe_budget = 64 * 1024 * 1024


class _ScaiProbeTooLarge(Exception):
    # The frame outgrew the in-memory persistability probe.
    pass


class _ScaiProbeSink(io.RawIOBase):
    # Buffer a dill frame in memory, abandoning it past a byte budget.

    def __init__(self, _budget):
        io.RawIOBase.__init__(self)
        self._chunks = []
        self._size = 0
        self._budget = _budget

    def writable(self):
        return True

    def write(self, _payload):
        _view = memoryview(_payload)
        self._size += _view.nbytes
        if self._size > self._budget:
            raise _ScaiProbeTooLarge()
        self._chunks.append(_view.tobytes())
        return _view.nbytes

    def frame_bytes(self):
        return b''.join(self._chunks)


def _scai_is_anndata(_o):
    try:
        _t = type(_o)
        return _t.__name__ == "AnnData" and "anndata" in (_t.__module__ or "").lower()
    except Exception:
        return False


# Handles and connections are the dangerous case: dill often "succeeds" on them
# by recording a path or fd, then reopens it on restore. On another machine that
# path may not exist, and a raising frame aborts the whole restore -- so an open
# loom/h5py/sqlite handle could cost the user every other variable. Exclude by
# type rather than waiting for a pickling error that may never come.
_scai_volatile_prefixes = """ + volatile_prefixes + """


def _scai_is_volatile(_o):
    _t = type(_o)
    _q = ((_t.__module__ or "") + "." + _t.__name__).lower()
    return _q.startswith(_scai_volatile_prefixes)


# Modules, functions, classes and methods are re-established by kernel init and
# notebook replay. They bloat the dump, and a by-reference frame is the single
# most likely thing to fail on restore in a different environment.
_scai_skip_kinds = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)

_live = {}
for _scai_name in list(globals().keys()):
    if not _scai_name.startswith('_'):
        _live[_scai_name] = globals()[_scai_name]

# Every AnnData is required no matter what it is called. Matching canonical
# names alone silently dropped things like `adata_hepatocyte`, which is exactly
# the state a resumed analysis cannot recompute.
#
# A backed AnnData wraps an open h5py handle that dill cannot pickle. It is
# still authoritative runtime state: silently removing it and committing the
# remaining namespace made the metadata claim `current` while a real analysis
# object had disappeared. Materialize one in-memory copy per object identity
# for the dump, preserving aliases. If materialization fails, abort the whole
# generation so the runtime-persistence gate remains armed on the live kernel.
_scai_materialized_backed = {}
for _scai_name, _scai_obj in list(_live.items()):
    if not _scai_is_anndata(_scai_obj):
        continue
    if getattr(_scai_obj, 'isbacked', False):
        _scai_object_id = id(_scai_obj)
        try:
            _scai_memory_obj = _scai_materialized_backed.get(_scai_object_id)
            if _scai_memory_obj is None:
                _scai_memory_obj = _scai_obj.to_memory()
                if getattr(_scai_memory_obj, 'isbacked', False):
                    raise RuntimeError('AnnData.to_memory() returned a backed object')
                _scai_materialized_backed[_scai_object_id] = _scai_memory_obj
            _live[_scai_name] = _scai_memory_obj
        except Exception as _e:
            raise RuntimeError(
                "backed AnnData could not be materialized for persistence: "
                + _scai_name + " (" + type(_e).__name__ + ")"
            ) from _e
    _required_names.add(_scai_name)

_missing_required = sorted(_n for _n in _required_names if _n not in _live)
if _missing_required:
    raise RuntimeError(
        "required runtime variables are missing: " + ", ".join(_missing_required)
    )

# Required values first so a failure aborts before any optional bytes are
# written; the rest of the namespace follows on a best-effort basis.
_to_dump = []
for _scai_name in sorted(_live):
    if _scai_name in _required_names:
        _to_dump.append((_scai_name, _live[_scai_name], True))
for _scai_name in sorted(_live):
    _scai_obj = _live[_scai_name]
    if _scai_name in _required_names:
        continue
    if isinstance(_scai_obj, _scai_skip_kinds) or _scai_is_volatile(_scai_obj):
        continue
    _to_dump.append((_scai_name, _scai_obj, False))

_skipped_vars = []
_saved_object_names = {}
_saved_names = []

# Write to a temp file.  The host signs and installs it together with committed
# generation metadata; kernel code must never replace the authoritative pkl.
_tmp_path = """ + context.escaped_path + """ + '.tmp'
_saved_count = 0
_compression_magic = b"SCAI_ZSTD_DILL_V1\\n"
try:
    with open(_tmp_path, 'wb') as _raw:
        _raw.write(_compression_magic)
        with zstd.ZstdCompressor(level=1).stream_writer(_raw, closefd=False) as _f:
            for _var_name, _obj, _is_required in _to_dump:
                _object_id = id(_obj)
                if _object_id in _saved_object_names:
                    _frame = (
                        "__SCAI_ALIAS__", _var_name,
                        _saved_object_names[_object_id],
                    )
                else:
                    _frame = (_var_name, _obj)
                # Required values may stream because any failure aborts the
                # generation. Optional values first serialize into a spooled
                # file, bounding memory without trusting a library's entire
                # top-level namespace to be dill-safe.
                # A required value proves it serializes before a byte of it
                # reaches the output stream. One that raises here is
                # deterministically unpicklable -- a retry is the identical
                # computation -- so it is dropped like an optional value and
                # the host releases it from the required set. Values past the
                # budget are the large analysis objects; they stream as before
                # and a failure there still aborts the generation.
                _probe_first = (
                    _is_required
                    and _var_name in _scai_host_required
                    and not _scai_is_anndata(_obj)
                )
                _streamed = _is_required and not _probe_first
                try:
                    if _probe_first:
                        _probe = _ScaiProbeSink(_scai_probe_budget)
                        try:
                            dill.dump(_frame, _probe)
                        except _ScaiProbeTooLarge:
                            _streamed = True
                            dill.dump(_frame, _f)
                        else:
                            _streamed = True
                            _f.write(_probe.frame_bytes())
                    elif _is_required:
                        dill.dump(_frame, _f)
                    else:
                        with tempfile.SpooledTemporaryFile(
                            max_size=8 * 1024 * 1024, mode='w+b'
                        ) as _frame_file:
                            dill.dump(_frame, _frame_file)
                            _frame_file.seek(0)
                            # Serialization is complete. A failure from this
                            # point may have partially modified the compressed
                            # stream, so it must abort the generation.
                            _streamed = True
                            while True:
                                _frame_chunk = _frame_file.read(1024 * 1024)
                                if not _frame_chunk:
                                    break
                                _f.write(_frame_chunk)
                except Exception as _e:
                    _skipped_vars.append((
                        _var_name,
                        type(_obj).__name__,
                        type(_e).__name__,
                        str(_e)[:200],
                    ))
                    _sv = _skipped_vars[-1]
                    print(f"__SCAI_SAVE_SKIPPED__\\t{_sv[0]}\\t{_sv[1]}\\t{_sv[2]}\\t{_sv[3]}")
                    if _streamed:
                        raise RuntimeError(
                            ("required runtime variable could not be persisted: "
                             if _is_required else
                             "streamed runtime variable failed mid-write, "
                             "aborting to avoid a desynced dump: ") + _var_name
                        ) from _e
                    continue
                _saved_object_names.setdefault(_object_id, _var_name)
                _saved_names.append(_var_name)
                _saved_count += 1
except Exception:
    try:
        os.remove(_tmp_path)
    except OSError:
        pass
    raise

print(f"Session saved: {_saved_count} variables")
print("__SCAI_SAVED_NAMES__:" + _scai_json.dumps(sorted(_saved_names)))
"""
    return save_code


def _release_structural_failures(sandbox, names) -> set[str]:
    """Drop provably unpicklable names from the tracker's required set."""
    tracker = getattr(sandbox, "runtime_dependency_tracker", None)
    if tracker is None:
        return set()
    releasable = {name for name in names if name in tracker.required_outputs}
    if not releasable:
        return set()
    tracker.release(
        releasable,
        lifecycle="superseded",
        reason=(
            "Value is structurally unpicklable; every retry repeats the same "
            "serialization, so it cannot be continuation state."
        ),
    )
    return releasable


def _attribute_failed_names(context, skipped, result, previously_demoted) -> set[str]:
    """Name the required values this attempt failed on, from every channel."""
    # A name released earlier in this same save stays attributable: the tracker
    # no longer holds it, so nothing else can name what failed.
    attributable = set(context.required_names) | set(previously_demoted or ())
    failed_names = {
        sv["name"] for sv in skipped if sv.get("name") in attributable
    }
    error_text = str(result.error or "")
    exact_failure_patterns = (
        r"required runtime variable could not be persisted:\s*([A-Za-z_]\w*)",
        r"backed AnnData could not be materialized for persistence:\s*([A-Za-z_]\w*)",
    )
    for pattern in exact_failure_patterns:
        failed_names.update(
            match.group(1)
            for match in re.finditer(pattern, error_text)
            if match.group(1) in attributable
        )
    missing_match = re.search(
        r"required runtime variables are missing:\s*([^\n]+)", error_text,
    )
    if missing_match:
        failed_names.update(
            name.strip()
            for name in missing_match.group(1).split(",")
            if name.strip() in attributable
        )
    return failed_names


def _reject_incomplete_runtime(
    context,
    skipped,
    result,
    *,
    bindings,
    allow_demotion: bool,
    previously_demoted,
) -> None:
    """Always raises: either to demote and retry, or to arm the gate."""
    sandbox = context.sandbox
    if skipped:
        bindings.logger.error(
            "[%s] save_session rejected incomplete runtime; required "
            "variables not persisted: %s",
            sandbox.session_id,
            [sv["name"] for sv in skipped],
        )
    bindings.logger.warning(f"Failed to save session: {result.error}")
    failed_names = _attribute_failed_names(
        context, skipped, result, previously_demoted
    )
    if allow_demotion:
        # Only values too large for the in-kernel probe reach this branch, so
        # the stream is already unusable and the staged file is gone. Releasing
        # the name makes the retry a different dump rather than the same
        # deterministic failure a second time.
        released = _release_structural_failures(sandbox, {
            sv["name"] for sv in skipped
            if sv.get("name") in failed_names
            and sv.get("exc") in _STRUCTURAL_PICKLE_ERRORS
        })
        if released:
            bindings.logger.error(
                "[%s] Required runtime variable(s) failed mid-stream and are "
                "structurally unpicklable; released and retrying the save "
                "without them: %s",
                sandbox.session_id,
                ", ".join(sorted(released)),
            )
            raise RequiredValueDemoted(released)

    poisoned: set[str] = set()
    tracker = getattr(sandbox, "runtime_dependency_tracker", None)
    if tracker is not None and failed_names:
        poisoned = tracker.record_persistence_failure(failed_names)
    context.record_generation_failure(
        result.error or "Runtime serialization failed.",
        failed_variables=sorted(failed_names),
        requires_recompute=bool(poisoned),
        recompute_variables=sorted(poisoned),
    )
    raise RuntimeError(
        f"Sandbox runtime save failed for session {sandbox.session_id}: "
        f"{result.error or 'unknown error'}"
    )


async def _execute_save(
    context: SaveContext,
    save_code: str,
    *,
    bindings,
    allow_demotion: bool = False,
    previously_demoted: set[str] | None = None,
) -> bool:
    sandbox = context.sandbox
    result = await sandbox._execute_unlocked(
        save_code,
        save_to_history=False,
        skip_safety_check=True,
    )

    skipped = []
    for line in (result.output or "").splitlines():
        if line.startswith("__SCAI_SAVE_SKIPPED__\t"):
            parts = line.split("\t")
            if len(parts) >= 5:
                skipped.append({
                    "name": parts[1],
                    "type": parts[2],
                    "exc": parts[3],
                    "msg": parts[4],
                })
    sandbox._last_save_skipped = skipped

    saved_names: set[str] = set()
    for line in (result.output or "").splitlines():
        if not line.startswith("__SCAI_SAVED_NAMES__:"):
            continue
        try:
            parsed_names = bindings.json.loads(line.split(":", 1)[1])
        except (bindings.json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed_names, list):
            saved_names = {str(name) for name in parsed_names}

    if result.success:
        if skipped:
            # Surfaced so a later restore that is missing a value has a
            # breadcrumb explaining why.
            bindings.logger.info(
                "[%s] Session saved; %d unpicklable value(s) skipped: %s",
                sandbox.session_id,
                len(skipped),
                ", ".join(f"{sv['name']}({sv['type']})" for sv in skipped),
            )
            # The in-kernel probe proved these before writing any of their
            # bytes, so this dump is complete apart from them. Release them
            # here or every later save re-runs the identical failing
            # serialization and the analysis stalls on a scratch variable.
            released = _release_structural_failures(sandbox, {
                sv["name"] for sv in skipped
                if sv.get("name") in context.required_names
            })
            if released:
                bindings.logger.error(
                    "[%s] Required runtime variable(s) are structurally "
                    "unpicklable and were released from the required set: %s",
                    sandbox.session_id,
                    ", ".join(sorted(released)),
                )
        pkl_path = bindings.Path(sandbox.notebook_state.session_path)
        staged_path = bindings.Path(str(pkl_path) + ".tmp")
        production_transaction = staged_path.exists()
        integrity_path = bindings._integrity_dir(sandbox.session_id)

        def _cleanup_unjournaled_products() -> None:
            """Remove only this attempt's unjournaled staged pkl."""
            journal_path = pkl_path.with_name(
                f"{pkl_path.name}.regular-save.json"
            )
            if journal_path.exists():
                return
            staged_path.unlink(missing_ok=True)

        manifest_path = None
        target_digest = None
        if production_transaction:
            from backend.utils.runtime_apply import runtime_h5ad_manifest_path

            manifest_path = runtime_h5ad_manifest_path(
                sandbox.session_id,
                states_dir=sandbox.notebook_state.state_dir,
            )
            try:
                target_digest = await bindings._run_durable_host_operation(
                    lambda: bindings._sha256_file(staged_path),
                    session_id=sandbox.session_id,
                    description="runtime digest calculation",
                )
            except BaseException as exc:
                _cleanup_unjournaled_products()
                context.record_generation_failure(
                    "Runtime digest calculation did not complete: "
                    f"{type(exc).__name__}."
                )
                raise

        def _commit_generation_manifest():
            if not production_transaction:
                return
            from backend.utils.runtime_apply import (
                write_runtime_h5ad_manifest,
            )

            write_runtime_h5ad_manifest(
                sandbox.session_id,
                {},
                authoritative_pkl_names=saved_names,
                states_dir=sandbox.notebook_state.state_dir,
            )

        def _retire_generation_sidecars():
            from backend.utils.runtime_apply import (
                quarantine_unreferenced_runtime_h5ads,
            )

            quarantine_unreferenced_runtime_h5ads(
                sandbox.session_id,
                set(),
                states_dir=sandbox.notebook_state.state_dir,
            )

        def _commit_generation_metadata():
            sandbox.saved_runtime_revision = context.revision
            updater = getattr(
                sandbox.notebook_state, "update_runtime_persistence", None
            )
            if callable(updater):
                updater(
                    "current",
                    current_revision=context.revision,
                    saved_revision=context.revision,
                    commit_source="regular_save",
                    runtime_digest=target_digest,
                )

        commit_outcome: Dict[str, Any] = {
            "completed": False,
            "error": None,
        }

        try:
            def _commit_on_host() -> None:
                try:
                    with context.lifecycle_file_lock(sandbox.session_id):
                        context.assert_session_generation_active(
                            sandbox.session_id,
                            expected_generation=context.expected_lifecycle_generation,
                        )
                        if staged_path.exists():
                            bindings._commit_pkl_transaction(
                                staged_path,
                                pkl_path,
                                integrity_path,
                                _commit_generation_metadata,
                                metadata_path=bindings.Path(sandbox.notebook_state.state_path),
                                manifest_path=manifest_path,
                                commit_manifest=_commit_generation_manifest,
                                retire_sidecars=_retire_generation_sidecars,
                                target_revision=context.revision,
                                authoritative_pkl_names=saved_names,
                                fault_hook=getattr(
                                    sandbox, "_regular_save_fault_hook", None
                                ),
                            )
                        else:
                            # Compatibility for test doubles and pre-transaction
                            # custom kernels. Production save code always leaves
                            # the staged path and never reaches this branch.
                            bindings._write_pkl_signature(pkl_path, integrity_path)
                            context.assert_session_generation_active(
                                sandbox.session_id,
                                expected_generation=context.expected_lifecycle_generation,
                            )
                            _commit_generation_metadata()
                except BaseException as exc:
                    commit_outcome["error"] = exc
                    raise
                else:
                    commit_outcome["completed"] = True

            await bindings._run_durable_host_operation(
                _commit_on_host,
                session_id=sandbox.session_id,
                description="runtime generation commit",
            )
            bindings.logger.info("Session saved successfully")
            await _publish_committed_projection(sandbox, bindings=bindings)
        except BaseException as exc:
            worker_error = commit_outcome.get("error")
            if isinstance(exc, bindings.asyncio.CancelledError):
                if commit_outcome.get("completed"):
                    # The worker published a complete generation; preserve
                    # its metadata before re-delivering cancellation.
                    raise
                _cleanup_unjournaled_products()
                if isinstance(worker_error, context.SessionGoneError):
                    context.restore_previous_persistence()
                else:
                    context.record_generation_failure(
                        "Runtime generation commit was cancelled after "
                        "the durable worker reached its terminal boundary."
                    )
                raise

            _cleanup_unjournaled_products()
            if isinstance(exc, context.SessionGoneError):
                # A tombstoned generation must not write new metadata.
                context.restore_previous_persistence()
                raise
            if isinstance(exc, context.SessionGenerationError):
                context.record_generation_failure(
                    "Runtime generation rejected by the session lifecycle."
                )
                raise
            context.record_generation_failure(
                f"Runtime commit failed: {type(exc).__name__}"
            )
            raise RuntimeError(
                f"Sandbox runtime commit failed for session {sandbox.session_id}"
            ) from exc
    else:
        _reject_incomplete_runtime(
            context,
            skipped,
            result,
            bindings=bindings,
            allow_demotion=allow_demotion,
            previously_demoted=previously_demoted,
        )

    # A rebuilt namespace is now on disk, so the pkl the latch was
    # protecting no longer exists to protect. Clearing here rather than
    # at the guard keeps the protection in place across a save that
    # raised, which would otherwise disarm it permanently.
    if getattr(sandbox, "_session_restore_failed", False):
        sandbox._session_restore_failed = False
        sandbox._restore_failed_at_revision = None
        sandbox._restore_failed_globals = None

    return True
