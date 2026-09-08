"""Runtime save phases for LocalJupyterSandbox."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from typing import Any, Dict

from backend.utils.runtime_names import PRIVATE_EXCLUSION_SCRIPT


class RuntimeCoverageError(RuntimeError):
    """The kernel did not report a usable coverage record for this generation."""


_COVERAGE_MARKERS = {
    "saved_names": "__SCAI_SAVED_NAMES__:",
    "excluded_by_kind": "__SCAI_EXCLUDED_KIND__:",
    "excluded_volatile": "__SCAI_EXCLUDED_VOLATILE__:",
}

# Spec 5.5's carried __main__ definitions. Deliberately NOT in the table above:
# that table is the completeness contract, and a producer missing one of its
# entries is treated as broken. A kernel script from a build before this change
# emits three markers, and it is not broken -- it is older. Parsed when present,
# absent without complaint.
_OPTIONAL_MARKERS = {
    "carried_definitions": "__SCAI_CARRIED_DEFINITIONS__:",
    "excluded_private": "__SCAI_EXCLUDED_PRIVATE__:",
}


def _parse_coverage_markers(output: str) -> dict | None:
    """Read the generation's coverage record, or say there is none.

    Three outcomes, and the difference between the last two matters:

    * every marker present and well formed -- the generation can describe
      itself, and a manifest is written from this record;
    * no marker at all -- this producer does not report coverage. That is a
      pre-transaction custom kernel or a test double, not a claim of
      completeness: the generation commits *without* a manifest and a later
      restore treats it as legacy, coverage unknown. Returning an empty record
      instead would be the real hazard, because an empty record asserts that
      nothing was excluded;
    * some markers, or malformed ones -- the producer is broken, and a
      generation whose own account of itself is truncated must not commit.
    """
    found: dict = {}
    optional: dict = {}
    for line in (output or "").splitlines():
        for key, prefix in _OPTIONAL_MARKERS.items():
            if line.startswith(prefix):
                try:
                    value = json.loads(line[len(prefix):])
                except json.JSONDecodeError:
                    # An optional record that cannot be read is dropped, not
                    # fatal. It describes what was brought in; the generation's
                    # completeness does not rest on it.
                    value = None
                if isinstance(value, list):
                    optional[key] = value
        for key, prefix in _COVERAGE_MARKERS.items():
            if not line.startswith(prefix):
                continue
            if key in found:
                raise RuntimeCoverageError(f"coverage marker repeated: {prefix}")
            try:
                parsed = json.loads(line[len(prefix):])
            except json.JSONDecodeError as exc:
                raise RuntimeCoverageError(
                    f"malformed coverage marker {prefix}: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise RuntimeCoverageError(
                    f"malformed coverage marker {prefix}: expected a list"
                )
            found[key] = parsed
    if not found:
        return None
    missing = sorted(set(_COVERAGE_MARKERS) - set(found))
    if missing:
        raise RuntimeCoverageError(
            "coverage record is partial; markers absent: "
            + ", ".join(_COVERAGE_MARKERS[key].strip(":") for key in missing)
        )
    found.update(optional)
    return found


def _classify_skipped(skipped, *, excluded_by_kind, excluded_volatile,
                      excluded_private=()):
    """One coverage record from the four channels the kernel reports.

    A value dill refused, one never offered because of its kind, one excluded
    as a volatile handle, and one excluded for having a private name are four
    different facts about the same generation; collapsing them loses the only
    signal that says whether a name is worth recomputing. The last defaults to
    empty, so a producer that predates it classifies as it used to.
    """
    entries: dict = {}
    for item in skipped or ():
        name = str(item.get("name") or "")
        if name:
            entries[name] = {
                "name": name,
                "type": str(item.get("type") or "unknown"),
                "reason": "serialization_failed",
            }
    for reason, source in (
        ("excluded_by_kind", excluded_by_kind),
        ("excluded_volatile", excluded_volatile),
        ("excluded_private", excluded_private),
    ):
        for item in source or ():
            name = str(item.get("name") or "")
            if name:
                entries.setdefault(name, {
                    "name": name,
                    "type": str(item.get("type") or "unknown"),
                    "reason": reason,
                })
    return [entries[name] for name in sorted(entries)]


def _carried_definitions_for_saved_names(
    coverage: dict | None,
    saved_names: set[str],
) -> list[dict] | None:
    """Return only carried definitions present in this dump generation.

    A ``__main__`` definition can be discovered by the kernel before dill
    rejects it (for example, a closure over a generator).  The definition is
    therefore present in the optional carried record even though its name is
    absent from ``saved_names``.  Keep the manifest's strict validation for
    malformed entries, but do not pass a valid, skipped name to it.
    """
    carried = (coverage or {}).get("carried_definitions") or None
    if carried is None:
        return None
    filtered = [
        entry
        for entry in carried
        if not (
            isinstance(entry, dict)
            and "name" in entry
            and str(entry.get("name")) not in saved_names
        )
    ]
    return filtered or None


@dataclass
class SaveContext:
    sandbox: Any
    restore_failed: bool
    revision: int
    expected_lifecycle_generation: int
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
    pairing: dict | None = None


async def save_session(sandbox, *, bindings):
    # One pass. The demotion retry existed so a deterministically unpicklable
    # *required* value could be released and the dump rebuilt without it; with
    # no required set such a value is simply skipped and recorded as coverage,
    # so the second pass would produce the identical dump.
    async with sandbox._exec_lock:
        return await save_session_locked(sandbox, bindings=bindings)


async def save_session_locked(sandbox, *, bindings):
    """Commit while the caller holds the execution lock through a larger capture."""
    context = await _prepare_save(sandbox, bindings=bindings)
    if isinstance(context, bool):
        if context:
            await _publish_committed_projection(sandbox, bindings=bindings)
        return context
    return await _execute_save(
        context, _build_save_code(context), bindings=bindings,
    )


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

    from backend.state import sessions as live_sessions
    from backend.utils.runtime_pairing import capture_pairing

    try:
        pairing = await bindings.asyncio.to_thread(
            capture_pairing, sandbox.session_id, live_sessions.get(sandbox.session_id),
            getattr(sandbox.notebook_state, "state_dir", None),
        )
    except Exception:
        _record_generation_failure("Could not read the workflow checkpoint identity")
        raise
    has_committed_dump = getattr(sandbox.notebook_state, "has_session_dump", None)
    if (
        not restore_failed
        and previous_persistence.get("committed_pairing") == pairing
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
    from backend.utils.runtime_types import (
        VOLATILE_RUNTIME_TYPE_PREFIXES,
    )
    return SaveContext(
        sandbox=sandbox,
        restore_failed=restore_failed,
        revision=revision,
        expected_lifecycle_generation=expected_lifecycle_generation,
        escaped_path=escaped_path,
        escaped_volatile_prefixes=repr(VOLATILE_RUNTIME_TYPE_PREFIXES),
        record_generation_failure=_record_generation_failure,
        restore_previous_persistence=_restore_previous_persistence,
        SessionGenerationError=SessionGenerationError,
        SessionGoneError=SessionGoneError,
        assert_session_generation_active=assert_session_generation_active,
        lifecycle_file_lock=lifecycle_file_lock,
        pairing=pairing,
    )


def _build_save_code(context: SaveContext) -> str:
    from backend.utils.runtime_types import (
        STRUCTURAL_PICKLE_ERRORS,
        VOLATILE_RUNTIME_TYPE_PREFIXES,
    )

    volatile_prefixes = (
        context.escaped_volatile_prefixes
        or repr(VOLATILE_RUNTIME_TYPE_PREFIXES)
    )
    structural_errors = repr(sorted(STRUCTURAL_PICKLE_ERRORS))
    save_code = """\
import dill as _scai_dill
import json as _scai_json
# Captured, not resolved at use: the kernel namespace binds `open` to the
# sandbox write guard, and an analysis cell is free to rebind it. A bare
# `open` in this script therefore fails every save as soon as a user
# defines one -- the same shadowing that once destroyed user variables
# named types, io, os, dill, tempfile and zstd.
import builtins as _scai_builtins
_scai_open = _scai_builtins.open
import os as _scai_os
import tempfile as _scai_tempfile
import types as _scai_types
import zstandard as _scai_zstd

# A refusal that is a property of the value is coverage (spec 4.4); a machine
# failure is not, and must abort rather than commit a generation that reports
# itself complete.
_scai_structural_errors = frozenset(""" + structural_errors + """)

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


# Imported functions, classes and methods are re-established by kernel init and
# notebook replay, they bloat the dump, and a by-reference frame is the single
# most likely thing to fail on restore in a different environment. The user's
# own definitions are none of those things: nothing re-establishes them, so a
# helper defined in round two was simply gone in round three (spec 5.5).
#
# Modules stay excluded unconditionally. They are the bulk the exclusion was
# aimed at, and a module is never the user's own state.
_scai_skip_kinds = (
    _scai_types.ModuleType,
    _scai_types.FunctionType,
    _scai_types.BuiltinFunctionType,
    _scai_types.MethodType,
    type,
)


def _scai_is_own_definition(_o):
    # A function or class defined in a cell, rather than imported. The kernel
    # sets __module__ to "__main__" for those.
    #
    # Bootstrap-owned callables are excluded even though they satisfy that: the
    # bootstrap recreates them on every start, and `open` is the write guard, so
    # carrying it lets a restore reinstate the saved generation's guard over the
    # fresh one.
    if getattr(_o, "_scai_bootstrap_owned", False):
        return False
    #
    # The kind test has to come first. An *instance* of a __main__ class also
    # reports __module__ == "__main__", because the lookup falls through to its
    # class -- so without it every user object would look like a definition.
    if isinstance(_o, _scai_types.ModuleType):
        return False
    if not isinstance(_o, _scai_skip_kinds):
        return False
    try:
        return getattr(_o, "__module__", None) == "__main__"
    except Exception:
        return False

""" + PRIVATE_EXCLUSION_SCRIPT + """
_live = {_n: globals()[_n] for _n in _scai_snapshot if not _n.startswith('_')}

# Every AnnData is required no matter what it is called. Matching canonical
# names alone silently dropped things like `adata_hepatocyte`, which is exactly
# the state a resumed analysis cannot recompute.
#
# A backed AnnData wraps an open h5py handle that dill cannot pickle. It is
# still authoritative runtime state: silently removing it and committing the
# remaining namespace made the metadata claim `current` while a real analysis
# object had disappeared. Materialize one in-memory copy per object identity
# for the dump, preserving aliases. If materialization fails, abort the whole
# generation rather than commit a dump that is missing analysis state.
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

# Every value takes the same path: no value is privileged, and none can abort
# the generation by failing to serialize. Spec 4.4 -- coverage warns, integrity
# blocks, and a value dill refuses is coverage.
_to_dump = []
_excluded_by_kind = []
_excluded_volatile = []
# Spec 5.5's carried definitions, reported so a restore can say what it brought
# back and, for a class, what is still not true about it.
_carried_definitions = []
for _scai_name in sorted(_live):
    _scai_obj = _live[_scai_name]
    # These two exclusions used to be silent, so a name dropped here was absent
    # from the dump and absent from every record of the dump. Reporting them
    # separately is what lets a restore say why a name is missing rather than
    # only that it is.
    if isinstance(_scai_obj, _scai_skip_kinds) and not _scai_is_own_definition(_scai_obj):
        _excluded_by_kind.append({
            "name": _scai_name, "type": type(_scai_obj).__name__,
        })
        continue
    if _scai_is_own_definition(_scai_obj):
        # Carried, and for a class the caveat travels with it. Every instance
        # was pickled in its own frame with its own memo, so dill embedded a
        # copy of the class in each of them: the restored name is a *different*
        # object from type(instance), and isinstance(old_instance, ThisClass)
        # is False. Disclosed rather than hidden -- before this change the name
        # was absent and the same code raised NameError, which is louder.
        if isinstance(_scai_obj, type):
            _carried_definitions.append({
                "name": _scai_name,
                "kind": "class",
                "note": "restored class is not type(instance) for instances "
                        "restored from other frames",
            })
        else:
            _carried_definitions.append({
                "name": _scai_name, "kind": "function",
            })
    if _scai_is_volatile(_scai_obj):
        _scai_t = type(_scai_obj)
        _excluded_volatile.append({
            "name": _scai_name,
            "type": (_scai_t.__module__ or "") + "." + _scai_t.__name__,
        })
        continue
    _to_dump.append((_scai_name, _scai_obj))

_skipped_vars = []
_saved_object_names = {}
_saved_names = []

# Write to a temp file.  The host signs and installs it together with committed
# generation metadata; kernel code must never replace the authoritative pkl.
_tmp_path = """ + context.escaped_path + """ + '.tmp'
_saved_count = 0
_compression_magic = b"SCAI_ZSTD_DILL_V1\\n"
try:
    with _scai_open(_tmp_path, 'wb') as _raw:
        _raw.write(_compression_magic)
        with _scai_zstd.ZstdCompressor(level=1).stream_writer(_raw, closefd=False) as _f:
            for _var_name, _obj in _to_dump:
                _object_id = id(_obj)
                if _object_id in _saved_object_names:
                    _frame = (
                        "__SCAI_ALIAS__", _var_name,
                        _saved_object_names[_object_id],
                    )
                else:
                    _frame = (_var_name, _obj)
                # Every value serializes into a spooled file first, so a
                # dill failure is discovered before a byte reaches the
                # compressed stream and the value can simply be skipped. Only
                # a failure after that point is unrecoverable: the zstd
                # compressor's state has already advanced, so the dump would be
                # desynced and the generation must abort.
                _streamed = False
                try:
                    with _scai_tempfile.SpooledTemporaryFile(
                        max_size=8 * 1024 * 1024, mode='w+b'
                    ) as _frame_file:
                        _scai_dill.dump(_frame, _frame_file)
                        _frame_file.seek(0)
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
                            "streamed runtime variable failed mid-write, "
                            "aborting to avoid a desynced dump: " + _var_name
                        ) from _e
                    if type(_e).__name__ not in _scai_structural_errors:
                        # The machine failed, not the value. A retry could
                        # reach a different outcome, so this is not coverage
                        # and the generation must not commit as complete.
                        raise RuntimeError(
                            "runtime persistence failed on the environment, "
                            "not on the value: " + _var_name
                            + " (" + type(_e).__name__ + ")"
                        ) from _e
                    continue
                _saved_object_names.setdefault(_object_id, _var_name)
                _saved_names.append(_var_name)
                _saved_count += 1
except Exception:
    try:
        _scai_os.remove(_tmp_path)
    except OSError:
        pass
    raise

print(f"Session saved: {_saved_count} variables")
print("__SCAI_SAVED_NAMES__:" + _scai_json.dumps(sorted(_saved_names)))
print("__SCAI_EXCLUDED_KIND__:" + _scai_json.dumps(
    sorted(_excluded_by_kind, key=lambda _e: _e["name"])))
print("__SCAI_EXCLUDED_VOLATILE__:" + _scai_json.dumps(
    sorted(_excluded_volatile, key=lambda _e: _e["name"])))
print("__SCAI_CARRIED_DEFINITIONS__:" + _scai_json.dumps(
    sorted(_carried_definitions, key=lambda _e: _e["name"])))
print("__SCAI_EXCLUDED_PRIVATE__:" + _scai_json.dumps(
    sorted(_excluded_private, key=lambda _e: _e["name"])))
"""
    return save_code


def _paired_sources(sandbox, *, bindings) -> dict:
    """The live files this generation has to be captured beside. Spec B2-1.

    Three, not two: §3.11 -- the executor's own step checkpoint holds its
    position and its account of what it has already done, and is bound to the
    generation by nothing else. It is absent at a round boundary reached after
    the executor finished, and absent is not an error.
    """
    from backend.utils.checkpoint_paths import checkpoint_sqlite_path

    states = bindings.Path(sandbox.notebook_state.state_dir)
    return {
        "state": bindings.Path(sandbox.notebook_state.state_path),
        "checkpoint": checkpoint_sqlite_path(states),
        "executor_checkpoint": states / f"{sandbox.session_id}_bioinfo_checkpoint.json",
    }


def _reject_incomplete_runtime(context, skipped, result, *, bindings) -> None:
    """Record a failed generation and raise. Always raises.

    A value that could not be serialized is coverage, not a rejection: the save
    script skips it, reports it, and commits. This path is reached only when the
    script itself failed -- a mid-write desync, an OSError, a lifecycle refusal
    -- so there is nothing to attribute to a required set and nothing to retry.
    """
    sandbox = context.sandbox
    skipped_names = sorted({sv["name"] for sv in skipped if sv.get("name")})
    if skipped_names:
        bindings.logger.warning(
            "[%s] runtime save failed; values skipped before the failure: %s",
            sandbox.session_id, skipped_names,
        )
    bindings.logger.warning(f"Failed to save session: {result.error}")
    context.record_generation_failure(
        result.error or "Runtime serialization failed.",
        failed_variables=skipped_names,
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
) -> bool:
    sandbox = context.sandbox
    result = await sandbox._execute_unlocked(
        save_code,
        records_history=False,
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

    # A coverage record describes a dump, so it is required exactly when a dump
    # was produced. The staged file is the same discriminator the commit below
    # uses to tell a real save from the test doubles and pre-transaction custom
    # kernels that never stage one: demanding a record from those would fail a
    # save that had no generation to describe in the first place. When a dump
    # does exist, missing or malformed markers fail the generation rather than
    # degrading to an empty record.
    _staged_probe = bindings.Path(
        str(bindings.Path(sandbox.notebook_state.session_path)) + ".tmp"
    )
    try:
        coverage = (
            _parse_coverage_markers(result.output or "")
            if result.success and _staged_probe.exists()
            else None
        )
    except RuntimeCoverageError as coverage_error:
        # The kernel may already have written a multi-gigabyte staged dump. It
        # can never be committed now, and nothing else reclaims it.
        try:
            _staged_probe.unlink(missing_ok=True)
        except OSError:
            bindings.logger.warning(
                "[%s] could not remove the staged dump of a generation whose "
                "coverage record was unusable",
                sandbox.session_id,
            )
        # Route this through the same channel as any other failed save rather
        # than letting it escape mid-function. An exception raised here unwinds
        # past the generation-failure bookkeeping while the caller still holds
        # the execution lock, and a later save waiting on that lock never
        # returns -- observed as a hang, not as an error.
        context.record_generation_failure(
            f"Runtime coverage record unusable: {coverage_error}",
        )
        raise RuntimeError(
            f"Sandbox runtime save failed for session {sandbox.session_id}: "
            f"{coverage_error}"
        ) from coverage_error
    saved_names: set[str] = (
        {str(name) for name in coverage["saved_names"]} if coverage else set()
    )
    skipped_record = (
        _classify_skipped(
            skipped,
            excluded_by_kind=coverage["excluded_by_kind"],
            excluded_volatile=coverage["excluded_volatile"],
            # `.get`, like the carried record: optional, not incomplete.
            excluded_private=coverage.get("excluded_private") or [],
        )
        if coverage
        else []
    )
    # `.get`, not a subscript: the three completeness markers are required of
    # every producer, this one is not. A kernel script from before §5.5's
    # narrowing reports no carried definitions and is not thereby incomplete.
    # None rather than [] so the manifest omits the key entirely, which keeps a
    # pre-§5.5 generation's record byte-identical to what it is today.
    carried_record = _carried_definitions_for_saved_names(
        coverage, saved_names,
    )

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
            # Nothing is released, because nothing was required. Each name is
            # already in the generation's coverage record with its reason, and
            # the next save will skip it the same way at the same cost.
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

        runtime_manifest_file = None
        target_digest = None
        if production_transaction:
            from backend.utils.runtime_manifest import runtime_manifest_path

            runtime_manifest_file = runtime_manifest_path(
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

        # The sidecar manifest recorded which names the pkl is authoritative
        # for. A1's runtime_manifest.json carries exactly that in saved_names,
        # so the older record is gone rather than kept in parallel.

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
                    committed_pairing=context.pairing,
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
                                runtime_manifest_path=runtime_manifest_file,
                                runtime_manifest={
                                    "state_revision": (context.pairing or {}).get("state_version", 0),
                                    "pairing": context.pairing,
                                    "saved_names": sorted(saved_names),
                                    "skipped_names": skipped_record,
                                    "carried_definitions": carried_record,
                                    "serializer_version": "dill-zstd-1",
                                },
                                target_revision=context.revision,
                                generation_root=bindings.Path(
                                    sandbox.notebook_state.state_dir
                                ) / bindings.GENERATION_PAIR_DIRNAME,
                                paired_sources=_paired_sources(sandbox, bindings=bindings),
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
        _reject_incomplete_runtime(context, skipped, result, bindings=bindings)

    # A rebuilt namespace is now on disk, so the pkl the latch was
    # protecting no longer exists to protect. Clearing here rather than
    # at the guard keeps the protection in place across a save that
    # raised, which would otherwise disarm it permanently.
    if getattr(sandbox, "_session_restore_failed", False):
        sandbox._session_restore_failed = False
        sandbox._restore_failed_at_revision = None
        sandbox._restore_failed_globals = None

    return True
