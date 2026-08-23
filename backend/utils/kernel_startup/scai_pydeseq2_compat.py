"""Keep PyDESeq2's local inference inside ScAI kernel process limits."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


_PROCESS_BACKENDS = {"loky", "multiprocessing"}


def install() -> bool:
    """Route PyDESeq2 process backends through joblib threads.

    Only PyDESeq2's imported binding is wrapped.  The global joblib API remains
    untouched for callers whose execution is not owned by the kernel adapter.
    """
    try:
        import pydeseq2.default_inference as inference
    except ModuleNotFoundError as exc:
        if exc.name == "pydeseq2":
            return False
        raise

    original = inference.parallel_backend
    if getattr(original, "_scai_thread_backend", False):
        return True

    @contextmanager
    def _thread_backend(
        backend: Any, *args: Any, **kwargs: Any,
    ) -> Iterator[Any]:
        selected = (
            "threading"
            if isinstance(backend, str) and backend.lower() in _PROCESS_BACKENDS
            else backend
        )
        options = dict(kwargs)
        if isinstance(selected, str) and selected.lower() == "threading":
            options.pop("inner_max_num_threads", None)
        with original(selected, *args, **options) as active:
            yield active

    _thread_backend._scai_thread_backend = True  # type: ignore[attr-defined]
    inference.parallel_backend = _thread_backend
    return True
