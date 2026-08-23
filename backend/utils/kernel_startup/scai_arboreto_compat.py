"""Narrow Arboreto compatibility for current Dask dataframe releases."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


_EMPTY_DELAYED_ERROR = "Must supply at least one delayed object"


def _wrapper(
    original: Callable[..., Any],
    metadata_schema: Any,
    from_pandas: Callable[..., Any],
) -> Callable[..., Any]:
    """Adapt Arboreto's unused empty metadata graph only."""

    @wraps(original)
    def _from_delayed_with_empty_meta(
        delayed_values: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if (
            isinstance(delayed_values, list)
            and not delayed_values
            and kwargs.get("meta") is metadata_schema
        ):
            return from_pandas(metadata_schema.copy(), npartitions=1)
        return original(delayed_values, *args, **kwargs)

    _from_delayed_with_empty_meta._scai_empty_meta_compat = True  # type: ignore[attr-defined]
    return _from_delayed_with_empty_meta


def install() -> bool:
    """Install only when Arboreto's exact empty delayed call is rejected."""
    try:
        import arboreto.core as core
        import dask.dataframe as dataframe
    except ModuleNotFoundError as exc:
        if exc.name in {"arboreto", "dask"}:
            return False
        raise

    original = core.from_delayed
    if getattr(original, "_scai_empty_meta_compat", False):
        return True

    try:
        original([], meta=core._META_SCHEMA)
    except TypeError as exc:
        if _EMPTY_DELAYED_ERROR not in str(exc):
            raise
    else:
        return True

    core.from_delayed = _wrapper(
        original,
        core._META_SCHEMA,
        dataframe.from_pandas,
    )
    return True
