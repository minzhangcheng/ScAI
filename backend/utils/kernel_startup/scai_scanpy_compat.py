"""Narrow Scanpy Scrublet compatibility for ScAI kernels."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any


def install() -> bool:
    """Add ``layer=`` only when the installed Scrublet lacks it."""
    try:
        import scanpy as sc
    except ModuleNotFoundError as exc:
        if exc.name == "scanpy":
            return False
        raise

    original = sc.pp.scrublet
    if getattr(original, "_scai_layer_compat", False):
        return True
    signature = inspect.signature(original)
    if "layer" in signature.parameters:
        return True

    @wraps(original)
    def _scrublet_with_layer(
        adata: Any,
        adata_sim: Any = None,
        *,
        layer: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if layer is None:
            return original(adata, adata_sim=adata_sim, **kwargs)
        if layer not in adata.layers:
            raise KeyError(f"AnnData layer {layer!r} is unavailable for scrublet")

        copy_result = bool(kwargs.pop("copy", False))
        target = adata.copy() if copy_result else adata
        original_x = target.X
        layer_values = target.layers[layer]
        target.X = (
            layer_values.copy()
            if hasattr(layer_values, "copy")
            else layer_values
        )
        try:
            original(target, adata_sim=adata_sim, copy=False, **kwargs)
        finally:
            target.X = original_x
        return target if copy_result else None

    parameters = list(signature.parameters.values())
    copy_index = next(
        (index for index, value in enumerate(parameters) if value.name == "copy"),
        len(parameters),
    )
    parameters.insert(
        copy_index,
        inspect.Parameter("layer", inspect.Parameter.KEYWORD_ONLY, default=None),
    )
    _scrublet_with_layer.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    _scrublet_with_layer._scai_layer_compat = True  # type: ignore[attr-defined]
    sc.pp.scrublet = _scrublet_with_layer
    return True
