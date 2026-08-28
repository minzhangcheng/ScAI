"""Database-prior and gene-set binding owners for the executor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional


async def _compute_database_priors(
    agent, adata_var, cluster_key, rank_key, organism, workspace_dir,
    tissue="", cluster_markers=None,
    *,
    bindings,
) -> Any:
    """Compute offline marker priors without making annotation depend on them.

    The Scanpy rank key remains the primary marker source. When that key is
    unavailable, the kernel falls back to the sanitized marker mapping that
    ``annotate_clusters`` is actually using. Every outcome is explicit so a
    database failure cannot be mistaken for a successful literature-only
    decision.
    """
    marker_priors_write = getattr(bindings, "marker_priors_write")
    DatabasePriorResult = getattr(bindings, "DatabasePriorResult")
    _ORA_KERNEL_SRC = getattr(bindings, "_ORA_KERNEL_SRC")
    _json = getattr(bindings, "_json")
    _compact_database_prior_error = getattr(bindings, "_compact_database_prior_error")
    logger = getattr(bindings, "logger")
    try:
        if not marker_priors_write(organism, workspace_dir):
            return DatabasePriorResult(
                "skipped", {},
                f"marker database has no entries for organism {organism!r}",
            )
        src = _ORA_KERNEL_SRC
        markers_json = _json.dumps(cluster_markers or {}, ensure_ascii=False)
        call = (
            f"\n_provided_cluster_markers = json.loads({_json.dumps(markers_json)})\n"
            f"_priors = compute_ora_priors({adata_var}, "
            f"'tables/marker_priors_input.tsv', {cluster_key!r}, {rank_key!r}, "
            f"tissue={str(tissue or '')!r}, "
            f"cluster_markers=_provided_cluster_markers)\n"
            f"import json as __j; print(__j.dumps(_priors))\n"
        )
        result = await agent.sandbox.execute(src + call, save_to_history=False,
                                            skip_safety_check=True)
        if not getattr(result, "success", False):
            raw_error = (
                getattr(result, "error", None)
                or getattr(result, "output", None)
                or "unknown sandbox execution error"
            )
            detail = _compact_database_prior_error(raw_error)
            logger.warning(
                f"[{agent.session_id}] DB priors sandbox execution failed: {raw_error}")
            return DatabasePriorResult("failed", {}, detail)
        out = getattr(result, "output", "") or ""
        line = next((ln for ln in out.strip().splitlines()[::-1]
                     if ln.strip().startswith("{")), None)
        if line is None:
            detail = "sandbox returned no JSON result"
            logger.warning(f"[{agent.session_id}] DB priors computation failed: {detail}")
            return DatabasePriorResult("failed", {}, detail)
        try:
            priors = _json.loads(line)
        except (TypeError, ValueError) as e:
            detail = _compact_database_prior_error(f"invalid JSON result: {e}")
            logger.warning(f"[{agent.session_id}] DB priors computation failed: {detail}")
            return DatabasePriorResult("failed", {}, detail)
        if not isinstance(priors, dict):
            detail = f"sandbox returned {type(priors).__name__}, expected object"
            logger.warning(f"[{agent.session_id}] DB priors computation failed: {detail}")
            return DatabasePriorResult("failed", {}, detail)
        if not priors:
            return DatabasePriorResult(
                "empty", {}, "no eligible marker-database overlaps passed the ORA threshold")
        marker_sources = {
            str(hit.get("marker_source") or "")
            for hits in priors.values() if isinstance(hits, list)
            for hit in hits if isinstance(hit, dict)
        }
        marker_sources.discard("")
        return DatabasePriorResult(
            "used", priors, marker_source="+".join(sorted(marker_sources)))
    except Exception as e:
        detail = _compact_database_prior_error(e)
        logger.warning(
            f"[{agent.session_id}] DB priors computation failed: {detail}",
            exc_info=True)
        return DatabasePriorResult("failed", {}, detail)


async def _bind_gene_set(
    agent, set_id: str, genes: List[str], label: str,
    *,
    bindings,
) -> Optional[str]:
    """Bind a fetched panel into the kernel; returns its variable name or None.

    Handing the genes over as data rather than asking the model to retype
    them is where a fetched panel stops silently becoming a remembered one
    again. `None` means the kernel would not take them, and the caller must
    inline the genes into its reply instead.
    """
    re = getattr(bindings, "re")
    compact_traceback = getattr(bindings, "compact_traceback")
    logger = getattr(bindings, "logger")
    var_name = f"gene_set_{re.sub(r'[^0-9A-Za-z_]', '_', set_id)[:60]}"
    try:
        result = await agent.sandbox.execute(
            f"{var_name} = {genes!r}",
            save_to_history=False, skip_safety_check=True,
        )
        # A kernel that will not take the assignment says so in the result
        # -- dead kernel, restart, timeout, MemoryError on a large panel --
        # far more often than it raises. Reporting the variable anyway is
        # the worst outcome available: the caller then suppresses the inline
        # gene list, and the next cell dies on NameError or the model
        # retypes the panel from memory, which is what this helper exists to
        # prevent.
        if result.error:
            compact_error = compact_traceback(result.error, max_chars=1000)
            logger.warning(
                f"[{agent.session_id}] {label} injection failed: {compact_error}"
            )
            return None
        return var_name
    except Exception as exc:
        logger.warning(
            "[%s] %s injection failed: %s",
            agent.session_id,
            label,
            compact_traceback(str(exc), max_chars=1000),
        )
        return None


async def _bind_gene_set_collection(
    agent, collection: str, path: Path,
    *,
    bindings,
) -> Optional[str]:
    """Bind a backend-cached GMT path into the analysis kernel."""
    re = getattr(bindings, "re")
    compact_traceback = getattr(bindings, "compact_traceback")
    logger = getattr(bindings, "logger")
    var_name = (
        "gene_set_collection_"
        + re.sub(r"[^0-9A-Za-z_]", "_", collection)[:60]
        + "_path"
    )
    try:
        result = await agent.sandbox.execute(
            f"{var_name} = {str(path)!r}",
            save_to_history=False,
            skip_safety_check=True,
        )
        if result.error:
            logger.warning(
                "[%s] gene-set collection path injection failed: %s",
                agent.session_id,
                compact_traceback(result.error, max_chars=1000),
            )
            return None
        return var_name
    except Exception as exc:
        logger.warning(
            "[%s] gene-set collection path injection failed: %s",
            agent.session_id,
            compact_traceback(str(exc), max_chars=1000),
        )
        return None
