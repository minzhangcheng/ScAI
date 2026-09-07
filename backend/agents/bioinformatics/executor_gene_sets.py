"""Small invariant helpers for gene-set fetch dispatch."""
from __future__ import annotations

from .executor_dispatch import ToolAction


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def resolve_fetch_organism(args, state) -> str:
    return str(args.get("organism") or getattr(state, "species", "") or "").strip()


def fetch_failure(error) -> str:
    return f"fetch_gene_set failed: {error}"


def partition_failure(error) -> str:
    return f"fetch_gene_set could not partition: {error}"


async def handle_fetch_gene_set(context, tool_call) -> ToolAction:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    from ...services.gene_set_fetch import (
        list_collections as list_gene_set_collections,
        load_collection, collection_cache_path,
        resolve_gene_set,
        resolve_partitioned_gene_set,
    )
    from ...services.gene_set_resolver import (
        DEFAULT_PROJECTION_SOURCE, MSIGDB_ORGANISMS,
        direction_sibling_ids, normalize_organism, suggest_similar,
    )

    set_id = str(args.get("set_id", "") or "").strip()
    collection = str(args.get("collection", "") or "").strip()
    version = str(args.get("version", "") or "").strip()
    organism = normalize_organism(
        resolve_fetch_organism(args, state)
    )

    native = organism in MSIGDB_ORGANISMS
    browse_organism = (
        organism if native else DEFAULT_PROJECTION_SOURCE
    )
    projection_note = "" if native else (
        f" The resource publishes these identifiers for "
        f"{browse_organism}, not for the session organism "
        f"{organism}: a set fetched by set_id is projected "
        f"from {browse_organism} through orthology and carries "
        f"its mapping rate."
    )

    if not organism:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                "fetch_gene_set rejected: no organism is known for this "
                "session and none was supplied."
            ),
        })
        return ToolAction.CONTINUE

    if args.get("list_collections"):
        # Discovery first: the identifiers are organism-specific
        # and a guess fails with a 404 that teaches nothing.
        try:
            available = await asyncio.to_thread(
                list_gene_set_collections, version, browse_organism
            )
            content = (
                f"Collections available for {browse_organism} in release "
                f"{version}: {', '.join(available)}. "
                f"Pick one and call again with collection=<id> and "
                f"either list_sets=true to browse or set_id=<id> to "
                f"fetch.{projection_note}"
            )
        except Exception as exc:
            content = (
                f"fetch_gene_set could not list collections for "
                f"{browse_organism} in release {version!r}: {exc}"
            )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": content,
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="fetch_gene_set",
            agent_reasoning=agent_reasoning,
            code_description=f"Listed collections for {organism} {version}",
        ))
        return ToolAction.CONTINUE

    if args.get("list_sets"):
        # Discovery path: never fabricate an identifier, look it up.
        try:
            available = await asyncio.to_thread(
                load_collection, version, collection, browse_organism
            )
        except Exception as exc:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    f"fetch_gene_set could not load {collection!r} "
                    f"v{version!r} for {organism}: {exc}"
                ),
            })
            analysis_log.append(AnalysisStep(
                step_number=step_count, step_type="fetch_gene_set",
                agent_reasoning=agent_reasoning,
                code_description=(
                    f"FAILED: browse {collection!r} v{version!r} "
                    f"({organism})"
                ),
            ))
            return ToolAction.CONTINUE
        hits = (suggest_similar(set_id, available.keys(), limit=25)
                if set_id else sorted(available)[:25])
        path_var = None
        if native:
            collection_path = collection_cache_path(
                version, collection, organism
            )
            path_var = await self._bind_gene_set_collection(
                collection, collection_path
            )
        if path_var:
            binding_note = (
                f" The complete versioned GMT is bound in the "
                f"sandbox as {path_var}. Pass that variable directly "
                f"as gene_sets= to gseapy.prerank/enrichr; do not "
                f"scan _WORKSPACE_DIR for the backend cache."
            )
        elif native:
            binding_note = (
                " The full native GMT could not be bound in the "
                "sandbox; fetch the named sets you need with "
                "set_id=<id>."
            )
        else:
            binding_note = (
                " GMT is NOT bound in the sandbox: it carries "
                f"{browse_organism} symbols and this session analyses "
                f"{organism}. Fetch the named sets you need with "
                "set_id=<id>, which projects them and reports the "
                "mapping rate."
            )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                f"{len(available)} sets in {collection} v{version} "
                f"({browse_organism}). Matching: "
                f"{', '.join(hits) or 'none'}."
                f"{binding_note}{projection_note}"
            ),
        })
        return ToolAction.CONTINUE

    if not set_id or not collection:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                "fetch_gene_set needs both collection and set_id to "
                "fetch. Call with list_collections=true to see the "
                "identifiers this organism publishes, then "
                "list_sets=true to browse that collection."
            ),
        })
        return ToolAction.CONTINUE

    if args.get("partition_by_direction"):
        parts, error = await asyncio.to_thread(
            resolve_partitioned_gene_set, set_id, organism,
            version=version, collection=collection,
        )
        if error:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": partition_failure(error),
            })
            analysis_log.append(AnalysisStep(
                step_number=step_count, step_type="fetch_gene_set",
                agent_reasoning=agent_reasoning,
                code_description=(
                    f"FAILED: partition {set_id!r} -- {str(error)[:140]}"
                ),
            ))
            return ToolAction.CONTINUE

        lines = [
            f"Split {set_id} into {len(parts)} direction-coherent "
            f"panels. Score them SEPARATELY -- that is the point of "
            f"the split: it keeps activators and repressors from "
            f"cancelling in one score. A higher score on the positive "
            f"panel is not by itself evidence that the process is more "
            f"active; for an activity question consider a footprint "
            f"method. Register each as its own plan and score."
        ]
        for part in parts:
            name = part.set_id.split("#")[-1]
            var = await self._bind_gene_set(
                part.set_id, part.genes, "partition"
            )
            bound = (
                f" bound as {var}" if var
                else f" genes: {', '.join(part.genes)}"
            )
            lines.append(
                f"[{name}] {part.n_genes} genes, "
                f"provenance {part.provenance}{bound}."
            )
            if name == "ambiguous":
                lines.append(
                    "The ambiguous part is annotated to both "
                    "directions; it cannot support a directional "
                    "claim and is separated so it does not count "
                    "toward either arm."
                )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": " ".join(lines),
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="fetch_gene_set",
            agent_reasoning=agent_reasoning,
            code_description=(
                f"Partitioned {set_id} into "
                + ", ".join(
                    f"{r.set_id.split('#')[-1]}({r.n_genes})"
                    for r in parts)
            ),
        ))
        return ToolAction.CONTINUE

    result, error = await asyncio.to_thread(
        resolve_gene_set, set_id, organism,
        version=version, collection=collection,
    )
    if error:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": fetch_failure(error),
        })
        # Record the attempt. Without this a failed fetch is
        # invisible in session.json, and a later reader
        # concludes the tool was never called -- which is the
        # opposite of what happened in the first run.
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="fetch_gene_set",
            agent_reasoning=agent_reasoning,
            code_description=f"FAILED: {set_id!r} -- {str(error)[:160]}",
        ))
        return ToolAction.CONTINUE

    var_name = await self._bind_gene_set(
        set_id, result.genes, "gene set"
    )
    injected = var_name is not None

    lines = [
        f"Fetched {result.set_id}: {result.n_genes} genes. "
        f"Provenance: {result.provenance}. "
        f"Pass this string as gene_list_source / panel_source."
    ]
    if injected:
        lines.append(
            f"The genes are bound in the sandbox as {var_name} -- use "
            f"that variable directly rather than retyping them."
        )
    else:
        lines.append(f"Genes: {', '.join(result.genes)}")
    lines.extend(result.warnings())
    lines.append(_method_rule_text("read-the-set-before-scoring-it"))
    if direction_sibling_ids(result.set_id):
        lines.append(
            "This set may be direction-mixed. Call again with "
            "partition_by_direction=true to split it into "
            "direction-coherent panels using the ontology's own "
            "regulation terms, rather than assembling one by hand."
        )
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": " ".join(lines),
    })

    analysis_log.append(AnalysisStep(
        step_number=step_count, step_type="fetch_gene_set",
        agent_reasoning=agent_reasoning,
        code_description=(
            f"Fetched {result.set_id} ({result.n_genes} genes) "
            f"from {result.provenance}"
        ),
    ))

    return ToolAction.CONTINUE
