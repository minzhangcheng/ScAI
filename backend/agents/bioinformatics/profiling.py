"""Read the data once, before planning, to learn how many replicates it holds.

TreatmentGroup.sample_count is a declaration nothing verified. Counting files
instead does not fix it and errs in both directions -- one specimen exported as
both .loom and .h5 counts as two, while one matrix holding several multiplexed
specimens counts as one. Only reading the data answers the question, so this
pass loads it and asks the model what it sees.

The loaded object stays in the kernel. The executor's first plan step then
verifies it instead of re-reading 280 MB from disk, and because verification is
still a code step the plan audit still gets its execution evidence.

Read calls do not disturb artifact provenance: _REPRESENTATION_MUTATORS is
{normalize_total, log1p, scale, normalize_per_cell}, so sc.read_* neither bumps
analysis_generation nor invalidates anything.
"""
import json
import logging
from typing import Any, Dict, List

from ...constants import NO_OUTPUT_TOKEN_CAP
from ...models.schemas import SampleCountRecord
from ...services.sample_design import canonical_modality
from .shared import (
    _ensure_list, UpstreamExhaustedError, compact_traceback,
    mutates_expression_representation,
)
from .tool_schemas import PROFILING_TOOLS, PROFILING_TOOL_NAMES

logger = logging.getLogger(__name__)

#: Backstop, not a target. One load cell plus roughly two inspection cells per
#: sequencing data type, with headroom for a retry after a read error; the pass
#: is expected to finish in three or four. Exhausting it ends the pass with
#: whatever was registered -- a group left without a record falls back to the
#: previous per-modality resolution, which is a correct outcome,
#: not a failure.
MAX_PROFILING_STEPS = 12

_SYSTEM_PROMPT = """You are inspecting uploaded sequencing data to determine \
how many biological replicates each experimental group actually contains. \
This is a read-only pass. Do not normalize, filter, cluster, or analyze.

Load every sequencing input with the appropriate reader and inspect it. What \
you are looking for is the number of distinct biological specimens, which is \
not the number of files:

- Two files of different formats may hold the same specimen. Compare cell \
counts, barcodes, and gene names before treating them as separate replicates.
- One matrix may hold several multiplexed specimens. Look at obs columns \
whose values partition the cells (sample, batch, orig.ident, donor, library), \
and at barcode suffixes.
- For a bulk expression table, inspect both axes, labels, value types, and any \
sample metadata before deciding where biological specimens are represented. \
When the data verify membership, register a separate record for each treatment \
group even though the uploaded file itself is ungrouped. Use an empty group only \
when the sample-to-group membership cannot be established from the data.

Declare every object later steps should reuse in persistent_outputs, so the \
analysis that follows does not have to read these files again.

Every execute_code call in this profiling pass must set \
execution_purpose to 'supplementary', plan_step_ids to an empty list, and \
input_profile_ids to an empty list. This \
pass runs before plan approval and cannot satisfy a plan requirement.

When you have inspected everything, call register_sample_counts once with one \
record per (group, data type) pair. Report what the data shows, not what the \
stated design says it should show. If a count cannot be established, register \
it with detected_count 0 and method 'undetermined' rather than guessing."""


class ProfilingMixin:
    """Pre-planning replicate-count detection. Composed into BioinformaticsAgent."""

    async def profile_data(self, state: Any) -> Dict[str, Any]:
        """Inspect the sequencing inputs and return measured replicate counts.

        Never raises for a modelling or execution problem: the caller treats a
        thin result as "no records", and the design resolver falls back.
        """
        design = state.build_experimental_design_summary()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"{design}\n\n"
                f"{_file_listing(state)}\n\n"
                "Inspect these inputs and register the replicate counts."
            )},
        ]

        records: List[SampleCountRecord] = []
        objects: List[str] = []
        summary = ""
        # Counts consecutive register_sample_counts calls that produced zero
        # usable records. A payload that could not be parsed, or that named no
        # real group, must not silently end the pass as "0 records registered"
        # -- the model is told what was wrong and gets another turn, unless it
        # has already had one and still sent nothing usable.
        failed_registration_attempts = 0

        for step in range(MAX_PROFILING_STEPS):
            try:
                call_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": PROFILING_TOOLS,
                    "tool_choice": "auto",
                    "temperature": self.config.temperature,
                    "max_tokens": NO_OUTPUT_TOKEN_CAP,
                    "prefer_stream": True,
                }
                if callable(getattr(self, "_budgeted_text_call", None)):
                    response = await self._budgeted_text_call(
                        self.llm_client.chat_with_tools,
                        **call_kwargs,
                    )
                else:
                    # Lightweight mixin consumers used by compatibility tests do
                    # not inherit BaseAgent. Production agents always take the
                    # shared observational budget path above.
                    response = await self.llm_client.chat_with_tools(**call_kwargs)
            except Exception as exc:                      # noqa: BLE001
                logger.warning("[Profiling] LLM call failed at step %d: %s", step, exc)
                return {"status": "failed", "records": records,
                        "objects": objects, "summary": f"LLM call failed: {exc}"}

            tool_calls = list(getattr(response, "tool_calls", None) or [])
            if not tool_calls:
                summary = (getattr(response, "content", "") or "")[:2000]
                break

            # Mirrors the executor's own replay construction (bioinformatics_agent.py):
            # raw_assistant_message carries provider-specific fields (e.g. Gemini's
            # thought_signature) that must be echoed back verbatim, so it takes
            # priority whenever the adapter supplied one.
            raw_assistant_message = getattr(response, "raw_assistant_message", None)
            if raw_assistant_message:
                messages.append(raw_assistant_message)
            else:
                messages.append({
                    "role": "assistant",
                    "content": getattr(response, "content", "") or "",
                    "tool_calls": [
                        {"id": call.id, "type": "function",
                         "function": {"name": call.name,
                                      "arguments": json.dumps(call.arguments)}}
                        for call in tool_calls
                    ],
                })

            done = False
            for call in tool_calls:
                name = call.name

                # Parsing already happened upstream (chat_with_tools returns
                # ToolCall, not a raw wire object); a malformed-JSON call surfaces
                # here as parse_error rather than as an exception to catch.
                if getattr(call, "parse_error", None):
                    messages.append(_tool_reply(
                        call, f"Arguments were not valid JSON: {call.parse_error}"))
                    continue

                if name not in PROFILING_TOOL_NAMES:
                    messages.append(_tool_reply(
                        call, f"Unknown tool '{name}' in this pass. "
                              f"Available: {sorted(PROFILING_TOOL_NAMES)}."))
                    continue
                args = call.arguments

                # These schemas carry no "strict": true, and this platform has a
                # documented history of models returning payloads that do not
                # match a declared schema. The coercions below give a useful
                # reply for the shapes already seen (wrong-typed description,
                # code, persistent_outputs, records); this catch-all is the
                # backstop for whatever shape neither of us has thought of yet --
                # mirrors the executor's own per-call try/except.
                try:
                    if name == "execute_code":
                        messages.append(_tool_reply(
                            call, await self._profiling_execute(args, objects)))
                    else:
                        raw_records = args.get("records")
                        accepted = _to_records(state, _ensure_list(raw_records))
                        coverage_error = _registration_coverage_error(
                            state, accepted,
                        ) if accepted else ""
                        if accepted and not coverage_error:
                            records = accepted
                            messages.append(_tool_reply(
                                call, f"Registered {len(records)} record(s)."))
                            done = True
                        else:
                            # A dropped payload (not a list, or every
                            # group_name unmatched) must not read as "the
                            # model registered nothing on purpose" -- tell it
                            # concretely what was wrong so it can correct
                            # itself, unless it has already had that chance.
                            failed_registration_attempts += 1
                            messages.append(_tool_reply(
                                call,
                                coverage_error or _explain_empty_registration(
                                    state, raw_records,
                                )))
                            if failed_registration_attempts >= 2:
                                done = True
                except UpstreamExhaustedError:
                    raise
                except Exception as exc:                  # noqa: BLE001
                    compact_error = compact_traceback(str(exc), max_chars=3000)
                    logger.warning(
                        "[Profiling] Tool '%s' failed: %s; continuing",
                        name, compact_error)
                    messages.append(_tool_reply(
                        call, f"Error processing tool '{name}': {compact_error}"))

            if done:
                break

        if not records and failed_registration_attempts:
            # The model tried to register counts and never landed a single
            # usable record -- a degraded pass, not a quiet success, so
            # _data_profiling's existing degraded branch must see it rather
            # than the resolver silently falling back to the declaration.
            return {"status": "failed", "records": records, "objects": objects,
                    "summary": summary or (
                        "register_sample_counts was called but no usable "
                        "records were ever accepted.")}
        return {"status": "completed", "records": records,
                "objects": objects, "summary": summary}

    async def _profiling_execute(self, args: Dict[str, Any],
                                 objects: List[str]) -> str:
        code = str(args.get("code") or "")
        if not code:
            return "No code provided; nothing to execute."

        # This pass shares execute_code's schema -- and its normalization-
        # inviting description -- with the executor, but must stay read-only:
        # a mutation here would bump analysis_generation and invalidate
        # same-round artifacts before any plan step has run. Refuse rather
        # than run it; the executor's own execute path is where normalization
        # belongs.
        if mutates_expression_representation(code):
            return (
                "Execution blocked: this profiling pass is read-only. Your "
                "execute_code call would change the expression representation "
                "(e.g. normalize_total, log1p, scale, or normalize_per_cell), "
                "which this pass must not do. Inspect the data without "
                "mutating it -- normalization belongs in the analysis plan, "
                "not in this measurement pass."
            )

        declared = [str(name) for name in _ensure_list(args.get("persistent_outputs"))
                    if str(name).isidentifier()]

        is_safe, security_error = self._validate_code(code)
        if not is_safe:
            logger.warning("[Profiling] Code validation failed: %s", security_error)
            return f"Execution blocked: {security_error}"

        description = str(args.get("description") or "Inspecting data...")
        await self._notify("status", description[:200])
        try:
            result = await self.sandbox.execute(
                code, records_history=True,
                declared_outputs=declared, track_globals=True,
            )
        except Exception as exc:                          # noqa: BLE001
            logger.warning("[Profiling] Execution raised: %s", exc)
            return f"Execution raised: {exc}"

        if getattr(result, "error", None):
            return f"Execution failed:\n{result.error}"

        for name in declared:
            if name not in objects:
                objects.append(name)
        return (getattr(result, "output", "") or "")[:8000] or "(no output)"


def _upload_reference(file_path: str) -> str:
    """Name an input the way a replayable cell must reference it.

    The absolute path is what `file_path` holds, and handing it to the model is
    how ten of them reached one run's LLM traffic and one of them reached a
    notebook cell. A cell that hard-codes `/data/sessions/<id>/uploads/...`
    cannot run anywhere but the machine that produced it, which is exactly what
    the replay contract is about.

    `_UPLOADS_DIR` is bound by the kernel bootstrap, which A2-9 made cell 1 of
    the notebook, so a cell written against it travels. A path that is not under
    the uploads root is reported as-is: mislabelling it would be worse than
    naming it.
    """
    from pathlib import PurePosixPath

    text = str(file_path or "")
    marker = "/uploads/"
    index = text.find(marker)
    if index == -1:
        return text
    relative = PurePosixPath(text[index + len(marker):])
    return f"_UPLOADS_DIR / {str(relative)!r}"


def _file_listing(state: Any) -> str:
    lines = [
        "Sequencing inputs (reference them through `_UPLOADS_DIR`, which the "
        "notebook's first cells define; do not hard-code an absolute path):"
    ]
    for data_file in (state.data_files or []):
        if getattr(data_file, "section", "") != "sequencing":
            continue
        group = state.get_group_by_id(getattr(data_file, "group_id", None))
        lines.append(
            f"  - {data_file.filename} | type={data_file.data_type} | "
            f"group={getattr(group, 'name', '(ungrouped)')} | "
            f"path={_upload_reference(data_file.file_path)}"
        )
    if len(lines) == 1:
        lines.append("  (none)")
    return "\n".join(lines)


def _tool_reply(call: Any, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.id, "content": content}


def _explain_empty_registration(state: Any, raw: Any) -> str:
    """Diagnose why a register_sample_counts call produced zero usable
    records, so the reply gives the model something to correct rather than an
    opaque "Registered 0 record(s).".

    ``_to_records`` only ever drops an entry for one reason: a non-empty
    group_name that matches no group (an empty group_name always succeeds, as
    ungrouped data). So once ``raw`` is confirmed to be a list of dicts, zero
    acceptances means every entry's group_name was unmatched.
    """
    if not isinstance(raw, list):
        return (
            f"'records' must be a JSON array of one object per (group, data "
            f"type) pair; got {type(raw).__name__} instead. Call "
            f"register_sample_counts again with records as an array."
        )
    dict_entries = [e for e in raw if isinstance(e, dict)]
    if not dict_entries:
        return (
            "'records' contained no usable entries -- each entry must be a "
            "JSON object with group_name, data_type, detected_count, method, "
            "and evidence. Call register_sample_counts again with at least "
            "one such record."
        )
    sent_names = sorted({str(e.get("group_name") or "").strip()
                          for e in dict_entries})
    valid_names = sorted(g.name for g in (state.treatment_groups or []))
    return (
        f"None of the group_name value(s) you sent ({sent_names}) matched a "
        f"group in this experimental design. Valid group names are: "
        f"{valid_names}. Call register_sample_counts again using one of "
        f"those exact names."
    )


def _to_records(state: Any, raw: List[Dict[str, Any]]) -> List[SampleCountRecord]:
    """Bind reported counts to real groups. Unknown group names are dropped.

    A name the design does not contain cannot be matched to a group id, so the
    record could never reach the sample-design resolver; keeping it would put an
    unusable row in front of the PI.
    """
    by_name = {g.name: g for g in (state.treatment_groups or [])}
    records: List[SampleCountRecord] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        group_name = str(entry.get("group_name") or "").strip()
        group = by_name.get(group_name)
        if group_name and group is None:
            logger.warning("[Profiling] Dropping record for unknown group %r",
                           group_name)
            continue
        try:
            detected = max(0, int(entry.get("detected_count", 0) or 0))
        except (TypeError, ValueError):
            detected = 0
        declared = getattr(group, "sample_count", None)
        method = str(entry.get("method") or "undetermined").strip() or "undetermined"
        if detected < 1:
            method = "undetermined"
        records.append(SampleCountRecord(
            group_id=getattr(group, "id", None),
            group_name=group_name,
            section="sequencing",
            data_type=str(entry.get("data_type") or "unknown"),
            detected_count=detected,
            declared_count=declared,
            method=method,
            evidence=str(entry.get("evidence") or "")[:500],
            conflicts=bool(detected >= 1 and declared is not None
                           and detected != declared),
        ))
    return records


def _registration_coverage_error(
    state: Any, records: List[SampleCountRecord],
) -> str:
    """Require every sequencing modality to have a complete design record.

    An ungrouped record is complete evidence that membership could not be
    established.  Otherwise every declared treatment group must be represented,
    including by an explicit detected_count=0 record.  Accepting a non-empty
    subset would make the omitted group invisible to the downstream gate.
    """
    expected_groups = {
        getattr(group, "id", None)
        for group in (getattr(state, "treatment_groups", None) or [])
    }
    expected_groups.discard(None)
    expected_modalities = {
        canonical_modality("sequencing", getattr(item, "data_type", ""))
        for item in (getattr(state, "data_files", None) or [])
        if getattr(item, "section", None) == "sequencing"
    }
    grouped: Dict[str, set] = {}
    unresolved: set[str] = set()
    for record in records:
        modality = canonical_modality(
            "sequencing", getattr(record, "data_type", ""),
        )
        group_id = getattr(record, "group_id", None)
        if group_id is None:
            unresolved.add(modality)
        else:
            grouped.setdefault(modality, set()).add(group_id)

    missing_modalities = sorted(
        modality for modality in expected_modalities
        if modality not in unresolved
        and not expected_groups.issubset(grouped.get(modality, set()))
    )
    if not missing_modalities:
        return ""
    return (
        "The sample-count registration is incomplete for: "
        + ", ".join(missing_modalities)
        + ". For each listed modality, send one record for every treatment "
        "group, using detected_count 0 when a group's count is undetermined; "
        "or send one empty-group record when column-to-group membership cannot "
        "be established at all. Inspect every sequencing input and call "
        "register_sample_counts again with the complete records array."
    )
