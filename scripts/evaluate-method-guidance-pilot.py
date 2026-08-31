#!/usr/bin/env python3
"""Generate frozen, opaque off/on plan pairs for the method-guidance pilot.

This development evaluator reads persisted JSONL planning requests, calls the
current configured Bioinformatics model, and writes only to a separate evidence
directory. It never loads, executes, or saves an AnalysisState.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


CASES = (
    ("case-01", "legacy", "7bc10a2b-b8f7-4c1a-ace8-cde783266f2e", 1351),
    ("case-02", "legacy", "7bc10a2b-b8f7-4c1a-ace8-cde783266f2e", 954),
    ("case-03", "legacy", "7bc10a2b-b8f7-4c1a-ace8-cde783266f2e", 622),
    ("case-04", "legacy", "7bc10a2b-b8f7-4c1a-ace8-cde783266f2e", 248),
    ("case-05", "legacy", "7bc10a2b-b8f7-4c1a-ace8-cde783266f2e", 100),
    ("case-06", "current", "e4306558-383e-49a3-a35f-d28ae38b1898", 259),
    ("case-07", "current", "e4306558-383e-49a3-a35f-d28ae38b1898", 104),
    ("case-08", "current", "0706bc07-fafa-49d9-aea1-5f58b869441d", 106),
    ("case-09", "current", "f219d62c-7f28-4368-a176-cb1a97a00bbe", 824),
    ("case-10", "current", "f219d62c-7f28-4368-a176-cb1a97a00bbe", 470),
)
ORDER_SEED = "method-guidance-b2-2026-08-31"
GUIDANCE_HEADER = """

## Non-binding Method Decision Guidance
The following cards compare unranked approaches. Use observed facts to choose
and justify an executable method and its material parameters. Do not treat a
card as a required method or threshold. Put unresolved material alternatives
in key_decision_points for PI review.

{guidance}"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl_line(path: Path, line_number: int) -> dict:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index == line_number:
                return json.loads(line)
    raise ValueError(f"{path} has no line {line_number}")


def _planning_prompt(row: dict) -> str:
    messages = row.get("messages") or []
    user_messages = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    ]
    if len(user_messages) != 1 or not user_messages[0].strip():
        raise ValueError("frozen call does not contain exactly one planning request")
    return user_messages[0]


def _insert_guidance(prompt: str, block: str) -> str:
    for anchor in ("\n## PI's Analysis Goals", "\n## Evidence Policy"):
        if anchor in prompt:
            return prompt.replace(anchor, block + anchor, 1)
    return prompt + block


def _condition_order(case_id: str) -> list[str]:
    digest = hashlib.sha256(f"{ORDER_SEED}:{case_id}".encode()).digest()
    return ["off", "on"] if digest[0] % 2 == 0 else ["on", "off"]


def _load_config(database: Path, secret_file: Path, output_dir: Path):
    os.environ["SCAI_CONFIG_SECRET"] = secret_file.read_text(encoding="utf-8").strip()
    os.environ.setdefault("SCAI_DATA_DIR", str(output_dir))

    from backend.config._config import SystemConfig
    from backend.utils.secret_redaction import open_system_config_payload

    database_copy = output_dir / "config-snapshot.db"
    shutil.copy2(database, database_copy)
    try:
        connection = sqlite3.connect(database_copy)
        try:
            row = connection.execute(
                "SELECT value FROM app_meta WHERE key=?",
                ("global_system_config",),
            ).fetchone()
        finally:
            connection.close()
    finally:
        database_copy.unlink(missing_ok=True)
    if not row:
        raise RuntimeError("global_system_config is unavailable")
    payload = open_system_config_payload(json.loads(row[0]))
    config = SystemConfig.model_validate(payload)
    config.api.require_key()
    return config


async def _generate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.database, args.secret_file, output_dir)

    from backend.services.method_decision_cards import (
        guidance_for_query,
        search_cards,
    )
    from backend.utils.llm_client import OpenAICompatibleClient
    from backend.utils.formatting import safe_parse_json

    model = config.models.bioinformatics_model
    temperature = config.temperatures.bioinformatics
    system_prompt = config.prompts.bioinformatics_prompt
    client = OpenAICompatibleClient(
        api_key=config.api.api_key,
        base_url=config.api.llm_base_url,
        proxy_url=config.api.proxy_url,
        session_id="method-guidance-b2",
        agent_name="bioinformatics-evaluation",
    )
    roots = {"current": args.current_data, "legacy": args.legacy_data}
    selected_cases = CASES[: args.case_limit] if args.case_limit else CASES
    run_meta = {
        "evaluation": "method-guidance-b2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "temperature": temperature,
        "base_url_host": config.api.llm_base_url.split("//", 1)[-1].split("/", 1)[0],
        "system_prompt_sha256": _sha256(system_prompt),
        "order_seed_sha256": _sha256(ORDER_SEED),
        "case_count": len(selected_cases),
        "source_commit": args.source_commit,
        "replay_mode": (
            "current system prompt plus exact persisted Bioinformatics planning "
            "request; guided condition inserts the current card block"
        ),
    }
    _write_json(output_dir / "run.json", run_meta)

    try:
        for case_id, store, session_id, line_number in selected_cases:
            pair_path = output_dir / "pairs" / f"{case_id}.json"
            mapping_path = output_dir / "mapping" / f"{case_id}.json"
            if args.resume and pair_path.exists() and mapping_path.exists():
                print(f"{case_id}: already complete", flush=True)
                continue

            log_path = roots[store] / "sessions" / session_id / "logs" / "llm.jsonl"
            row = _read_jsonl_line(log_path, line_number)
            if row.get("agent") != "bioinformatics" or row.get("interaction_type") != "chat":
                raise ValueError(f"{case_id} is not a Bioinformatics chat call")
            baseline_prompt = _planning_prompt(row)
            hits = search_cards(baseline_prompt)
            guidance = guidance_for_query(baseline_prompt)
            guided_prompt = _insert_guidance(
                baseline_prompt,
                GUIDANCE_HEADER.format(guidance=guidance),
            ) if guidance else baseline_prompt
            prompts = {"off": baseline_prompt, "on": guided_prompt}
            order = _condition_order(case_id)
            labels = {condition: label for condition, label in zip(order, ("A", "B"))}
            outputs: dict[str, dict] = {}

            for condition in order:
                label = labels[condition]
                started = time.perf_counter()
                request_prompt = prompts[condition]
                attempts = []
                response = None
                parsed = None
                for attempt_number in (1, 2):
                    response = await client.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": request_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        max_retries=2,
                    )
                    parsed = safe_parse_json(response.content, default=None)
                    parseable = bool(
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("steps"), list)
                        and parsed["steps"]
                    )
                    attempts.append({
                        "attempt": attempt_number,
                        "content": response.content,
                        "content_sha256": _sha256(response.content),
                        "prompt_sha256": _sha256(request_prompt),
                        "usage": response.usage,
                        "parseable_plan": parseable,
                    })
                    if parseable:
                        break
                    request_prompt = (
                        prompts[condition]
                        + "\n\nIMPORTANT: Return valid JSON only. The top-level "
                        "object must contain a non-empty steps array. Numeric IDs "
                        "and depends_on entries must be numbers. Do not include "
                        "analysis, commentary, or Markdown outside the JSON object. "
                        f"[retry:b2-{case_id}-{label.lower()}]"
                    )
                latency_ms = round((time.perf_counter() - started) * 1000)
                assert response is not None
                outputs[label] = {
                    "content": response.content,
                    "content_sha256": _sha256(response.content),
                    "model": response.model,
                    "usage": response.usage,
                    "latency_ms": latency_ms,
                    "prompt_sha256": _sha256(prompts[condition]),
                    "attempts": attempts,
                    "parseable_plan": bool(
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("steps"), list)
                        and parsed["steps"]
                    ),
                    "step_count": len(parsed.get("steps", []))
                    if isinstance(parsed, dict)
                    else 0,
                }
                print(
                    f"{case_id} {label}: complete in {latency_ms / 1000:.1f}s",
                    flush=True,
                )

            _write_json(
                pair_path,
                {
                    "case_id": case_id,
                    "source_pointer": f"{session_id}/logs/llm.jsonl:{line_number}",
                    "source_timestamp": row.get("timestamp"),
                    "frozen_request_sha256": _sha256(baseline_prompt),
                    "retrieved_card_ids": [card["id"] for card in hits],
                    "guidance_chars": len(guidance),
                    "outputs": outputs,
                },
            )
            _write_json(
                mapping_path,
                {
                    "case_id": case_id,
                    "condition_by_label": {
                        label: condition for condition, label in labels.items()
                    },
                },
            )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-data", type=Path, required=True)
    parser.add_argument("--legacy-data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(_generate(parser.parse_args()))


if __name__ == "__main__":
    main()
