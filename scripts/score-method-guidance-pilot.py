#!/usr/bin/env python3
"""Blindly score frozen A/B plan pairs for the method-guidance pilot."""
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


DIMENSIONS = (
    "question_alignment",
    "observed_fact_use",
    "non_equivalent_method_distinction",
    "material_parameter_rationale",
    "assumptions_uncertainty_limitations",
    "executable_specificity",
)
LEGACY_SESSION_ID = "7bc10a2b-b8f7-4c1a-ace8-cde783266f2e"
RUBRIC = """
You are the PI scientific reviewer for a blinded paired evaluation. Review the
frozen planning request and plans A and B without guessing which condition is
the intervention. Score each plan independently from 0 (materially deficient)
to 2 (strong) on exactly these dimensions:

1. question_alignment: alignment between method and requested question;
2. observed_fact_use: use of observed design/input facts;
3. non_equivalent_method_distinction: distinction between non-equivalent approaches;
4. material_parameter_rationale: rationale for material parameters;
5. assumptions_uncertainty_limitations: visibility of assumptions, uncertainty, and limitations;
6. executable_specificity: executable specificity without invented data or capabilities.

Then record preference as A, B, or tie. Do not reward length by itself. Record a
critical regression only if a plan answers a different question, invents a
material input/capability, loses a PI objective, or prescribes a universal
scientific threshold/method without support. Use an empty list when none exists.

Return valid JSON only with this shape:
{
  "scores": {
    "A": {"question_alignment": 0, "observed_fact_use": 0,
      "non_equivalent_method_distinction": 0, "material_parameter_rationale": 0,
      "assumptions_uncertainty_limitations": 0, "executable_specificity": 0},
    "B": {"question_alignment": 0, "observed_fact_use": 0,
      "non_equivalent_method_distinction": 0, "material_parameter_rationale": 0,
      "assumptions_uncertainty_limitations": 0, "executable_specificity": 0}
  },
  "preference": "A",
  "critical_regressions": {"A": [], "B": []},
  "rationale": "concise case-level comparison",
  "uncertainties": []
}
""".strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_jsonl_line(path: Path, line_number: int) -> dict:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index == line_number:
                return json.loads(line)
    raise ValueError(f"{path} has no line {line_number}")


def _planning_prompt(row: dict) -> str:
    user_messages = [
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    ]
    if len(user_messages) != 1 or not user_messages[0].strip():
        raise ValueError("frozen call does not contain exactly one planning request")
    return user_messages[0]


def _load_config(database: Path, secret_file: Path, output_dir: Path):
    os.environ["SCAI_CONFIG_SECRET"] = secret_file.read_text(encoding="utf-8").strip()
    os.environ.setdefault("SCAI_DATA_DIR", str(output_dir))

    from backend.config._config import SystemConfig
    from backend.utils.secret_redaction import open_system_config_payload

    database_copy = output_dir / "pi-config-snapshot.db"
    shutil.copy2(database, database_copy)
    try:
        connection = sqlite3.connect(database_copy)
        try:
            row = connection.execute(
                "SELECT value FROM app_meta WHERE key=?", ("global_system_config",)
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


def _valid_review(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("preference") not in {"A", "B", "tie"}:
        return False
    scores = value.get("scores")
    regressions = value.get("critical_regressions")
    if not isinstance(scores, dict) or not isinstance(regressions, dict):
        return False
    for label in ("A", "B"):
        label_scores = scores.get(label)
        if not isinstance(label_scores, dict) or set(label_scores) != set(DIMENSIONS):
            return False
        if any(type(label_scores[name]) is not int or not 0 <= label_scores[name] <= 2 for name in DIMENSIONS):
            return False
        findings = regressions.get(label)
        if not isinstance(findings, list) or any(not isinstance(item, str) for item in findings):
            return False
    return isinstance(value.get("rationale"), str) and isinstance(
        value.get("uncertainties"), list
    )


def _source_request(pair: dict, current_data: Path, legacy_data: Path) -> str:
    pointer, line_text = pair["source_pointer"].rsplit(":", 1)
    session_id = pointer.split("/", 1)[0]
    root = legacy_data if session_id == LEGACY_SESSION_ID else current_data
    row = _read_jsonl_line(root / "sessions" / pointer, int(line_text))
    prompt = _planning_prompt(row)
    if _sha256(prompt) != pair["frozen_request_sha256"]:
        raise ValueError(f"frozen request hash mismatch for {pair['case_id']}")
    return prompt


async def _score(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    score_dir = output_dir / "blind-scores"
    pair_paths = sorted((output_dir / "pairs").glob("case-*.json"))
    if args.case_limit:
        pair_paths = pair_paths[: args.case_limit]
    if not pair_paths:
        raise RuntimeError("no generated pairs are available")

    config = _load_config(args.database, args.secret_file, output_dir)
    from backend.utils.formatting import safe_parse_json
    from backend.utils.llm_client import OpenAICompatibleClient

    model = config.models.pi_model
    temperature = config.temperatures.pi
    client = OpenAICompatibleClient(
        api_key=config.api.api_key,
        base_url=config.api.llm_base_url,
        proxy_url=config.api.proxy_url,
        session_id="method-guidance-b2-blind-review",
        agent_name="pi-evaluation",
    )
    _write_json(
        output_dir / "blind-review-run.json",
        {
            "evaluation": "method-guidance-b2-blind-review",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "temperature": temperature,
            "base_url_host": config.api.llm_base_url.split("//", 1)[-1].split("/", 1)[0],
            "pi_system_prompt_sha256": _sha256(config.prompts.pi_prompt),
            "rubric_sha256": _sha256(RUBRIC),
            "pair_count": len(pair_paths),
            "source_commit": args.source_commit,
            "mapping_read_during_review": False,
        },
    )

    try:
        for pair_path in pair_paths:
            score_path = score_dir / pair_path.name
            if args.resume and score_path.exists():
                print(f"{pair_path.stem}: already scored", flush=True)
                continue
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            frozen_request = _source_request(pair, args.current_data, args.legacy_data)
            review_input = (
                f"{RUBRIC}\n\n## Frozen planning request\n{frozen_request}"
                f"\n\n## Plan A\n{pair['outputs']['A']['content']}"
                f"\n\n## Plan B\n{pair['outputs']['B']['content']}"
            )
            request_prompt = review_input
            attempts = []
            parsed = None
            response = None
            started = time.perf_counter()
            for attempt_number in (1, 2):
                response = await client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": config.prompts.pi_prompt},
                        {"role": "user", "content": request_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    max_retries=2,
                )
                parsed = safe_parse_json(response.content, default=None)
                valid = _valid_review(parsed)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "content": response.content,
                        "content_sha256": _sha256(response.content),
                        "prompt_sha256": _sha256(request_prompt),
                        "usage": response.usage,
                        "valid_review": valid,
                    }
                )
                if valid:
                    break
                request_prompt = (
                    review_input
                    + "\n\nIMPORTANT: Return only valid JSON matching the exact rubric "
                    + "shape and allowed score/preference values. "
                    + f"[retry:b2-review-{pair['case_id']}]"
                )
            latency_ms = round((time.perf_counter() - started) * 1000)
            if response is None or not _valid_review(parsed):
                raise RuntimeError(f"invalid PI review for {pair['case_id']}")
            _write_json(
                score_path,
                {
                    "case_id": pair["case_id"],
                    "pair_sha256": _sha256(pair_path.read_text(encoding="utf-8")),
                    "review": parsed,
                    "model": response.model,
                    "usage": response.usage,
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                },
            )
            print(f"{pair['case_id']}: scored in {latency_ms / 1000:.1f}s", flush=True)
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
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(_score(parser.parse_args()))


if __name__ == "__main__":
    main()
