#!/usr/bin/env python3
"""Run the locked frontend install and Vitest with isolated, auditable state."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from scai_frontend_parts.artifacts import FrontendError, json_hash, sha256_file
from scai_frontend_parts.identity import resolve_run_start_head
from scai_frontend_parts.npm import (
    REGISTRY, check_secret_key, normalize_bin, npm_ls_inventory, npm_selector_applies,
    parse_npm_config_sources, parse_npm_config_values, sanitize_output, tokenized,
    validate_npm_config_agreement, walk_npm_tree,
)
from scai_frontend_parts.results import parse_vitest
from scai_frontend_parts.runner import FrontendRun


def main(argv: list[str]) -> int:
    if argv != ["--install-and-test"]:
        print("Usage: scripts/scai-frontend-test --install-and-test", file=sys.stderr)
        return 2
    repo = Path(os.environ.get("SCAI_FRONTEND_RUNNER_PATH", __file__)).resolve().parents[1]
    run = FrontendRun(repo)

    def interrupted(signum: int, _frame: Any) -> None:
        run.failure = {"label": "frontend signal", "message": f"received signal {signum}; restoring source link"}
        raise KeyboardInterrupt

    for signal_number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, interrupted)
    status = "failed"
    try:
        run.prepare()
        run.run_install()
        run.link_and_test()
        status = "application_failure" if run.application_failure else "passed"
    except FrontendError as exc:
        run.failure = {"label": exc.label, "message": str(exc)}
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        run.failure = {"label": "frontend runner", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        close_error: FrontendError | None = None
        try:
            close_error = run.close()
        except Exception as exc:
            close_error = FrontendError("frontend close", f"{type(exc).__name__}: {exc}")
            run.failure = {"label": close_error.label, "message": str(close_error)}
    if close_error is not None:
        status = "failed"
    try:
        run.finish(status)
    except Exception as exc:
        status = "failed"
        run.failure = {"label": "frontend evidence", "message": f"{type(exc).__name__}: {exc}"}
        print(f"error: frontend evidence: {exc}", file=sys.stderr)
    finally:
        run.release_locks()
    if status == "passed":
        return 0
    print(f"error: frontend {status}: {run.failure or 'unknown failure'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
