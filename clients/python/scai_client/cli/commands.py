"""Handlers for the retained ``scai`` commands."""
import argparse
import getpass
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..client import Client, login, verify_exported_bindings
from ..config import AGENT_NAMES, load_config, resolve_data_files
from ..errors import ScaiError
from ._credentials import write_key
from ._format import Progress, human, log, session_rows, table


@contextmanager
def _client(args: argparse.Namespace) -> Iterator[Client]:
    from . import resolve_api_key
    client = Client(args.base_url, api_key=resolve_api_key(args), timeout=args.timeout)
    try:
        yield client
    finally:
        client.close()


def cmd_login(args: argparse.Namespace) -> int:
    username = args.username or os.environ.get("SCAI_USERNAME", "") or input("username: ").strip()
    password = args.password or os.environ.get("SCAI_PASSWORD", "") or getpass.getpass("password: ")
    if not username or not password:
        raise ScaiError("a username and a password are both required")
    account = login(args.base_url, username, password, timeout=args.timeout)
    if args.print_jwt:
        print(account["access_token"])
        return 0
    with Client(args.base_url, jwt=account["access_token"], timeout=args.timeout) as client:
        created = client.create_api_key(args.label)
    path = write_key(created["key"])
    log(f"API key stored in {path} (mode 600)")
    return 0


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    analysis = config.setdefault("analysis", {})
    if args.name:
        analysis["name"] = args.name
    if args.round_wait is not None:
        analysis["round_wait_seconds"] = args.round_wait
    if args.max_rounds is not None:
        analysis["max_rounds"] = args.max_rounds
    for assignment in args.model or []:
        agent, separator, model = assignment.partition("=")
        if not separator or agent not in AGENT_NAMES:
            raise ScaiError(f"--model expects <agent>=<model>; got {assignment!r}")
        config.setdefault("agents", {}).setdefault(agent, {})["model"] = model


def cmd_new(args: argparse.Namespace) -> int:
    config, config_path = load_config(args.config)
    _apply_overrides(config, args)
    files = resolve_data_files(config, config_path.parent)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    with _client(args) as client:
        job_id, manifest = client.create_analysis(config)
        ids = Client._validate_placeholder_manifest(config, files, manifest)
        for path, file_id in zip(files, ids):
            client.upload_data_file(job_id, path, placeholder_id=file_id)
        verify_exported_bindings(config, client.export_config(job_id), files)
        if args.no_start:
            print(job_id)
            return 0
        client.start_analysis(job_id)
        print(job_id)
        if args.watch:
            return cmd_watch(argparse.Namespace(
                base_url=args.base_url, api_key=args.api_key, timeout=args.timeout,
                session_id=job_id, poll=args.poll,
                watch_timeout=args.watch_timeout,
            ))
    return 0


def _watch(client: Client, session_id: str, poll: int, timeout: Optional[int]) -> int:
    start = time.time()
    while True:
        status = client.get_job(session_id).status()
        if status.get("status") in {"completed", "failed", "stopped"}:
            return 0 if status.get("status") == "completed" else 1
        if timeout is not None and time.time() - start > timeout:
            return 2
        time.sleep(poll)


def cmd_list(args: argparse.Namespace) -> int:
    with _client(args) as client:
        values = client.list_sessions()
    if args.json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
    elif values:
        table(session_rows(values), ["ID", "STATUS", "ROUND", "UPDATED", "NAME"])
    else:
        log("no analyses")
    return 0


def cmd_snapshots(args: argparse.Namespace) -> int:
    """List, restore or delete a session's decision snapshots.

    There is no ``create``: a snapshot is taken automatically before each
    decision an analysis asks about itself, and a command that made one on
    request would be a second producer with different semantics.
    """
    with _client(args) as client:
        if args.action == "restore":
            new_id = client.restore_snapshot(
                args.session_id, args.snapshot_id, wait=not args.no_wait,
            )
            print(new_id)
            return 0
        if args.action == "delete":
            client.delete_snapshot(args.session_id, args.snapshot_id)
            log(f"deleted {args.snapshot_id}")
            return 0

        listing = client.list_snapshots(args.session_id)
        if args.json:
            print(json.dumps(listing, indent=2, ensure_ascii=False))
            return 0
        if not listing.get("enabled"):
            log("snapshots are not enabled on this installation")
            return 0
        rows = listing.get("snapshots") or []
        if not rows:
            log("no snapshots")
        else:
            table(
                [
                    [
                        row["snapshot_id"],
                        row["gate"],
                        str(row["round"]),
                        str(row["retention_rank"]),
                        str(row["deletion_order"]),
                        human(row.get("bytes_total") or 0),
                        "superseded" if row.get("superseded") else "",
                        row["created_at"],
                    ]
                    for row in rows
                ],
                ["ID", "DECISION", "ROUND", "KEEP", "DROP", "SIZE", "", "CREATED"],
            )
            # What the session occupies, not the sum of the column above it:
            # snapshots share a dump through one inode and share one copy of
            # the inputs, so the total is smaller than the parts and is the
            # number that changes when one is given up.
            log(f"{human(listing.get('bytes_total') or 0)} on disk for this session")
        for row in listing.get("failed") or []:
            log(f"no snapshot for {row['snapshot_id']}: {row['reason']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with _client(args) as client:
        print(json.dumps(client.get_job(args.session_id).status(), indent=2, ensure_ascii=False))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    with _client(args) as client:
        return _watch(client, args.session_id, args.poll, args.watch_timeout)


def cmd_start(args: argparse.Namespace) -> int:
    with _client(args) as client:
        body = client.start_analysis(args.session_id)
    log(str(body.get("status", "")))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    with _client(args) as client:
        body = client.stop_session(args.session_id)
    print(json.dumps(body, ensure_ascii=False))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    with _client(args) as client:
        body = client.get_job(args.session_id).delete()
    print(json.dumps(body, ensure_ascii=False))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    with _client(args) as client:
        body = client.export_config(args.session_id)
    print(json.dumps(body, indent=2, ensure_ascii=False))
    return 0


def cmd_artifacts(args: argparse.Namespace) -> int:
    with _client(args) as client:
        job = client.get_job(args.session_id)
        if args.download:
            destination = Path(args.output or Path(args.download).name)
            job.download_artifact(args.download, destination)
            return 0
        values = job.artifacts()
    table([[str(value.get("name", "")), human(value.get("size") or 0)] for value in values], ["NAME", "SIZE"])
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    progress = Progress()
    with _client(args) as client:
        try:
            session_id = client.import_session(args.archive, trust=args.trust, on_progress=progress)
        finally:
            progress.done()
    print(session_id)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with _client(args) as client:
        path = client.export_session(args.session_id, args.output, include_runtime=args.include_runtime)
    print(path)
    return 0
