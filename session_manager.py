#!/usr/bin/env python3
"""Session management CLI using scai-client.

Usage:
    python session_manager.py list
    python session_manager.py import <archive_path>
    python session_manager.py export <session_id> [output_path]
"""
import argparse
import sys

from scai_client import Client


def list_sessions(client: Client) -> None:
    """List all sessions and their status."""
    sessions = client.list_sessions()
    if not sessions:
        print("No sessions found.")
        return

    print(f"{'Session ID':<12} {'Name':<30} {'Status':<20} {'Rounds':<8} {'Created':<20}")
    print("-" * 95)
    for s in sessions:
        print(
            f"{s.get('id', 'N/A'):<12} "
            f"{s.get('name', 'N/A')[:28]:<30} "
            f"{s.get('status', 'N/A'):<20} "
            f"{s.get('current_round', 0):<8} "
            f"{s.get('created_at', 'N/A')[:19]:<20}"
        )


def _format_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _print_progress(info: dict) -> None:
    """on_progress callback — single-line live status to stderr."""
    phase = info.get("phase") or info.get("status") or "pending"
    bp = info.get("bytes_processed") or 0
    bt = info.get("bytes_total") or 0
    fp = info.get("files_processed") or 0
    cf = info.get("current_file") or ""
    pct = f"{(bp / bt * 100):5.1f}%" if bt > 0 else "  ?  "
    line = f"[{phase:10s}] {pct}  {_format_bytes(bp)}/{_format_bytes(bt)}  files={fp}"
    if cf:
        line += f"  {cf[-50:]}"
    # \r overwrite + pad to clear leftover chars from the previous line
    sys.stderr.write("\r" + line.ljust(120))
    sys.stderr.flush()


def import_session(client: Client, archive_path: str, *, trust: bool = False) -> str:
    """Import a session from a local tar.gz archive.

    No client-side timeout — multi-GB extractions can legitimately take >1h.
    Server-side cancel is still available via the cancel endpoint.
    """
    print(f"Importing session from: {archive_path} (trust={trust})")
    try:
        session_id = client.import_session(
            archive_path,
            wait=True,
            timeout=None,
            trust=trust,
            on_progress=_print_progress,
        )
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()
    print(f"Session imported successfully. Session ID: {session_id}")
    return session_id


def export_session(
    client: Client,
    session_id: str,
    output_path: str | None = None,
    *,
    include_runtime: bool = False,
) -> None:
    """Export a session to a local tar.gz archive.

    ``include_runtime`` adds sandbox session pickles (live kernel state +
    per-snapshot pkl.zst). Default False produces a smaller, portable archive.
    """
    print(f"Exporting session: {session_id} (include_runtime={include_runtime})")
    path = client.export_session(session_id, output_path, include_runtime=include_runtime)
    print(f"Session exported successfully to: {path}")


def main():
    parser = argparse.ArgumentParser(description="ScAI Session Manager")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Server URL"
    )
    parser.add_argument(
        "--api-key", required=True, help="API key (starts with scai_)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all sessions and their status")

    imp = subparsers.add_parser("import", help="Import session from archive")
    imp.add_argument("archive", help="Path to .tar.gz archive")
    imp.add_argument(
        "--trust",
        action="store_true",
        help="Trust the archive and restore sandbox session pickles "
             "(dill.load = arbitrary code execution; only for archives you control)",
    )

    exp = subparsers.add_parser("export", help="Export session to archive")
    exp.add_argument("session_id", help="Session ID to export")
    exp.add_argument("output", nargs="?", help="Output file path (optional)")
    exp.add_argument(
        "--include-runtime",
        action="store_true",
        help="Include sandbox session pickles (live kernel state + snapshot pkl.zst). "
             "Larger archive, only re-importable on a trusted instance with --trust.",
    )

    args = parser.parse_args()

    # Use 24h timeout for large file uploads (41GB+ archives can take hours).
    # The import_session method also has its own poll timeout (None = no limit).
    client = Client(base_url=args.url, api_key=args.api_key, timeout=86400)

    try:
        if args.command == "list":
            list_sessions(client)
        elif args.command == "import":
            import_session(client, args.archive, trust=args.trust)
        elif args.command == "export":
            export_session(
                client, args.session_id, args.output,
                include_runtime=args.include_runtime,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()