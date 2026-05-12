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


def import_session(client: Client, archive_path: str) -> str:
    """Import a session from a local tar.gz archive."""
    print(f"Importing session from: {archive_path}")
    session_id = client.import_session(archive_path, wait=True, timeout=300)
    print(f"Session imported successfully. Session ID: {session_id}")
    return session_id


def export_session(client: Client, session_id: str, output_path: str | None = None) -> None:
    """Export a session to a local tar.gz archive."""
    print(f"Exporting session: {session_id}")
    path = client.export_session(session_id, output_path)
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

    exp = subparsers.add_parser("export", help="Export session to archive")
    exp.add_argument("session_id", help="Session ID to export")
    exp.add_argument("output", nargs="?", help="Output file path (optional)")

    args = parser.parse_args()

    client = Client(base_url=args.url, api_key=args.api_key)

    try:
        if args.command == "list":
            list_sessions(client)
        elif args.command == "import":
            import_session(client, args.archive)
        elif args.command == "export":
            export_session(client, args.session_id, args.output)
    finally:
        client.close()


if __name__ == "__main__":
    main()