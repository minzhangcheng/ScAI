"""The unified ``scai`` command-line entry point."""
import argparse
import os
import sys
from typing import Optional, Sequence

from .. import __version__
from ..errors import ScaiError
from . import commands
from ._credentials import key_path, read_key
from ._format import log


DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 3600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scai", description="Create and manage ScAI analyses.")
    parser.add_argument("--version", action="version", version=f"scai-client {__version__}")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SCAI_BASE_URL", DEFAULT_BASE_URL),
        help="server URL",
    )
    parser.add_argument("--api-key", default=None, help="API key (env SCAI_API_KEY)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    sub = parser.add_subparsers(dest="command", required=True)

    login_parser = sub.add_parser("login")
    login_parser.add_argument("--username", default=None)
    login_parser.add_argument("--password", default=None)
    login_parser.add_argument("--label", default="scai cli")
    login_parser.add_argument("--print-jwt", action="store_true")
    login_parser.set_defaults(handler=commands.cmd_login)

    new = sub.add_parser("new")
    new.add_argument("config")
    new.add_argument("--name", default=None)
    new.add_argument("--round-wait", type=int, default=None)
    new.add_argument("--max-rounds", type=int, default=None)
    new.add_argument("--model", action="append", default=None, metavar="AGENT=MODEL")
    new.add_argument("--dry-run", action="store_true")
    new.add_argument("--no-start", action="store_true")
    new.add_argument("--watch", action="store_true")
    new.add_argument("--poll", type=int, default=30)
    new.add_argument("--watch-timeout", type=int, default=None)
    new.set_defaults(handler=commands.cmd_new)

    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=commands.cmd_list)

    status = sub.add_parser("status")
    status.add_argument("session_id")
    status.set_defaults(handler=commands.cmd_status)

    watch = sub.add_parser("watch")
    watch.add_argument("session_id")
    watch.add_argument("--poll", type=int, default=30)
    watch.add_argument("--watch-timeout", type=int, default=None)
    watch.set_defaults(handler=commands.cmd_watch)

    start = sub.add_parser("start")
    start.add_argument("session_id")
    start.set_defaults(handler=commands.cmd_start)
    stop = sub.add_parser("stop")
    stop.add_argument("session_id")
    stop.set_defaults(handler=commands.cmd_stop)
    delete = sub.add_parser("delete")
    delete.add_argument("session_id")
    delete.set_defaults(handler=commands.cmd_delete)
    config = sub.add_parser("config")
    config.add_argument("session_id")
    config.set_defaults(handler=commands.cmd_config)
    artifacts = sub.add_parser("artifacts")
    artifacts.add_argument("session_id")
    artifacts.add_argument("--download", default=None)
    artifacts.add_argument("--output", default=None)
    artifacts.set_defaults(handler=commands.cmd_artifacts)
    importing = sub.add_parser("import")
    importing.add_argument("archive")
    importing.add_argument("--trust", action="store_true")
    importing.set_defaults(handler=commands.cmd_import)
    exporting = sub.add_parser("export")
    exporting.add_argument("session_id")
    exporting.add_argument("output", nargs="?", default=None)
    exporting.add_argument("--include-runtime", action="store_true")
    exporting.set_defaults(handler=commands.cmd_export)
    return parser


def resolve_api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("SCAI_API_KEY", "") or read_key()
    if not key:
        raise ScaiError(
            f"no credential: run `scai login`, pass --api-key, or set SCAI_API_KEY; "
            f"stored key path is {key_path()}"
        )
    return key


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        return args.handler(args)
    except (ScaiError, FileNotFoundError) as exc:
        log(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        log("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
