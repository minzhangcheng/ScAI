"""Small terminal-formatting helpers; progress is always sent to stderr."""
import sys
import time
from typing import Any, Dict, List, Sequence


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def human(size: float) -> str:
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KiB", "MiB", "GiB"):
        size /= 1024
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} GiB"


def table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)).rstrip())
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def session_rows(sessions: List[Dict[str, Any]]) -> List[List[str]]:
    return [
        [str(s.get("id", "")), str(s.get("status", "")),
         str(s.get("current_round", "")), str(s.get("updated_at", ""))[:19],
         str(s.get("name", ""))]
        for s in sessions
    ]


class Progress:
    def __init__(self) -> None:
        self._width = 0

    def __call__(self, info: Dict[str, Any]) -> None:
        phase = info.get("phase") or info.get("status") or "pending"
        processed = info.get("bytes_processed") or 0
        total = info.get("bytes_total") or 0
        percent = f"{processed / total * 100:5.1f}%" if total else "    ?"
        self.write(f"[{phase:>10}] {percent} {human(processed)}/{human(total)}")

    def write(self, line: str) -> None:
        sys.stderr.write("\r" + line.ljust(self._width))
        sys.stderr.flush()
        self._width = max(self._width, len(line))

    def done(self) -> None:
        if self._width:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._width = 0
