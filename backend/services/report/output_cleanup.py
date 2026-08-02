"""Reset report-owned derivative asset directories before regeneration."""

from __future__ import annotations

import shutil
from pathlib import Path


DERIVED_REPORT_ASSET_DIRS = ("figures", "tables")


def reset_report_asset_directories(output_dir: str | Path) -> None:
    """Recreate report-owned asset directories without touching source evidence.

    Regeneration reuses the same report directory. Without an explicit reset,
    figures and tables copied by an older report survive and can be rediscovered
    or packaged even when the new report no longer references them. Only the two
    directories fully owned by report generation are removed; report JSON/HTML,
    archives, notebooks, and interaction records are left in place.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for directory_name in DERIVED_REPORT_ASSET_DIRS:
        target = root / directory_name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
