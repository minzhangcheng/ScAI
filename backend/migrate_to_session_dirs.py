#!/usr/bin/env python3
"""
Migration script: move from type-based directory layout to session-centric layout.

Old structure:
    data/uploads/{session_id}/...
    data/states/{session_id}*
    data/workspace/{session_id}/...
    data/sessions/{session_id}.json
    data/reports/{session_id}/...
    data/logs/llm/sessions/{session_id}.jsonl

New structure:
    data/sessions/{session_id}/
        session.json
        workspace/
        reports/
        uploads/
        states/
        logs/llm.jsonl

Usage:
    python -m backend.migrate_to_session_dirs [--dry-run]

Run this script while the server is stopped.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import (
    DATA_BASE_DIR, UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR,
    SESSION_DIR, REPORTS_DIR, LOG_DIR,
    get_session_dir, get_session_metadata_path, get_session_workspace,
    get_session_states, get_session_uploads, get_session_reports, get_session_logs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def discover_session_ids() -> set:
    """Discover all session IDs from legacy directories."""
    ids = set()

    # From flat JSON files in sessions/
    if SESSION_DIR.exists():
        for f in SESSION_DIR.glob("*.json"):
            if f.is_file():
                ids.add(f.stem)

    # From upload subdirectories
    if UPLOAD_DIR.exists():
        for d in UPLOAD_DIR.iterdir():
            if d.is_dir():
                ids.add(d.name)

    # From workspace subdirectories
    if WORKSPACE_DIR.exists():
        for d in WORKSPACE_DIR.iterdir():
            if d.is_dir():
                ids.add(d.name)

    # From report subdirectories
    if REPORTS_DIR.exists():
        for d in REPORTS_DIR.iterdir():
            if d.is_dir():
                ids.add(d.name)

    # From state files
    if STATE_DIR.exists():
        for f in STATE_DIR.iterdir():
            if f.is_file():
                # Filenames like: {session_id}.ipynb, {session_id}_state.json, etc.
                name = f.stem
                # Strip known suffixes
                for suffix in ("_state", "_session", "_bioinfo_checkpoint", "_report_checkpoint"):
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                ids.add(name)

    # Filter out obviously non-session entries
    ids = {sid for sid in ids if sid and "/" not in sid and "\\" not in sid and ".." not in sid}
    return ids


def _move(src: Path, dst: Path, dry_run: bool):
    """Move src to dst, creating parent dirs."""
    if not src.exists():
        return
    if dry_run:
        logger.info(f"  [DRY-RUN] MOVE {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            # Merge: move contents
            for item in src.iterdir():
                target = dst / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                else:
                    logger.warning(f"  SKIP (already exists): {target}")
            # Remove empty source
            try:
                src.rmdir()
            except OSError:
                pass
        else:
            shutil.move(str(src), str(dst))
    else:
        if not dst.exists():
            shutil.move(str(src), str(dst))
        else:
            logger.warning(f"  SKIP (already exists): {dst}")


def migrate_session(session_id: str, dry_run: bool):
    """Migrate a single session to the new directory layout."""
    session_root = get_session_dir(session_id)

    # Check if already migrated
    new_meta = get_session_metadata_path(session_id)
    if new_meta.exists():
        logger.info(f"  Already migrated (session.json exists), skipping file moves")
        return

    logger.info(f"  Migrating session: {session_id}")

    if not dry_run:
        session_root.mkdir(parents=True, exist_ok=True)

    # 1. Session metadata: sessions/{id}.json -> sessions/{id}/session.json
    legacy_meta = SESSION_DIR / f"{session_id}.json"
    _move(legacy_meta, new_meta, dry_run)

    # 2. Uploads: uploads/{id}/ -> sessions/{id}/uploads/
    legacy_uploads = UPLOAD_DIR / session_id
    _move(legacy_uploads, get_session_uploads(session_id), dry_run)

    # 3. Workspace: workspace/{id}/ -> sessions/{id}/workspace/
    legacy_workspace = WORKSPACE_DIR / session_id
    _move(legacy_workspace, get_session_workspace(session_id), dry_run)

    # 4. Reports: reports/{id}/ -> sessions/{id}/reports/
    legacy_reports = REPORTS_DIR / session_id
    _move(legacy_reports, get_session_reports(session_id), dry_run)

    # 5. State files: states/{id}* -> sessions/{id}/states/
    if STATE_DIR.exists():
        states_dest = get_session_states(session_id)
        if not dry_run:
            states_dest.mkdir(parents=True, exist_ok=True)
        for sf in STATE_DIR.glob(f"{session_id}*"):
            if sf.is_file():
                _move(sf, states_dest / sf.name, dry_run)

    # 6. LLM session log: logs/llm/sessions/{id}.jsonl -> sessions/{id}/logs/llm.jsonl
    legacy_llm_log = LOG_DIR / "llm" / "sessions" / f"{session_id}.jsonl"
    _move(legacy_llm_log, get_session_logs(session_id) / "llm.jsonl", dry_run)

    # 7. Remap embedded paths in session.json
    if not dry_run and new_meta.exists():
        try:
            text = new_meta.read_text(encoding="utf-8")
            old_upload = str(UPLOAD_DIR / session_id)
            old_workspace = str(WORKSPACE_DIR / session_id)
            old_reports = str(REPORTS_DIR / session_id)
            new_upload = str(get_session_uploads(session_id))
            new_workspace = str(get_session_workspace(session_id))
            new_reports = str(get_session_reports(session_id))

            updated = text
            updated = updated.replace(old_upload, new_upload)
            updated = updated.replace(old_workspace, new_workspace)
            updated = updated.replace(old_reports, new_reports)

            if updated != text:
                new_meta.write_text(updated, encoding="utf-8")
                logger.info(f"  Remapped embedded paths in session.json")
        except Exception as e:
            logger.exception("  Failed to remap paths in session.json")


def main():
    parser = argparse.ArgumentParser(description="Migrate data to session-centric layout")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    logger.info(f"Data directory: {DATA_BASE_DIR}")
    logger.info(f"Dry run: {args.dry_run}")

    session_ids = discover_session_ids()
    if not session_ids:
        logger.info("No sessions found to migrate.")
        return

    logger.info(f"Found {len(session_ids)} session(s) to migrate")

    for sid in sorted(session_ids):
        try:
            migrate_session(sid, args.dry_run)
        except Exception as e:
            logger.exception(f"Failed to migrate session {sid}")

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
