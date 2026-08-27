"""Generation-bound lifecycle ownership for session export archives."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from starlette.responses import (
    FileResponse,
    Headers,
    MalformedRangeHeader,
    MutableHeaders,
    PlainTextResponse,
    RangeNotSatisfiable,
)

from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.state import (
    prepared_exports,
    prepared_exports_lock,
    sessions,
)
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_thread,
)


logger = logging.getLogger(__name__)


from backend.services.export_lifecycle_parts.archive import (
    EXPORT_OWNERSHIP_MARKER_SUFFIX,
    _SECURE_COPY_CHUNK_BYTES,
    _ZIP_MEMBER_OVERHEAD_BYTES,
    _Identity,
    SecureArchiveError,
    _SelectedArchiveMember,
    _secure_capabilities,
    _archive_identity,
    _inode_identity,
    _identity_matches,
    _relative_parts,
    _path_identity,
    _selected_member,
    _iter_regular_files,
    _select_archive_members,
    _open_root,
    _verify_root,
    _open_member,
    _verify_member,
    _reserve_space,
    _write_fd_exact,
    _validate_owned_destination,
    _verify_owned_destination,
    _unlink_owned_export_paths,
    build_secure_zip,
    copy_secure_file,
    export_ownership_marker_path,
    _regular_file_identity,
    create_export_ownership_marker,
    export_ownership_marker_matches,
    unlink_export_archive_and_marker,
)

from backend.services.export_lifecycle_parts.records import (
    _current_task,
    _record_generation,
    export_record_matches_generation,
    reserve_export_record,
    generation_bound_export_prefix,
    bind_export_path,
    publish_export_ready,
    claim_export_stream,
    _begin_stream,
    cleanup_export_record,
    abandon_export_record,
    mark_session_exports_for_deletion,
    cleanup_session_export_records,
)

from backend.services.export_lifecycle_parts.responses import (
    LifecycleFileResponse,
    OwnedArchiveFileResponse,
)

SecureArchiveError.__module__ = __name__
LifecycleFileResponse.__module__ = __name__
OwnedArchiveFileResponse.__module__ = __name__
