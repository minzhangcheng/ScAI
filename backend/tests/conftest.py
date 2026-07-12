import os

# SCAI_DATA_DIR is guaranteed writable by the repository-root conftest.py, which
# runs before the ``backend`` package (and its import-time data-dir creation) is
# imported. Keep a defensive setdefault for direct/standalone imports of this
# module.
os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_test_data")

import pytest_asyncio
from backend.utils.db import AppDB


@pytest_asyncio.fixture
async def app_db(tmp_path):
    """Provide an initialized AppDB instance for testing."""
    db = AppDB(tmp_path / "app.db")
    await db.init()
    return db
