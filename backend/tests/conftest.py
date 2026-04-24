import os

os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_test_data")

import pytest_asyncio
from backend.utils.db import AppDB


@pytest_asyncio.fixture
async def app_db(tmp_path):
    """Provide an initialized AppDB instance for testing."""
    db = AppDB(tmp_path / "app.db")
    await db.init()
    return db
