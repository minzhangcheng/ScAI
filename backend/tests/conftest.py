import os

# SCAI_DATA_DIR is guaranteed writable by the repository-root conftest.py, which
# runs before the ``backend`` package (and its import-time data-dir creation) is
# imported. Keep a defensive setdefault for direct/standalone imports of this
# module.
os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_test_data")

import pytest
import pytest_asyncio
from backend.utils.db import AppDB


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Pin asyncio fixture loop scope when unset (avoids pytest-asyncio deprecation)."""
    original_getini = config.getini

    def getini(name: str):
        value = original_getini(name)
        if name == "asyncio_default_fixture_loop_scope" and not value:
            return "function"
        return value

    config.getini = getini  # type: ignore[method-assign]


@pytest_asyncio.fixture
async def app_db(tmp_path):
    """Provide an initialized AppDB instance for testing."""
    db = AppDB(tmp_path / "app.db")
    await db.init()
    return db
