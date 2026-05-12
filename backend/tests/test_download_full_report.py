import inspect
import pytest

from backend.routes.reports import download_full_report


@pytest.mark.asyncio
async def test_download_full_report_does_not_call_ensure_session_loaded_again():
    """download_full_report must use the state from Depends() and NOT call
    ensure_session_loaded again."""
    source = inspect.getsource(download_full_report)
    assert "ensure_session_loaded" not in source, (
        "download_full_report should not call ensure_session_loaded; "
        "it should use the state parameter from Depends(require_session_ownership)"
    )


@pytest.mark.asyncio
async def test_download_full_report_uses_state_name_for_filename():
    """download_full_report should use state.name directly for the filename."""
    source = inspect.getsource(download_full_report)
    assert "state.name" in source, (
        "download_full_report should reference state.name for the filename"
    )