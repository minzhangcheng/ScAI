from backend.tests.split_facade import patch_facade
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("route_name, session_id", [
    ("export_state", "test_session_123"), ("export_prepare", "test_session_456"),
])
async def test_export_cleans_temp_file_on_build_failure(tmp_path, route_name, session_id):
    """Both export routes release their temporary archive if building fails."""
    from backend.routes import export_import

    export_route = getattr(export_import, route_name)

    isolated_temp = tmp_path / "upload_temp"
    isolated_temp.mkdir()

    request = MagicMock()
    request.state = MagicMock()
    request.state.user_id = "test_user"

    with patch_facade("backend.routes.export_import", "UPLOAD_TEMP_DIR", isolated_temp):
        with patch_facade("backend.routes.export_import", "require_session_ownership") as mock_own:
            mock_state = MagicMock()
            mock_state.id = session_id
            mock_state.name = "TestSession"
            mock_own.return_value = mock_state

            with patch_facade(
                "backend.routes.export_import", "reserve_export_record"
            ) as reserve:
                reserve.side_effect = lambda token, **kwargs: {
                    "token": token,
                    "session_id": kwargs["session_id"],
                    "generation": 1,
                    "state_ref": kwargs["state"],
                }
                with patch_facade("backend.routes.export_import", "bind_export_path"):
                    with patch_facade("backend.routes.export_import", "async_save_session"):
                        with patch_facade(
                            "backend.routes.export_import",
                            "run_in_owned_thread",
                            side_effect=RuntimeError("tar build failed"),
                        ):
                            with patch_facade("backend.routes.export_import", "validate_session_id"):
                                with pytest.raises(RuntimeError, match="tar build failed"):
                                    await export_route(session_id, request)

    leftover = list(isolated_temp.glob("*.tar.gz"))
    assert len(leftover) == 0, f"Leaked temp files after {route_name} failure: {leftover}"
