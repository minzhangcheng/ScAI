"""Tests for strict suffix matching in SESSION_AUTH_OPTIONAL_SUFFIXES.

P0-3: previous `suffix in path` substring check allowed bypass attacks like
/api/session/x/notexport/something matching "/export". Now anchored to the
session-id segment.
"""
from backend.middleware.auth import _is_auth_optional


class TestAuthOptionalSuffixes:
    # /export: terminal suffix and its own subpaths
    def test_export_exact(self):
        assert _is_auth_optional("/api/session/abc/export") is True

    def test_export_subpath_download(self):
        # /export/download is a real route — must remain auth-optional
        assert _is_auth_optional("/api/session/abc/export/download") is True

    def test_export_substring_in_middle_rejected(self):
        # Bypass attempt: /export embedded in middle of a different segment
        assert _is_auth_optional("/api/session/abc/notexport/extra") is False

    def test_export_substring_as_segment_part_rejected(self):
        # /exports (plural) is not /export — must not match
        assert _is_auth_optional("/api/session/abc/exports") is False

    def test_export_in_query_like_segment_rejected(self):
        # Path segment that merely contains the word "export"
        assert _is_auth_optional("/api/session/abc/myexport/data") is False

    # /report/: prefix-of-subpath after session id
    def test_report_html(self):
        assert _is_auth_optional("/api/session/abc/report/html") is True

    def test_report_figure(self):
        assert _is_auth_optional("/api/session/abc/report/figure/x.png") is True

    def test_report_substring_in_middle_rejected(self):
        # /report/ appearing further down the path is NOT the session-level route
        assert _is_auth_optional("/api/session/abc/something/report/html") is False

    # -report/: e.g. exploration-report/, annotation-report/, detailed-report/
    def test_exploration_report(self):
        assert _is_auth_optional("/api/session/abc/exploration-report/html") is True

    def test_detailed_report(self):
        assert _is_auth_optional("/api/session/abc/detailed-report/html") is True

    def test_dash_report_in_middle_rejected(self):
        assert _is_auth_optional("/api/session/abc/x/exploration-report/html") is False

    # /figure/: session-level figure endpoint
    def test_figure_png(self):
        assert _is_auth_optional("/api/session/abc/figure/fid/png") is True

    def test_figure_in_middle_rejected(self):
        # /figure/ buried under report/ is matched by /report/, but here we want
        # to ensure raw /figure/ doesn't match in arbitrary positions
        assert _is_auth_optional("/api/session/abc/x/figure/fid/png") is False

    # AUTH_OPTIONAL_PATHS prefix semantics (must keep working)
    def test_file_prefix_kept(self):
        assert _is_auth_optional("/api/file/something") is True

    def test_download_prefix_kept(self):
        assert _is_auth_optional("/api/download/something") is True

    # Leaf endpoints: the actual routes are /api/file and /api/download with the
    # path passed as a query string, so the exact path (no trailing slash) must
    # be auth-optional too — otherwise <img src>/download links 403.
    def test_file_exact_path(self):
        assert _is_auth_optional("/api/file") is True

    def test_download_exact_path(self):
        assert _is_auth_optional("/api/download") is True

    # Negative: unrelated paths
    def test_non_session_path(self):
        assert _is_auth_optional("/api/admin/users") is False

    def test_session_root_not_optional(self):
        assert _is_auth_optional("/api/session/abc/detail") is False


def test_path_traversal_does_not_match_suffix():
    assert _is_auth_optional("/api/session/x/report/../etc") is False
    assert _is_auth_optional("/api/session/x/./export") is False
