"""Authentication exemptions match complete route segments only."""
import pytest

from backend.middleware.auth import _is_auth_optional


@pytest.mark.parametrize("path, expected", [
    ('/api/session/x/report/../etc', False),
    ('/api/session/x/./export', False),
    ('/api/session/abc/export', False),
    ('/api/session/abc/export/prepare', False),
    ('/api/session/abc/export/download', True),
    ('/api/session/abc/notexport/extra', False),
    ('/api/session/abc/exports', False),
    ('/api/session/abc/myexport/data', False),
    ('/api/session/abc/report/html', True),
    ('/api/session/abc/report/figure/x.png', True),
    ('/api/session/abc/something/report/html', False),
    ('/api/session/abc/exploration-report/html', True),
    ('/api/session/abc/detailed-report/html', True),
    ('/api/session/abc/x/exploration-report/html', False),
    ('/api/session/abc/figure/fid/png', True),
    ('/api/session/abc/x/figure/fid/png', False),
    ('/api/file/something', True),
    ('/api/download/something', True),
    ('/api/file', True),
    ('/api/download', True),
    ('/api/admin/users', False),
    ('/api/session/abc/detail', False),
])
def test_auth_optional_route_boundaries(path, expected):
    assert _is_auth_optional(path) is expected
