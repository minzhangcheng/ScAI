def test_setup_invalidates_admin_cache():
    from backend.middleware import auth as mw
    mw._set_admin_cache(False)
    mw.invalidate_admin_initialized_cache()
    assert mw._admin_initialized_cache is None or mw._admin_initialized_cache["value"] is None
