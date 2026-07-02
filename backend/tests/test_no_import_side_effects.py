import logging
import importlib


def test_import_does_not_reconfigure_root_logging():
    before = list(logging.getLogger().handlers)
    import backend.main
    importlib.reload(backend.main)
    after = list(logging.getLogger().handlers)
    assert before == after
