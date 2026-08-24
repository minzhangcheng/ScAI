from backend.tests.split_facade import binds, split_source


def test_h5ad_checkpoint_count_guard_is_removed():
    facade = "backend.agents.bioinformatics_agent"
    source = split_source(facade)

    assert binds(facade, "workspace_h5ad_write_guard") == []
    assert "MAX_WORKSPACE_H5AD_FILES" not in source
    assert "keep at most 1-2 checkpoint files" not in source
