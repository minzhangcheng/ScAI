from backend.workflow.synthesis import build_disabled_pathology_warning


def test_disabled_pathology_with_image_inputs_returns_generic_warning():
    files = [{"section": "biochemistry", "data_type": "image", "filename": "panel.png"}]
    warning = build_disabled_pathology_warning(files, pathology_enabled=False)
    assert warning
    assert "pathology" in warning.lower()
    assert "image" in warning.lower()


def test_enabled_pathology_suppresses_boundary_warning():
    files = [{"section": "biochemistry", "data_type": "image", "filename": "panel.png"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=True) == ""


def test_non_image_inputs_do_not_warn():
    files = [{"section": "biochemistry", "data_type": "table", "filename": "values.csv"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=False) == ""
