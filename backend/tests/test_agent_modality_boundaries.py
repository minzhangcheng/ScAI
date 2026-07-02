from backend.workflow.synthesis import build_disabled_pathology_warning


def test_disabled_pathology_with_pathology_image_warns():
    files = [{"section": "pathology", "data_type": "H&E", "filename": "slide.png"}]
    warning = build_disabled_pathology_warning(files, pathology_enabled=False)
    assert warning
    assert "pathology" in warning.lower()


def test_disabled_pathology_with_biochemistry_image_does_not_warn():
    # A biochemistry-section image is not pathology data and must not trigger
    # the pathology modality-boundary warning.
    files = [{"section": "biochemistry", "data_type": "image", "filename": "panel.png"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=False) == ""


def test_disabled_pathology_with_pending_pathology_upload_warns():
    # Section-based pending uploads carry the section under the "type" key.
    files = [{"type": "pathology", "filename": "biopsy.png"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=False)


def test_disabled_pathology_with_pending_chat_image_does_not_warn():
    # Generic PI-chat uploads (type is 'image'/'file') are not pathology-designated.
    files = [{"type": "image", "filename": "sketch.png"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=False) == ""


def test_enabled_pathology_suppresses_boundary_warning():
    files = [{"section": "pathology", "data_type": "H&E", "filename": "slide.png"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=True) == ""


def test_non_pathology_inputs_do_not_warn():
    files = [{"section": "biochemistry", "data_type": "table", "filename": "values.csv"}]
    assert build_disabled_pathology_warning(files, pathology_enabled=False) == ""
