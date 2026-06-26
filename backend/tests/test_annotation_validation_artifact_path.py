from backend.workflow.annotation_validation import resolve_artifact_path

WS = "/data/sessions/sid123/workspace"


def test_keeps_absolute_path_inside_workspace():
    p = "/data/sessions/sid123/workspace/annotation_validation/m.jpg"
    assert resolve_artifact_path(p, WS) == p


def test_reroutes_stray_absolute_path_by_basename():
    p = "/workspace/cmz/ScAI/scai/data/sessions/sid123/workspace/annotation_validation/m.jpg"
    assert resolve_artifact_path(p, WS) == "/data/sessions/sid123/workspace/annotation_validation/m.jpg"


def test_relative_with_workspace_prefix():
    assert resolve_artifact_path("workspace/annotation_validation/m.jpg", WS) == \
           "/data/sessions/sid123/workspace/annotation_validation/m.jpg"


def test_relative_subdir():
    assert resolve_artifact_path("annotation_validation/m.jpg", WS) == \
           "/data/sessions/sid123/workspace/annotation_validation/m.jpg"


def test_bare_filename():
    assert resolve_artifact_path("m.jpg", WS) == "/data/sessions/sid123/workspace/m.jpg"


def test_empty_path_is_safe():
    out = resolve_artifact_path("", WS)
    assert out.startswith(WS)
