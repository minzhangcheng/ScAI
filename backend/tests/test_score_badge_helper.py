from backend.services.report import variants


def test_score_badge_known_levels():
    assert 'badge-high">High' in variants._render_score_badge("high confidence")
    assert 'badge-medium">Medium' in variants._render_score_badge("medium")
    assert 'badge-low">Low' in variants._render_score_badge("low")


def test_score_badge_fallback_class_is_parameterized():
    assert 'badge-medium"' in variants._render_score_badge("weird", fallback_class="badge-medium")
    assert 'badge-na"' in variants._render_score_badge("weird", fallback_class="badge-na")


def test_score_badge_escapes_and_truncates():
    out = variants._render_score_badge("<x>" * 20, fallback_class="badge-na")
    assert "<x>" not in out
    assert "&lt;x&gt;" in out
