from backend.config._config import SystemConfig, default_config


def test_validation_flag_default_true():
    assert SystemConfig().enable_annotation_validation_round is True
    assert default_config.enable_annotation_validation_round is True


def test_validation_flag_overridable():
    assert SystemConfig(enable_annotation_validation_round=False).enable_annotation_validation_round is False


def test_validation_timeout_and_iter_defaults():
    cfg = SystemConfig()
    assert cfg.annotation_validation_timeout_seconds == 600
    assert cfg.annotation_validation_max_iterations == 5


def test_validation_timeout_and_iter_overridable():
    cfg = SystemConfig(annotation_validation_timeout_seconds=90, annotation_validation_max_iterations=3)
    assert cfg.annotation_validation_timeout_seconds == 90
    assert cfg.annotation_validation_max_iterations == 3


import pytest
from pydantic import ValidationError


def test_timeout_rejects_non_positive():
    with pytest.raises(ValidationError):
        SystemConfig(annotation_validation_timeout_seconds=0)


def test_max_iterations_bounds():
    with pytest.raises(ValidationError):
        SystemConfig(annotation_validation_max_iterations=0)
    with pytest.raises(ValidationError):
        SystemConfig(annotation_validation_max_iterations=9)


def test_validation_limits_clamps_bad_values():
    from types import SimpleNamespace
    from backend.workflow.annotation import AnnotationMixin
    host = AnnotationMixin()
    host.config = SimpleNamespace(annotation_validation_max_iterations=99,
                                  annotation_validation_timeout_seconds=0)
    assert host._validation_limits() == (8, 600)
