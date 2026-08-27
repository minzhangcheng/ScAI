"""Compatibility facade for post-execution plan audit owners."""
from .post_execution_review_parts import audit as _audit
from .post_execution_review_parts import methods as _methods

globals().update(
    {name: value for module in (_methods, _audit) for name, value in vars(module).items()
     if not name.startswith("__")}
)
