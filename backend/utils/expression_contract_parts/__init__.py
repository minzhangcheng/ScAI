"""Internal owners behind the expression_contract compatibility facade."""

from .ast import (  # noqa: F401
    _call_module_root,
    _mutation_target_root,
    _root_name,
    _value_makes_fresh_object,
    extract_sensitive_expression_inputs,
    has_direct_expression_contract_call,
    instrument_expression_contract_calls,
    mutates_expression_representation,
    prepares_expression_before_sensitive_call,
)
from .constants import (  # noqa: F401
    _FRESH_OBJECT_CALLS,
    _MARKER,
    _NON_EXPRESSION_CALL_ROOTS,
    _REPRESENTATION_MUTATORS,
    _RUNTIME_CALL_HELPER,
    _SENSITIVE_CALLS,
)
from .kernel import (  # noqa: F401
    build_expression_preflight_code,
    build_expression_runtime_code,
)
from .provenance import (  # noqa: F401
    ROUND_START_KEY,
    _MAX_NAMED_STALE_ARTIFACTS,
    describe_invalidated_artifacts,
    parse_expression_contracts,
    record_executed_representation_change,
    record_expression_contract,
    record_invalidated_file_stems,
    record_representation_change,
    record_round_start,
    round_start_ns,
    strip_expression_contract_markers,
    unchanged_artifact_stems,
    workspace_artifact_versions,
)
