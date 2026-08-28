#!/usr/bin/env python3
"""Immutable source-tree and test-outcome proofs for the ScAI runner."""

if __package__:
    from .scai_test_tree_parts.discovery import (
        ArtifactBytes, ContractError, _fail, _sha256_bytes, _sha256_file,
        read_direct_children_nofollow,
    )
    from .scai_test_tree_parts.evidence import (
        _hash_json, _normalize_detail, _phase_result, _phase_signature, _record_signature,
        _validate_phases,
        parse_junit, parse_junit_bytes, project_pytest_9_1_1_junit, publish_evidence,
        reject_kernel_gated_live_skips, validate_collection, validate_outcomes,
    )
    from .scai_test_tree_parts.identity import (
        _manifest, _manifest_digest, _validate_manifest_payload, build_commit_manifest,
        build_index_manifest, build_worktree_manifest, compare_index, compare_post_summary,
        load_manifest, verify_final_commit,
    )
    from .scai_test_tree_parts.profiles import cli
else:
    from scai_test_tree_parts.discovery import (
        ArtifactBytes, ContractError, _fail, _sha256_bytes, _sha256_file,
        read_direct_children_nofollow,
    )
    from scai_test_tree_parts.evidence import (
        _hash_json, _normalize_detail, _phase_result, _phase_signature, _record_signature,
        _validate_phases,
        parse_junit, parse_junit_bytes, project_pytest_9_1_1_junit, publish_evidence,
        reject_kernel_gated_live_skips, validate_collection, validate_outcomes,
    )
    from scai_test_tree_parts.identity import (
        _manifest, _manifest_digest, _validate_manifest_payload, build_commit_manifest,
        build_index_manifest, build_worktree_manifest, compare_index, compare_post_summary,
        load_manifest, verify_final_commit,
    )
    from scai_test_tree_parts.profiles import cli


if __name__ == "__main__":
    raise SystemExit(cli())
