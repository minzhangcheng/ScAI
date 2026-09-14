"""Real-kernel contract for the AnnData checkpoint capture bridge.

Requires ``SCAI_RUN_KERNEL_TESTS=1`` (the supported kernel profile).  It proves
the non-kernel parts cannot: the pinned writer path, top-level anchor alias
identity, and host-side immutable blob publication.
"""
from __future__ import annotations

import pytest

from backend.services.adata_checkpoints.layout import verify_blob
from backend.services.adata_checkpoints.storage import publish_or_reuse


@pytest.mark.asyncio
async def test_real_h5ad_capture_bridge_and_anchor_identity(sandbox_with_adata):
    sandbox = sandbox_with_adata
    digest_before = await sandbox.execute(
        "import hashlib as _h; "
        "print(_h.sha256(adata.X.tobytes()).hexdigest())",
        records_history=False,
        skip_safety_check=True,
        mutates_namespace=False,
    )
    assert digest_before.success

    candidate = sandbox._checkpoint_candidate_path()
    captured = await sandbox.serialize_adata_candidate_locked(candidate)
    assert captured.get("ok") is True, captured
    assert candidate.is_file()
    assert int(captured.get("n_obs") or 0) == 50
    assert int(captured.get("n_vars") or 0) == 10

    # Source object must be unchanged by a successful capture.
    digest_after = await sandbox.execute(
        "import hashlib as _h; "
        "print(_h.sha256(adata.X.tobytes()).hexdigest())",
        records_history=False,
        skip_safety_check=True,
        mutates_namespace=False,
    )
    assert digest_after.success
    assert (
        digest_before.output.strip().splitlines()[-1]
        == digest_after.output.strip().splitlines()[-1]
    )

    published = publish_or_reuse(
        sandbox.session_id,
        candidate,
        allowed_root=sandbox._session_tmp,
    )
    verify_blob(
        published.identity.absolute_path,
        expected_sha256=published.identity.sha256,
        expected_size=published.identity.size,
    )

    registry_name = "scai_adata_registry_integration"
    anchor_name = "scai_adata_anchor_integration"
    registered = await sandbox.register_canonical_anchor_locked(
        registry_name=registry_name,
        anchor_name=anchor_name,
        adata_lineage_id="integration-lineage",
        ancestor_checkpoint_id=None,
    )
    assert registered.get("ok") is True, registered

    queried = await sandbox.query_adata_registry_locked(registry_name=registry_name)
    assert queried["canonical"]["same_as_anchor"] is True
    assert queried["canonical"]["adata_lineage_id"] == "integration-lineage"
    assert sandbox.host_managed_global_names() == {registry_name, anchor_name}

    # Repeated registration must not trip the R0.1 method/attribute shadowing.
    repeated = await sandbox.register_canonical_anchor_locked(
        registry_name=registry_name,
        anchor_name=anchor_name,
        adata_lineage_id="integration-lineage-2",
        ancestor_checkpoint_id=None,
    )
    assert repeated.get("ok") is True, repeated

    copied = await sandbox.apply_adata_source_locked(
        mode="checkpoint_copy",
        registry_name=registry_name,
        preparation_id="prep-integration-copy",
        application_token="token-integration-copy",
        scope_id="scope-integration-copy",
        owner_kind="plan",
        owner_id="plan-integration-copy",
        selection_signature="a" * 64,
        session_lineage_id="main",
        checkpoint_path=str(published.identity.absolute_path),
        expected_n_obs=50,
        expected_n_vars=10,
        adata_lineage_id="copy-lineage",
        ancestor_checkpoint_id=None,
        binding_id="binding-integration-copy",
        binding_name="scai_work_integration_copy",
        anchor_name="scai_anchor_integration_copy",
    )
    assert copied.get("ok") is True, copied
    before_collision = await sandbox.execute(
        "_canonical_before_collision = id(adata)\n"
        "scai_promote_collision = object()",
        records_history=False,
        skip_safety_check=True,
    )
    assert before_collision.success
    refused_promotion = await sandbox.apply_adata_source_locked(
        mode="promote_live",
        registry_name=registry_name,
        preparation_id="prep-integration-promote",
        application_token="token-integration-promote",
        scope_id="scope-integration-promote",
        owner_kind="plan",
        owner_id="plan-integration-promote",
        selection_signature="b" * 64,
        session_lineage_id="main",
        adata_lineage_id="copy-lineage",
        ancestor_checkpoint_id=None,
        binding_id="binding-integration-copy",
        binding_name="scai_work_integration_copy",
        anchor_name="scai_promote_collision",
    )
    assert refused_promotion.get("ok") is False
    assert refused_promotion.get("reason") == "anchor_collision"
    unchanged = await sandbox.execute(
        "print(id(adata) == _canonical_before_collision)",
        records_history=False,
        skip_safety_check=True,
        mutates_namespace=False,
    )
    assert unchanged.success
    assert unchanged.output.strip().splitlines()[-1] == "True"

    # A user-created adata remains detectable as a declared persistent output
    # after Host anchors were installed.
    declared = await sandbox.execute(
        "adata.obs['post_anchor'] = 'kept'",
        declared_outputs=["adata"],
        skip_safety_check=True,
    )
    assert declared.success, declared.error
