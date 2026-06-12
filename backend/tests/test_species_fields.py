"""Tests for species-related fields on AnalysisState and constants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from backend.models.schemas import AnalysisState, TaskStatus


class TestSpeciesFields:
    """Verify species fields exist with correct defaults."""

    def test_species_defaults_empty(self):
        state = AnalysisState(user_id="test")
        assert state.species == ""
        assert state.species_resolved is False
        assert state.species_input_info is None

    def test_species_genes_validated_default(self):
        state = AnalysisState(user_id="test")
        assert state.species_genes_validated is False

    def test_species_roundtrip(self):
        state = AnalysisState(user_id="test", species="Homo sapiens", species_resolved=True, species_genes_validated=True)
        data = state.model_dump(mode="json")
        restored = AnalysisState.model_validate(data)
        assert restored.species == "Homo sapiens"
        assert restored.species_resolved is True
        assert restored.species_genes_validated is True

    def test_species_input_info_roundtrip(self):
        info = {"source": "pi_inference", "suggestions": ["Homo sapiens"], "context": "test"}
        state = AnalysisState(user_id="test", species_input_info=info)
        data = state.model_dump(mode="json")
        restored = AnalysisState.model_validate(data)
        assert restored.species_input_info["source"] == "pi_inference"

    def test_waiting_species_input_status(self):
        assert TaskStatus.WAITING_SPECIES_INPUT == "waiting_species_input"
        state = AnalysisState(user_id="test", status=TaskStatus.WAITING_SPECIES_INPUT)
        assert state.status == TaskStatus.WAITING_SPECIES_INPUT


class TestSpeciesConstants:
    """Verify species-related constants."""

    def test_wait_input_reasons(self):
        from backend.constants import WaitInputReason
        assert WaitInputReason.SPECIES_INPUT == "species_input"
        assert WaitInputReason.GENE_SPECIES_MISMATCH == "gene_species_mismatch"

    def test_state_sync_change_types(self):
        from backend.constants import StateSyncChangeType
        assert StateSyncChangeType.SPECIES_INPUT == "species_input"
        assert StateSyncChangeType.GENE_SPECIES_MISMATCH == "gene_species_mismatch"
        assert StateSyncChangeType.SPECIES_RESOLVED == "species_resolved"
        assert StateSyncChangeType.SPECIES_AUTO_SKIPPED == "species_auto_skipped"


class TestSpeciesInputRequest:
    """Verify SpeciesInputRequest validation."""

    def test_valid_input(self):
        from backend.models.api_requests import SpeciesInputRequest
        req = SpeciesInputRequest(input="Homo sapiens")
        assert req.input == "Homo sapiens"

    def test_strips_whitespace(self):
        from backend.models.api_requests import SpeciesInputRequest
        req = SpeciesInputRequest(input="  Mus musculus  ")
        assert req.input == "Mus musculus"

    def test_rejects_empty(self):
        from backend.models.api_requests import SpeciesInputRequest
        with pytest.raises(Exception):
            SpeciesInputRequest(input="")


def test_to_api_dict_includes_species_fields():
    state = AnalysisState(
        user_id="u1",
        species="Homo sapiens",
        species_resolved=True,
        species_genes_validated=True,
        species_input_info={"source": "pi_inference", "suggestions": [], "context": "x"},
    )
    payload = state.to_api_dict()
    assert payload["species"] == "Homo sapiens"
    assert payload["species_resolved"] is True
    assert payload["species_genes_validated"] is True
    assert payload["species_input_info"]["source"] == "pi_inference"
