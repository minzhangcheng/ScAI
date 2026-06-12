"""Tests for species-input interaction endpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from backend.models.api_requests import SpeciesInputRequest


class TestSpeciesInputRequest:
    def test_valid_request(self):
        req = SpeciesInputRequest(input="Homo sapiens")
        assert req.input == "Homo sapiens"

    def test_empty_rejected(self):
        with pytest.raises(Exception):
            SpeciesInputRequest(input="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(Exception):
            SpeciesInputRequest(input="   ")

    def test_strips_whitespace(self):
        req = SpeciesInputRequest(input="  mouse  ")
        assert req.input == "mouse"
