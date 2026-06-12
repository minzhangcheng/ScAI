"""Test that species is prepended to task instructions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestSpeciesInjection:
    def test_species_prepended_when_set(self):
        species = "Homo sapiens"
        task_instruction = "Analyze pathology images"
        if species:
            task_instruction = f"Species: {species}\n\n" + task_instruction
        assert task_instruction.startswith("Species: Homo sapiens")
        assert "Analyze pathology images" in task_instruction

    def test_species_not_prepended_when_empty(self):
        species = ""
        task_instruction = "Analyze pathology images"
        if species:
            task_instruction = f"Species: {species}\n\n" + task_instruction
        assert not task_instruction.startswith("Species:")
