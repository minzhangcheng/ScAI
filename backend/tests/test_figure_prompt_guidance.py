"""Task 3: the bioinformatics prompt must steer final/report figures through
generate_publication_figure and warn against nested results/ figure save paths.
Verified by inspecting the prompt-building source (agent construction needs LLM
config, so we avoid instantiation)."""
import inspect

from backend.agents.bioinformatics_agent import BioinformaticsAgent

_GUIDANCE_MARKER = "generate_publication_figure for figures intended for the final report"
_SUBDIR_WARNING_MARKER = "do NOT save figures into nested results/"


def _all_prompt_source() -> str:
    prop_obj = getattr(BioinformaticsAgent, "_static_system_prompt")
    fget = prop_obj.fget
    parts = [inspect.getsource(fget)]
    return "\n".join(parts)


def test_prompt_recommends_publication_figure_for_final_figures():
    assert _GUIDANCE_MARKER in _all_prompt_source()


def test_prompt_warns_against_nested_results_figure_paths():
    assert _SUBDIR_WARNING_MARKER in _all_prompt_source()
