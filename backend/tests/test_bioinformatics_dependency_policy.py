import re

from backend.tests.split_facade import split_source


def test_bioinformatics_prompt_does_not_instruct_direct_pip_install():
    # Scanning the facade alone would leave the moved code free to shell out to pip.
    text = split_source("backend.agents.bioinformatics_agent")
    assert re.search(r"subprocess\.run\(\s*\[\s*['\"]pip['\"]\s*,\s*['\"]install['\"]", text) is None
    assert "report the missing dependency" in text.lower()
