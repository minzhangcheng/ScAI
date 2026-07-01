from pathlib import Path
import re


def test_bioinformatics_prompt_does_not_instruct_direct_pip_install():
    text = Path("backend/agents/bioinformatics_agent.py").read_text(encoding="utf-8")
    assert re.search(r"subprocess\.run\(\s*\[\s*['\"]pip['\"]\s*,\s*['\"]install['\"]", text) is None
    assert "report the missing dependency" in text.lower()
