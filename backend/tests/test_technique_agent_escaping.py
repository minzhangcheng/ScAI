import ast

from backend.agents.technique_agent import TechniqueAgent


def _syntax_is_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def test_inspect_code_prevents_injection():
    code = TechniqueAgent._build_inspect_code('os"; import os; os.system("x")')
    assert _syntax_is_valid(code), "Injection must not break syntax"


def test_dir_code_prevents_injection():
    code = TechniqueAgent._build_dir_code('os"; import os; os.system("x")')
    assert _syntax_is_valid(code), "Injection must not break syntax"


def test_inspect_code_works_normally():
    code = TechniqueAgent._build_inspect_code("numpy.ndarray")
    assert _syntax_is_valid(code)


def test_dir_code_works_normally():
    code = TechniqueAgent._build_dir_code("numpy")
    assert _syntax_is_valid(code)
