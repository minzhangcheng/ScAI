import ast
from backend.routes.chat import _build_figure_setup_code


def test_malicious_fig_dir_does_not_break_out():
    evil = '/tmp/x"; import os; os.system("touch /tmp/pwned")\n#'
    code = _build_figure_setup_code(evil)
    tree = ast.parse(code)
    assert not any(isinstance(n, ast.Call) and getattr(getattr(n, "func", None), "attr", "") == "system"
                   for n in ast.walk(tree))
