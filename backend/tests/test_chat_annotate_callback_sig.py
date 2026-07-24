from backend.tests.split_facade import split_source
import inspect
import ast
from pathlib import Path


def test_chat_annotate_callback_accepts_new_kwargs():
    src = split_source("backend.routes.chat")
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.AsyncFunctionDef) and n.name == "annotate_callback"]
    assert fns, "annotate_callback not found"
    args = {a.arg for a in fns[0].args.args} | {a.arg for a in fns[0].args.kwonlyargs}
    assert "analysis_log" in args and "database_priors" in args
