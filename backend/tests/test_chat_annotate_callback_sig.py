import inspect
import ast
from pathlib import Path


def test_chat_annotate_callback_accepts_new_kwargs():
    src = Path("routes/chat.py").read_text() if Path("routes/chat.py").exists() \
        else Path("backend/routes/chat.py").read_text()
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.AsyncFunctionDef) and n.name == "annotate_callback"]
    assert fns, "annotate_callback not found"
    args = {a.arg for a in fns[0].args.args} | {a.arg for a in fns[0].args.kwonlyargs}
    assert "analysis_log" in args and "database_priors" in args
