"""Bounded read-only kernel facts for publication-figure code revisions."""
from __future__ import annotations

import ast
import json
import logging

logger = logging.getLogger(__name__)
_MARKER = "__SCAI_FIGURE_INPUTS__:"


async def figure_input_context(sandbox, original_code: str, *, preferred_names=()) -> str:
    """Inspect names/shapes/columns, never expression values or analysis outputs."""
    if sandbox is None or getattr(sandbox, "kc", None) is None:
        return "Live input inventory unavailable; do not assume new variables or columns exist."
    try:
        preferred = sorted({
            node.id for node in ast.walk(ast.parse(original_code))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        })[:128]
    except SyntaxError:
        preferred = []
    preferred = list(dict.fromkeys([
        name for name in preferred_names if isinstance(name, str) and name.isidentifier()
    ] + preferred))[:128]
    code = f'''
def _scai_figure_input_inventory():
    import json
    from itertools import islice
    namespace = globals()
    preferred = {preferred!r}
    names = sorted(name for name in namespace if not name.startswith('_'))
    ordered = [name for name in preferred if name in namespace]
    ordered += [name for name in names if name not in ordered]
    rows = []
    def describe(name, value):
        kind = type(value)
        module = kind.__module__
        if not module.startswith(('pandas.', 'anndata.', 'numpy', 'scipy.sparse')):
            return None
        row = {{'name': name[:120], 'type': kind.__name__}}
        if hasattr(value, 'shape'):
            row['shape'] = list(value.shape)[:4]
        if module.startswith('pandas.') and hasattr(value, 'columns'):
            row['columns'] = [str(item)[:100] for item in value.columns[:80]]
            row['column_count'] = len(value.columns)
        if module.startswith('anndata.'):
            row['obs_columns'] = [str(item)[:100] for item in value.obs.columns[:80]]
            row['obs_column_count'] = len(value.obs.columns)
        return row
    for name in ordered:
        value = namespace[name]
        row = describe(name, value)
        if row is not None:
            rows.append(row)
        elif type(value) is dict:
            # Result dictionaries are common analysis outputs. Inspect only
            # one bounded level of metadata, never values or arbitrary objects.
            for key, child in islice(value.items(), 20):
                if not isinstance(key, str) or len(key) > 120:
                    continue
                path = name + '[' + repr(key) + ']'
                child_row = describe(path, child)
                if child_row is not None:
                    child_row['name'] = path
                    rows.append(child_row)
                if len(rows) >= 60:
                    break
        if len(rows) >= 60:
            break
    return {{'objects': rows, 'limits': {{'objects': 60, 'columns_per_object': 80,
                                       'dictionary_entries': 20, 'dictionary_depth': 1}},
            'missing_referenced_names': [name for name in preferred if name not in namespace][:40]}}
try:
    print({_MARKER!r} + __import__('json').dumps(_scai_figure_input_inventory()))
finally:
    del _scai_figure_input_inventory
'''
    try:
        result = await sandbox.execute(
            code, records_history=False, skip_safety_check=True,
            mutates_namespace=False,
        )
        if result.success:
            for line in (result.output or "").splitlines():
                if line.startswith(_MARKER):
                    payload = json.loads(line[len(_MARKER):])
                    if isinstance(payload, dict):
                        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Figure input inventory unavailable: %s", type(exc).__name__)
    return "Live input inventory unavailable; do not assume new variables or columns exist."
