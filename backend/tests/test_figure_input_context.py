"""Figure revisions receive measured object metadata without altering analysis."""
import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from backend.agents.bioinformatics.figure_context import figure_input_context


@pytest.mark.asyncio
async def test_figure_inventory_measures_existing_columns_without_data_or_mutation():
    table = pd.DataFrame({"condition": ["secret-group"], "value": [12345]})
    namespace = {"bulk_table": table}

    async def execute(code, **kwargs):
        assert kwargs == {
            "records_history": False, "skip_safety_check": True,
            "mutates_namespace": False,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        return SimpleNamespace(success=True, output=output.getvalue())

    sandbox = SimpleNamespace(kc=object(), execute=AsyncMock(side_effect=execute))
    result = await figure_input_context(sandbox, "bulk_table['genotype']; bulk_raw_matrix")
    inventory = json.loads(result)
    assert inventory["objects"] == [{
        "name": "bulk_table", "type": "DataFrame", "shape": [1, 2],
        "columns": ["condition", "value"], "column_count": 2,
    }]
    assert inventory["missing_referenced_names"] == ["bulk_raw_matrix"]
    assert "secret-group" not in result
    assert "12345" not in result
    assert namespace["bulk_table"] is table
    assert "_scai_figure_input_inventory" not in namespace


@pytest.mark.asyncio
async def test_failed_inventory_is_explicit_and_nonblocking():
    sandbox = SimpleNamespace(kc=object(), execute=AsyncMock(
        return_value=SimpleNamespace(success=False, output="", error="unavailable"),
    ))
    assert "unavailable" in await figure_input_context(sandbox, "plot()")
