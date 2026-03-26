import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.agents.technique_agent import TechniqueAgent
from backend.models.schemas import AnalysisState


class _ConcreteTechniqueAgent(TechniqueAgent):
    """Concrete subclass for testing (BaseAgent.process is abstract)."""
    async def process(self, state: AnalysisState, instruction: str, **kwargs):
        return {"result": "test"}


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def agent(mock_llm):
    return _ConcreteTechniqueAgent(llm_client=mock_llm, model="test-model")


def test_agent_type_technique_exists():
    from backend.models.schemas import AgentType
    assert hasattr(AgentType, "TECHNIQUE")
    assert AgentType.TECHNIQUE.value == "technique"


def test_technique_model_config():
    from backend.config import SystemConfig
    config = SystemConfig()
    assert hasattr(config.models, "technique_model")
    assert config.models.technique_model == "google/gemini-3.1-flash-lite-preview"


def test_technique_temperature_config():
    from backend.config import SystemConfig
    config = SystemConfig()
    assert hasattr(config.temperatures, "technique")
    assert config.temperatures.technique == 0.1
    assert config.temperatures.get("technique") == 0.1


def test_build_inspect_code_basic():
    """Test that _build_inspect_code generates valid Python code."""
    code = TechniqueAgent._build_inspect_code("scanpy.tl.umap")
    assert "inspect" in code
    assert "scanpy.tl" in code
    assert "umap" in code
    assert "getdoc" in code or "signature" in code


def test_build_inspect_code_top_level_module():
    code = TechniqueAgent._build_inspect_code("scanpy")
    assert "scanpy" in code


def test_parse_error_for_lookup_attribute_error():
    error = """Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'decoupler' has no attribute 'get_collectri'"""
    result = TechniqueAgent._parse_error_for_lookup(error)
    assert result is not None
    assert result["error_type"] == "AttributeError"
    assert result["module"] == "decoupler"
    assert result["attribute"] == "get_collectri"


def test_parse_error_for_lookup_type_error():
    error = """Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: umap() got an unexpected keyword argument 'init_pos'"""
    result = TechniqueAgent._parse_error_for_lookup(error)
    assert result is not None
    assert result["error_type"] == "TypeError"
    assert result["function"] == "umap"


def test_parse_error_for_lookup_import_error():
    error = """Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: cannot import name 'run_ulm' from 'decoupler'"""
    result = TechniqueAgent._parse_error_for_lookup(error)
    assert result is not None
    assert result["error_type"] == "ImportError"
    assert result["module"] == "decoupler"
    assert result["attribute"] == "run_ulm"


def test_parse_error_for_lookup_non_api_error():
    """Non-API errors (KeyError, MemoryError) should return None."""
    error = "MemoryError: Unable to allocate 4.00 GiB"
    result = TechniqueAgent._parse_error_for_lookup(error)
    assert result is None


def test_guess_function_path_from_import():
    code = "from scanpy.tl import umap\numap(adata)"
    result = TechniqueAgent._guess_function_path("umap", code)
    assert result == "scanpy.tl.umap"


def test_guess_function_path_from_alias():
    code = "import scanpy as sc\nsc.tl.umap(adata)"
    result = TechniqueAgent._guess_function_path("umap", code)
    assert result == "scanpy.tl.umap"


@pytest.mark.asyncio
async def test_lookup_docs_with_mock_sandbox(agent, mock_llm):
    """Test full lookup_docs flow with mocked sandbox."""
    mock_llm.chat = AsyncMock(return_value=MagicMock(
        content="**Signature:** `scanpy.tl.umap(adata, ...)`\n**Key Parameters:**\n- `init_pos`: initialization method"
    ))

    mock_sandbox = MagicMock()
    mock_sandbox.execute = AsyncMock(return_value=MagicMock(
        output="Signature: scanpy.tl.umap(adata, min_dist=0.5, ...)\n\nDocstring:\nEmbed the neighborhood graph...",
        error=None,
    ))

    result = await agent.lookup_docs(mock_sandbox, "scanpy.tl.umap")
    assert "umap" in result.lower()
    mock_sandbox.execute.assert_called_once()
    mock_llm.chat.assert_called_once()


@pytest.mark.asyncio
async def test_lookup_from_error_attribute_error(agent, mock_llm):
    """Test lookup_from_error for AttributeError."""
    mock_llm.chat = AsyncMock(return_value=MagicMock(
        content="The `get_collectri` function was removed in decoupler v2. Use `dc.op.collectri()` instead."
    ))

    mock_sandbox = MagicMock()
    mock_sandbox.execute = AsyncMock(return_value=MagicMock(
        output="Available in decoupler: ['op', 'mt', 'pp', 'pl']",
        error=None,
    ))

    error = "AttributeError: module 'decoupler' has no attribute 'get_collectri'"
    result = await agent.lookup_from_error(mock_sandbox, error, "import decoupler as dc\ndc.get_collectri()")
    assert result
    mock_sandbox.execute.assert_called()


@pytest.mark.asyncio
async def test_lookup_from_error_non_api_error_returns_empty(agent):
    """Non-API errors should return empty string."""
    mock_sandbox = MagicMock()

    result = await agent.lookup_from_error(mock_sandbox, "MemoryError: Unable to allocate", "")
    assert result == ""
    mock_sandbox.execute.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_docs_fallback_on_llm_failure(agent, mock_llm):
    """When LLM fails, should fallback to truncated raw docstring."""
    mock_llm.chat = AsyncMock(side_effect=Exception("LLM timeout"))

    mock_sandbox = MagicMock()
    raw_doc = "Signature: scanpy.tl.umap(adata)\n\nDocstring:\nEmbed the neighborhood graph using UMAP."
    mock_sandbox.execute = AsyncMock(return_value=MagicMock(
        output=raw_doc,
        error=None,
    ))

    result = await agent.lookup_docs(mock_sandbox, "scanpy.tl.umap")
    assert "umap" in result.lower()
    assert "neighborhood" in result.lower()
