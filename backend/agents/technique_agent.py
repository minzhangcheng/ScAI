"""
Technique Agent - Lightweight API documentation lookup and summarization.

Uses a cheap LLM to extract docstrings from installed packages via the sandbox
and summarize them into concise API references. Can also fetch web documentation
as a fallback.
"""
import asyncio
import re
import logging
from typing import Optional, Dict, Any, AsyncGenerator

from .base import BaseAgent, AgentConfig
from ..models.schemas import AgentType, AnalysisState
from ..utils.error_parsing import API_ERROR_TYPES, extract_error_type
from ..utils.llm_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """You are a technical documentation assistant. Given raw Python docstring output, produce a concise API reference.

FORMAT (strict, no extra text):
**Signature:** `function_path(param1, param2, ...)`
**Key Parameters:**
- `param1` (type): description
- `param2` (type): description
(only list the 5-8 most important parameters)
**Returns:** description
**Example:**
```python
one short usage example
```

If the input includes a list of available attributes (from dir()), highlight the 3-5 most relevant ones to the user's error.

Keep total output under 25 lines."""


class TechniqueAgent(BaseAgent):
    """
    Lightweight technical support agent for API documentation lookup.

    Not an autonomous agent - only called by Supervisor (on repeated errors)
    or by BioinformaticsAgent (via lookup_documentation tool).
    """

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        temperature: float = 0.1,
        system_prompt: str = "",
    ):
        super().__init__(
            agent_type=AgentType.TECHNIQUE,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt or _SUMMARIZE_PROMPT,
            config=AgentConfig(temperature=temperature),
        )

    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Abstract method implementation. TechniqueAgent is not called via process() —
        it uses lookup_docs() and lookup_from_error() directly.
        This stub satisfies the BaseAgent ABC contract.
        """
        return {
            "agent": self.agent_type.value,
            "status": "skipped",
            "message": "TechniqueAgent does not use the standard process() workflow. "
                       "Use lookup_docs() or lookup_from_error() instead.",
        }

    @staticmethod
    def _build_inspect_code(function_path: str) -> str:
        """Build Python code to extract docstring and signature from an installed package."""
        parts = function_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attr_name = parts
            return f"""
import inspect
try:
    import {module_path.split('.')[0]}
    mod = __import__("{module_path}", fromlist=["{attr_name}"])
    obj = getattr(mod, "{attr_name}", None)
    if obj is None:
        print(f"'{attr_name}' not found in {module_path}.")
        print(f"Available attributes: {{[a for a in dir(mod) if not a.startswith('_')]}}")
    else:
        sig = ""
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            pass
        doc = inspect.getdoc(obj) or "No docstring available."
        print(f"Signature: {function_path}{{sig}}")
        print(f"\\nDocstring:\\n{{doc[:3000]}}")
except ImportError as e:
    print(f"Cannot import: {{e}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
        else:
            # Top-level module
            return f"""
import inspect
try:
    import {function_path}
    doc = inspect.getdoc({function_path}) or "No docstring available."
    print(f"Module: {function_path}")
    print(f"\\nDocstring:\\n{{doc[:2000]}}")
    print(f"\\nPublic API: {{[a for a in dir({function_path}) if not a.startswith('_')][:50]}}")
except ImportError as e:
    print(f"Cannot import: {{e}}")
"""

    @staticmethod
    def _build_dir_code(module_path: str) -> str:
        """Build code to list available attributes of a module."""
        return f"""
try:
    import {module_path.split('.')[0]}
    mod = __import__("{module_path}", fromlist=["__name__"])
    attrs = [a for a in dir(mod) if not a.startswith('_')]
    print(f"Available in {module_path}: {{attrs}}")
except ImportError as e:
    print(f"Cannot import: {{e}}")
"""

    @staticmethod
    def _parse_error_for_lookup(error_text: str) -> Optional[Dict[str, str]]:
        """
        Parse an error traceback to extract information for documentation lookup.

        Returns dict with keys: error_type, module, attribute, function
        Returns None if the error is not an API-related error worth looking up.
        """
        if not error_text:
            return None

        error_type = extract_error_type(error_text)
        if error_type not in API_ERROR_TYPES:
            return None

        result = {"error_type": error_type}

        # AttributeError: module 'X' has no attribute 'Y'
        m = re.search(
            r"AttributeError: module ['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)['\"]",
            error_text,
        )
        if m:
            result["module"] = m.group(1)
            result["attribute"] = m.group(2)
            return result

        # TypeError: func() got an unexpected keyword argument 'X'
        m = re.search(
            r"TypeError: (\w+)\(\) got an unexpected keyword argument",
            error_text,
        )
        if m:
            result["function"] = m.group(1)
            return result

        # ImportError: cannot import name 'X' from 'Y'
        m = re.search(
            r"ImportError: cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]",
            error_text,
        )
        if m:
            result["attribute"] = m.group(1)
            result["module"] = m.group(2)
            return result

        # ModuleNotFoundError: No module named 'X'
        m = re.search(
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
            error_text,
        )
        if m:
            result["module"] = m.group(1)
            return result

        # Generic: at least return what we have
        return result

    async def lookup_docs(
        self,
        sandbox,
        function_path: str,
        question: str = "",
    ) -> str:
        """
        Look up documentation for a function and return a concise summary.

        Args:
            sandbox: LocalJupyterSandbox instance
            function_path: e.g. 'scanpy.tl.umap', 'decoupler.op.collectri'
            question: optional specific question about the API

        Returns:
            Concise API summary (under 25 lines)
        """
        # Step 1: Extract raw docstring from sandbox
        code = self._build_inspect_code(function_path)
        result = await sandbox.execute(code)
        raw_doc = result.output if result.output else ""
        if result.error:
            raw_doc += f"\nError during lookup: {result.error}"

        if not raw_doc.strip() or "Cannot import" in raw_doc:
            return f"Could not find documentation for `{function_path}`. Package may not be installed."

        # Step 2: Summarize with cheap LLM
        user_msg = f"Summarize this Python API documentation:\n\n{raw_doc[:4000]}"
        if question:
            user_msg += f"\n\nSpecific question: {question}"

        try:
            response = await self.llm_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.config.temperature,
                max_tokens=1024,
            )
            return response.content if response.content else raw_doc[:2000]
        except Exception as e:
            logger.warning(f"Technique Agent LLM summarization failed: {e}")
            return raw_doc[:2000]

    async def lookup_from_error(
        self,
        sandbox,
        error_text: str,
        failed_code: str = "",
    ) -> str:
        """
        Automatically look up docs based on an error message.

        Called by Supervisor when it detects consecutive same-type API errors.

        Returns:
            Concise error explanation + correct API usage
        """
        parsed = self._parse_error_for_lookup(error_text)
        if not parsed:
            return ""

        doc_parts = []

        # Build parallel tasks for independent lookups
        tasks = {}
        module = parsed.get("module", "")
        attr = parsed.get("attribute", "")

        if module and attr:
            dir_code = self._build_dir_code(module)
            tasks["dir"] = sandbox.execute(dir_code)
            tasks["doc"] = self.lookup_docs(sandbox, f"{module}.{attr}")
            # Pre-fetch module-level docs as fallback (avoids sequential await later)
            tasks["doc_fallback"] = self.lookup_docs(sandbox, module)
        elif module:
            tasks["doc"] = self.lookup_docs(sandbox, module)

        if parsed.get("function"):
            func_path = self._guess_function_path(parsed["function"], failed_code)
            if func_path:
                tasks["func_doc"] = self.lookup_docs(sandbox, func_path)

        if tasks:
            results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

            if "dir" in results and results["dir"].output:
                doc_parts.append(f"Available in `{module}`:\n{results['dir'].output}")
            if "doc" in results:
                doc = results["doc"]
                if "Could not find" in doc and "doc_fallback" in results:
                    doc = results["doc_fallback"]
                doc_parts.append(doc)
            if "func_doc" in results:
                doc_parts.append(results["func_doc"])

        if not doc_parts:
            return ""

        combined = "\n\n".join(doc_parts)

        user_msg = (
            f"The user's code failed with this error:\n```\n{error_text[:500]}\n```\n\n"
            f"Here is the relevant API documentation:\n{combined[:3000]}\n\n"
            f"Provide:\n"
            f"1. Brief explanation of why the error occurred (1 line)\n"
            f"2. The correct API usage (code example)\n"
            f"Keep it under 15 lines total."
        )

        try:
            response = await self.llm_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.config.temperature,
                max_tokens=1024,
            )
            return response.content if response.content else combined[:1500]
        except Exception as e:
            logger.warning(f"Technique Agent error summarization failed: {e}")
            return combined[:1500]

    @staticmethod
    def _guess_function_path(func_name: str, code: str) -> str:
        """Try to guess the full dotted path of a function from the code's imports."""
        # Match: from X.Y import func_name
        m = re.search(rf"from\s+([\w.]+)\s+import\s+.*\b{re.escape(func_name)}\b", code)
        if m:
            return f"{m.group(1)}.{func_name}"
        # Match: import X; X.func_name(...)
        m = re.search(rf"import\s+([\w.]+)", code)
        if m:
            if re.search(rf"{re.escape(m.group(1))}\.{re.escape(func_name)}\s*\(", code):
                return f"{m.group(1)}.{func_name}"
        # Match common aliases: sc.tl.func, dc.mt.func
        aliases = {"sc": "scanpy", "np": "numpy", "pd": "pandas", "dc": "decoupler"}
        for alias, full in aliases.items():
            m = re.search(rf"{re.escape(alias)}\.([\w.]*\.)?{re.escape(func_name)}\s*\(", code)
            if m:
                mid = m.group(1) or ""
                return f"{full}.{mid}{func_name}"
        return ""

