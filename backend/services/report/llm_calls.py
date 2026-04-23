"""LLM calls mixin for DetailedReportGenerator."""
from typing import List, Dict, Any, Optional

from backend.utils.report_html import _safe_parse_json


# Tools for Bioinformatics Agent to generate report figures/tables
REPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code to generate figures or tables. Use scanpy, matplotlib, seaborn, pandas. Always save figures to specified path and tables to CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must save output to the specified file paths."
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief title for the figure/table (e.g., 'UMAP visualization of cell clusters')"
                    },
                    "caption": {
                        "type": "string",
                        "description": "DETAILED figure caption for publication (50-100 words). Describe: what is shown, key observations, sample sizes, statistical details. Example: 'UMAP visualization of 15,234 cells from control and treatment groups, colored by Leiden clustering (resolution=0.8). A total of 12 distinct clusters were identified, with cluster 0 (3,421 cells, 22.4%) representing the largest population.'"
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["figure", "table", "both"],
                        "description": "Type of output this code generates"
                    }
                },
                "required": ["code", "description", "caption", "output_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_pi",
            "description": "Request guidance from PI Agent on what data/figures are needed, or ask for feedback on generated results",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or request for PI guidance"
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context (e.g., what has been generated so far, current data status)"
                    }
                },
                "required": ["question", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_for_review",
            "description": "Submit current section output to PI for review. PI will decide if more data/figures are needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what has been generated for this section"
                    },
                    "figures_generated": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of figure names/descriptions generated"
                    },
                    "tables_generated": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of table names generated"
                    }
                },
                "required": ["summary", "figures_generated"]
            }
        }
    }
]


class LLMCallsMixin:
    """Provides LLM call methods for report generation."""

    async def _pi_chat(self, prompt: str, temperature: float = 0.3, max_tokens: int = 32768) -> str:
        """Chat with PI Agent"""
        self._set_llm_context("report_pi")
        response = await self.llm_client.chat(
            model=self.pi_model,
            messages=[
                {"role": "system", "content": "You are a senior biomedical research PI writing a scientific report. You MUST respond exclusively in English."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content if response else ""

    async def _chat_with_vision_fallback(
        self,
        text_prompt: str,
        images: List[str],
        temperature: float = 0.3,
        max_tokens: int = 32768,
        request_timeout: float = 120.0,
    ) -> str:
        """Vision call with agent fallback chain: PI -> bio -> supervisor -> text-only."""
        import logging
        logger = logging.getLogger(__name__)

        candidates = []
        if self.vision_settings.get("pi", True) and self.pi_model:
            candidates.append(("pi", self.pi_model))
        if self.vision_settings.get("bioinformatics", True) and self.bio_model:
            candidates.append(("bioinformatics", self.bio_model))
        if self.vision_settings.get("supervisor", True) and self.supervisor_model:
            candidates.append(("supervisor", self.supervisor_model))

        if not candidates:
            logger.info(f"[{self._session_id}] No vision-capable model for report, using text-only")
            return await self._pi_chat(text_prompt, temperature, max_tokens)

        for agent_name, model in candidates:
            try:
                self._set_llm_context(f"report_{agent_name}")
                response = await self.llm_client.chat_with_vision(
                    model=model,
                    text_prompt=text_prompt,
                    images=images,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                )
                return response.content if response and response.content else ""
            except Exception as e:
                logger.warning(
                    f"[{self._session_id}] Vision failed with {agent_name} model ({model}): {e}"
                )
                continue

        logger.warning(f"[{self._session_id}] All vision models failed, falling back to text-only")
        self._set_llm_context("report_pi")
        return await self._pi_chat(text_prompt, temperature, max_tokens)

    async def _bio_chat_with_tools(
        self,
        messages: List[Dict],
        temperature: float = 0.2
    ):
        """Chat with Bioinformatics Agent using tools"""
        self._set_llm_context("report_bio")
        return await self.llm_client.chat_with_tools(
            model=self.bio_model,
            messages=messages,
            tools=REPORT_TOOLS,
            temperature=temperature,
            max_tokens=32768
        )