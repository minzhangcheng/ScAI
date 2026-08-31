"""Text-diagram review.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from typing import List, Dict, Any
import logging

from ...utils.formatting import safe_parse_json, format_revision_history
from ...constants import REASONING_MODEL_MIN_TOKENS

logger = logging.getLogger(__name__)


from ...models.schemas import AnalysisState


# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ...utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402

from .shared import (  # noqa: F401
    _SPECIFIC_SECTION_IDS,
    _SECTION_ID_TO_DEFAULT_TITLE,
    _PLACEHOLDER_SECTION_TITLES,
    SupervisorRecord,
    SupervisorEvalResult,
    _CRITICAL_OUTPUT_PATTERNS,
    _PERCENTAGE_PATTERN,
    _INF_NAN_PATTERN,
    _SUPERVISOR_SYSTEM_PROMPT,
    _serialize_record,
    persist_graphic_abstract,
)


class DiagramsMixin:
    """Text-diagram review."""

    async def review_text_diagram(
        self,
        state: AnalysisState,
        text_diagram: str,
        revision_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review an ASCII/Unicode text-mode graphic abstract as the final fallback reviewer
        (used when both the PI and Bioinformatics agent fail). Pure text, no vision needed.

        Raises on any failure so the workflow can decide to skip review entirely.
        """
        logger.info(f"[{self.session_id}][Supervisor] Reviewing text diagram (final fallback)")

        history_text = format_revision_history(revision_history)

        system_prompt = (
            "You are a supervising reviewer checking a text-mode (ASCII/Unicode) pathway "
            "diagram for a publication. You are the final fallback reviewer. Read the "
            "diagram character by character and respond strictly in the requested JSON."
        )

        user_message = f"""## Review the text diagram

**Project:** {state.name}
**Objective:** {state.initial_instruction[:300] if state.initial_instruction else 'Biomedical research'}

{history_text}

**Diagram:**
```
{text_diagram}
```

Respond in JSON:

```json
{{
    "quality_score": 1-10,
    "feedback": "Plain-English assessment",
    "strengths": ["..."],
    "issues": ["..."],
    "suggestions": ["Concrete text edits"],
    "must_fix": ["Critical omissions"]
}}
```
"""
        self.refresh_runtime_config_if_needed()
        self._set_llm_context()
        response = await self._budgeted_text_call(
            self.llm_client.chat,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Strict review
            max_tokens=REASONING_MODEL_MIN_TOKENS,  # reasoning models need headroom or content comes back empty
            prefer_stream=True,
        )
        response_text = response.content if response and response.content else ""
        if not response_text.strip():
            raise ValueError("Supervisor text diagram review returned empty response")

        # Parse JSON using standardized implementation
        data = safe_parse_json(response_text)
        if not data or "feedback" not in data:
            logger.warning(f"[{self.session_id}][Supervisor] Failed to parse diagram review JSON: {response_text}")
            raise ValueError("Supervisor text diagram review did not return parseable JSON")

        return {
            "quality_score": int(data.get("quality_score", 5)),
            "feedback": data.get("feedback", ""),
            "strengths": data.get("strengths", []),
            "issues": data.get("issues", []),
            "suggestions": data.get("suggestions", []),
            "must_fix": data.get("must_fix", [])
        }
