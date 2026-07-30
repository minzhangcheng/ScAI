# Publication-figure generation, review loop, and revision.
# Moved verbatim out of backend/agents/bioinformatics_agent.py; composed into BioinformaticsAgent.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import asyncio
import functools
import json
import logging
import base64
import io
import uuid
import re
import ast
import textwrap
import time
from pathlib import Path
from datetime import datetime

import aiofiles

from ..base import BaseAgent, AgentConfig
from ...utils.settings_provider import AgentSettingsProvider
from ...utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult, _TPL_NOOP_INSTALL,
)
from ...utils.checkpoint import CheckpointManager
import inspect
import json as _json
from backend.services import ora_kernel
from backend.services.marker_priors import MarkerDB, summarize_database_priors

from ...utils.formatting import format_revision_history, truncate_for_llm
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)


from .shared import (  # noqa: F401
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
    Tuple,
    asyncio,
    functools,
    json,
    logging,
    base64,
    io,
    uuid,
    re,
    ast,
    textwrap,
    time,
    Path,
    datetime,
    aiofiles,
    BaseAgent,
    AgentConfig,
    AgentSettingsProvider,
    validate_code_safety,
    DANGEROUS_PATTERNS,
    extract_code_from_markdown,
    LocalJupyterSandbox,
    CodeExecutionResult,
    _TPL_NOOP_INSTALL,
    CheckpointManager,
    inspect,
    _json,
    ora_kernel,
    MarkerDB,
    summarize_database_priors,
    _ORA_KERNEL_SRC,
    format_revision_history,
    truncate_for_llm,
    MECHANISM_EVIDENCE_POLICY,
    MAX_WORKSPACE_H5AD_FILES,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _static_h5ad_path,
    workspace_h5ad_write_guard,
    _OVERCLAIM_WORDS,
    _NEGATION_WORDS,
    _MAX_PROGRAMS,
    _MAX_GENES,
    _MAX_OBS_LEN,
    _DEFAULT_MISSING,
    _REQUIRED_FIELDS,
    _ensure_list,
    _overclaim_unnegated,
    validate_mechanism_program_payload,
    TOOLS,
    KNOWN_TOOL_NAMES,
    build_gate_skip_note,
    count_saved_figures,
    AgentType,
    AnalysisPhase,
    AnalysisState,
    MessageRole,
    CodeExecution,
    DataDrivenMechanismProgram,
    PublicationFigure,
    FigureReviewStatus,
    AnalysisStep,
    PlanStage,
    DATA_TYPE_SCRNASEQ,
    DATA_TYPE_BULK_RNASEQ,
    OpenAICompatibleClient,
    ToolCall,
    UpstreamExhaustedError,
    EMPTY_CONTENT_PLACEHOLDER,
    tool_calls_are_paired,
    append_missing_tool_results,
    PLAN_REVISION_NEEDED_TAG,
    strip_ansi,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_invalidated_file_stems,
    record_representation_change,
    unchanged_artifact_stems,
    workspace_artifact_versions,
    SupervisorEvalResult,
    _DYNAMIC_CONTEXT_PREFIX,
    _APPROVED_PLAN_PREFIX,
    _CELL_TYPE_ASSIGN_RE,
    _FAILED_EVIDENCE_OUTPUT_RE,
    build_approved_plan_context,
    inject_revised_plan_context,
    execution_supports_evidence,
    successful_evidence_executions,
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
    _build_swallowed_error_feedback,
    _MARKER_DB,
    _get_marker_db,
    marker_priors_write,
)


class FiguresMixin:
    """Publication-figure generation, review loop, and revision."""

    async def _handle_generate_publication_figure(
        self,
        code: str,
        figure_name: str,
        description: str,
        figure_type: str,
        state: AnalysisState,
        review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None
    ) -> str:
        """
        Handle generate_publication_figure tool call.
        Generates the figure, saves in dual format, and optionally gets PI review.

        Args:
            code: Python plotting code
            figure_name: Name for the figure file
            description: Figure description
            figure_type: Type of figure
            state: Analysis state to store the figure
            review_callback: Optional callback for PI review

        Returns:
            Tool result message
        """
        logger.info(f"[Bioinformatics] Generating publication figure: {figure_name}")

        # Validate code before execution
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return f"Error: Code validation failed - {error_msg}"

        # Ensure figures directory exists
        figures_dir = Path(self.sandbox.working_dir) / "figures"

        # Clean up figure name
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in figure_name)
        jpg_path = figures_dir / f"{safe_name}.jpg"
        pdf_path = figures_dir / f"{safe_name}.pdf"

        # Wrap code to save both JPG and PDF; paths are repr()'d inside the
        # shared wrapper to prevent injection.
        wrapped_code = _build_publication_figure_wrapper(code, figures_dir, jpg_path, pdf_path)

        expression_specs = extract_sensitive_expression_inputs(code)
        representation_will_change = mutates_expression_representation(code)
        preexisting_artifact_versions = (
            workspace_artifact_versions(self.sandbox.working_dir)
            if representation_will_change else {}
        )
        if (
            expression_specs
            and not prepares_expression_before_sensitive_call(code)
        ):
            preflight = await self.sandbox.execute(
                build_expression_preflight_code(expression_specs),
                save_to_history=False,
                skip_safety_check=True,
            )
            if preflight.error:
                return f"Figure generation blocked: {strip_ansi(preflight.error)}"
            record_expression_contract(
                state, parse_expression_contracts(preflight.output)
            )

        # Execute the code
        result = await self.sandbox.execute(wrapped_code, skip_safety_check=True)

        if result.error:
            result.error = strip_ansi(result.error)
            error_msg = f"Figure generation failed: {result.error}"
            logger.error(f"[Bioinformatics] {error_msg}")
            await self._notify("error", error_msg)
            return error_msg

        record_representation_change(state, code)
        if representation_will_change:
            record_invalidated_file_stems(
                state,
                unchanged_artifact_stems(
                    self.sandbox.working_dir, preexisting_artifact_versions
                ),
            )

        # Check if files were created
        if not jpg_path.exists():
            error_msg = f"JPG file was not created: {jpg_path}"
            logger.error(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Detect blank figures (empty white JPG with bbox_inches='tight' is typically <2KB)
        if jpg_path.stat().st_size < 2000:
            error_msg = f"Figure appears to be blank (file size: {jpg_path.stat().st_size} bytes). The plotting code may not have generated visible content."
            logger.warning(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Read JPG and compress to base64
        png_b64 = self._compress_image_to_base64(jpg_path, max_width=800)

        # Create PublicationFigure object (png_path field stores JPG path for web display)
        sandbox_state = state.sandbox_state or {}
        figure = PublicationFigure(
            id=str(uuid.uuid4()),
            name=safe_name,
            description=description,
            figure_type=figure_type,
            png_path=str(jpg_path),
            pdf_path=str(pdf_path),
            png_base64=png_b64,
            review_status=FigureReviewStatus.PENDING,
            round_number=int(state.current_round or 0),
            analysis_generation=int(sandbox_state.get("analysis_generation", 0) or 0),
            expression_provenance=dict(
                sandbox_state.get("expression_contract_v1") or {}
            ),
        )

        # Send figure to frontend for display
        await self._notify("figure", png_b64)

        # If review callback provided, get PI review
        # Allow up to 3 revisions (4 reviews total: initial + 3 revisions)
        if review_callback:
            revision_count = 0
            current_code = code
            revision_history = []  # Accumulate all rounds of PI feedback

            while revision_count <= 3:
                logger.info(f"[Bioinformatics] Submitting figure for PI review (attempt {revision_count + 1})")
                await self._notify("status", f"PI reviewing figure: {figure_name}...")

                review = await review_callback(figure)

                if review.get('approved', False):
                    quality_score = review.get('quality_score', 5)
                    # Mark figures approved under leniency (after multiple revisions with non-perfect score)
                    if revision_count >= 2 and quality_score < 8:
                        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
                        logger.info(f"[Bioinformatics] Figure {figure_name} accepted with reservations (score={quality_score}, revisions={revision_count})")
                    else:
                        figure.review_status = FigureReviewStatus.APPROVED
                        logger.info(f"[Bioinformatics] Figure {figure_name} approved by PI")
                    figure.revision_history = revision_history
                    self._upsert_publication_figure(state, figure)
                    await self._notify("status", f"Figure {figure_name} approved by PI")
                    return f"Figure {figure_name} has been approved by PI and saved.\nQuality score: {quality_score}/10\nFeedback: {review.get('feedback', 'Approved')}"

                # PI requested revision
                feedback = review.get('feedback', '')
                suggestions = review.get('suggestions', [])
                must_fix = review.get('must_fix', [])

                # If PI flagged a data/analysis issue that can't be fixed by revising plot code,
                # break out of the revision loop and defer to the main analysis flow
                if review.get('requires_reanalysis', False):
                    logger.warning(f"[Bioinformatics] Figure {figure_name} requires upstream reanalysis, skipping revision loop")
                    await self._notify("status", f"Figure {figure_name}: PI identified data-level issue, deferring to analysis")
                    figure.review_status = FigureReviewStatus.REVISION_REQUESTED
                    figure.pi_feedback = f"[REQUIRES REANALYSIS] {feedback}"
                    self._upsert_publication_figure(state, figure)
                    return (
                        f"Figure {figure_name} has a data-level issue that cannot be fixed by revising the plot code. "
                        f"PI feedback: {feedback}\n"
                        f"Please address this in the analysis pipeline and regenerate the figure later."
                    )

                # Record this round's feedback in cumulative history
                revision_history.append({
                    "round": revision_count + 1,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "must_fix": must_fix,
                    "quality_score": review.get('quality_score', None)
                })

                figure.pi_feedback = feedback
                figure.revision_history = revision_history

                if revision_count >= 3:
                    # Out of revision budget
                    break

                # Detect stalled revision loop: if the quality score hasn't
                # improved across the last 2 rounds, further revisions are
                # unlikely to help — accept the current version to avoid
                # burning time and tokens.
                if len(revision_history) >= 2:
                    prev_score = revision_history[-2].get("quality_score")
                    curr_score = revision_history[-1].get("quality_score")
                    if prev_score is not None and curr_score is not None and curr_score <= prev_score:
                        logger.warning(
                            f"[Bioinformatics] Figure {figure_name}: quality score stalled "
                            f"({prev_score} -> {curr_score}), accepting current version"
                        )
                        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
                        self._upsert_publication_figure(state, figure)
                        await self._notify("status", f"Figure {figure_name} accepted (revision stalled)")
                        return (
                            f"Figure {figure_name} accepted with reservations (quality score did not improve "
                            f"after revision: {prev_score} -> {curr_score}).\n"
                            f"Last feedback: {feedback}\n"
                            f"Please move on to the next analysis step."
                        )

                # Try to revise the figure
                await self._notify("status", f"Revising figure based on PI feedback...")
                logger.info(f"[Bioinformatics] Revising figure {figure_name}: {feedback}")

                # Generate revised code with full revision history
                try:
                    revised_code = await self._revise_figure_code(
                        current_code, feedback, suggestions, must_fix, figure_type, state, figure.png_base64,
                        revision_history=revision_history
                    )
                except UpstreamExhaustedError:
                    raise
                except Exception as revise_err:
                    logger.error(f"[Bioinformatics] _revise_figure_code failed: {revise_err}, keeping current version")
                    await self._notify("status", f"Figure revision failed ({revise_err}), keeping current version")
                    break

                # Only increment revision count after successful code generation
                revision_count += 1
                figure.revision_count = revision_count

                # Validate revised code before execution
                is_valid, error_msg = self._validate_code(revised_code)
                if not is_valid:
                    logger.warning(f"[Bioinformatics] Revised code failed validation: {error_msg}")
                    break

                # Re-execute with revised code (rebuild wrapper instead of str.replace to avoid corruption)
                revised_wrapped = _build_publication_figure_wrapper(revised_code, figures_dir, jpg_path, pdf_path)
                result = await self.sandbox.execute(
                    revised_wrapped, skip_safety_check=True
                )

                if not result.error and jpg_path.exists():
                    # Detect blank revised figure
                    if jpg_path.stat().st_size < 2000:
                        logger.warning(f"[Bioinformatics] Revised figure appears blank (size: {jpg_path.stat().st_size} bytes)")
                        break
                    # Update figure with new image
                    figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                    await self._notify("figure", figure.png_base64)
                    current_code = revised_code
                else:
                    result.error = strip_ansi(result.error) if result.error else result.error
                    logger.warning(f"[Bioinformatics] Revision failed: {result.error}")
                    error_msg = str(result.error)[:1000]

                    # Technique Agent documentation lookup for technical errors
                    extra_context = ""
                    if self.technique_agent:
                        err_type = extract_error_type(error_msg)
                        if err_type in TECHNICAL_ERROR_TYPES:
                            logger.info(f"[Bioinformatics] Looking up docs for {err_type} error")
                            doc_hint = await self.technique_agent.lookup_from_error(
                                self.sandbox, error_msg, revised_code
                            )
                            if doc_hint:
                                extra_context = f"\n\n[API Documentation]\n{doc_hint}"
                                logger.info(f"[Bioinformatics] Got documentation hint for {err_type}")

                    # Auto-retry: feed error + docs back to LLM
                    logger.info(f"[Bioinformatics] Auto-retrying revision with error context")
                    try:
                        retry_code = await self._revise_figure_code(
                            revised_code,
                            f"The previous code revision caused a runtime error:\n{error_msg}\nFix this error while keeping ALL the style improvements from the revision.{extra_context}",
                            [f"Fix the runtime error: {error_msg}"],
                            ["The code must execute without errors"],
                            figure_type, state, figure.png_base64,
                            revision_history=revision_history
                        )
                        is_valid_retry, err_retry = self._validate_code(retry_code)
                        if is_valid_retry:
                            retry_wrapped = _build_publication_figure_wrapper(retry_code, figures_dir, jpg_path, pdf_path)
                            retry_result = await self.sandbox.execute(
                                retry_wrapped, skip_safety_check=True
                            )
                            if not retry_result.error and jpg_path.exists() and jpg_path.stat().st_size >= 2000:
                                logger.info(f"[Bioinformatics] Auto-retry succeeded for {figure_name}")
                                figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                                await self._notify("figure", figure.png_base64)
                                current_code = retry_code
                                continue  # Back to review loop
                            else:
                                logger.warning(f"[Bioinformatics] Auto-retry also failed: {retry_result.error}")
                    except UpstreamExhaustedError:
                        raise
                    except Exception as retry_err:
                        logger.warning(f"[Bioinformatics] Auto-retry exception: {retry_err}")
                    break  # Both attempts failed

            # After 3 attempts, save with revision_requested status
            figure.review_status = FigureReviewStatus.REVISION_REQUESTED
            self._upsert_publication_figure(state, figure)
            logger.warning(f"[Bioinformatics] Figure {figure_name} not fully approved after {revision_count} revisions")
            return f"Figure {figure_name} saved after {revision_count} revision attempts. Status: needs further revision.\nLast feedback: {figure.pi_feedback}"

        # No review callback, just save the figure
        self._upsert_publication_figure(state, figure)
        return f"Figure {figure_name} has been generated and saved.\nJPG: {jpg_path}\nPDF: {pdf_path}"

    def _upsert_publication_figure(self, state, figure):
        """Replace existing figure with same name, or append if new."""
        refreshed_stems = {
            Path(str(figure.name)).stem,
            Path(str(figure.png_path)).stem,
        }
        state.invalidated_artifact_stems = [
            stem for stem in (state.invalidated_artifact_stems or [])
            if stem not in refreshed_stems
        ]
        existing_idx = next(
            (i for i, f in enumerate(state.publication_figures) if f.name == figure.name),
            None
        )
        if existing_idx is not None:
            state.publication_figures[existing_idx] = figure
        else:
            state.publication_figures.append(figure)

    def _compress_image_to_base64(self, image_path: Path, max_width: int = 800) -> str:
        """
        Compress an image and convert to base64 for efficient transmission.

        Args:
            image_path: Path to the image file
            max_width: Maximum width for the compressed image

        Returns:
            Base64 encoded JPEG string
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                # Calculate resize ratio
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    # Handle PIL version compatibility
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS  # For older PIL versions
                    img = img.resize(new_size, resample_method)

                # Convert to RGB if necessary (handle RGBA, P modes)
                if img.mode in ('RGBA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        # Safely get alpha channel - check split() returns enough channels
                        channels = img.split()
                        if len(channels) >= 4:
                            background.paste(img, mask=channels[3])
                        else:
                            background.paste(img)
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            logger.exception("[Bioinformatics] Failed to compress image")
            # Fallback: read raw file and encode
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    async def _revise_figure_code(
        self,
        original_code: str,
        feedback: str,
        suggestions: List[str],
        must_fix: List[str],
        figure_type: str,
        state: AnalysisState = None,
        image_base64: str = None,
        revision_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate revised plotting code based on PI feedback.

        Args:
            original_code: The original plotting code
            feedback: Overall feedback from PI
            suggestions: List of improvement suggestions
            must_fix: List of critical issues that must be fixed
            figure_type: Type of figure
            state: Analysis state with treatment group info
            image_base64: Base64 encoded current figure image for visual review
            revision_history: Accumulated feedback from all revision rounds (text only)

        Returns:
            Revised Python plotting code
        """
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        must_fix_text = "\n".join(f"- {m}" for m in must_fix) if must_fix else "None"

        exp_design_constraint = ""
        if state and state.treatment_groups:
            group_names = ', '.join(g.name for g in state.treatment_groups)
            exp_design_constraint = f"""
**CRITICAL - Experimental Design Constraint:**
The actual experimental groups are: {group_names}
You must ONLY use data from these groups. Do NOT create or reference groups that don't exist.
If the PI feedback asks you to add groups not in the experimental design, IGNORE that part of the feedback and only plot data for the actual groups.
"""

        figure_context = "**Current Figure:** An image of the current figure is attached. Review it carefully to understand what needs to be changed.\n\n" if image_base64 else ""

        # Build revision history section (show previous rounds, exclude current which is shown below)
        history_section = ""
        if revision_history and len(revision_history) > 1:
            past_rounds = revision_history[:-1]
            history_parts = []
            for entry in past_rounds:
                round_num = entry.get('round', '?')
                hist_feedback = entry.get('feedback', '')
                hist_suggestions = entry.get('suggestions', [])
                hist_must_fix = entry.get('must_fix', [])
                round_text = f"### Round {round_num}\n"
                round_text += f"- Feedback: {hist_feedback}\n"
                if hist_suggestions:
                    round_text += f"- Suggestions: {'; '.join(str(s) for s in hist_suggestions)}\n"
                if hist_must_fix:
                    round_text += f"- Must Fix: {'; '.join(str(m) for m in hist_must_fix)}\n"
                history_parts.append(round_text)
            history_section = (
                "**Previous Revision History (CRITICAL - read carefully):**\n"
                "The following issues were identified and some were FIXED in prior rounds.\n"
                "You MUST:\n"
                "1. Keep ALL previously applied fixes (do NOT revert them)\n"
                "2. Address the NEW issues from the current round\n"
                "3. Before submitting, mentally verify each previous fix is still present in your code\n\n"
                + "\n".join(history_parts) + "\n"
            )

        prompt = f"""Please revise the following plotting code based on PI feedback.

{figure_context}**Original Code:**
```python
{original_code}
```

**Figure Type:** {figure_type}
{exp_design_constraint}
{history_section}**PI Feedback (Current Round):** {feedback}

**Improvement Suggestions (Current Round):**
{suggestions_text}

**Critical Issues (MUST FIX - Current Round):**
{must_fix_text}

**Requirements:**
1. Fix all critical issues listed above
2. Address as many suggestions as possible
3. CRITICAL: Do NOT revert previously applied fixes. If Round 1 changed grid lines to dashed, keep them dashed. If Round 2 added axis labels, keep them.
4. Maintain the same overall figure structure
5. Keep using the same data variables (assume they exist), but REMOVE any fabricated/dummy data for groups not in the experimental design
6. Before calling set_xticklabels()/set_yticklabels(), ALWAYS call set_xticks()/set_yticks() first with matching count
7. For pandas Categorical columns, use .cat.add_categories() before assigning new category values
8. Use legend_.get_legend_handles_labels() instead of legend_.legendHandles (deprecated in newer matplotlib)
9. For ax.bar()/ax.barh(), the 3rd positional arg IS the bar thickness (width for bar, height for barh) — never also pass it as a width=/height= keyword, or you get "got multiple values for argument". Use the explicit keyword form only: ax.barh(y, values, height=0.7, ...)
10. Return ONLY the revised Python code, no explanations

**Common fixes for publication figures:**
- Add/improve axis labels and titles
- Adjust font sizes (plt.rcParams['font.size'] = 12)
- Add legends with proper placement
- Use colorblind-friendly palettes
- Add scale bars or annotations
- Improve color contrast
- Remove unnecessary elements

Return only the Python code:"""

        self._set_llm_context()
        # Use slightly lower temperature for code revision (needs precision)
        revision_temp = min(self.config.temperature, 0.2)
        if image_base64 and self.has_vision:
            # Use vision API so the agent can see the current figure
            try:
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=prompt,
                    images=[image_base64],
                    temperature=revision_temp,
                    max_tokens=32768
                )
            except Exception as vision_err:
                _is_vision_unsupported = (
                    "NotFoundError" in type(vision_err).__name__
                    or "404" in str(vision_err)
                    or "1210" in str(vision_err)  # ZhipuAI: API params error (non-vision model)
                )
                if _is_vision_unsupported:
                    # Model does not support vision input — disable for this session and fall back
                    logger.warning(
                        f"[Bioinformatics] Model {self.model} does not support vision "
                        f"({vision_err}), disabling vision for this session"
                    )
                    self.has_vision = False
                    prompt = prompt.replace(
                        "**Current Figure:** An image of the current figure is attached. "
                        "Review it carefully to understand what needs to be changed.\n\n",
                        "**Current Figure:** (Image not available — model does not support vision. "
                        "Revise code based on PI feedback text only.)\n\n",
                    )
                    response = await self.llm_client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=revision_temp,
                        max_tokens=32768
                    )
                else:
                    raise
        else:
            response = await self.llm_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=revision_temp,
                max_tokens=32768
            )

        response_text = response.content if response and response.content else None
        if not response_text:
            logger.warning("[Bioinformatics] Empty response from LLM for code revision")
            return original_code  # Return unchanged code

        # Extract code from markdown response
        code = extract_code_from_markdown(response_text)

        return code.strip()
