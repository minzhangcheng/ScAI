"""PI review mixin for DetailedReportGenerator."""
from typing import Dict, Any, List, Optional

from backend.constants import NO_OUTPUT_TOKEN_CAP
from backend.utils.report_html import _safe_parse_json


class PIReviewMixin:
    """Provides PI review methods for report generation."""

    @staticmethod
    def _empty_analysis_summary() -> Dict[str, Any]:
        """Return an empty analysis summary with the expected schema."""
        return {
            "analysis_approach": "",
            "methodology_overview": "",
            "workflow_summary": "",
            "key_results": "",
            "figure_descriptions": {},
            "parameters_used": {},
        }

    async def _pi_review_section(
        self,
        section_plan: Dict[str, Any],
        figures_generated: List[Dict],
        tables_generated: List[Dict],
        findings: Dict[str, Any],
        language: str
    ) -> Dict[str, Any]:
        """
        PI Agent reviews the generated section output and decides if more data is needed.

        Returns:
            Dict with:
            - satisfied: bool - whether PI is satisfied with current output
            - feedback: str - feedback for bioinformatics agent if not satisfied
            - missing_elements: List[str] - specific items still needed
        """
        import json
        import logging

        logger = logging.getLogger(__name__)

        # Build summary of what was generated
        fig_summary = "\n".join([
            f"- {f.get('name', 'Unknown')}: {f.get('caption', '')[:100]}"
            for f in figures_generated
        ]) if figures_generated else "No figures generated"

        tbl_summary = "\n".join([
            f"- {t.get('name', 'Unknown')}"
            for t in tables_generated
        ]) if tables_generated else "No tables generated"

        # Build expected outputs from plan
        expected_figs = "\n".join([
            f"- {f['name']}: {f['description']}"
            for f in section_plan.get("figures", [])
        ]) if section_plan.get("figures") else "No specific figures required"

        expected_tbls = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in section_plan.get("tables", [])
        ]) if section_plan.get("tables") else "No specific tables required"

        prompt = f"""You are the PI reviewing the bioinformatics output for a report section.

## Section: {section_plan.get('title', 'Unknown')}
Description: {section_plan.get('description', '')}
Analysis Type: {section_plan.get('analysis_type', 'general')}

## Expected Outputs (from plan):
### Figures:
{expected_figs}

### Tables:
{expected_tbls}

## Actually Generated:
### Figures ({len(figures_generated)} total):
{fig_summary}

### Tables ({len(tables_generated)} total):
{tbl_summary}

## Research Context:
Key findings: {json.dumps(findings.get('bioinformatics', [])[:5], ensure_ascii=False)}

## Task
Evaluate whether the generated outputs adequately cover the section requirements.

Consider:
1. Are all required figures present?
2. Do the figures address the section's goals?
3. Is the data quality sufficient for publication?
4. Are there important visualizations missing?

Reply in English

Return JSON:
{{
    "satisfied": true/false,
    "quality_score": 1-10,
    "feedback": "Detailed feedback for the bioinformatics agent",
    "missing_elements": ["List of specific figures/tables still needed"],
    "improvement_suggestions": ["Suggestions for improving existing figures"]
}}

Be strict but fair. For a publication-quality report:
- Each section should have comprehensive visualizations
- Key findings should be clearly illustrated
- Statistical information should be included where relevant
"""

        response = await self._pi_chat(prompt, temperature=0.3)

        try:
            result = _safe_parse_json(response)
            if result:
                return {
                    "satisfied": result.get("satisfied", False),
                    "quality_score": result.get("quality_score", 5),
                    "feedback": result.get("feedback", ""),
                    "missing_elements": result.get("missing_elements", []),
                    "improvement_suggestions": result.get("improvement_suggestions", [])
                }
        except Exception as e:
            logger.exception("Failed to parse PI review")

        # Retry once before falling back to heuristic
        try:
            logger.info("Retrying PI review LLM call after JSON parse failure")
            retry_response = await self._pi_chat(prompt, temperature=0.2)
            retry_result = _safe_parse_json(retry_response)
            if retry_result:
                return {
                    "satisfied": retry_result.get("satisfied", False),
                    "quality_score": retry_result.get("quality_score", 5),
                    "feedback": retry_result.get("feedback", ""),
                    "missing_elements": retry_result.get("missing_elements", []),
                    "improvement_suggestions": retry_result.get("improvement_suggestions", [])
                }
        except Exception as e:
            logger.exception("PI review retry also failed")

        # Default: not satisfied if fewer figures than planned
        min_figures = len(section_plan.get("figures", []))
        is_satisfied = len(figures_generated) >= max(min_figures, 2)

        return {
            "satisfied": is_satisfied,
            "quality_score": 5 if is_satisfied else 3,
            "feedback": "Please ensure all required figures are generated." if not is_satisfied else "Acceptable output.",
            "missing_elements": [],
            "improvement_suggestions": []
        }

    async def _pi_guide_bio_agent(
        self,
        question: str,
        context: str,
        section_plan: Dict[str, Any],
        findings: Dict[str, Any],
        language: str
    ) -> str:
        """PI provides guidance to bioinformatics agent during report generation"""
        import json

        prompt = f"""You are the PI guiding the bioinformatics analyst in generating report figures.

## Section Being Generated: {section_plan.get('title', '')}
{section_plan.get('description', '')}

## Bioinformatics Agent's Question:
{question}

## Context:
{context}

## Research Background:
{json.dumps(findings.get('bioinformatics', [])[:10], ensure_ascii=False)}

Provide specific analytical guidance

Be specific and actionable. Include:
1. What specific visualizations to create
2. What data/metrics to include
3. How to make figures publication-quality
4. Any statistical considerations
"""

        return await self._pi_chat(prompt, temperature=0.4)

    # Phase classification constants (shared with _pi_review_analysis)
    _PHASE_NAMES = {
        "qc": "Quality Control & Filtering",
        "normalization": "Normalization & Feature Selection",
        "integration": "Batch Correction & Integration",
        "dimensionality_reduction": "Dimensionality Reduction",
        "clustering": "Cell Clustering",
        "annotation": "Cell Type Annotation",
        "differential_expression": "Differential Expression Analysis",
        "pathway": "Pathway & Functional Analysis",
        "trajectory": "Trajectory & RNA Velocity Analysis",
        "communication": "Cell-Cell Communication Analysis",
        "visualization": "Visualization",
        "other": "Additional Analysis",
    }
    _PHASE_ORDER = list(_PHASE_NAMES.keys())
    _CATEGORY_KEYWORDS = {
        "qc": ["n_genes_by_counts", "pct_counts_mt", "filter_cells", "filter_genes",
                "mito_genes", "quality_control", "doublet", "scrublet"],
        "normalization": ["normalize_total", "log1p", "highly_variable_genes",
                          "sc.pp.scale", "regress_out"],
        "integration": ["harmonypy", "scanorama", "bbknn", "harmony_integrate",
                        "ingest", "sc.external.pp"],
        "dimensionality_reduction": ["sc.tl.pca", "sc.tl.umap", "sc.tl.tsne",
                                     "sc.pp.pca", "n_comps"],
        "clustering": ["sc.tl.leiden", "sc.tl.louvain", "sc.pp.neighbors",
                       "resolution=", "n_neighbors="],
        "annotation": ["cell_type", "annotate", "marker_genes",
                       "sc.tl.rank_genes_groups"],
        "differential_expression": ["rank_genes_groups", "wilcoxon", "t-test",
                                    "volcano", "de_results", "deg_", "logfoldchange"],
        "pathway": ["enrichr", "gsea", "pathway", "go_", "kegg", "gseapy",
                    "decoupler", "gene_set"],
        "trajectory": ["scvelo", "dpt", "pseudotime", "paga", "velocity",
                       "cellrank", "diffusion_map"],
        "communication": ["liana", "cellchat", "ligand", "receptor",
                          "squidpy", "omnipath", "cell_communication"],
    }

    async def _pi_review_phase(
        self,
        phase: Dict[str, Any],
        figure_images: Dict[str, str],
        table_summaries: Dict[str, str],
        state: "AnalysisState",
        language: str
    ) -> str:
        """
        PI reviews one analysis phase with actual code, figures, and table data.
        Uses vision API when figures are available for this phase.

        Returns structured text summary of methodology, workflow, and results for this phase.
        """
        phase_name = phase["phase_name"]
        phase_id = phase["phase_id"]
        cells = phase["cells"]

        # Build code text from cells
        code_parts = []
        output_parts = []
        phase_saved_figures = set()

        for j, cell in enumerate(cells):
            code = cell["code"]
            output = cell.get("output", "")
            code_parts.append(f"# --- Cell {j+1} ---\n{code}")
            if output:
                output_parts.append(f"[Cell {j+1} output]: {output[:500]}")
            for fname in cell.get("saved_figures", []):
                phase_saved_figures.add(fname)

        code_text = "\n\n".join(code_parts)
        output_text = "\n".join(output_parts) if output_parts else "(No output)"

        # Truncate code if too long — keep first 3 and last 2 cells in full, summarize middle
        if len(code_text) > 8000 and len(cells) > 5:
            first_cells = cells[:3]
            last_cells = cells[-2:]
            middle_count = len(cells) - 5
            first_code = "\n\n".join(f"# --- Cell {j+1} ---\n{c['code']}" for j, c in enumerate(first_cells))
            last_code = "\n\n".join(f"# --- Cell {len(cells)-1+j} ---\n{c['code']}" for j, c in enumerate(last_cells))
            code_text = f"{first_code}\n\n# [... {middle_count} intermediate cells omitted for brevity ...]\n\n{last_code}"

        if len(output_text) > 3000:
            output_text = output_text[:3000] + "\n... (truncated)"

        # Match figures to this phase
        phase_images = []
        # 1. Match by saved filenames in cells
        for fname in phase_saved_figures:
            if fname in figure_images:
                phase_images.append(figure_images[fname])
        # 2. Fallback: match by inferred type
        if not phase_images:
            type_to_phase = {
                "qc": ["qc"], "normalization": [], "integration": [],
                "dimensionality_reduction": ["umap", "scatter"],
                "clustering": ["umap", "clustering"], "annotation": ["annotation", "umap"],
                "differential_expression": ["volcano", "heatmap"],
                "pathway": ["pathway"], "trajectory": ["trajectory", "velocity"],
                "communication": [], "visualization": [],
            }
            matching_types = type_to_phase.get(phase_id, [])
            for fname, img_b64 in figure_images.items():
                fig_type = self._infer_figure_type(fname)
                if fig_type in matching_types and len(phase_images) < 5:
                    phase_images.append(img_b64)

        # Cap at 5 images per phase
        phase_images = phase_images[:5]

        # Build table context for this phase
        table_context = ""
        if table_summaries:
            relevant_tables = []
            for tbl_name, tbl_summary in table_summaries.items():
                tbl_lower = tbl_name.lower()
                # Simple relevance matching
                if phase_id == "differential_expression" and ("de_" in tbl_lower or "deg" in tbl_lower or "marker" in tbl_lower):
                    relevant_tables.append(tbl_summary)
                elif phase_id == "pathway" and ("pathway" in tbl_lower or "enrich" in tbl_lower or "gsea" in tbl_lower):
                    relevant_tables.append(tbl_summary)
                elif phase_id == "annotation" and ("marker" in tbl_lower or "cluster" in tbl_lower):
                    relevant_tables.append(tbl_summary)
            if relevant_tables:
                table_context = "\n\n## Relevant Table Data\n" + "\n\n".join(relevant_tables[:3])

        # Build prompt
        figure_note = f"\n[{len(phase_images)} figure(s) attached for visual inspection]" if phase_images else "\n[No figures attached for this phase]"
        if phase_saved_figures:
            figure_note += f"\nFigure filenames: {', '.join(sorted(phase_saved_figures))}"

        lang_instruction = "Write 200-400 words. Be specific about parameters and quantitative results."

        prompt = f"""You are a Principal Investigator reviewing your bioinformatics team's single-cell RNA sequencing analysis.
Read the analysis code below carefully and examine the accompanying figures.

## Analysis Phase: {phase_name}

## Python Code Executed:
```python
{code_text}
```

## Code Output:
{output_text}

## Figures Generated in This Phase:
{figure_note}
{table_context}

## Task:
Provide a detailed technical summary of this analysis phase covering:
1. **Methods**: What specific algorithms, parameters, and tools were used? (e.g., "Leiden clustering with resolution=0.8", "Harmony integration across batches")
2. **Workflow**: What was the step-by-step process in this phase?
3. **Results**: Based on the code output and figures, what were the key results? (e.g., "Identified 15 clusters", "Filtered from 25,000 to 18,500 cells")
4. **Figure descriptions**: For each figure you can see, describe what it actually shows visually.

{lang_instruction}
"""

        # Use vision API if we have images (fallback chain handled internally), otherwise text-only
        if phase_images:
            return await self._chat_with_vision_fallback(
                text_prompt=prompt, images=phase_images,
                temperature=0.3, max_tokens=NO_OUTPUT_TOKEN_CAP, request_timeout=600.0,
            )
        else:
            return await self._pi_chat(prompt, temperature=0.3)

    async def _pi_synthesize_review(
        self,
        phase_reviews: List[Dict[str, Any]],
        state: "AnalysisState",
        language: str
    ) -> Dict[str, Any]:
        """
        PI synthesizes all phase reviews into a comprehensive analysis summary.

        Returns dict with methodology_overview, workflow_summary, key_results,
        figure_descriptions, and parameters_used.
        """
        # Format phase reviews
        reviews_text = ""
        for pr in phase_reviews:
            reviews_text += f"\n### {pr['phase_name']}\n{pr['summary']}\n"

        lang_instruction = "Write all content in English"

        prompt = f"""You are a Principal Investigator writing a comprehensive summary of your team's
bioinformatics analysis. Below are detailed reviews of each analysis phase.

## Research Project
Title: {state.name}
Objective: {state.initial_instruction}

## Phase-by-Phase Reviews:
{reviews_text}

## Task:
Synthesize these phase reviews into a comprehensive analysis summary.
{lang_instruction}

Return a JSON object with this structure:
{{
    "methodology_overview": "200-300 word overview of ALL methods, parameters, and tools used across the entire analysis pipeline, from raw data processing to final results",
    "workflow_summary": "Step-by-step workflow as a numbered list, from raw data to final results. Include specific methods and parameters at each step.",
    "key_results": "All key quantitative results discovered: cell counts before/after filtering, number of clusters, cell types identified, significant genes found, key pathways, etc.",
    "figure_descriptions": {{"filename.jpg": "Description of what this figure shows based on visual inspection"}},
    "parameters_used": {{"parameter_name": "value and brief rationale"}}
}}

IMPORTANT:
- Be specific about methods (e.g., "Leiden at resolution 0.8" not just "clustering")
- Include ALL quantitative results from the analysis
- The figure_descriptions should only include figures you actually observed in the reviews
- The parameters_used should list key analysis parameters (resolution, n_neighbors, thresholds, etc.)
"""
        import logging
        logger = logging.getLogger(__name__)

        try:
            response = await self._pi_chat(prompt, temperature=0.3)
        except Exception as e:
            logger.exception("PI synthesis LLM call failed")
            all_reviews = "\n\n".join(pr["summary"] for pr in phase_reviews if pr["summary"])
            fallback = self._empty_analysis_summary()
            fallback["methodology_overview"] = all_reviews[:2000] if all_reviews else ""
            return fallback

        try:
            result = _safe_parse_json(response)
            if result and "methodology_overview" in result:
                logger.info("PI synthesis successful")
                return result
        except Exception as e:
            logger.exception("Failed to parse PI synthesis")

        # Fallback: construct basic summary from raw review texts
        logger.warning("PI synthesis JSON parsing failed, constructing fallback summary")
        all_reviews = "\n\n".join(pr["summary"] for pr in phase_reviews if pr["summary"])
        fallback = self._empty_analysis_summary()
        fallback["methodology_overview"] = all_reviews[:2000] if all_reviews else ""
        return fallback

    async def _pi_comprehensive_review(
        self,
        code_digest: str,
        figure_images: Dict[str, str],
        table_summaries: Dict[str, str],
        state: "AnalysisState",
        language: str
    ) -> Dict[str, Any]:
        """
        PI Agent performs a single comprehensive review of the entire analysis.

        Reads the complete code digest, views all figures via vision API, and reads
        table data. Writes a from-scratch summary covering analysis approach,
        workflow, and results.

        Returns analysis_summary dict compatible with downstream report generation.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Build table context
        table_text = ""
        if table_summaries:
            table_parts = list(table_summaries.values())[:10]
            table_text = "\n\n".join(table_parts)

        # Build figure filename list for the prompt
        figure_filenames = list(figure_images.keys())
        figures_list = "\n".join(f"  {i+1}. {fname}" for i, fname in enumerate(figure_filenames))

        # Collect all images for vision API
        all_images = list(figure_images.values())

        lang_instruction = "Write all content in English. analysis_approach 300-500 words, analysis_results 300-500 words."

        exploration_context = ""
        if state.exploration_report and hasattr(state.exploration_report, "research_direction_candidates") and state.exploration_report.research_direction_candidates:
            exploration_context += "### Initial Exploration Report (Possible Directions):\n"
            for idx, c in enumerate(state.exploration_report.research_direction_candidates):
                exploration_context += f"{idx+1}. {c.get('direction', '')}\n   Rationale: {c.get('rationale', '')}\n"

        if state.user_chosen_directions:
            exploration_context += "\n### User Selected Directions (Modifications):\n"
            for d in state.user_chosen_directions:
                exploration_context += f"- {d.get('direction', '')} (Notes: {d.get('notes', '')})\n"
        elif getattr(state, "user_chosen_direction", None):
            exploration_context += f"\n### User Selected Direction:\n- {state.user_chosen_direction}\n"

        if exploration_context:
            exploration_context = f"\n## Phase 1 Exploration & User Directions\n{exploration_context}\n"

        method_decision_context = ""
        if state.method_decisions:
            lines = []
            profiles = {item.id: item for item in state.bulk_input_profiles}
            for decision in state.method_decisions:
                profile = profiles.get(decision.input_profile_id)
                lines.append(
                    f"- Bulk input {decision.input_profile_id} "
                    f"(scale={getattr(profile, 'declared_scale', 'unknown')}, "
                    f"profile_status={getattr(profile, 'profile_status', 'unknown')}): "
                    f"claim_level={decision.claim_level}; method={decision.method}; "
                    f"rationale={decision.rationale}; execution={decision.execution_id}."
                )
            method_decision_context = (
                "\n## Recorded Bulk Method Decisions\n" + "\n".join(lines) + "\n"
            )

        prompt = f"""You are a Principal Investigator. Your bioinformatics team has completed a single-cell RNA sequencing analysis. Below is the complete analysis notebook (all code and outputs in chronological order), along with all result figures and table data.

Your task: Read through the analysis code carefully from beginning to end, examine every attached figure, and review the table data. Then write a comprehensive summary FROM SCRATCH that demonstrates your deep understanding of the entire analysis.

## Research Project
Title: {state.name}
Objective: {state.initial_instruction}
{exploration_context}
## Complete Analysis Notebook (Code + Outputs)
{code_digest}

## Table Data Summaries
{table_text if table_text else "(No tables available)"}

## Result Figures ({len(figure_filenames)} figures attached for visual inspection)
{figures_list if figures_list else "(No figures available)"}

{self._build_score_definitions_context(state)}
{method_decision_context}

## Task
{lang_instruction}

Return a JSON object with this exact structure:
{{
    "analysis_approach": "Overall analysis strategy and scientific rationale. Explain: What was the overall analysis strategy? How was the analysis pipeline designed? What scientific questions guided the analysis? How did findings from earlier steps inform later analysis decisions? What was the rationale behind key methodological choices?",

    "analysis_workflow": "Complete step-by-step analysis pipeline. Write as a numbered list from raw data processing to final results. Include specific methods, tools, parameters, and software used at each step. Example: '1. Data loading: Read 10x Genomics data with sc.read_10x_h5(), loaded N cells × M genes...'",

    "analysis_results": "All key quantitative findings. Include: cell counts before/after filtering, number of clusters identified, cell types annotated, number of significant DEGs found, key pathways enriched, and any other quantitative results. Be specific with numbers.",

    "figure_descriptions": {{
        "filename.jpg": "Description of what this figure actually shows visually and its scientific significance (50-100 words per figure)"
    }},

    "parameters_used": {{
        "parameter_name": "value and brief rationale"
    }},

    "score_definitions": [
        {{"score_name": "name", "method": "computation method", "description": "what it measures", "gene_list_source": "source of genes"}}
    ]
}}

IMPORTANT:
- Read the code carefully to understand the COMPLETE analysis pipeline
- For analysis_approach, explain the THINKING behind the analysis — not just what was done, but WHY
- For analysis_workflow, be specific about methods and parameters (e.g., "Leiden clustering at resolution=0.8" not just "clustering")
- For analysis_results, include ALL quantitative results you can find in the code outputs
- For figure_descriptions, describe what you actually SEE in each attached figure
- For parameters_used, list key analysis parameters from the code (resolution, n_neighbors, thresholds, etc.)
- For score_definitions, identify ALL scoring/enrichment computations in the code and document what each score measures and how it was computed
"""

        try:
            if all_images:
                response = await self._chat_with_vision_fallback(
                    text_prompt=prompt, images=all_images[:20],
                    temperature=0.3, max_tokens=NO_OUTPUT_TOKEN_CAP, request_timeout=600.0,
                )
            else:
                response = await self._pi_chat(prompt, temperature=0.3, max_tokens=NO_OUTPUT_TOKEN_CAP)
        except Exception as e:
            logger.exception("PI comprehensive review failed")
            return self._empty_analysis_summary()

        # Parse JSON response
        try:
            result = _safe_parse_json(response)
            if result:
                analysis_approach = result.get("analysis_approach", "")
                analysis_workflow = result.get("analysis_workflow", "")
                analysis_results = result.get("analysis_results", "")
                figure_descriptions = result.get("figure_descriptions", {})
                parameters_used = result.get("parameters_used", {})
                # Map to backward-compatible schema
                return {
                    "analysis_approach": analysis_approach,
                    "methodology_overview": f"{analysis_approach}\n\n{analysis_workflow}",
                    "workflow_summary": analysis_workflow,
                    "key_results": analysis_results,
                    "figure_descriptions": figure_descriptions if isinstance(figure_descriptions, dict) else {},
                    "parameters_used": parameters_used if isinstance(parameters_used, dict) else {},
                }
        except Exception as e:
            logger.exception("Failed to parse PI comprehensive review response")

        # Fallback: use raw response as methodology_overview
        fallback = self._empty_analysis_summary()
        fallback["methodology_overview"] = response[:3000] if response else ""
        return fallback

    async def _pi_review_analysis(
        self,
        state: "AnalysisState",
        analysis_record: Dict[str, Any],
        findings: Dict[str, Any],
        existing_figures: List[Dict[str, Any]],
        existing_tables: List[Dict[str, Any]],
        language: str
    ) -> Dict[str, Any]:
        """
        PI Agent performs comprehensive review of the bioinformatics analysis.
        Reads the analysis notebook (.ipynb) directly, views figures via vision API,
        and reads table data. Produces a from-scratch summary of analysis approach,
        workflow, and results in a single comprehensive review call.

        Returns analysis_summary dict for downstream report generation.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Step 1: Read notebook directly
        await self._report_progress(17, "Reading analysis notebook...")
        notebook_cells = self._read_notebook(state)

        if not notebook_cells:
            logger.warning("No notebook cells found, returning empty analysis summary")
            return self._empty_analysis_summary()

        logger.info(f"Read {len(notebook_cells)} code cells from notebook")

        # Step 2: Build chronological code digest
        await self._report_progress(19, "Building code digest...")
        code_digest = self._build_notebook_digest(notebook_cells)
        logger.info(f"Built code digest: {len(code_digest)} chars")

        # Step 3: Load figure images for vision API
        await self._report_progress(21, "Loading figure images...")
        figure_images = self._prepare_figure_images(existing_figures)

        # Step 4: Load table summaries
        table_summaries = self._prepare_table_summaries(existing_tables, analysis_state=state)

        # Step 5: Single comprehensive PI review
        await self._report_progress(23, "PI Agent reviewing complete analysis from scratch...")
        analysis_summary = await self._pi_comprehensive_review(
            code_digest, figure_images, table_summaries, state, language
        )

        logger.info(f"Comprehensive analysis review complete: "
                     f"approach={bool(analysis_summary.get('analysis_approach'))}, "
                     f"workflow={bool(analysis_summary.get('workflow_summary'))}, "
                     f"results={bool(analysis_summary.get('key_results'))}")
        return analysis_summary
