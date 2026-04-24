# ScAI User Guide

A step-by-step walkthrough of using ScAI to run a complete multi-omics analysis, from first-run configuration to final report. If you need installation instructions or a high-level overview, see the [README](../README.md).

## 1. Configure API Keys

On first launch, a configuration dialog appears. You need to set up:

- **OpenRouter API Key** (required): Used by all AI agents for analysis. Get one at https://openrouter.ai/keys
- **PubMed Email** (required): Required by NCBI for PubMed literature searches. Use any valid email address.
- **Google AI API Key** (optional): Enables Gemini Deep Research mode for comprehensive literature surveys. Get one at https://aistudio.google.com/app/apikey. When not set, the system uses PubMed + LLM mode instead.
- **Proxy URL** (optional): HTTP/HTTPS proxy if your network requires one for API access.

You can update these settings at any time via the gear icon in the header.

## 2. Create an Analysis Session

Click **"New Analysis"** in the sidebar. Enter a clear description of your research objective. The more specific you are, the better the agents can tailor their analysis. Examples:

> Analyze the liver tissue phenotype of Alb-Cre; Genex-flox/flox knockout mice, focusing on cell type composition changes, fibrosis-related signaling pathways, and immune cell infiltration

> Compare NASH model mouse liver with control, identify key differentially expressed genes and enriched pathways in hepatocytes and macrophages

## 3. Define Treatment Groups

In the **Data** panel, define your experimental groups before uploading data:

- Click **"Add Group"** to create groups (e.g., "Control", "NASH Model", "Treatment A").
- Set the **sample count** for each group (e.g., 3 biological replicates).
- Each uploaded file will be assigned to a specific group and sample number.

This group structure is used by all agents to organize comparisons and ensure figures reference the correct experimental conditions.

## 4. Upload Data

Upload your research data in the **Data** panel. Each file is assigned to a treatment group and sample.

**Pathology Images**
- Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP, SVG.
- Staining types: H&E or custom (specify the stain name).
- Large images are automatically split into tiles for analysis.
- Assign each image to its treatment group and sample number.

**Sequencing Data**
- Primary format: `.h5ad` (AnnData) — recommended for single-cell data.
- Also supports: `.loom` files.
- Sequencing types: scRNA-seq, snRNA-seq, bulk RNA-seq, or other.
- Assign to treatment group and sample number.

**Biochemistry Data**
- **Manual entry**: Enter indicator values directly (e.g., ALT = 45 U/L).
- **File upload**: Upload data tables (CSV/Excel) or images of experimental results.
- Assign to treatment group and sample number.

All uploaded files can be downloaded back via the download buttons in the Data panel.

## 5. Start Analysis

Click **"Start Analysis"** in the Analysis panel. The system runs a **two-phase analysis workflow**.

### Phase 1 — Exploration (Round 1)

Seven steps run sequentially to build a complete, unbiased picture of your data:

1. **Round Initialization** — Pending user inputs are processed, round counter increments, status sets to RUNNING.
2. **PI Planning** — PI Agent generates an unbiased, comprehensive analysis plan with specific tasks for each agent.
3. **Biochemistry Analysis** — Extracts and interprets biochemical indicators from structured data and uploaded images.
4. **Pathology Analysis** — Tiles histology images, runs visual analysis, synthesizes findings cross-referenced with biochemistry results.
5. **Bioinformatics Analysis** — Autonomous Jupyter-based single-cell analysis (up to 200 steps):
   - QC → Normalization → Clustering → Cell Annotation.
   - **Cell Annotation Confirmation** — Analysis pauses here. A dialog shows each cluster's cell type, confidence level (high/medium/low), and key marker genes. Low-confidence clusters require manual confirmation or correction. High-confidence results auto-confirm after a countdown.
   - DEG → GO/KEGG Pathway Enrichment → Cell-Cell Communication.
6. **Literature Research** — Searches and synthesizes scientific literature (Deep Research, PubMed+LLM, or quick search mode).
7. **PI Synthesis** — PI Agent synthesizes findings from all agents and generates a structured **Exploration Report** covering cluster overview, data quality warnings, heterogeneity findings, pathway changes, and research direction candidates. A dialog appears for you to select a direction for Phase 2 (AI-suggested candidates with innovation/novelty/reliability scores, a custom direction, or let PI decide automatically).

### Phase 2 — Innovation (Round 2+)

Following rounds focus on the selected research direction for deeper, hypothesis-driven analysis:

- **Direction Modes**: Single direction, multiple directions (processed sequentially via an automatic queue), or PI-selected best directions.
- **Selective Agent Activation**: PI decides which agents run each round — saving time and API costs. Bioinformatics always runs.
- **Multi-Direction Correlation**: After all direction rounds complete, PI automatically synthesizes findings across directions.
- **Convergence Detection**: Analysis stops automatically after 2 consecutive rounds with no new findings.
- **Feedback Loop**: After each round, provide text guidance, upload new data, or let the countdown auto-continue.

The Bioinformatics Agent runs code in a local Jupyter kernel — you can see the code and outputs in real-time. All figures are automatically saved as both PNG and high-resolution PDF.

## 6. Provide Feedback Between Rounds

After each round, a **Round Feedback Dialog** appears showing:

- Summary of findings from each agent.
- A configurable countdown timer (default: 5 minutes).

You have three options:

- **Provide guidance**: Type instructions to steer the next round (e.g., "Focus on the Wnt signaling pathway" or "Compare macrophage subtypes between groups").
- **Upload additional data**: Add more files targeted to specific agents.
- **Skip / Auto-continue**: Let the countdown expire or click "Continue" to proceed with automatic analysis.

Providing input or uploading files on the final round automatically extends the analysis by one additional round.

## 7. View Results

Switch to the **Results** panel to see:

- **Round-by-round findings** from each agent.
- **Integrated report** summarizing the overall analysis.
- **Publication figures** generated by the Bioinformatics Agent — with PI review status and download buttons (PNG/PDF).
- **Graphic abstract** — an AI-generated visual summary of the research.

## 8. Generate Detailed Report

In the **Report** panel, click **"Generate Report"** to create a comprehensive HTML report:

- PI Agent first reviews all analysis notebooks and figures.
- The system then generates a structured report with 9 chapters covering all analysis aspects.
- Each chapter includes relevant figures, data tables, and narrative text.
- The report appendix documents project settings, agent configurations, uploaded data, and analysis history.

Download the complete report including all figures and tables via the download button.

The 9 chapters cover:

1. Quality Control (QC)
2. Cell Clustering
3. Cell Type Annotation
4. Cell Proportion Analysis
5. Differential Expression Analysis
6. Pathway Enrichment Analysis (GO/KEGG/GSEA)
7. Cell-Cell Communication Analysis
8. Trajectory Analysis
9. Targeted Analysis (custom)

## 9. Chat with PI Agent

The **Chat** panel provides a free-form conversational interface:

- Ask questions about the analysis results (e.g., "What are the most significant findings in the macrophage population?").
- Request additional analysis (e.g., "Generate a volcano plot comparing group A vs group B for hepatocytes").
- Upload images or data files for on-the-fly analysis — you can target files to specific agents.
- PI can display figures from the analysis or generate new ones in real-time.
- For complex questions, PI automatically delegates to specialized agents and synthesizes their responses.

## 10. Customize Agent Prompts

In the **Prompts** panel, you can customize each agent's behavior:

- Modify the system prompt for any agent to adjust its focus.
- Configure which LLM model each agent uses.
- Adjust temperature settings for more creative or deterministic outputs.
- Set literature search parameters (mode, max search rounds).

Common customizations:

- Adjust Pathology Agent's focus for specific organs or staining methods.
- Add domain-specific terminology for Bioinformatics Agent.
- Change Literature Agent search strategy for your research area.

## 11. Session Management

**Pause and Resume**

- Click "Pause" to stop analysis at any time — all state is preserved.
- Click "Resume" to continue from where you left off; the Jupyter environment is fully restored.

**Export and Import**

- **Export**: Click the export button in the sidebar to download your session as a compressed archive. This includes all data, analysis state, figures, and reports.
- **Import**: Click "Import Session" in the sidebar to restore a previously exported session.

**Session Browser**

- All sessions are listed in the sidebar with status indicators.
- Click any session to view or resume it.
- Session name and description can be edited directly in the sidebar.

## Operational Notes

1. Sequencing data analysis may take a while depending on dataset size — please be patient.
2. Deep Research mode may take up to 60 minutes per query and costs ~$2–5 per task.
3. All code executions are automatically saved as Jupyter Notebooks (`.ipynb`).
4. When pausing/resuming analysis, the complete Python environment is automatically restored.
5. Network interruptions are handled gracefully — the interface automatically reconnects and restores message history.

## See Also

- [README](../README.md) — installation, quick start, architecture overview.
- [HANDOVER](../HANDOVER.md) — developer and operator handover (module map, deployment, dev guide).
- [Python client](../clients/python/README.md) — programmatic access via the Python SDK.
