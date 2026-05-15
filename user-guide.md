
# ScAI User Guide

**Version**: 0.24.5 · **Last Updated**: 2026-05-15

> Also available in: [中文 (Chinese)](user-guide-zh.md)

ScAI (Single-Cell Sequencing Analysis Driven by AI) is a multi-agent AI-driven multi-omics biomedical research system. It integrates single-cell sequencing (scRNA-seq / snRNA-seq), bulk RNA-seq, pathology images (H&amp;E / IHC), and biochemical markers into a unified iterative analysis workflow, with eight specialized AI agents collaborating to complete the analysis, and a **Human-in-the-Loop (HITL)** design allowing researchers to participate in decision-making at key nodes.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation &amp; Startup](#2-installation--startup)
3. [First-time Configuration](#3-first-time-configuration)
4. [Interface Overview](#4-interface-overview)
5. [Creating an Analysis Session](#5-creating-an-analysis-session)
6. [Uploading Data](#6-uploading-data)
7. [Configuring Agents](#7-configuring-agents)
8. [Starting Analysis](#8-starting-analysis)
9. [Human-in-the-Loop (HITL)](#9-human-in-the-loop-hitl)
10. [Innovation Phase](#10-innovation-phase)
11. [Viewing Results](#11-viewing-results)
12. [Generating Reports](#12-generating-reports)
13. [Session Management](#13-session-management)
14. [Chat Feature](#14-chat-feature)
15. [User Management](#15-user-management)
16. [FAQ](#16-faq)

---

## 1. Introduction

### 1.1 What is ScAI

ScAI is an AI analysis platform for biomedical researchers. Simply upload your experimental data (sequencing files, pathology images, biochemical markers), and ScAI's eight AI agents will automatically collaborate to complete data analysis and interpretation, ultimately generating a publication-quality HTML report with 9 chapters.

### 1.2 Core Features

- **Eight-Agent Collaboration**: PI (Principal Investigator) Agent formulates analysis plans and synthesizes results; Bioinformatics Agent autonomously executes code analysis in Jupyter environment; Pathology Agent analyzes tissue slice images; Biochemistry Agent interprets biochemical markers; Literature Agent conducts literature research; Painter Agent generates graphical abstracts; Supervisor Agent monitors analysis quality in real-time; Technique Agent provides API documentation lookup.
- **Dual-Phase Workflow**: Unbiased Exploration phase produces comprehensive analysis reports and research directions; Innovation phase deeply tracks selected directions with automatic convergence detection.
- **Human-in-the-Loop**: Intervene at key nodes such as cell type annotation confirmation, research direction selection, round feedback; pause/resume analysis at any time.
- **Multi-Omics Integration**: Automatically cross-reference results from scRNA-seq, snRNA-seq, bulk RNA-seq, pathology images, and biochemical markers.
- **Publication-Ready Output**: PI-reviewed figures (PNG + high-resolution PDF), 9-chapter HTML report, AI-generated graphical abstract.
- **Model-Agnostic**: Supports any OpenAI-compatible API endpoints, including cloud commercial APIs (DeepSeek, Moonshot, Zhipu, MiniMax, Alibaba/Tongyi Qianwen, etc.), OpenRouter aggregation platform, Google Gemini, and open-source models locally deployed via Ollama.

### 1.3 Use Cases

- Animal model phenotype analysis (gene knockout/knockin mice, disease models)
- Tumor vs normal tissue comparative analysis
- Drug toxicology — tissue damage assessment after treatment
- Organ-specific research (liver, heart, kidney, brain, etc.)
- Developmental biology — cross-developmental stage comparison
- Multi-omics integrated disease mechanism research

---

## 2. Installation &amp; Startup

### Method A — Docker (Recommended)

This method requires no manual configuration of Python or Node.js environments and is suitable for most users.

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v ./scai-dep:/dep \
  -v ./scai-data:/data \
  ghcr.io/minzhangcheng/scai:latest
```

Parameter explanation:
- `-p 3000:3000`: Maps container port 3000 to host port 3000.
- `-v ./scai-dep:/dep`: Persists Python and Node.js dependencies (automatically installed on first startup, reused on subsequent startups).
- `-v ./scai-data:/data`: Persists analysis data, uploaded files, reports, and logs.

After startup, open a browser and visit **http://localhost:3000**. Use `docker logs -f scai` to view runtime logs.

To build the image yourself:
```bash
docker build -t ghcr.io/minzhangcheng/scai:latest .
```

### Method B — Startup Script

Suitable for users with Python and Node.js development environments.

```bash
git clone &lt;repository-url&gt;
cd scai
./start.sh
```

Custom host and port:
```bash
./start.sh --host 0.0.0.0 --port 3000 --backport 8000
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Listening address | `127.0.0.1` |
| `--port` | Frontend port | `3000` |
| `--backport` | Backend port | `8000` |

The script automatically creates a virtual environment, installs dependencies, and starts the backend (FastAPI) and frontend (React). Use `./stop.sh` to stop services.

**Environment Requirements**: Python 3.11+ and Node.js LTS.

---

## 3. First-time Configuration

### 3.1 Admin Registration

When accessing ScAI for the first time, the system will prompt you to create an administrator account:

1. On the registration page, fill in **username** (3-64 characters, supporting letters, numbers, underscores, and hyphens), password (at least 8 characters), and email (optional).
2. After clicking register, this user becomes the system administrator.
3. Subsequent new user registrations require administrator approval before use.

### 3.2 API Configuration

After registration, the system will pop up the API configuration panel. This is required for ScAI to call AI models.

**Required**:
- **Default LLM Server**: Fill in the OpenAI-compatible API endpoint address. For example:
  - OpenRouter: `https://openrouter.ai/api/v1`
  - DeepSeek: `https://api.deepseek.com/v1`
  - Zhipu AI: `https://open.bigmodel.cn/api/paas/v4`
  - Alibaba Tongyi Qianwen: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - Local Ollama: `http://localhost:11434/v1`
- **API Key**: API key for the corresponding platform.

**Optional**:
- **PubMed Email**: Identity used by Literature Agent when calling NCBI E-utilities API (NCBI requires an email to increase request frequency limits).
- **Proxy Address**: Configure if network environment requires proxy to access external APIs.
- **Google AI API Key**: Required if you want to use Google Gemini Deep Research for deep literature research.
- **Unsupported Parameter Strategy**: Controls whether to skip or error when a model doesn't support a parameter (like temperature). Default is skip.

### 3.3 Read-Only Mode

If you don't have an API Key yet, you can click "Read-Only Mode" to enter the system. In read-only mode:
- You can browse existing analysis results, reports, and figures.
- You cannot create new analysis sessions or start new analysis tasks.
- You can reconfigure API Key at any time via the "API Settings" button to switch to full mode.

---

## 4. Interface Overview

ScAI's main interface is divided into three areas:

### 4.1 Left Sidebar

Click the menu button (three horizontal lines icon) in the top-left corner to expand/collapse the sidebar. The sidebar contains:
- **New Analysis**: Create a new analysis session.
- **Open Session**: Browse and switch between existing analysis sessions.
- **API Settings**: Reconfigure API Key and server address.
- **User Menu**: View personal information, change password, switch language, etc.
- **Session List**: Shows all saved sessions, click to quickly switch.

### 4.2 Top Tab Bar

The top of the main interface has six tabs corresponding to the six stages of the analysis workflow:

| Tab | Icon | Purpose |
|-----|------|---------|
| **Data** | 📄 | Upload and manage experimental data (sequencing files, pathology images, biochemical markers) |
| **Agents** (Prompts) | ⚙️ | Configure AI models, temperature, and system prompts for each agent |
| **Analysis** | 🧪 | Start analysis, monitor progress, view round summaries, perform HITL interactions |
| **Results** | 📊 | View figures and data files generated from analysis |
| **Report** | 📋 | Generate and download HTML reports |
| **Chat** | 💬 | Free conversation with PI Agent, upload files for interpretation |

### 4.3 Right Message Stream

When an analysis session is selected, a **Message Stream Panel** appears on the right, showing output messages, status updates, and generated figures from each agent in real-time. You can drag the divider on the left side of the panel to adjust the width.

### 4.4 Language Switching

The language selector in the top-right corner supports **Chinese** and **English** switching. The entire interface language takes effect immediately after switching.

---

## 5. Creating an Analysis Session

### 5.1 New Session

1. Click the **"New Analysis"** (+ icon) button in the sidebar.
2. In the pop-up dialog, fill in:
   - **Analysis Name** (optional): Give this analysis a name, e.g., "ZnT1/Ctr1 DKO Mouse Liver Analysis".
   - **Research Goal** (required): Describe your research background, experimental design, and analysis goals in detail. The system pre-fills an example description that you can modify directly.

   &gt; **Tip**: The more detailed your research goal description, the more precise the analysis plan formulated by the PI Agent. It's recommended to include: research background, experimental model, information about control and treatment groups, and specific scientific questions you wish to explore.

3. Click the **"Create"** button.

After creation, the system automatically opens the session, and you can see the session name, current round, and status in the top title bar.

### 5.2 Treatment Groups

In the **Data** tab, you can set treatment groups (i.e., experimental groups):

1. Click **"Add"** in the "Treatment Groups" area.
2. Fill in **group name** (e.g., "WT Control", "DKO"), **description**, and **sample count**.
3. Repeat the above steps to add all experimental groups.

Treatment group information is used by the PI Agent for differential analysis comparisons and result interpretation.

---

## 6. Uploading Data

ScAI supports three main data types, managed in three sections in the **Data** tab.

### 6.1 Pathology Images

Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP, SVG.

1. In the "Pathology Images" area, first select the data type from the dropdown menu (e.g., H&amp;E, IHC).
2. Click the upload area or drag files into it. Large images (over 1024 pixels) are automatically split into small tiles for analysis by the Pathology Agent.
3. After upload is complete, each file can be assigned to a treatment group.

### 6.2 Sequencing Data

Supported formats: `.h5ad` (AnnData), `.loom` (Loom), CSV/TSV (bulk RNA-seq count matrix).

1. In the "Sequencing Data" area, select the data type from the dropdown menu: **scRNA-seq**, **bulk RNA-seq**, **Loom**, or **Other**.
2. Click the upload area or drag files into it.
3. Before uploading, ensure `.h5ad` files contain necessary preprocessing information (e.g., gene expression matrix, cell metadata, etc.).

### 6.3 Biochemical Markers

Biochemical markers can be entered in two ways:

**Method A — Manual Entry**:
1. Click the "Add Marker" button.
2. Fill in marker name, value, unit, and normal reference range (optional).
3. Select the treatment group it belongs to.

**Method B — File Upload**:
1. In the "Biochemical Data" area, select the data type: **CSV/Excel** or **Image** (e.g., Western blot band images, biochemical test report photos, etc.).
2. After uploading the file, the Biochemistry Agent will automatically extract values and markers from it.

### 6.4 File Management

Uploaded files are displayed in a list, where you can:
- Click the filename to **download** the original file.
- Click the trash can icon to **delete** the file.
- View the treatment group assignment of the file.

---

## 7. Configuring Agents

In the **Agents** tab, you can configure AI models and parameters for each agent individually. Each agent in ScAI can use different models and API endpoints, allowing you to freely combine based on task complexity (e.g., flagship model for core reasoning, lightweight model for real-time monitoring).

### 7.1 Agent List

| Agent | Icon | Responsibility | Recommended Model Type |
|-------|------|----------------|-----------------------|
| **PI Agent** | 👤 | Formulate analysis plans, synthesize results from all parties, review figures, drive conversations | Powerful reasoning models (e.g., Claude Opus, GPT-4o) |
| **Bioinformatics Agent** | 💻 | Autonomously execute single-cell / bulk RNA-seq analysis in Jupyter kernel | Powerful code models (e.g., Claude Sonnet, DeepSeek-Coder) |
| **Pathology Agent** | 🔬 | Analyze histology slice images (H&amp;E and special stains) | Vision-capable models |
| **Biochemistry Agent** | 🧪 | Interpret biochemical experiment markers | General reasoning models |
| **Literature Agent** | 📚 | Literature research: supports PubMed search, Deep Research, and Google Gemini Deep Research | Long-context models, or specialized Deep Research models |
| **Painter Agent** | 🎨 | Generate graphical abstracts | Image generation capable models |
| **Supervisor Agent** | 👁️ | Real-time monitoring of bioinformatics analysis execution, escalate issues to PI | Lightweight fast models (e.g., Claude Haiku, Gemini Flash) |
| **Technique Agent** | 🔧 | API documentation lookup and error summarization | Lightweight models |

### 7.2 Configuration Options

Click each agent card to expand detailed settings:

- **Model**: Enter model name or ID. E.g., `claude-opus-4-7`, `deepseek-chat`, `gpt-4o`. Different agents can use different models from the same API endpoint, or models from completely different providers.
- **Temperature**: Controls randomness of output, range 0-2. Lower values (e.g., 0.1-0.3) produce more deterministic and consistent output, suitable for data analysis; higher values (e.g., 0.7-1.0) produce more diverse output, suitable for creative tasks.
- **LLM Server Override**: If an agent needs to use a different API endpoint than the global setting, fill it in here. Leave blank to use the global setting.
- **System Prompt**: Each agent has a default system prompt. Advanced users can modify the prompt to adjust agent behavior.
- **Vision Toggle**: For agents that need to analyze images (Pathology Agent, Literature Agent, etc.), you can enable vision capabilities. Note: After enabling Vision, you must use a multimodal-capable model.

### 7.3 Round Wait Time

At the bottom of the sidebar or in session configuration, you can set **Round Wait Time**. In multi-round analysis, the system waits for a period of time (default 30 seconds) after each Innovation round ends, giving you time to review current results and decide whether to provide feedback. You can set this value to 0 to disable waiting, or set it to a longer time for thorough review.

---

## 8. Starting Analysis

### 8.1 Pre-Start Checklist

Before starting analysis, please confirm:

1. ✅ API Key is configured and valid.
2. ✅ Analysis session is created and research goal is filled in.
3. ✅ At least one type of data is uploaded (pathology images, sequencing data, or biochemical markers).
4. ✅ Each agent's model configuration is correct.
5. ✅ WebSocket connection status in the top-right corner shows green "Connected".

### 8.2 Start Exploration Phase

1. Switch to the **Analysis** tab.
2. Click the **"Start Analysis"** button.

The system will start the **Exploration phase**:
- PI Agent first analyzes all data you uploaded and formulates a comprehensive analysis plan.
- Each specialized agent (Pathology, Bioinformatics, Biochemistry, Literature) executes their respective analysis tasks in parallel.
- The right message stream panel shows each agent's progress in real-time: code execution, figure generation, analysis findings, etc.
- If the Bioinformatics Agent executes code through the Jupyter kernel, you can see code output and generated figures in the message stream.

### 8.3 Real-time Monitoring

During analysis runtime:

- **Message Stream Panel**: Shows all agent messages in chronological order, different message types are marked with different colors (agent messages, system status, errors, etc.).
- **Analysis Panel**: Top shows current status ("Analysis in progress"), and completed rounds and finding count.
- **WebSocket Status**: Connection indicator in the title bar should remain green. If disconnected and turns yellow, the system will automatically attempt to reconnect.

### 8.4 Pause and Stop

- **Pause**: Click the pause button, the system will pause after completing the current step. Suitable for when you need to review results mid-way. After pausing, you can click "Continue" to resume analysis.
- **Stop**: Click the square stop button, a confirmation dialog will pop up. After stopping, analysis terminates, but generated results and figures are retained.

---

## 9. Human-in-the-Loop (HITL)

ScAI designs three key human-in-the-loop interaction nodes in the Exploration phase, allowing you to intervene at important decision points.

### 9.1 Cell Type Annotation Confirmation

After the Bioinformatics Agent performs clustering analysis on single-cell data, the Literature Agent conducts literature search to infer cell types. The system will pop up an **Annotation Confirmation Dialog**, containing:

- **Cluster List**: Each cluster and its corresponding cell type inference result.
- **Confidence**: PI Agent's confidence score for each annotation.
- **Candidate Cell Types**: For uncertain annotations, you can expand to view alternative cell types.
- **Actions**: For each cluster, you can **Confirm**, **Modify** (select alternative type or custom input), or **Mark as Unknown**.

&gt; **Important**: Please carefully review annotation results. Correct cell type identification is the foundation for all subsequent analysis. If you find incorrect annotations, be sure to modify them.

### 9.2 Exploration Report Review

After the Exploration phase is complete, the PI Agent generates an **Exploration Report**, summarizing:
- Data overview and quality assessment.
- Key findings from each agent.
- Proposed research directions.

You need to review this report and confirm if the analysis direction is reasonable. If you need supplementary analysis, you can submit **Supplementary Analysis Requests** at this stage.

### 9.3 Research Direction Selection

After the Exploration Report is generated, the system will pop up a **Direction Selection Dialog**, listing 2-5 research directions proposed by the PI Agent based on analysis findings. For each direction, the PI Agent will explain:

- Research hypothesis and innovation.
- Recommended analysis methods.
- Expected biological findings.

You need to **select 1-3 directions** to enter the Innovation phase for deep analysis. You can also choose to let the PI Agent decide directions autonomously.

### 9.4 Human Guidance

At any time during analysis runtime, you can input instructions via the **Human Guidance** feature:

1. Click the "Human Guidance" button in the analysis panel.
2. Input your guidance content in the pop-up dialog. For example:
   - "Please focus on the interaction between Kupffer cells and hepatic stellate cells."
   - "I'm particularly interested in iron metabolism-related genes, please emphasize this aspect in the analysis."
3. You can also upload additional data files along with your guidance, and the system will incorporate the new data into subsequent analysis.

---

## 10. Innovation Phase

### 10.1 Multi-Direction Analysis

After selecting research directions in the Exploration phase, the Innovation phase begins. Each selected direction starts an independent analysis round:

- Each direction has a specialized analysis plan formulated by the PI Agent.
- Bioinformatics Agent, Literature Agent, and other specialized agents perform deep analysis targeting that direction.
- Analysis results from multiple directions undergo **cross-correlation analysis** to discover complementarity and potential connections between directions.

### 10.2 Round Iteration &amp; Convergence

The Innovation phase continues for multiple rounds, after each round:

1. PI Agent synthesizes all results from the round and assesses whether the predetermined goals for that direction have been achieved.
2. The system judges whether to continue through **convergence detection**. If new findings are saturated (no major new findings for 2 consecutive rounds), analysis for that direction automatically ends.
3. If round wait time is set, the system pauses between rounds, giving you time to review results and provide feedback.

### 10.3 Supplementary Analysis

During the Innovation phase, you can request supplementary analysis at any time:

1. Click the "Supplementary Analysis" button in the analysis panel.
2. Input the additional analysis content or specific questions you want.
3. PI Agent will integrate the supplementary analysis into the current analysis workflow.

---

## 11. Viewing Results

### 11.1 Results Panel

Switch to the **Results** tab to view all files generated during analysis:

- **Figures**: All figures generated by Bioinformatics Agent and Pathology Agent (PNG format), and high-resolution PDF versions of each figure.
- **Data Tables**: Differentially expressed gene lists, pathway enrichment results, cell proportion matrices, etc. (CSV format).
- **Code Records**: Jupyter Notebooks (`.ipynb` format) executed by Bioinformatics Agent, for analysis reproduction.

### 11.2 Figure Gallery

Figures are displayed in a grid, where you can:
- Click a figure for **full-screen preview**.
- Click the download button to **download** PNG or PDF versions.
- View figure **metadata** (generation time,所属 analysis step, etc.).

### 11.3 Round Summaries

In the **Analysis** tab, you can expand summary cards for each completed round to view:
- List of agent tasks executed in that round.
- Key findings from each agent (markdown format text summary).
- Finding count statistics.

---

## 12. Generating Reports

### 12.1 Report Generation

When analysis is complete (all selected directions have converged or you manually stopped), you can generate the final report:

1. Switch to the **Report** tab.
2. Click the **"Generate Report"** button.
3. The system starts calling the report generation workflow, and you can see generation progress in the right message stream panel.

Report generation includes the following steps:
- **Narrative Construction**: Supervisor Agent extracts key steps and findings from analysis logs to construct an analysis narrative.
- **Figure Selection**: Automatically selects the most representative figures based on quality scores.
- **PI Review**: PI Agent reviews report content and figure quality.
- **HTML Rendering**: Generates interactive HTML report.
- **Graphical Abstract Generation**: Painter Agent creates a summary graphical abstract.

### 12.2 Report Structure

The generated HTML report contains 9 chapters:

1. **Cover**: Project title, date, ScAI version.
2. **Project Info**: Research goal, data overview, analysis method summary.
3. **Abstract**: Concise summary of main findings.
4. **Introduction**: Research background and scientific questions.
5. **Methods**: Detailed data processing and analysis methods (including software packages and parameters).
6. **Results**: Results presentation organized by agent and analysis direction, including figures and data tables.
7. **Discussion**: Cross-agent comprehensive analysis interpretation.
8. **Conclusion**: Key findings summary and future research direction suggestions.
9. **Appendix**: Reproducibility information (Python version, dependency package list, file checksums, executable code commands).

### 12.3 Report Operations

- **Online Preview**: After generation is complete, the report is displayed embedded directly in the report panel.
- **Download HTML**: Click the download button to get the complete HTML file, viewable offline in any browser.
- **Regenerate**: If not satisfied with the report, you can modify prompts or configuration and click "Regenerate".
- **Download Reproducibility Package**: Independently download the reproducibility appendix containing environment information, dependency snapshots, and statistics.

---

## 13. Session Management

### 13.1 Session Switching

In the sidebar session list, click any session to switch. The current session automatically saves state.

### 13.2 Snapshot Management

A snapshot is a complete backup of analysis state, containing:
- Current analysis state (rounds, findings, figures, etc.).
- Complete state of Jupyter kernel (variables, data objects, etc.).

**Create Snapshot**:
1. Click the camera icon ("Create Snapshot") in the analysis panel.
2. Input snapshot name and description.
3. Click confirm.

**Restore Snapshot**:
1. Open the Snapshot Panel.
2. Select the snapshot to restore from the snapshot list.
3. Click "Restore". The system will roll back analysis state and Jupyter kernel to the snapshot state.

**Snapshot Tree**: Multiple snapshot and restore operations form a tree structure, and you can visually view and select different analysis path branches in the snapshot panel.

&gt; **Tip**: Creating snapshots at key analysis nodes (after annotation confirmation, before direction selection) is a good practice. This way, if subsequent analysis directions are not ideal, you can quickly return to key nodes to reselect.

### 13.3 Export Session

The export function packages the entire session into an archive file:

1. Select "Export Session" in the sidebar.
2. Select export format: **tar.gz** or **zip**.
3. Wait for background packaging to complete (large files may take several minutes).
4. Browser download is automatically triggered after packaging is complete.

Export content includes: all uploaded raw data, analysis state, generated figures and reports, Jupyter Notebooks, reproducibility information.

### 13.4 Import Session

1. Select "Import Session" in the sidebar.
2. Select a previously exported `.tar.gz`, `.zip`, or `.json` file.
3. Wait for import to complete, the session will appear in the session list.

Imported sessions retain complete analysis state, and you can continue analysis or regenerate reports on top of them.

---

## 14. Chat Feature

### 14.1 PI Chat

Switch to the **Chat** tab, and you can have free conversations with the PI Agent:

- Directly input questions, PI Agent will answer based on the current analysis session context.
- Chat history is retained, PI Agent will remember previous conversation content (up to 20 Q&amp;A pairs retained).
- Markdown format replies are supported, including tables, code blocks, etc.

**Typical Uses**:
- "Please explain the biological significance of the differentially expressed genes discovered by the Bioinformatics Agent in round 3."
- "Help me design a follow-up analysis scheme targeting the liver fibrosis pathway."
- "Summarize the main findings of all analyses so far."

### 14.2 Upload Files in Chat

In the chat input box, you can upload files (pathology images, sequencing files, etc.) via the attachment button. PI Agent will analyze file content and integrate it into the answer. This is different from formal analysis sessions — it's one-time interpretation and Q&amp;A.

### 14.3 Agent Delegation

In chat, PI Agent can **delegate** tasks to other specialized agents:

- When your question involves literature research, PI Agent may delegate to Literature Agent.
- When involving bioinformatics analysis questions, PI Agent may start Bioinformatics Agent to execute code in Jupyter kernel.
- Delegation process and results are displayed in real-time in the chat window.

---

## 15. User Management

### 15.1 Registration &amp; Login

- **Registration**: Click "Register" on the login page, fill in username, password, and email (optional) to submit a registration application.
- **Login**: Log in using username and password.
- **User Status**:
  - `pending`: Default status for newly registered users, requires administrator approval.
  - `active`: Approved, can normally use all features.
  - `readonly`: Can only browse existing analysis results, cannot create new analysis or modify data.

### 15.2 Administrator Features

Administrators (first registered user is admin by default) can access the **Admin Panel**:

- **User Approval**: View pending user list, approve or reject registration applications.
- **User Management**: View all users, modify user permission levels.
- **Cascading Cleanup**: Delete users and all their associated sessions and data.

The admin panel is accessed via URL path `/admin`, or by clicking the "Admin" option in the user menu.

### 15.3 Personal Information Modification

Through the user menu (top-right corner avatar icon) you can:
- View and modify personal information.
- Change password.
- Change email address.
- Switch to other sessions.

---

## 16. FAQ

### 16.1 API Configuration Related

**Q: How to get an API Key?**

API Key acquisition methods for major AI platforms:
- **OpenRouter**: Register an openrouter.ai account, generate a Key in the settings page. OpenRouter can aggregate calls to multiple models, suitable for users wanting to try different models.
- **DeepSeek**: Register at platform.deepseek.com, create in API Keys page.
- **Zhipu AI**: Register at open.bigmodel.cn, create in API Keys page.
- **Alibaba Tongyi Qianwen**: Create in Alibaba Cloud DashScope console.
- **Local Ollama**: No API Key needed, model address fill in `http://localhost:11434/v1`.

**Q: What to do if getting "API Key verification failed" error?**

1. Confirm API Key is not truncated or contains extra spaces.
2. Confirm LLM server address is correct (note no extra slashes at the end).
3. Confirm server is accessible (network firewall/proxy settings).
4. Try clicking "Reset API Key" and re-entering.

### 16.2 Data Format Related

**Q: Error uploading .h5ad file?**

1. Confirm file is valid AnnData format (can verify with `anndata.read_h5ad()`).
2. Confirm file does not contain non-serializable custom objects.
3. For very large files (&gt; 5GB), it's recommended to first preprocess (filter low-expression genes, subsample, etc.) to reduce file size.

**Q: What stains are supported for pathology images?**

Pathology Agent does not rely on hard-coded stain type recognition, but directly analyzes image content through vision AI. Supports H&amp;E, IHC, Masson trichrome, PAS, and other common stains, as well as custom stains. You can select or customize stain types in the data type dropdown menu.

**Q: What format is needed for biochemical marker images?**

Supports common image formats like PNG, JPEG. Biochemical experiment report photos, Western blot band images, ELISA plate reader result images can all be directly uploaded, and the Biochemistry Agent will use vision AI to automatically recognize and extract information.

### 16.3 Analysis Process Related

**Q: How long does analysis take?**

Analysis time depends on data volume and complexity. A single Exploration phase usually takes 10-30 minutes. Each Innovation round takes 5-15 minutes. The system has a 1.5-hour timeout protection mechanism.

**Q: Can I close the browser during analysis?**

Yes. Analysis runs on the server side, closing the browser does not affect the analysis process. After reopening, the system will automatically reconnect WebSocket and restore the real-time message stream. Note: Sessions without activity for a long time (over 1 hour) may be automatically unloaded to disk by the server, but data will not be lost.

**Q: What to do if Bioinformatics Agent reports an error?**

Bioinformatics Agent has built-in automatic error repair mechanism (up to 3 retries). Common errors include:
- Package version compatibility issues: Agent will automatically try different APIs.
- Insufficient memory: Agent will adjust analysis strategy (e.g., subsampling).
- 3 consecutive same-type errors will trigger Supervisor intervention and may escalate to PI Agent to adjust analysis strategy.

If an error causes analysis to interrupt, you can restore from the most recent snapshot after fixing the problem.

**Q: How to judge if analysis is "good enough"?**

The system automatically detects convergence — if no major new findings for two consecutive rounds, it will recommend ending analysis. Additionally, you can judge by the following indicators:
- Whether the exploration report comprehensively covers your scientific questions.
- Whether key pathways have been fully analyzed.
- Whether PI Agent's comprehensive summary gives clear biological conclusions.

### 16.4 Model Configuration Recommendations

**Q: How to choose models for each agent?**

A typical cost-effective configuration scheme:

| Agent | Recommended Model | Notes |
|-------|-------------------|-------|
| PI Agent | claude-opus-4-7 or gpt-4o | Core reasoning, needs strongest model |
| Bioinformatics Agent | claude-sonnet-4-6 or deepseek-chat | Code generation, needs strong code capability |
| Pathology Agent | claude-sonnet-4-6 (Vision) | Image analysis, needs multimodal |
| Biochemistry Agent | deepseek-chat | Numerical interpretation, medium model sufficient |
| Literature Agent | Google Gemini 2.5 Pro or deepseek-chat | Long-context processing |
| Supervisor | claude-haiku-4-5 or gemini-flash | Fast monitoring, lightweight model sufficient |
| Technique | claude-haiku-4-5 | API documentation lookup, lightweight model sufficient |
| Painter | Image generation capable model | Needs endpoint supporting `/v1/images/generations` |

**Q: Can I use locally deployed open-source models?**

Yes. After deploying open-source models via Ollama, set LLM server address to `http://localhost:11434/v1`, and use the model name from Ollama (e.g., `qwen3`, `llama4`). Note: Open-source model reasoning capability may not be as good as commercial models, it's recommended at least PI Agent uses a stronger commercial model.

### 16.5 Other Questions

**Q: How to reproduce analysis results?**

ScAI provides multi-layer reproduction guarantees:
1. Exported session contains complete analysis state and code records.
2. Report includes reproducibility appendix, listing Python version, dependency packages and versions, file checksums.
3. All code from Bioinformatics Agent is saved in Jupyter Notebooks and can be run independently.

**Q: Can multiple users use the same ScAI instance simultaneously?**

Yes. Each user has their own account and sessions, no interference. Administrators manage all users through the admin panel.

**Q: Is data secure?**

- All data is stored locally on the server (`/data` directory).
- Uploaded files do not leave your server (except analysis request content sent to LLM API).
- API Key is encrypted and stored on the server side, not exposed to frontend.
- If you have strict data security requirements, it's recommended to use local Ollama deployed open-source models, completely independent of external APIs.

---

&gt; **Tip**: This manual updates with ScAI versions. If manual content is found to be inconsistent with the actual interface, please refer to the latest version. For issues not covered, welcome to submit an Issue.

---

**ScAI · Single Cell Sequencing Analysis Driven by AI · © 2025–2026 Minzhang Cheng**

