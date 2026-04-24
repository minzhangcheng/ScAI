# ScAI — Single-Cell Sequencing Analysis Driven by AI

**ScAI** is a multi-agent AI system for multi-omics biomedical research. It integrates single-cell sequencing (scRNA-seq / snRNA-seq), bulk RNA-seq, pathology image analysis (H&E / IHC), and biochemical indicators into a unified, iterative workflow. Powered by specialized agents and a **human-in-the-loop** design, ScAI is **provider-agnostic**: any OpenAI-compatible endpoint works, with per-agent routing so you can mix flagship and open-source models to balance quality and cost. Verified endpoints include cloud aggregators (OpenRouter, n1n.ai), direct provider APIs (DeepSeek, Moonshot, Zhipu, MiniMax, Aliyun/Qwen), Google Gemini (for Deep Research literature mode), and **locally deployed open-source models via Ollama**.

**Version**: 0.20.9 · **License**: Apache 2.0

## Quick Start

### Option A — Docker (recommended)

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v ./scai-dep:/dep \
  -v ./scai-data:/data \
  ghcr.io/minzhangcheng/scai:latest
```

- `/dep` — Python venv and Node dependencies (installed on first run, persisted).
- `/data` — session data, uploads, reports, logs.

Open **http://localhost:3000**. View logs with `docker logs -f scai`.

To build locally: `docker build -t ghcr.io/minzhangcheng/scai:latest .`

### Option B — Startup script

```bash
git clone <repository-url>
cd scai
./start.sh
```

Customize host/ports:

```bash
./start.sh --host 0.0.0.0 --port 3000 --backport 8000
```

| Option       | Description      | Default     |
|--------------|------------------|-------------|
| `--host`     | Listen address   | `127.0.0.1` |
| `--port`     | Frontend port    | `3000`      |
| `--backport` | Backend port     | `8000`      |

The script creates a virtual environment, installs dependencies, and starts both backend and frontend. Stop with `./stop.sh`.

> First-run configuration (API keys, PubMed email, optional proxy) is performed in the UI. See the [User Guide](docs/user-guide.md) for full details.

## Core Features

- **Eight-agent collaboration** — a PI Agent plans and synthesizes; specialist agents run bioinformatics, pathology, biochemistry, literature, and graphic-abstract tasks; a Supervisor monitors execution.
- **Two-phase workflow** — an unbiased Exploration round produces a structured report and proposed research directions; Innovation rounds pursue selected directions, with automatic convergence detection.
- **Human-in-the-loop** — pause/resume any time; confirm cell-type annotations; pick research directions; give text guidance or upload new data between rounds.
- **Multi-omics integration** — scRNA-seq, snRNA-seq, bulk RNA-seq, pathology images, and biochemistry values are cross-referenced automatically.
- **Provider-agnostic LLM layer** — each agent independently selects its model and endpoint; any OpenAI-compatible API works (cloud or self-hosted), including **local Ollama** deployments of open-source models for fully offline use.
- **Publication-ready output** — PI-reviewed figures (PNG + high-res PDF), a 9-chapter HTML report with appendix, and an AI-generated graphic abstract.
- **Session lifecycle** — pause, resume, snapshot, export, and import complete sessions (data, state, notebooks, figures, reports).

## Agents

| Agent           | Role                                                                                          |
|-----------------|-----------------------------------------------------------------------------------------------|
| PI              | Plans analysis, synthesizes findings across agents, reviews figures, drives chat.             |
| Bioinformatics  | Autonomous scRNA/snRNA/bulk RNA-seq analysis in a local Jupyter kernel (up to 200 steps).     |
| Pathology       | Tiles and analyzes histology images (H&E and custom stains).                                  |
| Biochemistry    | Extracts and interprets biochemical indicators from tables and images.                        |
| Literature      | Literature synthesis via Deep Research models, Gemini Deep Research, or PubMed + LLM.         |
| Painter         | Generates graphic abstracts summarizing results.                                              |
| Supervisor      | Real-time oversight of bioinformatics execution; escalates issues to PI.                      |
| Technique       | API documentation lookup and error summarization for other agents.                            |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Web Frontend (React / Vite)                 │
│   Data · Prompts · Analysis · Results · Report · Chat        │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST + WebSocket
┌───────────────────────────▼──────────────────────────────────┐
│                   Backend (FastAPI / Uvicorn)                │
│   Routes · Auth · Session Store · Workflow Engine            │
│                                                              │
│   ┌────┐  ┌────────┐  ┌─────────┐  ┌─────────┐               │
│   │ PI │  │Bioinfo │  │Pathology│  │Biochem  │               │
│   └──▲─┘  └───┬────┘  └─────────┘  └─────────┘               │
│      │       │                                               │
│   ┌──┴──┐ ┌──▼───────┐ ┌───────────┐ ┌──────────┐            │
│   │Super│ │ Painter  │ │Literature │ │Technique │            │
│   └─────┘ └──────────┘ └───────────┘ └──────────┘            │
├──────────────────────────────────────────────────────────────┤
│  LLM endpoints (OpenAI-compatible: cloud or local Ollama)    │
│  · Jupyter kernel · Disk · SQLite (users)                    │
│  · Google APIs (optional)                                    │
└──────────────────────────────────────────────────────────────┘
```

For module-level detail (routes, services, agents, state, deployment, dev workflow), see [HANDOVER.md](HANDOVER.md).

## Supported Data Formats

| Data type         | Formats                                 | Notes                                   |
|-------------------|-----------------------------------------|-----------------------------------------|
| Pathology images  | PNG, JPEG, GIF, BMP, TIFF, WebP, SVG    | Large images auto-tiled at 1024 px      |
| Sequencing data   | `.h5ad` (AnnData), `.loom`              | scRNA-seq, snRNA-seq, bulk RNA-seq      |
| Biochemistry data | Manual entry, CSV / Excel, images       | Values with units and reference ranges  |
| Session archives  | `.tar.gz`, `.zip`, `.json`              | For export / import                     |

## Use Cases

- Animal model phenotyping (KO / KI mice, disease models).
- Tumor vs. normal-tissue comparative analysis.
- Drug toxicology — tissue damage assessment after treatment.
- Organ-specific research (liver, heart, kidney, brain, …).
- Developmental biology — comparisons across stages.
- Disease mechanism research with multi-omics integration.

## Documentation

- **[User Guide](docs/user-guide.md)** — step-by-step walkthrough of a full analysis.
- **[HANDOVER](HANDOVER.md)** — architecture snapshot, module map, deployment, and developer guide.
- **[Python client](clients/python/README.md)** — programmatic access to ScAI sessions.

## License

Apache License 2.0
