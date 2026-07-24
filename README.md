# ScAI — Single-Cell Sequencing Analysis Driven by AI

**ScAI** is a multi-agent AI system for multi-omics biomedical research. It integrates single-cell sequencing (scRNA-seq / snRNA-seq), bulk RNA-seq, pathology image analysis (H&E / IHC), and biochemical indicators into a unified, iterative workflow. Powered by specialized agents and a **human-in-the-loop** design, ScAI is **provider-agnostic**: any OpenAI-compatible endpoint works, with per-agent routing so you can mix flagship and open-source models to balance quality and cost. Verified endpoints include cloud aggregators (OpenRouter, n1n.ai), direct provider APIs (DeepSeek, Moonshot, Zhipu, MiniMax, Aliyun/Qwen), Google Gemini (for Deep Research literature mode), and **locally deployed open-source models via Ollama**.

**Version**: 0.35.3 · **License**: Apache 2.0

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

Common Docker environment flags (pass with `-e`):

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v ./scai-dep:/dep \
  -v ./scai-data:/data \
  -e PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  -e SCAI_ENABLE_SNAPSHOT_V2=true \
  ghcr.io/minzhangcheng/scai:latest
```

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

| Option       | Description                | Default     |
|--------------|----------------------------|-------------|
| `--host`     | Listen address             | `127.0.0.1` |
| `--port`     | Frontend port              | `3000`      |
| `--backport` | Backend port               | `8000`      |

The script creates a virtual environment, installs dependencies, and starts both backend and frontend. Stop with `./stop.sh`.

Common script environment variables:

```bash
SCAI_ENABLE_SNAPSHOT_V2=true SCAI_DATA_DIR=/custom/data ./start.sh
```

> Full environment variable reference is below. First-run configuration (API keys, PubMed email, optional proxy) is performed in the UI. See the [User Guide](user-guide.md) for full details.

## Environment Variables

| Variable | Default | Applies to | Purpose |
|----------|---------|------------|---------|
| `SCAI_DATA_DIR` | `../data` (relative to backend) | `start.sh` / backend | Root for session data, uploads, logs. Docker always uses `/data` (volume-mounted). |
| `SCAI_JWT_SECRET` | auto-generated | Docker / `start.sh` | JWT signing secret; set explicitly for multi-instance deployments |
| `SCAI_CONFIG_SECRET` | auto-generated | Docker / `start.sh` | Encryption key for per-user LLM config secrets in SQLite |
| `SCAI_ENABLE_REPRODUCIBILITY` | disabled | Docker / `start.sh` | Enable reproducibility bundle generation (`1`/`true`/`yes`) |
| `SCAI_ENABLE_SNAPSHOT_V2` | disabled | Docker / `start.sh` | Enable Snapshot V2 checkpoint system (`1`/`true`/`yes`) |
| `SCAI_MODE` | `prod` | Docker only | Runtime mode: `prod` or `dev` |
| `PIP_INDEX_URL` | unset | Docker only | Custom PyPI mirror for `pip install` (e.g. `https://mirrors.aliyun.com/pypi/simple/`) |
| `SCAI_AUTO_FETCH_MARKERS` | `0` | `start.sh` only | Auto-fetch marker gene databases on first startup (`1` to enable) |
| `VITE_API_BASE_URL` | `/api` | Frontend build | API base URL override (Vite build-time) |
| `VITE_WS_BASE_URL` | `ws://localhost:8000/ws` | Frontend build | WebSocket URL override (Vite build-time) |

> `SCAI_HOST` and `SCAI_BACKPORT` are internal variables set automatically by the startup scripts for the Vite proxy; manual setting is generally unnecessary.

## Core Features

- **Eight-agent collaboration** — a PI Agent plans and synthesizes; specialist agents run bioinformatics, pathology, biochemistry, literature, and graphic-abstract tasks; a Supervisor monitors execution.
- **Two-phase workflow** — an unbiased Exploration round produces a structured report and proposed research directions; Innovation rounds pursue selected directions, with automatic convergence detection.
- **Human-in-the-loop** — pause/resume any time; confirm cell-type annotations; pick research directions; give text guidance or upload new data between rounds.
- **Multi-omics integration** — scRNA-seq, snRNA-seq, bulk RNA-seq, pathology images, and biochemistry values are cross-referenced automatically.
- **Provider-agnostic LLM layer** — each agent independently selects its model and endpoint; any OpenAI-compatible API works (cloud or self-hosted), including **local Ollama** deployments of open-source models for fully offline use.
- **Publication-ready output** — PI-reviewed figures (PNG + high-res PDF), a 9-chapter HTML report with appendix, and an AI-generated graphic abstract.
- **Session lifecycle** — pause, resume, export, and import complete sessions (data, state, notebooks, figures, reports), with an optional Snapshot V2 checkpoint subsystem.

### Snapshot V2 safety contract

Snapshot V2 is **disabled by default**. To opt in, set the environment variable before backend startup and restart ScAI:

```bash
SCAI_ENABLE_SNAPSHOT_V2=true ./start.sh
```

For Docker, add `-e SCAI_ENABLE_SNAPSHOT_V2=true` to `docker run`. Accepted truthy values are `1`, `true`, and `yes` (case-insensitive). Disabling Snapshot V2 does not delete existing snapshot data; setting the variable and restarting restores access to it.

When disabled, ScAI does not create automatic workflow, report, or chat snapshots; Snapshot V2 management, restore, import, export, and failure-action APIs return `SNAPSHOT_V2_DISABLED`; and the frontend hides Snapshot V2 controls. When explicitly enabled, Snapshot V2 is the authoritative checkpoint lifecycle and the following safety contracts apply:

| Topic | Behavior |
|-------|----------|
| Fail-closed large runtime | Required system snapshots commit only with complete, verified runtime. Oversized or failed runtime capture does **not** publish a state-only partial snapshot. Workflow-boundary, post-report, and post-chat failures may be skipped with reduced restorability; pre-restore remains fail-closed with no Skip. |
| V1 restore cutover | Legacy V1 restore always returns **HTTP 410**. V1 inventory, export, and delete remain available as non-restorable management only. |
| Runtime unavailable (`reset` / `keep`) | Restore of a snapshot without usable runtime requires an explicit choice: **`reset`** restores business state and workspace, quarantines live runtime, and blocks resume until a full re-run from original inputs; **`keep`** retains the current live runtime and records a **persistent** risk acceptance (not an exact restore). |
| Import `trust_runtime=false` (default) | Imported archives never materialize dill/runtime objects unless trust is explicitly enabled. Imported snapshots report unavailable runtime and require `reset`/`keep` for continuation. |
| Session-local layout | Paths follow `backend/services/snapshot_v2/paths.py`: `sessions/{id}/v2`, `objects/`, `staging/`, `journals/`, `committed/`, and `integrity/`. Layout is not migrated in this release. |
| AnnData alias residual | Snapshot V2 preserves supported top-level values and workflow continuation, but it does not guarantee Python object alias identity when the same AnnData object is also embedded inside an arbitrary nested container. Code must not rely on identity checks such as `nested["adata"] is adata` across restore. |

Release evidence for design acceptance criteria: [docs/verification/snapshot-v2-acceptance.md](docs/verification/snapshot-v2-acceptance.md).

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

- **[User Guide](user-guide.md)** — step-by-step walkthrough of a full analysis.
- **[HANDOVER](HANDOVER.md)** — architecture snapshot, module map, deployment, and developer guide.
- **[Python client](clients/python/README.md)** — programmatic access to ScAI sessions.

## License

Apache License 2.0
