# ScAI - AI-Driven Biomedical Analysis

ScAI is a general-purpose, multi-agent platform for biomedical analysis. It coordinates sequencing analysis, pathology imaging, biochemical indicators, literature research, and human review in one iterative workflow. Models can be configured per agent through OpenAI-compatible endpoints, Google APIs, or local deployments.

**Version**: 0.38.1 | **License**: Apache 2.0

## Core Capabilities

- Multi-agent planning, execution, synthesis, and supervision.
- scRNA-seq, snRNA-seq, bulk RNA-seq, pathology images, and biochemical data.
- Human-in-the-loop plan review, annotation confirmation, direction selection, and round guidance.
- Local Jupyter execution with publication-ready figures and integrated reports.
- Persistent sessions, portable export/import, browser UI, and Python client.

## Workflow

ScAI begins with broad exploration and annotation, continues with approved downstream analysis directions, integrates evidence across rounds and modalities, and produces final reports. Users remain in control at review gates and can pause, resume, or redirect the analysis.

## Docker Quick Start

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v ./scai-dep:/dep \
  -v ./scai-data:/data \
  ghcr.io/minzhangcheng/scai:latest
```

Open [http://localhost:3000](http://localhost:3000). The `/dep` volume persists dependencies and `/data` persists application data. See the [Administrator Manual](docs/admin-manual.md) for deployment options.

## Direct Run

```bash
git clone https://github.com/minzhangcheng/ScAI.git scai
cd scai
./start.sh --prod
```

Requirements: the application supports Python 3.11+, but the current startup script specifically invokes `python3.12` when creating its virtual environment; install Python 3.12 and Node.js LTS for this path. The script installs dependencies and starts the frontend and backend. Stop with `./stop.sh`; use `./start.sh --dev` for development.

## Minimal Usage

1. Open the web UI and complete administrator setup.
2. Configure LLM credentials and models.
3. Create a session, define groups, and upload data.
4. Start analysis and respond to review gates.
5. Inspect results, generate reports, or export the session.

See the [User Guide](docs/user-guide.md) or [中文用户指南](docs/user-guide-cn.md) for the complete walkthrough.

## Python Client

```bash
pip install -e clients/python
```

```python
from scai_client import Client

client = Client(base_url="http://localhost:3000", api_key="scai_your_key_here")
job = client.submit("analysis-config.json")
job.wait()
```

See the [Python client documentation](clients/python/README.md) for authentication, sessions, uploads, analysis, and export/import.

## Documentation

- [User Guide](docs/user-guide.md)
- [中文用户指南](docs/user-guide-cn.md)
- [Administrator Manual](docs/admin-manual.md)
- [管理员手册](docs/admin-manual-zh.md)
- [Technical Handover](docs/HANDOVER.md)
- [Python Client](clients/python/README.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)

## Acknowledgements

The methodological rule corpus in `backend/data/method_rules.yaml` -- the
statistical rules and literature citations ScAI applies to enrichment,
differential-expression and clustering analyses -- is distilled from
[bioSkills](https://github.com/GPTomics/bioSkills) (MIT licence, Copyright (c)
2026 GPTomics). bioSkills targets an R/Bioconductor stack; each rule was
re-expressed for ScAI's Python sandbox, so the Python translations, rule
identifiers and enforcement mechanisms are ScAI's own. No bioSkills source file
is vendored here.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full MIT licence
text, the exact commit referenced, and the external data resources ScAI fetches
at runtime.

## License

Apache License 2.0

Portions of `backend/data/method_rules.yaml` are derived from MIT-licensed
material; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
