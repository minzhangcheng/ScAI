# scai-client

Python SDK for the ScAI batch analysis API. Submit single-cell analyses from scripts.

## Install

```bash
pip install scai-client
```

## Quick Start

```python
from scai_client import Client

# Authenticate with an API key (create one at http://localhost:8000/api-keys)
client = Client(base_url="http://localhost:8000", api_key="scai_your_key_here")

# Submit an analysis from a config file
job = client.submit("my-config.json")

# Wait for completion (polls every 60s)
job.wait(timeout=3600)

# Download results
job.download_export("results.zip")
```

## Config File

The config file follows the ConfigV1 schema. Export one from the web UI to use as a template.

## Known Limitations (v0.1.0)

- No automatic resume after interrupted uploads — you must re-submit
- SHA256 verification is performed server-side at upload completion

## License

MIT
