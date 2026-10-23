# ScAI Developer Handover

This guide describes the current architecture and the boundaries that matter when
maintaining ScAI. Release history is in [CHANGELOG.md](CHANGELOG.md). For setup,
usage, and operations, see the [README](../README.md), [User Guide](user-guide.md),
and [Administrator Manual](admin-manual.md).

**Version**: 0.64.8 · **Last updated**: 2026-10-23

## Architecture and source map

ScAI combines a React/Vite frontend, a FastAPI backend, LangGraph orchestration,
and a local Jupyter kernel for each analysis session. REST handles configuration,
uploads, and lifecycle operations; WebSockets stream progress and results.
Long-running workflows execute as background asyncio tasks. Session artifacts
are stored on disk, while SQLite holds users, encrypted credentials, and global
application metadata.

| Location | Responsibility |
| --- | --- |
| `backend/main.py`, `backend/routes/` | Application lifecycle, HTTP routes, and WebSockets. |
| `backend/constants.py` | Version, graph-node registry, workflow signals, and node mappings. |
| `backend/workflow/` | Planning, execution, review, synthesis, and report orchestration. |
| `backend/agents/` | PI, Bioinformatics, Literature, and optional specialist agents. |
| `backend/services/` | Session storage, task ownership, uploads, reports, and artifact services. |
| `backend/models/` | Pydantic request and state schemas. |
| `backend/utils/` | LLM protocols, settings resolution, Jupyter execution, and runtime persistence. |
| `backend/state.py` | Active session, workflow, task, lock, and WebSocket registries. |
| `frontend/src/components/`, `frontend/src/pages/` | Application UI and configuration pages. |
| `frontend/src/hooks/api/`, `frontend/src/stores/` | REST/WebSocket integration and Zustand state. |
| `clients/python/` | Python client and batch API. |
| `backend/tests/`, `test-profiles.toml`, `scripts/scai-test` | Backend tests and the supported test runner. |

Several large modules expose compatibility facades over split implementation
packages. For example, `backend/workflow/core.py` composes the workflow mixins;
frontend API hooks and stores also retain stable entry modules. Change the
implementation owner while preserving public imports and existing patch points.

## Workflow and configuration

`SingleCellAnalysisWorkflow` starts at `restore_entry`, which selects the correct
continuation node. Ordinary runs proceed through initialization, data profiling,
PI planning, optional modality analyses, Bioinformatics execution, literature
research, PI synthesis, and user feedback. Conditional routes handle plan
revision, further rounds, integration, and reporting. Stage 1 covers initial
exploration and annotation; Stage 2 performs the approved downstream analyses.
The final report sequence is summary, graphic abstract, then detailed report.

`WORKFLOW_GRAPH_NODES` in `backend/constants.py` is authoritative and is checked
against `backend/workflow/graph.py`. Adding or renaming a node also requires
updating its agent/result mappings, restore targets, serialized state consumers,
and relevant graph and restore tests. A restored user decision continues after
its parked gate; replaying the gate would duplicate the action.

Agent configuration resolves through `AgentSettingsProvider` in three layers:

1. Global defaults in `backend/config/_config.py`.
2. Per-user credentials and settings through `backend/utils/config_resolver.py`.
3. Session overrides on `AnalysisState`.

The provider reads the current session settings on each access. Agent code uses
it instead of reading global configuration directly. PI, Bioinformatics, and
Literature are core agents; unavailable core models require user intervention.
Pathology, Biochemistry, Painter, Supervisor, and Technique are optional. Most
agents inherit `BaseAgent`; `SupervisorAgent` exposes a parallel interface.
LLM protocol handling and English output enforcement live in
`backend/utils/llm_client.py`.

The Python client supports session submission, completion monitoring, and export
downloads; see its [README](../clients/python/README.md). Routine decision gates
use `analysis.round_wait_seconds` (default 300). Exceptional LLM failures use a
separate recovery policy: the first two dialogs retry automatically and the
third waits for an explicit action. Annotation-tool failures require an explicit
Retry, Skip, or Terminate choice. These recovery paths do not change the routine
wait setting.

## Scientific decisions and evidence

The PI agent owns scientific planning, method review, evidence interpretation,
and synthesis. Host code may measure design facts, identify missing provenance,
and attach observations for PI review. Those observations do not reject tools,
force supplementary work, change a scientific result's status, withhold
evidence, or block reports. Authentication, parseable schemas, execution and
cohort identity, persistence integrity, resource limits, and artifact linkage
remain technical contracts enforced by the host.

Method guidance is stored in `backend/data/method_rules.yaml` and retrieved
through `backend/services/method_rules.py`. Its enforcement values are
`advisory` and `guidance`. Compatibility APIs in
`backend/services/statistical_gates.py` always expose `blocking == []`, all
findings as advisories, and `passed is True`. Scientific acceptance belongs to
the PI, including when provenance is incomplete or methods are debatable.

Bioinformatics tools register plans, scores, methods, sample counts, findings,
tables, and figures alongside their execution identities. Preserve that linkage
through persistence and report assembly. A completed execution proves that code
ran; scientific sufficiency requires separate review. Retain older registered
plans and scores so the PI can assess the full record.

The analysis environment is Python-only. There is no R runtime or R service;
method guidance records relevant limitations of the Python alternatives.
Prompts and defaults remain neutral to organism, tissue, disease, and research
direction.

The Data tab lists uploaded files and manually entered biochemistry indicators.
Image-extracted values and legacy indicators remain available to analysis but
are excluded from the input table and counts. Preserve indicator origin and
provenance notes through session reload and configuration export/import.

## Session storage and recovery

The in-process `sessions` mapping holds active state. Durable session records
include `session.json`, workspace files, uploads, notebook and checkpoint data,
runtime state, reports, and logs. `backend/services/session_store.py` manages
disk writes and eviction; per-session locks serialize mutations. A UI status or
live kernel alone does not establish that a workflow or report is complete.

Ordinary saves persist supported top-level kernel values to
`<session>/states/<id>_session.pkl`. A signed runtime manifest records the
revision, saved and skipped names, and serializer version. Integrity signatures
are stored under the installation's integrity directory.
`_commit_pkl_transaction` in `backend/utils/sandbox_parts/regular_save.py`
commits the dump, signatures, metadata, and manifest together using a
crash-recoverable journal. Recovery completes or rolls back an interrupted
transaction before the next load.

The same transaction pairs a generation with notebook metadata, a SQLite backup
of the LangGraph checkpoint, and the Bioinformatics step checkpoint when present.
These copies live under `<session>/states/generation/<revision>/`. When live
state has advanced beyond a committed runtime dump, continuation can restore
this pair and disclose the rollback through `runtime_persistence.pair_restore`.
Retirement remains the fallback when no usable pair exists. `session.json`, the
notebook itself, and workspace files have separate write paths; the paired
checkpoint carries the corresponding analysis state.

A failed save keeps the previous committed generation and reports the failure
without parking the analysis on a new gate. Unsupported values are named in
`skipped_names`; a failed write does not publish a partial dump.
`backend/services/legacy_state_sanitizer.py` clears obsolete control fields on
load and import. It refuses an unfinished legacy restore transaction, including
an unreadable journal, because the session files may describe different points
in the run. Such a transaction must be completed or rolled back before running
the session.

Decision snapshots are an **experimental, opt-in** facility under
`backend/services/round_snapshots/`, enabled by an absolute `SCAI_SNAPSHOT_DIR`.
They copy an ordinarily committed generation before annotation confirmation,
direction decisions, and round acceptance; they do not add another serialization
mechanism. Capture is automatic, and restoration creates a new session with an
appropriate workflow continuation. Reports and logs are not copied. Automatic
pruning is disabled unless `SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB` is configured.
Keep independent backups; decision snapshots are not backups.

For notebook reproducibility, see [Notebook Replay](notebook-replay.md). That
procedure compares a recorded run with a fresh kernel in the same installation;
it is separate from production runtime recovery.

## Kernel runtime boundary

On Linux, the trusted launcher in `backend/utils/kernel_startup/launcher.py`
installs Landlock before importing workspace code. Startup requires Landlock
ABI 3 or newer. Native writes are limited to three per-session roots: the
workspace, the dill state directory, and the private kernel temporary directory.
The surrounding session directory remains host-owned. The roots must exist and
must not be symlinks or resolve to `/`.

The trusted startup code also installs a CPython audit hook that blocks ordinary
local process creation. Use threads or a remote scheduler. Local Dask clusters
use threaded workers; PyTorch `DataLoader` requires `num_workers=0`.
`DASK_DATAFRAME__QUERY_PLANNING=false` retains the dataframe behavior required by
arboreto. The shared `/dev/shm` is not kernel-writable.

This boundary does not restrict reads or network access. Non-Linux enforcement
is best effort, native FFI can bypass Python audit events, and the runtime state
directory remains kernel-writable. It is not a complete host or confidential-data
sandbox.

## Figures and reports

PI vision review judges scientific accuracy and presentation. Supervisor review
is advisory and cannot turn a successfully rendered figure into an execution
failure. The renderer in `backend/services/figure_layout.py` separately measures
layout inside the live kernel before saving. Objective blocking collisions can
exclude a figure from publication and gallery eligibility. An absent or invalid
measurement is unverified; consumers use the measured severity and shared
eligibility helpers instead of inferring acceptance from a filename or review
label.

Publication identity follows registered artifacts, logical figure IDs, and
revision lineage. Keep reports and galleries linked to the materialized output
of the recorded execution. Files remaining on disk do not automatically make
an earlier revision eligible for publication.

New analyses save PI-authored round chapters for reuse in final reports, with
global methods and synthesis sections. Captions and artifact references remain
bound to their saved versions; later limitations are disclosed separately.
The live detailed-report path is `backend/agents/supervisor_agent.py`; the
fallback is `backend/services/report/generator.py`. Keep both aligned with the
shared report services. Scientific caveats are disclosed to readers and the PI;
host validation checks structure, artifact references, and renderability.

## Accounts and access

Users can change their username after current-password verification;
administrators can rename non-deleting accounts. User IDs remain stable, so
sessions, API keys, and per-user configuration survive a rename. Preserve
username uniqueness, account-status checks, and write-time authorization in
`backend/routes/auth.py`, `backend/routes/admin.py`, and `backend/utils/db.py`.

## Development and verification

Run `./start.sh --dev` for the development server. Deployment, environment
variables, volumes, backup, and administration are documented in the
[Administrator Manual](admin-manual.md) and [Chinese Administrator Manual](admin-manual-zh.md).

The supported Python test entry point is `scripts/scai-test`:

```bash
scripts/scai-test fast backend/tests/test_version_consistency.py
scripts/scai-test client
scripts/scai-test kernel
scripts/scai-test backend-full
scripts/scai-test -- <raw pytest args>
```

Choose the smallest profile that covers the changed behavior. The runner uses
the immutable, digest-verified dependency image, a read-only checkout, disabled
network access, and separate evidence storage. It does not fall back to host
pytest. The frontend entry point is
`scripts/scai-frontend-test --install-and-test`; its isolated npm installation
requires network access and is a separate operation. See
[Test Strategy](test-strategy.md) for the complete environment and evidence
contract.

Each supported run records source and image identity, collection, outcomes, and
artifact evidence under `test-results/`. A passing focused run covers only its
selected tests. Test evidence does not establish that an application image was
built or deployed, that an existing session was repaired, or that a scientific
conclusion is valid. Preserve executable bits and trusted startup file modes
when copying a checkout.

When changing routes, update registration and `backend/tests/route_snapshot.json`.
When changing workflow or serialized state, check restore paths and older
sessions as well as ordinary execution. Keep schemas at route boundaries and
business logic in services. Add regression coverage for the affected behavior,
run the relevant supported checks, and run `git diff --check`.

For a release, synchronize the backend and Python client versions, README,
this guide, and both administrator manuals. Add release notes to
[CHANGELOG.md](CHANGELOG.md), retaining earlier entries. Record source checks,
image builds, and deployment as separate outcomes.
