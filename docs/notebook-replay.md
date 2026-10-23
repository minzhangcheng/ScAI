# Notebook Replay

`scripts/scai-notebook-replay` compares a session's recorded notebook with a new
execution in the same installation. This document retains the declared variables
and tolerances of the replay contract. A divergent result is a finding to
investigate; it does not justify changing the comparison thresholds.

## Scope

The claim is that, given the same input files, executing every recorded code cell
in order on a fresh kernel and clean workspace reproduces the recorded cell
outcomes and declared variables within the tolerances below. The kernel has no
inherited state and uses the same installation's packages, including the modules
under `backend/utils/kernel_startup`.

Replay does not require ScAI's production launcher or its full filesystem and
process restrictions. The recorded bootstrap handles ScAI-only setup
conditionally, and the harness adds the kernel-startup modules to the replay
path so the compatibility shims are available. If those modules are missing, the
harness warns and the result supports a weaker reproducibility claim.

The comparison does not claim byte-identical figures or other artifacts,
unchanging external network resources, equal runtime or memory use, or
reproducibility across installations and package versions. It is also different
from ordinary Jupyter "Restart & Run All": the harness continues through errors
because recorded failures are part of the original execution.

## Recorded outcomes

Every code cell is compared as `ok` or `raised`. For raised cells, the exception
type must also agree; traceback text is not compared. A previously successful
cell that now raises is a divergence. A previously failing cell that now succeeds
is also a divergence. The same recorded exception in both runs is a match.

Network-related divergences should be identified separately in the interpretation
of the replay report. The current CLI does not automatically classify or exempt
them from its machine-readable `passed` flag.

## Declared variables and tolerances

| Variable or property | Measurement | Tolerance |
| --- | --- | --- |
| Primary matrix shape | `adata.shape` | Exact |
| Observation names | Digest of `sorted(adata.obs_names)` | Exact |
| Variable names | Digest of `sorted(adata.var_names)` | Exact |
| `adata.X` sum | `float(X.sum())` | Relative `1e-6` |
| `adata.X` non-zero count | `int((X != 0).sum())` | Exact |
| Cluster labels | Value counts for each categorical `obs` column, sorted as `(label, count)` pairs | Exact multiset |
| Number of clusters | `len(unique)` per categorical `obs` column | Exact |
| Embedding shapes | `adata.obsm[k].shape` for each `k` | Exact |
| Embedding values | `adata.obsm[k]`, mean absolute difference over finite entries | Relative `1e-4` |
| Highly variable genes | `set(adata.var_names[adata.var.highly_variable])` | Exact |
| Declared session outputs | Names from `state.code_history[*].declared_persistent_outputs` present in each namespace | Exact set |

For a scalar replayed value `a` and original value `b`, the bound is
`abs(a - b) <= rtol * max(1, abs(b))`. NaN matches NaN; replacing a finite value
with NaN is a divergence. For embeddings, the implementation compares the mean
absolute difference against `1e-4 * max(1, mean(abs(original_finite_values)))`.
The positions and kinds of non-finite embedding values must agree exactly.

A variable absent from both runs is reported as `absent`, not passed. Presence
in only one run is a divergence. Missing or failed extraction evidence is
`unmeasured`. The current fingerprint format records embedding values rather
than aggregate summaries; legacy fingerprints without those values cannot
establish an embedding match.

Implementation details matter when interpreting the measurements:

- The current extractor uses sparse `X.nnz` when available. Explicitly stored
  zeros can therefore affect that count even though the declared quantity is
  the mathematical non-zero count.
- Cluster comparisons use category counts, including object and boolean columns
  in the current extractor. They do not establish identical label assignments
  for every cell; the cluster count is implied by the compared count lists.
- Declared output comparisons measure namespace presence, not the values of
  arbitrary output objects.

## Running a comparison

Use a completed session and retain its notebook and input files. Capture the
original fingerprint **before** starting replay. The CLI reads the original
committed runtime dump; the underlying procedure also permits measurement from
the original live kernel, but the CLI has no live-kernel fingerprint mode.

```bash
scripts/scai-notebook-replay fingerprint /path/to/session /tmp/original-fingerprint.json
scripts/scai-notebook-replay replay /path/to/session /tmp/original-fingerprint.json /tmp/replay-report.json --workspace /path/to/clean-workspace
```

The fingerprint command requires a directory containing exactly one runtime
dump and an unambiguous session state JSON for output declarations. The replay
command requires exactly one notebook below the supplied directory. Select a
more precise directory when an archive contains multiple candidates.

Provide a new, empty workspace for each comparison and make the same input files
available at the paths the notebook expects. The harness creates the workspace
if necessary but does not clear an existing one or copy inputs. It sets
`SCAI_REPLAY_WORKSPACE`, `SCAI_REPLAY_SESSION_TMP`, and `SCAI_REPLAY_STATE_DIR`
for the recorded bootstrap. Older notebooks without bootstrap cells warn that
these overrides cannot take effect and do not support a contract verdict.

The harness executes all code cells, records outcomes, extracts the replayed
variables, and writes a JSON report containing both fingerprints and their
comparisons. A CLI success requires no cell-outcome divergences and no variable
marked `diverged` or `unmeasured`. The declared contract allows separately
identified network divergences in its interpretation; retain those findings
with the report instead of changing the CLI result or tolerances.

Preserve the report, environment identity, inputs, and any identified limitations
with the reproducibility record. This procedure measures notebook replay; it
does not determine scientific validity or alter workflow acceptance.
