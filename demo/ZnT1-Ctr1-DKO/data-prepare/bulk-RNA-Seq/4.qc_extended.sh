#!/bin/bash
# Compatibility entry point for the verified DKO preprocessing workflow.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR="$(cd "${WORK_DIR:-.}" && pwd -P)"
bash "${SCRIPT_DIR}/scripts/04_qc_aggregate.sh"
