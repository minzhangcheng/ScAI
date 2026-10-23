#!/bin/bash
# Compatibility entry point for the verified DKO preprocessing workflow.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR="$(cd "${WORK_DIR:-.}" && pwd -P)"
bash "${SCRIPT_DIR}/scripts/01_qc_trim.sh"
bash "${SCRIPT_DIR}/scripts/02_star_align.sh"
bash "${SCRIPT_DIR}/scripts/03_rsem_quant.sh"
