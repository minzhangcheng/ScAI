#!/usr/bin/env bash
# Run velocyto run10x on each cellranger output to produce .loom files for pseudotime / RNA velocity.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${HOME}/refdata-gex-GRCm39-2024-A"
GTF="${REF_DIR}/genes/genes.gtf"
MASK_GTF="${BASE_DIR}/mm39_rmsk.gtf"
OUT_DIR="${BASE_DIR}/results"

# shellcheck disable=SC1091
source "${HOME}/.miniconda3/etc/profile.d/conda.sh"
conda activate sc

if [[ ! -f "${GTF}" ]]; then
    echo "[velocyto] Missing GTF: ${GTF}; please run 00_setup.sh first" >&2
    exit 1
fi
if [[ ! -f "${MASK_GTF}" ]]; then
    echo "[velocyto] Missing repeat mask GTF: ${MASK_GTF}; please run 2.rmsk.sh first" >&2
    exit 1
fi

SAMPLES=("WT" "DKO")
CORES="${CORES:-$(nproc)}"

for sample in "${SAMPLES[@]}"; do
    cr_dir="${OUT_DIR}/${sample}"
    loom_file="${cr_dir}/velocyto/${sample}.loom"
    if [[ -f "${loom_file}" ]]; then
        echo "[velocyto] ${sample} already has loom, skipping"
        continue
    fi
    if [[ ! -f "${cr_dir}/outs/possorted_genome_bam.bam" ]]; then
        echo "[velocyto] ${sample} missing cellranger BAM, skipping" >&2
        continue
    fi
    echo "[velocyto] start ${sample}"
    velocyto run10x -@ "${CORES}" -m "${MASK_GTF}" "${cr_dir}" "${GTF}"
    echo "[velocyto] ${sample} done -> ${loom_file}"
done

echo "[velocyto] all finished"
