#!/usr/bin/env bash
# Run cellranger count on each sample sequentially (snRNA-seq; v10 includes introns by default).
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAWDATA="Archive/snRNA-seq"
FASTQ="${BASE_DIR}/fastq"
CR=/home/minzhang/cellranger-10.0.0/cellranger
REF_DIR="${HOME}/refdata-gex-GRCm39-2024-A"
OUT_DIR="${BASE_DIR}/results"

mkdir -p ${FASTQ}
mkdir -p ${OUT_DIR}

ln -sfn "${RAWDATA}/WT-LGJ1033_L3_1.fq.gz" \
        "${FASTQ}/WT_S1_L001_R1_001.fastq.gz"
ln -sfn "${RAWDATA}/WT-LGJ1033_L3_2.fq.gz" \
        "${FASTQ}/WT_S1_L001_R2_001.fastq.gz"
ln -sfn "${RAWDATA}/DKO-LGJ1032_L3_1.fq.gz" \
        "${FASTQ}/DKO_S1_L001_R1_001.fastq.gz"
ln -sfn "${RAWDATA}/DKO-LGJ1032_L3_2.fq.gz" \
        "${FASTQ}/DKO_S1_L001_R2_001.fastq.gz"

SAMPLES=("WT" "DKO")

CORES="${CORES:-$(nproc)}"
MEM_GB="${MEM_GB:-64}"

cd "${OUT_DIR}"

for sample in "${SAMPLES[@]}"; do

    out_sample_dir="${OUT_DIR}/${sample}"
    if [[ -f "${out_sample_dir}/outs/web_summary.html" ]]; then
        echo "[cellranger] ${sample} already finished, skipping"
        continue
    fi
    if [[ -d "${out_sample_dir}" ]]; then
        echo "[cellranger] removing stale pipestance ${out_sample_dir}"
        rm -rf "${out_sample_dir}"
    fi

    echo "[cellranger] start ${sample}"
    "${CR}" count \
        --id="${sample}" \
        --transcriptome="${REF_DIR}" \
        --fastqs=${FASTQ} \
        --sample="${sample}" \
        --create-bam=true \
        --localcores="${CORES}" \
        --localmem="${MEM_GB}"
    echo "[cellranger] ${sample} finished"
done

echo "[cellranger] all finished"
