#!/bin/bash
# =============================================================================
# 01_qc_trim.sh — FastQC + Adapter Trimming
# =============================================================================
# Runs FastQC on raw reads, then trim_galore (with FastQC on trimmed reads).
# Output: fastqc/ + fastq_trim/
# =============================================================================

set -euo pipefail

# Load config and library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib.sh"

# --- Setup -------------------------------------------------------------------
ensure_dir "${FASTQC_DIR}"
ensure_dir "${FASTQ_TRIM_DIR}"
ensure_dir "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/01_qc_trim.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log_info "Starting QC and trimming..."

# --- Check prerequisites -----------------------------------------------------
check_tool fastqc
check_tool trim_galore

# --- Discover and link samples -----------------------------------------------
discover_and_link_samples "${FASTQ_DIR}" "${FASTQ_MERGED_DIR}"

# --- FastQC on raw reads -----------------------------------------------------
log_info "Running FastQC on raw reads..."
pids=()
for fq in "${FASTQ_MERGED_DIR}"/*.fq.gz; do
    (
        sample=$(basename "$fq")
        log_info "  FastQC: $sample"
        fastqc -t 4 -o "${FASTQC_DIR}" "$fq" > "${LOG_DIR}/fastqc_${sample}.log" 2>&1
    ) &
    pids+=($!)
    while [[ $(jobs -r -p | wc -l) -ge ${MAX_FASTQC_JOBS} ]]; do
        sleep 1
    done
done
wait_all "${pids[@]}"
log_info "FastQC complete."

# --- trim_galore -------------------------------------------------------------
log_info "Running trim_galore..."
pids=()
for r1 in "${FASTQ_MERGED_DIR}"/*_1.fq.gz; do
    [[ -e "$r1" ]] || continue
    sample=$(basename "$r1" | sed 's/_1\.fq\.gz//')
    r2="${FASTQ_MERGED_DIR}/${sample}_2.fq.gz"

    (
        log_info "  Trimming: $sample"
        trim_galore \
            --output_dir "${FASTQ_TRIM_DIR}" \
            --gzip \
            --cores 2 \
            --clip_R1 "${TRIM_LENGTH}" \
            --clip_R2 "${TRIM_LENGTH}" \
            --fastqc \
            --fastqc_args "-t 2 -o ${FASTQC_DIR}" \
            --paired "$r1" "$r2" \
            > "${LOG_DIR}/trim_${sample}.log" 2>&1

        # Rename trim_galore output to consistent naming
        for r in 1 2; do
            src="${FASTQ_TRIM_DIR}/${sample}_${r}_val_${r}.fq.gz"
            dst="${FASTQ_TRIM_DIR}/${sample}_${r}.fq.gz"
            if [[ -f "$src" ]]; then
                mv -f "$src" "$dst"
            elif [[ ! -f "$dst" ]]; then
                die "trim_galore output missing for ${sample} R${r}: expected $src"
            fi
        done
    ) &
    pids+=($!)
    while [[ $(jobs -r -p | wc -l) -ge ${MAX_TRIM_JOBS} ]]; do
        sleep 1
    done
done
wait_all "${pids[@]}"

log_info "Trimming complete. Output in ${FASTQ_TRIM_DIR}"
