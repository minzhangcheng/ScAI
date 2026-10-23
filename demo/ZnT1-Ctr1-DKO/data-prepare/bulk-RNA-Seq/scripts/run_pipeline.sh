#!/bin/bash
# =============================================================================
# run_pipeline.sh — Master Orchestrator
# =============================================================================
# Runs all pipeline steps in order:
#   01_qc_trim.sh → 02_star_align.sh → 03_rsem_quant.sh → 04_qc_aggregate.sh → 05_merge_counts.R
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration -----------------------------------------------------------
WORK_DIR="${WORK_DIR:-.}"
export WORK_DIR=$(cd "$WORK_DIR" && pwd -P)
LOG_DIR="${WORK_DIR}/log/run_pipeline"
mkdir -p "${LOG_DIR}"

# Master log
MASTER_LOG="${LOG_DIR}/pipeline.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log_info() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO: $*"
}

log_error() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR: $*" >&2
}

die() {
    log_error "$*"
    exit 1
}

# --- Step runner -------------------------------------------------------------
run_step() {
    local step_num="$1"
    local script_name="$2"
    local description="$3"
    local script_path="${SCRIPT_DIR}/${script_name}"

    log_info ""
    log_info "========================================"
    log_info "STEP ${step_num}: ${description}"
    log_info "========================================"

    local -a command=(bash "$script_path")
    if [[ "$script_name" == *.R ]]; then
        command=(Rscript "$script_path" "$WORK_DIR")
    fi

    local start_time end_time
    start_time=$(date '+%s')

    if "${command[@]}"; then
        end_time=$(date '+%s')
        local elapsed=$((end_time - start_time))
        log_info "STEP ${step_num} COMPLETED in ${elapsed}s"
        return 0
    else
        end_time=$(date '+%s')
        local elapsed=$((end_time - start_time))
        log_error "STEP ${step_num} FAILED after ${elapsed}s"
        return 1
    fi
}

# --- Check prerequisites -----------------------------------------------------
log_info "========================================"
log_info "RNA-Seq Pipeline: FASTQ → merged_counts"
log_info "Working directory: ${WORK_DIR}"
log_info "========================================"

# Check conda environment
if ! conda info --envs 2>/dev/null | grep -q "^rna-seq "; then
    die "Conda environment 'rna-seq' not found. Run: conda env list"
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate rna-seq

# Check genome files exist
if [[ ! -f "${WORK_DIR}/GRCm39.primary_assembly.genome.fa" ]]; then
    die "Genome FASTA not found: ${WORK_DIR}/GRCm39.primary_assembly.genome.fa"
fi
if [[ ! -f "${WORK_DIR}/gencode.vM39.annotation.gtf" ]]; then
    die "Annotation GTF not found: ${WORK_DIR}/gencode.vM39.annotation.gtf"
fi

# Check FASTQ input
if [[ ! -d "${WORK_DIR}/fastq" ]]; then
    die "FASTQ directory not found: ${WORK_DIR}/fastq"
fi

# --- Run steps ---------------------------------------------------------------
FAILED=0

run_step "01" "01_qc_trim.sh"     "QC and Trimming"         || exit 1
run_step "02" "02_star_align.sh" "STAR Two-Pass Alignment" || exit 1
run_step "03" "03_rsem_quant.sh" "RSEM Quantification"     || exit 1
run_step "04" "04_qc_aggregate.sh" "Extended QC"             || exit 1
run_step "05" "05_merge_counts.R" "Merge Counts"            || exit 1

# --- Summary -----------------------------------------------------------------
log_info ""
log_info "========================================"
if (( FAILED == 0 )); then
    log_info "PIPELINE COMPLETE — ALL STEPS SUCCEEDED"
    log_info "Output: ${WORK_DIR}/expr/merged_counts.csv"
else
    log_error "PIPELINE FAILED — ${FAILED} STEP(S) FAILED"
    log_error "Check logs in: ${LOG_DIR}"
    exit 1
fi
log_info "========================================"
