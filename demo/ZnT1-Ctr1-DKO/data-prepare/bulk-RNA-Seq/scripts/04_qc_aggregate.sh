#!/bin/bash
# =============================================================================
# 04_qc_aggregate.sh — Extended QC (RSeQC + MultiQC)
# =============================================================================
# 1. Index all BAM files
# 2. Run RSeQC (read_distribution, infer_experiment)
# 3. Generate MultiQC report
# =============================================================================

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib.sh"

# --- Setup -------------------------------------------------------------------
ensure_dir "${RSEQC_DIR}"
ensure_dir "${MULTIQC_DIR}"
ensure_dir "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/04_qc_aggregate.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log_info "Starting extended QC..."

# --- Check prerequisites -----------------------------------------------------
for tool in samtools read_distribution.py infer_experiment.py gtfToGenePred genePredToBed multiqc; do
    check_tool "$tool"
done

# --- Collect BAMs -----------------------------------------------------------
bams=( "${ALIGN_DIR}"/*.Aligned.sortedByCoord.out.bam )
if (( ${#bams[@]} == 0 )); then
    die "No BAM files found in ${ALIGN_DIR}"
fi

log_info "Found ${#bams[@]} BAM file(s)"

# --- Index BAMs -------------------------------------------------------------
log_info "Indexing BAM files..."
count=0
total=${#bams[@]}
for bam in "${bams[@]}"; do
    count=$((count + 1))
    if [[ ! -f "${bam}.bai" ]]; then
        log_info "  [${count}/${total}] Indexing $(basename "$bam")"
        samtools index -@ "${THREADS}" "$bam"
    fi
done

# --- Convert GTF to BED12 for RSeQC ------------------------------------------
ANOT_BED="${WORK_DIR}/gencode.vM39.annotation.bed12"
if [[ ! -f "${ANOT_BED}" ]]; then
    log_info "Converting GTF to BED12..."
    tmp_gp="${WORK_DIR}/tmp.$$.genePred"
    tmp_bed="${ANOT_BED}.tmp.$$"
    trap 'rm -f "${tmp_gp}" "${tmp_bed}"' EXIT
    gtfToGenePred "${ANNOT_GTF}" "${tmp_gp}"
    genePredToBed "${tmp_gp}" "${tmp_bed}"
    mv "${tmp_bed}" "${ANOT_BED}"
    rm -f "${tmp_gp}"
    trap - EXIT
fi

# --- RSeQC ------------------------------------------------------------------
log_info "Running RSeQC..."
for bam in "${bams[@]}"; do
    sample=$(basename "${bam}" .Aligned.sortedByCoord.out.bam)
    log_info "  RSeQC: ${sample}"

    read_distribution.py -i "${bam}" -r "${ANOT_BED}" \
        > "${RSEQC_DIR}/${sample}.read_dist.txt" \
        2> "${LOG_DIR}/${sample}.read_dist.err"

    infer_experiment.py -r "${ANOT_BED}" -i "${bam}" \
        > "${RSEQC_DIR}/${sample}.infer_exp.txt" \
        2> "${LOG_DIR}/${sample}.infer_exp.err"
done

# --- MultiQC ----------------------------------------------------------------
log_info "Generating MultiQC report..."
multiqc "${FASTQC_DIR}" "${FASTQ_TRIM_DIR}" "${ALIGN_DIR}" "${RSEQC_DIR}" "${EXPR_DIR}" \
    -o "${MULTIQC_DIR}" --force \
    > "${LOG_DIR}/multiqc.log" 2>&1

log_info "Extended QC complete. Report: ${MULTIQC_DIR}/multiqc_report.html"
