#!/bin/bash
# =============================================================================
# 03_rsem_quant.sh — RSEM Quantification
# =============================================================================
# Builds RSEM reference (if needed) and quantifies expression from
# STAR transcriptome BAMs.
# Output: expr/*.genes.results, expr/*.isoforms.results
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib.sh"

# --- Setup -------------------------------------------------------------------
ensure_dir "${EXPR_DIR}"
ensure_dir "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/03_rsem_quant.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log_info "Starting RSEM quantification..."

# --- Check prerequisites -----------------------------------------------------
check_tool rsem-calculate-expression

# --- Infer strandedness (RSeQC) ---------------------------------------------
# Use infer_experiment.py on a subset of BAMs to determine strandedness
log_info "Inferring library strandedness..."

RSEM_STRANDEDNESS="none"
if [[ -d "${RSEQC_DIR}" ]]; then
    forward_n=0; reverse_n=0; none_n=0
    for bam in "${ALIGN_DIR}"/*.Aligned.sortedByCoord.out.bam; do
        [[ -e "$bam" ]] || continue
        sample=$(basename "$bam" .Aligned.sortedByCoord.out.bam)
        infer_out="${RSEQC_DIR}/${sample}.infer_exp.txt"
        if [[ -f "$infer_out" ]]; then
            forward_fraction=$(awk -F': ' '/1\+\+,1--,2\+-,2-\+/{print $2}' "$infer_out" 2>/dev/null || echo "0")
            reverse_fraction=$(awk -F': ' '/1\+-,1-\+,2\+\+,2--/{print $2}' "$infer_out" 2>/dev/null || echo "0")
            # Classify
            if awk -v f="$forward_fraction" -v r="$reverse_fraction" 'BEGIN {exit (f >= 0.80 && f-r >= 0.40) ? 0 : 1}'; then
                forward_n=$((forward_n + 1))
            elif awk -v f="$forward_fraction" -v r="$reverse_fraction" 'BEGIN {exit (r >= 0.80 && r-f >= 0.40) ? 0 : 1}'; then
                reverse_n=$((reverse_n + 1))
            else
                none_n=$((none_n + 1))
            fi
        fi
    done

    if (( forward_n > 0 && reverse_n == 0 )); then
        RSEM_STRANDEDNESS="forward"
    elif (( reverse_n > 0 && forward_n == 0 )); then
        RSEM_STRANDEDNESS="reverse"
    fi
    log_info "Inferred strandedness: ${RSEM_STRANDEDNESS} (forward=$forward_n reverse=$reverse_n none=$none_n)"
else
    log_warn "RSeQC directory not found, using strandedness=none"
fi

# --- Build RSEM Reference ----------------------------------------------------
RSEM_VERSION=$(rsem-calculate-expression --version 2>&1 | head -n 1)
RSEM_SIG=$(make_signature "RSEM=${RSEM_VERSION}" "${GENOME_FA}" "${ANNOT_GTF}")

if rsem_index_complete "${RSEM_REF}" "$RSEM_SIG"; then
    log_info "RSEM reference exists, skipping."
else
    log_info "Building RSEM reference..."
    ensure_dir "$(dirname "${RSEM_REF}")"
    rsem-prepare-reference -p "${THREADS}" --gtf "${ANNOT_GTF}" "${GENOME_FA}" "${RSEM_REF}" \
        > "${LOG_DIR}/rsem_prepare_reference.log" 2>&1
    printf '%s\n' "$RSEM_SIG" > "${RSEM_REF}.pipeline_signature"
    log_info "RSEM reference built."
fi

# --- Quantify Expression -----------------------------------------------------
log_info "Quantifying expression for all samples..."

rsem_thread=$(( THREADS / MAX_RSEM_JOBS ))
(( rsem_thread < 1 )) && rsem_thread=1

pids=()
for bam in "${ALIGN_DIR}"/*.Aligned.toTranscriptome.out.bam; do
    [[ -e "$bam" ]] || continue
    sample=$(basename "$bam" .Aligned.toTranscriptome.out.bam)

    (
        log_info "  Quantifying: $sample"
        rsem-calculate-expression -p "${rsem_thread}" --alignments --paired-end \
            --strandedness "${RSEM_STRANDEDNESS}" \
            "$bam" \
            "${RSEM_REF}" \
            "${EXPR_DIR}/${sample}" \
            > "${LOG_DIR}/rsem_${sample}.log" 2>&1
        log_info "  Done: $sample"
    ) &
    pids+=($!)
    while [[ $(jobs -r -p | wc -l) -ge ${MAX_RSEM_JOBS} ]]; do
        sleep 1
    done
done

wait_all "${pids[@]}"

log_info "RSEM quantification complete. Results in ${EXPR_DIR}"
