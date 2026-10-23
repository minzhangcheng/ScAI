#!/bin/bash
# =============================================================================
# 02_star_align.sh — STAR Two-Pass Alignment
# =============================================================================
# 1. Build 1st genome index (with GTF)
# 2. 1st pass alignment → SJ.out.tab per sample
# 3. Filter and merge SJs (annotated + novel with ≥min reads)
# 4. Build 2nd genome index (with filtered SJs)
# 5. 2nd pass alignment → BAM + transcriptome BAM
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib.sh"

# --- Setup -------------------------------------------------------------------
ensure_dir "${ALIGN_DIR}"
ensure_dir "${TMP_DIR:-${WORK_DIR}/tmp}"
ensure_dir "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/02_star_align.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log_info "Starting STAR two-pass alignment..."

# --- Check prerequisites -----------------------------------------------------
check_tool STAR
check_tool samtools

# --- Ensure trimmed FASTQs exist ---------------------------------------------
if [[ ! -d "${FASTQ_TRIM_DIR}" ]] || [[ -z "$(ls -A "${FASTQ_TRIM_DIR}"/*.fq.gz 2>/dev/null)" ]]; then
    die "Trimmed FASTQ files not found in ${FASTQ_TRIM_DIR}. Run 01_qc_trim.sh first."
fi

# --- 1st Genome Index --------------------------------------------------------
STAR_VERSION=$(STAR --version 2>&1 | head -n 1)
SIG_1ST=$(make_signature "STAR=${STAR_VERSION};sjdbOverhang=${SJDB_OVERHANG};mode=first" \
    "${GENOME_FA}" "${ANNOT_GTF}")

if star_index_complete "${STAR_GENOME_INDEX}" "$SIG_1ST"; then
    log_info "1st genome index exists, skipping."
else
    log_info "Building 1st genome index..."
    rm -rf "${STAR_GENOME_INDEX}" && mkdir -p "${STAR_GENOME_INDEX}"
    STAR --runMode genomeGenerate \
        --genomeDir "${STAR_GENOME_INDEX}" \
        --genomeFastaFiles "${GENOME_FA}" \
        --sjdbOverhang "${SJDB_OVERHANG}" \
        --sjdbGTFfile "${ANNOT_GTF}" \
        --runThreadN "${THREADS}" \
        > "${LOG_DIR}/star_index_1st.log" 2>&1
    printf '%s\n' "$SIG_1ST" > "${STAR_GENOME_INDEX}/.pipeline_signature"
    log_info "1st genome index built."
fi

# --- 1st Pass Alignment ------------------------------------------------------
log_info "Running 1st pass alignment..."

# Determine read command for gzipped files
read_method="zcat"

for r1 in "${FASTQ_TRIM_DIR}"/*_1.fq.gz; do
    [[ -e "$r1" ]] || continue
    sample=$(basename "$r1" | sed 's/_1\.fq\.gz//')
    r2="${FASTQ_TRIM_DIR}/${sample}_2.fq.gz"

    log_info "  1st pass: $sample"
    STAR --genomeDir "${STAR_GENOME_INDEX}" \
        --readFilesIn "$r1" "$r2" \
        --runThreadN "${THREADS}" \
        --outFileNamePrefix "${TMP_DIR:-${WORK_DIR}/tmp}/${sample}." \
        --outFilterMultimapScoreRange 1 \
        --outFilterMultimapNmax 20 \
        --outFilterMismatchNmax 10 \
        --alignIntronMax 500000 \
        --alignMatesGapMax 1000000 \
        --sjdbScore 2 \
        --alignSJDBoverhangMin 1 \
        --genomeLoad NoSharedMemory \
        --readFilesCommand "${read_method}" \
        --outFilterMatchNminOverLread 0.33 \
        --outFilterScoreMinOverLread 0.33 \
        --sjdbOverhang "${SJDB_OVERHANG}" \
        --outSAMstrandField intronMotif \
        --outSAMtype None \
        --outSAMmode None \
        > "${LOG_DIR}/star_1st_${sample}.log" 2>&1

    # Merge SJ files
    cat "${TMP_DIR:-${WORK_DIR}/tmp}/${sample}.SJ.out.tab" >> "${ALIGN_DIR}/combined_SJ.out.tab"
done

# --- Filter and Sort SJs -----------------------------------------------------
log_info "Filtering splice junctions..."

awk -v min="${MIN_SJ_READS}" '$6 == 1 || $7 >= min' "${ALIGN_DIR}/combined_SJ.out.tab" \
    > "${ALIGN_DIR}/filtered_combined_SJ.out.tab"
LC_ALL=C sort -k1,1 -k2,2n -k3,3n "${ALIGN_DIR}/filtered_combined_SJ.out.tab" | uniq \
    > "${ALIGN_DIR}/unique_combined_SJ.out.tab"

if [[ ! -s "${ALIGN_DIR}/unique_combined_SJ.out.tab" ]]; then
    die "No usable splice junctions after filtering"
fi

# --- 2nd Genome Index --------------------------------------------------------
SIG_2ND=$(make_signature "STAR=${STAR_VERSION};sjdbOverhang=${SJDB_OVERHANG};mode=second" \
    "${GENOME_FA}" "${ALIGN_DIR}/unique_combined_SJ.out.tab")

if star_index_complete "${STAR_INDEX}" "$SIG_2ND"; then
    log_info "2nd genome index exists, skipping."
else
    log_info "Building 2nd genome index..."
    rm -rf "${STAR_INDEX}" && mkdir -p "${STAR_INDEX}"
    STAR --runMode genomeGenerate \
        --genomeDir "${STAR_INDEX}" \
        --genomeFastaFiles "${GENOME_FA}" \
        --sjdbOverhang "${SJDB_OVERHANG}" \
        --runThreadN "${THREADS}" \
        --sjdbFileChrStartEnd "${ALIGN_DIR}/unique_combined_SJ.out.tab" \
        > "${LOG_DIR}/star_index_2nd.log" 2>&1
    printf '%s\n' "$SIG_2ND" > "${STAR_INDEX}/.pipeline_signature"
    log_info "2nd genome index built."
fi

# --- 2nd Pass Alignment ------------------------------------------------------
log_info "Running 2nd pass alignment..."

for r1 in "${FASTQ_TRIM_DIR}"/*_1.fq.gz; do
    [[ -e "$r1" ]] || continue
    sample=$(basename "$r1" | sed 's/_1\.fq\.gz//')
    r2="${FASTQ_TRIM_DIR}/${sample}_2.fq.gz"

    log_info "  2nd pass: $sample"
    STAR --genomeDir "${STAR_INDEX}" \
        --readFilesIn "$r1" "$r2" \
        --runThreadN "${THREADS}" \
        --outFileNamePrefix "${ALIGN_DIR}/${sample}." \
        --sjdbGTFfile "${ANNOT_GTF}" \
        --outFilterMultimapScoreRange 1 \
        --outFilterMultimapNmax 20 \
        --outFilterMismatchNmax 10 \
        --alignIntronMax 500000 \
        --alignMatesGapMax 1000000 \
        --sjdbScore 2 \
        --genomeLoad NoSharedMemory \
        --limitBAMsortRAM 30000000000 \
        --readFilesCommand "${read_method}" \
        --outFilterMatchNminOverLread 0.33 \
        --outFilterScoreMinOverLread 0.33 \
        --sjdbOverhang "${SJDB_OVERHANG}" \
        --outSAMstrandField intronMotif \
        --outSAMattributes NH HI NM MD AS XS \
        --outSAMunmapped Within \
        --outSAMtype BAM SortedByCoordinate \
        --outSAMheaderHD @HD VN:1.4 \
        --outSAMattrRGline "ID:$sample SM:$sample PL:ILLUMINA LB:Lib_$sample" \
        --quantMode GeneCounts TranscriptomeSAM \
        --chimSegmentMin 12 \
        --chimJunctionOverhangMin 12 \
        --chimOutType Junctions WithinBAM SoftClip \
        --chimOutJunctionFormat 1 \
        --chimMainSegmentMultNmax 1 \
        --alignSJstitchMismatchNmax 5 -1 5 5 \
        --alignSJDBoverhangMin 10 \
        --chimSegmentReadGapMax 3 \
        > "${LOG_DIR}/star_2nd_${sample}.log" 2>&1

done

log_info "STAR alignment complete. BAMs in ${ALIGN_DIR}"
