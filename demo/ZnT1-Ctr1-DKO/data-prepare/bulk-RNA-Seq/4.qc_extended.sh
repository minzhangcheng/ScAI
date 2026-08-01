#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Basic parameter definitions
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
align_dir=${work_dir}/align
fastqc_dir=${work_dir}/fastqc
fastq_trim_dir=${work_dir}/fastq_trim
log_dir=${work_dir}/log/4.qc_extended.sh
expr_dir=${work_dir}/expr
multiqc_dir=${work_dir}/multiqc
rseqc_dir=${work_dir}/rseqc
anot_gtf=${work_dir}/gencode.vM36.annotation.gtf
anot_bed=${work_dir}/gencode.vM36.annotation.bed12
thread=32

mkdir -p "${multiqc_dir}" "${rseqc_dir}" "${log_dir}" "${expr_dir}"

exec > >(tee -a "${log_dir}/qc_extended.log") 2>&1
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting Extended QC..."

# 0. Tool availability check — avoids silent skips when conda env isn't active.
for tool in samtools read_distribution.py infer_experiment.py gtfToGenePred genePredToBed multiqc; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "Error: '$tool' not found in PATH. Activate the 'rna-seq' conda env first." >&2
        exit 1
    }
done

# Collect BAM list once; empty directory should fail loudly rather than pass a literal glob to samtools.
bams=( "${align_dir}"/*.Aligned.sortedByCoord.out.bam )
if (( ${#bams[@]} == 0 )); then
    echo "Error: No BAM files found in ${align_dir}" >&2
    exit 1
fi

# 1. BAM Indexing
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Indexing BAM files..."
total=${#bams[@]}
count=0
for bam in "${bams[@]}"; do
    count=$((count + 1))
    if [ ! -f "${bam}.bai" ]; then
        echo "[${count}/${total}] Indexing ${bam} ..."
        samtools index -@ "${thread}" "${bam}"
    fi
done

# 2. RSeQC (requires BED12)
# Convert GTF to BED12 via UCSC tools. Write to a temp file and atomically rename
# so an interrupted conversion won't leave a half-written BED12 that later runs trust.
if [ ! -f "${anot_bed}" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Converting GTF to BED12 for RSeQC..."
    tmp_gp="${work_dir}/tmp.$$.genePred"
    tmp_bed="${anot_bed}.tmp.$$"
    trap 'rm -f "${tmp_gp}" "${tmp_bed}"' EXIT
    gtfToGenePred "${anot_gtf}" "${tmp_gp}"
    genePredToBed "${tmp_gp}" "${tmp_bed}"
    mv "${tmp_bed}" "${anot_bed}"
    rm -f "${tmp_gp}"
    trap - EXIT
fi

for bam in "${bams[@]}"; do
    sample=$(basename "${bam}" .Aligned.sortedByCoord.out.bam)
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Running RSeQC for ${sample}..."

    # Read distribution — keep stderr out of the result file so downstream parsers aren't polluted.
    read_distribution.py -i "${bam}" -r "${anot_bed}" \
        > "${rseqc_dir}/${sample}.read_dist.txt" \
        2> "${log_dir}/${sample}.read_dist.err"

    # Gene body coverage is disabled by default — slow on large cohorts.
    # geneBody_coverage.py -r "${anot_bed}" -i "${bam}" -o "${rseqc_dir}/${sample}"

    # Infer experiment (strandedness)
    infer_experiment.py -r "${anot_bed}" -i "${bam}" \
        > "${rseqc_dir}/${sample}.infer_exp.txt" \
        2> "${log_dir}/${sample}.infer_exp.err"
done

# 3. MultiQC — scan only QC-producing subdirs; work_dir contains multi-GB FASTA/GTF.
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Generating MultiQC report..."
multiqc "${fastqc_dir}" "${fastq_trim_dir}" "${align_dir}" "${rseqc_dir}" "${expr_dir}" \
    -o "${multiqc_dir}" --force

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Extended QC complete. Report in ${multiqc_dir}"
