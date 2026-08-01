#!/bin/bash
set -euo pipefail

# conda activate pytorch

# Basic parameter definitions
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
genome_file=GRCm39.primary_assembly.genome.fa
anot_file=gencode.vM36.annotation.gtf
read_length=150
trim_length=10
thread=32
max_io_jobs=8
min_SJ_reads=5


# Paths
genome_fa=${work_dir}/${genome_file}
anot_gtf=${work_dir}/${anot_file}
fastq_raw_dir=${work_dir}/fastq
fastq_dir=${work_dir}/fastq_merged
align_dir=${work_dir}/align
expr_dir=${work_dir}/expr
tmp_dir=${work_dir}/tmp
rsem_reference=${work_dir}/rsem_reference/rsem_ref
log_dir=${work_dir}/log/3.align.sh
fastq_suffix=fq.gz
fastqc_dir=${work_dir}/fastqc
fastq_trim_dir=${work_dir}/fastq_trim
# Create directories
mkdir -p ${fastq_dir} ${align_dir} ${tmp_dir} ${expr_dir} ${log_dir} $(dirname ${rsem_reference}) ${fastqc_dir} ${fastq_trim_dir} ${work_dir}/star_genome_index ${work_dir}/star_index

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting alignment and quantification..."
exec > >(tee -a "${log_dir}/align.log") 2>&1

# star_index_complete: return 0 if dir contains all essential files from a successful genomeGenerate.
star_index_complete() {
    local dir="$1" f
    for f in SA SAindex Genome chrLength.txt chrName.txt chrStart.txt genomeParameters.txt; do
        [ -f "${dir}/${f}" ] || return 1
    done
    return 0
}

# wait_all: wait on all pids, collect failures without set -e aborting the loop on first failure.
wait_all() {
    local pid fail=0
    for pid in "$@"; do
        wait "$pid" || fail=$((fail + 1))
    done
    if (( fail > 0 )); then
        echo "Error: $fail parallel job(s) failed" >&2
        exit 1
    fi
}


# Create symlinks: unify two naming conventions into {Group}_{ID}_1.fq.gz / _2.fq.gz
# Ctrl is renamed to WT
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Creating symlinks for samples..."

# link_one: refuse to silently overwrite an existing symlink pointing elsewhere.
link_one() {
    local s="$1" d="$2"
    if [ ! -e "$s" ]; then
        echo "Error: source not found: $s" >&2; exit 1
    fi
    if [ -L "$d" ] && [ "$(readlink "$d")" != "$s" ]; then
        echo "Error: symlink conflict: $d already points to $(readlink "$d"), refusing to overwrite with $s" >&2
        exit 1
    fi
    ln -sf "$s" "$d"
}

# Pattern A: {id}-{group}_1.fq.gz (e.g., 2-DKO_1.fq.gz, 3-Ctrl_1.fq.gz)
for fq in "${fastq_raw_dir}"/[0-9]*-*_1.fq.gz; do
    [ -e "$fq" ] || continue
    fname=$(basename "$fq")
    id=$(echo "$fname" | sed 's/-.*//')
    group=$(echo "$fname" | sed 's/^[0-9]*-//;s/_1\.fq\.gz//')
    group=${group/Ctrl/WT}
    link_one "$fq"                       "${fastq_dir}/${group}_${id}_1.fq.gz"
    link_one "${fq/_1.fq.gz/_2.fq.gz}"   "${fastq_dir}/${group}_${id}_2.fq.gz"
done

# Pattern B: {group}-{id}_L1_UDI{xxx}.R1.fastq.gz (e.g., DKO-790_L1_UDI075.R1.fastq.gz)
for fq in "${fastq_raw_dir}"/*.R1.fastq.gz; do
    [ -e "$fq" ] || continue
    fname=$(basename "$fq")
    group=$(echo "$fname" | sed 's/-.*//')
    id=$(echo "$fname" | sed 's/^[^-]*-//;s/_L1_.*//')
    group=${group/Ctrl/WT}
    link_one "$fq"                              "${fastq_dir}/${group}_${id}_1.fq.gz"
    link_one "${fq/.R1.fastq.gz/.R2.fastq.gz}"  "${fastq_dir}/${group}_${id}_2.fq.gz"
done

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Symlinks created. Samples:"
ls ${fastq_dir}/*_1.${fastq_suffix} | xargs -I{} basename {} | sed "s/_1.${fastq_suffix}//"


# Quality control of raw sequencing data
pids=()
for i in ${fastq_dir}/*.${fastq_suffix}; do
    (
        sample=$(basename "$i")
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Performing FastQC quality check on: $sample..."
        fastqc -t 4 -o "${fastqc_dir}" "${fastq_dir}/${sample}" > "${log_dir}/fastqc_${sample}.log" 2>&1
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished performing FastQC quality check on: $sample..."
    ) &
    pids+=($!)
    while [[ $(jobs -r -p | wc -l) -ge $max_io_jobs ]]; do
        sleep 1
    done
done
wait_all "${pids[@]}"


# Trim adaptor and index
pids=()
for i in ${fastq_dir}/*_1.${fastq_suffix}; do
    (
        sample=$(basename "$i")
        sample=${sample%"_1.$fastq_suffix"}
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Trimming adaptor and indexing on: $sample..."
        trim_galore --output_dir "${fastq_trim_dir}" --gzip --cores 4 \
            --clip_R1 "$trim_length" --clip_R2 "$trim_length" \
            --fastqc --fastqc_args "-t 4 -o ${fastqc_dir}" \
            --paired "${fastq_dir}/${sample}_1.${fastq_suffix}" "${fastq_dir}/${sample}_2.${fastq_suffix}" \
            > "${log_dir}/trim_${sample}.log" 2>&1
        for r in 1 2; do
            src="${fastq_trim_dir}/${sample}_${r}_val_${r}.fq.gz"
            dst="${fastq_trim_dir}/${sample}_${r}.fq.gz"
            if [ -f "$src" ]; then
                mv -f "$src" "$dst"
            elif [ ! -f "$dst" ]; then
                echo "Error: trim_galore output missing for ${sample} R${r}: expected $src" >&2
                exit 1
            fi
        done
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished trimming adaptor and indexing on $sample."
    ) &
    pids+=($!)
    while [[ $(jobs -r -p | wc -l) -ge $max_io_jobs ]]; do
        sleep 1
    done
done
wait_all "${pids[@]}"
fastq_dir=${fastq_trim_dir}


# Generate initial STAR index (approximately 2 hours and 10 minutes)
if star_index_complete "${work_dir}/star_genome_index"; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] STAR first index already exists, skipping."
else
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Indexing genome with STAR (First Indexing)..."
    rm -rf "${work_dir}/star_genome_index" && mkdir -p "${work_dir}/star_genome_index"
    STAR \
        --runMode genomeGenerate \
        --genomeDir "${work_dir}/star_genome_index" \
        --genomeFastaFiles "${genome_fa}" \
        --sjdbOverhang $((read_length - trim_length - 1)) \
        --sjdbGTFfile "${anot_gtf}" \
        --runThreadN "${thread}" \
        > "${log_dir}/index_1.log" 2>&1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished indexing genome with STAR (First Indexing)..."
fi


# First alignment (approximately 5 minutes per sample)
if [[ ${fastq_suffix} == *gz ]]; then
    read_method="zcat"
else
    read_method="cat"
fi
> ${align_dir}/combined_SJ.out.tab

for i in ${fastq_dir}/*_1.${fastq_suffix}; do
    sample=$(basename "$i")
    sample=${sample%"_1.$fastq_suffix"}
    # Check if input files exist
    if [ ! -f "${fastq_dir}/${sample}_1.${fastq_suffix}" ] || [ ! -f "${fastq_dir}/${sample}_2.${fastq_suffix}" ]; then
        echo "Error: Missing FASTQ files for sample ${sample}" >&2
        continue
    fi
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] First Aligning sample: $sample..."
    STAR \
        --genomeDir ${work_dir}/star_genome_index \
        --readFilesIn ${fastq_dir}/${sample}_1.${fastq_suffix} ${fastq_dir}/${sample}_2.${fastq_suffix} \
        --runThreadN ${thread} \
        --outFileNamePrefix ${tmp_dir}/${sample}. \
        --outFilterMultimapScoreRange 1 \
        --outFilterMultimapNmax 20 \
        --outFilterMismatchNmax 10 \
        --alignIntronMax 500000 \
        --alignMatesGapMax 1000000 \
        --sjdbScore 2 \
        --alignSJDBoverhangMin 1 \
        --genomeLoad NoSharedMemory \
        --readFilesCommand $read_method \
        --outFilterMatchNminOverLread 0.33 \
        --outFilterScoreMinOverLread 0.33 \
        --sjdbOverhang $((read_length - trim_length - 1)) \
        --outSAMstrandField intronMotif \
        --outSAMtype None \
        --outSAMmode None \
        > ${log_dir}/alignment_1_${sample}.log 2>&1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished first Aligning sample: $sample..."
    # Merge SJ files
    cat ${tmp_dir}/${sample}.SJ.out.tab >> ${align_dir}/combined_SJ.out.tab
done


# Filter and sort SJ files. Keep all annotated junctions ($6==1) plus novel junctions with enough unique reads.
awk -v min_SJ_reads="$min_SJ_reads" '$6 == 1 || $7 >= min_SJ_reads' "${align_dir}/combined_SJ.out.tab" \
    > "${align_dir}/filtered_combined_SJ.out.tab"
LC_ALL=C sort -k1,1 -k2,2n -k3,3n "${align_dir}/filtered_combined_SJ.out.tab" | uniq > "${align_dir}/unique_combined_SJ.out.tab"

if [ ! -s "${align_dir}/unique_combined_SJ.out.tab" ]; then
    echo "Error: unique_combined_SJ.out.tab is empty — first-pass alignment produced no usable junctions." >&2
    exit 1
fi


# Second STAR indexing (approximately 2 hours and 10 minutes)
if star_index_complete "${work_dir}/star_index"; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] STAR second index already exists, skipping."
else
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Indexing genome with STAR (Second Indexing)..."
    rm -rf "${work_dir}/star_index" && mkdir -p "${work_dir}/star_index"
    STAR \
        --runMode genomeGenerate \
        --genomeDir "${work_dir}/star_index" \
        --genomeFastaFiles "$genome_fa" \
        --sjdbOverhang $((read_length - trim_length - 1)) \
        --runThreadN "$thread" \
        --sjdbFileChrStartEnd "${align_dir}/unique_combined_SJ.out.tab" \
        > "${log_dir}/index_2.log" 2>&1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished indexing genome with STAR (Second Indexing)..."
fi


# Second alignment (approximately 10 minutes per sample)
for i in ${fastq_dir}/*_1.${fastq_suffix}; do
    sample=$(basename "$i")
    sample=${sample%"_1.$fastq_suffix"}
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Second Aligning sample: $sample..."
    STAR \
        --genomeDir ${work_dir}/star_index \
        --readFilesIn ${fastq_dir}/${sample}_1.${fastq_suffix} ${fastq_dir}/${sample}_2.${fastq_suffix} \
        --runThreadN ${thread} \
        --outFileNamePrefix ${align_dir}/${sample}. \
        --sjdbGTFfile ${anot_gtf} \
        --outFilterMultimapScoreRange 1 \
        --outFilterMultimapNmax 20 \
        --outFilterMismatchNmax 10 \
        --alignIntronMax 500000 \
        --alignMatesGapMax 1000000 \
        --sjdbScore 2 \
        --genomeLoad NoSharedMemory \
        --limitBAMsortRAM 30000000000 \
        --readFilesCommand $read_method \
        --outFilterMatchNminOverLread 0.33 \
        --outFilterScoreMinOverLread 0.33 \
        --sjdbOverhang $((read_length - trim_length - 1)) \
        --outSAMstrandField intronMotif \
        --outSAMattributes NH HI NM MD AS XS \
        --outSAMunmapped Within \
        --outSAMtype BAM SortedByCoordinate \
        --outSAMheaderHD @HD VN:1.4 \
        --outSAMattrRGline ID:$sample SM:$sample PL:ILLUMINA LB:Lib_$sample \
        --quantMode GeneCounts TranscriptomeSAM \
        --chimSegmentMin 12 \
        --chimJunctionOverhangMin 12 \
        --chimOutType Junctions WithinBAM SoftClip \
        --chimMainSegmentMultNmax 1 \
        --alignSJstitchMismatchNmax 5 -1 5 5 \
        --alignSJDBoverhangMin 10 \
        --chimSegmentReadGapMax 3 \
        > ${log_dir}/alignment_2_${sample}.log 2>&1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished second Aligning sample: $sample..."
done


# RSEM expression calculation. Check all index files so a half-finished run is rebuilt.
rsem_index_ok=1
for ext in .seq .grp .ti .chrlist .transcripts.fa .idx.fa .n2g.idx.fa; do
    [ -f "${rsem_reference}${ext}" ] || rsem_index_ok=0
done
if (( rsem_index_ok == 0 )); then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Indexing genome with RSEM ..."
    rsem-prepare-reference -p "$thread" --gtf "${anot_gtf}" "${genome_fa}" "${rsem_reference}" \
        > "${log_dir}/rsem_index.log" 2>&1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished RSEM indexing."
else
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] RSEM reference already exists, skipping indexing."
fi

rsem_thread=$(( thread / max_io_jobs ))
(( rsem_thread < 1 )) && rsem_thread=1
pids=()
for i in "${fastq_dir}"/*_1.${fastq_suffix}; do
    (
        sample=$(basename "$i")
        sample=${sample%"_1.$fastq_suffix"}
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Calculating expression level of sample: $sample..."
        rsem-calculate-expression -p "$rsem_thread" --calc-ci --bam --paired-end \
            "${align_dir}/${sample}.Aligned.toTranscriptome.out.bam" \
            "${rsem_reference}" \
            "${expr_dir}/${sample}" \
            > "${log_dir}/rsem_${sample}.log" 2>&1
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished calculating expression level for $sample."
    ) &
    pids+=($!)
    while [[ $(jobs -r -p | wc -l) -ge $max_io_jobs ]]; do
        sleep 1
    done
done
wait_all "${pids[@]}"


rm -rf ${tmp_dir}
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Alignment and expression level calculation complete!"
