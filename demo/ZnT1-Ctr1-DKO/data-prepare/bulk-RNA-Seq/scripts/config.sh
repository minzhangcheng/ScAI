#!/bin/bash
# =============================================================================
# config.sh — RNA-Seq Pipeline Configuration
# =============================================================================
# Centralized parameter definitions. Source this file from any pipeline step.
# All paths are relative to the pipeline working directory.
# =============================================================================

# --- Core paths -------------------------------------------------------------
WORK_DIR="$(cd "${WORK_DIR:-.}" && pwd -P)"
GENOME_FA="${WORK_DIR}/GRCm39.primary_assembly.genome.fa"
ANNOT_GTF="${WORK_DIR}/gencode.vM39.annotation.gtf"

# --- Input / Output directories ---------------------------------------------
FASTQ_DIR="${WORK_DIR}/fastq"
FASTQ_MERGED_DIR="${WORK_DIR}/fastq_merged"
FASTQC_DIR="${WORK_DIR}/fastqc"
FASTQ_TRIM_DIR="${WORK_DIR}/fastq_trim"
ALIGN_DIR="${WORK_DIR}/align"
EXPR_DIR="${WORK_DIR}/expr"
STAR_GENOME_INDEX="${WORK_DIR}/star_genome_index"
STAR_INDEX="${WORK_DIR}/star_index"
RSEM_REF="${WORK_DIR}/rsem_reference/rsem_ref"
RSEQC_DIR="${WORK_DIR}/rseqc"
MULTIQC_DIR="${WORK_DIR}/multiqc"
LOG_DIR="${WORK_DIR}/log"
CONFIG_DIR="${WORK_DIR}/config"

# --- Computational parameters -----------------------------------------------
THREADS="${THREADS:-32}"
READ_LENGTH="${READ_LENGTH:-150}"
TRIM_LENGTH="${TRIM_LENGTH:-10}"
SJDB_OVERHANG=$(( READ_LENGTH - TRIM_LENGTH - 1 ))

# --- Parallel job limits ----------------------------------------------------
MAX_FASTQC_JOBS="${MAX_FASTQC_JOBS:-4}"
MAX_TRIM_JOBS="${MAX_TRIM_JOBS:-2}"
MAX_RSEM_JOBS="${MAX_RSEM_JOBS:-8}"

# --- Quality thresholds -----------------------------------------------------
MIN_SJ_READS="${MIN_SJ_READS:-5}"      # Min unique reads for novel SJ retention

# --- Sample naming ----------------------------------------------------------
# Pattern A: {id}-{group}_1.fq.gz  (e.g. 4-Ctrl_1.fq.gz → WT_4)
# Pattern B: {group}-{id}_L{lane}_UDI{xxx}.R1.fastq.gz (e.g. DKO-790_L1_UDI075.R1.fastq.gz → DKO_790)

# --- Logging ----------------------------------------------------------------
TIMESTAMP_FORMAT="+%Y-%m-%d %H:%M:%S"
