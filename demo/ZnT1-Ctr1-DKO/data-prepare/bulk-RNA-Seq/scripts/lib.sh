#!/bin/bash
# =============================================================================
# lib.sh — RNA-Seq Pipeline Shared Functions
# =============================================================================
# Source this file after config.sh to get shared utilities.
# =============================================================================

set -uo pipefail

# --- Logging -----------------------------------------------------------------
log_info() {
    echo "[$(date "${TIMESTAMP_FORMAT:-+%Y-%m-%d %H:%M:%S}")] INFO: $*"
}

log_warn() {
    echo "[$(date "${TIMESTAMP_FORMAT:-+%Y-%m-%d %H:%M:%S}")] WARN: $*" >&2
}

log_error() {
    echo "[$(date "${TIMESTAMP_FORMAT:-+%Y-%m-%d %H:%M:%S}")] ERROR: $*" >&2
}

# --- Error handling ----------------------------------------------------------
die() {
    log_error "$*"
    exit 1
}

# --- Directory creation ------------------------------------------------------
ensure_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir" || die "Failed to create directory: $dir"
    fi
}

# --- Conda environment check -----------------------------------------------
check_conda_env() {
    local env_name="$1"
    if ! conda info --envs 2>/dev/null | grep -q "^${env_name} "; then
        die "Conda environment '${env_name}' not found. Run: conda env list"
    fi
}

# --- Tool availability check -------------------------------------------------
check_tool() {
    local tool="$1"
    command -v "$tool" >/dev/null 2>&1 || die "Required tool not found: $tool"
}

# --- Wait for all background jobs, fail if any exit non-zero -----------------
wait_all() {
    local pid fail=0
    for pid in "$@"; do
        wait "$pid" || fail=$((fail + 1))
    done
    if (( fail > 0 )); then
        die "$fail parallel job(s) failed"
    fi
}

# --- SHA256 signature for caching --------------------------------------------
make_signature() {
    local tool_version="$1"; shift
    {
        printf '%s\n' "$tool_version"
        printf '%s\n' "$@"
        sha256sum "$@" 2>/dev/null || cat "$@" 2>/dev/null
    } | sha256sum | awk '{print $1}'
}

# --- Check if STAR index is complete -----------------------------------------
star_index_complete() {
    local dir="$1" expected_sig="$2"
    local f
    for f in SA SAindex Genome chrLength.txt chrName.txt chrStart.txt genomeParameters.txt; do
        [[ -f "${dir}/${f}" ]] || return 1
    done
    [[ -f "${dir}/.pipeline_signature" ]] || return 1
    [[ "$(<"${dir}/.pipeline_signature")" == "$expected_sig" ]]
}

# --- Check if RSEM index is complete -----------------------------------------
rsem_index_complete() {
    local ref="$1" expected_sig="$2"
    local ext rsem_ok=1
    for ext in .seq .grp .ti .chrlist .transcripts.fa .idx.fa .n2g.idx.fa; do
        [[ -f "${ref}${ext}" ]] || rsem_ok=0
    done
    [[ -f "${ref}.pipeline_signature" ]] || rsem_ok=0
    if (( rsem_ok == 1 )) && [[ "$(<"${ref}.pipeline_signature")" != "$expected_sig" ]]; then
        rsem_ok=0
    fi
    (( rsem_ok == 1 ))
}

# --- Safe symlink creation ---------------------------------------------------
link_one() {
    local src="$1" dst="$2"
    [[ -e "$src" ]] || die "Source not found: $src"
    src=$(realpath "$src")
    if [[ -L "$dst" ]] && [[ "$(readlink "$dst")" != "$src" ]]; then
        die "Symlink conflict: $dst already points elsewhere, refusing to overwrite"
    fi
    ln -sf "$src" "$dst"
}

# --- Sample discovery and renaming -------------------------------------------
# Discovers FASTQ pairs from FASTQ_DIR and creates symlinks in FASTQ_MERGED_DIR
# with unified naming: {Group}_{ID}_1.fq.gz / {Group}_{ID}_2.fq.gz
# Ctrl → WT
# ---------------------------------------------------------------------------
discover_and_link_samples() {
    local fastq_dir="$1" merged_dir="$2"
    ensure_dir "$merged_dir"

    log_info "Discovering samples from $fastq_dir ..."

    # Pattern A: {id}-{group}_1.fq.gz
    local fq fname id group
    for fq in "${fastq_dir}"/[0-9]*-*_1.fq.gz; do
        [[ -e "$fq" ]] || continue
        fname=$(basename "$fq")
        id=$(echo "$fname" | sed 's/-.*//')
        group=$(echo "$fname" | sed 's/^[0-9]*-//;s/_1\.fq\.gz//')
        group=${group/Ctrl/WT}
        link_one "$fq" "${merged_dir}/${group}_${id}_1.fq.gz"
        link_one "${fq/_1.fq.gz/_2.fq.gz}" "${merged_dir}/${group}_${id}_2.fq.gz"
    done

    # Pattern B: {group}-{id}_L{lane}_UDI{xxx}.R1.fastq.gz
    # Multiple lanes for same sample are concatenated
    declare -A lane_r1_by_sample=()
    for fq in "${fastq_dir}"/*.R1.fastq.gz; do
        [[ -e "$fq" ]] || continue
        fname=$(basename "$fq")
        group=$(echo "$fname" | sed 's/-.*//')
        id=$(echo "$fname" | sed 's/^[^-]*-//;s/_L[0-9][0-9]*_.*//')
        group=${group/Ctrl/WT}
        local sample="${group}_${id}"
        lane_r1_by_sample["$sample"]+="${fq}"$'\n'
    done

    local sample mate_fq
    for sample in "${!lane_r1_by_sample[@]}"; do
        local -a r1_files=() r2_files=()
        while IFS= read -r fq; do
            [[ -n "$fq" ]] || continue
            mate_fq=${fq/.R1.fastq.gz/.R2.fastq.gz}
            [[ -f "$mate_fq" ]] || die "Missing mate for $fq: $mate_fq"
            r1_files+=("$fq")
            r2_files+=("$mate_fq")
        done <<< "${lane_r1_by_sample[$sample]}"

        if (( ${#r1_files[@]} == 1 )); then
            link_one "${r1_files[0]}" "${merged_dir}/${sample}_1.fq.gz"
            link_one "${r2_files[0]}" "${merged_dir}/${sample}_2.fq.gz"
        else
            log_info "Merging ${#r1_files[@]} lanes for ${sample}"
            cat "${r1_files[@]}" > "${merged_dir}/${sample}_1.fq.gz"
            cat "${r2_files[@]}" > "${merged_dir}/${sample}_2.fq.gz"
        fi
    done

    # Report discovered samples
    local samples=()
    for fq in "${merged_dir}"/*_1.fq.gz; do
        [[ -e "$fq" ]] || continue
        samples+=("$(basename "$fq" | sed 's/_1\.fq\.gz//')")
    done

    if (( ${#samples[@]} == 0 )); then
        die "No samples discovered in $fastq_dir"
    fi

    log_info "Discovered ${#samples[@]} sample(s): ${samples[*]}"
}
