#!/bin/bash
set -euo pipefail

# Basic parameter definitions
work_dir="$(cd "${WORK_DIR:-.}" && pwd -P)"
LOG_DIR="${work_dir}/log/2.prepare.sh"
mkdir -p "$LOG_DIR"
cd "$work_dir"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting data preparation..."
exec > >(tee -a "$LOG_DIR/prepare.log") 2>&1

# Download mouse GRCm39 genome and annotation from GENCODE vM39 only when the
# corresponding decompressed file is absent. Checksum verification is deferred.
if [ ! -s "GRCm39.primary_assembly.genome.fa" ]; then
    wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M39/GRCm39.primary_assembly.genome.fa.gz
fi
if [ ! -s "gencode.vM39.annotation.gtf" ]; then
    wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M39/gencode.vM39.annotation.gtf.gz
fi

# Check integrity (min 10MB expected)
if { [ ! -s "GRCm39.primary_assembly.genome.fa" ] && \
     { [ ! -s "GRCm39.primary_assembly.genome.fa.gz" ] || [ "$(stat -c%s "GRCm39.primary_assembly.genome.fa.gz")" -lt 10000000 ]; }; } || \
   { [ ! -s "gencode.vM39.annotation.gtf" ] && \
     { [ ! -s "gencode.vM39.annotation.gtf.gz" ] || [ "$(stat -c%s "gencode.vM39.annotation.gtf.gz")" -lt 10000000 ]; }; }; then
    echo "Error: Download failed or files are corrupted."
    exit 1
fi

[ -s "GRCm39.primary_assembly.genome.fa" ] || gzip -d GRCm39.primary_assembly.genome.fa.gz
[ -s "gencode.vM39.annotation.gtf" ] || gzip -d gencode.vM39.annotation.gtf.gz
