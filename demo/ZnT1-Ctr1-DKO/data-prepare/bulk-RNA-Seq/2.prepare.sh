#!/bin/bash
set -euo pipefail

# Basic parameter definitions
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${work_dir}/log/2.prepare.sh"
mkdir -p "$LOG_DIR"
cd $work_dir

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting data preparation..."
exec > >(tee -a "$LOG_DIR/prepare.log") 2>&1

# Download mouse GRCm39 genome and annotation from GENCODE vM36
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M36/GRCm39.primary_assembly.genome.fa.gz
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M36/gencode.vM36.annotation.gtf.gz

# Check integrity (min 10MB expected)
if [ $(stat -c%s "GRCm39.primary_assembly.genome.fa.gz") -lt 10000000 ] || [ $(stat -c%s "gencode.vM36.annotation.gtf.gz") -lt 10000000 ]; then
    echo "Error: Download failed or files are corrupted."
    exit 1
fi

gzip -d GRCm39.primary_assembly.genome.fa.gz
gzip -d gencode.vM36.annotation.gtf.gz
