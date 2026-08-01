#!/bin/bash
set -euo pipefail

#==============================================================
# RNA-Seq Pipeline Environment Setup Script
# Creates a Conda environment and installs all dependencies
#==============================================================

ENV_NAME="rna-seq"
LOG_DIR="./log/1.install.sh"
mkdir -p "$LOG_DIR"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting environment setup..."
exec > >(tee -a "$LOG_DIR/install.log") 2>&1

# Check if mamba is installed; if so, use it for faster installation
CONDA_CMD="conda"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Found mamba, using it for faster package installation."
fi

# 1. Create environment and install core bioinformatics tools and main R packages
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Creating Conda environment: $ENV_NAME ..."
$CONDA_CMD create -y -n "$ENV_NAME" \
    -c conda-forge -c bioconda \
    fastqc \
    trim-galore \
    cutadapt \
    star \
    rsem \
    samtools \
    multiqc \
    rseqc \
    ucsc-gtftogenepred \
    ucsc-genepredtobed \
    rmats \
    star-fusion \
    r-base=4.3 \
    r-dplyr \
    r-readr \
    r-tidyr \
    r-ggplot2 \
    r-pheatmap \
    r-ggrepel \
    r-ggdendro \
    r-ggvenndiagram \
    r-msigdbr \
    r-wgcna \
    r-igraph \
    r-ggraph \
    r-remotes \
    bioconductor-tximport \
    bioconductor-deseq2 \
    bioconductor-clusterprofiler \
    bioconductor-org.mm.eg.db \
    bioconductor-enrichplot \
    bioconductor-limma \
    bioconductor-edger \
    bioconductor-rtracklayer \
    bioconductor-pathview \
    bioconductor-decoupler \
    bioconductor-biomart \
    bioconductor-gsva \
    bioconductor-hdf5array \
    bioconductor-rhdf5 \
    bioconductor-rhdf5filters \
    r-xml \
    r-xml2 \
    r-rcurl \
    r-fs \
    libuv \
    xz \
    pkg-config

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Primary tools and packages successfully installed via Conda."

# 2. Activate environment and install remaining R packages (e.g., immunedeconv)
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Installing extra R packages via GitHub..."

# Use conda run to execute Rscript in the environment
# Use Rscript to install immunedeconv (if conda failed to install it)
conda run -n "$ENV_NAME" Rscript -e '
if (!requireNamespace("immunedeconv", quietly = TRUE)) {
    message("Installing immunedeconv from GitHub...")
    remotes::install_github("icbi-lab/immunedeconv")
} else {
    message("immunedeconv already installed.")
}
'

echo "=========================================================="
echo "Environment installation complete!"
echo "To use this environment, please run the following command:"
echo "    conda activate $ENV_NAME"
echo "=========================================================="
