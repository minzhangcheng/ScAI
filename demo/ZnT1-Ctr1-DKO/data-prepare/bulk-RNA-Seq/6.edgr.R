# ─── Auto-check and install missing R packages ─────────────────────────────
required_bioc <- c("edgeR", "tximport")
required_cran <- c("dplyr", "readr", "ggplot2", "ggrepel", "BiocManager")

for (pkg in required_cran) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}
for (pkg in required_bioc) {
  if (!requireNamespace(pkg, quietly = TRUE)) BiocManager::install(pkg, ask = FALSE)
}

library(dplyr)
library(readr)
library(edgeR)
library(tximport)
library(ggplot2)
library(ggrepel)

# Path settings
work_dir <- "./"
expr_dir  <- file.path(work_dir, "expr")
deg_dir   <- file.path(work_dir, "deg")
log_dir   <- file.path(work_dir, "log/6.edgr.R")
dir.create(deg_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

# Significance thresholds
FDR_CUTOFF    <- 0.05
LOGFC_CUTOFF  <- 1

# Redirect stdout to log file; ensure sink closes even on error
sink(file.path(log_dir, "edgr.log"), append = FALSE, split = TRUE)
on.exit(try(sink(), silent = TRUE), add = TRUE)

# ─── After clustering analysis, add sample names to exclude here ───
exclude_samples <- c()  # e.g., c("WT_83", "DKO_67")

# Read RSEM expression files
expr_files <- list.files(expr_dir, pattern = "\\.genes\\.results$", full.names = TRUE)
sample_names <- sub("\\.genes\\.results$", "", basename(expr_files))
names(expr_files) <- sample_names
cat("Detected samples:", paste(sample_names, collapse = ", "), "\n")

# Exclude specified samples
if (length(exclude_samples) > 0) {
  keep_idx <- !(sample_names %in% exclude_samples)
  expr_files <- expr_files[keep_idx]
  sample_names <- sample_names[keep_idx]
  cat("After exclusion:", paste(sample_names, collapse = ", "), "\n")
}

# Load RSEM data via tximport (recommended; provides counts + effective length)
cat("Loading RSEM data via tximport...\n")
txi <- tximport(expr_files, type = "rsem", txIn = FALSE, txOut = FALSE)

# Group info: extract group name from sample name (WT_3 -> WT, DKO_790 -> DKO)
group_names <- sub("_[0-9]+$", "", sample_names)
group <- factor(group_names, levels = c("WT", "DKO"))
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)
rownames(design) <- sample_names

cat("Group info:\n")
print(table(group))

# Build DGEList with gene-length offset (tximport + edgeR recommended workflow)
cts <- txi$counts
normMat <- txi$length
# Guard against zero effective lengths
normMat[normMat == 0] <- 1
# Obtain per-observation scaling factors for length, adjusted to avoid changing the magnitude of the counts
normMat <- normMat / exp(rowMeans(log(normMat)))
normCts <- cts / normMat
# Effective library sizes to account for composition biases
eff_lib <- calcNormFactors(normCts) * colSums(normCts)
# Combine length factors with effective library sizes into log-scale offset
normMat <- sweep(normMat, 2, eff_lib, "*")
normMat <- log(normMat)

dge <- DGEList(cts, group = group)
dge <- scaleOffset(dge, normMat)

# Low expression filtering (using edgeR recommended filterByExpr)
keep <- filterByExpr(dge, design)
dge  <- dge[keep, , keep.lib.sizes = FALSE]
cat("Genes after filtering:", nrow(dge), "\n")

# Estimate dispersion (offset already scaled; no need for calcNormFactors)
dge <- estimateDisp(dge, design)
bcv <- sqrt(dge$common.dispersion)
cat("Common dispersion:", dge$common.dispersion, "\n")
cat("BCV:", bcv, "\n\n")
if (bcv > 0.4) {
  warning(sprintf("High BCV (%.3f) — check sample quality or biological variability.", bcv))
}

# Save dispersion parameters
disp_df <- data.frame(
  CommonDisp = dge$common.dispersion,
  BCV = bcv
)
write_csv(disp_df, file.path(deg_dir, "edgeR_dispersion_parameters.csv"))

# Diagnostic plots: MDS and BCV
pdf(file.path(deg_dir, "edgeR_diagnostics.pdf"), width = 8, height = 6)
plotMDS(dge, col = as.integer(group), main = "MDS plot")
legend("topright", legend = levels(group), col = seq_along(levels(group)), pch = 16)
plotBCV(dge, main = "BCV plot")
dev.off()

# Fit GLM
fit <- glmFit(dge, design)

# Differential expression analysis: DKO vs WT
contrast <- makeContrasts(DKO - WT, levels = design)
lrt    <- glmLRT(fit, contrast = contrast)
result <- topTags(lrt, n = nrow(dge))$table
result <- cbind(gene_id = rownames(result), result)

# Save all results
all_out <- file.path(deg_dir, "DKO_vs_WT_DEG.csv")
write_csv(result, all_out)

# Filter significant DEGs
sig <- result %>% filter(FDR < FDR_CUTOFF, abs(logFC) > LOGFC_CUTOFF)
sig_out <- file.path(deg_dir, "DKO_vs_WT_significant_DEG.csv")
write_csv(sig, sig_out)

sig_up   <- sig %>% filter(logFC >  LOGFC_CUTOFF)
sig_down <- sig %>% filter(logFC < -LOGFC_CUTOFF)

cat(sprintf("DKO_vs_WT: Total genes: %d | Significant DEGs: %d (Up: %d, Down: %d)\n",
            nrow(result), nrow(sig), nrow(sig_up), nrow(sig_down)))

# ─────────────────────────────────────────────
# Volcano Plot
# ─────────────────────────────────────────────
volcano_df <- result %>%
  mutate(
    neg_log10_FDR = -log10(FDR),
    significance = case_when(
      FDR < FDR_CUTOFF & logFC >  LOGFC_CUTOFF ~ "Up",
      FDR < FDR_CUTOFF & logFC < -LOGFC_CUTOFF ~ "Down",
      TRUE                                     ~ "NS"
    )
  )

# Label top 20 significant genes (sorted by FDR)
top_genes <- volcano_df %>%
  filter(significance != "NS") %>%
  arrange(FDR) %>%
  head(20)

p <- ggplot(volcano_df, aes(x = logFC, y = neg_log10_FDR, color = significance)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("Up" = "#E63946", "Down" = "#457B9D", "NS" = "grey70")) +
  geom_vline(xintercept = c(-LOGFC_CUTOFF, LOGFC_CUTOFF), linetype = "dashed", color = "grey40") +
  geom_hline(yintercept = -log10(FDR_CUTOFF), linetype = "dashed", color = "grey40") +
  geom_text_repel(data = top_genes, aes(label = gene_id), size = 2.5, max.overlaps = 30) +
  labs(
    title = "DKO vs WT — Volcano Plot",
    x = "log2 Fold Change",
    y = "-log10(FDR)",
    color = "Significance"
  ) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave(file.path(deg_dir, "volcano_DKO_vs_WT.pdf"), p, width = 10, height = 8)
ggsave(file.path(deg_dir, "volcano_DKO_vs_WT.jpg"), p, width = 10, height = 8, dpi = 300)

cat("\nAnalysis complete! Results saved to:", deg_dir, "\n")
# sink() is closed by the on.exit handler registered earlier
