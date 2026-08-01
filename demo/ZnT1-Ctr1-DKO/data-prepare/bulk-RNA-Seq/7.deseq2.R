# ─── Auto-check and install missing R packages ─────────────────────────────
required_bioc <- c("DESeq2", "edgeR", "apeglm", "tximport")
required_cran <- c("dplyr", "readr", "ggplot2", "ggrepel", "ggVennDiagram", "BiocManager")

for (pkg in required_cran) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}
for (pkg in required_bioc) {
  if (!requireNamespace(pkg, quietly = TRUE)) BiocManager::install(pkg, ask = FALSE)
}

library(DESeq2)
library(edgeR)
library(tximport)
library(apeglm)
library(dplyr)
library(readr)
library(ggplot2)
library(ggrepel)
library(ggVennDiagram)

# Path settings
work_dir <- "./"
expr_dir  <- file.path(work_dir, "expr")
deg_dir   <- file.path(work_dir, "deg")
log_dir   <- file.path(work_dir, "log/7.deseq2.R")
dir.create(deg_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

# Significance thresholds
FDR_CUTOFF   <- 0.05
LOGFC_CUTOFF <- 1

# Redirect stdout to log file; ensure sink closes even on error
sink(file.path(log_dir, "deseq2.log"), append = FALSE, split = TRUE)
on.exit(try(sink(), silent = TRUE), add = TRUE)

# ─── After clustering analysis, add sample names to exclude here ───
exclude_samples <- c()  # e.g., c("WT_83", "DKO_67")

# 1. Prepare sample info and file paths
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
if (length(expr_files) == 0) stop("No samples left after exclusion; check exclude_samples.")

# Use tximport to load RSEM data (recommended approach)
cat("Loading RSEM data via tximport...\n")
txi <- tximport(expr_files, type = "rsem", txIn = FALSE, txOut = FALSE)

# 2. Build ColData
group_names <- sub("_[0-9]+$", "", sample_names)
stopifnot(all(group_names %in% c("WT", "DKO")))
colData <- data.frame(
  condition = factor(group_names, levels = c("WT", "DKO"))
)
rownames(colData) <- sample_names

cat("Group info:\n")
print(table(colData$condition))

# 3. Differential expression analysis (DESeq2)
dds <- DESeqDataSetFromTximport(txi,
                                colData = colData,
                                design = ~ condition)

# Unified pre-filtering logic (using edgeR::filterByExpr to be consistent with 6.edgr.R)
# Build a temporary DGEList so filterByExpr uses the same lib.size/group semantics as edgeR script
design_tmp <- model.matrix(~condition, data = colData)
dge_tmp <- edgeR::DGEList(counts = txi$counts, group = colData$condition)
keep <- edgeR::filterByExpr(dge_tmp, design = design_tmp)
dds <- dds[keep, ]
cat("Genes after filtering:", nrow(dds), "\n")

# Run differential analysis
dds <- DESeq(dds)

# Get differential results and perform LFC Shrinkage (apeglm method recommended)
res_names <- resultsNames(dds)
coef_name <- res_names[grep("condition_DKO_vs_WT", res_names)]
if (length(coef_name) > 1) {
  warning("Multiple coefficient names match 'condition_DKO_vs_WT'; using the first: ", coef_name[1])
}

if (length(coef_name) > 0) {
  cat("Performing LFC Shrinkage (apeglm)... \n")
  res <- lfcShrink(dds, coef = coef_name[1], type = "apeglm")
} else {
  cat("Warning: No matching coefficient name found, skipping LFC Shrinkage.\n")
  res <- results(dds, contrast = c("condition", "DKO", "WT"))
}

res_df <- as.data.frame(res)
res_df$gene_id <- rownames(res_df)
res_df <- res_df %>% select(gene_id, everything())

# Save all DESeq2 results
write_csv(res_df, file.path(deg_dir, "DKO_vs_WT_DESeq2_DEG.csv"))

# Filter significant DEGs (padj < FDR_CUTOFF & |log2FC| > LOGFC_CUTOFF)
sig_deseq2 <- res_df %>%
  filter(!is.na(padj) & padj < FDR_CUTOFF & abs(log2FoldChange) > LOGFC_CUTOFF)
write_csv(sig_deseq2, file.path(deg_dir, "DKO_vs_WT_significant_DESeq2_DEG.csv"))

sig_up   <- sig_deseq2 %>% filter(log2FoldChange >  LOGFC_CUTOFF)
sig_down <- sig_deseq2 %>% filter(log2FoldChange < -LOGFC_CUTOFF)

cat(sprintf("DESeq2 (DKO_vs_WT): Total genes: %d | Significant DEGs: %d (Up: %d, Down: %d)\n",
            nrow(res_df), nrow(sig_deseq2), nrow(sig_up), nrow(sig_down)))

# ─────────────────────────────────────────────
# Diagnostic plots: PCA + Dispersion estimates + MA
# ─────────────────────────────────────────────
pdf(file.path(deg_dir, "DESeq2_diagnostics.pdf"), width = 8, height = 6)
on.exit(try(dev.off(), silent = TRUE), add = TRUE)

vsd <- tryCatch(vst(dds, blind = FALSE), error = function(e) {
  cat("vst() failed (likely too few samples); falling back to varianceStabilizingTransformation.\n")
  tryCatch(varianceStabilizingTransformation(dds, blind = FALSE), error = function(e2) NULL)
})
if (!is.null(vsd)) {
  print(plotPCA(vsd, intgroup = "condition") +
          labs(title = "PCA (VST-transformed counts)") +
          theme_bw() +
          theme(plot.title = element_text(hjust = 0.5)))
} else {
  cat("Skipping PCA plot — variance stabilization failed.\n")
}
plotDispEsts(dds, main = "Dispersion estimates")
plotMA(res, main = "MA plot (DKO vs WT, shrunken LFC)", ylim = c(-5, 5))
dev.off()

# ─────────────────────────────────────────────
# Volcano Plot
# ─────────────────────────────────────────────
volcano_df <- res_df %>%
  filter(!is.na(padj)) %>%
  mutate(
    neg_log10_padj = -log10(padj),
    significance = case_when(
      padj < FDR_CUTOFF & log2FoldChange >  LOGFC_CUTOFF ~ "Up",
      padj < FDR_CUTOFF & log2FoldChange < -LOGFC_CUTOFF ~ "Down",
      TRUE                                               ~ "NS"
    )
  )

top_genes <- volcano_df %>%
  filter(significance != "NS") %>%
  arrange(padj) %>%
  head(20)

p_volcano <- ggplot(volcano_df, aes(x = log2FoldChange, y = neg_log10_padj, color = significance)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("Up" = "#E63946", "Down" = "#457B9D", "NS" = "grey70")) +
  geom_vline(xintercept = c(-LOGFC_CUTOFF, LOGFC_CUTOFF), linetype = "dashed", color = "grey40") +
  geom_hline(yintercept = -log10(FDR_CUTOFF), linetype = "dashed", color = "grey40") +
  geom_text_repel(data = top_genes, aes(label = gene_id), size = 2.5, max.overlaps = 30) +
  labs(
    title = "DKO vs WT — Volcano Plot (DESeq2)",
    x = "log2 Fold Change (shrunken)",
    y = "-log10(padj)",
    color = "Significance"
  ) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave(file.path(deg_dir, "volcano_DKO_vs_WT_DESeq2.pdf"), p_volcano, width = 10, height = 8)
ggsave(file.path(deg_dir, "volcano_DKO_vs_WT_DESeq2.jpg"), p_volcano, width = 10, height = 8, dpi = 300)

# 4. Compare with edgeR results (Venn Diagram)
edger_file <- file.path(deg_dir, "DKO_vs_WT_significant_DEG.csv")
if (file.exists(edger_file)) {
  sig_edger <- read_csv(edger_file, show_col_types = FALSE)

  venn_list <- list(
    edgeR  = sig_edger$gene_id,
    DESeq2 = sig_deseq2$gene_id
  )

  p_venn <- ggVennDiagram(venn_list) +
    scale_fill_gradient(low = "white", high = "#E63946") +
    labs(title = "Overlap of Significant DEGs: edgeR vs DESeq2") +
    theme(plot.title = element_text(hjust = 0.5))

  ggsave(file.path(deg_dir, "edger_vs_deseq2_venn.pdf"), p_venn, width = 8, height = 6)
  ggsave(file.path(deg_dir, "edger_vs_deseq2_venn.jpg"), p_venn, width = 8, height = 6, dpi = 300)

  # Save intersect genes as "high confidence" genes
  intersect_genes <- intersect(sig_edger$gene_id, sig_deseq2$gene_id)
  cat(sprintf("Intersect gene count: %d\n", length(intersect_genes)))

  sig_intersect <- sig_deseq2 %>% filter(gene_id %in% intersect_genes)
  write_csv(sig_intersect, file.path(deg_dir, "DKO_vs_WT_high_confidence_DEG.csv"))
} else {
  cat("edgeR result file not found; skipping Venn comparison.\n")
}

cat("\nDESeq2 analysis complete! Results saved to:", deg_dir, "\n")
# sink() is closed by the on.exit handler registered earlier
