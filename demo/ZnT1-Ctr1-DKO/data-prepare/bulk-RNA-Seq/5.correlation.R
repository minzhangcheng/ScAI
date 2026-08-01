#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# ─── Auto-check and install missing R packages ─────────────────────────────
required_cran <- c("readr", "dplyr", "tidyr", "pheatmap", "ggplot2", "ggrepel")

for (pkg in required_cran) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}

library(readr)
library(dplyr)
library(tidyr)
library(pheatmap)
library(ggplot2)
library(ggrepel)

# --- 1. Set paths ---
work_dir <- "./"
INPUT_DIR  <- "./expr"
OUTPUT_DIR <- "./correlation"
LOG_DIR    <- "./log/5.correlation.R"
FILE_SUFFIX <- ".genes.results"

# Ensure output directory exists
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

# Redirect stdout and stderr to log file
sink(file.path(LOG_DIR, "correlation.log"), append = FALSE, split = TRUE)

# Group colors
GROUP_COLORS <- c("WT" = "#457B9D", "DKO" = "#E63946")

cat("Starting analysis, input directory:", INPUT_DIR, "\n")
cat("Output directory:", OUTPUT_DIR, "\n\n")

# --- 2. Read and merge all sample data ---
file_pattern <- file.path(INPUT_DIR, paste0("*", FILE_SUFFIX))
all_files <- sort(Sys.glob(file_pattern))

if (length(all_files) < 2) {
  stop("Error: Need at least 2 *", FILE_SUFFIX, " files in ", INPUT_DIR, " (found ", length(all_files), ")")
}

cat("Found", length(all_files), "sample files:\n")
all_sample_dfs <- list()
sample_names <- character()

for (file_path in all_files) {
  sample_name <- sub(FILE_SUFFIX, "", basename(file_path))
  sample_names <- c(sample_names, sample_name)
  cat("  - Reading:", sample_name, "\n")

  df <- read_tsv(file_path, col_types = cols(), show_col_types = FALSE)
  df <- df %>% select(gene_id, TPM) %>% filter(!is.na(gene_id)) %>% distinct(gene_id, .keep_all = TRUE)
  colnames(df)[2] <- sample_name
  all_sample_dfs[[sample_name]] <- df
}

cat("\n...Merging expression profiles for all samples...\n")
expression_matrix <- Reduce(function(x, y) full_join(x, y, by = "gene_id"), all_sample_dfs)
expression_matrix[is.na(expression_matrix)] <- 0

# Filter: keep only genes with TPM > 1 in at least one sample
expression_matrix <- expression_matrix %>% filter(rowSums(select(., -gene_id) > 1) >= 1)

cat("Merged expression matrix (before log transform) dimensions:", nrow(expression_matrix), "x", ncol(expression_matrix) - 1, "(genes x samples)\n")

# --- 3. Extract group information ---
groups <- sub("_[0-9]+$", "", sample_names)
group_colors_vec <- GROUP_COLORS[groups]

# --- 4. Data transformation: Log2(TPM + 1) ---
expr_data <- expression_matrix %>% select(-gene_id)
log_transformed <- log2(expr_data + 1)
rownames(log_transformed) <- expression_matrix$gene_id

cat("Log2(TPM + 1) transformation complete.\n")

# --- 5. Calculate Pearson correlation matrix and save ---
cat("\n--- Calculating Pearson correlation ---\n")
corr_matrix <- cor(log_transformed, method = "pearson")

corr_matrix_path <- file.path(OUTPUT_DIR, "pearson_correlation_matrix.csv")
write.csv(corr_matrix, corr_matrix_path, quote = FALSE)
cat("Correlation matrix saved to:", corr_matrix_path, "\n")

# --- 6. Draw correlation heatmap ---
# Order columns by sample name
corr_matrix_ordered <- corr_matrix[sample_names, sample_names]

# Clustering annotations
annotation_col <- data.frame(Group = groups, row.names = sample_names)
annotation_colors <- list(Group = GROUP_COLORS)

# Helper: run base-R plotting expression to both PDF and JPG
save_plot <- function(base_path, width, height, expr) {
  pdf(paste0(base_path, ".pdf"), width = width, height = height)
  eval(expr)
  dev.off()
  jpeg(paste0(base_path, ".jpg"), width = width * 100, height = height * 100,
       quality = 100, res = 100)
  eval(expr)
  dev.off()
}

# Dynamic breaks: lower bound = min(corr), upper = 1. Avoids collapsing low correlations to one color.
heatmap_breaks <- seq(min(corr_matrix_ordered), 1, length.out = 100)

heatmap_base <- file.path(OUTPUT_DIR, "sample_correlation_heatmap")
save_plot(heatmap_base, 12, 10, quote({
  pheatmap(
    corr_matrix_ordered,
    annotation_col = annotation_col,
    annotation_colors = annotation_colors,
    annotation_legend = FALSE,
    show_annotation_names = FALSE,
    display_numbers = TRUE,
    number_format = "%.2f",
    fontsize_number = 8,
    color = colorRampPalette(c("#457B9D", "white", "#E63946"))(100),
    breaks = heatmap_breaks,
    treeheight_col = 30,
    treeheight_row = 30,
    main = "Sample Correlation Heatmap (Pearson, Log2(TPM+1))"
  )
}))
cat("Correlation heatmap saved to:", paste0(heatmap_base, ".{pdf,jpg}"), "\n")

# --- 7. Hierarchical clustering dendrogram ---
cat("\n--- Drawing hierarchical clustering dendrogram ---\n")
distance_matrix <- as.dist(1 - corr_matrix)
hc <- hclust(distance_matrix, method = "average")
dendro <- as.dendrogram(hc)

dendro_base <- file.path(OUTPUT_DIR, "hierarchical_clustering_dendrogram")
save_plot(dendro_base, 12, 6, quote({
  par(mar = c(8, 5, 3, 3))
  plot(dendro,
       xlab = "Sample",
       ylab = "Distance (1 - r)",
       main = "Hierarchical Clustering Dendrogram (1 - Pearson Correlation)",
       cex.main = 1.3,
       cex.lab = 1.1,
       cex.axis = 0.9,
       las = 2)
}))
cat("Hierarchical clustering dendrogram saved to:", paste0(dendro_base, ".{pdf,jpg}"), "\n")

# --- 8. Draw PCA plot (colored by group) ---
cat("\n--- Calculating and drawing PCA ---\n")
pca_result <- prcomp(t(log_transformed), scale. = TRUE)
pc_scores <- as.data.frame(pca_result$x)
pc_scores$sample <- rownames(pc_scores)
pc_scores$group <- groups
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

pca_plot <- ggplot(pc_scores, aes(x = PC1, y = PC2, color = group, label = sample)) +
  geom_point(size = 8, alpha = 0.7, stroke = 1) +
  scale_color_manual(values = GROUP_COLORS) +
  geom_text_repel(size = 3, max.overlaps = 20) +
  labs(
    title = "PCA of Samples (Log2(TPM+1))",
    x = sprintf("PC1 (%.2f%%)", explained_variance[1] * 100),
    y = sprintf("PC2 (%.2f%%)", explained_variance[2] * 100),
    color = "Group"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(OUTPUT_DIR, "pca_plot.jpg"), pca_plot, width = 12, height = 10, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "pca_plot.pdf"), pca_plot, width = 12, height = 10)
cat("PCA plot saved to:", file.path(OUTPUT_DIR, "pca_plot.jpg"), "\n")

# --- 9. Draw MDS plot (colored by group) ---
cat("\n--- Calculating and drawing MDS ---\n")
mds_result <- cmdscale(distance_matrix, k = 2, eig = TRUE)
mds_scores <- as.data.frame(mds_result$points)
colnames(mds_scores) <- c("Dim1", "Dim2")
mds_scores$sample <- rownames(mds_scores)
mds_scores$group <- groups

mds_plot <- ggplot(mds_scores, aes(x = Dim1, y = Dim2, color = group, label = sample)) +
  geom_point(size = 8, alpha = 0.7, stroke = 1) +
  scale_color_manual(values = GROUP_COLORS) +
  geom_text_repel(size = 3, max.overlaps = 20) +
  labs(
    title = "MDS Plot (Distance = 1 - Pearson Correlation)",
    x = "Dimension 1",
    y = "Dimension 2",
    color = "Group"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(OUTPUT_DIR, "mds_plot.jpg"), mds_plot, width = 12, height = 10, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "mds_plot.pdf"), mds_plot, width = 12, height = 10)
cat("MDS plot saved to:", file.path(OUTPUT_DIR, "mds_plot.jpg"), "\n")

cat("\n--- Analysis complete! ---\n")

# Close log connection
sink()
