# ─── Auto-check and install missing R packages ─────────────────────────────
required_bioc <- c("rtracklayer")
required_cran <- c("dplyr", "readr", "BiocManager")

for (pkg in required_cran) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}
for (pkg in required_bioc) {
  if (!requireNamespace(pkg, quietly = TRUE)) BiocManager::install(pkg, ask = FALSE)
}

library(rtracklayer)
library(dplyr)
library(readr)

# Correct execution order:
# 2.prepare.sh → 3.align.sh → 5.correlation.R (clustering QC) → 6.edgr.R → 8.annot.R → 9.enrichment.R

work_dir <- "./"
expr_dir <- file.path(work_dir, "expr")
deg_dir  <- file.path(work_dir, "deg")
log_dir  <- file.path(work_dir, "log/8.annot.R")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

# Redirect stdout to log file; ensure sink closes even on error
sink(file.path(log_dir, "annot.log"), append = FALSE, split = TRUE)
on.exit(try(sink(), silent = TRUE), add = TRUE)

# ─── 1. Parse GTF, get gene_id → gene_name mapping ───
gtf_path <- file.path(work_dir, "gencode.vM36.annotation.gtf")
if (!file.exists(gtf_path)) stop("GTF file not found: ", gtf_path)
cat("Parsing GTF file...\n")
gtf <- import(gtf_path)
gene_map <- as.data.frame(gtf[gtf$type == "gene"]) %>%
  select(gene_id, gene_name, gene_type)
cat(sprintf("Parsed %d genes\n\n", nrow(gene_map)))

# ─── 2. Merge all sample expression matrices ───
if (!dir.exists(expr_dir)) stop("Expression directory not found: ", expr_dir)
expr_files <- list.files(expr_dir, pattern = "\\.genes\\.results$", full.names = TRUE)
if (length(expr_files) == 0) stop("No .genes.results files found in ", expr_dir)
sample_names <- sub("\\.genes\\.results$", "", basename(expr_files))
cat("Detected samples:", paste(sample_names, collapse = ", "), "\n")

# Read TPM and expected_count for each sample
read_expr <- function(f, col) {
  df <- read_tsv(f, col_types = cols(), show_col_types = FALSE)
  if (any(duplicated(df$gene_id))) {
    warning(sprintf("Duplicate gene_ids in %s — keeping first occurrence", basename(f)))
  }
  df %>% select(gene_id, all_of(col)) %>% distinct(gene_id, .keep_all = TRUE)
}

# Merge TPM
tpm_list <- lapply(seq_along(expr_files), function(i) {
  df <- read_expr(expr_files[i], "TPM")
  colnames(df)[2] <- sample_names[i]
  df
})
tpm_matrix <- Reduce(function(x, y) full_join(x, y, by = "gene_id"), tpm_list)
tpm_na <- sum(is.na(tpm_matrix))
if (tpm_na > 0) cat(sprintf("Warning: %d NAs in TPM matrix replaced with 0 (gene_id inconsistency across samples)\n", tpm_na))
tpm_matrix[is.na(tpm_matrix)] <- 0

# Merge counts
count_list <- lapply(seq_along(expr_files), function(i) {
  df <- read_expr(expr_files[i], "expected_count")
  colnames(df)[2] <- sample_names[i]
  df
})
count_matrix <- Reduce(function(x, y) full_join(x, y, by = "gene_id"), count_list)
count_na <- sum(is.na(count_matrix))
if (count_na > 0) cat(sprintf("Warning: %d NAs in count matrix replaced with 0 (gene_id inconsistency across samples)\n", count_na))
count_matrix[is.na(count_matrix)] <- 0

# Add gene_name and output
annotate_and_save <- function(mat, out_path) {
  mat <- left_join(mat, gene_map, by = "gene_id") %>%
    select(gene_id, gene_name, gene_type, everything())
  write_csv(mat, out_path)
  cat(sprintf("Saved: %s (%d genes x %d samples)\n",
              out_path, nrow(mat), ncol(mat) - 3))
}

annotate_and_save(tpm_matrix, file.path(expr_dir, "merged_TPM.csv"))
annotate_and_save(count_matrix, file.path(expr_dir, "merged_counts.csv"))

# ─── 3. Add gene_name annotation to DEG results ───
cat("\nAnnotating DEG files...\n")
if (!dir.exists(deg_dir)) stop("DEG directory not found: ", deg_dir, " (run 6.edgr.R first)")
deg_files <- list.files(deg_dir, pattern = "DEG\\.csv$", full.names = TRUE)
if (length(deg_files) == 0) stop("No DEG.csv files found in ", deg_dir)

for (f in deg_files) {
  deg <- read_csv(f, col_types = cols(), show_col_types = FALSE)
  deg <- left_join(deg, gene_map, by = "gene_id") %>%
    select(gene_id, gene_name, gene_type, everything())
  out_path <- sub("\\.csv$", "_annot.csv", f)
  write_csv(deg, out_path)
  cat(sprintf("  %s → %s (%d genes)\n", basename(f), basename(out_path), nrow(deg)))
}

cat("\nAnnotation complete!\n")
# sink() is closed by the on.exit handler registered earlier
