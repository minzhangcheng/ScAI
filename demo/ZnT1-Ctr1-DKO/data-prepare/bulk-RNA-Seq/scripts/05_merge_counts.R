#!/usr/bin/env Rscript
# =============================================================================
# 05_merge_counts.R — Merge RSEM Results into Annotated Count Matrix
# =============================================================================
# Reads all expr/*.genes.results files, extracts expected_count,
# merges into a single matrix, and annotates with gene_name and gene_type
# from the GTF file.
#
# Usage: Rscript 05_merge_counts.R [work_dir]
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
work_dir <- if (length(args) >= 1) args[1] else "."
work_dir <- normalizePath(work_dir, mustWork = FALSE)
if (!dir.exists(work_dir)) {
  stop("Working directory does not exist: ", work_dir)
}

# --- Paths ------------------------------------------------------------------
EXPR_DIR     <- file.path(work_dir, "expr")
GTF_PATH     <- file.path(work_dir, "gencode.vM39.annotation.gtf")
OUTPUT_PATH  <- file.path(EXPR_DIR, "merged_counts.csv")
LOG_DIR      <- file.path(work_dir, "log", "05_merge_counts")

# Ensure log directory exists
if (!dir.exists(LOG_DIR)) {
  dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)
}

# Redirect output to log
LOG_FILE <- file.path(LOG_DIR, "merge_counts.log")
con <- file(LOG_FILE, open = "wt")
sink(con, split = TRUE)
sink(con, type = "message")
on.exit({
  try(sink(type = "message"), silent = TRUE)
  try(sink(), silent = TRUE)
  try(close(con), silent = TRUE)
}, add = TRUE)

cat("[", as.character(Sys.time()), "] Starting merge counts...\n")

# --- Validation -------------------------------------------------------------
stopifnot_msg <- function(cond, msg) {
  if (!cond) stop(msg, call. = FALSE)
}

stopifnot_msg(dir.exists(EXPR_DIR),
              paste("Expression directory not found:", EXPR_DIR))

stopifnot_msg(file.exists(GTF_PATH),
              paste("GTF annotation file not found:", GTF_PATH))

# --- Discover RSEM result files ---------------------------------------------
result_files <- sort(list.files(EXPR_DIR,
                                pattern = "\\.genes\\.results$",
                                full.names = TRUE))

stopifnot_msg(length(result_files) > 0,
              paste("No *.genes.results files found in", EXPR_DIR))

sample_names <- sub("\\.genes\\.results$", "", basename(result_files))

stopifnot_msg(!any(duplicated(sample_names)),
              paste("Duplicate sample names:",
                    paste(sample_names[duplicated(sample_names)], collapse = ", ")))

cat(sprintf("Found %d sample(s): %s\n",
            length(sample_names),
            paste(sample_names, collapse = ", ")))

# --- Parse GTF for gene annotations -----------------------------------------
cat("Parsing GTF annotation...\n")

gtf_lines <- readLines(GTF_PATH, warn = FALSE)
gene_lines <- gtf_lines[grepl("\\tgene\\t", gtf_lines)]

extract_attr <- function(lines, key) {
  pat <- paste0(key, " \"([^\"]+)\"")
  m <- regmatches(lines, regexec(pat, lines))
  vapply(m, function(x) if (length(x) >= 2) x[2] else NA_character_,
         character(1))
}

gene_map <- data.frame(
  gene_id   = extract_attr(gene_lines, "gene_id"),
  gene_name = extract_attr(gene_lines, "gene_name"),
  gene_type = extract_attr(gene_lines, "gene_type"),
  stringsAsFactors = FALSE
)

gene_map <- gene_map[!is.na(gene_map$gene_id), ]
gene_map <- gene_map[!duplicated(gene_map$gene_id), ]

stopifnot_msg(nrow(gene_map) > 0, "No valid gene entries parsed from GTF.")

cat(sprintf("Parsed %d gene annotations from GTF.\n", nrow(gene_map)))

# --- Read expected_count from each sample -----------------------------------
cat("Reading expected counts from RSEM results...\n")

read_counts <- function(path, sample) {
  df <- utils::read.delim(path, stringsAsFactors = FALSE,
                          check.names = FALSE, na.strings = "")

  stopifnot_msg(all(c("gene_id", "expected_count") %in% colnames(df)),
                paste("Missing required columns in", basename(path)))

  dup_ids <- df$gene_id[duplicated(df$gene_id)]
  stopifnot_msg(length(dup_ids) == 0,
                paste("Duplicate gene_ids in", basename(path)))

  df <- df[, c("gene_id", "expected_count")]
  colnames(df)[2] <- sample
  df
}

count_list <- Map(read_counts, result_files, sample_names)

# --- Merge all sample count matrices ----------------------------------------
cat("Merging count matrices...\n")

merged <- Reduce(function(x, y) merge(x, y, by = "gene_id", all = TRUE),
                 count_list)

na_count <- sum(is.na(merged[, -1]))
if (na_count > 0) {
  warning(sprintf("%d NA values in merged count matrix — replacing with 0.",
                   na_count))
  merged[, -1][is.na(merged[, -1])] <- 0
}

# --- Add gene annotations ---------------------------------------------------
cat("Adding gene name annotations...\n")

merged_annot <- merge(gene_map, merged, by = "gene_id", all.x = FALSE, all.y = TRUE)

# Reorder columns
merged_annot <- merged_annot[, c("gene_id", "gene_name", "gene_type",
                                  setdiff(colnames(merged_annot),
                                          c("gene_id", "gene_name", "gene_type")))]

# --- Validate output --------------------------------------------------------
total_genes   <- nrow(merged_annot)
total_samples <- ncol(merged_annot) - 3

cat(sprintf("Output matrix: %d genes x %d samples\n", total_genes, total_samples))

stopifnot_msg(total_genes > 10000,
              paste("Suspiciously few genes:", total_genes))

stopifnot_msg(all(merged_annot[, -c(1:3)] >= 0, na.rm = TRUE),
              "Negative count values detected.")

# --- Write output -----------------------------------------------------------
cat("Writing", OUTPUT_PATH, "...\n")

if (!dir.exists(EXPR_DIR)) {
  dir.create(EXPR_DIR, recursive = TRUE, showWarnings = FALSE)
}

utils::write.csv(merged_annot, OUTPUT_PATH, row.names = FALSE, quote = TRUE)

stopifnot_msg(file.exists(OUTPUT_PATH) && file.size(OUTPUT_PATH) > 0,
              paste("Failed to write output file:", OUTPUT_PATH))

cat("Done. Output:", OUTPUT_PATH, "\n")

# The on.exit handler will close sinks and connection
