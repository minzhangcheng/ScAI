#!/usr/bin/env bash
# Fetch the mm39 RepeatMasker annotation from UCSC and convert it to a GTF usable with velocyto -m.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_GTF="${BASE_DIR}/mm39_rmsk.gtf"
TMP_TSV="${BASE_DIR}/mm39_rmsk.tsv"

echo "[rmsk] Method A: querying mm39.rmsk via UCSC genome-mysql ..."
if ! mysql --skip-ssl --user=genome --host=genome-mysql.soe.ucsc.edu -A -D mm39 -N -B \
        -e "SELECT genoName, genoStart, genoEnd, strand, repName, repClass, repFamily, swScore
            FROM rmsk
            ORDER BY genoName, genoStart;" \
        > "${TMP_TSV}" 2>/tmp/rmsk_mysql.err; then
    echo "[rmsk] mysql failed: $(cat /tmp/rmsk_mysql.err)" >&2
    echo "[rmsk] Method B: falling back to HTTP download of rmsk.txt.gz ..."
    RMSK_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm39/database/rmsk.txt.gz"
    curl -fsSL "${RMSK_URL}" -o "${TMP_TSV}.gz"
    # rmsk.txt columns: bin swScore milliDiv milliDel milliIns genoName genoStart genoEnd
    #                   genoLeft strand repName repClass repFamily ...
    # Extract: genoName(6) genoStart(7) genoEnd(8) strand(10) repName(11) repClass(12) repFamily(13) swScore(2)
    zcat "${TMP_TSV}.gz" \
        | awk 'BEGIN{FS=OFS="\t"} {print $6,$7,$8,$10,$11,$12,$13,$2}' \
        | sort -k1,1 -k2,2n > "${TMP_TSV}"
    rm -f "${TMP_TSV}.gz"
fi

echo "[rmsk] Got $(wc -l < "${TMP_TSV}") records, converting to GTF ..."
awk 'BEGIN{FS=OFS="\t"}
{
    chrom=$1; start=$2+1; end=$3; strand=$4;
    repName=$5; repClass=$6; repFamily=$7; score=$8;
    # velocyto requires feature=exon with gene_id / transcript_id attributes
    attr="gene_id \"" repName "\"; transcript_id \"" repName "\"; family_id \"" repFamily "\"; class_id \"" repClass "\";";
    print chrom, "rmsk", "exon", start, end, score, strand, ".", attr;
}' "${TMP_TSV}" > "${OUT_GTF}"

rm -f "${TMP_TSV}"

echo "[rmsk] Done: ${OUT_GTF}"
echo "[rmsk] Lines: $(wc -l < "${OUT_GTF}")"
echo "[rmsk] Chromosomes covered:"
cut -f1 "${OUT_GTF}" | sort -u | head -30
