# scripts/fetch_markers.py
"""下载 PanglaoDB + CellMarker 2.0，投影为固定 schema，原子写到 backend/data/markers/。

后端调用入口: fetch_and_write(dest_dir)（经 asyncio.to_thread 调用，勿走 shell）。
CLI: python scripts/fetch_markers.py
"""
from __future__ import annotations
import hashlib
import os
import logging
from io import BytesIO
from pathlib import Path
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_COLS = ["source", "species", "cell_type", "gene_symbol", "tissue", "evidence", "ref"]
_DEST_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "markers"
_SPECIES_MAP = {"Mm": "mouse", "mouse": "mouse", "Mus musculus": "mouse",
                "Hs": "human", "human": "human", "Homo sapiens": "human"}
# Case-insensitive view: CellMarker2 ships "Mouse"/"Human", PanglaoDB "Mm"/"Hs".
_SPECIES_LOOKUP = {k.lower(): v for k, v in _SPECIES_MAP.items()}


def map_species_tokens(value) -> list[str]:
    """Map one raw species cell to the list of schema species it covers.

    PanglaoDB encodes a marker that applies to both organisms as the single
    value ``"Mm Hs"`` -- 7507 of its 8286 rows (90.6%). A plain ``.map()``
    against _SPECIES_MAP yields NaN for those and they were silently dropped,
    leaving only the 402 mouse-only + 344 human-only rows. Splitting the cell
    and emitting one row per species recovers them.

    Unrecognised tokens are skipped rather than failing the row, so a single
    upstream typo (PanglaoDB has 3 rows with species "4") cannot take out an
    otherwise valid dual-species marker.
    """
    raw = str(value).strip()
    direct = _SPECIES_LOOKUP.get(raw.lower())
    if direct:
        return [direct]
    out: list[str] = []
    for token in raw.replace(",", " ").split():
        mapped = _SPECIES_LOOKUP.get(token.lower())
        if mapped and mapped not in out:
            out.append(mapped)
    return out

# 上游来源（如失效需在此更新）
PANGLAODB_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
CELLMARKER2_URL = "https://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_All.xlsx"
# Integrity pin for Cell_marker_All.xlsx.
#   - Set to a real SHA-256 to fetch over the raw-IP endpoint (verify=False) with
#     content verified by the checksum (available AND integrity-checked).
#   - Leave None to fail-secure: the download requires a valid TLS chain
#     (verify=True); a raw IP without a valid cert will fail and CellMarker2 is
#     skipped rather than trusted unverified. Compute a digest once via:
#       curl -sk <CELLMARKER2_URL> | sha256sum
CELLMARKER2_SHA256 = None


def project_to_schema(raw: pd.DataFrame, source: str, gene_col: str,
                      celltype_col: str, species_col: str,
                      tissue_col: str | None = None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["gene_symbol"] = raw[gene_col].astype(str).str.strip()
    out["cell_type"] = raw[celltype_col].astype(str).str.strip()
    # One raw row may cover several species ("Mm Hs"); explode to one row each.
    out["species"] = raw[species_col].map(map_species_tokens)
    out["tissue"] = (raw[tissue_col].astype(str).str.strip()
                     if tissue_col and tissue_col in raw else "")
    out["source"] = source
    out["evidence"] = ""
    out["ref"] = ""
    out = out.explode("species", ignore_index=True)
    out = out[out["species"].isin(("mouse", "human"))]
    out = out[(out["gene_symbol"] != "") & (out["cell_type"] != "")]
    return out[SCHEMA_COLS].drop_duplicates().reset_index(drop=True)


def atomic_write_tsv(df: pd.DataFrame, dest: Path, source_version: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    from datetime import date
    with open(tmp, "w") as f:
        f.write(f"# SOURCE_VERSION={source_version} DOWNLOAD_DATE={date.today().isoformat()}\n")
        df.to_csv(f, sep="\t", index=False)
    os.replace(tmp, dest)  # 原子 rename，避免半截文件被读


def _download_panglaodb() -> pd.DataFrame:
    raw = pd.read_csv(PANGLAODB_URL, sep="\t", compression="gzip")
    # "organ" is PanglaoDB's tissue field; without it every row landed in the
    # table tissue-blind, so no tissue-restricted prior was ever possible.
    return project_to_schema(raw, "PanglaoDB", gene_col="official gene symbol",
                             celltype_col="cell type", species_col="species",
                             tissue_col="organ")


def _http_get_bytes(url: str, verify: bool = True) -> bytes:
    with httpx.Client(verify=verify, timeout=120, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def _download_cellmarker2() -> pd.DataFrame:
    # Security posture: never accept unverified data over the unauthenticated
    # raw-IP endpoint. Two safe modes:
    #   1. A SHA-256 is pinned  -> integrity is guaranteed by the checksum, so
    #      the untrusted transport (verify=False) is acceptable.
    #   2. No SHA-256 pinned    -> require a valid TLS chain (verify=True); if the
    #      endpoint has no valid certificate the fetch fails and CellMarker2 is
    #      skipped (fail-secure). Pin CELLMARKER2_SHA256 to enable it over raw IP.
    if CELLMARKER2_SHA256:
        content = _http_get_bytes(CELLMARKER2_URL, verify=False)
        digest = hashlib.sha256(content).hexdigest()
        if digest != CELLMARKER2_SHA256:
            raise ValueError(f"CellMarker2 checksum mismatch: {digest}")
    else:
        content = _http_get_bytes(CELLMARKER2_URL, verify=True)
    raw = pd.read_excel(BytesIO(content))
    return project_to_schema(raw, "CellMarker2", gene_col="Symbol",
                             celltype_col="cell_name", species_col="species",
                             tissue_col="tissue_type")


_STALE_LOCK_SECONDS = 1800  # 30 分钟后视锁为陈旧（进程崩溃残留）


def _acquire_lock(lock: Path):
    """O_CREAT|O_EXCL 文件锁；陈旧（>30min）则夺锁。返回 fd 或 None。"""
    import time
    try:
        return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
            if age > _STALE_LOCK_SECONDS:
                logger.warning(f"stale marker fetch lock ({age:.0f}s); reclaiming")
                lock_tmp = lock.with_suffix(lock.suffix + ".tmp")
                fd = os.open(lock_tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.replace(lock_tmp, lock)
                return fd
        except OSError:
            pass
        return None


def fetch_and_write(dest_dir: Path | None = None) -> bool:
    """下载并原子写两表。带文件锁防并发（陈旧锁自动夺取）。失败返回 False（不抛）。

    注：CellMarker2 为 .xlsx，pd.read_excel 需 openpyxl；若缺则该源失败但 PanglaoDB 仍写（补充源失败不阻断）。
    """
    dest = Path(dest_dir) if dest_dir else _DEST_DIR
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / ".fetch.lock"
    fd = _acquire_lock(lock)
    if fd is None:
        logger.warning("marker fetch already in progress (lock present); skipping")
        return False
    try:
        atomic_write_tsv(_download_panglaodb(), dest / "panglaodb.tsv", "PanglaoDB_27_Mar_2020")
        try:
            atomic_write_tsv(_download_cellmarker2(), dest / "cellmarker2.tsv", "CellMarker2.0")
        except Exception as e:
            logger.warning(f"CellMarker2 fetch failed (PanglaoDB still written): {e}")
        return True
    except Exception as e:
        logger.warning(f"marker fetch failed: {e}")
        return False
    finally:
        os.close(fd)
        try:
            os.unlink(lock)
        except OSError:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = fetch_and_write()
    raise SystemExit(0 if ok else 1)
