# scripts/fetch_markers.py
"""下载 PanglaoDB + CellMarker 2.0，投影为固定 schema，原子写到 backend/data/markers/。

后端调用入口: fetch_and_write(dest_dir)（经 asyncio.to_thread 调用，勿走 shell）。
CLI: python scripts/fetch_markers.py
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_COLS = ["source", "species", "cell_type", "gene_symbol", "tissue", "evidence", "ref"]
_DEST_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "markers"
_SPECIES_MAP = {"Mm": "mouse", "mouse": "mouse", "Mus musculus": "mouse",
                "Hs": "human", "human": "human", "Homo sapiens": "human"}

# 上游来源（如失效需在此更新）
PANGLAODB_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
CELLMARKER2_URL = "http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_All.xlsx"


def project_to_schema(raw: pd.DataFrame, source: str, gene_col: str,
                      celltype_col: str, species_col: str,
                      tissue_col: str | None = None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["gene_symbol"] = raw[gene_col].astype(str).str.strip()
    out["cell_type"] = raw[celltype_col].astype(str).str.strip()
    out["species"] = raw[species_col].map(_SPECIES_MAP)
    out["tissue"] = raw[tissue_col].astype(str) if tissue_col and tissue_col in raw else ""
    out["source"] = source
    out["evidence"] = ""
    out["ref"] = ""
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
    return project_to_schema(raw, "PanglaoDB", gene_col="official gene symbol",
                             celltype_col="cell type", species_col="species")


def _download_cellmarker2() -> pd.DataFrame:
    raw = pd.read_excel(CELLMARKER2_URL)
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
                os.unlink(lock)
                return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
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
