# scripts/fetch_markers.py
"""下载 PanglaoDB + CellMarker 3.0，投影为固定 schema，原子写到运行时数据目录。

目标目录是 ``$SCAI_DATA_DIR/markers``（见 default_dest_dir），**不是**代码目录:
这些表是缓存，放在代码树里会被容器镜像固化，而"文件已存在就不重下"意味着镜像
里那一份会比后续任何投影修复都活得久。

写出的每张表首行带 ``SCHEMA_REVISION``，由 ``backend/services/marker_priors.py``
校验；revision 不符即视为陈旧并重下。

后端调用入口: fetch_and_write(dest_dir)（经 asyncio.to_thread 调用，勿走 shell）。
CLI: python scripts/fetch_markers.py

数据来源与许可（fetch 时会把 LICENSE 一并写到目标目录）:
  PanglaoDB      主源       CC BY-NC 4.0  https://panglaodb.se/
  CellMarker 3.0 补稀有/亚型  见官网       https://bio-bigdata.hrbmu.edu.cn/CellMarker/
覆盖 human / mouse。仅离线非商业研究用途；商用需自行复核许可。
补充源失败只 warning，不阻断主源写出。
"""
from __future__ import annotations
import os
import logging
from io import BytesIO
from pathlib import Path
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_COLS = ["source", "species", "cell_type", "gene_symbol", "tissue", "evidence", "ref"]

#: Every table this script is expected to leave behind, primary source first.
#: The "already current, skip the network" short-circuit below is keyed on all of
#: them. Keying it on panglaodb.tsv alone meant that the first fetch which
#: succeeded for the primary and failed for the supplementary (whose failure is
#: only a warning) stamped panglaodb.tsv at the current revision and thereby
#: made cellmarker.tsv unfetchable forever — every later call returned True
#: without a request, so a deployment ran on PanglaoDB-only priors permanently.
EXPECTED_TABLES = ("panglaodb.tsv", "cellmarker.tsv")

#: Stamped into every table written here and validated by
#: ``marker_priors.MARKER_SCHEMA_REVISION`` — the two must be bumped together.
#: 2 = dual-species rows expanded + PanglaoDB's ``organ`` carried as ``tissue``.
#: 3 = CellMarker reachable at last (authenticated domain), so the supplementary
#:     source is present for the first time. Revision 2 tables are PanglaoDB-only
#:     and must be re-fetched to gain it — which is exactly what a bump buys.
SCHEMA_REVISION = 3

#: Written next to the tables so the attribution travels with the data rather
#: than living in the code tree the data is deliberately kept out of.
DATA_LICENSE_TEXT = (
    "PanglaoDB: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).\n"
    "CellMarker 3.0: see https://bio-bigdata.hrbmu.edu.cn/CellMarker/ for terms.\n"
    "Downloaded for offline non-commercial research use only; not redistributed.\n"
)

_SPECIES_MAP = {"Mm": "mouse", "mouse": "mouse", "Mus musculus": "mouse",
                "Hs": "human", "human": "human", "Homo sapiens": "human"}
# Case-insensitive view: CellMarker ships "Mouse"/"Human", PanglaoDB "Mm"/"Hs".
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


def default_dest_dir() -> Path:
    """``$SCAI_DATA_DIR/markers``, or ``<repo>/../data/markers`` when unset.

    Mirrors ``backend.config.DATA_BASE_DIR`` exactly — this module must stay
    importable without the backend package (it is loaded by path from both the
    CLI and ``marker_priors.trigger_fetch``), so the rule is duplicated rather
    than imported, and a test pins the two together.
    """
    repo_root = Path(__file__).resolve().parent.parent
    base = Path(os.environ.get(
        "SCAI_DATA_DIR", str((repo_root.parent / "data").resolve())
    )).expanduser().resolve()
    return base / "markers"


# 上游来源（如失效需在此更新）
PANGLAODB_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
#: panglaodb.se rejects requests whose User-Agent looks like a scraper (a plain
#: curl UA gets 403 "we don't permit curl access anymore, b/c bots"). Identify
#: the client honestly and explicitly instead of inheriting whatever UA the
#: HTTP library of the day happens to send.
USER_AGENT = "ScAI/marker-fetch (single-cell analysis platform; offline ORA priors)"
#: CellMarker's own domain, which presents a valid TLS chain. The previous URL
#: was a raw IP (``https://117.50.127.228/...``) that could not be authenticated,
#: so the fetch had to choose between a pinned checksum and refusing to run —
#: unpinned, it refused, and the supplementary source was never once available.
#: On the real domain ``verify=True`` both authenticates and works.
#:
#: Upstream is now CellMarker 3.0 and ships ``.zip``-wrapped TSV rather than
#: ``.xlsx``. ``single_cell_marker`` is the single-cell-derived subset — the
#: relevant one for cluster annotation, and 12 MB against 71 MB for the full
#: table, whose uncompressed 811 MB buys only bulk/antibody-derived rows.
CELLMARKER_URL = "https://bio-bigdata.hrbmu.edu.cn/CellMarker/file/single_cell_marker.zip"
#: Columns actually projected. The file carries 22 (pmid, journal, method
#: details, ...); reading four keeps a 148 MB TSV to a manageable frame.
CELLMARKER_COLS = ["species", "cell_name", "symbol", "tissue_type"]


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
        f.write(f"# SOURCE_VERSION={source_version} SCHEMA_REVISION={SCHEMA_REVISION} "
                f"DOWNLOAD_DATE={date.today().isoformat()}\n")
        df.to_csv(f, sep="\t", index=False)
    os.replace(tmp, dest)  # 原子 rename，避免半截文件被读


def _download_panglaodb() -> pd.DataFrame:
    # Read the bytes ourselves rather than handing the URL to pandas: pandas
    # would send urllib's UA, and this host 403s UAs it reads as scrapers.
    raw = pd.read_csv(BytesIO(_http_get_bytes(PANGLAODB_URL)),
                      sep="\t", compression="gzip")
    # "organ" is PanglaoDB's tissue field; without it every row landed in the
    # table tissue-blind, so no tissue-restricted prior was ever possible.
    return project_to_schema(raw, "PanglaoDB", gene_col="official gene symbol",
                             celltype_col="cell type", species_col="species",
                             tissue_col="organ")


def _http_get_bytes(url: str, verify: bool = True) -> bytes:
    with httpx.Client(verify=verify, timeout=120, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def _download_cellmarker() -> pd.DataFrame:
    # verify=True unconditionally: the source is now an authenticated domain, so
    # there is no availability/integrity trade-off left to make. Never relax this
    # to verify=False -- an unauthenticated marker table steers cell annotation.
    content = _http_get_bytes(CELLMARKER_URL, verify=True)
    raw = pd.read_csv(BytesIO(content), sep="\t", compression="zip",
                      usecols=CELLMARKER_COLS, dtype=str)
    return project_to_schema(raw, "CellMarker", gene_col="symbol",
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


def is_current_revision(path: Path) -> bool:
    """True when ``path`` carries this module's SCHEMA_REVISION in its header."""
    try:
        # readline, not read_text: the table is multi-MB and the header is one
        # line, so slurping the file to reach it costs the whole table.
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return False
    if not first.startswith("#"):
        return False
    return f"SCHEMA_REVISION={SCHEMA_REVISION}" in first.lstrip("#").split()


def fetch_and_write(dest_dir: Path | None = None, force: bool = False) -> bool:
    """Download and atomically write both tables into dest_dir.

    Defaults to ``default_dest_dir()``. File-locked against concurrent runs, a
    stale lock being taken over. Returns False on failure rather than raising.

    With ``force=False`` the network is skipped only when EVERY table in
    ``EXPECTED_TABLES`` already carries the current SCHEMA_REVISION, so the
    startup script can call this unconditionally. Keying the skip on the primary
    table alone made a supplementary source that failed to download permanently
    unrecoverable: the primary stayed current, so every later call returned
    before reaching the network. Anything at the wrong revision -- a table
    written by an older projection, or one absent because its last download
    failed -- is fetched again, and only that table is.

    CellMarker is supplementary: its failure only warns, and PanglaoDB is still
    written, so annotation keeps the primary source's priors.
    """
    dest = Path(dest_dir) if dest_dir else default_dest_dir()
    dest.mkdir(parents=True, exist_ok=True)
    # Per-table, not per-run: a primary that is already current must not be
    # re-downloaded just because the supplementary source is missing, and a
    # missing supplementary source must not be masked by a current primary.
    stale = [name for name in EXPECTED_TABLES
             if force or not is_current_revision(dest / name)]
    if not stale:
        logger.info(f"marker DB already at schema revision {SCHEMA_REVISION}; skipping fetch")
        return True
    lock = dest / ".fetch.lock"
    fd = _acquire_lock(lock)
    if fd is None:
        logger.warning("marker fetch already in progress (lock present); skipping")
        return False
    try:
        if "panglaodb.tsv" in stale:
            atomic_write_tsv(_download_panglaodb(), dest / "panglaodb.tsv", "PanglaoDB_27_Mar_2020")
        # Attribution travels with the data, which lives outside the code tree.
        # Rewritten on every fetch pass, supplementary-only ones included, so a
        # directory that gains cellmarker.tsv later still carries both notices.
        (dest / "LICENSE").write_text(DATA_LICENSE_TEXT)
        if "cellmarker.tsv" in stale:
            try:
                atomic_write_tsv(_download_cellmarker(), dest / "cellmarker.tsv", "CellMarker3.0")
            except Exception as e:
                logger.warning(f"CellMarker fetch failed (PanglaoDB still written): {e}")
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
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=None,
                    help="target directory (default: $SCAI_DATA_DIR/markers)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the table is already current")
    ns = ap.parse_args()
    dest = ns.dest or default_dest_dir()
    logger.info(f"marker DB destination: {dest}")
    raise SystemExit(0 if fetch_and_write(dest, force=ns.force) else 1)
