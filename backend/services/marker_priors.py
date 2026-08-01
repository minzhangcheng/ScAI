"""离线 marker 数据库加载 / 物种筛选 / 合并去重 / 写 workspace。仅数据，无统计。"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .gene_set_resolver import normalize_organism

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["source", "species", "cell_type", "gene_symbol", "tissue", "evidence", "ref"]
_TSV_NAMES = ("panglaodb.tsv", "cellmarker.tsv")  # PanglaoDB 优先

#: Projection revision the loader accepts. ``fetch_markers.py`` stamps it into
#: the header of every table it writes; a table carrying any other value is
#: ignored here and re-fetched.
#:
#: This exists because the tables are a *cache* whose correct contents changed.
#: Revision 1 mapped PanglaoDB's species column with a plain lookup, so the
#: single value ``"Mm Hs"`` -- 7507 of the file's 8286 rows -- became NaN and was
#: dropped, leaving 746 rows (~9%) with an empty tissue column. Revision 2
#: expands dual-species rows and carries ``organ`` as ``tissue``. Revision 3 adds
#: the CellMarker table, which had never once been fetched because its only
#: endpoint was an unauthenticatable raw IP. Bump this whenever a fix changes what
#: a correct table holds, or deployments that already downloaded once will keep
#: serving the old projection forever.
MARKER_SCHEMA_REVISION = 3


def normalize_species(organism: str) -> str:
    """One alias table governs the whole platform; see gene_set_resolver."""
    return normalize_organism(organism)


def runtime_markers_dir() -> Path:
    """Runtime home of the marker TSVs: ``$SCAI_DATA_DIR/markers``.

    Deliberately outside the code tree. When these lived in ``backend/data/``
    they were captured by the container image, and since nothing re-downloads a
    file that already exists, an image shipped one frozen table to every
    deployment built from it. On the data volume the tables are what they are --
    a cache -- and are fetched by whichever release is actually running.
    """
    from backend.config import DATA_BASE_DIR
    return Path(DATA_BASE_DIR) / "markers"


def parse_table_header(path: Path) -> Dict[str, str]:
    """Parse the leading ``# KEY=VALUE ...`` line written by fetch_markers.

    Returns ``{}`` for a missing, unreadable or header-less file so callers can
    treat "no metadata" and "wrong metadata" alike.
    """
    try:
        # readline, not read_text: these tables are multi-MB and the header is
        # one line, so slurping the file to reach it costs the whole table.
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
    except Exception:
        return {}
    if not first.startswith("#"):
        return {}
    out: Dict[str, str] = {}
    for token in first.lstrip("#").split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def _header_is_current(header: Dict[str, str]) -> bool:
    return header.get("SCHEMA_REVISION") == str(MARKER_SCHEMA_REVISION)


def _is_current_revision(path: Path) -> bool:
    return _header_is_current(parse_table_header(path))


#: Separator for the multi-valued tissue field. ``ora_kernel`` splits on it.
TISSUE_SEP = ";"


def _join_tissues(values) -> str:
    """Union the organs a (cell type, gene) pair is annotated in.

    Taking ``"first"`` here -- what the dedup used to do -- discarded all but one
    organ per pair. CellMarker annotates the same marker across many tissues, so
    that collapsed e.g. CD8 T cells to a single arbitrary organ and made the
    kernel's tissue check answer "not in this tissue" for cell types that are.
    """
    seen = {}
    for value in values:
        if value is None:
            continue
        for part in str(value).split(TISSUE_SEP):
            part = part.strip()
            # str() on a missing pandas value yields the literal 'nan'; it is not
            # an organ, and neither is a stringified None.
            if part and part.lower() not in ('nan', 'none'):
                seen[part] = None
    return TISSUE_SEP.join(seen)


class MarkerDB:
    def __init__(self, markers_dir: Optional[Path] = None):
        self.markers_dir = Path(markers_dir) if markers_dir else runtime_markers_dir()
        self._versions: Dict[str, str] = {}
        self._by_species: Dict[str, List[dict]] = {}
        self._loaded = False

    def usable(self) -> bool:
        """True only when a table written by the CURRENT projection is on disk.

        Mere existence is not enough — see MARKER_SCHEMA_REVISION. Callers use
        this to decide whether to fetch, so a stale table must read as absent.
        """
        return any(p.exists() and _is_current_revision(p)
                   for p in (self.markers_dir / n for n in _TSV_NAMES))

    def missing_tables(self) -> List[str]:
        """Expected tables absent or stale at the current projection revision.

        ``usable`` asks whether the DB can annotate at all; this asks whether it
        is running on the sources this revision was cut for. The two differ
        exactly when one download succeeded and another failed, and nothing
        downstream can tell: annotation simply proceeds on a narrower evidence
        base than intended, with no signal anywhere.
        """
        return [n for n in _TSV_NAMES
                if not ((self.markers_dir / n).exists()
                        and _is_current_revision(self.markers_dir / n))]

    def _load(self) -> None:
        if self._loaded:
            return
        frames = []
        for name in _TSV_NAMES:
            path = self.markers_dir / name
            if not path.exists():
                continue
            header = parse_table_header(path)
            if not _header_is_current(header):
                logger.warning(
                    "ignoring stale %s (SCHEMA_REVISION=%s, expected %s); "
                    "re-fetch required before DB priors are available",
                    path, header.get("SCHEMA_REVISION", "absent"),
                    MARKER_SCHEMA_REVISION)
                continue
            self._versions[name.replace(".tsv", "")] = header.get(
                "SOURCE_VERSION", "unknown")
            df = pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{path} missing columns {missing}")
            frames.append(df[REQUIRED_COLS])
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged["species"] = merged["species"].map(normalize_species)
            key = ["species", "cell_type", "gene_symbol"]
            agg = (merged.groupby(key, sort=False)
                          .agg({"source": lambda s: "+".join(dict.fromkeys(s)),
                                "tissue": _join_tissues,
                                "evidence": "first", "ref": "first"})
                          .reset_index())
            for species, grp in agg.groupby("species"):
                self._by_species[species] = grp.to_dict("records")
        self._loaded = True

    def source_version(self, source_key: str) -> str:
        self._load()
        return self._versions.get(source_key, "unknown")

    def is_species_supported(self, organism: str) -> bool:
        self._load()
        return normalize_species(organism) in self._by_species

    def markers_for(self, organism: str) -> List[dict]:
        self._load()
        return self._by_species.get(normalize_species(organism), [])

    def reload(self) -> None:
        """Clear cached data so next access re-reads from disk.
        Called after trigger_fetch() downloads fresh TSV files."""
        self._loaded = False
        self._versions = {}
        self._by_species = {}

    def write_workspace_input(self, organism: str, workspace_dir) -> Optional[Path]:
        """写筛选后的 cell_type/gene_symbol/source 到 workspace/tables/。物种不支持返回 None。"""
        rows = self.markers_for(organism)
        if not rows:
            return None
        tables = Path(workspace_dir) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        out = tables / "marker_priors_input.tsv"
        # tissue travels through so the kernel can label each prior with the
        # organ its cell type came from. Adding CellMarker took the candidate
        # pool from 178 cell types to ~1000, many of them fine-grained entries
        # from unrelated organs whose small gene sets score well on two hits; an
        # organ label lets the reader discount those instead of being misled.
        df = pd.DataFrame(rows)[["cell_type", "gene_symbol", "source", "tissue"]]
        df.to_csv(out, sep="\t", index=False)
        return out


_fetch_mod = None


def trigger_fetch(force: bool = False) -> bool:
    """同步下载 marker DB 到 runtime_markers_dir()（供 asyncio.to_thread 调用）。

    失败返回 False，不抛。``force`` 绕过"已是当前 revision 则跳过"的短路，供
    HITL 重试路径重下一个已存在但可疑的文件。
    """
    global _fetch_mod
    if _fetch_mod is None:
        import importlib.util
        path = Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py"
        spec = importlib.util.spec_from_file_location("fetch_markers", path)
        _fetch_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_fetch_mod)
    try:
        return bool(_fetch_mod.fetch_and_write(runtime_markers_dir(), force=force))
    except Exception as e:
        logger.warning(f"trigger_fetch failed: {e}")
        return False


_topped_up = False


def top_up_missing_tables(db) -> bool:
    """Best-effort, once per process: re-fetch tables absent at this revision.

    A supplementary source whose download failed leaves the DB usable, so the
    gate that would have re-fetched it never fires and nothing asks again --
    the deployment annotates on the remaining source indefinitely, which is the
    opposite of why the revision was cut.

    Deliberately quiet: it never blocks a run and never escalates to the user,
    so a source that stays unreachable costs one attempt per process rather
    than a prompt per session. Call it off the event loop (asyncio.to_thread);
    it performs network I/O.
    """
    global _topped_up
    if _topped_up:
        return False
    _topped_up = True
    if not db.usable():
        return False  # the ordinary gate owns this case, and it can prompt.
    missing = db.missing_tables()
    if not missing:
        return False
    logger.warning(
        "marker DB is usable but %s absent at SCHEMA_REVISION %s; attempting a "
        "one-off top-up fetch", ", ".join(missing), MARKER_SCHEMA_REVISION)
    return trigger_fetch()


def summarize_database_priors(database_priors, total_clusters: int) -> Optional[str]:
    if not database_priors:
        return None
    n = len(database_priors)
    strong = [f"cluster {cid}→{h['cell_type']}"
              for cid, hits in database_priors.items()
              for h in hits if h.get("strong")]
    s = f"{n}/{total_clusters} clusters got DB priors"
    if strong:
        s += "; strong: " + ", ".join(strong[:5])
    return s
