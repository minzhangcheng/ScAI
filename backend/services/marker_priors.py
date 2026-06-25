"""离线 marker 数据库加载 / 物种筛选 / 合并去重 / 写 workspace。仅数据，无统计。"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["source", "species", "cell_type", "gene_symbol", "tissue", "evidence", "ref"]
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "markers"
_TSV_NAMES = ("panglaodb.tsv", "cellmarker2.tsv")  # PanglaoDB 优先
_SPECIES_NORM = {"mouse": "mouse", "mus musculus": "mouse", "mmusculus": "mouse", "mm": "mouse",
                 "human": "human", "homo sapiens": "human", "hsapiens": "human", "hs": "human"}


def normalize_species(organism: str) -> str:
    return _SPECIES_NORM.get(str(organism).strip().lower(), str(organism).strip().lower())


class MarkerDB:
    def __init__(self, markers_dir: Optional[Path] = None):
        self.markers_dir = Path(markers_dir) if markers_dir else _DEFAULT_DIR
        self._versions: Dict[str, str] = {}
        self._by_species: Dict[str, List[dict]] = {}
        self._loaded = False

    def present(self) -> bool:
        return any((self.markers_dir / n).exists() for n in _TSV_NAMES)

    def _parse_version(self, path: Path) -> str:
        try:
            first = path.read_text().splitlines()[0]
            if first.startswith("#") and "SOURCE_VERSION=" in first:
                return first.split("SOURCE_VERSION=", 1)[1].split()[0]
        except Exception:
            pass
        return "unknown"

    def _load(self) -> None:
        if self._loaded:
            return
        frames = []
        for name in _TSV_NAMES:
            path = self.markers_dir / name
            if not path.exists():
                continue
            self._versions[name.replace(".tsv", "")] = self._parse_version(path)
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
                                "tissue": "first", "evidence": "first", "ref": "first"})
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
        df = pd.DataFrame(rows)[["cell_type", "gene_symbol", "source"]]
        df.to_csv(out, sep="\t", index=False)
        return out


_fetch_mod = None


def trigger_fetch() -> bool:
    """同步下载 marker DB（供 asyncio.to_thread 调用）。失败返回 False，不抛。"""
    global _fetch_mod
    if _fetch_mod is None:
        import importlib.util
        path = Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_markers.py"
        spec = importlib.util.spec_from_file_location("fetch_markers", path)
        _fetch_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_fetch_mod)
    try:
        return bool(_fetch_mod.fetch_and_write())
    except Exception as e:
        logger.warning(f"trigger_fetch failed: {e}")
        return False


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
