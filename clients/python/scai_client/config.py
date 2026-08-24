"""ConfigV1 loading and deterministic local data-file resolution."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


AGENT_NAMES = (
    "pi", "pathology", "bioinformatics", "biochemistry",
    "literature", "painter", "supervisor", "technique",
)


def load_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path]:
    """Load JSON and return it together with its resolved source path."""
    path = Path(config_path).resolve()
    try:
        return json.loads(path.read_text()), path
    except FileNotFoundError:
        raise FileNotFoundError(f"config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from None


def resolve_data_files(config: Dict[str, Any], base_dir: Path) -> List[Path]:
    """Resolve every declared local path before creating a remote session."""
    resolved: List[Path] = []
    missing: List[str] = []
    for entry in config.get("data", {}).get("files", []):
        candidate = Path(entry["local_path"])
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        try:
            local = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            missing.append(str(candidate))
            continue
        if not local.is_file():
            missing.append(str(candidate))
            continue
        resolved.append(local)
    if missing:
        raise FileNotFoundError(
            f"data.files references {len(missing)} path(s) that do not exist: "
            + ", ".join(missing)
        )
    return resolved
