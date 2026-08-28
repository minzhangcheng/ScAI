"""Npm configuration, commands, and installed-tree verification."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    ensure_directory, fail, json_hash, lstat_or_none, path_inside, require_regular,
    sha256_bytes, sha256_file,
)

REGISTRY = "https://registry.npmjs.org/"
CONFIG_KEYS = ("cache", "userconfig", "globalconfig", "registry", "audit", "fund", "update-notifier")
SECRET_KEYS = {"_auth", "_authtoken", "_password", "password", "token", "cert", "key"}
REGISTRY_SECRET = re.compile(
    r"^//[^/]+(?:/[^:]*)?/:(_auth|_authtoken|_password|password|token|cert|key)$",
    re.IGNORECASE,
)
NODE_PLATFORM_QUERY = "JSON.stringify({platform:process.platform,arch:process.arch})"

def normalize_path_value(value: Any, expected: str) -> Any:
    if isinstance(expected, str) and (expected.startswith("/") or "/" in expected):
        return str(Path(str(value)).resolve(strict=False))
    return value


def check_secret_key(key: str, value: Any) -> None:
    normalized = key.lower()
    if REGISTRY_SECRET.fullmatch(normalized):
        fail("npm config secret", f"registry-scoped credential key is forbidden: {key}")
    if normalized in SECRET_KEYS and value not in (None, "", False):
        fail("npm config secret", f"credential key is forbidden: {key}")


def expected_config(install_dir: Path, cache_dir: Path, userconfig: Path, globalconfig: Path) -> dict[str, Any]:
    return {
        "cache": str(cache_dir.resolve()),
        "userconfig": str(userconfig.resolve()),
        "globalconfig": str(globalconfig.resolve()),
        "registry": REGISTRY,
        "audit": False,
        "fund": False,
        "update-notifier": False,
    }
def normalize_config(value: Any, expected: Any) -> Any:
    if isinstance(expected, bool):
        if not isinstance(value, bool):
            fail("npm config values", f"boolean config value is malformed: {value!r}")
        return value
    if isinstance(expected, str) and expected.startswith("/"):
        return str(Path(str(value)).resolve(strict=False))
    return value


def parse_npm_config_values(raw: bytes, expected: dict[str, Any], empty_hashes: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        fail("npm config values", f"JSON output is malformed: {exc}")
    if not isinstance(value, dict):
        fail("npm config values", "JSON output is not an object")
    for key, item in value.items():
        if isinstance(key, str):
            check_secret_key(key, item)
    if value.get("json") is not True:
        fail("npm config values", "command-owned json=true is missing")
    if value.get("location") != "project" or not isinstance(value.get("location"), str):
        fail("npm config values", "command-owned location=project is missing or malformed")
    effective: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key not in value:
            fail("npm config values", f"planned override is missing: {key}")
        actual = normalize_config(value[key], expected_value)
        if actual != expected_value:
            fail("npm config values", f"planned override drift: {key}={actual!r}")
        effective[key] = actual
    builtin = {key: item for key, item in value.items() if key not in set(expected) | {"json", "location"}}
    return {
        "schema": 1,
        "command_key": "npm_config_values",
        "command_owned": {"json": True, "location": "project"},
        "effective_overrides": effective,
        "builtin_values_sha256": json_hash(builtin),
        "empty_config_hashes": empty_hashes,
        "raw_stdout_sha256": sha256_bytes(raw),
    }


def parse_config_value(text: str) -> Any:
    value = text.strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except ValueError:
            return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def validate_npm_config_agreement(values: dict[str, Any], sources: dict[str, Any]) -> None:
    value_overrides = values.get("effective_overrides")
    source_overrides = sources.get("final_effective_overrides")
    if (
        not isinstance(value_overrides, dict)
        or not isinstance(source_overrides, dict)
        or set(value_overrides) != set(CONFIG_KEYS)
        or set(source_overrides) != set(CONFIG_KEYS)
        or value_overrides != source_overrides
    ):
        fail("npm config agreement", "JSON and plain npm config overrides disagree")
    if values.get("command_owned") != {"json": True, "location": "project"}:
        fail("npm config agreement", "JSON command-owned config is not exact")
    if sources.get("command_owned") != {"location": "project"}:
        fail("npm config agreement", "plain command-owned config is not exact")


def parse_npm_config_sources(raw: bytes, expected: dict[str, Any], empty_hashes: dict[str, str], install_dir: Path) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail("npm config sources", f"plain output is not UTF-8: {exc}")
    section_class = "npm builtin/default"
    section_path: str | None = None
    observed: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    environment_values: dict[str, Any] = {}
    command_owned: dict[str, Any] = {}
    file_keys: set[str] = set()
    environment_sections = 0
    cli_sections = 0
    header = re.compile(r'^\s*;\s*"([^"]+)" config from\s*(.*?)\s*$')
    assignment = re.compile(r"^\s*([A-Za-z0-9_./:@-]+)\s*=\s*(.*?)\s*$")
    for line in text.splitlines():
        match = header.match(line)
        if match:
            name, source = match.groups()
            lowered = name.lower()
            source_path = source.strip() or None
            if name == "env":
                if source_path != "environment":
                    fail("npm config sources", f"unexpected reserved env config source: {source_path}")
                environment_sections += 1
                if environment_sections != 1:
                    fail("npm config sources", "duplicate environment config section")
                section_class = "environment"
                section_path = source_path
            elif name == "cli":
                if source_path != "command line options":
                    fail("npm config sources", f"unexpected reserved cli config source: {source_path}")
                cli_sections += 1
                if cli_sections != 1:
                    fail("npm config sources", "duplicate command-owned CLI config section")
                section_class = "command-owned CLI"
                section_path = source_path
            elif "builtin" in lowered or "default" in lowered:
                section_class = "npm builtin/default"
                section_path = source_path
            elif source_path in {str((install_dir / ".npmrc").resolve()), str((install_dir / "global.npmrc").resolve())}:
                section_class = "empty config file"
                section_path = source_path
            else:
                fail("npm config sources", f"unexpected npm config source: {name} {source_path}")
            observed.append({"class": section_class, "path": section_path})
            continue
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        match = assignment.match(line)
        if not match:
            fail("npm config sources", f"unrecognized config output line: {line!r}")
        key, raw_value = match.groups()
        if key == "location" and section_class != "command-owned CLI":
            fail("npm config sources", f"location assignment is forbidden in {section_class}")
        value = parse_config_value(raw_value)
        check_secret_key(key, value)
        if section_class == "environment":
            if key not in expected:
                fail("npm config sources", f"unexpected environment config key: {key}")
            if key in environment_values:
                fail("npm config sources", f"duplicate environment config key: {key}")
            environment_values[key] = value
            final[key] = value
        elif section_class == "command-owned CLI":
            if key != "location":
                fail("npm config sources", f"unexpected command-owned CLI config key: {key}")
            if key in command_owned:
                fail("npm config sources", "duplicate command-owned CLI location")
            if not isinstance(value, str) or value != "project":
                fail("npm config sources", "command-owned CLI location must be exact string project")
            command_owned[key] = value
        elif section_class == "empty config file":
            file_keys.add(key)
        else:
            final[key] = value
    if environment_sections and set(environment_values) != set(expected):
        fail("npm config sources", f"environment source key set drift: {sorted(environment_values)}")
    if cli_sections != 1 or command_owned != {"location": "project"}:
        fail("npm config sources", "exact command-owned CLI location=project section is required")
    if file_keys:
        fail("npm config sources", "an allowed npm config file was not empty")
    for name, relative in (("userconfig", ".npmrc"), ("globalconfig", "global.npmrc")):
        path = install_dir / relative
        require_regular(path, "npm config file")
        if path.stat().st_size != 0 or sha256_file(path) != empty_hashes[name]:
            fail("npm config file", f"allowed npm config file changed: {name}")
    for key, expected_value in expected.items():
        if key not in final:
            fail("npm config sources", f"final source value is missing: {key}")
        actual = normalize_config(final[key], expected_value)
        if actual != expected_value:
            fail("npm config sources", f"plain/JSON config drift: {key}={actual!r}")
    return {
        "schema": 1,
        "command_key": "npm_config_sources",
        "allowed_sources": ["npm builtin/default", "environment", "empty config file", "command-owned CLI"],
        "command_owned": {"location": "project"},
        "observed_sources": observed,
        "final_effective_overrides": {key: normalize_config(final[key], expected[key]) for key in expected},
        "empty_config_hashes": empty_hashes,
        "raw_stdout_sha256": sha256_bytes(raw),
    }


def tokenized(value: Any, repo: Path, result_dir: Path | None = None, install_dir: Path | None = None) -> Any:
    if isinstance(value, str):
        result = value
        replacements: list[tuple[str, str]] = []
        if install_dir is not None:
            replacements.append((str(install_dir.resolve()), "<install-dir>"))
        if result_dir is not None:
            replacements.append((str(result_dir.resolve()), "<result-dir>"))
        replacements.append((str(repo.resolve()), "<repo-root>"))
        for source, target in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
            result = result.replace(source, target)
        return result
    if isinstance(value, list):
        return [tokenized(item, repo, result_dir, install_dir) for item in value]
    if isinstance(value, dict):
        return {key: tokenized(item, repo, result_dir, install_dir) for key, item in value.items()}
    return value


def command_env(node: Path, npm: Path, install_dir: Path, runtime_dir: Path) -> dict[str, str]:
    # Resolve node/npm before entering this environment.  Do not copy the
    # caller's PATH, node option variables, npmrc, or upper-case npm configuration.
    path_parts = list(dict.fromkeys([str(node.parent), str(npm.parent), *os.defpath.split(os.pathsep)]))
    home = runtime_dir / "home"
    cache = runtime_dir / "cache"
    xdg = runtime_dir / "xdg"
    for directory in (home, cache, xdg):
        ensure_directory(directory, "frontend isolated directories")
    userconfig = install_dir / ".npmrc"
    globalconfig = install_dir / "global.npmrc"
    return {
        "PATH": os.pathsep.join(path_parts),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg),
        "XDG_CACHE_HOME": str(cache),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "npm_config_cache": str(cache),
        "npm_config_userconfig": str(userconfig),
        "npm_config_globalconfig": str(globalconfig),
        "npm_config_registry": REGISTRY,
        "npm_config_audit": "false",
        "npm_config_fund": "false",
        "npm_config_update_notifier": "false",
    }


def empty_config_file(path: Path) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    require_regular(path, "npm config file")
    if path.stat().st_size != 0:
        fail("npm config file", f"config file is not empty: {path}")
    return sha256_file(path)


def verify_config_files(paths: dict[str, Path], expected_hashes: dict[str, str]) -> None:
    """Reopen the two command-owned files before each npm invocation."""
    for name, path in paths.items():
        require_regular(path, "npm config file")
        if path.stat().st_size != 0 or sha256_file(path) != expected_hashes[name]:
            fail("npm config file", f"command-owned config changed: {name}")


def command_record(
    name: str,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    result_dir: Path,
    repo: Path,
    install_dir: Path,
    stdout_name: str,
    stderr_name: str,
) -> tuple[dict[str, Any], bytes, bytes]:
    stdout_path = result_dir / stdout_name
    stderr_path = result_dir / stderr_name
    start = time.time()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
    end = time.time()
    raw_stdout = stdout_path.read_bytes()
    raw_stderr = stderr_path.read_bytes()
    stdout_data = sanitize_output(raw_stdout, repo, result_dir, install_dir)
    stderr_data = sanitize_output(raw_stderr, repo, result_dir, install_dir)
    stdout_path.write_bytes(stdout_data)
    stderr_path.write_bytes(stderr_data)
    record = {
        "argv": tokenized(argv, repo, result_dir, install_dir),
        "cwd": tokenized(str(cwd), repo, result_dir, install_dir),
        "environment": tokenized(sorted(env.items()), repo, result_dir, install_dir),
        "utc_start_epoch": start,
        "utc_end_epoch": end,
        "duration_seconds": round(max(0.0, end - start), 6),
        "exit_code": process.returncode,
        "stdout": stdout_name,
        "stderr": stderr_name,
        "stdout_sha256": sha256_bytes(stdout_data),
        "stderr_sha256": sha256_bytes(stderr_data),
    }
    return record, raw_stdout, raw_stderr


def sanitize_output(raw: bytes, repo: Path, result_dir: Path, install_dir: Path) -> bytes:
    """Keep command artifacts useful without publishing checkout-local paths."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        fail("frontend output", "command output is not UTF-8")
    for source, target in sorted(
        (
            (str(install_dir.resolve()), "<install-dir>"),
            (str(result_dir.resolve()), "<result-dir>"),
            (str(repo.resolve()), "<repo-root>"),
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    ):
        text = text.replace(source, target)
    return text.encode("utf-8")


def expected_name_from_key(key: str) -> str:
    suffix = key.rsplit("node_modules/", 1)[-1]
    parts = suffix.split("/")
    if parts[0].startswith("@"):
        if len(parts) < 2:
            fail("npm tree", f"scoped package path is malformed: {key}")
        return "/".join(parts[:2])
    return parts[0]


def canonicalize_bin_target(value: Any) -> str:
    """Canonicalize the one npm 9 bin-target lexical equivalence."""
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        fail("npm tree", "package bin target is malformed")
    canonical = value[2:] if value.startswith("./") else value
    if not canonical or canonical.startswith("/"):
        fail("npm tree", "package bin target is malformed")
    if any(not part or part in {".", ".."} for part in canonical.split("/")):
        fail("npm tree", "package bin target is malformed")
    if value not in {canonical, f"./{canonical}"}:
        fail("npm tree", "package bin target is malformed")
    return canonical


def normalize_bin(name: str, value: Any) -> dict[str, str]:
    if isinstance(value, str):
        entries = ((name.rsplit("/", 1)[-1], value),)
    elif isinstance(value, dict):
        entries = value.items()
    else:
        fail("npm tree", f"package bin declaration is malformed: {name}")
    normalized: dict[str, str] = {}
    for command, target in entries:
        if not isinstance(command, str) or not command:
            fail("npm tree", f"package bin declaration is malformed: {name}")
        normalized[command] = canonicalize_bin_target(target)
    return normalized


def npm_selector_applies(selector: object, actual: str, label: str) -> bool:
    """Apply npm positive/negative os or cpu selector semantics."""
    if selector is None:
        return True
    if not isinstance(selector, list) or not selector or any(not isinstance(item, str) or not item for item in selector):
        fail("npm tree", f"lock {label} selector is malformed")
    negatives = {item[1:] for item in selector if item.startswith("!")}
    if actual in negatives:
        return False
    positives = {item for item in selector if not item.startswith("!")}
    return not positives or actual in positives


def optional_omission_reason(
    locked: dict[str, object],
    node_platform: str,
    node_arch: str,
    npm_reported_optional_paths: frozenset[str],
    lock_path: str,
) -> str | None:
    """Return the exact host-owned reason for an omitted optional package."""
    if locked.get("optional") is not True:
        return None
    if "os" in locked and not npm_selector_applies(locked["os"], node_platform, "os"):
        return f"lock os selector excludes {node_platform}"
    if "cpu" in locked and not npm_selector_applies(locked["cpu"], node_arch, "cpu"):
        return f"lock cpu selector excludes {node_arch}"
    if lock_path in npm_reported_optional_paths:
        return "npm ls reported an optional omission"
    return None


def npm_ls_inventory(
    value: dict[str, object],
) -> tuple[
    frozenset[tuple[str, str]],
    tuple[tuple[str, str, str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    """Retain npm 9 identities and its logical parent graph without treating it as a disk tree."""
    identities: set[tuple[str, str]] = set()
    logical_nodes: list[tuple[str, str, str, str]] = []
    empty_edges: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()

    def visit(dependencies: Any, parent: str) -> None:
        if dependencies is None:
            return
        if not isinstance(dependencies, dict):
            fail("npm ls", "dependencies is not an object")
        for name, detail in sorted(dependencies.items()):
            if not isinstance(name, str) or not name or not isinstance(detail, dict):
                fail("npm ls", "dependency entry is malformed")
            parts = name.split("/")
            if (
                "\0" in name
                or "\\" in name
                or (name.startswith("@") and (len(parts) != 2 or any(not part for part in parts)))
                or (not name.startswith("@") and len(parts) != 1)
                or any(part in {".", ".."} for part in parts)
            ):
                fail("npm ls", f"dependency name is malformed: {name}")
            path = f"{parent}/node_modules/{name}" if parent else f"node_modules/{name}"
            if path in seen_paths:
                fail("npm ls", f"duplicate logical dependency path: {path}")
            seen_paths.add(path)
            if detail == {}:
                empty_edges.append((path, parent, name))
                continue
            version = detail.get("version")
            if not isinstance(version, str) or not version:
                fail("npm ls", f"dependency version is missing: {name}")
            identities.add((name, version))
            logical_nodes.append((path, parent, name, version))
            visit(detail.get("dependencies"), path)

    visit(value.get("dependencies"), "")
    return frozenset(identities), tuple(logical_nodes), tuple(empty_edges)


def require_contained_directory(path: Path, root_real: Path) -> Path:
    info = lstat_or_none(path)
    if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        fail("npm tree", f"invalid directory: {path}")
    real = path.resolve(strict=True)
    if not path_inside(real, root_real):
        fail("npm tree", f"directory escapes installation: {path}")
    return real


def require_target_file(path: Path, package_dir: Path, root_real: Path) -> None:
    info = lstat_or_none(path)
    if info is None or not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
        fail("npm tree", f"bin target is not a regular file: {path}")
    real = path.resolve(strict=True)
    if not path_inside(real, package_dir) or not path_inside(real, root_real):
        fail("npm tree", f"bin target escapes package/install: {path}")


def walk_npm_tree(
    install_dir: Path,
    npm_ls: dict[str, Any],
    node_platform: str,
    node_arch: str,
    npm_inventory: tuple[
        frozenset[tuple[str, str]],
        tuple[tuple[str, str, str, str], ...],
        tuple[tuple[str, str, str], ...],
    ] | None = None,
) -> dict[str, Any]:
    node_modules = install_dir / "node_modules"
    root_real = require_contained_directory(node_modules, install_dir.resolve())
    lock_path = install_dir / "package-lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail("npm tree", f"package lock is malformed: {exc}")
    if lock.get("lockfileVersion") != 3 or not isinstance(lock.get("packages"), dict):
        fail("npm tree", "package-lock.json is not npm lockfile version 3")
    lock_packages = {key: value for key, value in lock["packages"].items() if key}
    npm_identities, npm_logical_nodes, npm_empty_edges = npm_inventory or npm_ls_inventory(npm_ls)
    for locked in lock_packages.values():
        if isinstance(locked, dict):
            for label in ("os", "cpu"):
                if label in locked:
                    npm_selector_applies(locked[label], node_platform if label == "os" else node_arch, label)
    records: list[dict[str, Any]] = []
    installed_metadata: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, dict[str, Any]]] = {}
    bin_dirs: set[str] = set()
    hidden_paths: list[str] = []

    def register_package(package_dir: Path) -> None:
        require_contained_directory(package_dir, root_real)
        key = package_dir.relative_to(install_dir).as_posix()
        metadata = package_dir / "package.json"
        require_regular(metadata, "invalid package metadata")
        if not path_inside(metadata.resolve(), root_real):
            fail("npm tree", f"package metadata escapes installation: {key}")
        try:
            actual = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            fail("npm tree", f"package metadata is malformed at {key}: {exc}")
        locked = lock_packages.get(key)
        if not isinstance(locked, dict) or actual.get("name") != expected_name_from_key(key) or actual.get("version") != locked.get("version"):
            fail("npm tree", f"installed/lock drift at {key}")
        if not isinstance(locked.get("resolved"), str) or not isinstance(locked.get("integrity"), str):
            fail("npm tree", f"lock identity is incomplete at {key}")
        installed_metadata[key] = actual
        actual_bin = normalize_bin(actual.get("name", ""), actual["bin"]) if "bin" in actual else {}
        locked_bin = normalize_bin(actual.get("name", ""), locked["bin"]) if "bin" in locked else {}
        if actual_bin != locked_bin:
            fail("npm tree", f"package bin drift at {key}")
        records.append({"path": key, "name": actual["name"], "version": actual["version"], "resolved": locked["resolved"], "integrity": locked["integrity"]})
        if actual_bin:
            # The final node_modules ancestor is the .bin directory level.
            marker = key.rfind("/node_modules/")
            level = key[: marker + len("/node_modules")] if marker >= 0 else "node_modules"
            expected = declarations.setdefault(level, {})
            for command, target in actual_bin.items():
                if not target or Path(target).is_absolute() or any(part == ".." for part in Path(target).parts):
                    fail("npm tree", f"bin target is unsafe at {key}: {command}")
                target_path = package_dir / target
                require_target_file(target_path, package_dir, root_real)
                if command in expected:
                    fail("npm tree", f"duplicate bin declaration at {level}/{command}")
                expected[command] = {
                    "target": target_path,
                    "target_rel": target_path.relative_to(install_dir).as_posix(),
                    "package": key,
                }
        nested = package_dir / "node_modules"
        if lstat_or_none(nested) is not None:
            visit_node_modules(nested)

    def visit_node_modules(directory: Path) -> None:
        level = directory.relative_to(install_dir).as_posix()
        require_contained_directory(directory, root_real)
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.name == ".package-lock.json":
                require_regular(entry, "npm hidden lock")
                if not path_inside(entry.resolve(), root_real):
                    fail("npm tree", "hidden npm lock escapes installation")
                hidden_paths.append(entry.relative_to(install_dir).as_posix())
            elif entry.name == ".bin":
                require_contained_directory(entry, root_real)
                bin_dirs.add(level)
            elif entry.name.startswith("@"):
                require_contained_directory(entry, root_real)
                for child in sorted(entry.iterdir(), key=lambda item: item.name):
                    register_package(child)
            else:
                register_package(entry)

    visit_node_modules(node_modules)
    installed = {record["path"] for record in records}

    def resolve_node_module(parent: str, name: str, available: set[str]) -> str | None:
        """Resolve one dependency using Node's nearest ancestor node_modules order."""
        current = parent
        while current:
            candidate = f"{current}/node_modules/{name}"
            if candidate in available:
                return candidate
            marker = current.rfind("/node_modules/")
            current = current[:marker] if marker >= 0 else ""
        candidate = f"node_modules/{name}"
        return candidate if candidate in available else None

    logical_physical = {"": ""}
    for logical_path, logical_parent, name, version in npm_logical_nodes:
        parent = logical_physical.get(logical_parent)
        if parent is None:
            fail("npm ls", f"logical dependency parent is unknown: {logical_path}")
        physical = resolve_node_module(parent, name, installed)
        if physical is None:
            fail("npm ls", f"logical dependency is absent from the installed tree: {logical_path}")
        locked = lock_packages.get(physical)
        if not isinstance(locked, dict) or locked.get("version") != version:
            fail("npm ls", f"logical dependency identity differs from the lock: {logical_path}")
        logical_physical[logical_path] = physical

    all_lock_paths = set(lock_packages)
    npm_reported_optional_paths: set[str] = set()
    root_locked = lock["packages"].get("")
    for logical_path, logical_parent, name in npm_empty_edges:
        parent = logical_physical.get(logical_parent)
        if parent is None:
            fail("npm tree", f"npm ls empty dependency parent is unknown: {logical_path}")
        parent_locked = root_locked if not parent else lock_packages.get(parent)
        if not isinstance(parent_locked, dict):
            fail("npm tree", f"npm ls empty dependency parent is not locked: {logical_path}")
        parent_actual = None if not parent else installed_metadata.get(parent)
        if parent and not isinstance(parent_actual, dict):
            fail("npm tree", f"npm ls empty dependency parent is not installed: {logical_path}")

        optional_dependencies = parent_locked.get("optionalDependencies", {})
        if not isinstance(optional_dependencies, dict):
            fail("npm tree", f"parent optionalDependencies is malformed: {logical_path}")
        if name in optional_dependencies:
            optional_spec = optional_dependencies[name]
            if not isinstance(optional_spec, str) or not optional_spec:
                fail("npm tree", f"parent optional dependency is malformed: {logical_path}")
            if parent_actual is not None and parent_actual.get("optionalDependencies") != optional_dependencies:
                fail("npm tree", f"installed/lock optional dependency declaration drift: {logical_path}")
            target = resolve_node_module(parent, name, all_lock_paths)
            target_locked = lock_packages.get(target) if target is not None else None
            if (
                target is None
                or not isinstance(target_locked, dict)
                or not isinstance(target_locked.get("version"), str)
                or not target_locked["version"]
                or target_locked.get("optional") is not True
                or target in installed
            ):
                fail("npm tree", f"npm ls optional dependency target is invalid: {logical_path}")
            has_selector = "os" in target_locked or "cpu" in target_locked
            selector_reason = optional_omission_reason(
                target_locked, node_platform, node_arch, frozenset(), target
            )
            if has_selector and selector_reason is None:
                fail("npm tree", f"npm ls optional dependency selector still applies: {logical_path}")
            npm_reported_optional_paths.add(target)
            continue

        peer_dependencies = parent_locked.get("peerDependencies", {})
        peer_meta = parent_locked.get("peerDependenciesMeta", {})
        if not isinstance(peer_dependencies, dict) or not isinstance(peer_meta, dict):
            fail("npm tree", f"parent peer dependency metadata is malformed: {logical_path}")
        peer_spec = peer_dependencies.get(name)
        meta = peer_meta.get(name)
        if (
            not isinstance(peer_spec, str)
            or not peer_spec
            or not isinstance(meta, dict)
            or meta.get("optional") is not True
        ):
            fail("npm tree", f"npm ls reported an unknown or required empty dependency: {logical_path}")
        if parent_actual is not None and (
            parent_actual.get("peerDependencies") != peer_dependencies
            or parent_actual.get("peerDependenciesMeta") != peer_meta
        ):
            fail("npm tree", f"installed/lock optional peer declaration drift: {logical_path}")
        if resolve_node_module(parent, name, all_lock_paths) is not None:
            fail("npm tree", f"npm ls optional peer unexpectedly has a lock target: {logical_path}")

    optional_omissions: list[dict[str, str]] = []
    for key, locked in lock_packages.items():
        if key in installed:
            continue
        if isinstance(locked, dict):
            reason = optional_omission_reason(
                locked, node_platform, node_arch, frozenset(npm_reported_optional_paths), key
            )
            if reason is not None:
                optional_omissions.append({"path": key, "reason": reason})
                continue
        fail("npm tree", f"locked package is absent from installation: {key}")
    if set(declarations) != bin_dirs:
        fail("npm bins", f".bin level set drift: declarations={sorted(declarations)}, dirs={sorted(bin_dirs)}")
    bin_records: list[dict[str, Any]] = []
    for level in sorted(bin_dirs):
        directory = install_dir / level / ".bin"
        expected = declarations[level]
        actual_names = sorted(item.name for item in directory.iterdir())
        if actual_names != sorted(expected):
            fail("npm bins", f"bin name set drift at {level}")
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            declaration = expected[entry.name]
            info = entry.lstat()
            target = declaration["target"]
            if stat.S_ISLNK(info.st_mode):
                raw_target = os.readlink(entry)
                if Path(raw_target).is_absolute():
                    fail("npm bins", f"absolute bin target: {entry}")
                if (entry.parent / raw_target).resolve(strict=True) != target.resolve(strict=True):
                    fail("npm bins", f"wrong bin target: {entry}")
                kind = "symlink"
            elif stat.S_ISREG(info.st_mode) and info.st_mode & stat.S_IXUSR:
                require_target_file(entry, root_real, root_real)
                if entry.read_bytes() != target.read_bytes():
                    fail("npm bins", f"copied bin bytes drift: {entry}")
                kind = "regular"
            else:
                fail("npm bins", f"unsafe bin entry: {entry}")
            bin_records.append({"path": entry.relative_to(install_dir).as_posix(), "kind": kind, "target": declaration["target_rel"], "target_sha256": sha256_file(target)})
    if "node_modules/.bin/vitest" not in {item["path"] for item in bin_records}:
        fail("npm bins", "vitest shim is missing")
    hidden = node_modules / ".package-lock.json"
    require_regular(hidden, "npm hidden lock")
    try:
        hidden_payload = json.loads(hidden.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail("npm hidden lock", f"hidden lock is malformed: {exc}")
    if hidden_payload.get("lockfileVersion") != 3 or not isinstance(hidden_payload.get("packages"), dict):
        fail("npm hidden lock", "hidden lock is not npm version 3")
    hidden_installed = {key for key in hidden_payload["packages"] if key}
    if hidden_installed != installed:
        fail("npm hidden lock", "hidden lock package subset differs from installed tree")
    expected_ls = {(item["name"], item["version"]) for item in records}
    if npm_identities != expected_ls:
        fail("npm ls", "npm ls dependency identities differ from installed tree")
    return {
        "schema": 1,
        "lockfile_version": 3,
        "packages": sorted(records, key=lambda item: item["path"]),
        "optional_omissions": sorted(optional_omissions, key=lambda item: item["path"]),
        "hidden_lock": {"path": hidden.relative_to(install_dir).as_posix(), "sha256": sha256_file(hidden)},
        "bins": sorted(bin_records, key=lambda item: item["path"]),
        "npm_ls_sha256": json_hash(npm_ls),
    }
