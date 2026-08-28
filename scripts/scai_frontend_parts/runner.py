"""Lifecycle owner for one isolated frontend test run."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactBytes, FrontendError, ensure_directory, fsync_directory, json_hash,
    lstat_or_none, read_direct_children_nofollow, require_directory, require_regular,
    safe_lock, sha256_bytes, sha256_file, unlock, write_json, write_sealed_file,
)
from .identity import (
    input_hashes, load_tree_helper, resolve_run_start_head, source_node_modules_state,
    source_snapshot,
)
from .npm import (
    NODE_PLATFORM_QUERY, command_env, command_record, empty_config_file, expected_config,
    npm_ls_inventory, parse_npm_config_sources, parse_npm_config_values, tokenized,
    validate_npm_config_agreement, verify_config_files, walk_npm_tree,
)
from .results import parse_vitest

class FrontendRun:
    def __init__(self, repo: Path):
        self.repo = repo
        self.frontend = repo / "frontend"
        self.tree = load_tree_helper(repo)
        self.base_commit = ""
        self.source_before: dict[str, Any] | None = None
        self.source_after: dict[str, Any] | None = None
        self.package_hash = ""
        self.lock_hash = ""
        self.input_key = ""
        self.input_namespace: Path | None = None
        self.install_dir: Path | None = None
        self.runtime_dir: Path | None = None
        self.result_dir: Path | None = None
        self.run_id = ""
        self.global_fd: int | None = None
        self.keyed_fd: int | None = None
        self.original_link: tuple[str, str | None] | None = None
        self.linked = False
        self.env: dict[str, str] = {}
        self.node: Path | None = None
        self.npm: Path | None = None
        self.node_platform = ""
        self.node_arch = ""
        self.commands: dict[str, Any] = {}
        self.tree_payload: dict[str, Any] | None = None
        self.vitest_payload: dict[str, Any] | None = None
        self.config_values: dict[str, Any] | None = None
        self.config_sources: dict[str, Any] | None = None
        self.config_files: dict[str, Path] = {}
        self.empty_config_hashes: dict[str, str] = {}
        self.failure: dict[str, str] | None = None
        self.application_failure = False
        self._registered_result_names: set[str] = set()
        self.started = time.time()

    def fail(self, label: str, message: str) -> None:
        self.failure = {"label": label, "message": message}
        raise FrontendError(label, message)

    def register_result_name(self, name: str) -> None:
        """Admit one fixed direct-child artifact name before its command/write."""
        fixed = {
            "npm-config-values.json", "npm-config-sources.json", "npm-ls.json", "npm-tree.json",
            "vitest.json", "pre-source-manifest.json", "post-source-manifest.json",
        }
        command_output = re.fullmatch(r"[A-Za-z0-9_-]+\.(?:stdout|stderr)\.(?:txt|json|log)", name or "")
        if name not in fixed and command_output is None:
            self.fail("frontend evidence", f"result artifact name is not command-owned: {name!r}")
        if name in self._registered_result_names:
            self.fail("frontend evidence", f"result artifact name registered twice: {name}")
        self._registered_result_names.add(name)

    def load_result_artifacts(self) -> dict[str, ArtifactBytes]:
        """Require exactly registered direct children and load each once no-follow."""
        if self.result_dir is None:
            self.fail("frontend evidence", "result directory is unavailable")
        return read_direct_children_nofollow(self.result_dir, set(self._registered_result_names))

    def record_command(
        self,
        name: str,
        argv: list[str],
        cwd: Path,
        stdout_name: str,
        stderr_name: str,
    ) -> tuple[dict[str, Any], bytes, bytes]:
        assert self.result_dir and self.install_dir
        self.register_result_name(stdout_name)
        self.register_result_name(stderr_name)
        return command_record(
            name, argv, cwd, self.env, self.result_dir, self.repo, self.install_dir, stdout_name, stderr_name
        )

    def prepare(self) -> None:
        require_directory(self.frontend, "frontend source")
        npmrc = self.frontend / ".npmrc"
        if lstat_or_none(npmrc) is not None:
            self.fail("source npmrc", "frontend/.npmrc must be absent")
        self.original_link = source_node_modules_state(self.frontend)
        self.base_commit = resolve_run_start_head(self.repo)
        self.source_before = source_snapshot(self.repo, self.tree, self.base_commit)
        self.package_hash, self.lock_hash, self.input_key = input_hashes(self.frontend)
        self.input_namespace = self.repo / "test-results" / "frontend-installations" / self.input_key
        canonical_repo = str(self.repo.resolve())
        global_name = sha256_bytes(canonical_repo.encode())
        self.global_fd = safe_lock(Path("/tmp") / f"scai-frontend-{global_name}.lock", "frontend global lock")
        ensure_directory(self.repo / "test-results", "frontend evidence root")
        ensure_directory(self.repo / "test-results" / "frontend-installations", "frontend input namespace root")
        ensure_directory(self.input_namespace, "frontend input namespace")
        ensure_directory(self.input_namespace / "runs", "frontend input runs")
        self.keyed_fd = safe_lock(self.input_namespace / ".install.lock", "frontend keyed lock")
        latest_source = source_snapshot(self.repo, self.tree, self.base_commit)
        latest_hashes = input_hashes(self.frontend)
        latest_link = source_node_modules_state(self.frontend)
        if lstat_or_none(self.frontend / ".npmrc") is not None:
            self.fail("source npmrc", "frontend/.npmrc appeared while locks were acquired")
        if latest_source != self.source_before or latest_hashes != (self.package_hash, self.lock_hash, self.input_key) or latest_link != self.original_link:
            self.fail("frontend source identity", "source or frontend inputs changed while locks were acquired")
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
        self.install_dir = self.input_namespace / "runs" / self.run_id
        try:
            self.install_dir.mkdir(mode=0o700)
        except FileExistsError:
            self.fail("frontend install", "run directory already exists")
        self.result_dir = self.repo / "test-results" / f"{self.run_id}-frontend"
        try:
            self.result_dir.mkdir(mode=0o700)
        except FileExistsError:
            self.fail("frontend evidence", "result directory already exists")
        for source_name in ("package.json", "package-lock.json"):
            source = self.frontend / source_name
            destination = self.install_dir / source_name
            shutil.copyfile(source, destination)
            require_regular(destination, "frontend copied input")
        if sha256_file(self.install_dir / "package.json") != self.package_hash or sha256_file(self.install_dir / "package-lock.json") != self.lock_hash:
            self.fail("frontend inputs", "copied input hash differs from source")
        self.runtime_dir = self.install_dir / ".scai-runtime"
        ensure_directory(self.runtime_dir, "frontend runtime directory")
        self.node = Path(shutil.which("node") or "")
        self.npm = Path(shutil.which("npm") or "")
        if not self.node or not self.node.is_file() or not self.npm or not self.npm.is_file():
            self.fail("frontend tools", "node and npm are not resolvable")
        self.env = command_env(self.node, self.npm, self.install_dir, self.runtime_dir)
        self.config_files = {
            "userconfig": self.install_dir / ".npmrc",
            "globalconfig": self.install_dir / "global.npmrc",
        }
        empty_hashes = {
            "userconfig": empty_config_file(self.install_dir / ".npmrc"),
            "globalconfig": empty_config_file(self.install_dir / "global.npmrc"),
        }
        self.empty_config_hashes = empty_hashes
        self.commands["empty_config_hashes"] = empty_hashes
        platform_record, platform_raw, platform_err = self.record_command(
            "node_platform",
            [str(self.node), "-p", NODE_PLATFORM_QUERY],
            self.install_dir,
            "node-platform.stdout.json",
            "node-platform.stderr.log",
        )
        if platform_record["exit_code"] != 0 or platform_err:
            self.fail("frontend tools", "node platform query failed")
        try:
            parsed_platform = json.loads(platform_raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.fail("frontend tools", f"node platform query is malformed: {exc}")
        if (
            not isinstance(parsed_platform, dict)
            or set(parsed_platform) != {"platform", "arch"}
            or not isinstance(parsed_platform["platform"], str)
            or not parsed_platform["platform"]
            or not isinstance(parsed_platform["arch"], str)
            or not parsed_platform["arch"]
        ):
            self.fail("frontend tools", "node platform query returned an invalid identity")
        self.node_platform = parsed_platform["platform"]
        self.node_arch = parsed_platform["arch"]
        self.commands["node_platform"] = {**platform_record, "parsed": parsed_platform}
        version_commands = {}
        for name, executable in (("node_version", self.node), ("npm_version", self.npm)):
            if name == "npm_version":
                verify_config_files(self.config_files, self.empty_config_hashes)
            record, stdout, stderr = self.record_command(
                name, [str(executable), "--version"], self.install_dir,
                f"{name}.stdout.txt", f"{name}.stderr.log",
            )
            if record["exit_code"] != 0 or stderr:
                self.fail("frontend tools", f"{name} command failed")
            version_commands[name] = {**record, "version": stdout.decode("utf-8", "replace").strip()}
        self.commands["versions"] = version_commands
        expected = expected_config(self.install_dir, self.result_dir / "cache", self.install_dir / ".npmrc", self.install_dir / "global.npmrc")
        # The npm cache is created by command_env under the isolated install runtime;
        # use the exact environment path as the planned effective value.
        expected["cache"] = str(Path(self.env["npm_config_cache"]).resolve())
        verify_config_files(self.config_files, self.empty_config_hashes)
        values_record, values_raw, values_err = self.record_command(
            "npm_config_values", [str(self.npm), "config", "list", "--json"], self.install_dir,
            "npm-config-values.stdout.json", "npm-config-values.stderr.log",
        )
        self.commands["npm_config_values"] = values_record
        if values_record["exit_code"] != 0 or values_err:
            self.fail("npm config values", "npm config list --json failed or wrote stderr")
        self.config_values = parse_npm_config_values(values_raw, expected, empty_hashes)
        verify_config_files(self.config_files, self.empty_config_hashes)
        sources_record, sources_raw, sources_err = self.record_command(
            "npm_config_sources", [str(self.npm), "config", "list"], self.install_dir,
            "npm-config-sources.stdout.txt", "npm-config-sources.stderr.log",
        )
        self.commands["npm_config_sources"] = sources_record
        if sources_record["exit_code"] != 0 or sources_err:
            self.fail("npm config sources", "npm config list failed or wrote stderr")
        self.config_sources = parse_npm_config_sources(sources_raw, expected, empty_hashes, self.install_dir)
        validate_npm_config_agreement(self.config_values, self.config_sources)
        verify_config_files(self.config_files, self.empty_config_hashes)
        self.register_result_name("npm-config-values.json")
        self.register_result_name("npm-config-sources.json")
        write_json(self.result_dir / "npm-config-values.json", self.config_values)
        write_json(self.result_dir / "npm-config-sources.json", self.config_sources)

    def run_install(self) -> None:
        assert self.install_dir and self.result_dir and self.npm
        verify_config_files(self.config_files, self.empty_config_hashes)
        record, _stdout, _stderr = self.record_command(
            "npm_ci", [str(self.npm), "ci"], self.install_dir, "npm-ci.stdout.txt", "npm-ci.stderr.log"
        )
        self.commands["npm_ci"] = record
        if record["exit_code"] != 0:
            self.fail("npm ci", f"npm ci exited {record['exit_code']}")
        verify_config_files(self.config_files, self.empty_config_hashes)
        record, stdout, stderr = self.record_command(
            "npm_ls", [str(self.npm), "ls", "--all", "--json"], self.install_dir,
            "npm-ls.stdout.json", "npm-ls.stderr.log",
        )
        self.commands["npm_ls"] = record
        if record["exit_code"] != 0:
            self.fail("npm ls", f"npm ls exited {record['exit_code']}")
        try:
            npm_ls = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.fail("npm ls", f"npm ls JSON is malformed: {exc}")
        if not isinstance(npm_ls, dict) or npm_ls.get("problems"):
            self.fail("npm ls", "npm ls reports problems")
        npm_inventory = npm_ls_inventory(npm_ls)
        if not isinstance(self.node_platform, str) or not self.node_platform or not isinstance(self.node_arch, str) or not self.node_arch:
            self.fail("npm tree", "node platform identity is unavailable")
        self.register_result_name("npm-ls.json")
        write_json(self.result_dir / "npm-ls.json", npm_ls)
        self.tree_payload = walk_npm_tree(self.install_dir, npm_ls, self.node_platform, self.node_arch, npm_inventory)
        self.register_result_name("npm-tree.json")
        write_json(self.result_dir / "npm-tree.json", self.tree_payload)

    def verify_vitest_shim(self) -> None:
        assert self.install_dir and self.tree_payload
        expected = next((item for item in self.tree_payload["bins"] if item["path"] == "node_modules/.bin/vitest"), None)
        if expected is None:
            self.fail("npm bins", "verified vitest shim is missing")
        link = self.install_dir / "node_modules/.bin/vitest"
        if not link.is_symlink() or link.resolve(strict=True).relative_to(self.install_dir.resolve()).as_posix() != expected["target"]:
            self.fail("npm bins", "vitest shim changed after tree validation")

    def link_and_test(self) -> None:
        assert self.install_dir and self.result_dir and self.npm
        helper = self.repo / "scripts" / "link_frontend_node_modules.sh"
        require_regular(helper, "frontend link helper")
        vitest_path = self.result_dir / "vitest.json"
        try:
            self.linked = True
            self.register_result_name("link.stdout.txt")
            self.register_result_name("link.stderr.log")
            with (self.result_dir / "link.stdout.txt").open("wb") as stdout, (self.result_dir / "link.stderr.log").open("wb") as stderr:
                subprocess.run(["bash", str(helper), str(self.frontend), str(self.install_dir / "node_modules")], cwd=self.repo, env=self.env, stdout=stdout, stderr=stderr, check=True)
            self.verify_vitest_shim()
            verify_config_files(self.config_files, self.empty_config_hashes)
            self.register_result_name("vitest.json")
            record, _stdout, _stderr = self.record_command(
                "npm_test", [str(self.npm), "test", "--", "--reporter=json", f"--outputFile={vitest_path}"],
                self.frontend, "npm-test.stdout.txt", "npm-test.stderr.log",
            )
            self.commands["npm_test"] = record
            self.vitest_payload = parse_vitest(vitest_path, record["exit_code"])
            if record["exit_code"] != 0:
                self.application_failure = True
        except subprocess.CalledProcessError as exc:
            self.fail("frontend link", f"link helper exited {exc.returncode}")

    def restore_link(self) -> None:
        if not self.linked:
            return
        path = self.frontend / "node_modules"
        original_kind, original_target = self.original_link or ("absent", None)
        info = lstat_or_none(path)
        if info is not None and not stat.S_ISLNK(info.st_mode):
            self.fail("frontend restoration", "helper left a non-symlink node_modules path")
        if original_kind == "absent":
            if info is not None:
                path.unlink()
        elif original_kind == "symlink" and original_target is not None:
            temporary = self.frontend / f".node_modules.restore.{uuid.uuid4().hex}"
            os.symlink(original_target, temporary)
            os.replace(temporary, path)
        else:
            self.fail("frontend restoration", "original node_modules state is invalid")
        final_kind, final_target = source_node_modules_state(self.frontend)
        if (final_kind, final_target) != (original_kind, original_target):
            self.fail("frontend restoration", "frontend/node_modules was not restored exactly")
        self.linked = False

    def check_source_after(self) -> None:
        if lstat_or_none(self.frontend / ".npmrc") is not None:
            self.fail("source npmrc", "frontend/.npmrc was created during frontend run")
        if self.original_link is not None and source_node_modules_state(self.frontend) != self.original_link:
            self.fail("frontend source identity", "frontend/node_modules state changed during frontend run")
        self.source_after = source_snapshot(self.repo, self.tree, self.base_commit)
        if self.source_before != self.source_after:
            self.fail("frontend source identity", "source manifest changed during frontend run")

    def artifact_manifest(self, artifacts: dict[str, ArtifactBytes]) -> dict[str, Any]:
        """Build the detached index from the one trusted artifact byte snapshot."""
        entries = [
            {"path": item.name, "size": item.size, "sha256": item.sha256}
            for item in (artifacts[name] for name in sorted(artifacts))
        ]
        return {"schema": 1, "entries": entries, "digest": json_hash(entries)}

    def evidence(self, status: str, manifest: dict[str, Any]) -> dict[str, Any]:
        assert self.result_dir
        source = {"base_commit": self.base_commit, "manifest": self.source_before, "manifest_after": self.source_after}
        installation = tokenized({"namespace": str(self.input_namespace) if self.input_namespace else None, "run_dir": str(self.install_dir) if self.install_dir else None}, self.repo, self.result_dir, self.install_dir)
        if self.run_id and installation.get("run_dir") == "<install-dir>":
            installation["run_dir"] = f"<install-dir>/{self.run_id}"
        return {
            "schema": 1,
            "run_id": self.run_id,
            "status": status,
            "source": tokenized(source, self.repo, self.result_dir, self.install_dir),
            "node_version": self.commands.get("versions", {}).get("node_version", {}).get("version"),
            "npm_version": self.commands.get("versions", {}).get("npm_version", {}).get("version"),
            "package_json_sha256": self.package_hash,
            "package_lock_sha256": self.lock_hash,
            "input_key": self.input_key,
            "installation": installation,
            "effective_environment": tokenized(self.env, self.repo, self.result_dir, self.install_dir),
            "npm_config_values": tokenized(self.config_values, self.repo, self.result_dir, self.install_dir),
            "npm_config_sources": tokenized(self.config_sources, self.repo, self.result_dir, self.install_dir),
            "commands": tokenized(self.commands, self.repo, self.result_dir, self.install_dir),
            "tree": tokenized(self.tree_payload, self.repo, self.result_dir, self.install_dir),
            "vitest": tokenized(self.vitest_payload, self.repo, self.result_dir, self.install_dir),
            "error": tokenized(self.failure, self.repo, self.result_dir, self.install_dir),
            "artifacts": manifest,
            "utc_start_epoch": self.started,
            "utc_end_epoch": time.time(),
            "duration_seconds": round(max(0.0, time.time() - self.started), 6),
        }

    def finish(self, status: str) -> None:
        if self.result_dir is None:
            return
        if self.source_before is not None:
            self.register_result_name("pre-source-manifest.json")
            write_json(self.result_dir / "pre-source-manifest.json", self.source_before)
        if self.source_after is not None:
            self.register_result_name("post-source-manifest.json")
            write_json(self.result_dir / "post-source-manifest.json", self.source_after)
        fsync_directory(self.result_dir)
        artifacts = self.load_result_artifacts()
        manifest = self.artifact_manifest(artifacts)
        manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        evidence = self.evidence(status, manifest)
        evidence_payload = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        write_sealed_file(self.result_dir, "artifact-manifest.json", manifest_payload)
        write_sealed_file(self.result_dir, "evidence.json", evidence_payload)

    def close(self) -> FrontendError | None:
        """Restore the source link and prove final source identity while locked."""
        try:
            self.restore_link()
            if self.source_before is not None:
                self.check_source_after()
        except FrontendError as exc:
            self.failure = {"label": exc.label, "message": str(exc)}
            return exc
        return None

    def release_locks(self) -> None:
        """Release keyed then global locks; safe to call once in the outer finally."""
        keyed_fd, global_fd = self.keyed_fd, self.global_fd
        self.keyed_fd = None
        self.global_fd = None
        try:
            unlock(keyed_fd)
        finally:
            unlock(global_fd)
