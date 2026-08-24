"""Focused contracts for the unified ``scai`` entry point."""

import importlib
import importlib.util
import re
import stat
from pathlib import Path

import pytest


def _load_cli():
    if importlib.util.find_spec("scai_client.cli") is None:
        pytest.fail("scai_client.cli is required for the unified entry point")
    return importlib.import_module("scai_client.cli")


def _parse(cli, *argv):
    return cli.build_parser().parse_args(list(argv))


def test_every_retained_subcommand_has_a_handler():
    cli = _load_cli()
    commands = (
        ["login"], ["new", "config.json"], ["list"], ["status", "s1"],
        ["watch", "s1"], ["start", "s1"], ["stop", "s1"],
        ["delete", "s1"], ["config", "s1"], ["artifacts", "s1"],
        ["import", "archive.tar.gz"], ["export", "s1"],
    )
    for argv in commands:
        assert callable(_parse(cli, *argv).handler), argv


def test_module_entrypoint_version_and_bare_invocation():
    cli = _load_cli()
    package = importlib.import_module("scai_client")
    root = Path(__file__).resolve().parents[3]
    constants = (root / "backend/constants.py").read_text()
    expected = re.search(r'^VERSION\s*=\s*["\']([^"\']+)', constants, re.MULTILINE)
    assert expected and package.__version__ == expected.group(1)
    with pytest.raises(SystemExit):
        cli.main([])


def test_stored_key_is_created_mode_0600(tmp_path, monkeypatch):
    _load_cli()
    credentials = importlib.import_module("scai_client.cli._credentials")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = credentials.write_key("scai_secret")
    assert path.read_text().strip() == "scai_secret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_print_jwt_stores_nothing(tmp_path, monkeypatch, capsys):
    cli = _load_cli()
    commands = importlib.import_module("scai_client.cli.commands")
    credentials = importlib.import_module("scai_client.cli._credentials")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        commands,
        "login",
        lambda *_args, **_kwargs: {
            "access_token": "jwt-token",
            "username": "u",
            "role": "user",
            "status": "active",
        },
    )

    assert commands.cmd_login(
        _parse(cli, "login", "--username", "u", "--password", "p", "--print-jwt")
    ) == 0
    assert capsys.readouterr().out.strip() == "jwt-token"
    assert not credentials.key_path().exists()
