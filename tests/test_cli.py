from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from bixolon_scanner import cli


def test_unified_cli_dispatches_arguments_without_reparsing(monkeypatch) -> None:
    observed: list[list[str]] = []
    fake = SimpleNamespace(main=lambda: observed.append(list(sys.argv)))
    monkeypatch.setattr(cli, "import_module", lambda name: fake)

    cli.main(["evaluate", "worker", "--provider", "cpu"])

    assert observed == [["bixolon evaluate worker", "--provider", "cpu"]]


def test_unified_cli_help_exposes_only_the_active_version_command(capsys) -> None:
    cli.main(["--help"])

    output = capsys.readouterr().out
    assert "active commands:" in output
    assert "bundle verify" in output
    assert "worker" not in output
    assert "evaluate" not in output
    assert "scanner-2.0" not in output
    assert "bread-1.1" not in output


def test_unified_cli_can_list_optional_diagnostics(capsys) -> None:
    cli.main(["--help-diagnostics"])

    output = capsys.readouterr().out
    assert "operational compatibility commands:" in output
    assert "optional diagnostics:" in output
    assert "worker" in output
    assert "evaluate benchmark" in output
    assert "scanner-2.0" not in output


def test_unified_cli_can_list_legacy_compatibility_commands(capsys) -> None:
    cli.main(["--help-legacy"])

    output = capsys.readouterr().out
    assert "legacy compatibility commands:" in output
    assert "evaluate scanner-2.0" in output
    assert "evaluate bread-1.1-runtime" in output


def test_command_support_registries_do_not_overlap() -> None:
    registries = (
        cli.ACTIVE_COMMANDS,
        cli.COMPATIBILITY_COMMANDS,
        cli.DIAGNOSTIC_COMMANDS,
        cli.LEGACY_COMMANDS,
    )

    assert cli.COMMANDS == {
        path: target for registry in registries for path, target in registry.items()
    }
    assert sum(map(len, registries)) == len(cli.COMMANDS)
    assert set(cli.ACTIVE_COMMANDS) == {("bundle", "verify")}


def test_unified_cli_rejects_unknown_commands() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["experiment", "archived-prototype"])

    assert caught.value.code == 2


def test_legacy_console_targets_and_unified_targets_share_callables() -> None:
    from bixolon_scanner.evaluation.worker import main as canonical
    from bixolon_scanner.training.evaluate_worker import main as legacy

    assert legacy is canonical


def test_scanner_2_commands_are_canonical() -> None:
    assert ("evaluate", "scanner-2.0") in cli.COMMANDS
    assert ("evaluate", "scanner-2.0-parity") in cli.COMMANDS
    assert ("evaluate", "scanner-2.0-embedder-parity") in cli.COMMANDS
    assert ("evaluate", "scanner-2.0-packaged-worker-smoke") in cli.COMMANDS
    assert ("model", "export-embedder") in cli.COMMANDS
    assert ("model", "export-dinov2-embedder") in cli.COMMANDS
    assert ("catalog", "activate") in cli.COMMANDS
    assert ("bundle", "verify") in cli.COMMANDS
    assert not any(path[0] == "release" for path in cli.COMMANDS)
    assert ("experiment", "bread-catalog-backbone-probe") not in cli.COMMANDS
