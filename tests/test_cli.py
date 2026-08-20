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


def test_unified_cli_help_lists_stable_groups(capsys) -> None:
    cli.main(["--help"])

    output = capsys.readouterr().out
    for group in (
        "worker",
        "data",
        "train",
        "evaluate",
        "model",
        "experiment",
        "operations",
        "catalog",
        "release",
        "tools",
    ):
        assert group in output


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
    assert ("evaluate", "scanner-2.0-private-preflight") in cli.COMMANDS
    assert ("evaluate", "scanner-2.0-private") in cli.COMMANDS
    assert ("model", "export-embedder") in cli.COMMANDS
    assert ("model", "export-dinov2-embedder") in cli.COMMANDS
    assert ("catalog", "activate") in cli.COMMANDS
    assert ("release", "lock-scanner-2.0") in cli.COMMANDS
    assert ("release", "promote-scanner-2.0") in cli.COMMANDS
    assert ("release", "promote-scanner-2.0-owner-waiver") in cli.COMMANDS
    assert ("experiment", "bread-catalog-backbone-probe") not in cli.COMMANDS
