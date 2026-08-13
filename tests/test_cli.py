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
