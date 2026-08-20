"""Unified CLI with compatibility access to the Worker command."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable

from .command_registry import (
    ACTIVE_COMMANDS,
    COMMANDS,
    COMPATIBILITY_COMMANDS,
    DIAGNOSTIC_COMMANDS,
    LEGACY_COMMANDS,
    CommandTarget,
)
from .worker.cli import serve


def _append_commands(
    lines: list[str], title: str, registry: dict[tuple[str, ...], CommandTarget]
) -> None:
    lines.extend((title, *(f"  {' '.join(path)}" for path in sorted(registry)), ""))


def _help(*, include_diagnostics: bool = False, include_legacy: bool = False) -> str:
    lines = ["usage: bixolon <group> <command> [options]", ""]
    _append_commands(lines, "active commands:", ACTIVE_COMMANDS)
    if include_diagnostics:
        _append_commands(lines, "operational compatibility commands:", COMPATIBILITY_COMMANDS)
        _append_commands(lines, "optional diagnostics:", DIAGNOSTIC_COMMANDS)
    else:
        lines.append("Use `bixolon --help-diagnostics` to list optional tools.")
    if include_legacy:
        _append_commands(lines, "legacy compatibility commands:", LEGACY_COMMANDS)
    else:
        lines.append("Use `bixolon --help-legacy` to list archived compatibility commands.")
    lines.append("")
    lines.append("Use `bixolon <group> <command> --help` for command-specific options.")
    return "\n".join(lines)


def _resolve(argv: list[str]) -> tuple[tuple[str, ...], list[str], CommandTarget] | None:
    for length in (2, 1):
        path = tuple(argv[:length])
        target = COMMANDS.get(path)
        if target is not None:
            return path, argv[length:], target
    return None


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        print(_help())
        return
    if arguments == ["--help-legacy"]:
        print(_help(include_legacy=True))
        return
    if arguments == ["--help-diagnostics"]:
        print(_help(include_diagnostics=True))
        return
    resolved = _resolve(arguments)
    if resolved is None:
        print(_help(), file=sys.stderr)
        raise SystemExit(2)
    path, remaining, (module_name, function_name) = resolved
    module = import_module(module_name)
    command: Callable[[], None] = getattr(module, function_name)
    original = sys.argv
    try:
        sys.argv = [f"bixolon {' '.join(path)}", *remaining]
        command()
    finally:
        sys.argv = original


__all__ = [
    "ACTIVE_COMMANDS",
    "COMMANDS",
    "COMPATIBILITY_COMMANDS",
    "DIAGNOSTIC_COMMANDS",
    "LEGACY_COMMANDS",
    "main",
    "serve",
]
