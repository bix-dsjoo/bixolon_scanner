from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from bixolon_scanner.worker import cli as worker_cli
from bixolon_scanner.worker import logging as worker_logging


def test_configure_logging_uses_null_handler_without_stderr(monkeypatch) -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        monkeypatch.setattr(worker_logging.sys, "stderr", None)

        worker_logging.configure_logging("WARNING")

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.NullHandler)
        assert root.level == logging.WARNING
    finally:
        root.handlers.clear()
        root.handlers.extend(previous_handlers)
        root.setLevel(previous_level)


def test_configure_logging_can_explicitly_disable_stderr() -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        worker_logging.configure_logging("INFO", use_stderr=False)

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.NullHandler)
    finally:
        root.handlers.clear()
        root.handlers.extend(previous_handlers)
        root.setLevel(previous_level)


def test_serve_uses_windows_selector_loop_and_disables_websockets(monkeypatch) -> None:
    settings = SimpleNamespace(log_level="INFO", host="127.0.0.1", port=8123)
    app = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(worker_cli, "WorkerSettings", lambda: settings)
    monkeypatch.setattr(worker_cli, "configure_logging", lambda _level, **_kwargs: None)
    monkeypatch.setattr(worker_cli, "create_app", lambda *, settings: app)
    monkeypatch.setattr(worker_cli.sys, "platform", "win32")

    def fake_run(received_app: object, **kwargs: object) -> None:
        captured["app"] = received_app
        captured.update(kwargs)

    monkeypatch.setattr(worker_cli.uvicorn, "run", fake_run)

    worker_cli.serve()

    assert captured["app"] is app
    assert captured["ws"] == "none"
    assert captured["loop"] is worker_cli._windows_selector_loop_factory
    loop = worker_cli._windows_selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()
