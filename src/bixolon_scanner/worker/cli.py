from __future__ import annotations

import uvicorn

from .api import create_app
from .logging import configure_logging
from .settings import WorkerSettings


def serve() -> None:
    settings = WorkerSettings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings=settings), host=settings.host, port=settings.port, log_config=None
    )
