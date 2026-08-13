from __future__ import annotations

import uvicorn

from .api import create_app
from .config import WorkerSettings
from .observability import configure_logging


def serve() -> None:
    settings = WorkerSettings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings=settings), host=settings.host, port=settings.port, log_config=None
    )
