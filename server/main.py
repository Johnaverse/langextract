"""LangExtract FastAPI server entry point."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from server.errors import register_handlers
from server.routes import VERSION, router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Body-size limit middleware
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BODY_MB = 10


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds max_bytes with 413."""

    def __init__(self, app: FastAPI, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> JSONResponse:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": (
                            f"Request body exceeds the "
                            f"{self.max_bytes // (1024 * 1024)} MB limit."
                        ),
                    }
                },
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    max_mb = int(os.getenv("MAX_BODY_SIZE_MB", str(_DEFAULT_MAX_BODY_MB)))
    max_bytes = max_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        logger.info(
            "LangExtract API v%s started. Max body size: %d MB.", VERSION, max_mb
        )
        yield

    app = FastAPI(
        title="LangExtract API",
        description=(
            "HTTP API for the LangExtract structured extraction library. "
            "POST text or a URL with few-shot examples and receive structured "
            "extractions with precise character positions."
        ),
        version=VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(MaxBodySizeMiddleware, max_bytes=max_bytes)
    register_handlers(app)
    app.include_router(router)

    return app


app = create_app()
