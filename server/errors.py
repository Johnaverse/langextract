"""Maps LangExtract exceptions to HTTP error responses."""
from __future__ import annotations

import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langextract.core import exceptions as lx_exc
from langextract.prompt_validation import PromptAlignmentError


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


def register_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app."""

    @app.exception_handler(ValueError)
    async def value_error_handler(req: Request, exc: ValueError) -> JSONResponse:
        return _err(422, "INVALID_INPUT", str(exc))

    @app.exception_handler(PromptAlignmentError)
    async def alignment_error_handler(
        req: Request, exc: PromptAlignmentError
    ) -> JSONResponse:
        return _err(422, "PROMPT_ALIGNMENT_ERROR", str(exc))

    @app.exception_handler(http_requests.RequestException)
    async def url_fetch_error_handler(
        req: Request, exc: http_requests.RequestException
    ) -> JSONResponse:
        return _err(502, "URL_FETCH_ERROR", str(exc))

    @app.exception_handler(lx_exc.InferenceConfigError)
    async def config_error_handler(
        req: Request, exc: lx_exc.InferenceConfigError
    ) -> JSONResponse:
        return _err(400, "INFERENCE_CONFIG_ERROR", str(exc))

    @app.exception_handler(lx_exc.InferenceRuntimeError)
    async def runtime_error_handler(
        req: Request, exc: lx_exc.InferenceRuntimeError
    ) -> JSONResponse:
        return _err(502, "INFERENCE_RUNTIME_ERROR", str(exc))

    @app.exception_handler(lx_exc.InferenceOutputError)
    async def output_error_handler(
        req: Request, exc: lx_exc.InferenceOutputError
    ) -> JSONResponse:
        return _err(502, "INFERENCE_OUTPUT_ERROR", str(exc))

    @app.exception_handler(lx_exc.ProviderError)
    async def provider_error_handler(
        req: Request, exc: lx_exc.ProviderError
    ) -> JSONResponse:
        return _err(502, "PROVIDER_ERROR", str(exc))

    @app.exception_handler(lx_exc.LangExtractError)
    async def langextract_error_handler(
        req: Request, exc: lx_exc.LangExtractError
    ) -> JSONResponse:
        return _err(500, "LANGEXTRACT_ERROR", str(exc))

    @app.exception_handler(Exception)
    async def generic_error_handler(req: Request, exc: Exception) -> JSONResponse:
        return _err(500, "INTERNAL_ERROR", "An unexpected error occurred.")
