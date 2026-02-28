"""Pydantic request and response schemas for the LangExtract API."""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Module-level defaults — read once at startup, overridable via env vars.
# Each can be overridden per-request by supplying the field in the JSON body.
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID: str = os.getenv("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
_DEFAULT_API_KEY: str | None = os.getenv("LANGEXTRACT_API_KEY")
_DEFAULT_MODEL_URL: str | None = os.getenv("LANGEXTRACT_MODEL_URL")

_DEFAULT_MAX_CHAR_BUFFER: int = int(os.getenv("LANGEXTRACT_MAX_CHAR_BUFFER", "1000"))


def _default_temperature() -> float | None:
    raw = os.getenv("LANGEXTRACT_TEMPERATURE", "").strip()
    return float(raw) if raw else None


# ---------------------------------------------------------------------------
# Sub-models used inside requests (mirror langextract.core.data)
# ---------------------------------------------------------------------------

class CharIntervalRequest(BaseModel):
    start_pos: int | None = None
    end_pos: int | None = None


class ExtractionRequest(BaseModel):
    extraction_class: str
    extraction_text: str
    char_interval: CharIntervalRequest | None = None
    description: str | None = None
    attributes: dict[str, str | list[str]] | None = None


class ExampleDataRequest(BaseModel):
    text: str
    extractions: list[ExtractionRequest] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Main request body
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    # Input source — exactly one of text or url must be provided
    text: str | None = Field(default=None, description="Source text to extract from.")
    url: str | None = Field(
        default=None,
        description="URL to fetch source text from. Provide either text or url, not both.",
    )

    # Required extraction parameters
    prompt_description: str = Field(
        ..., description="Natural-language instructions describing what to extract."
    )
    examples: list[ExampleDataRequest] = Field(
        ..., min_length=1, description="At least one few-shot example is required."
    )

    # Model selection
    model_id: str = Field(
        default_factory=lambda: _DEFAULT_MODEL_ID,
        description=(
            "LLM model identifier recognized by LangExtract. "
            "Defaults to LANGEXTRACT_MODEL_ID env var, or 'gemini-2.5-flash'."
        ),
    )
    api_key: str | None = Field(
        default_factory=lambda: _DEFAULT_API_KEY,
        description=(
            "Provider API key. Defaults to LANGEXTRACT_API_KEY env var, "
            "then GEMINI_API_KEY / OPENAI_API_KEY env vars."
        ),
    )
    model_url: str | None = Field(
        default_factory=lambda: _DEFAULT_MODEL_URL,
        description=(
            "Base URL for self-hosted or on-prem models. "
            "Use for Ollama (e.g. http://localhost:11434) or "
            "LM Studio (e.g. http://localhost:1234/v1). "
            "Defaults to LANGEXTRACT_MODEL_URL env var."
        ),
    )

    # Tuning knobs — same defaults as lx.extract(), overridable via env vars
    max_char_buffer: int = Field(default_factory=lambda: _DEFAULT_MAX_CHAR_BUFFER, gt=0)
    temperature: float | None = Field(default_factory=_default_temperature, ge=0.0, le=2.0)
    batch_length: int = Field(default=10, gt=0)
    max_workers: int = Field(default=10, gt=0)
    additional_context: str | None = None
    extraction_passes: int = Field(default=1, ge=1)
    context_window_chars: int | None = Field(default=None, ge=0)
    use_schema_constraints: bool = True

    @model_validator(mode="after")
    def _validate_input_source(self) -> "ExtractRequest":
        has_text = self.text is not None
        has_url = self.url is not None
        if has_text == has_url:  # both set or neither set
            raise ValueError("Provide exactly one of 'text' or 'url', not both (or neither).")
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CharIntervalResponse(BaseModel):
    start_pos: int | None
    end_pos: int | None


class ExtractionResponse(BaseModel):
    extraction_class: str
    extraction_text: str
    char_interval: CharIntervalResponse | None
    alignment_status: str | None
    extraction_index: int | None
    group_index: int | None
    description: str | None
    attributes: dict[str, Any] | None


class AnnotatedDocumentResponse(BaseModel):
    document_id: str
    text: str | None
    extractions: list[ExtractionResponse]


class ExtractResponse(BaseModel):
    documents: list[AnnotatedDocumentResponse]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
