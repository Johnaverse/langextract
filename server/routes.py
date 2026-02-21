"""Endpoint handlers for the LangExtract API."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

import langextract as lx
from langextract import data_lib
from langextract import io as lx_io
from langextract.core import data as lx_data

from server.models import (
    AnnotatedDocumentResponse,
    CharIntervalResponse,
    ExampleDataRequest,
    ExtractRequest,
    ExtractResponse,
    ExtractionResponse,
    HealthResponse,
)

router = APIRouter()

# Single source of truth — imported by main.py to keep version consistent.
VERSION = "1.1.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_examples(req_examples: list[ExampleDataRequest]) -> list[lx_data.ExampleData]:
    """Convert request ExampleDataRequest list to lx_data.ExampleData objects."""
    out = []
    for ex in req_examples:
        extractions = [
            lx_data.Extraction(
                extraction_class=e.extraction_class,
                extraction_text=e.extraction_text,
                char_interval=(
                    lx_data.CharInterval(
                        start_pos=e.char_interval.start_pos,
                        end_pos=e.char_interval.end_pos,
                    )
                    if e.char_interval
                    else None
                ),
                description=e.description,
                attributes=e.attributes,
            )
            for e in ex.extractions
        ]
        out.append(lx_data.ExampleData(text=ex.text, extractions=extractions))
    return out


def _serialize_adoc(adoc: Any) -> AnnotatedDocumentResponse:
    """Serialize an AnnotatedDocument to the response model."""
    raw = data_lib.annotated_document_to_dict(adoc)
    extractions = [
        ExtractionResponse(
            extraction_class=e.get("extraction_class", ""),
            extraction_text=e.get("extraction_text", ""),
            char_interval=(
                CharIntervalResponse(**e["char_interval"])
                if e.get("char_interval")
                else None
            ),
            alignment_status=e.get("alignment_status"),
            extraction_index=e.get("extraction_index"),
            group_index=e.get("group_index"),
            description=e.get("description"),
            attributes=e.get("attributes"),
        )
        for e in (raw.get("extractions") or [])
    ]
    return AnnotatedDocumentResponse(
        document_id=raw["document_id"],
        text=raw.get("text"),
        extractions=extractions,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


@router.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    # Resolve source text
    if req.url is not None:
        text = lx_io.download_text_from_url(req.url, show_progress=False)
    else:
        text = req.text

    examples = _build_examples(req.examples)

    result = lx.extract(
        text_or_documents=text,
        prompt_description=req.prompt_description,
        examples=examples,
        model_id=req.model_id,
        api_key=req.api_key,
        model_url=req.model_url,
        max_char_buffer=req.max_char_buffer,
        temperature=req.temperature,
        batch_length=req.batch_length,
        max_workers=req.max_workers,
        additional_context=req.additional_context,
        extraction_passes=req.extraction_passes,
        context_window_chars=req.context_window_chars,
        use_schema_constraints=req.use_schema_constraints,
        show_progress=False,
        fetch_urls=False,
    )

    if isinstance(result, list):
        docs = [_serialize_adoc(d) for d in result]
    else:
        docs = [_serialize_adoc(result)]

    return ExtractResponse(documents=docs)
