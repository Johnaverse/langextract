# LangExtract API Server

HTTP interface for the [LangExtract](https://github.com/google-gemini/langextract) structured
extraction library. POST unstructured text (or a URL) together with a few-shot prompt and receive
structured JSON extractions with precise character positions.

---

## Quick Start

### Docker (recommended)

```bash
docker pull ghcr.io/<your-org>/langextract/server:latest

docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your-key \
  ghcr.io/<your-org>/langextract/server:latest
```

### Local development

```bash
# From the repo root
pip install -r server/requirements.txt
uvicorn server.main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

---

## Environment Variables

All variables are optional. Request-body fields always take precedence over env-var defaults.

### Server configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_BODY_SIZE_MB` | `10` | Maximum request body size in MB. Returns 413 if exceeded. |

### Model defaults

Set these to avoid repeating model config in every request. Any field supplied in the request
body overrides the corresponding env var for that request.

| Variable | Default | Description |
|---|---|---|
| `LANGEXTRACT_MODEL_ID` | `gemini-2.5-flash` | Default LLM model identifier. |
| `LANGEXTRACT_API_KEY` | _(none)_ | Default API key for the chosen provider. |
| `LANGEXTRACT_MODEL_URL` | _(none)_ | Base URL for self-hosted models (Ollama / LM Studio). |
| `LANGEXTRACT_TEMPERATURE` | _(model default)_ | Sampling temperature, `0.0`–`2.0`. |
| `LANGEXTRACT_MAX_CHAR_BUFFER` | `1000` | Character buffer per extraction window. |

### Provider API keys and endpoints

Read directly by LangExtract. Set whichever matches your chosen model.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(none)_ | Google Gemini API key. |
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (auto-used for Ollama model IDs). |

### Docker example with model defaults

```bash
docker run -p 8000:8000 \
  -e LANGEXTRACT_MODEL_ID=gpt-4o-mini \
  -e OPENAI_API_KEY=sk-... \
  -e LANGEXTRACT_TEMPERATURE=0.2 \
  -e MAX_BODY_SIZE_MB=50 \
  ghcr.io/<your-org>/langextract/server:latest
```

---

## API Reference

### `GET /health`

Returns server status and version.

**Response `200`**

```json
{
  "status": "ok",
  "version": "1.1.1"
}
```

---

### `POST /extract`

Run structured extraction on a piece of text or a URL.

**Request body** (`application/json`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | `string` | one of `text`/`url` | — | Source text to extract from. |
| `url` | `string` | one of `text`/`url` | — | URL to fetch source text from. |
| `prompt_description` | `string` | yes | — | Natural-language extraction instructions. |
| `examples` | `Example[]` | yes (≥ 1) | — | Few-shot examples. |
| `model_id` | `string` | no | `LANGEXTRACT_MODEL_ID` | LLM model identifier. |
| `api_key` | `string` | no | `LANGEXTRACT_API_KEY` | Provider API key. |
| `model_url` | `string` | no | `LANGEXTRACT_MODEL_URL` | Base URL for self-hosted models (Ollama / LM Studio). |
| `temperature` | `float` | no | `LANGEXTRACT_TEMPERATURE` | Sampling temperature `0.0`–`2.0`. |
| `max_char_buffer` | `int > 0` | no | `LANGEXTRACT_MAX_CHAR_BUFFER` | Chars per extraction window. |
| `batch_length` | `int > 0` | no | `10` | Documents per LLM batch. |
| `max_workers` | `int > 0` | no | `10` | Parallel worker threads. |
| `extraction_passes` | `int ≥ 1` | no | `1` | Extraction passes over each chunk. |
| `additional_context` | `string` | no | — | Extra context injected into the prompt. |
| `context_window_chars` | `int ≥ 0` | no | — | Context window override in chars. |
| `use_schema_constraints` | `bool` | no | `true` | Enforce schema constraints on output. |

**`Example` object**

```json
{
  "text": "Alice joined Acme Corp in 2021.",
  "extractions": [
    {
      "extraction_class": "person",
      "extraction_text": "Alice"
    }
  ]
}
```

Each extraction inside an example also accepts optional `char_interval` (`start_pos`, `end_pos`),
`description`, and `attributes` fields.

**Response `200`**

```json
{
  "documents": [
    {
      "document_id": "doc_0",
      "text": "...",
      "extractions": [
        {
          "extraction_class": "person",
          "extraction_text": "Alice",
          "char_interval": { "start_pos": 0, "end_pos": 5 },
          "alignment_status": "exact",
          "extraction_index": 0,
          "group_index": 0,
          "description": null,
          "attributes": null
        }
      ]
    }
  ]
}
```

---

## Error Responses

All errors share the same envelope shape:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Human-readable description."
  }
}
```

| HTTP | Code | Cause |
|---|---|---|
| `400` | `INFERENCE_CONFIG_ERROR` | Bad model / API-key configuration. |
| `413` | `REQUEST_TOO_LARGE` | Body exceeds `MAX_BODY_SIZE_MB`. |
| `422` | `INVALID_INPUT` | Pydantic field validation error or bad input value. |
| `422` | `PROMPT_ALIGNMENT_ERROR` | Few-shot examples failed alignment checks. |
| `502` | `URL_FETCH_ERROR` | Failed to fetch the provided URL. |
| `502` | `INFERENCE_RUNTIME_ERROR` | LLM call failed at runtime. |
| `502` | `INFERENCE_OUTPUT_ERROR` | LLM returned unparseable output. |
| `502` | `PROVIDER_ERROR` | Provider-level error (rate limit, quota, etc.). |
| `500` | `LANGEXTRACT_ERROR` | Unexpected library error. |
| `500` | `INTERNAL_ERROR` | Unexpected server error. |

---

## curl Examples

### Basic extraction

```bash
curl -s http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice joined Acme Corp in 2021 as a software engineer.",
    "prompt_description": "Extract people and the companies they work for.",
    "examples": [
      {
        "text": "Bob works at Globex since 2019.",
        "extractions": [
          {"extraction_class": "person", "extraction_text": "Bob"},
          {"extraction_class": "company", "extraction_text": "Globex"}
        ]
      }
    ],
    "api_key": "YOUR_GEMINI_KEY"
  }' | jq .
```

### Extract from a URL

```bash
curl -s http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article.txt",
    "prompt_description": "Extract dates and events.",
    "examples": [
      {
        "text": "The conference was held on March 5, 2024.",
        "extractions": [
          {"extraction_class": "date", "extraction_text": "March 5, 2024"}
        ]
      }
    ]
  }'
```

### Override model per-request

```bash
curl -s http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "...",
    "prompt_description": "...",
    "examples": [...],
    "model_id": "gpt-4o-mini",
    "api_key": "sk-...",
    "temperature": 0.1
  }'
```

---

## Local Model Providers

### Ollama

Ollama is auto-detected from the model ID — any model name that matches an Ollama pattern
(e.g. `llama3.2:1b`, `gemma2:2b`, `mistral:7b`, `qwen2.5:7b`) or a HuggingFace-style ID
(e.g. `meta-llama/Llama-3.2-1B-Instruct`) is routed to Ollama automatically.

The `OLLAMA_BASE_URL` env var (default `http://localhost:11434`) is picked up automatically.
Override it per-request with `model_url`.

**Run Ollama locally and point the server at it:**

```bash
# Start Ollama and pull a model
ollama pull llama3.2:1b

# Run the server
docker run -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e LANGEXTRACT_MODEL_ID=llama3.2:1b \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  ghcr.io/<your-org>/langextract/server:latest
```

**Or override per-request:**

```bash
curl -s http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "...",
    "prompt_description": "...",
    "examples": [...],
    "model_id": "llama3.2:1b",
    "model_url": "http://localhost:11434"
  }'
```

---

### LM Studio

LM Studio exposes an OpenAI-compatible API. Use any `gpt-*` model ID (it is only used for
routing to the OpenAI-compatible provider) and point `model_url` at LM Studio's server.
LM Studio's default port is **1234**.

**Server-wide default (Docker):**

```bash
docker run -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e LANGEXTRACT_MODEL_ID=gpt-4o \
  -e LANGEXTRACT_MODEL_URL=http://host.docker.internal:1234/v1 \
  ghcr.io/<your-org>/langextract/server:latest
```

**Or override per-request:**

```bash
curl -s http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "...",
    "prompt_description": "...",
    "examples": [...],
    "model_id": "gpt-4o",
    "model_url": "http://localhost:1234/v1",
    "api_key": "lm-studio"
  }'
```

> `api_key` can be any non-empty string — LM Studio does not validate it.
