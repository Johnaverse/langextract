# ---- build stage: install all dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy only what pip needs to resolve deps before copying full source.
# This layer is cached as long as pyproject.toml / requirements don't change.
COPY pyproject.toml README.md ./
COPY langextract/ ./langextract/
COPY server/requirements.txt ./server/requirements.txt

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r server/requirements.txt


# ---- runtime stage: lean final image ----
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY langextract/ ./langextract/
COPY server/ ./server/
COPY pyproject.toml README.md ./

# Install the langextract package itself (no deps, already installed above)
RUN pip install --no-deps --no-cache-dir -e .

USER appuser

# ---- Server configuration ----
# Max request body size in MB (override with -e MAX_BODY_SIZE_MB=50)
ENV MAX_BODY_SIZE_MB=10

# ---- Model defaults (all overridable per-request in the JSON body) ----
# Default LLM model. E.g. gemini-2.5-flash | gpt-4o-mini | llama3.2:1b
ENV LANGEXTRACT_MODEL_ID=gemini-2.5-flash
# Default API key (optional; also accepts GEMINI_API_KEY / OPENAI_API_KEY)
ENV LANGEXTRACT_API_KEY=
# Default sampling temperature 0.0–2.0 (empty = model default)
ENV LANGEXTRACT_TEMPERATURE=
# Default character buffer per extraction window
ENV LANGEXTRACT_MAX_CHAR_BUFFER=1000
# Base URL for self-hosted models (Ollama or LM Studio). Empty = use provider default.
# Ollama example:    http://localhost:11434
# LM Studio example: http://host.docker.internal:1234/v1
ENV LANGEXTRACT_MODEL_URL=

# ---- Provider-specific (read directly by LangExtract) ----
# Ollama base URL — used automatically for ollama/* model IDs
ENV OLLAMA_BASE_URL=http://localhost:11434

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
