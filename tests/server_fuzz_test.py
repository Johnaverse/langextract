"""Fuzz / property-based tests for the LangExtract FastAPI server.

Uses the Hypothesis library to generate random inputs and verify that:
  - The server never crashes (always returns a valid JSON response)
  - Input validation is exhaustive (invalid inputs always → 4xx, not 5xx)
  - The error envelope shape is always consistent

Install Hypothesis before running:
  pip install hypothesis
"""
from __future__ import annotations

import json
import string
from unittest import mock

from absl.testing import absltest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from server.main import create_app


# ---------------------------------------------------------------------------
# Shared client (created once per module to avoid repeated startup overhead)
# ---------------------------------------------------------------------------

_APP = create_app()
_CLIENT = TestClient(_APP, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Printable text that could plausibly appear in real documents
_text_st = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),  # no surrogates
    min_size=1,
    max_size=2000,
)

# Arbitrary strings including edge cases (empty, unicode, control chars)
_any_str_st = st.text(min_size=0, max_size=500)

# A valid single extraction for use inside examples
_extraction_st = st.fixed_dict(
    {
        "extraction_class": st.text(
            alphabet=string.ascii_lowercase + "_", min_size=1, max_size=50
        ),
        "extraction_text": _text_st,
    }
)

# A valid example entry
_example_st = st.fixed_dict(
    {
        "text": _text_st,
        "extractions": st.lists(_extraction_st, min_size=1, max_size=5),
    }
)

# A minimal valid payload with generated text and prompt
_valid_payload_st = st.fixed_dict(
    {
        "text": _text_st,
        "prompt_description": _text_st,
        "examples": st.lists(_example_st, min_size=1, max_size=3),
        "model_id": st.sampled_from(
            ["gemini-2.5-flash", "gpt-4o-mini", "llama3.2:1b"]
        ),
        "api_key": st.just("test-key"),
    }
)


def _fake_adoc_dict(text: str = "text") -> dict:
    return {
        "document_id": "doc_fuzz",
        "text": text,
        "extractions": [],
    }


class _FakeDoc:
    """Minimal stand-in for AnnotatedDocument used in fuzz tests."""


# ---------------------------------------------------------------------------
# Fuzz: server never crashes on arbitrary text/prompt inputs
# ---------------------------------------------------------------------------

class TestExtractNeverCrashesOnValidStructure(absltest.TestCase):
    """With a mocked LLM, any well-structured payload must return 200."""

    @given(payload=_valid_payload_st)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5000,
    )
    def test_valid_structure_always_returns_200(self, payload):
        with (
            mock.patch("server.routes.lx.extract") as mock_extract,
            mock.patch(
                "server.routes.data_lib.annotated_document_to_dict"
            ) as mock_to_dict,
        ):
            mock_extract.return_value = _FakeDoc()
            mock_to_dict.return_value = _fake_adoc_dict(payload["text"])

            resp = _CLIENT.post("/extract", json=payload)
            self.assertEqual(
                resp.status_code,
                200,
                f"Expected 200 for valid payload, got {resp.status_code}: {resp.text}",
            )

    @given(payload=_valid_payload_st)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5000,
    )
    def test_valid_structure_response_is_valid_json(self, payload):
        with (
            mock.patch("server.routes.lx.extract") as mock_extract,
            mock.patch(
                "server.routes.data_lib.annotated_document_to_dict"
            ) as mock_to_dict,
        ):
            mock_extract.return_value = _FakeDoc()
            mock_to_dict.return_value = _fake_adoc_dict()

            resp = _CLIENT.post("/extract", json=payload)
            try:
                response_body = resp.json()
            except (ValueError, json.JSONDecodeError) as exc:
                self.fail(f"Response is not valid JSON: {exc}\nBody: {resp.text}")
            self.assertIn("documents", response_body)


# ---------------------------------------------------------------------------
# Fuzz: arbitrary bytes body never crashes the server
# ---------------------------------------------------------------------------

class TestExtractHandlesArbitraryBodies(absltest.TestCase):
    """The server must respond with a valid error for any garbage input body."""

    @given(body=st.binary(min_size=0, max_size=4096))
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=3000,
    )
    def test_arbitrary_bytes_body_returns_error_not_crash(self, body):
        resp = _CLIENT.post(
            "/extract",
            content=body,
            headers={"content-type": "application/json"},
        )
        # Must never be a 5xx caused by an unhandled exception
        self.assertNotEqual(
            resp.status_code,
            500,
            f"Server crashed (500) on binary body: {body[:50]!r}",
        )
        # Response must still be parseable JSON
        try:
            resp.json()
        except (ValueError, json.JSONDecodeError):
            self.fail(f"Server returned non-JSON for binary body: {resp.text[:200]}")


# ---------------------------------------------------------------------------
# Fuzz: missing or wrong-typed fields in JSON payload always yield 4xx
# ---------------------------------------------------------------------------

class TestExtractValidationIsExhaustive(absltest.TestCase):
    """Malformed payloads must always produce 4xx, never 2xx or 5xx."""

    @given(
        field=st.sampled_from(["prompt_description", "examples", "model_id"]),
        value=st.one_of(
            st.none(),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.binary(),
        ),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=3000,
    )
    def test_wrong_type_for_required_field_returns_4xx(self, field, value):
        payload = {
            "text": "Some text",
            "prompt_description": "Extract things.",
            "examples": [
                {
                    "text": "foo",
                    "extractions": [
                        {"extraction_class": "x", "extraction_text": "foo"}
                    ],
                }
            ],
        }
        payload[field] = value

        # binary values cannot be JSON-serialized — skip those cases
        try:
            serialized = json.dumps(payload)
        except (TypeError, ValueError):
            return

        resp = _CLIENT.post(
            "/extract",
            content=serialized.encode(),
            headers={"content-type": "application/json"},
        )
        self.assertGreaterEqual(resp.status_code, 400, "Expected error for bad field")
        self.assertLess(resp.status_code, 500, "Expected 4xx, not 5xx, for bad input")

    @given(
        extra_fields=st.dictionaries(
            keys=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
            values=_any_str_st,
            max_size=10,
        )
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=3000,
    )
    def test_extra_unknown_fields_do_not_crash(self, extra_fields):
        """Pydantic should silently ignore unknown fields; server must not crash."""
        payload = {
            "text": "Some text",
            "prompt_description": "Extract things.",
            "examples": [
                {
                    "text": "foo",
                    "extractions": [
                        {"extraction_class": "x", "extraction_text": "foo"}
                    ],
                }
            ],
            **extra_fields,
        }
        try:
            serialized = json.dumps(payload)
        except (TypeError, ValueError):
            return

        with (
            mock.patch("server.routes.lx.extract") as mock_extract,
            mock.patch(
                "server.routes.data_lib.annotated_document_to_dict"
            ) as mock_to_dict,
        ):
            mock_extract.return_value = _FakeDoc()
            mock_to_dict.return_value = _fake_adoc_dict()

            resp = _CLIENT.post(
                "/extract",
                content=serialized.encode(),
                headers={"content-type": "application/json"},
            )
            self.assertNotEqual(
                resp.status_code,
                500,
                f"Server crashed on extra fields: {extra_fields}",
            )


# ---------------------------------------------------------------------------
# Fuzz: numeric parameter boundary testing
# ---------------------------------------------------------------------------

class TestExtractNumericBoundaries(absltest.TestCase):
    """Property tests for numeric parameter validation."""

    @given(max_char_buffer=st.integers(min_value=-10000, max_value=10000))
    @settings(max_examples=100, deadline=3000)
    def test_max_char_buffer_validation(self, max_char_buffer):
        payload = {
            "text": "text",
            "prompt_description": "Extract.",
            "examples": [
                {
                    "text": "foo",
                    "extractions": [
                        {"extraction_class": "x", "extraction_text": "foo"}
                    ],
                }
            ],
            "api_key": "key",
            "max_char_buffer": max_char_buffer,
        }
        resp = _CLIENT.post("/extract", json=payload)
        if max_char_buffer <= 0:
            self.assertEqual(
                resp.status_code,
                422,
                f"Expected 422 for max_char_buffer={max_char_buffer}",
            )
        else:
            self.assertNotEqual(
                resp.status_code,
                422,
                f"Unexpectedly rejected max_char_buffer={max_char_buffer}",
            )

    @given(temperature=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
    @settings(max_examples=100, deadline=3000)
    def test_temperature_validation(self, temperature):
        payload = {
            "text": "text",
            "prompt_description": "Extract.",
            "examples": [
                {
                    "text": "foo",
                    "extractions": [
                        {"extraction_class": "x", "extraction_text": "foo"}
                    ],
                }
            ],
            "api_key": "key",
            "temperature": temperature,
        }
        resp = _CLIENT.post("/extract", json=payload)
        if temperature < 0.0 or temperature > 2.0:
            self.assertEqual(
                resp.status_code,
                422,
                f"Expected 422 for temperature={temperature}",
            )
        else:
            self.assertNotEqual(
                resp.status_code,
                422,
                f"Unexpectedly rejected temperature={temperature}",
            )

    @given(extraction_passes=st.integers(min_value=-5, max_value=20))
    @settings(max_examples=50, deadline=3000)
    def test_extraction_passes_validation(self, extraction_passes):
        payload = {
            "text": "text",
            "prompt_description": "Extract.",
            "examples": [
                {
                    "text": "foo",
                    "extractions": [
                        {"extraction_class": "x", "extraction_text": "foo"}
                    ],
                }
            ],
            "api_key": "key",
            "extraction_passes": extraction_passes,
        }
        resp = _CLIENT.post("/extract", json=payload)
        if extraction_passes < 1:
            self.assertEqual(
                resp.status_code,
                422,
                f"Expected 422 for extraction_passes={extraction_passes}",
            )
        else:
            self.assertNotEqual(
                resp.status_code,
                422,
                f"Unexpectedly rejected extraction_passes={extraction_passes}",
            )


# ---------------------------------------------------------------------------
# Fuzz: error envelope shape is always consistent
# ---------------------------------------------------------------------------

class TestErrorEnvelopeShape(absltest.TestCase):
    """Every error response from the server must have the same envelope shape."""

    @given(
        payload=st.fixed_dict(
            {
                "text": _any_str_st,
                "prompt_description": _any_str_st,
                "examples": st.lists(_example_st, min_size=0, max_size=2),
                "max_char_buffer": st.integers(min_value=-100, max_value=100),
            }
        )
    )
    @settings(
        max_examples=150,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=3000,
    )
    def test_4xx_responses_always_have_error_envelope(self, payload):
        try:
            serialized = json.dumps(payload)
        except (TypeError, ValueError):
            return

        resp = _CLIENT.post(
            "/extract",
            content=serialized.encode(),
            headers={"content-type": "application/json"},
        )

        if 400 <= resp.status_code < 500:
            try:
                response_body = resp.json()
            except (ValueError, json.JSONDecodeError):
                self.fail(
                    f"4xx response was not valid JSON: status={resp.status_code}"
                )
            self.assertIn(
                "error",
                response_body,
                f"4xx response missing 'error' key: {response_body}",
            )
            self.assertIn(
                "code",
                response_body["error"],
                f"error envelope missing 'code': {response_body}",
            )
            self.assertIn(
                "message",
                response_body["error"],
                f"error envelope missing 'message': {response_body}",
            )


if __name__ == "__main__":
    absltest.main()
