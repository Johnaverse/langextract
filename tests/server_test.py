"""Unit tests for the LangExtract FastAPI server."""
from __future__ import annotations

import json
import os
from unittest import mock

import requests as http_requests
from absl.testing import absltest
from absl.testing import parameterized
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server.main import create_app
from server.models import (
    ExampleDataRequest,
    ExtractRequest,
    ExtractionRequest,
)
from langextract.core import exceptions as lx_exc
from langextract.prompt_validation import PromptAlignmentError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> TestClient:
    """Create a fresh TestClient for each test."""
    return TestClient(create_app(), raise_server_exceptions=False)


def _minimal_payload(**overrides) -> dict:
    """Minimal valid /extract request payload."""
    base = {
        "text": "Marie Curie was a physicist.",
        "prompt_description": "Extract people.",
        "examples": [
            {
                "text": "Isaac Newton studied physics.",
                "extractions": [
                    {"extraction_class": "person", "extraction_text": "Isaac Newton"}
                ],
            }
        ],
        "model_id": "gemini-2.5-flash",
        "api_key": "test-key",
    }
    base.update(overrides)
    return base


def _fake_annotated_doc() -> dict:
    """Return a minimal fake AnnotatedDocument dict as data_lib would produce."""
    return {
        "document_id": "doc_abc123",
        "text": "Marie Curie was a physicist.",
        "extractions": [
            {
                "extraction_class": "person",
                "extraction_text": "Marie Curie",
                "char_interval": {"start_pos": 0, "end_pos": 11},
                "alignment_status": "match_exact",
                "extraction_index": 0,
                "group_index": None,
                "description": None,
                "attributes": None,
            }
        ],
    }


class _FakeAnnotatedDoc:
    """Duck-typed stand-in for AnnotatedDocument used to satisfy isinstance checks."""


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------

class TestExtractRequestValidation(absltest.TestCase):
    """Tests for Pydantic schema validation on ExtractRequest."""

    def _make_example(self) -> ExampleDataRequest:
        return ExampleDataRequest(
            text="foo",
            extractions=[
                ExtractionRequest(extraction_class="thing", extraction_text="foo")
            ],
        )

    def test_valid_text_request_passes(self):
        req = ExtractRequest(
            text="hello world",
            prompt_description="Extract things.",
            examples=[self._make_example()],
        )
        self.assertEqual(req.text, "hello world")
        self.assertIsNone(req.url)

    def test_valid_url_request_passes(self):
        req = ExtractRequest(
            url="https://example.com/doc.txt",
            prompt_description="Extract things.",
            examples=[self._make_example()],
        )
        self.assertEqual(req.url, "https://example.com/doc.txt")
        self.assertIsNone(req.text)

    def test_both_text_and_url_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractRequest(
                text="some text",
                url="https://example.com/doc.txt",
                prompt_description="Extract things.",
                examples=[self._make_example()],
            )

    def test_neither_text_nor_url_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractRequest(
                prompt_description="Extract things.",
                examples=[self._make_example()],
            )

    def test_empty_examples_list_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractRequest(
                text="some text",
                prompt_description="Extract things.",
                examples=[],
            )

    def test_defaults_are_applied(self):
        req = ExtractRequest(
            text="hello",
            prompt_description="Extract.",
            examples=[self._make_example()],
        )
        self.assertEqual(req.model_id, "gemini-2.5-flash")
        self.assertEqual(req.max_char_buffer, 1000)
        self.assertEqual(req.batch_length, 10)
        self.assertEqual(req.max_workers, 10)
        self.assertEqual(req.extraction_passes, 1)
        self.assertTrue(req.use_schema_constraints)
        self.assertIsNone(req.api_key)
        self.assertIsNone(req.temperature)

    def test_negative_max_char_buffer_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractRequest(
                text="hello",
                prompt_description="Extract.",
                examples=[self._make_example()],
                max_char_buffer=0,
            )

    def test_temperature_out_of_range_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractRequest(
                text="hello",
                prompt_description="Extract.",
                examples=[self._make_example()],
                temperature=3.0,
            )


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint(absltest.TestCase):
    """Tests for GET /health."""

    def setUp(self):
        super().setUp()
        self.client = _make_client()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_returns_ok_status(self):
        resp = self.client.get("/health")
        body = resp.json()
        self.assertEqual(body["status"], "ok")

    def test_health_returns_version(self):
        resp = self.client.get("/health")
        body = resp.json()
        self.assertIn("version", body)
        self.assertIsInstance(body["version"], str)


# ---------------------------------------------------------------------------
# Extract endpoint — happy path tests
# ---------------------------------------------------------------------------

class TestExtractEndpointSuccess(absltest.TestCase):
    """Tests for POST /extract happy paths."""

    def setUp(self):
        super().setUp()
        self.client = _make_client()

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_text_returns_200(self, mock_extract, mock_to_dict):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 200)

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_returns_documents_list(self, mock_extract, mock_to_dict):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        resp = self.client.post("/extract", json=_minimal_payload())
        body = resp.json()
        self.assertIn("documents", body)
        self.assertIsInstance(body["documents"], list)
        self.assertLen(body["documents"], 1)

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_returns_extractions_with_char_interval(
        self, mock_extract, mock_to_dict
    ):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        resp = self.client.post("/extract", json=_minimal_payload())
        body = resp.json()
        extraction = body["documents"][0]["extractions"][0]
        self.assertEqual(extraction["extraction_class"], "person")
        self.assertEqual(extraction["extraction_text"], "Marie Curie")
        self.assertEqual(extraction["char_interval"]["start_pos"], 0)
        self.assertEqual(extraction["char_interval"]["end_pos"], 11)

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_passes_text_to_lx_extract(self, mock_extract, mock_to_dict):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        self.client.post("/extract", json=_minimal_payload(text="Custom input text."))

        call_kwargs = mock_extract.call_args.kwargs
        self.assertEqual(call_kwargs["text_or_documents"], "Custom input text.")

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_forces_show_progress_false(self, mock_extract, mock_to_dict):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        self.client.post("/extract", json=_minimal_payload())

        call_kwargs = mock_extract.call_args.kwargs
        self.assertFalse(call_kwargs["show_progress"])

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_forces_fetch_urls_false(self, mock_extract, mock_to_dict):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        self.client.post("/extract", json=_minimal_payload())

        call_kwargs = mock_extract.call_args.kwargs
        self.assertFalse(call_kwargs["fetch_urls"])

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_passes_optional_params(self, mock_extract, mock_to_dict):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        self.client.post(
            "/extract",
            json=_minimal_payload(max_char_buffer=500, temperature=0.2, extraction_passes=2),
        )

        call_kwargs = mock_extract.call_args.kwargs
        self.assertEqual(call_kwargs["max_char_buffer"], 500)
        self.assertEqual(call_kwargs["temperature"], 0.2)
        self.assertEqual(call_kwargs["extraction_passes"], 2)

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_handles_list_result_from_lx(self, mock_extract, mock_to_dict):
        """When lx.extract returns a list, all docs are included in response."""
        mock_extract.return_value = [_FakeAnnotatedDoc(), _FakeAnnotatedDoc()]
        mock_to_dict.return_value = _fake_annotated_doc()

        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertLen(resp.json()["documents"], 2)

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_with_no_extractions_returns_empty_list(
        self, mock_extract, mock_to_dict
    ):
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = {
            "document_id": "doc_empty",
            "text": "No entities here.",
            "extractions": [],
        }

        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertLen(resp.json()["documents"][0]["extractions"], 0)

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_extract_with_null_char_interval_in_response(
        self, mock_extract, mock_to_dict
    ):
        mock_extract.return_value = _FakeAnnotatedDoc()
        raw = _fake_annotated_doc()
        raw["extractions"][0]["char_interval"] = None
        mock_to_dict.return_value = raw

        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertIsNone(resp.json()["documents"][0]["extractions"][0]["char_interval"])


# ---------------------------------------------------------------------------
# Extract endpoint — URL fetch tests
# ---------------------------------------------------------------------------

class TestExtractEndpointUrlFetch(absltest.TestCase):
    """Tests for POST /extract when a url is provided instead of text."""

    def setUp(self):
        super().setUp()
        self.client = _make_client()

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    @mock.patch("server.routes.lx_io.download_text_from_url")
    def test_url_is_fetched_and_passed_to_extract(
        self, mock_download, mock_extract, mock_to_dict
    ):
        mock_download.return_value = "Fetched document text."
        mock_extract.return_value = _FakeAnnotatedDoc()
        mock_to_dict.return_value = _fake_annotated_doc()

        payload = _minimal_payload()
        del payload["text"]
        payload["url"] = "https://example.com/doc.txt"

        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 200)

        mock_download.assert_called_once_with(
            "https://example.com/doc.txt", show_progress=False
        )
        self.assertEqual(
            mock_extract.call_args.kwargs["text_or_documents"], "Fetched document text."
        )

    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    @mock.patch("server.routes.lx_io.download_text_from_url")
    def test_url_fetch_failure_returns_502(
        self, mock_download, mock_extract, mock_to_dict
    ):
        mock_download.side_effect = http_requests.RequestException("Connection refused")

        payload = _minimal_payload()
        del payload["text"]
        payload["url"] = "https://broken.example.com/doc.txt"

        resp = self.client.post("/extract", json=payload)
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()["error"]["code"], "URL_FETCH_ERROR")


# ---------------------------------------------------------------------------
# Extract endpoint — validation / input error tests
# ---------------------------------------------------------------------------

class TestExtractEndpointValidationErrors(absltest.TestCase):
    """Tests for input validation error responses."""

    def setUp(self):
        super().setUp()
        self.client = _make_client()

    def test_missing_examples_returns_422(self):
        payload = _minimal_payload()
        del payload["examples"]
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_empty_examples_returns_422(self):
        payload = _minimal_payload(examples=[])
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_both_text_and_url_returns_422(self):
        payload = _minimal_payload(url="https://example.com/doc.txt")
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_neither_text_nor_url_returns_422(self):
        payload = _minimal_payload()
        del payload["text"]
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_missing_prompt_description_returns_422(self):
        payload = _minimal_payload()
        del payload["prompt_description"]
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_invalid_json_body_returns_422(self):
        resp = self.client.post(
            "/extract",
            content=b"not json at all",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_max_char_buffer_zero_returns_422(self):
        payload = _minimal_payload(max_char_buffer=0)
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_temperature_above_2_returns_422(self):
        payload = _minimal_payload(temperature=2.5)
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)

    def test_extraction_passes_zero_returns_422(self):
        payload = _minimal_payload(extraction_passes=0)
        self.assertEqual(self.client.post("/extract", json=payload).status_code, 422)


# ---------------------------------------------------------------------------
# Extract endpoint — LLM error propagation tests
# ---------------------------------------------------------------------------

class TestExtractEndpointErrors(absltest.TestCase):
    """Tests that LangExtract exceptions map to correct HTTP status codes."""

    def setUp(self):
        super().setUp()
        self.client = _make_client()

    @mock.patch("server.routes.lx.extract")
    def test_inference_config_error_returns_400(self, mock_extract):
        mock_extract.side_effect = lx_exc.InferenceConfigError("Bad API key")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "INFERENCE_CONFIG_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_inference_runtime_error_returns_502(self, mock_extract):
        mock_extract.side_effect = lx_exc.InferenceRuntimeError("Timeout")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()["error"]["code"], "INFERENCE_RUNTIME_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_inference_output_error_returns_502(self, mock_extract):
        mock_extract.side_effect = lx_exc.InferenceOutputError("No output")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()["error"]["code"], "INFERENCE_OUTPUT_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_provider_error_returns_502(self, mock_extract):
        mock_extract.side_effect = lx_exc.ProviderError("Provider down")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()["error"]["code"], "PROVIDER_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_prompt_alignment_error_returns_422(self, mock_extract):
        mock_extract.side_effect = PromptAlignmentError("Alignment failed")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "PROMPT_ALIGNMENT_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_value_error_returns_422(self, mock_extract):
        mock_extract.side_effect = ValueError("examples required")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "INVALID_INPUT")

    @mock.patch("server.routes.lx.extract")
    def test_langextract_base_error_returns_500(self, mock_extract):
        mock_extract.side_effect = lx_exc.LangExtractError("Unknown LX error")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json()["error"]["code"], "LANGEXTRACT_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_unexpected_exception_returns_500(self, mock_extract):
        mock_extract.side_effect = RuntimeError("Unexpected crash")
        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json()["error"]["code"], "INTERNAL_ERROR")

    @mock.patch("server.routes.lx.extract")
    def test_error_response_has_message_field(self, mock_extract):
        mock_extract.side_effect = lx_exc.InferenceConfigError("Specific message")
        resp = self.client.post("/extract", json=_minimal_payload())
        body = resp.json()
        self.assertIn("message", body["error"])
        self.assertIn("Specific message", body["error"]["message"])


# ---------------------------------------------------------------------------
# Body size middleware tests
# ---------------------------------------------------------------------------

class TestBodySizeMiddleware(absltest.TestCase):
    """Tests for the configurable body size limit middleware."""

    def test_request_within_limit_is_allowed(self):
        with mock.patch.dict(os.environ, {"MAX_BODY_SIZE_MB": "10"}):
            client = _make_client()
        small_body = json.dumps(_minimal_payload()).encode()
        resp = client.post(
            "/extract",
            content=small_body,
            headers={
                "content-type": "application/json",
                "content-length": str(len(small_body)),
            },
        )
        self.assertNotEqual(resp.status_code, 413)

    def test_request_exceeding_limit_returns_413(self):
        with mock.patch.dict(os.environ, {"MAX_BODY_SIZE_MB": "1"}):
            client = _make_client()
        oversized_body = b"x" * (2 * 1024 * 1024)  # 2 MB
        resp = client.post(
            "/extract",
            content=oversized_body,
            headers={
                "content-type": "application/json",
                "content-length": str(len(oversized_body)),
            },
        )
        self.assertEqual(resp.status_code, 413)

    def test_413_response_has_error_envelope(self):
        with mock.patch.dict(os.environ, {"MAX_BODY_SIZE_MB": "1"}):
            client = _make_client()
        oversized_body = b"x" * (2 * 1024 * 1024)
        resp = client.post(
            "/extract",
            content=oversized_body,
            headers={
                "content-type": "application/json",
                "content-length": str(len(oversized_body)),
            },
        )
        body = resp.json()
        self.assertIn("error", body)
        self.assertEqual(body["error"]["code"], "REQUEST_TOO_LARGE")

    @mock.patch.dict(os.environ, {"MAX_BODY_SIZE_MB": "100"})
    def test_custom_limit_from_env_var_is_respected(self):
        client = _make_client()
        large_body = b"x" * (50 * 1024 * 1024)  # 50 MB — within 100 MB limit
        resp = client.post(
            "/extract",
            content=large_body,
            headers={
                "content-type": "application/json",
                "content-length": str(len(large_body)),
            },
        )
        self.assertNotEqual(resp.status_code, 413)


# ---------------------------------------------------------------------------
# Parameterized tests: extraction attributes in response
# ---------------------------------------------------------------------------

class TestExtractionResponseFields(parameterized.TestCase):
    """Verify each field of ExtractionResponse is correctly serialized."""

    def setUp(self):
        super().setUp()
        self.client = _make_client()

    @parameterized.named_parameters(
        dict(
            testcase_name="string_attributes",
            attributes={"dosage": "10mg", "route": "oral"},
        ),
        dict(
            testcase_name="list_attributes",
            attributes={"symptoms": ["fever", "cough"]},
        ),
        dict(
            testcase_name="null_attributes",
            attributes=None,
        ),
    )
    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_attributes_are_returned_correctly(
        self, mock_extract, mock_to_dict, attributes
    ):
        mock_extract.return_value = _FakeAnnotatedDoc()
        raw = _fake_annotated_doc()
        raw["extractions"][0]["attributes"] = attributes
        mock_to_dict.return_value = raw

        resp = self.client.post("/extract", json=_minimal_payload())
        self.assertEqual(resp.status_code, 200)
        returned_attrs = resp.json()["documents"][0]["extractions"][0]["attributes"]
        self.assertEqual(returned_attrs, attributes)

    @parameterized.named_parameters(
        dict(testcase_name="match_exact", status="match_exact"),
        dict(testcase_name="match_fuzzy", status="match_fuzzy"),
        dict(testcase_name="match_lesser", status="match_lesser"),
        dict(testcase_name="null_status", status=None),
    )
    @mock.patch("server.routes.data_lib.annotated_document_to_dict")
    @mock.patch("server.routes.lx.extract")
    def test_alignment_status_is_returned_correctly(
        self, mock_extract, mock_to_dict, status
    ):
        mock_extract.return_value = _FakeAnnotatedDoc()
        raw = _fake_annotated_doc()
        raw["extractions"][0]["alignment_status"] = status
        mock_to_dict.return_value = raw

        resp = self.client.post("/extract", json=_minimal_payload())
        returned_status = resp.json()["documents"][0]["extractions"][0][
            "alignment_status"
        ]
        self.assertEqual(returned_status, status)


if __name__ == "__main__":
    absltest.main()
