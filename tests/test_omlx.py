"""Tests for the oMLX provider."""

import unittest
from unittest import mock

from langextract.core import exceptions


class OMLXInputTokenLimitTest(unittest.TestCase):
  """Tests for max_input_tokens enforcement in OMLXLanguageModel."""

  @mock.patch('openai.OpenAI')
  def test_default_max_input_tokens(self, mock_openai_cls):
    """Default max_input_tokens should be 65536."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model')
    self.assertEqual(model.max_input_tokens, 65536)

  @mock.patch('openai.OpenAI')
  def test_custom_max_input_tokens(self, mock_openai_cls):
    """max_input_tokens should be configurable."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=32768)
    self.assertEqual(model.max_input_tokens, 32768)

  @mock.patch('openai.OpenAI')
  def test_check_input_tokens_under_limit(self, mock_openai_cls):
    """Should not raise when input is under the limit."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=1000)
    messages = [{'role': 'user', 'content': 'short prompt'}]
    # Should not raise.
    model._check_input_tokens(messages)

  @mock.patch('openai.OpenAI')
  def test_check_input_tokens_over_limit(self, mock_openai_cls):
    """Should raise InferenceConfigError when input exceeds limit."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=10)
    # 200 chars / 4 = 50 estimated tokens, well over limit of 10
    long_content = 'a' * 200
    messages = [{'role': 'user', 'content': long_content}]

    with self.assertRaises(exceptions.InferenceConfigError) as cm:
      model._check_input_tokens(messages)
    self.assertIn('max_input_tokens', str(cm.exception))
    self.assertIn('OOM', str(cm.exception))

  @mock.patch('openai.OpenAI')
  def test_check_input_tokens_disabled_when_zero(self, mock_openai_cls):
    """Setting max_input_tokens=0 should disable the check."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=0)
    long_content = 'a' * 1000000
    messages = [{'role': 'user', 'content': long_content}]
    # Should not raise.
    model._check_input_tokens(messages)

  @mock.patch('openai.OpenAI')
  def test_check_counts_all_messages(self, mock_openai_cls):
    """Token estimation should sum across all messages."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=10)
    # Two messages: 100 + 100 = 200 chars / 4 = 50 tokens > 10
    messages = [
        {'role': 'system', 'content': 'a' * 100},
        {'role': 'user', 'content': 'b' * 100},
    ]

    with self.assertRaises(exceptions.InferenceConfigError):
      model._check_input_tokens(messages)

  @mock.patch('openai.OpenAI')
  def test_process_single_prompt_rejects_large_input(self, mock_openai_cls):
    """_process_single_prompt should raise before calling the API."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=10)
    long_prompt = 'x' * 500

    with self.assertRaises(exceptions.InferenceRuntimeError) as cm:
      model._process_single_prompt(long_prompt, {})
    self.assertIn('max_input_tokens', str(cm.exception))

    # Verify the OpenAI client was never called.
    mock_openai_cls.return_value.chat.completions.create.assert_not_called()

  @mock.patch('openai.OpenAI')
  def test_recommended_max_char_buffer(self, mock_openai_cls):
    """recommended_max_char_buffer should be ~70% of token budget * 4."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=65536)
    # 65536 * 0.7 = 45875.2 -> int = 45875, * 4 = 183500
    expected = int(65536 * 0.7) * 4
    self.assertEqual(model.recommended_max_char_buffer, expected)

  @mock.patch('openai.OpenAI')
  def test_recommended_max_char_buffer_disabled(self, mock_openai_cls):
    """recommended_max_char_buffer should return 1000 when limit is 0."""
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(model_id='test-model', max_input_tokens=0)
    self.assertEqual(model.recommended_max_char_buffer, 1000)


if __name__ == '__main__':
  unittest.main()
