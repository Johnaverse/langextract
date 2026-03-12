"""oMLX provider for LangExtract.

This provider enables using local oMLX models with LangExtract's extract()
function. oMLX exposes an OpenAI-compatible API, so this provider reuses
the OpenAI client with oMLX-specific defaults.

No API key is required since oMLX runs locally on your machine.

Usage with extract():
    import langextract as lx

    result = lx.extract(
        text_or_documents="Isaac Asimov was a prolific science fiction writer.",
        model_id="Qwen3.5-35B-A3B-4bit",
        prompt_description="Extract the person's name and field",
        examples=[example],
    )

Direct provider instantiation:
    from langextract.providers.omlx import OMLXLanguageModel

    model = OMLXLanguageModel(
        model_id="Qwen3.5-35B-A3B-4bit",
        base_url="http://localhost:8000/v1",
    )

Prerequisites:
    1. Install oMLX
    2. Download and load a model (e.g., Qwen3.5-35B-A3B-4bit)
    3. Start the oMLX server
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
from typing import Any, Iterator, Sequence

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router

_OMLX_DEFAULT_BASE_URL = os.getenv(
    'OMLX_BASE_URL', 'http://localhost:8000/v1'
)


@router.register(
    *patterns.OMLX_PATTERNS,
    priority=patterns.OMLX_PRIORITY,
)
@dataclasses.dataclass(init=False)
class OMLXLanguageModel(base_model.BaseLanguageModel):
  """Language model inference using oMLX's OpenAI-compatible API."""

  model_id: str = 'Qwen3.5-35B-A3B-4bit'
  base_url: str = _OMLX_DEFAULT_BASE_URL
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  max_workers: int = 10
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @property
  def requires_fence_output(self) -> bool:
    """oMLX JSON mode returns raw JSON without fences."""
    if self.format_type == data.FormatType.JSON:
      return False
    return super().requires_fence_output

  def __init__(
      self,
      model_id: str = 'Qwen3.5-35B-A3B-4bit',
      base_url: str = _OMLX_DEFAULT_BASE_URL,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = None,
      max_workers: int = 10,
      api_key: str | None = None,
      **kwargs,
  ) -> None:
    """Initialize the oMLX language model.

    Args:
      model_id: The model ID loaded in oMLX (e.g., 'Qwen3.5-35B-A3B-4bit').
      base_url: Base URL for oMLX server. Defaults to http://localhost:8000/v1.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      api_key: API key for the oMLX server. Falls back to OMLX_API_KEY env var.
      **kwargs: Additional parameters passed through.
    """
    try:
      # pylint: disable=import-outside-toplevel
      import openai
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'oMLX provider requires openai package. '
          'Install with: pip install langextract[openai]'
      ) from e

    self.model_id = model_id
    self.base_url = base_url
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers

    # Use api_key from request, then env var, then placeholder
    resolved_api_key = api_key or os.getenv('OMLX_API_KEY') or 'omlx'
    self._client = openai.OpenAI(
        api_key=resolved_api_key,
        base_url=self.base_url,
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = kwargs or {}

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      system_message = ''
      if self.format_type == data.FormatType.JSON:
        system_message = (
            'You are a helpful assistant that responds in JSON format.'
        )
      elif self.format_type == data.FormatType.YAML:
        system_message = (
            'You are a helpful assistant that responds in YAML format.'
        )

      messages = [{'role': 'user', 'content': prompt}]
      if system_message:
        messages.insert(0, {'role': 'system', 'content': system_message})

      api_params = {
          'model': self.model_id,
          'messages': messages,
      }

      temp = config.get('temperature', self.temperature)
      if temp is not None:
        api_params['temperature'] = temp

      # Note: oMLX does not support response_format 'json_object'.
      # JSON output is guided by the system prompt instead.

      if (v := config.get('max_output_tokens')) is not None:
        api_params['max_tokens'] = v
      if (v := config.get('top_p')) is not None:
        api_params['top_p'] = v
      for key in [
          'frequency_penalty',
          'presence_penalty',
          'seed',
          'stop',
      ]:
        if (v := config.get(key)) is not None:
          api_params[key] = v

      response = self._client.chat.completions.create(**api_params)

      output_text = response.choices[0].message.content

      return core_types.ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'oMLX API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via oMLX's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    merged_kwargs = self.merge_kwargs(kwargs)

    config = {}

    temp = merged_kwargs.get('temperature', self.temperature)
    if temp is not None:
      config['temperature'] = temp
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']

    for key in [
        'frequency_penalty',
        'presence_penalty',
        'seed',
        'stop',
    ]:
      if key in merged_kwargs:
        config[key] = merged_kwargs[key]

    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[core_types.ScoredOutput | None] = [None] * len(
            batch_prompts
        )
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'Parallel inference error: {str(e)}', original=e
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]
