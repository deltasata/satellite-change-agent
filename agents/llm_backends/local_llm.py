"""Variant (2) backend: an open-weight model served by vLLM on one A10, reached over
its OpenAI-compatible HTTP API. Structured outputs (guided decoding) make the server
emit only tokens that keep the JSON valid against the schema."""

from __future__ import annotations

import os

from openai import OpenAI


class LocalLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 thinking: bool | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        if thinking is None:  # experiment switch: EO_LLM_THINKING=1 lets Qwen3 reason before answering
            thinking = os.environ.get("EO_LLM_THINKING", "0") == "1"
        self.base_url = base_url or os.environ.get("EO_LLM_URL", "http://127.0.0.1:8101/v1")
        self.client = OpenAI(base_url=self.base_url, api_key="local-no-key")  # vLLM needs no key
        self.model = model or self.client.models.list().data[0].id
        self.name = f"local:{self.model}" + (":thinking" if thinking else "")
        self.thinking, self.temperature, self.max_tokens = thinking, temperature, max_tokens
        self.n_calls = 0

    def raw_json(self, system: str, prompt: str, json_schema: dict) -> str:
        self.n_calls += 1
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens + (2048 if self.thinking else 0),
            response_format={"type": "json_schema",
                             "json_schema": {"name": "answer", "schema": json_schema, "strict": True}},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        return r.choices[0].message.content or ""
