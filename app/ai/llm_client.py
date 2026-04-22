"""OpenAI-compatible LLM client. Defaults to NVIDIA NIM; swap endpoints via env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class LLMConfig:
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "google/gemma-3-27b-it"
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            model=os.getenv("LLM_MODEL", "google/gemma-3-27b-it"),
            api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )


class LLMClient:
    """Thin wrapper over the OpenAI-compatible NVIDIA NIM endpoint."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client: Any = None

    def _lazy_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Run `pip install openai>=1.40`."
            ) from exc

        if not self.config.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Get a free key at "
                "https://build.nvidia.com and export it in .env."
            )
        self._client = OpenAI(
            base_url=self.config.base_url, api_key=self.config.api_key
        )
        logger.info(
            f"LLMClient ready | model={self.config.model} base_url={self.config.base_url}"
        )
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> Any:
        """Single-shot chat completion. Returns the raw OpenAI response object."""
        client = self._lazy_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return client.chat.completions.create(**kwargs)

    def chat_text(self, prompt: str, system: str | None = None) -> str:
        """Convenience: plain-text in, plain-text out."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.chat(messages)
        return response.choices[0].message.content or ""
