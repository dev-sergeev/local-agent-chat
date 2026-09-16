from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_openai import StreamChunkTimeoutError

from .providers import ModelStreamTimeout, create_chat_model
from .settings import LLMRetryConfig, ModelProfile

T = TypeVar("T")
ModelFactory = Callable[..., Any]

_RESERVED_MODEL_ARGUMENTS = frozenset(
    {
        "model",
        "api_key",
        "access_token",
        "credentials",
        "user",
        "password",
        "auth_url",
        "base_url",
        "max_retries",
        "timeout",
        "stream_chunk_timeout",
    }
)


@dataclass(frozen=True)
class RetryBlock:
    """Apply bounded recovery policies to every LLM model.

    The provider SDK owns transient-error classification, backoff, and
    `Retry-After` handling for one inference. A separate budget may resume a
    model handler after a zero-chunk stream timeout. It never replays an Agent
    graph, Turn, middleware side effect, or tool execution.
    """

    config: LLMRetryConfig
    model_factory: ModelFactory = create_chat_model

    def create_model(self, profile: ModelProfile, **kwargs: Any) -> Any:
        """Create a model whose provider retry behavior cannot be bypassed."""

        reserved_overrides = _RESERVED_MODEL_ARGUMENTS.intersection(kwargs)
        if reserved_overrides:
            names = ", ".join(sorted(reserved_overrides))
            raise ValueError(f"RetryBlock reserves model arguments: {names}")

        credentials: dict[str, str] = {}
        if profile.api_key:
            credentials["api_key"] = profile.api_key
        if profile.base_url:
            credentials["base_url"] = profile.base_url

        return self.model_factory(
            profile.model,
            **kwargs,
            **credentials,
            max_retries=self.config.max_retries,
            timeout=self.config.request_timeout_seconds,
            stream_chunk_timeout=self.config.stream_chunk_timeout_seconds,
        )

    async def run_auxiliary(self, awaitable_factory: Callable[[], Awaitable[T]]) -> T:
        """Run a non-Turn LLM operation inside one total timeout budget."""

        async with asyncio.timeout(self.config.auxiliary_timeout_seconds):
            return await awaitable_factory()

    async def run_streaming_model(
        self, awaitable_factory: Callable[[], Awaitable[T]]
    ) -> T:
        """Retry only a model handler that timed out before its first chunk."""

        retries = 0
        while True:
            try:
                return await awaitable_factory()
            except (StreamChunkTimeoutError, ModelStreamTimeout) as error:
                if error.chunks_received != 0 or retries >= self.config.stream_retries:
                    raise
                retries += 1
