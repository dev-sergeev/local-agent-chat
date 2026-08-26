from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain_openai import StreamChunkTimeoutError

from .settings import LLMRetryConfig, ModelProfile

T = TypeVar("T")
ModelFactory = Callable[..., Any]

_RESERVED_MODEL_ARGUMENTS = frozenset(
    {
        "model",
        "api_key",
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
    model_factory: ModelFactory

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

    def summarization_middleware(self, model: Any, backend: Any) -> Any:
        """Create summarization with retry immediately around the model handler.

        Deep Agents composes LangChain's summarizer, which otherwise adds three
        broad `Runnable.with_retry()` attempts on top of the provider policy.
        Replacing that runnable keeps `LLM_MAX_RETRIES` authoritative while
        retaining context compaction and overflow recovery. The custom wrapper
        retries inside summarization, so its backend offloads run only once.
        """

        middleware = _ResilientSummarizationMiddleware(self, model, backend)
        helper = getattr(middleware, "_lc_helper", None)
        if helper is None or not hasattr(helper, "_summary_model"):
            raise RuntimeError(
                "Unsupported Deep Agents summarization middleware: "
                "cannot install the configured LLM retry policy"
            )
        helper._summary_model = model
        return middleware

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
            except StreamChunkTimeoutError as error:
                if error.chunks_received != 0 or retries >= self.config.stream_retries:
                    raise
                retries += 1


class _ResilientSummarizationMiddleware(SummarizationMiddleware):
    """Keep stream retry inside Deep Agents' side-effecting wrapper."""

    def __init__(self, retry_block: RetryBlock, model: Any, backend: Any) -> None:
        defaults = compute_summarization_defaults(model)
        super().__init__(
            model=model,
            backend=backend,
            trigger=defaults["trigger"],
            keep=defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=defaults["truncate_args_settings"],
        )
        self._retry_block = retry_block

    @property
    def name(self) -> str:
        """Replace the built-in slot in main and general-purpose agents."""

        return "SummarizationMiddleware"

    async def awrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        async def retrying_handler(inner_request: Any) -> Any:
            return await self._retry_block.run_streaming_model(
                lambda: handler(inner_request)
            )

        return await super().awrap_model_call(request, retrying_handler)

    async def _acreate_summary(self, messages_to_summarize: list[Any]) -> str:
        create_summary = super()._acreate_summary
        return await self._retry_block.run_streaming_model(
            lambda: create_summary(messages_to_summarize)
        )
