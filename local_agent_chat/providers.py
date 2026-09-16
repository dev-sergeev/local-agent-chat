"""Provider construction and GigaChat's bounded asynchronous inference boundary."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from gigachat.exceptions import ResponseError
from langchain import chat_models
from langchain_gigachat import GigaChat

from .settings import parse_model


class ModelStreamTimeout(TimeoutError):
    def __init__(self, chunks_received: int) -> None:
        self.chunks_received = chunks_received
        super().__init__("Timed out waiting for a model stream chunk.")


def _transient(error: Exception) -> bool:
    return isinstance(error, (httpx.TransportError, TimeoutError)) or (
        isinstance(error, ResponseError)
        and error.status_code in {408, 429, 500, 502, 503, 504}
    )


def _retry_delay(error: Exception, attempt: int) -> float:
    if isinstance(error, ResponseError) and error.headers:
        value = error.headers.get("retry-after")
        if value:
            try:
                return max(0.0, min(60.0, float(value)))
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    return max(
                        0.0,
                        min(60.0, (date - datetime.now(timezone.utc)).total_seconds()),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
    return min(60.0, 0.5 * 2**attempt)


def _raise_provider_error(error: Exception) -> None:
    # SDK ResponseError includes response bodies and headers. Never expose those
    # in the UI: an endpoint may echo the Authorization header in an error.
    if isinstance(error, ResponseError):
        if error.status_code in {401, 403}:
            raise ValueError(
                "GigaChat authorization failed. Update the access token in the "
                "profile's .env variable and restart localchat."
            ) from None
        raise ValueError(
            f"GigaChat API request failed (HTTP {error.status_code})."
        ) from None
    raise error


class GigaChatModel(GigaChat):
    """Keep SDK retries off: its stream retry can replay an emitted chunk.

    The application only uses async inference. HTTP retries live here, before
    the first chunk; RetryBlock separately handles zero-chunk stalls. Neither
    boundary can replay a graph or a completed tool call.
    """

    stream_chunk_timeout: float = 120.0

    def _build_stream_chunk(self, chunk, first_chunk):
        # The SDK emits None for absent cache usage, but LangChain's chunk
        # aggregation requires an integer (the wrapper only handles missing keys).
        if usage := chunk.get("usage"):
            chunk = {
                **chunk,
                "usage": {
                    **usage,
                    "precached_prompt_tokens": usage.get("precached_prompt_tokens")
                    or 0,
                },
            }
        return super()._build_stream_chunk(chunk, first_chunk)

    def _get_client_init_kwargs(self) -> dict[str, Any]:
        return {
            **super()._get_client_init_kwargs(),
            "max_retries": 0,
            # Do not pick up unrelated OAuth/password credentials from the host.
            "credentials": "",
            "user": "",
            "password": "",
        }

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if kwargs.get("stream", self.streaming):
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
        for attempt in range((self.max_retries or 0) + 1):
            try:
                async with asyncio.timeout(self.timeout):
                    return await super()._agenerate(
                        messages, stop, run_manager, **kwargs
                    )
            except Exception as error:
                if not _transient(error) or attempt >= (self.max_retries or 0):
                    _raise_provider_error(error)
                await asyncio.sleep(_retry_delay(error, attempt))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        chunks_received = 0
        for attempt in range((self.max_retries or 0) + 1):
            try:
                async with aclosing(
                    super()._astream(messages, stop, run_manager, **kwargs)
                ) as stream:
                    while True:
                        try:
                            async with asyncio.timeout(self.stream_chunk_timeout):
                                chunk = await anext(stream)
                        except StopAsyncIteration:
                            return
                        except TimeoutError as error:
                            raise ModelStreamTimeout(chunks_received) from error
                        chunks_received += 1
                        yield chunk
            except ModelStreamTimeout:
                raise
            except Exception as error:
                if (
                    chunks_received
                    or not _transient(error)
                    or attempt >= (self.max_retries or 0)
                ):
                    _raise_provider_error(error)
                await asyncio.sleep(_retry_delay(error, attempt))


def create_chat_model(model: str, **kwargs: Any):
    provider, model_id = parse_model(model)
    if provider == "openai":
        return chat_models.init_chat_model(model, **kwargs)
    token = kwargs.pop("api_key", None)
    if not token:
        raise ValueError(
            "GigaChat requires an access token in the profile's api_key_env."
        )
    unknown = kwargs.keys() - GigaChatModel.model_fields.keys()
    if unknown:
        raise ValueError(
            f"Unsupported GigaChat model options: {', '.join(sorted(unknown))}"
        )
    # GigaChat's pre-init validator marks every field as explicitly set. Recent
    # LangChain therefore treats its default streaming=False as a hard opt-out.
    kwargs.setdefault("streaming", not kwargs.get("disable_streaming", True))
    return GigaChatModel(model=model_id, access_token=token, **kwargs)
