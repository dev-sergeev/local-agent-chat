"""Provider construction with bounded retries around a single inference."""

from __future__ import annotations

import asyncio
from asyncio import sleep
from contextlib import aclosing
from typing import Any

import httpx
from gigachat.exceptions import ResponseError
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI, StreamChunkTimeoutError
from openai import APIConnectionError, APIStatusError

from .retry_policy import retry_delay
from .settings import parse_model


class ModelStreamTimeout(TimeoutError):
    def __init__(self, chunks_received: int) -> None:
        self.chunks_received = chunks_received
        super().__init__("Timed out waiting for a model stream chunk.")


def _transient(error: Exception) -> bool:
    return isinstance(
        error, (httpx.TransportError, TimeoutError, APIConnectionError)
    ) or (
        isinstance(error, (ResponseError, APIStatusError))
        and (error.status_code in {408, 409, 429} or 500 <= error.status_code < 600)
    )


def _retry_delay(error: Exception, attempt: int) -> float:
    headers = None
    if isinstance(error, ResponseError):
        headers = error.headers
    elif isinstance(error, APIStatusError):
        headers = error.response.headers
    return retry_delay(attempt, headers)


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


class _AsyncInferenceRetries:
    """Own retries once, before output; SDK retries stay disabled."""

    inference_retries: int = 10
    inference_timeout: float = 60.0
    stream_chunk_timeout: float = 120.0

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if kwargs.get("stream", self.streaming):
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
        for attempt in range(self.inference_retries + 1):
            try:
                async with asyncio.timeout(self.inference_timeout):
                    return await super()._agenerate(
                        messages, stop, run_manager, **kwargs
                    )
            except Exception as error:
                if not _transient(error) or attempt >= self.inference_retries:
                    _raise_provider_error(error)
                await sleep(_retry_delay(error, attempt))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        chunks_received = 0
        for attempt in range(self.inference_retries + 1):
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
            except (ModelStreamTimeout, StreamChunkTimeoutError):
                raise
            except Exception as error:
                if (
                    chunks_received
                    or not _transient(error)
                    or attempt >= self.inference_retries
                ):
                    _raise_provider_error(error)
                await sleep(_retry_delay(error, attempt))


class OpenAIModel(_AsyncInferenceRetries, ChatOpenAI):
    """Use the same retry schedule for every OpenAI-compatible endpoint."""


class GigaChatModel(_AsyncInferenceRetries, GigaChat):
    """Use ready access tokens and normalize streaming usage from the SDK."""

    def _build_stream_chunk(self, chunk, first_chunk):
        # LangChain's aggregation requires integer cache usage, not SDK None.
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


def create_chat_model(model: str, **kwargs: Any):
    provider, model_id = parse_model(model)
    kwargs["inference_retries"] = kwargs.pop("max_retries", 10)
    kwargs["inference_timeout"] = kwargs.get("timeout", 60.0)
    kwargs["max_retries"] = 0
    if provider == "openai":
        return OpenAIModel(model=model_id, **kwargs)
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
