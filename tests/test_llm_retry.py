import asyncio
import json
from dataclasses import FrozenInstanceError

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import StreamChunkTimeoutError
from openai import APIConnectionError, BadRequestError, InternalServerError

from local_agent_chat.agent_context import ContextSummary
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.settings import AgentConfig, LLMRetryConfig, ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(
        id="local",
        label="Local model",
        model="openai:test-model",
        api_key_env="TEST_API_KEY",
        api_key="secret",
        base_url="https://models.example/v1",
        streaming=True,
    )


def test_retry_config_is_immutable() -> None:
    config = LLMRetryConfig()

    with pytest.raises(FrozenInstanceError):
        config.max_retries = 4  # type: ignore[misc]


def test_create_model_applies_retry_policy_and_profile_credentials() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    expected_model = object()

    def model_factory(model: str, **kwargs: object) -> object:
        calls.append((model, kwargs))
        return expected_model

    block = RetryBlock(
        LLMRetryConfig(
            max_retries=5,
            request_timeout_seconds=45.0,
            stream_chunk_timeout_seconds=90.0,
            auxiliary_timeout_seconds=20.0,
        ),
        model_factory,
    )

    model = block.create_model(_profile(), max_tokens=32, reasoning_effort="none")

    assert model is expected_model
    assert calls == [
        (
            "openai:test-model",
            {
                "max_tokens": 32,
                "reasoning_effort": "none",
                "api_key": "secret",
                "base_url": "https://models.example/v1",
                "max_retries": 5,
                "timeout": 45.0,
                "stream_chunk_timeout": 90.0,
            },
        )
    ]


def test_create_model_omits_empty_optional_profile_credentials() -> None:
    captured: dict[str, object] = {}
    profile = ModelProfile(
        id="environment",
        label="Environment credentials",
        model="openai:test-model",
        api_key_env="TEST_API_KEY",
        api_key=None,
    )

    def model_factory(_model: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    RetryBlock(LLMRetryConfig(), model_factory).create_model(profile)

    assert "api_key" not in captured
    assert "base_url" not in captured


@pytest.mark.parametrize(
    "reserved_key",
    [
        "model",
        "api_key",
        "base_url",
        "max_retries",
        "timeout",
        "stream_chunk_timeout",
    ],
)
def test_create_model_rejects_reserved_overrides(reserved_key: str) -> None:
    calls = 0

    def model_factory(_model: str, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    block = RetryBlock(LLMRetryConfig(), model_factory)

    with pytest.raises(ValueError, match=reserved_key):
        block.create_model(_profile(), **{reserved_key: "override"})

    assert calls == 0


@pytest.mark.asyncio
async def test_run_auxiliary_returns_result_from_fresh_awaitable() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return "done"

    block = RetryBlock(
        LLMRetryConfig(auxiliary_timeout_seconds=1.0),
        lambda *_args, **_kwargs: object(),
    )

    assert await block.run_auxiliary(operation) == "done"
    assert calls == 1


@pytest.mark.asyncio
async def test_run_auxiliary_enforces_total_timeout_and_cancels_operation() -> None:
    cancelled = asyncio.Event()

    async def operation() -> None:
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    block = RetryBlock(
        LLMRetryConfig(auxiliary_timeout_seconds=0.01),
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(TimeoutError):
        await block.run_auxiliary(operation)

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_zero_chunk_model_retry_stops_at_the_configured_limit() -> None:
    calls = 0

    async def stalled_call() -> None:
        nonlocal calls
        calls += 1
        raise StreamChunkTimeoutError(0.01, chunks_received=0)

    block = RetryBlock(
        LLMRetryConfig(stream_retries=2),
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(StreamChunkTimeoutError):
        await block.run_streaming_model(stalled_call)

    assert calls == 3


@pytest.mark.asyncio
async def test_model_retry_never_restarts_after_a_provider_chunk() -> None:
    calls = 0

    async def partial_call() -> None:
        nonlocal calls
        calls += 1
        raise StreamChunkTimeoutError(0.01, chunks_received=1)

    block = RetryBlock(
        LLMRetryConfig(stream_retries=3),
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(StreamChunkTimeoutError):
        await block.run_streaming_model(partial_call)

    assert calls == 1


@pytest.mark.asyncio
async def test_model_retry_returns_the_recovered_result() -> None:
    calls = 0

    async def recovering_call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StreamChunkTimeoutError(0.01, chunks_received=0)
        return "recovered"

    block = RetryBlock(
        LLMRetryConfig(stream_retries=1),
        lambda *_args, **_kwargs: object(),
    )

    assert await block.run_streaming_model(recovering_call) == "recovered"
    assert calls == 2


def _completion_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "recovered"},
                    "finish_reason": "stop",
                }
            ],
        },
        request=request,
    )


def _real_model_block(*, max_retries: int) -> RetryBlock:
    return RetryBlock(
        LLMRetryConfig(
            max_retries=max_retries,
            request_timeout_seconds=1.0,
            stream_chunk_timeout_seconds=1.0,
            auxiliary_timeout_seconds=5.0,
        ),
    )


@pytest.mark.asyncio
async def test_openai_provider_retries_transient_http_failures() -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests < 3:
            return httpx.Response(
                503,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "temporarily unavailable"}},
                request=request,
            )
        return _completion_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = _real_model_block(max_retries=2).create_model(
            _profile(), http_async_client=client
        )
        response = await model.ainvoke("hello")

    assert response.text == "recovered"
    assert requests == 3


@pytest.mark.asyncio
async def test_openai_provider_does_not_retry_permanent_http_failures() -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            400,
            json={"error": {"message": "invalid request"}},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = _real_model_block(max_retries=3).create_model(
            _profile(), http_async_client=client
        )
        with pytest.raises(BadRequestError):
            await model.ainvoke("hello")

    assert requests == 1


@pytest.mark.asyncio
async def test_zero_max_retries_disables_nested_summarization_retry() -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            503,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "temporarily unavailable"}},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        block = _real_model_block(max_retries=0)
        model = block.create_model(_profile(), http_async_client=client)
        middleware = ContextSummary(model, AgentConfig(), block)
        with pytest.raises(InternalServerError):
            await middleware._acreate_summary([HumanMessage(content="hello")])

    assert requests == 1


class _BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "partial"},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        raise httpx.ReadError("stream disconnected")


@pytest.mark.asyncio
async def test_openai_provider_does_not_restart_a_started_stream() -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_BrokenStream(),
            request=request,
        )

    received: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = _real_model_block(max_retries=3).create_model(
            _profile(), http_async_client=client
        )
        # Newer SDKs wrap transport failures in APIConnectionError; older
        # versions propagate ReadError directly. Both must preserve no replay.
        with pytest.raises((httpx.ReadError, APIConnectionError)) as failure:
            async for chunk in model.astream("hello"):
                received.append(str(chunk.content))

    cause = failure.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, httpx.ReadError)
    assert str(cause) == "stream disconnected"
    assert received == ["partial"]
    assert requests == 1
