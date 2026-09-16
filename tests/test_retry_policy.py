import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
from langchain_core.messages import HumanMessage
from openai import InternalServerError

from local_agent_chat import providers
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.retry_policy import retry_delay
from local_agent_chat.settings import LLMRetryConfig, ModelProfile


def test_default_backoff_reaches_five_minutes_on_tenth_retry():
    assert LLMRetryConfig().max_retries == 10
    assert [retry_delay(i) for i in range(10)] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        300,
    ]
    assert retry_delay(1000000) == 300


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Retry-After": "30"}, 30),
        ({"retry-after-ms": "2500"}, 2.5),
        ({"retry-after": "0"}, 1),
        ({"retry-after": "1000"}, 300),
        ({"retry-after": "-2"}, 1),
        ({"retry-after": "nan"}, 1),
        ({"retry-after": "inf"}, 1),
        ({"retry-after": "invalid"}, 1),
        ({"retry-after-ms": "invalid", "retry-after": "3"}, 3),
    ],
)
def test_server_delay_cannot_shorten_backoff_or_exceed_cap(headers, expected):
    assert retry_delay(0, headers) == expected
    assert retry_delay(5, {"retry-after": "1"}) == 32


def test_retry_after_http_date():
    date = datetime.now(timezone.utc) + timedelta(seconds=60)
    assert 58 <= retry_delay(0, {"Retry-After": format_datetime(date)}) <= 60


@pytest.mark.parametrize("provider", ["openai", "gigachat"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_retries_ten_times_on_real_http_boundary(
    monkeypatch, provider, streaming
):
    attempts, delays = [], []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(providers, "sleep", sleep)

    def respond(request):
        attempts.append(request)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    async with httpx.AsyncClient(
        base_url="https://model.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        profile = ModelProfile(
            "test",
            "Test",
            f"{provider}:test",
            "TEST_KEY",
            "test-key",
            "https://model.example/v1",
            streaming,
        )
        arguments = {"streaming": streaming}
        if provider == "openai":
            arguments["http_async_client"] = client
        model = RetryBlock(LLMRetryConfig()).create_model(profile, **arguments)
        if provider == "gigachat":
            model._client._aclient_instance = client
            assert model._client._settings.max_retries == 0
        else:
            assert model.root_async_client.max_retries == 0
        with pytest.raises(InternalServerError if provider == "openai" else ValueError):
            if streaming:
                async for _chunk in model.astream([HumanMessage(content="hello")]):
                    pytest.fail("An unavailable provider must not emit chunks")
            else:
                await model.ainvoke("hello")
    assert len(attempts) == 11
    assert delays == [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]


@pytest.mark.parametrize("provider", ["openai", "gigachat"])
async def test_cancelling_backoff_releases_model_without_another_attempt(
    monkeypatch, provider
):
    waiting = asyncio.Event()
    attempts = []

    async def sleep(delay):
        assert delay == 1
        waiting.set()
        await asyncio.Event().wait()

    def respond(request):
        attempts.append(request)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    monkeypatch.setattr(providers, "sleep", sleep)
    async with httpx.AsyncClient(
        base_url="https://model.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        block = RetryBlock(LLMRetryConfig())
        profile = ModelProfile(
            "test",
            "Test",
            f"{provider}:test",
            "KEY",
            "test-key",
            "https://model.example/v1",
        )
        model = block.create_model(
            profile, **({"http_async_client": client} if provider == "openai" else {})
        )
        if provider == "gigachat":
            model._client._aclient_instance = client
        task = asyncio.create_task(
            block.run_streaming_model(lambda: model.ainvoke("hello"))
        )
        await asyncio.wait_for(waiting.wait(), 2)
        assert block.busy
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not block.busy
    assert len(attempts) == 1
