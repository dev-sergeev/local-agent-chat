import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from local_agent_chat.auxiliary_labels import AuxiliaryLabels
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.settings import LLMRetryConfig, ModelProfile


def profile(profile_id: str = "local") -> ModelProfile:
    return ModelProfile(
        profile_id, profile_id.title(), "openai:test", "TEST_KEY", "key"
    )


def labels_with(
    tmp_path: Path,
    responses: list[str | BaseException],
) -> tuple[AuxiliaryLabels, list[tuple[str, dict]], list[list]]:
    calls: list[list] = []
    init_calls: list[tuple[str, dict]] = []

    class FakeModel:
        async def ainvoke(self, messages):
            calls.append(messages)
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return AIMessage(content=response)

    def factory(model: str, **kwargs):
        init_calls.append((model, kwargs))
        return FakeModel()

    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    bindings.open("chat-1", "local")
    labels = AuxiliaryLabels(
        (profile(),),
        bindings,
        RetryBlock(LLMRetryConfig(), factory),
    )
    return labels, init_calls, calls


@pytest.mark.asyncio
async def test_chat_label_uses_profile_retry_policy_and_cached_model(
    tmp_path: Path,
) -> None:
    labels, init_calls, calls = labels_with(
        tmp_path,
        [
            "Аудит проекта перед публикацией",
            "Разбор ошибки загрузки файла",
        ],
    )

    chat = await labels.describe_chat("chat-1", "Проведи полный аудит проекта")
    second_chat = await labels.describe_chat("chat-1", "Исправь ошибку загрузки файла")

    assert chat == "Аудит проекта перед публикацией"
    assert second_chat == "Разбор ошибки загрузки файла"
    assert init_calls == [
        (
            "openai:test",
            {
                "api_key": "key",
                "max_tokens": 32,
                "max_retries": 3,
                "reasoning_effort": "none",
                "stream_chunk_timeout": 120.0,
                "timeout": 60.0,
            },
        )
    ]
    assert "<user-request>" in calls[0][1][1]
    assert "Проведи полный аудит проекта" in calls[0][1][1]
    assert "Исправь ошибку загрузки файла" in calls[1][1][1]


@pytest.mark.asyncio
async def test_chat_label_provider_failure_is_cosmetic(tmp_path: Path) -> None:
    labels, _init_calls, _calls = labels_with(
        tmp_path, [RuntimeError("provider unavailable")]
    )

    assert await labels.describe_chat("chat-1", "Сделай что-нибудь") is None


@pytest.mark.asyncio
async def test_chat_label_invalid_output_and_factory_failure_are_cosmetic(
    tmp_path: Path,
) -> None:
    labels, _init_calls, _calls = labels_with(tmp_path, ["Короткий заголовок"])
    assert await labels.describe_chat("chat-1", "Сделай что-нибудь") is None

    bindings = ChatBindings(tmp_path / "broken.sqlite3", ("local",))
    bindings.open("chat-1", "local")

    def fail_factory(*_args, **_kwargs):
        raise RuntimeError("cannot construct model")

    broken = AuxiliaryLabels(
        (profile(),),
        bindings,
        RetryBlock(LLMRetryConfig(), fail_factory),
    )
    assert await broken.describe_chat("chat-1", "Сделай что-нибудь") is None


@pytest.mark.asyncio
async def test_missing_or_stale_binding_does_not_call_provider(tmp_path: Path) -> None:
    factory_calls = 0

    def factory(*_args, **_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("provider must not be called")

    database = tmp_path / "bindings.sqlite3"
    bindings = ChatBindings(database, ("removed",))
    bindings.open("stale-chat", "removed")
    labels = AuxiliaryLabels(
        (profile("current"),),
        bindings,
        RetryBlock(LLMRetryConfig(), factory),
    )

    assert await labels.describe_chat("missing-chat", "request") is None
    assert await labels.describe_chat("stale-chat", "request") is None
    assert factory_calls == 0


@pytest.mark.asyncio
async def test_cancellation_is_not_hidden_as_a_cosmetic_failure(tmp_path: Path) -> None:
    labels, _init_calls, _calls = labels_with(tmp_path, [asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await labels.describe_chat("chat-1", "request")
