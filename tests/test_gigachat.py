import asyncio
import json

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from local_agent_chat.agent_context import ContextSummary, ModelGuardrails
from local_agent_chat.auxiliary_labels import AuxiliaryLabels
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.providers import (
    GigaChatModel,
    ModelStreamTimeout,
    create_chat_model,
)
from local_agent_chat.settings import AgentConfig, LLMRetryConfig, ModelProfile


def profile(streaming=True):
    return ModelProfile(
        "giga",
        "GigaChat",
        "gigachat:GigaChat-2",
        "GIGACHAT_ACCESS_TOKEN",
        "test-access-token",
        "https://giga.example/v1",
        streaming,
    )


def completion(content="answer", function_call=None):
    message = {"role": "assistant", "content": content}
    if function_call:
        message["function_call"] = function_call
    return {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "function_call" if function_call else "stop",
            }
        ],
        "created": 1,
        "model": "GigaChat-2",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def chunk(content="part"):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
                "created": 1,
                "model": "GigaChat-2",
                "object": "chat.completion.chunk",
            }
        )
        + "\n\n"
    ).encode()


def block_for(client, **limits):
    def factory(model, **kwargs):
        instance = create_chat_model(model, **kwargs)
        instance._client._aclient_instance = client
        return instance

    return RetryBlock(LLMRetryConfig(**limits), factory)


async def test_native_gigachat_uses_token_url_and_model(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "unrelated-oauth-key")
    monkeypatch.setenv("GIGACHAT_PASSWORD", "unrelated-password")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(
        base_url="https://giga.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        model = block_for(client).create_model(profile(), max_tokens=20)
        assert isinstance(model, GigaChatModel)
        assert model.base_url == "https://giga.example/v1"
        assert model._client._settings.max_retries == 0
        assert not model._client._use_auth
        result = await model.ainvoke("hello")
    assert result.content == "answer"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://giga.example/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-access-token"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "GigaChat-2"
    assert payload["max_tokens"] == 20
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize("streaming", [False, True])
async def test_gigachat_retries_transient_requests_only(streaming):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(
                503, headers={"retry-after": "0"}, json={"message": "try again"}
            )
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=chunk() + b"data: [DONE]\n\n",
            )
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(
        base_url="https://giga.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        model = block_for(client, max_retries=2).create_model(
            profile(), streaming=streaming
        )
        if streaming:
            assert (
                "".join([part.content async for part in model.astream("hello")])
                == "part"
            )
        else:
            assert (await model.ainvoke("hello")).content == "answer"
    assert calls == 3


@pytest.mark.parametrize("code", [400, 401, 403, 503])
async def test_gigachat_errors_are_bounded_and_do_not_expose_response_secrets(code):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            code,
            json={"message": "test-access-token"},
            headers={"x-secret": "test-access-token"},
        )

    async with httpx.AsyncClient(
        base_url="https://giga.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        model = block_for(client, max_retries=0).create_model(profile())
        with pytest.raises(ValueError) as error:
            await model.ainvoke("hello")
    assert calls == 1
    assert "test-access-token" not in str(error.value)
    if code in {401, 403}:
        assert "restart localchat" in str(error.value)


class StallingStream(httpx.AsyncByteStream):
    def __init__(self, emit_first=False, disconnect=False):
        self.emit_first = emit_first
        self.disconnect = disconnect
        self.closed = False

    async def __aiter__(self):
        if self.emit_first:
            yield chunk()
        if self.disconnect:
            raise httpx.ReadError("disconnected")
        await asyncio.sleep(60)

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(
    "emit_first,disconnect", [(False, False), (True, False), (True, True)]
)
async def test_gigachat_stream_timeout_cleanup_and_no_replay(emit_first, disconnect):
    streams = []

    def respond(request):
        stream = StallingStream(emit_first, disconnect)
        streams.append(stream)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )

    async with httpx.AsyncClient(
        base_url="https://giga.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        block = block_for(
            client, max_retries=3, stream_retries=1, stream_chunk_timeout_seconds=0.01
        )
        model = block.create_model(profile(), streaming=True)
        received = []

        async def infer():
            async for part in model.astream("hello"):
                received.append(part.content)

        with pytest.raises(httpx.ReadError if disconnect else ModelStreamTimeout):
            await block.run_streaming_model(infer)
    assert len(streams) == (1 if emit_first else 2)
    assert received == (["part"] if emit_first else [])
    assert all(stream.closed for stream in streams)


async def test_gigachat_request_timeout_has_finite_attempts():
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(60)

    async with httpx.AsyncClient(
        base_url="https://giga.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        model = block_for(
            client, max_retries=0, request_timeout_seconds=0.01
        ).create_model(profile(False))
        with pytest.raises(TimeoutError):
            await model.ainvoke("hello")
    assert calls == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_gigachat_agent_tools_summary_and_titles(tmp_path, streaming):
    payloads = []
    failed_after_tool = False
    tool_calls = 0

    def answer(payload, body):
        if not payload.get("stream"):
            return httpx.Response(200, json=body)
        for choice in body["choices"]:
            choice["delta"] = choice.pop("message")
        body["object"] = "chat.completion.chunk"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("data: " + json.dumps(body) + "\n\ndata: [DONE]\n\n").encode(),
        )

    @tool
    def read_note() -> str:
        """Read the uploaded note."""
        nonlocal tool_calls
        tool_calls += 1
        return "PIN-ALPHA"

    def respond(request):
        nonlocal failed_after_tool
        payload = json.loads(request.content)
        payloads.append(payload)
        assert "reasoning_effort" not in payload
        messages = payload["messages"]
        if payload["max_tokens"] == 32:
            return httpx.Response(200, json=completion("Чтение загруженной заметки"))
        if payload["max_tokens"] == 1000:
            return httpx.Response(
                200, json=completion("The project code is PIN-ALPHA.")
            )
        if messages[-1]["role"] == "function":
            if not failed_after_tool:
                failed_after_tool = True
                return httpx.Response(503, headers={"retry-after": "0"})
            return answer(payload, completion("PIN-ALPHA"))
        assert payload["functions"][0]["name"] == "read_note"
        return answer(payload, completion("", {"name": "read_note", "arguments": {}}))

    async with httpx.AsyncClient(
        base_url="https://giga.example/v1/", transport=httpx.MockTransport(respond)
    ) as client:
        block = block_for(client, max_retries=1)
        model = block.create_model(
            profile(streaming), max_tokens=2000, disable_streaming=not streaming
        )
        graph = create_agent(
            model, tools=[read_note], middleware=[ModelGuardrails(AgentConfig(), block)]
        )
        result = await graph.ainvoke({"messages": [("user", "Read my note")]})
        assert result["messages"][-1].content == "PIN-ALPHA"
        assert tool_calls == 1
        summary = ContextSummary(
            block.create_model(profile(False), max_tokens=1000, disable_streaming=True),
            AgentConfig(),
            block,
        )
        assert "PIN-ALPHA" in await summary._acreate_summary(
            [HumanMessage(content="PIN-ALPHA")]
        )
        bindings = ChatBindings(tmp_path / "bindings.sqlite3", ["giga"])
        bindings.open("chat")
        labels = AuxiliaryLabels([profile()], bindings, block)
        assert (
            await labels.describe_chat("chat", "Read my note")
            == "Чтение загруженной заметки"
        )
    assert len(payloads) == 5


def test_gigachat_rejects_unsupported_openai_options():
    with pytest.raises(ValueError, match="extra_body"):
        RetryBlock(LLMRetryConfig()).create_model(
            profile(), extra_body={"reasoning": {}}
        )
