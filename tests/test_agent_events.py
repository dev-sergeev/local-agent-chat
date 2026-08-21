from local_agent_chat.agent_events import public_text, safe_text


def test_safe_text_redacts_structured_and_embedded_credentials(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret-value")

    rendered = safe_text(
        {
            "api_key": "structured-secret",
            "command": (
                "OPENAI_API_KEY=inline-secret-value "
                "curl -H 'Authorization: Bearer bearer-secret-value' "
                "https://example.test/environment-secret-value"
            ),
        }
    )

    assert "structured-secret" not in rendered
    assert "inline-secret-value" not in rendered
    assert "bearer-secret-value" not in rendered
    assert "environment-secret-value" not in rendered
    assert rendered.count("<redacted>") >= 3


def test_safe_text_bounds_large_tool_output() -> None:
    rendered = safe_text("a" * 7000, max_chars=120)

    assert "пропущено" in rendered
    assert len(rendered) < 220
    assert rendered.startswith("a" * 40)
    assert rendered.endswith("a" * 30)


def test_public_text_ignores_reasoning_blocks() -> None:
    content = [
        {"type": "reasoning", "text": "private reasoning"},
        {"type": "text", "text": "public "},
        {"type": "output_text", "text": {"value": "answer"}},
    ]

    assert public_text(content) == "public answer"
