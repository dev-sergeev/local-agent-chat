from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .agent_events import message_text, safe_text
from .chat_bindings import ChatBindings
from .chat_titles import normalize_chat_title
from .llm_retry import RetryBlock
from .prompts import (
    CHAT_TITLE_SYSTEM_PROMPT,
    TOOL_TITLE_RETRY_PROMPT,
    TOOL_TITLE_SYSTEM_PROMPT,
)
from .settings import ModelProfile
from .tool_titles import normalize_tool_title


class AuxiliaryLabels:
    """Generate cosmetic Chat and tool labels without affecting a Turn."""

    def __init__(
        self,
        models: Iterable[ModelProfile],
        bindings: ChatBindings,
        retry: RetryBlock,
    ) -> None:
        self._models = {model.id: model for model in models}
        self._bindings = bindings
        self._retry = retry
        self._label_models: dict[str, Any] = {}

    def _model(self, profile_id: str) -> Any:
        model = self._label_models.get(profile_id)
        if model is None:
            model = self._retry.create_model(
                self._models[profile_id],
                max_tokens=32,
                reasoning_effort="none",
            )
            self._label_models[profile_id] = model
        return model

    def _profile_id(self, chat_id: str) -> str | None:
        binding = self._bindings.get(chat_id)
        if binding is None or binding.profile_id not in self._models:
            return None
        return binding.profile_id

    async def describe_tool(
        self, chat_id: str, name: str, input_text: str
    ) -> str | None:
        """Return a compact tool label, or ``None`` for the neutral UI label."""

        profile_id = self._profile_id(chat_id)
        if profile_id is None:
            return None
        payload = safe_text(input_text, max_chars=2000)
        messages = [
            ("system", TOOL_TITLE_SYSTEM_PROMPT),
            (
                "user",
                f"Инструмент: {name}\n<tool-input>\n{payload}\n</tool-input>",
            ),
        ]
        try:
            model = self._model(profile_id)
            response = await self._retry.run_auxiliary(lambda: model.ainvoke(messages))
            rejected = message_text(response)
            title = normalize_tool_title(rejected)
            if title is not None:
                return title
            response = await self._retry.run_auxiliary(
                lambda: model.ainvoke(
                    [
                        *messages,
                        ("assistant", safe_text(rejected, max_chars=500)),
                        ("user", TOOL_TITLE_RETRY_PROMPT),
                    ]
                )
            )
        except Exception:  # noqa: BLE001 - labels are cosmetic
            return None
        return normalize_tool_title(message_text(response))

    async def describe_chat(self, chat_id: str, request_text: str) -> str | None:
        """Return a semantic Chat title, or ``None`` for the persisted fallback."""

        profile_id = self._profile_id(chat_id)
        if profile_id is None:
            return None
        payload = safe_text(request_text, max_chars=4000)
        try:
            model = self._model(profile_id)
            response = await self._retry.run_auxiliary(
                lambda: model.ainvoke(
                    [
                        ("system", CHAT_TITLE_SYSTEM_PROMPT),
                        ("user", f"<user-request>\n{payload}\n</user-request>"),
                    ]
                )
            )
        except Exception:  # noqa: BLE001 - labels are cosmetic
            return None
        return normalize_chat_title(message_text(response))
