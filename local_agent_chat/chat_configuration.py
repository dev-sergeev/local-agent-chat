from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

from .agent_modes import AgentMode
from .chat_bindings import ChatBinding, ChatBindings

type HasUserRequest = Callable[[str], Awaitable[bool]]


class ChatConfigurations:
    """Coordinate a Chat's persisted Model Profile and Agent Mode."""

    def __init__(
        self,
        bindings: ChatBindings,
        ordered_profile_ids: Iterable[str],
        has_user_request: HasUserRequest,
    ) -> None:
        profiles = tuple(dict.fromkeys(ordered_profile_ids))
        if not profiles:
            raise ValueError("At least one Model Profile must be available")
        if any(not profile_id for profile_id in profiles):
            raise ValueError("Model Profile identifiers must not be empty")

        self._bindings = bindings
        self._ordered_profiles = profiles
        self._available_profiles = frozenset(profiles)
        self._has_user_request = has_user_request

    def _ensure_active(self, chat_id: str) -> None:
        if self._bindings.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")

    def current(self, chat_id: str) -> ChatBinding | None:
        self._ensure_active(chat_id)
        binding = self._bindings.get(chat_id)
        self._ensure_active(chat_id)
        return binding

    def _profile_for(self, chat_id: str, profile_hints: tuple[str | None, ...]) -> str:
        persisted = self.current(chat_id)
        if persisted is not None and persisted.profile_id in self._available_profiles:
            return persisted.profile_id
        return next(
            (
                profile_id
                for profile_id in profile_hints
                if profile_id in self._available_profiles
            ),
            self._ordered_profiles[0],
        )

    def open(self, chat_id: str, *profile_hints: str | None) -> ChatBinding:
        profile_id = self._profile_for(chat_id, profile_hints)
        return self._bindings.open(chat_id, profile_id)

    def select_mode(
        self,
        chat_id: str,
        requested: AgentMode,
        *profile_hints: str | None,
    ) -> ChatBinding:
        requested = AgentMode(requested)
        self.open(chat_id, *profile_hints)
        try:
            return self._bindings.select_mode(chat_id, requested)
        except ValueError:
            authoritative = self.current(chat_id)
            if (
                authoritative is not None
                and authoritative.mode_locked
                and authoritative.mode is not requested
            ):
                return authoritative
            raise

    def accept_message(self, chat_id: str, *profile_hints: str | None) -> ChatBinding:
        self.open(chat_id, *profile_hints)
        return self._bindings.lock(chat_id)

    async def recover(self, chat_id: str, *profile_hints: str | None) -> ChatBinding:
        binding = self.open(chat_id, *profile_hints)
        if binding.mode_locked:
            return binding
        if await self._has_user_request(chat_id):
            return self._bindings.lock(chat_id)
        return binding
