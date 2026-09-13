from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps


class _PendingWrites:
    def __init__(self) -> None:
        self.owner = asyncio.current_task()
        self.tasks: set[asyncio.Task] = set()

    async def drain(self) -> None:
        # Message/Step.send() schedule persistence with create_task. Let those
        # coroutines enter the data layer, then wait for every scheduled write.
        await asyncio.sleep(0)
        results = []
        while self.tasks:
            pending, self.tasks = self.tasks, set()
            results.extend(await asyncio.gather(*pending, return_exceptions=True))
        for result in results:
            if isinstance(result, BaseException):
                raise result


_pending_writes: ContextVar[_PendingWrites | None] = ContextVar(
    "chainlit_pending_writes", default=None
)


def track_step_write(method):
    @wraps(method)
    async def tracked(*args, **kwargs):
        scope = _pending_writes.get()
        task = asyncio.current_task()
        if scope is not None and task is not None and task is not scope.owner:
            scope.tasks.add(task)
        return await method(*args, **kwargs)

    return tracked


@asynccontextmanager
async def step_writes() -> AsyncIterator[None]:
    """Finish Chainlit's background writes before deciding a Turn transaction."""
    scope = _PendingWrites()
    token = _pending_writes.set(scope)
    try:
        try:
            yield
        except BaseException:
            # Drain before rollback so a late UI write cannot resurrect a
            # discarded continuation. Preserve the original Turn failure.
            await asyncio.shield(asyncio.gather(scope.drain(), return_exceptions=True))
            raise
        else:
            await scope.drain()
    finally:
        _pending_writes.reset(token)
