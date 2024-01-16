from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from sneakers_ml.bot.models import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T], ABC):
    _session: AsyncSession
    _entity: type[T]

    __slots__ = ("_session", "_entity")

    def __init__(self, session: AsyncSession, entity: type[T]) -> None:
        self._session = session
        self._entity = entity

    async def save(self, model: Base) -> None:
        self._session.add(model)
        await self._session.commit()

    async def get(self, pk: int) -> T | None:
        return await self._session.get(entity=self._entity, ident=pk)

    if TYPE_CHECKING:
        create: Callable[..., Awaitable[T]]

    else:

        @abstractmethod
        async def create(self, *args: Any, **kwargs: Any) -> T:
            pass
