from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User

from sneakers_ml.bot.loggers import MultilineLogger

if TYPE_CHECKING:
    from sneakers_ml.bot.models import DBUser
    from sneakers_ml.bot.service.database import Repository


class UserAccessMiddleware(BaseMiddleware):
    logger = MultilineLogger().logger

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any | None:
        aiogram_user: User | None = data.get("event_from_user")
        if aiogram_user is None:
            return await handler(event, data)
        repository: Repository = data["repository"]
        try:
            user: DBUser | None = await repository.user.get(pk=aiogram_user.id)
            if user:
                data["user"] = user
        except Exception as e:
            self.logger.error("Error requesting repository: %v", e)

        return await handler(event, data)
