from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.utils.callback_answer import CallbackAnswerMiddleware

from sneakers_ml.bot.encoding import json
from sneakers_ml.bot.handlers import image, main
from sneakers_ml.bot.middlewares import (
    CommitMiddleware,
    DBSessionMiddleware,
    RetryRequestMiddleware,
    UserAccessMiddleware,
    UserAutoCreationMiddleware,
)
from sneakers_ml.bot.service.database import create_pool

if TYPE_CHECKING:
    from .settings import Settings


def _setup_outer_middlewares(dispatcher: Dispatcher, settings: Settings) -> None:
    pool = dispatcher["session_pool"] = create_pool(dsn=settings.build_postgres_dsn())

    dispatcher.update.outer_middleware(DBSessionMiddleware(session_pool=pool))
    dispatcher.update.outer_middleware(UserAccessMiddleware())


def _setup_inner_middlewares(dispatcher: Dispatcher) -> None:
    UserAutoCreationMiddleware().setup_inner(router=dispatcher)
    CommitMiddleware().setup_inner(router=dispatcher)
    dispatcher.callback_query.middleware(CallbackAnswerMiddleware())


def create_dispatcher(settings: Settings) -> Dispatcher:
    """
    :return: Configured ``Dispatcher`` with installed middlewares and included routers
    """
    dispatcher: Dispatcher = Dispatcher(
        name="main_dispatcher",
        settings=settings,
    )
    dispatcher.include_routers(main.router, image.router)
    if settings.enable_db_storage:
        _setup_outer_middlewares(dispatcher=dispatcher, settings=settings)
    _setup_inner_middlewares(dispatcher=dispatcher)
    return dispatcher


def create_bot(settings: Settings) -> Bot:
    """
    :return: Configured ``Bot`` with retry request middleware
    """
    session: AiohttpSession = AiohttpSession(json_loads=json.decode, json_dumps=json.encode)
    session.middleware(RetryRequestMiddleware())
    return Bot(
        token=settings.bot_token.get_secret_value(),
        parse_mode=ParseMode.HTML,
        session=session,
    )
