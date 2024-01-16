from typing import Any, Final

from aiogram import F, Router
from aiogram.filters import ExceptionTypeFilter
from aiogram.methods import TelegramMethod
from aiogram.types import ErrorEvent

router: Final[Router] = Router(name=__name__)


@router.error(ExceptionTypeFilter(Exception), F.update.message)
async def handle_some_error(error: ErrorEvent) -> TelegramMethod[Any]:
    return error.update.message.answer(text="Упс, произошла какая-то ошибка! Извините =(")
