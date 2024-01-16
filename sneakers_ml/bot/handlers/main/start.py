from typing import Any, Final

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.methods import TelegramMethod
from aiogram.types import Message
from aiogram.utils.markdown import hbold

router: Final[Router] = Router(name=__name__)


@router.message(CommandStart())
async def start_command(message: Message) -> TelegramMethod[Any]:
    return message.answer(
        text=f"Hello, {hbold(message.from_user.full_name)}!\n"
        "Just send me a photo of you sneaker and I'll try to predict it's brand"
    )
