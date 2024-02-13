import logging
from typing import BinaryIO

import aiohttp
from aiogram import Dispatcher
from aiogram.filters import CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from aiohttp.client_exceptions import ClientResponseError
from aiohttp.formdata import FormData

from sneakers_ml.bot.filters.image import ImageFilter

dispatcher = Dispatcher(storage=MemoryStorage())
logger = logging.getLogger(__name__)
CLASSIFY_URL = "http://localhost:8000/classify-brand/upload/"


@dispatcher.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(
        f"Hello, {hbold(message.from_user.full_name)}!\n"
        + "Just send me a photo of you sneaker and I'll try to predict it's brand"
    )


@dispatcher.message(ImageFilter())
async def on_image(message: Message, image_file_id: str):
    file: BinaryIO = await message.bot.download(image_file_id)
    file.seek(0)
    await message.answer("Predicting brand for your photo...")
    predictions = None

    try:
        async with aiohttp.ClientSession() as session:
            data = FormData()
            data.add_field("image", file, filename="report.xls")
            data.add_field("username", message.from_user.username)
            resp = await session.post(CLASSIFY_URL, data=data)
            content = await resp.json()
            resp.raise_for_status()
            predictions = content
    except ClientResponseError as e:
        logger.error("Got response error from classify-brand: %s\n%s", e, content)
        await message.answer("Oh no! Could not predict brand, sorry =(")

    if predictions is not None:
        await message.answer("\n".join([f"{hbold(model)}: {hbold(predictions[model])}" for model in predictions]))
    else:
        await message.answer("Oh no! Could not predict brand, sorry =(")
