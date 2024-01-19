from typing import BinaryIO

import aiohttp
from aiogram import Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from PIL import Image

from sneakers_ml.app.models.brand_classify import BrandClassifyRequest
from sneakers_ml.bot.filters.image import ImageFilter

dispatcher = Dispatcher()


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
    image = Image.open(file)
    await message.answer("Predicting brand for your photo...")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.post(
            url="http://localhost:8000/classify-brand/",  # TODO: change to prod url
            json=BrandClassifyRequest(
                image=image.tobytes().decode("utf-8"),
                name=str(message.message_id),
                from_username=message.from_user.username,
            ).model_dump(),
        ) as response:
            predictions = await response.json()
            await message.answer("\n".join([f"{hbold(model)}: {hbold(predictions[model])}" for model in predictions]))
