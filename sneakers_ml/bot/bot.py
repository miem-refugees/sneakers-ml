import asyncio
import logging
import sys
from os import getenv
from typing import BinaryIO

from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from PIL import Image

from sneakers_ml.bot.brand_predictor import BrandPredictor
from sneakers_ml.bot.image import ImageFilter

TOKEN = getenv("BOT_TOKEN")

dp = Dispatcher()
image_router = Router()
dp.include_router(image_router)
bot = Bot(TOKEN, parse_mode=ParseMode.HTML)

brand_predictor = BrandPredictor()


@dp.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer(
        f"Hello, {hbold(message.from_user.full_name)}!\n"
        f"Just send me a photo of you sneaker and I'll try to predict it's brand"
    )


@image_router.message(ImageFilter())
async def image_handler(message: types.Message, photo_file_id: str):
    await message.answer("Downloading your photo...")
    file: BinaryIO = await bot.download(photo_file_id)
    file.seek(0)
    image = Image.open(file)
    await message.answer("Predicting brand for your photo...")
    predicted = brand_predictor.predict(image)
    await message.answer("\n".join([f"{hbold(model)}: {hbold(predicted[model])}" for model in predicted]))


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
