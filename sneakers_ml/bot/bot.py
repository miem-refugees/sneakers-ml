import asyncio
import logging
import sys
from os import getenv
from typing import BinaryIO

from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from PIL import Image

from sneakers_ml.bot.brand_predictor import BrandPredictor
from sneakers_ml.bot.image import ImageFilter

dp = Dispatcher()


class TgBot:
    def __init__(self, *routers: Router):
        token = getenv("BOT_TOKEN")
        if token is None:
            raise RuntimeError("No tg token by env: BOT_TOKEN")
        self.telegram = Bot(token, parse_mode=ParseMode.HTML)
        self.brand_predictor = BrandPredictor()

    @dp.message(CommandStart())
    async def command_start_handler(message: Message):
        await message.answer(
            f"Hello, {hbold(message.from_user.full_name)}!\n"
            f"Just send me a photo of you sneaker and I'll try to predict it's brand"
        )

    @dp.message(Command("/help"))
    async def command_help_handler(message: Message):
        await message.answer("Send me any photo of sneaker and I'll try to predict it's brand =)")

    @dp.message(ImageFilter())
    async def image_handler(self, message: types.Message, photo_file_id: str):
        await message.answer("Downloading your photo...")
        file: BinaryIO = await self.telegram.download(photo_file_id)
        file.seek(0)
        image = Image.open(file)
        await message.answer("Predicting brand for your photo...")
        predicted = self.brand_predictor.predict(image)
        await message.answer("\n".join([f"{hbold(model)}: {hbold(predicted[model])}" for model in predicted]))


async def main():
    bot = TgBot(dp)
    await dp.start_polling(bot.telegram)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
