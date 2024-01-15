from typing import Any, BinaryIO, Final

from aiogram import Router
from aiogram.methods import TelegramMethod
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from PIL import Image

from sneakers_ml.bot.filters import ImageFilter
from sneakers_ml.bot.ml import BrandPredictor

router: Final[Router] = Router(name=__name__)
brand_predictor = BrandPredictor()


@router.message(ImageFilter())
async def on_image(message: Message, image_file_id: str) -> TelegramMethod[Any]:
    file: BinaryIO = await message.bot.download(image_file_id)
    file.seek(0)
    image = Image.open(file)
    await message.answer("Predicting brand for your photo...")
    predicted = brand_predictor.predict(image)
    await message.answer("\n".join([f"{hbold(model)}: {hbold(predicted[model])}" for model in predicted]))
