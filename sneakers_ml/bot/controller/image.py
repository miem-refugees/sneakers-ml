import aiohttp
from aiohttp import ClientResponseError, FormData
from loguru import logger
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes


class ImageController:
    def __init__(self, api_url: str, classify_brand_func=None) -> None:
        self.api_url = api_url
        self.classify_brand_url = f"{api_url}/classify-brand/upload/"
        self.classify_brand = classify_brand_func if classify_brand_func is not None else self.classify_brand

    async def classify_brand(self, image: bytearray, username: str, filename: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                data = FormData()
                data.add_field("image", image, filename=filename)
                data.add_field("username", username)
                resp = await session.post(self.classify_brand_url, data=data)
                content = await resp.json()
                resp.raise_for_status()
                predictions = content
                return "\n".join([f"*{model}*: *{predictions[model]}*" for model in predictions])
        except ClientResponseError as e:
            logger.error("Got response error from classify-brand: %s\n%s", e, content)
        except Exception as e:
            logger.error("Got exception while trying to classify brand: %s", e)

        return "Oh no! Could not predict brand, sorry =("

    async def image_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        file_info = await context.bot.get_file(update.message.photo[-1].file_id)
        image: bytearray = await file_info.download_as_bytearray()
        await update.message.reply_text("Predicting brand for your photo...")

        classification_result = await self.classify_brand(image, update.message.from_user.username, file_info.file_id)

        await update.message.reply_text(classification_result, parse_mode=ParseMode.MARKDOWN_V2)
