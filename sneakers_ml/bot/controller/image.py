from aiohttp import ClientResponseError, ClientSession, FormData
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from sneakers_ml.bot.utils.escape import escape
from sneakers_ml.bot.utils.timer import timed


class ImageController:
    def __init__(self, api_url: str, logger, classify_brand_func=None) -> None:
        self.api_url = api_url
        self._logger = logger
        self.classify_brand_url = f"{api_url}/classify-brand/upload/"

        # This is used for testing purposes
        if classify_brand_func is not None:
            self.classify_brand = classify_brand_func

    @timed
    async def classify_brand(self, image: bytearray, username: str, filename: str, logger) -> str:
        try:
            async with ClientSession() as session:
                data = FormData()
                data.add_field("image", image, filename=filename)
                data.add_field("username", username)
                resp = await session.post(self.classify_brand_url, data=data)
                content = await resp.json()
                resp.raise_for_status()
                predictions = content
                logger.debug("Got predictions from server: {}", predictions)
                joined = "\n".join(
                    [f"*{escape(model)}*: _{escape(''.join(predictions[model]))}_" for model in predictions]
                )
                return joined
        except ClientResponseError as e:
            logger.error("Got response error from classify-brand: {}\n{}", e, content)
        except Exception as e:
            logger.error("Got exception while trying to classify brand: {}", e)

        return escape("Oh no! Could not predict brand, sorry =(")

    async def image_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger = self._logger.bind(req_id=update.update_id)
        file_info = await context.bot.get_file(update.message.photo[-1].file_id)
        logger.debug("Got file info: {}", file_info.to_json())
        image: bytearray = await file_info.download_as_bytearray()
        logger.debug("Downloaded image with size: {} bytes", len(image))
        await update.message.reply_text("Predicting brand for your photo...")

        classification_result = await self.classify_brand(
            image, update.message.from_user.username, file_info.file_id, logger=logger
        )

        await update.message.reply_text(classification_result, parse_mode=ParseMode.MARKDOWN_V2)
