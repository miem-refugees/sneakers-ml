import logging

import aiohttp
from aiohttp.client_exceptions import ClientResponseError
from aiohttp.formdata import FormData
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from sneakers_ml.bot.config import Config

logger = logging.getLogger()
CLASSIFY_URL = "http://localhost:8000/classify-brand/upload/"


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Hello, {update.message.from_user.full_name}!\n"
        + "Just send me a photo of you sneaker and I'll try to predict it's brand"
    )


async def image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_info = await context.bot.get_file(update.message.photo[-1].file_id)
    image: bytearray = await file_info.download_as_bytearray()
    await update.message.reply_text("Predicting brand for your photo...")
    predictions = None

    try:
        async with aiohttp.ClientSession() as session:
            data = FormData()
            data.add_field("image", image, filename=file_info.file_id)
            data.add_field("username", update.message.from_user.username)
            resp = await session.post(CLASSIFY_URL, data=data)
            content = await resp.json()
            resp.raise_for_status()
            predictions = content
    except ClientResponseError as e:
        logger.error("Got response error from classify-brand: %s\n%s", e, content)
        await update.message.reply_text("Oh no! Could not predict brand, sorry =(")

    if predictions is not None:
        await update.message.reply_text(
            "\n".join([f"*{model}*: *{predictions[model]}*" for model in predictions]), parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await update.message.reply_text("Oh no! Could not predict brand, sorry =(")


def main():
    config = Config()

    application = Application.builder().token(config.bot_token).build()

    application.add_handler(CommandHandler(["start", "help"], start))
    application.add_handler(MessageHandler(filters=filters.PHOTO, callback=image))

    logger.info("Started polling with %d handlers", len(application.handlers))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
