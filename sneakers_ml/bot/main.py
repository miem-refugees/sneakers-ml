from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from sneakers_ml.bot.config import Config
from sneakers_ml.bot.controller.common import CommonController
from sneakers_ml.bot.controller.image import ImageController


def main():
    config = Config()

    # Controllers
    image_controller = ImageController(config.api_url)
    common_controller = CommonController()

    application = Application.builder().token(config.bot_token).build()
    application.add_handler(CommandHandler(["start", "help"], common_controller.start_handler))
    application.add_handler(MessageHandler(filters=filters.PHOTO, callback=image_controller.image_handler))

    logger.info("Started polling with {} handlers: ", len(application.handlers.values()))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
