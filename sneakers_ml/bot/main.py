from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from sneakers_ml.bot.config import Config
from sneakers_ml.bot.controller.common import CommonController
from sneakers_ml.bot.controller.image import ImageController
from sneakers_ml.bot.logger import new_logger


def main():
    config = Config()
    logger = new_logger(config)

    # Controllers
    image_controller = ImageController(config.api_url, logger)
    common_controller = CommonController(logger)

    application = Application.builder().token(config.bot_token).build()
    application.add_error_handler(common_controller.error_handler)
    application.add_handler(CommandHandler(["start", "help"], common_controller.start_handler))
    application.add_handler(MessageHandler(filters=filters.PHOTO, callback=image_controller.image_handler))

    with logger.contextualize(req_id="init"):
        logger.debug("Loaded config: {}", config)
        logger.info("Started polling with {} handlers: ", len(application.handlers.values()))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
