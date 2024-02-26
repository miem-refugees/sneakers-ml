from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes


class CommonController:
    def __init__(self, logger):
        self._logger = logger

    @staticmethod
    async def start_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            f"Hello, {update.message.from_user.full_name}!\n"
            + "Just send me a photo of you sneaker and I'll try to predict it's brand"
        )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            raise context.error
        except TelegramError as e:
            self._logger.bind(req_id=update.update_id).critical("A Telegram error occurred: {}", e)
            raise e
        except Exception as e:
            self._logger.bind(req_id=update.update_id).error("Update {} caused error {}", update, e)
            await update.message.reply_text("Sorry, something has broken =(\nTry again later")
            raise e
