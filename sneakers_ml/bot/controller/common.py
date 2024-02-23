from telegram import Update
from telegram.ext import ContextTypes


class CommonController:
    def __init__(self):
        pass

    @staticmethod
    async def start_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            f"Hello, {update.message.from_user.full_name}!\n"
            + "Just send me a photo of you sneaker and I'll try to predict it's brand"
        )
