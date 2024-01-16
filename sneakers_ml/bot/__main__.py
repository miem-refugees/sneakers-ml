from aiogram import Bot, Dispatcher

from sneakers_ml.bot.factories import create_bot, create_dispatcher
from sneakers_ml.bot.loggers import setup_logger
from sneakers_ml.bot.runners import run_polling, run_webhook
from sneakers_ml.bot.settings import Settings


def main() -> None:
    setup_logger()
    settings: Settings = Settings()
    dispatcher: Dispatcher = create_dispatcher(settings=settings)
    bot: Bot = create_bot(settings=settings)
    if settings.use_webhook:
        return run_webhook(dispatcher=dispatcher, bot=bot, settings=settings)
    return run_polling(dispatcher=dispatcher, bot=bot)


if __name__ == "__main__":
    main()
