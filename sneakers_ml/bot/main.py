import asyncio
import logging
import sys
from os import getenv

from aiogram import Bot
from aiogram.enums import ParseMode

from sneakers_ml.bot.dispatcher import dispatcher


async def main():
    bot = Bot(getenv("BOT_TOKEN"), parse_mode=ParseMode.HTML)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
