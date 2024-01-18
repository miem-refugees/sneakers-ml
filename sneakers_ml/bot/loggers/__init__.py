import logging
import os

from sneakers_ml.bot.loggers.multiline import MultilineLogger

__all__: list[str] = ["database", "webhook", "ml", "setup_logger", "MultilineLogger"]

webhook: logging.Logger = logging.getLogger("bot.webhook")
database: logging.Logger = logging.getLogger("bot.database")
ml: logging.Logger = logging.getLogger("bot.ml")
api: logging.Logger = logging.getLogger("api")


def setup_logger(level: int = logging.INFO) -> None:
    for name in ["aiogram.middlewares", "aiogram.event", "aiohttp.access"]:
        logging.getLogger(name).setLevel(os.environ.get("LOGLEVEL", logging.DEBUG))

    logging.basicConfig(
        format="%(asctime)s %(levelname)s | %(name)s: %(message)s",
        datefmt="[%H:%M:%S]",
        level=level,
    )
