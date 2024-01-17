import logging

from sneakers_ml.bot.loggers.multiline import MultilineLogger

__all__: list[str] = ["database", "webhook", "ml", "setup_logger", "MultilineLogger"]

webhook: logging.Logger = logging.getLogger("bot.webhook")
database: logging.Logger = logging.getLogger("bot.database")
ml: logging.Logger = logging.getLogger("bot.ml")


def setup_logger(level: int = logging.INFO) -> None:
    for name in ["aiogram.middlewares", "aiogram.event", "aiohttp.access"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s | %(name)s: %(message)s",
        datefmt="[%H:%M:%S]",
        level=level,
    )
