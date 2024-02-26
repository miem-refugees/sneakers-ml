import sys

from loguru import logger

from sneakers_ml.bot.config import Config


def new_logger(config: Config):
    logger.remove(0)
    if config.container_logging:
        logger.add(
            sys.stdout,
            level=config.log_level,
            format="[req_id={extra[req_id]}] [{level}] [{message}]",
            backtrace=True,
            colorize=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[req_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            backtrace=True,
            colorize=True,
        )
    return logger
