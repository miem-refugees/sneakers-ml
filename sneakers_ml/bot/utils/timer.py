from functools import wraps
from time import time


def timed(func):
    """This async decorator prints the execution time for the decorated function."""

    @wraps(func)
    async def wrapper(*args, logger, **kwargs):
        start = time()
        result = await func(*args, logger, **kwargs)
        end = time()
        logger.debug(f"{func.__name__} ran in {round(end - start, 2)}s")
        return result

    return wrapper
