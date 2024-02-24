import os
from typing import Any


def get_or_default(key: str, default: Any) -> Any:
    return os.environ.get(key) or default


def get_or_raise(key: str) -> Any:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} environment variable is not set")
    return value


class Config:
    bot_token: str = get_or_raise("BOT_TOKEN")
    api_url: str = get_or_default("API_HOST", "http://localhost:8000")

    log_level: str = get_or_default("LOG_LEVEL", "DEBUG")
    container_logging: bool = bool(get_or_default("CONTAINER_LOGGING", False))

    def __str__(self):
        return f"Config(bot_token=***, api_url={self.api_url}, container_logging={self.container_logging})"

    def __repr__(self):
        return self.__str__()
