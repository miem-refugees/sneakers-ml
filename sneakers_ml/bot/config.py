import os


def get_or_default(key: str, default: str):
    return os.environ.get(key) or default


def get_or_raise(key: str):
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} environment variable is not set")
    return value


class Config:
    bot_token: str = get_or_raise("BOT_TOKEN")
    api_url: str = get_or_default("API_HOST", "http://localhost:8000")
