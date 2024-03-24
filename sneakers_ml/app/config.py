from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


# Configs from repository root
class Config(BaseSettings):
    model_config = SettingsConfigDict()

    class Redis(BaseSettings):
        host: str = "localhost"
        port: int = 6379

    ml_config_path: str = "config"
    env_file: str = ".env"
    redis: Redis = Redis()


config = Config()
logger.info("Loaded config: {}", config)
