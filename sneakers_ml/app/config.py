from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


# Configs from repository root
class Config(BaseSettings):
    model_config = SettingsConfigDict()

    redis_host: str = "localhost"
    redis_port: int = 6379

    ml_config_path: str = "config"


config = Config()
logger.info("Loaded config: {}", config)
