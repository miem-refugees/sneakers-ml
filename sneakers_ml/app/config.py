from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


# Configs from repository root
class Config(BaseSettings):
    model_config = SettingsConfigDict()

    ml_config_path: str = "config"
    env_file: str = ".env"


config = Config()
logger.info("Loaded config: {}", config)
