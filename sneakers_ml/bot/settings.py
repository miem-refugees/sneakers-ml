import secrets

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bot_token: SecretStr
    use_webhook: bool = False
    drop_pending_updates: bool = False

    enable_db_storage: bool = False
    postgres_host: str = "localhost"
    postgres_db: str = "sneakers_ml_db"
    postgres_password: str = "postgres"
    postgres_port: str = "5432"
    postgres_user: str = "postgres"
    postgres_data: str = "/var/lib/postgresql/data"

    webhook_base_url: str = ""
    webhook_path: str = "/webhook"
    webhook_port: int = 80
    webhook_host: str = ""
    webhook_secret_token: str = Field(default_factory=secrets.token_urlsafe)
    reset_webhook: bool = True

    model_config = SettingsConfigDict(env_file_encoding="utf-8", env_file=".env")

    def build_postgres_dsn(self) -> str:
        return (
            "postgresql+asyncpg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def build_webhook_url(self) -> str:
        return f"{self.webhook_base_url}{self.webhook_path}"
