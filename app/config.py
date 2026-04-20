"""Centralized configuration using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"

    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    models_dir: Path = Path("./data/models")

    fleet_size: int = 50
    simulation_days: int = 30
    sample_interval_minutes: int = 1
    failure_injection_rate: float = 0.05

    DUCKDB_PATH: Path = Path("./data/mining.duckdb")

    TEMP_NORMAL_MAX: float = 70.0
    TEMP_THROTTLE: float = 78.0
    TEMP_WARNING: float = 85.0
    TEMP_EMERGENCY: float = 95.0
    TEMP_REFERENCE: float = 25.0

    VOLTAGE_MAX_DEVIATION: float = 0.10
    VOLTAGE_NOMINAL: float = 340.0

    CLOCK_MAX_CHANGE_PCT: float = 0.05
    MIN_COMMAND_INTERVAL_SEC: int = 300
    MAX_FLEET_OVERCLOCK_PCT: float = 0.20

    TE_BETA: float = 0.008
    TE_ETA_ENV_MIN: float = 0.70
    AUX_POWER_PCT: float = 0.02

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for d in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
