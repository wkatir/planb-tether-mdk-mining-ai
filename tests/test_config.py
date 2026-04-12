import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import settings


class TestSettings:
    def test_settings_loads_correctly(self):
        assert settings.app_env == "development"
        assert settings.log_level == "INFO"
        assert settings.fleet_size == 50
        assert settings.simulation_days == 30

    def test_settings_data_directories(self):
        assert settings.data_dir == Path("./data")
        assert settings.raw_data_dir == Path("./data/raw")
        assert settings.processed_data_dir == Path("./data/processed")
        assert settings.models_dir == Path("./data/models")

    def test_settings_physical_constants(self):
        assert settings.temp_throttle == 95.0
        assert settings.temp_warning == 90.0
        assert settings.temp_shutdown == 105.0
        assert settings.voltage_nominal == 5.0
        assert settings.freq_nominal == 600.0
        assert settings.thermal_time_constant == 300.0
        assert settings.heat_transfer_coeff == 0.1

    def test_settings_generation_params(self):
        assert settings.ambient_temp_mean == 25.0
        assert settings.ambient_temp_std == 5.0
        assert settings.energy_price_mean == 0.08
        assert settings.energy_price_std == 0.02
        assert settings.hashprice_mean == 0.05
        assert settings.hashprice_std == 0.01

    def test_settings_api_config(self):
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000


class TestEnsureDirs:
    def test_ensure_dirs_creates_directories(self, tmp_path):
        from app.config import Settings

        test_settings = Settings(
            DATA_DIR=tmp_path / "data",
            RAW_DATA_DIR=tmp_path / "data" / "raw",
            PROCESSED_DATA_DIR=tmp_path / "data" / "processed",
            MODELS_DIR=tmp_path / "data" / "models",
        )
        test_settings.ensure_dirs()
        assert (tmp_path / "data").exists()
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "data" / "processed").exists()
        assert (tmp_path / "data" / "models").exists()

    def test_ensure_dirs_is_idempotent(self, tmp_path):
        from app.config import Settings

        test_settings = Settings(
            DATA_DIR=tmp_path / "data",
            RAW_DATA_DIR=tmp_path / "data" / "raw",
            PROCESSED_DATA_DIR=tmp_path / "data" / "processed",
            MODELS_DIR=tmp_path / "data" / "models",
        )
        test_settings.ensure_dirs()
        test_settings.ensure_dirs()
        assert (tmp_path / "data").exists()
