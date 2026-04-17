"""Tests for configuration module."""

from pathlib import Path

from app.config import settings


class TestSettings:
    def test_thermal_thresholds(self):
        assert settings.TEMP_NORMAL_MAX == 70.0
        assert settings.TEMP_THROTTLE == 78.0
        assert settings.TEMP_WARNING == 85.0
        assert settings.TEMP_EMERGENCY == 95.0

    def test_safety_bounds(self):
        assert settings.VOLTAGE_MAX_DEVIATION == 0.10
        assert settings.CLOCK_MAX_CHANGE_PCT == 0.05
        assert settings.MIN_COMMAND_INTERVAL_SEC == 300
        assert settings.MAX_FLEET_OVERCLOCK_PCT == 0.20

    def test_kpi_parameters(self):
        assert settings.TE_BETA == 0.008
        assert settings.TE_ETA_ENV_MIN == 0.70
        assert settings.AUX_POWER_PCT == 0.02
        assert settings.TEMP_REFERENCE == 25.0

    def test_data_generation_defaults(self):
        assert settings.fleet_size == 50
        assert settings.simulation_days == 30
        assert settings.sample_interval_minutes == 1
        assert settings.failure_injection_rate == 0.05

    def test_api_config(self):
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_data_paths_are_path_objects(self):
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.raw_data_dir, Path)
        assert isinstance(settings.processed_data_dir, Path)
        assert isinstance(settings.models_dir, Path)


class TestEnsureDirs:
    def test_ensure_dirs_creates_directories(self, tmp_path):
        from app.config import Settings

        s = Settings(
            data_dir=tmp_path / "d",
            raw_data_dir=tmp_path / "d" / "raw",
            processed_data_dir=tmp_path / "d" / "proc",
            models_dir=tmp_path / "d" / "models",
        )
        s.ensure_dirs()
        assert (tmp_path / "d").exists()
        assert (tmp_path / "d" / "raw").exists()
