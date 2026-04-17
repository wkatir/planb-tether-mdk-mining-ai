"""Tests for safety constraints and ASIC specs."""

import pytest
from app.config import settings


class TestSafetyThresholds:
    def test_throttle_before_emergency(self):
        assert settings.TEMP_THROTTLE < settings.TEMP_EMERGENCY

    def test_warning_between_throttle_and_emergency(self):
        assert settings.TEMP_THROTTLE < settings.TEMP_WARNING < settings.TEMP_EMERGENCY

    def test_voltage_deviation_reasonable(self):
        assert 0.05 <= settings.VOLTAGE_MAX_DEVIATION <= 0.15

    def test_command_rate_limit_minimum(self):
        assert settings.MIN_COMMAND_INTERVAL_SEC >= 60

    def test_fleet_overclock_cap(self):
        assert settings.MAX_FLEET_OVERCLOCK_PCT <= 0.30


class TestASICSpecs:
    def test_s21_pro_exists(self):
        from app.data.asic_specs import ANTMINER_S21_PRO

        assert ANTMINER_S21_PRO.hashrate_th == 234.0
        assert ANTMINER_S21_PRO.power_watts == 3510.0
        assert ANTMINER_S21_PRO.efficiency_jth == 15.0

    def test_registry_has_four_models(self):
        from app.data.asic_specs import ASIC_REGISTRY

        assert len(ASIC_REGISTRY) == 4
        assert "antminer_s21_pro" in ASIC_REGISTRY

    def test_efficiency_matches_power_over_hashrate(self):
        from app.data.asic_specs import ASIC_REGISTRY

        for key, spec in ASIC_REGISTRY.items():
            calc = spec.power_watts / spec.hashrate_th
            assert abs(calc - spec.efficiency_jth) < 0.5, (
                f"{key}: {calc:.1f} != {spec.efficiency_jth}"
            )

    def test_all_specs_have_valid_cooling_type(self):
        from app.data.asic_specs import ASIC_REGISTRY

        for key, spec in ASIC_REGISTRY.items():
            assert spec.cooling_type in ("air", "hydro"), (
                f"{key}: invalid cooling type {spec.cooling_type}"
            )
