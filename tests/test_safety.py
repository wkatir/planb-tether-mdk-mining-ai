import time
from unittest.mock import MagicMock

import pytest


class TestSafetyConstraints:
    def test_decision_engine_temp_threshold(self):
        try:
            from app.control.safety import DecisionEngine
        except ImportError:
            try:
                from app.rl.safety import DecisionEngine
            except ImportError:
                pytest.skip("DecisionEngine not implemented yet")

        engine = DecisionEngine()
        mock_state = MagicMock()
        mock_state.T = 94.0
        result = engine.should_throttle(mock_state)
        assert result is False, "Should not throttle below 95°C"

        mock_state.T = 95.0
        result = engine.should_throttle(mock_state)
        assert result is True, "Should throttle at or above 95°C"

        mock_state.T = 100.0
        result = engine.should_throttle(mock_state)
        assert result is True, "Should throttle above 95°C"

    def test_voltage_deviation_limit(self):
        try:
            from app.control.safety import DecisionEngine
        except ImportError:
            try:
                from app.rl.safety import DecisionEngine
            except ImportError:
                pytest.skip("DecisionEngine not implemented yet")

        engine = DecisionEngine()
        nominal_voltage = 5.0

        deviation_5_percent = nominal_voltage * 1.05
        assert engine.is_voltage_safe(deviation_5_percent), (
            f"±5% deviation should be safe: {deviation_5_percent}V"
        )

        deviation_10_percent = nominal_voltage * 1.10
        assert not engine.is_voltage_safe(deviation_10_percent), (
            f"±10% deviation should NOT be safe: {deviation_10_percent}V"
        )

        deviation_minus_10_percent = nominal_voltage * 0.90
        assert not engine.is_voltage_safe(deviation_minus_10_percent), (
            f"-10% deviation should NOT be safe: {deviation_minus_10_percent}V"
        )

    def test_rate_limiting(self):
        try:
            from app.control.safety import DecisionEngine
        except ImportError:
            try:
                from app.rl.safety import DecisionEngine
            except ImportError:
                pytest.skip("DecisionEngine not implemented yet")

        engine = DecisionEngine()
        device_id = "miner_001"

        assert engine.can_send_command(device_id), "First command should be allowed"

        engine.record_command(device_id, "throttle")
        assert not engine.can_send_command(device_id), (
            "Command within 5 min should be rate limited"
        )

        engine._last_command_time[device_id] = time.time() - 301
        assert engine.can_send_command(device_id), (
            "Command after 5 min should be allowed"
        )


class TestVoltageBounds:
    def test_voltage_nominal_from_config(self):
        from app.config import settings

        assert settings.voltage_nominal == 5.0

    def test_asic_voltage_from_specs(self):
        from app.data.asic_specs import ANTMINER_S21

        assert ANTMINER_S21.chip_voltage_v == 0.15
        assert ANTMINER_S21.max_temp_c == 95.0
