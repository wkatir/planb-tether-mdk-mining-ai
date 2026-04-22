"""Tests for the three-tier DecisionEngine (Safety > AI > Operator)."""

from datetime import datetime, timedelta

import pytest

from app.config import settings
from app.control.decision_engine import (
    ActionType,
    ControlCommand,
    DecisionEngine,
    DeviceState,
    RateLimitEntry,
)


def _base_state(**overrides) -> DeviceState:
    state: DeviceState = {
        "device_id": "miner_001",
        "temperature": 60.0,
        "voltage": settings.VOLTAGE_NOMINAL,
        "clock_speed": 500.0,
        "hash_rate": 234.0,
        "power_draw": 3510.0,
        "health_score": 0.9,
        "ambient_temp": 25.0,
        "energy_price": 0.04,
        "hash_price": 50.0,
    }
    state.update(overrides)
    return state


class TestSafetyLayer:
    def test_thermal_throttle_forces_underclock(self) -> None:
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        state = _base_state(temperature=settings.TEMP_THROTTLE + 1.0)

        cmd = engine.get_action("miner_001", state)

        assert cmd.safety_override is True
        assert cmd.action_type == ActionType.UNDERCLOCK
        assert cmd.reason == "temp_throttle"
        assert cmd.clock_multiplier < 1.0

    def test_voltage_out_of_range_forces_underclock(self) -> None:
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        bad_voltage = settings.VOLTAGE_NOMINAL * 1.15
        state = _base_state(voltage=bad_voltage)

        cmd = engine.get_action("miner_001", state)

        assert cmd.safety_override is True
        assert cmd.action_type == ActionType.UNDERCLOCK
        assert cmd.reason == "voltage_protection"

    def test_safety_wins_over_ai(self) -> None:
        """When BOTH temp and health are bad, safety still takes precedence."""
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        state = _base_state(
            temperature=settings.TEMP_THROTTLE + 5.0, health_score=0.2
        )

        cmd = engine.get_action("miner_001", state)

        assert cmd.safety_override is True
        assert cmd.reason == "temp_throttle"


class TestAILayer:
    def test_low_health_recommends_underclock(self) -> None:
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        state = _base_state(health_score=0.3, temperature=60.0)

        cmd = engine.get_action("miner_001", state)

        assert cmd.safety_override is False
        assert cmd.action_type == ActionType.UNDERCLOCK
        assert cmd.reason == "low_health"

    def test_warning_temp_underclocks_softly(self) -> None:
        """Pre-throttle band: TEMP_NORMAL_MAX < temp < TEMP_THROTTLE."""
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        warning_temp = (settings.TEMP_NORMAL_MAX + settings.TEMP_THROTTLE) / 2
        state = _base_state(temperature=warning_temp)

        cmd = engine.get_action("miner_001", state)

        # Inside the pre-throttle band -> AI layer only, not safety.
        assert cmd.safety_override is False
        assert cmd.action_type == ActionType.UNDERCLOCK
        assert cmd.reason == "temp_warning"

    def test_nominal_state_is_noop(self) -> None:
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        state = _base_state()

        cmd = engine.get_action("miner_001", state)

        assert cmd.action_type == ActionType.NOOP
        assert cmd.reason == "nominal"

    def test_rl_not_ready_returns_placeholder(self) -> None:
        engine = DecisionEngine()
        # rl_agent_ready stays False by default
        state = _base_state()

        cmd = engine.get_action("miner_001", state)

        assert cmd.action_type == ActionType.PLACEHOLDER
        assert cmd.reason == "rl_not_ready"


class TestRateLimit:
    def test_rate_limit_blocks_second_command(self) -> None:
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)
        state = _base_state()

        first = engine.get_action("miner_001", state)
        second = engine.get_action("miner_001", state)

        assert first.reason == "nominal"
        assert second.action_type == ActionType.NOOP
        assert second.reason == "rate_limited"

    def test_rate_limit_expires_after_interval(self) -> None:
        engine = DecisionEngine()
        engine.set_rl_agent_ready(True)

        # Simulate a command from the past, beyond the rate-limit window.
        past = datetime.now() - timedelta(
            seconds=DecisionEngine.RATE_LIMIT_SECONDS + 1
        )
        engine._last_command_time["miner_001"] = RateLimitEntry(
            last_command_time=past, last_action=ActionType.NOOP
        )

        cmd = engine.get_action("miner_001", _base_state())
        assert cmd.reason != "rate_limited"


class TestControlCommandContract:
    def test_is_dataclass_not_dict(self) -> None:
        """Regression: ControlCommand must expose attributes, not dict keys."""
        cmd = ControlCommand(
            action_type=ActionType.NOOP,
            clock_multiplier=1.0,
            reason="test",
            safety_override=False,
        )
        # attribute access should work
        assert cmd.action_type == ActionType.NOOP
        assert cmd.reason == "test"
        # dict access should NOT work
        with pytest.raises(TypeError):
            _ = cmd["reason"]  # type: ignore[index]
