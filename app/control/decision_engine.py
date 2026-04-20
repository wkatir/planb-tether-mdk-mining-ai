from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TypedDict

from loguru import logger

from app.config import settings


class ActionType(str, Enum):
    UNDERCLOCK = "underclock"
    OVERCLOCK = "overclock"
    SHUTDOWN = "shutdown"
    NOOP = "noop"
    PLACEHOLDER = "placeholder"


class ControlCommand(TypedDict):
    action_type: ActionType
    clock_multiplier: float
    reason: str
    safety_override: bool
    timestamp: datetime


class DeviceState(TypedDict):
    device_id: str
    temperature: float
    voltage: float
    clock_speed: float
    hash_rate: float
    power_draw: float
    health_score: float | None
    ambient_temp: float
    energy_price: float
    hash_price: float


@dataclass
class RateLimitEntry:
    last_command_time: datetime
    last_action: ActionType


class DecisionEngine:
    RATE_LIMIT_SECONDS: int = 300
    VOLTAGE_TOLERANCE: float = 0.10
    CLOCK_STEP_LIMIT: float = 0.05

    def __init__(self) -> None:
        self._last_command_time: dict[str, RateLimitEntry] = {}
        self._rl_agent_ready: bool = False
        logger.info("DecisionEngine initialized")

    def get_action(self, device_id: str, state: DeviceState) -> ControlCommand:
        logger.debug(f"Evaluating action for device {device_id}")
        ai_action = self._get_ai_recommendation(state)

        if not self._apply_rate_limit(device_id):
            logger.info(f"Rate limit active for device {device_id}, returning NOOP")
            return ControlCommand(
                action_type=ActionType.NOOP,
                clock_multiplier=1.0,
                reason="rate_limited",
                safety_override=False,
                timestamp=datetime.now(),
            )

        safety_result = self._check_safety(ai_action, state)
        if safety_result is not None:
            logger.warning(
                f"Safety override for device {device_id}: {safety_result.reason}"
            )
            self._record_command(device_id, safety_result["action_type"])
            return ControlCommand(
                action_type=safety_result.action_type,
                clock_multiplier=safety_result.clock_multiplier,
                reason=safety_result.reason,
                safety_override=True,
                timestamp=datetime.now(),
            )

        self._record_command(device_id, ai_action["action_type"])
        return ControlCommand(
            action_type=ai_action.action_type,
            clock_multiplier=ai_action.clock_multiplier,
            reason=ai_action.reason,
            safety_override=False,
            timestamp=datetime.now(),
        )

    def _check_safety(
        self, action: ControlCommand, state: DeviceState
    ) -> ControlCommand | None:
        if state["temperature"] >= settings.TEMP_THROTTLE:
            logger.warning(
                f"Temperature {state['temperature']}°C exceeds throttle threshold "
                f"{settings.TEMP_THROTTLE}°C for device {state['device_id']}"
            )
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.8,
                reason="temp_throttle",
                safety_override=True,
                timestamp=datetime.now(),
            )

        voltage_deviation = (
            abs(state["voltage"] - settings.VOLTAGE_NOMINAL) / settings.VOLTAGE_NOMINAL
            if settings.VOLTAGE_NOMINAL > 0
            else 0.0
        )
        if voltage_deviation > self.VOLTAGE_TOLERANCE:
            logger.warning(
                f"Voltage deviation {voltage_deviation:.2%} exceeds tolerance "
                f"{self.VOLTAGE_TOLERANCE:.2%} for device {state['device_id']}"
            )
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.9,
                reason="voltage_protection",
                safety_override=True,
                timestamp=datetime.now(),
            )

        if action.action_type in (ActionType.UNDERCLOCK, ActionType.OVERCLOCK):
            clock_delta = abs(action.clock_multiplier - 1.0)
            if clock_delta > self.CLOCK_STEP_LIMIT:
                logger.warning(
                    f"Clock change {clock_delta:.2%} exceeds step limit "
                    f"{self.CLOCK_STEP_LIMIT:.2%} for device {state['device_id']}"
                )
                limited_multiplier = (
                    1.0 - self.CLOCK_STEP_LIMIT
                    if action.clock_multiplier < 1.0
                    else 1.0 + self.CLOCK_STEP_LIMIT
                )
                return ControlCommand(
                    action_type=ActionType.UNDERCLOCK
                    if action.clock_multiplier < 1.0
                    else ActionType.OVERCLOCK,
                    clock_multiplier=limited_multiplier,
                    reason="clock_step_limited",
                    safety_override=True,
                    timestamp=datetime.now(),
                )

        return None

    def _apply_rate_limit(self, device_id: str) -> bool:
        if device_id not in self._last_command_time:
            return True

        entry = self._last_command_time[device_id]
        elapsed = datetime.now() - entry.last_command_time
        if elapsed < timedelta(seconds=self.RATE_LIMIT_SECONDS):
            logger.debug(
                f"Rate limit check for {device_id}: "
                f"{elapsed.total_seconds():.1f}s elapsed (min: {self.RATE_LIMIT_SECONDS}s)"
            )
            return False

        return True

    def _record_command(self, device_id: str, action: ActionType) -> None:
        self._last_command_time[device_id] = RateLimitEntry(
            last_command_time=datetime.now(),
            last_action=action,
        )

    def _get_ai_recommendation(self, state: DeviceState) -> ControlCommand:
        if not self._rl_agent_ready:
            logger.debug("RL agent not ready, returning placeholder recommendation")
            return ControlCommand(
                action_type=ActionType.PLACEHOLDER,
                clock_multiplier=1.0,
                reason="rl_not_ready",
                safety_override=False,
                timestamp=datetime.now(),
            )

        health = state.get("health_score")
        if health is not None and health < 0.5:
            logger.debug(f"Low health score {health}, recommending underclock")
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.9,
                reason="low_health",
                safety_override=False,
                timestamp=datetime.now(),
            )

        temp = state["temperature"]
        if temp > settings.TEMP_WARNING:
            logger.debug(f"Temperature {temp}°C above warning, recommending underclock")
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.95,
                reason="temp_warning",
                safety_override=False,
                timestamp=datetime.now(),
            )

        logger.debug("RL agent recommending nominal operation")
        return ControlCommand(
            action_type=ActionType.NOOP,
            clock_multiplier=1.0,
            reason="nominal",
            safety_override=False,
            timestamp=datetime.now(),
        )

    def set_rl_agent_ready(self, ready: bool) -> None:
        self._rl_agent_ready = ready
        logger.info(f"RL agent ready status set to: {ready}")
