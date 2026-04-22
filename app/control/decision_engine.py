"""
app/control/decision_engine.py — Three-tier safety layer.

Hierarchy (enforced in this order):
    1. Safety Layer   — hard thermal/voltage/rate limits (non-negotiable)
    2. AI Layer       — RL agent or LLM recommendation
    3. Operator Layer — manual commands (still gated by Safety)
"""

from dataclasses import dataclass, field
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


@dataclass
class ControlCommand:
    """Immutable control decision emitted by the engine."""

    action_type: ActionType
    clock_multiplier: float
    reason: str
    safety_override: bool
    timestamp: datetime = field(default_factory=datetime.now)


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
            )

        safety_override = self._check_safety(ai_action, state)
        if safety_override is not None:
            logger.warning(
                f"Safety override for device {device_id}: {safety_override.reason}"
            )
            self._record_command(device_id, safety_override.action_type)
            return safety_override

        self._record_command(device_id, ai_action.action_type)
        return ai_action

    def _check_safety(
        self, action: ControlCommand, state: DeviceState
    ) -> ControlCommand | None:
        if state["temperature"] >= settings.TEMP_THROTTLE:
            logger.warning(
                f"Temperature {state['temperature']}C exceeds throttle threshold "
                f"{settings.TEMP_THROTTLE}C for device {state['device_id']}"
            )
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.8,
                reason="temp_throttle",
                safety_override=True,
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
                    action_type=(
                        ActionType.UNDERCLOCK
                        if action.clock_multiplier < 1.0
                        else ActionType.OVERCLOCK
                    ),
                    clock_multiplier=limited_multiplier,
                    reason="clock_step_limited",
                    safety_override=True,
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
                f"{elapsed.total_seconds():.1f}s elapsed "
                f"(min: {self.RATE_LIMIT_SECONDS}s)"
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
            )

        health = state.get("health_score")
        if health is not None and health < 0.5:
            logger.debug(f"Low health score {health}, recommending underclock")
            # 4% underclock — safely under the 5% step-limit (avoids
            # float-comparison flakiness at the boundary) so the AI
            # recommendation survives gating instead of being clipped.
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.96,
                reason="low_health",
                safety_override=False,
            )

        temp = state["temperature"]
        # Pre-throttle warning band: between TEMP_NORMAL_MAX and TEMP_THROTTLE.
        # Above TEMP_THROTTLE the Safety layer takes over — this branch must
        # live strictly below it or it becomes dead code.
        if settings.TEMP_NORMAL_MAX < temp < settings.TEMP_THROTTLE:
            logger.debug(f"Temperature {temp}C in warning band, recommending underclock")
            return ControlCommand(
                action_type=ActionType.UNDERCLOCK,
                clock_multiplier=0.97,
                reason="temp_warning",
                safety_override=False,
            )

        logger.debug("RL agent recommending nominal operation")
        return ControlCommand(
            action_type=ActionType.NOOP,
            clock_multiplier=1.0,
            reason="nominal",
            safety_override=False,
        )

    def set_rl_agent_ready(self, ready: bool) -> None:
        self._rl_agent_ready = ready
        logger.info(f"RL agent ready status set to: {ready}")
