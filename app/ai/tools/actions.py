"""Soft action recommendation: a pure-Python rule on top of ML outputs."""

from __future__ import annotations

from typing import Any

from app.ai.fleet_repository import FleetRepository
from app.ai.tools.base import Tool
from app.config import settings


def decide_action(
    row: dict[str, Any],
    temp_throttle: float = settings.TEMP_THROTTLE,
    error_count: int = settings.ACTION_ERROR_COUNT,
    ete_breakeven: float = settings.ETE_BREAKEVEN,
) -> tuple[str, str]:
    """Map a device snapshot to (action, reason). First matching rule wins."""
    temp = float(row.get("chip_temperature_c") or 0.0)
    errors = int(row.get("error_count") or 0)
    ete = row.get("economic_te")

    if temp >= temp_throttle:
        return "underclock", "thermal_throttle"
    if errors > error_count:
        return "investigate", "high_error_rate"
    if ete is not None and ete > ete_breakeven:
        return "investigate", "unprofitable_ete_gt_1"
    return "noop", "nominal"


def action_tools(repo: FleetRepository) -> list[Tool]:
    def recommend_action(device_id: str) -> dict[str, Any]:
        row = repo.device_status(device_id)
        if row is None:
            return {"error": f"unknown device_id={device_id}"}
        action, reason = decide_action(row)
        return {
            "device_id": device_id,
            "recommended_action": action,
            "reason": reason,
            "note": "soft recommendation; DecisionEngine gates execution",
            "snapshot": row,
        }

    return [
        Tool(
            name="recommend_action",
            description=(
                "Given a device_id, returns a suggested control action "
                "('underclock', 'noop', 'investigate') with plain-language "
                "reasoning. This is a SOFT recommendation - the DecisionEngine "
                "safety layer still gates execution."
            ),
            handler=recommend_action,
            properties={"device_id": {"type": "string"}},
            required=("device_id",),
        )
    ]
