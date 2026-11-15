"""Read-only fleet tools: thin adapters from the LLM to FleetRepository."""

from __future__ import annotations

from typing import Any

from app.ai.fleet_repository import FleetRepository
from app.ai.tools.base import Tool


def fleet_tools(repo: FleetRepository) -> list[Tool]:
    def get_device_status(device_id: str) -> Any:
        status = repo.device_status(device_id)
        return status or {"error": f"unknown device_id={device_id}"}

    return [
        Tool(
            name="get_fleet_summary",
            description=(
                "Aggregated fleet KPIs: miner count, avg TE/ETE, profitable vs "
                "loss-making miners, avg temperature."
            ),
            handler=repo.fleet_summary,
        ),
        Tool(
            name="get_device_status",
            description=(
                "Latest telemetry + computed KPIs for a single miner. Returns "
                "hashrate, temperature, power, TE, ETE, operating mode, model."
            ),
            handler=get_device_status,
            properties={"device_id": {"type": "string", "description": "e.g. 'miner_007'"}},
            required=("device_id",),
        ),
        Tool(
            name="list_miners_at_risk",
            description=(
                "Miners whose avg temperature or error count crossed risk "
                "thresholds. Already pre-ranked by severity."
            ),
            handler=lambda limit=10: repo.miners_at_risk(limit=limit),
            properties={
                "limit": {
                    "type": "integer",
                    "description": "Max number of miners to return.",
                    "default": 10,
                }
            },
        ),
        Tool(
            name="list_unprofitable_miners",
            description=(
                "Miners whose average ETE is at or above break-even (they cost "
                "more to run than they earn), worst first, with profit density."
            ),
            handler=lambda limit=10: repo.unprofitable_miners(limit=limit),
            properties={
                "limit": {
                    "type": "integer",
                    "description": "Max number of miners to return.",
                    "default": 10,
                }
            },
        ),
    ]
