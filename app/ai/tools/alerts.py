"""Alert tool: the only tool with a side effect outside the process."""

from __future__ import annotations

from app.ai.alerts import AlertSender
from app.ai.tools.base import Tool


def alert_tools(sender: AlertSender) -> list[Tool]:
    return [
        Tool(
            name="send_operator_alert",
            description=(
                "Draft and send an email alert about a miner. Used when a "
                "critical failure is predicted. Falls back to logging if "
                "SMTP env vars are not configured (safe for PoC)."
            ),
            handler=sender.send,
            properties={
                "device_id": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            required=("device_id", "subject", "body"),
        )
    ]
