"""Tools exposed to the LLM agent: read-only views over pre-computed ML outputs."""

from __future__ import annotations

from app.ai.alerts import AlertSender
from app.ai.fleet_repository import FleetRepository
from app.ai.skills import Skill
from app.ai.tools.actions import action_tools
from app.ai.tools.alerts import alert_tools
from app.ai.tools.base import Tool, Toolbox, ToolResult
from app.ai.tools.fleet import fleet_tools
from app.ai.tools.skill_tools import skill_tools

__all__ = ["Tool", "ToolResult", "Toolbox", "build_toolbox"]


def build_toolbox(repo: FleetRepository, alert_sender: AlertSender, skills: list[Skill]) -> Toolbox:
    """Every tool the project knows about. Agents get a restricted view of it."""
    return Toolbox(
        [
            *fleet_tools(repo),
            *action_tools(repo),
            *alert_tools(alert_sender),
            *skill_tools(skills),
        ]
    )
