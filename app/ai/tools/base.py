"""Tool primitives: a Tool couples its JSON schema with its handler; a Toolbox dispatches."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.ai.fleet_repository import KpiUnavailableError
from app.ai.skills import SkillError


@dataclass
class ToolResult:
    name: str
    content: str


@dataclass(frozen=True)
class Tool:
    """One capability exposed to the LLM. The handler returns a JSON-serializable payload."""

    name: str
    description: str
    handler: Callable[..., Any]
    properties: dict[str, dict[str, Any]] | None = None
    required: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        """OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties or {},
                    "required": list(self.required),
                },
            },
        }


class Toolbox:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def names(self) -> list[str]:
        return sorted(self._tools)

    def restrict(self, allowed: Iterable[str]) -> Toolbox:
        """A new Toolbox holding only `allowed` tools. Unknown names are a config error."""
        allowed = list(allowed)
        unknown = sorted(set(allowed) - set(self._tools))
        if unknown:
            raise ValueError(f"unknown tools {unknown}; available: {self.names()}")
        return Toolbox(self._tools[name] for name in dict.fromkeys(allowed))

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool requested by the LLM and return its JSON payload.

        Failures are returned as `{"error": ...}` so the model can read them and
        recover, instead of aborting the conversation.
        """
        tool = self._tools.get(name)
        if tool is None:
            payload: Any = {"error": f"tool {name!r} is not available to this agent"}
        else:
            try:
                payload = tool.handler(**arguments)
            except TypeError as exc:
                payload = {"error": f"bad args: {exc}"}
            except (KpiUnavailableError, SkillError) as exc:
                payload = {"error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Tool {name} crashed")
                payload = {"error": str(exc)}
        return ToolResult(name=name, content=json.dumps(payload))
