"""Fleet operator assistant: a profile-driven agent whose skills decide what it knows and can do."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.ai.alerts import AlertSender
from app.ai.fleet_repository import FleetRepository
from app.ai.llm_client import LLMClient
from app.ai.profiles import DEFAULT_PROFILE, AgentProfile, ProfileError, load_profile
from app.ai.prompts import build_system_prompt
from app.ai.skills import SkillRegistry
from app.ai.tools import build_toolbox
from app.ai.tools.skill_tools import LOAD_SKILL, READ_SKILL_RESOURCE, RUN_SKILL_SCRIPT


@dataclass
class AgentTrace:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class FleetAgent:
    """Orchestration only: prompt -> LLM -> tool calls -> answer.

    Identity comes from the profile, knowledge from its skills, and the tool set
    is the union of what those skills declare in `allowed-tools`.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        profile: AgentProfile | str = DEFAULT_PROFILE,
        repository: FleetRepository | None = None,
        skills: SkillRegistry | None = None,
        alert_sender: AlertSender | None = None,
    ) -> None:
        self.client = client or LLMClient()
        self.profile = load_profile(profile) if isinstance(profile, str) else profile
        self.skills = (skills or SkillRegistry.discover()).select(self.profile.skills)

        everything = build_toolbox(
            repository or FleetRepository(), alert_sender or AlertSender(), self.skills
        )
        try:
            self.toolbox = everything.restrict(self._granted_tools())
        except ValueError as exc:
            # A typo in a skill's allowed-tools or in the profile's extra_tools.
            raise ProfileError(f"{self.profile.name}: {exc}") from exc

        missing = sorted(set(self.profile.preload) - set(self.toolbox.names()))
        if missing:
            raise ProfileError(
                f"{self.profile.name}: preload tools {missing} are not granted by its skills"
            )

    def _granted_tools(self) -> list[str]:
        granted = [tool for skill in self.skills for tool in skill.allowed_tools]
        granted += self.profile.extra_tools
        if self.profile.tool_calling:
            if self.profile.skill_activation == "on_demand":
                granted.append(LOAD_SKILL)
            if any(s.resources("references") + s.resources("assets") for s in self.skills):
                granted.append(READ_SKILL_RESOURCE)
            if any(s.resources("scripts") for s in self.skills):
                granted.append(RUN_SKILL_SCRIPT)
        return granted

    def _preload(self, trace: AgentTrace) -> dict[str, str]:
        """Run the profile's preload tools so the model starts with a fleet snapshot."""
        snapshot: dict[str, str] = {}
        for name in self.profile.preload:
            result = self.toolbox.dispatch(name, {})
            snapshot[name] = result.content
            trace.tool_calls.append(
                {"name": name, "preloaded": True, "result_preview": result.content[:200]}
            )
        return snapshot

    def system_prompt(self, trace: AgentTrace | None = None) -> str:
        snapshot = self._preload(trace or AgentTrace(answer=""))
        return build_system_prompt(self.profile, self.skills, snapshot)

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> AgentTrace:
        trace = AgentTrace(answer="")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt(trace)},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        tools = self.toolbox.schemas() if self.profile.tool_calling else None

        for step in range(self.profile.max_steps):
            response = self.client.chat(
                messages=messages,
                tools=tools,
                temperature=self.profile.temperature,
                max_tokens=self.profile.max_tokens,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                trace.answer = msg.content or ""
                return trace

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info(f"[{self.profile.name}][step {step}] tool={name} args={args}")
                result = self.toolbox.dispatch(name, args)
                trace.tool_calls.append(
                    {
                        "name": name,
                        "arguments": args,
                        "result_preview": result.content[:200],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result.content,
                    }
                )

        trace.answer = (
            "I hit my tool-call budget without reaching a final answer. "
            "Try rephrasing or ask a more specific question."
        )
        return trace


if __name__ == "__main__":
    agent = FleetAgent()
    trace = agent.ask("Give me a fleet summary and flag the 3 worst miners.")
    print(trace.answer)
