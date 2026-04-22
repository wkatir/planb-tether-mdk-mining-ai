"""Fleet operator assistant: context-preloaded chat, plus a tool-calling fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.ai.llm_client import LLMClient
from app.ai.tools import TOOL_SCHEMAS, dispatch_tool, get_fleet_summary, list_miners_at_risk

SYSTEM_PROMPT_BASE = """You are MDK Copilot, an assistant for Bitcoin mining
operators running on Tether's Mining Development Kit (MDK).

Rules:
1. Numbers come from the ML layer. Never invent KPIs or health scores.
2. Be concise and operator-oriented. Use bullets when listing miners.
3. Surface risk first (miners with high temperature, high errors, or
   unprofitable ETE) before general summaries.
4. Your action recommendations are advisory. The Decision Engine has the
   final word on any clock/voltage change.
""".strip()


@dataclass
class AgentTrace:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class FleetAgent:
    def __init__(self, client: LLMClient | None = None, max_steps: int = 4) -> None:
        self.client = client or LLMClient()
        self.max_steps = max_steps

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> AgentTrace:
        summary = get_fleet_summary()
        at_risk = list_miners_at_risk(limit=10)

        context_block = (
            "Current fleet snapshot (pre-computed by the ML layer):\n\n"
            f"FLEET_SUMMARY = {summary.content}\n\n"
            f"MINERS_AT_RISK = {at_risk.content}\n"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE + "\n\n" + context_block},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        response = self.client.chat(messages=messages)
        answer = response.choices[0].message.content or ""

        return AgentTrace(
            answer=answer,
            tool_calls=[
                {"name": "get_fleet_summary", "result_preview": summary.content[:200]},
                {"name": "list_miners_at_risk", "result_preview": at_risk.content[:200]},
            ],
        )

    def ask_with_tools(
        self, question: str, history: list[dict[str, Any]] | None = None
    ) -> AgentTrace:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        trace = AgentTrace(answer="")

        for step in range(self.max_steps):
            response = self.client.chat(messages=messages, tools=TOOL_SCHEMAS)
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
                logger.info(f"[step {step}] tool={name} args={args}")
                result = dispatch_tool(name, args)
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
