"""Command-line entry point for the agent layer.

Usage:
    python -m app.ai.cli --list-agents
    python -m app.ai.cli --list-skills
    python -m app.ai.cli --agent maintenance --show-prompt --demo
    python -m app.ai.cli --agent finance --call-tool list_unprofitable_miners --demo
    python -m app.ai.cli --agent maintenance --call-tool get_device_status \
        --args device_id=miner_007 --demo
    python -m app.ai.cli --agent copilot --ask "What needs attention?" --demo

`--demo` reads examples/kpi_sample.csv instead of data/mining.duckdb, so the
agent layer can be inspected without running the ML pipeline. Only `--ask`
needs an LLM API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dotenv import load_dotenv

from app.ai.agent import FleetAgent
from app.ai.fleet_repository import FleetRepository
from app.ai.profiles import DEFAULT_PROFILE, ProfileError, list_profiles
from app.ai.skills import SkillError, SkillRegistry
from app.config import PROJECT_ROOT

DEMO_CSV = PROJECT_ROOT / "examples" / "kpi_sample.csv"


def _parse_tool_args(raw: str) -> dict[str, Any]:
    """Accept JSON, or `key=value,key=value` (no shell quoting needed)."""
    if raw.lstrip().startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("--args must be a JSON object")
        return parsed
    pairs = [item.partition("=") for item in raw.split(",") if item.strip()]
    if any(not sep for _, sep, _ in pairs):
        raise ValueError("--args must be JSON or key=value pairs")
    return {key.strip(): value.strip() for key, _, value in pairs}


def _print_agents() -> None:
    for profile in list_profiles():
        mode = f"{profile.skill_activation} skills, tools {'on' if profile.tool_calling else 'off'}"
        print(f"{profile.name:<12} {profile.display_name} ({mode})")
        print(f"{'':<12} {profile.description}")
        print(f"{'':<12} skills: {', '.join(profile.skills)}\n")


def _print_skills() -> None:
    registry = SkillRegistry.discover()
    for name in registry.names():
        skill = registry.get(name)
        print(f"{skill.name}\n  {skill.description}")
        print(f"  allowed-tools: {' '.join(skill.allowed_tools)}")
        print(f"  resources: {', '.join(skill.resources()) or '-'}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MDK Mining AI - agent layer")
    parser.add_argument("--agent", default=DEFAULT_PROFILE, help="profile name in agents/")
    parser.add_argument("--list-agents", action="store_true")
    parser.add_argument("--list-skills", action="store_true")
    parser.add_argument("--show-prompt", action="store_true", help="print system prompt + tools")
    parser.add_argument("--call-tool", metavar="NAME", help="run one of the agent's tools")
    parser.add_argument(
        "--args", default="{}", help="tool arguments: JSON or key=value,key=value"
    )
    parser.add_argument("--ask", metavar="QUESTION", help="send a question to the LLM")
    parser.add_argument("--demo", action="store_true", help=f"use {DEMO_CSV.name} as the data")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")

    try:
        if args.list_agents:
            _print_agents()
            return 0
        if args.list_skills:
            _print_skills()
            return 0

        repository = FleetRepository.from_csv(DEMO_CSV) if args.demo else FleetRepository()
        agent = FleetAgent(profile=args.agent, repository=repository)
    except (ProfileError, SkillError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.show_prompt:
        print(agent.system_prompt())
        exposed = agent.toolbox.names() if agent.profile.tool_calling else []
        print(f"\n--- tools exposed to the LLM: {', '.join(exposed) or 'none'}")
        return 0

    if args.call_tool:
        try:
            tool_args = _parse_tool_args(args.args)
        except ValueError as exc:  # includes json.JSONDecodeError
            print(f"error: {exc}", file=sys.stderr)
            return 2
        result = agent.toolbox.dispatch(args.call_tool, tool_args)
        print(json.dumps(json.loads(result.content), indent=2))
        return 0

    if args.ask:
        try:
            trace = agent.ask(args.ask)
        except RuntimeError as exc:  # missing API key or openai package
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(trace.answer)
        print("\n--- tool calls")
        for call in trace.tool_calls:
            print(json.dumps(call))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
