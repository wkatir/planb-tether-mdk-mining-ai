"""Agent profiles: declarative personalization loaded from `agents/<name>.toml`.

A profile says who the agent is (role, rules, output format), which skills it
owns, and how it runs (tool calling, skill activation, LLM sampling). No Python
changes are needed to add or retune an agent.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from app.config import settings

DEFAULT_PROFILE = "copilot"
SKILL_ACTIVATIONS = ("eager", "on_demand")


class ProfileError(ValueError):
    """An agent profile is missing or malformed."""


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    description: str
    role: str
    skills: tuple[str, ...]
    rules: tuple[str, ...] = ()
    output_format: str = ""
    # "eager": skill instructions are injected in the system prompt.
    # "on_demand": only name + description are injected; the model calls load_skill.
    skill_activation: str = "eager"
    tool_calling: bool = False
    # Tools executed before the first LLM call; their output becomes the fleet snapshot.
    preload: tuple[str, ...] = ()
    extra_tools: tuple[str, ...] = ()
    max_steps: int = 4
    temperature: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.skill_activation not in SKILL_ACTIVATIONS:
            raise ProfileError(
                f"{self.name}: skill_activation must be one of {SKILL_ACTIVATIONS}, "
                f"got {self.skill_activation!r}"
            )
        if self.skill_activation == "on_demand" and not self.tool_calling:
            raise ProfileError(f"{self.name}: on_demand skills require tool_calling = true")
        if self.max_steps < 1:
            raise ProfileError(f"{self.name}: max_steps must be >= 1")


def _from_toml(path: Path) -> AgentProfile:
    with path.open("rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)

    data.setdefault("name", path.stem)
    known = {f.name for f in fields(AgentProfile)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ProfileError(f"{path.name}: unknown keys {unknown}")

    for key in ("skills", "rules", "preload", "extra_tools"):
        if key in data:
            data[key] = tuple(data[key])
    for key in ("role", "output_format"):
        if key in data:
            data[key] = data[key].strip()
    try:
        return AgentProfile(**data)
    except TypeError as exc:
        raise ProfileError(f"{path.name}: {exc}") from exc


def load_profile(name: str = DEFAULT_PROFILE, agents_dir: Path | None = None) -> AgentProfile:
    root = Path(agents_dir or settings.agents_dir)
    path = root / f"{name}.toml"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.toml"))
        raise ProfileError(f"unknown agent {name!r}; available: {available}")
    return _from_toml(path)


def list_profiles(agents_dir: Path | None = None) -> list[AgentProfile]:
    root = Path(agents_dir or settings.agents_dir)
    return [_from_toml(p) for p in sorted(root.glob("*.toml"))]
