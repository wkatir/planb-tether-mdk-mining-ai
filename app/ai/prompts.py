"""System prompt assembly: fixed guardrails + profile + skills + live fleet snapshot."""

from __future__ import annotations

from app.ai.profiles import AgentProfile
from app.ai.skills import Skill
from app.config import settings

# Safety invariants of the "ML first, LLM on top" design. They live in code, not
# in agents/*.toml, so that no profile can drop or weaken them.
GUARDRAILS = (
    "Numbers come from the ML layer. Never invent KPIs or health scores.",
    (
        "Your action recommendations are advisory. The Decision Engine has the "
        "final word on any clock/voltage change."
    ),
    (
        "Use only the tools and skills listed here. If a request needs a capability "
        "you do not have, say so instead of improvising."
    ),
)


def active_thresholds() -> dict[str, float]:
    """Thresholds the tools apply, surfaced to the model so skills never restate numbers."""
    return {
        "temp_throttle_c": settings.TEMP_THROTTLE,
        "temp_warning_c": settings.TEMP_WARNING,
        "temp_emergency_c": settings.TEMP_EMERGENCY,
        "risk_temp_c": settings.RISK_TEMP_C,
        "risk_avg_errors": settings.RISK_AVG_ERRORS,
        "action_error_count": settings.ACTION_ERROR_COUNT,
        "ete_breakeven": settings.ETE_BREAKEVEN,
    }


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _skills_section(profile: AgentProfile, skills: list[Skill]) -> str:
    if profile.skill_activation == "on_demand":
        catalog = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        return (
            "# Skills\n"
            "You have the skills below. Before answering, call `load_skill` for the "
            "one that matches the request and follow its instructions.\n\n" + catalog
        )

    blocks = ["# Skills\nFollow the procedure of the skill that matches the request."]
    for skill in skills:
        block = f"## Skill: {skill.name}\n{skill.description}\n\n{skill.instructions()}"
        if profile.tool_calling:
            # The model can pull references itself; keep the prompt small.
            listed = skill.resources()
            if listed:
                block += "\n\nResources (read with `read_skill_resource`): " + ", ".join(listed)
        else:
            # No tool calling: the model cannot pull, so references are pushed.
            for ref in skill.resources("references") + skill.resources("assets"):
                block += f"\n\n### {skill.name}/{ref}\n{skill.read_resource(ref)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_system_prompt(
    profile: AgentProfile, skills: list[Skill], snapshot: dict[str, str] | None = None
) -> str:
    """Compose the prompt as labelled sections, most stable content first."""
    sections = [
        f"# Role\n{profile.role}",
        f"# Guardrails (non-negotiable)\n{_numbered(GUARDRAILS)}",
    ]
    if profile.rules:
        sections.append(f"# Operating rules\n{_numbered(profile.rules)}")
    if skills:
        sections.append(_skills_section(profile, skills))
    if profile.output_format:
        sections.append(f"# Output format\n{profile.output_format}")

    thresholds = "\n".join(f"- {k} = {v}" for k, v in active_thresholds().items())
    sections.append(f"# Active thresholds\n{thresholds}")

    if snapshot:
        blocks = "\n\n".join(f"{name.upper()} = {content}" for name, content in snapshot.items())
        sections.append(f"# Fleet snapshot (pre-computed by the ML layer)\n{blocks}")

    return "\n\n".join(sections)
