"""Meta-tools that let the LLM pull skill instructions, references and scripts on demand."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from app.ai.skills import Skill, SkillError
from app.ai.tools.base import Tool

LOAD_SKILL = "load_skill"
READ_SKILL_RESOURCE = "read_skill_resource"
RUN_SKILL_SCRIPT = "run_skill_script"
SCRIPT_TIMEOUT_SEC = 10


def skill_tools(skills: list[Skill]) -> list[Tool]:
    """Tools scoped to `skills`: an agent can only reach the skills of its own profile."""
    by_name = {s.name: s for s in skills}

    def _skill(name: str) -> Skill:
        if name not in by_name:
            raise SkillError(f"skill {name!r} is not enabled; enabled: {sorted(by_name)}")
        return by_name[name]

    def load_skill(name: str) -> dict[str, Any]:
        skill = _skill(name)
        return {
            "skill": skill.name,
            "instructions": skill.instructions(),
            "resources": skill.resources(),
        }

    def read_skill_resource(name: str, path: str) -> dict[str, Any]:
        return {"skill": name, "path": path, "content": _skill(name).read_resource(path)}

    def run_skill_script(name: str, script: str, payload: dict[str, Any] | None = None) -> Any:
        target = _skill(name).resource_path(script)
        if target.parent.name != "scripts" or target.suffix != ".py":
            raise SkillError(f"{script!r} is not a script of skill {name!r}")
        # Isolated interpreter (-I): the script only sees the stdlib and its stdin.
        proc = subprocess.run(
            [sys.executable, "-I", str(target)],
            input=json.dumps(payload or {}),
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SEC,
            check=False,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or f"exit code {proc.returncode}"}
        return json.loads(proc.stdout)

    skill_name = {"type": "string", "enum": sorted(by_name)}
    return [
        Tool(
            name=LOAD_SKILL,
            description=(
                "Load the full instructions of one of your skills. Call this "
                "before answering a request that matches the skill's description."
            ),
            handler=load_skill,
            properties={"name": skill_name},
            required=("name",),
        ),
        Tool(
            name=READ_SKILL_RESOURCE,
            description=(
                "Read a reference or asset file bundled with a skill, e.g. "
                "'references/failure-modes.md'. Paths are listed by load_skill."
            ),
            handler=read_skill_resource,
            properties={"name": skill_name, "path": {"type": "string"}},
            required=("name", "path"),
        ),
        Tool(
            name=RUN_SKILL_SCRIPT,
            description=(
                "Run a deterministic helper script bundled with a skill, e.g. "
                "'scripts/rank_severity.py'. `payload` is passed as JSON on stdin; "
                "the script's JSON output is returned."
            ),
            handler=run_skill_script,
            properties={
                "name": skill_name,
                "script": {"type": "string"},
                "payload": {"type": "object"},
            },
            required=("name", "script"),
        ),
    ]
