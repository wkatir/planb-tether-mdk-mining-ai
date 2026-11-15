"""Skill discovery and loading.

A skill is a folder holding a SKILL.md (YAML frontmatter + markdown instructions)
plus optional `references/`, `assets/` and `scripts/` resources. Only the
frontmatter is read at discovery time; instructions and resources are read when
an agent actually needs them (progressive disclosure).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

SKILL_FILE = "SKILL.md"
RESOURCE_DIRS = ("references", "assets", "scripts")
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class SkillError(ValueError):
    """A skill is missing, malformed, or a resource path is not allowed."""


def parse_frontmatter(text: str) -> tuple[dict[str, str | dict[str, str]], str]:
    """Split a SKILL.md into (frontmatter, body).

    Supports the subset of YAML the skill format needs: `key: value` scalars and
    one level of nested mapping (used by `metadata:`). Kept dependency-free on
    purpose; anything fancier is rejected instead of silently misread.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError("SKILL.md must start with a '---' frontmatter block")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        raise SkillError("frontmatter block is never closed with '---'") from None

    meta: dict[str, str | dict[str, str]] = {}
    nested: dict[str, str] | None = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        key, sep, value = raw.strip().partition(":")
        if not sep:
            raise SkillError(f"invalid frontmatter line: {raw!r}")
        key, value = key.strip(), value.strip().strip("\"'")
        if raw[0].isspace():
            if nested is None:
                raise SkillError(f"unexpected indentation: {raw!r}")
            nested[key] = value
        elif value:
            meta[key] = value
            nested = None
        else:
            nested = {}
            meta[key] = nested

    return meta, "\n".join(lines[end + 1 :]).strip()


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, path: Path) -> Skill:
        skill_file = path / SKILL_FILE
        if not skill_file.is_file():
            raise SkillError(f"{path} has no {SKILL_FILE}")
        meta, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))

        name, description = meta.get("name"), meta.get("description")
        if not isinstance(name, str) or not _NAME_RE.match(name) or len(name) > MAX_NAME_LEN:
            raise SkillError(f"{skill_file}: 'name' must be lowercase-hyphenated, got {name!r}")
        if name != path.name:
            raise SkillError(f"{skill_file}: name {name!r} must match folder {path.name!r}")
        if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_LEN:
            raise SkillError(f"{skill_file}: 'description' is required (max 1024 chars)")

        allowed = meta.get("allowed-tools", "")
        metadata = meta.get("metadata", {})
        return cls(
            name=name,
            description=description,
            path=path,
            allowed_tools=tuple(allowed.split()) if isinstance(allowed, str) else (),
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def instructions(self) -> str:
        """The markdown body of SKILL.md (read on demand, not at discovery)."""
        _, body = parse_frontmatter((self.path / SKILL_FILE).read_text(encoding="utf-8"))
        return body

    def resources(self, kind: str | None = None) -> list[str]:
        """Relative posix paths of bundled files, optionally for one resource dir."""
        dirs = (kind,) if kind else RESOURCE_DIRS
        return sorted(
            f.relative_to(self.path).as_posix()
            for d in dirs
            for f in (self.path / d).rglob("*")
            if f.is_file() and "__pycache__" not in f.parts
        )

    def resource_path(self, relative: str) -> Path:
        """Resolve a bundled file, refusing anything outside the resource dirs."""
        root = self.path.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise SkillError(f"{relative!r} escapes skill {self.name!r}")
        parts = target.relative_to(root).parts
        if len(parts) < 2 or parts[0] not in RESOURCE_DIRS or not target.is_file():
            raise SkillError(
                f"skill {self.name!r} has no resource {relative!r}; available: {self.resources()}"
            )
        return target

    def read_resource(self, relative: str) -> str:
        return self.resource_path(relative).read_text(encoding="utf-8")


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills = {s.name: s for s in skills}

    @classmethod
    def discover(cls, root: Path | None = None) -> SkillRegistry:
        """Load every `<root>/<skill-name>/SKILL.md`. A malformed skill fails loudly."""
        root = Path(root or settings.skills_dir)
        if not root.is_dir():
            raise SkillError(f"skills directory not found: {root}")
        return cls(
            Skill.from_dir(d)
            for d in sorted(root.iterdir())
            if d.is_dir() and (d / SKILL_FILE).is_file()
        )

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillError(f"unknown skill {name!r}; available: {self.names()}") from None

    def select(self, names: Iterable[str]) -> list[Skill]:
        return [self.get(n) for n in names]
