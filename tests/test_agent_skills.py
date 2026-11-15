"""Tests for the agent layer: skills, profiles, tools and the orchestration loop.

No network and no ML pipeline: the repository is seeded from the sample CSV and
the LLM is replaced by a scripted client.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.ai.agent import FleetAgent
from app.ai.alerts import AlertConfig, AlertSender
from app.ai.fleet_repository import FleetRepository, KpiUnavailableError
from app.ai.profiles import AgentProfile, ProfileError, list_profiles, load_profile
from app.ai.prompts import GUARDRAILS, build_system_prompt
from app.ai.skills import Skill, SkillError, SkillRegistry, parse_frontmatter
from app.ai.tools.actions import decide_action
from app.config import PROJECT_ROOT, settings

SAMPLE_CSV = PROJECT_ROOT / "examples" / "kpi_sample.csv"


class ScriptedLLM:
    """Stands in for LLMClient: replays canned responses and records every call."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def say(text):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call(tool, /, **arguments):
    function = SimpleNamespace(name=tool, arguments=json.dumps(arguments))
    tool_call = SimpleNamespace(id=f"call_{tool}", function=function)
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(scope="module")
def repo():
    return FleetRepository.from_csv(SAMPLE_CSV)


@pytest.fixture
def make_agent(repo, tmp_path):
    def _make(profile, *responses):
        llm = ScriptedLLM(*responses)
        sender = AlertSender(AlertConfig(log_file=tmp_path / "alerts.log"))
        return FleetAgent(client=llm, profile=profile, repository=repo, alert_sender=sender), llm

    return _make


class TestSkillFormat:
    def test_frontmatter_scalars_and_nested_metadata(self):
        meta, body = parse_frontmatter(
            '---\nname: demo\ndescription: "Does X: and Y"\nmetadata:\n  version: "1.0"\n---\n# Body\n'
        )
        assert meta == {
            "name": "demo",
            "description": "Does X: and Y",
            "metadata": {"version": "1.0"},
        }
        assert body == "# Body"

    @pytest.mark.parametrize("text", ["# no frontmatter", "---\nname: x\n", "---\nbroken line\n---\n"])
    def test_malformed_frontmatter_is_rejected(self, text):
        with pytest.raises(SkillError):
            parse_frontmatter(text)

    def test_name_must_match_folder(self, tmp_path):
        folder = tmp_path / "some-skill"
        folder.mkdir()
        (folder / "SKILL.md").write_text("---\nname: other\ndescription: d\n---\n", encoding="utf-8")
        with pytest.raises(SkillError, match="must match folder"):
            Skill.from_dir(folder)

    def test_every_bundled_skill_is_valid_and_documented(self):
        registry = SkillRegistry.discover()
        assert registry.names() == [
            "fleet-triage",
            "miner-diagnosis",
            "operator-alert",
            "profitability-analysis",
        ]
        for name in registry.names():
            skill = registry.get(name)
            assert "Use when" in skill.description
            assert skill.allowed_tools
            assert "## Procedure" in skill.instructions()
            assert skill.resources(), f"{name} ships no resources"
            # Every resource the instructions point to must exist.
            for resource in skill.resources():
                assert resource in skill.instructions()

    def test_resource_access_is_confined_to_the_skill(self):
        skill = SkillRegistry.discover().get("fleet-triage")
        assert "P1" in skill.read_resource("references/severity-matrix.md")
        for path in ("../../app/config.py", "SKILL.md", "references/missing.md"):
            with pytest.raises(SkillError):
                skill.read_resource(path)


class TestProfiles:
    def test_bundled_profiles_reference_existing_skills(self):
        registry = SkillRegistry.discover()
        profiles = list_profiles()
        assert [p.name for p in profiles] == ["copilot", "finance", "maintenance"]
        for profile in profiles:
            assert registry.select(profile.skills)

    def test_unknown_profile_lists_available(self):
        with pytest.raises(ProfileError, match="copilot"):
            load_profile("does-not-exist")

    def test_typo_in_profile_key_is_rejected(self, tmp_path):
        (tmp_path / "bad.toml").write_text(
            'display_name = "x"\ndescription = "x"\nrole = "x"\nskils = []\n', encoding="utf-8"
        )
        with pytest.raises(ProfileError, match="skils"):
            load_profile("bad", agents_dir=tmp_path)

    def test_on_demand_requires_tool_calling(self):
        with pytest.raises(ProfileError, match="tool_calling"):
            AgentProfile(
                name="x", display_name="x", description="x", role="x",
                skills=("fleet-triage",), skill_activation="on_demand", tool_calling=False,
            )


class TestRepositoryAndRules:
    def test_fleet_summary_counts_every_miner(self, repo):
        summary = repo.fleet_summary()
        assert summary["total_miners"] == 8
        assert summary["profitable"] + summary["loss_making"] == 8

    def test_at_risk_respects_threshold(self, repo):
        at_risk = repo.miners_at_risk()
        assert at_risk and all(m["avg_temp"] > settings.RISK_TEMP_C for m in at_risk)
        assert repo.miners_at_risk(temp_c=200.0) == []

    def test_unprofitable_sorted_worst_first(self, repo):
        etes = [m["avg_ete"] for m in repo.unprofitable_miners()]
        assert etes == sorted(etes, reverse=True)
        assert all(ete >= settings.ETE_BREAKEVEN for ete in etes)

    def test_unknown_device_is_none(self, repo):
        assert repo.device_status("miner_999") is None

    def test_missing_database_raises_domain_error(self, tmp_path):
        with pytest.raises(KpiUnavailableError):
            FleetRepository(db_path=tmp_path / "nope.duckdb").fleet_summary()

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            ({"chip_temperature_c": 90, "error_count": 99, "economic_te": 2}, "underclock"),
            ({"chip_temperature_c": 60, "error_count": 31, "economic_te": 0.5}, "investigate"),
            ({"chip_temperature_c": 60, "error_count": 0, "economic_te": 1.5}, "investigate"),
            ({"chip_temperature_c": 60, "error_count": 0, "economic_te": None}, "noop"),
        ],
    )
    def test_decide_action_first_matching_rule_wins(self, row, expected):
        assert decide_action(row)[0] == expected


class TestPersonalization:
    """The profile + its skills must change what the agent knows and can do."""

    def test_tools_are_the_union_of_skill_grants(self, make_agent):
        finance, _ = make_agent("finance")
        maintenance, _ = make_agent("maintenance")

        assert "send_operator_alert" not in finance.toolbox.names()
        assert "recommend_action" not in finance.toolbox.names()
        assert {"send_operator_alert", "recommend_action", "load_skill", "run_skill_script"} <= set(
            maintenance.toolbox.names()
        )
        assert "list_unprofitable_miners" not in maintenance.toolbox.names()

    def test_ungranted_tool_is_refused_at_dispatch(self, make_agent, tmp_path):
        finance, _ = make_agent("finance")
        result = finance.toolbox.dispatch(
            "send_operator_alert", {"device_id": "miner_000", "subject": "s", "body": "b"}
        )
        assert "not available" in json.loads(result.content)["error"]
        assert not (tmp_path / "alerts.log").exists()

    def test_guardrails_are_in_every_prompt(self, make_agent):
        for name in ("copilot", "finance", "maintenance"):
            agent, _ = make_agent(name)
            prompt = agent.system_prompt()
            assert all(rule in prompt for rule in GUARDRAILS)

    def test_eager_without_tools_pushes_instructions_and_references(self, make_agent):
        copilot, _ = make_agent("copilot")
        prompt = copilot.system_prompt()
        assert "## Skill: fleet-triage" in prompt
        assert "# Severity matrix" in prompt  # reference inlined: the model cannot pull it
        assert "LIST_MINERS_AT_RISK = [" in prompt

    def test_eager_with_tools_lists_references_instead_of_inlining(self, make_agent):
        finance, _ = make_agent("finance")
        prompt = finance.system_prompt()
        assert "## Skill: profitability-analysis" in prompt
        assert "references/kpi-glossary.md" in prompt
        assert "## ETE bands" not in prompt

    def test_on_demand_only_shows_the_catalog(self, make_agent):
        maintenance, _ = make_agent("maintenance")
        prompt = maintenance.system_prompt()
        assert "- miner-diagnosis:" in prompt
        assert "## Procedure" not in prompt

    def test_preload_must_be_granted_by_a_skill(self, repo):
        profile = AgentProfile(
            name="x", display_name="x", description="x", role="x",
            skills=("profitability-analysis",), preload=("list_miners_at_risk",),
        )
        with pytest.raises(ProfileError, match="list_miners_at_risk"):
            FleetAgent(client=ScriptedLLM(), profile=profile, repository=repo)

    def test_prompt_builder_without_skills_or_snapshot(self):
        profile = AgentProfile(name="x", display_name="x", description="x", role="R", skills=())
        prompt = build_system_prompt(profile, [])
        assert prompt.startswith("# Role\nR")
        assert "# Skills" not in prompt and "# Fleet snapshot" not in prompt


class TestAgentLoop:
    def test_tool_less_profile_answers_in_one_call_with_profile_sampling(self, make_agent):
        agent, llm = make_agent("copilot", say("Fleet is nominal."))
        trace = agent.ask("How is the fleet?")

        assert trace.answer == "Fleet is nominal."
        assert [c["name"] for c in trace.tool_calls] == list(agent.profile.preload)
        assert llm.calls[0]["tools"] is None
        assert llm.calls[0]["temperature"] == 0.2

    def test_on_demand_skill_then_tool_then_answer(self, make_agent):
        agent, llm = make_agent(
            "maintenance",
            call("load_skill", name="miner-diagnosis"),
            call("recommend_action", device_id="miner_007"),
            say("miner_007: underclock."),
        )
        trace = agent.ask("Why is miner_007 hot?")

        assert trace.answer == "miner_007: underclock."
        assert [c["name"] for c in trace.tool_calls] == [
            "list_miners_at_risk",  # preloaded
            "load_skill",
            "recommend_action",
        ]
        # The skill body only reaches the model after it asked for it.
        tool_messages = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
        assert "## Procedure" in json.loads(tool_messages[0]["content"])["instructions"]
        assert json.loads(tool_messages[1]["content"])["recommended_action"] in {
            "underclock", "investigate", "noop",
        }

    def test_alert_goes_to_local_log_when_smtp_is_not_configured(self, make_agent, tmp_path):
        agent, _ = make_agent(
            "maintenance",
            call("send_operator_alert", device_id="miner_007", subject="P1 miner_007: hot", body="b"),
            say("Alerts sent: miner_007"),
        )
        agent.ask("Alert the operator about miner_007")
        assert "[ALERT][miner_007] P1 miner_007: hot" in (tmp_path / "alerts.log").read_text()

    def test_skill_script_runs_through_the_agent(self, make_agent):
        agent, llm = make_agent(
            "maintenance",
            call(
                "run_skill_script",
                name="fleet-triage",
                script="scripts/rank_severity.py",
                payload={"miners": [{"device_id": "a", "avg_temp": 86.0}]},
            ),
            say("done"),
        )
        agent.ask("Triage")
        output = json.loads(llm.calls[-1]["messages"][-1]["content"])
        assert output["ranked"][0]["priority"] == "P1"

    def test_step_budget_is_enforced(self, make_agent):
        agent, llm = make_agent("finance", *[call("get_fleet_summary")] * 4)
        trace = agent.ask("loop forever")
        assert "tool-call budget" in trace.answer
        assert len(llm.calls) == agent.profile.max_steps


class TestSeverityScript:
    SCRIPT = PROJECT_ROOT / "skills" / "fleet-triage" / "scripts" / "rank_severity.py"

    def _run(self, payload):
        return subprocess.run(
            [sys.executable, "-I", str(self.SCRIPT)],
            input=payload, capture_output=True, text=True, check=False,
        )

    def test_matrix_and_ordering(self):
        miners = [
            {"device_id": "warm", "avg_temp": 81.0, "avg_errors": 0},
            {"device_id": "errors", "avg_temp": 60.0, "avg_errors": 120},
            {"device_id": "hot", "avg_temp": 88.0, "avg_errors": 0},
            {"device_id": "fine", "avg_temp": 70.0, "avg_errors": 1},
        ]
        out = json.loads(self._run(json.dumps({"miners": miners})).stdout)
        assert [(m["device_id"], m["priority"]) for m in out["ranked"]] == [
            ("hot", "P1"), ("errors", "P1"), ("warm", "P2"), ("fine", "P3"),
        ]
        assert out["counts"] == {"P1": 2, "P2": 1, "P3": 1}

    def test_thresholds_can_be_overridden(self):
        payload = {"miners": [{"device_id": "a", "avg_temp": 81.0}], "thresholds": {"temp_warning_c": 81}}
        assert json.loads(self._run(json.dumps(payload)).stdout)["ranked"][0]["priority"] == "P1"

    def test_bad_input_exits_non_zero(self):
        proc = self._run("not json")
        assert proc.returncode == 2 and "miners" in proc.stderr
