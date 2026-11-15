---
name: fleet-triage
description: Rank at-risk miners into P1/P2 priorities and report them risk-first. Use when the operator asks for a fleet overview, the worst miners, what needs attention, or a shift handover summary.
allowed-tools: get_fleet_summary list_miners_at_risk
metadata:
  version: "1.1"
  domain: operations
---

# Fleet triage

Turn the at-risk list into a short, prioritized work queue for the operator on shift.

## Procedure

1. Get the data: `LIST_MINERS_AT_RISK` and `GET_FLEET_SUMMARY` from the fleet
   snapshot if present, otherwise call `list_miners_at_risk` and `get_fleet_summary`.
2. Assign each miner a priority with the severity matrix in
   `references/severity-matrix.md`, using the values under "Active thresholds".
   If you have the `run_skill_script` tool, do not rank by hand; call
   `run_skill_script(name="fleet-triage", script="scripts/rank_severity.py",
   payload={"miners": <the at-risk list>, "thresholds": <active thresholds>})`.
   It applies the same matrix deterministically.
3. Report P1 first, then P2, in the order the script returns (hottest first).
   The order of `list_miners_at_risk` itself is not a priority ranking.
4. Close with one line of fleet context (total miners, how many are at risk,
   profitable vs loss-making).

## Rules

- One line per miner: `device_id (model) - priority - the metric that triggered it`.
- Quote the triggering value with its unit (`84.6 C`, `62 errors avg`). Do not
  round a value across a threshold.
- Priorities come from averages. State that, and point P2 miners to a
  diagnosis of their latest reading (the `miner-diagnosis` skill).
- An empty at-risk list is a valid result: say the fleet is nominal. Do not pad
  the answer with healthy miners.
- If the tool returns `{"error": ...}`, relay the error and the fix it suggests.
  Do not estimate the fleet state.

## Script

`scripts/rank_severity.py` takes a JSON file path (or JSON on stdin) and prints JSON.
Payload shape, with example values:

```json
{"miners": [{"device_id": "miner_007", "avg_temp": 84.6, "avg_errors": 0.0}],
 "thresholds": {"risk_temp_c": 80, "temp_warning_c": 85, "risk_avg_errors": 50}}
```

`thresholds` is optional; pass the active thresholds so the script and the tools agree.
