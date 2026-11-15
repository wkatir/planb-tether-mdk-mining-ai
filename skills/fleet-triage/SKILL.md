---
name: fleet-triage
description: Rank at-risk miners into P1/P2/P3 priorities and report them risk-first. Use when the operator asks for a fleet overview, the worst miners, what needs attention, or a shift handover summary.
allowed-tools: get_fleet_summary list_miners_at_risk
metadata:
  version: "1.0"
  domain: operations
---

# Fleet triage

Turn the at-risk list into a short, prioritized work queue for the operator on shift.

## Procedure

1. Get the data: `LIST_MINERS_AT_RISK` and `GET_FLEET_SUMMARY` from the fleet
   snapshot if present, otherwise call `list_miners_at_risk` and `get_fleet_summary`.
2. Assign each miner a priority with the severity matrix in
   `references/severity-matrix.md`, using the values under "Active thresholds".
   If you can run scripts, use `scripts/rank_severity.py` instead of ranking by
   hand: it applies the same matrix deterministically.
3. Report P1 first, then P2, then P3. Within a priority keep the script/tool order.
4. Close with one line of fleet context (total miners, how many are at risk,
   profitable vs loss-making).

## Rules

- One line per miner: `device_id (model) - priority - the metric that triggered it`.
- Quote the triggering value with its unit (`84.6 C`, `62 errors avg`). Do not
  round a value across a threshold.
- An empty at-risk list is a valid result: say the fleet is nominal. Do not pad
  the answer with healthy miners.
- If the tool returns `{"error": ...}`, relay the error and the fix it suggests.
  Do not estimate the fleet state.
- Triage only ranks. For the cause of a specific miner, the `miner-diagnosis`
  skill applies.

## Script

`scripts/rank_severity.py` reads JSON on stdin and prints JSON:

```json
{"miners": [{"device_id": "miner_007", "avg_temp": 84.6, "avg_errors": 0.0}],
 "thresholds": {"risk_temp_c": 80, "temp_warning_c": 85, "risk_avg_errors": 50}}
```

`thresholds` is optional; pass the active thresholds so the script and the tools agree.
