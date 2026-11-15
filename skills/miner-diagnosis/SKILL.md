---
name: miner-diagnosis
description: Diagnose a single miner - match its latest telemetry to a probable failure mode (thermal, hashboard, PSU) and state the recommended action. Use when the operator names a device_id or asks why a miner is hot, slow, erroring or inefficient.
allowed-tools: get_device_status recommend_action
metadata:
  version: "1.0"
  domain: maintenance
---

# Miner diagnosis

Explain what is probably wrong with one miner and what the operator should do,
without overstating what the telemetry can prove.

## Procedure

1. Call `get_device_status` for the device. If it returns
   `unknown device_id`, say so and stop; never guess a similar id.
2. Compare the reading with the nominal values for its model and with the
   signatures in `references/failure-modes.md`.
3. Call `recommend_action` for the device. Its `recommended_action` and
   `reason` are the answer to "what should we do"; you explain it, you do not
   override it.
4. Answer in this order:
   - **Reading**: the 3-4 values that matter, with units.
   - **Probable cause**: one failure mode, or "no clear signature".
   - **Confidence**: high / medium / low, and what would raise it.
   - **Action**: the recommended action and who executes it.

## Rules

- A diagnosis is a hypothesis from one snapshot. Say "consistent with", not "is".
- If you have no tools, work from the fleet snapshot and state that you could
  not pull the device's latest reading.
- Voltage is not visible to you. PSU problems can only be suspected from power
  behaviour; always ask for an on-site check before a PSU swap.
- `underclock` is executed by the Decision Engine, never by the operator by
  hand and never by you.
- When two modes fit, name the more dangerous one first (thermal > PSU > hashboard).
