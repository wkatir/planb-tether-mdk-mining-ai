# Severity matrix

Thresholds are named, not numbered: read their current values from
"Active thresholds". First matching row wins.

| Priority | Condition (per-miner averages) | Meaning | Expected response |
|---|---|---|---|
| P1 | `avg_temp >= temp_warning_c` **or** `avg_errors >= 2 x risk_avg_errors` | Damage or shutdown is likely within the shift | Act now: diagnose, expect an underclock from the Decision Engine |
| P2 | `avg_temp >= risk_temp_c` **or** `avg_errors >= risk_avg_errors` | Degrading, still inside safe operating limits | Diagnose this shift, schedule maintenance |

`list_miners_at_risk` only returns miners above `risk_temp_c` or
`risk_avg_errors`, so everything it returns is P2 or P1. A miner that is not in
the list has no priority.

Notes:

- **Averages hide spikes.** A miner in failure for the last few hours can still
  average P2 while its latest reading is far above `temp_warning_c`. Say so for
  every P2 miner: its ranking is by average, and its latest reading needs a
  `miner-diagnosis` before the shift ends.
- `temp_emergency_c` is not a triage level. At that temperature the Safety
  layer has already forced a shutdown; report it as an incident, not a ticket.
- Temperature outranks errors when both trigger the same priority: thermal
  damage is irreversible, error bursts are often transient.
- With the synthetic generator, `error_count` is capped per reading, so the
  error conditions rarely fire; in practice the ranking is driven by
  temperature. The error rows matter for real telemetry.
