# Severity matrix

Thresholds are named, not numbered: read their current values from
"Active thresholds". First matching row wins.

| Priority | Condition (per-miner averages) | Meaning | Expected response |
|---|---|---|---|
| P1 | `avg_temp >= temp_warning_c` **or** `avg_errors >= 2 x risk_avg_errors` | Damage or shutdown is likely within the shift | Act now: diagnose, expect an underclock from the Decision Engine |
| P2 | `avg_temp >= risk_temp_c` **or** `avg_errors >= risk_avg_errors` | Degrading, still inside safe operating limits | Diagnose this shift, schedule maintenance |
| P3 | In the at-risk list but below both P2 conditions | Borderline | Watch; re-check next shift |

Notes:

- `temp_emergency_c` is not a triage level. At that temperature the Safety
  layer has already forced a shutdown; report it as an incident, not a ticket.
- Temperature outranks errors when both trigger the same priority: thermal
  damage is irreversible, error bursts are often transient.
- Averages hide spikes. A P3 miner whose latest reading is above
  `temp_throttle_c` deserves a diagnosis even though its average is fine.
