---
name: operator-alert
description: Draft and send an operator alert email for a miner using the standard template. Use when a miner is P1, when the operator asks to notify, escalate or alert someone, or when a diagnosis ends in an action that cannot wait for the next shift.
allowed-tools: get_device_status send_operator_alert
metadata:
  version: "1.0"
  domain: maintenance
---

# Operator alert

An alert interrupts someone. Send few, and make each one actionable without
opening the dashboard.

## When to send

Send only if at least one is true:

- the miner is P1 in the triage matrix;
- its latest `chip_temperature_c` is at or above `temp_warning_c`;
- the operator explicitly asked for an alert.

Otherwise answer in chat and say no alert was needed.

## Procedure

1. Call `get_device_status` so the alert carries the latest reading, not an average.
2. Fill `assets/alert-template.md`. Every placeholder must come from tool
   output; if a value is missing write `n/a`, never an estimate.
3. Call `send_operator_alert(device_id, subject, body)` once per miner.
4. Report the delivery result exactly as returned: `sent`, `logged_locally`
   (SMTP not configured, the alert is in the local log) or `smtp_error`.

## Rules

- Subject: `<SEVERITY> <device_id>: <symptom in 6 words or fewer>`. The tool
  adds the `[MDK][device_id]` prefix; do not add it yourself.
- One miner per alert. No fleet summaries by email.
- Never send the same alert twice in a conversation; on `smtp_error` report the
  failure instead of retrying.
- The body states the recommended action as advisory. Clock and voltage
  changes belong to the Decision Engine.
- Plain text only: no markdown, no tables.
