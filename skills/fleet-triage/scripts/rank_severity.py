"""Rank at-risk miners into P1/P2/P3. Stdlib only: JSON on stdin, JSON on stdout.

Usage:
    python rank_severity.py < payload.json     # {"miners": [...], "thresholds": {...}}

Implements references/severity-matrix.md. Missing thresholds fall back to the
project defaults (app/config.py); pass the active ones to stay in sync.
"""

from __future__ import annotations

import json
import sys

DEFAULT_THRESHOLDS = {"risk_temp_c": 80.0, "temp_warning_c": 85.0, "risk_avg_errors": 50.0}


def classify(miner: dict, th: dict) -> tuple[str, list[str]]:
    temp = float(miner.get("avg_temp") or 0.0)
    errors = float(miner.get("avg_errors") or 0.0)

    if temp >= th["temp_warning_c"] or errors >= 2 * th["risk_avg_errors"]:
        priority = "P1"
    elif temp >= th["risk_temp_c"] or errors >= th["risk_avg_errors"]:
        priority = "P2"
    else:
        priority = "P3"

    reasons = []
    if temp >= th["risk_temp_c"]:
        reasons.append(f"avg_temp {temp:.1f} C")
    if errors >= th["risk_avg_errors"]:
        reasons.append(f"avg_errors {errors:.1f}")
    return priority, reasons or ["below P2 thresholds"]


def rank(miners: list[dict], th: dict) -> list[dict]:
    ranked = []
    for miner in miners:
        priority, reasons = classify(miner, th)
        ranked.append({**miner, "priority": priority, "reasons": reasons})
    # P1 first; inside a priority, hottest first (temperature outranks errors).
    ranked.sort(key=lambda m: (m["priority"], -float(m.get("avg_temp") or 0.0)))
    return ranked


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        miners = payload["miners"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"input must be JSON with a 'miners' list: {exc!r}", file=sys.stderr)
        return 2

    th = {**DEFAULT_THRESHOLDS, **(payload.get("thresholds") or {})}
    ranked = rank(miners, th)
    counts = {p: sum(m["priority"] == p for m in ranked) for p in ("P1", "P2", "P3")}
    json.dump({"counts": counts, "ranked": ranked}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
