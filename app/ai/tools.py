"""Tools exposed to the LLM agent: read-only views over pre-computed ML outputs."""

from __future__ import annotations

import json
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger

from app.config import settings


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_fleet_summary",
            "description": (
                "Aggregated fleet KPIs: miner count, avg TE/ETE, profitable vs "
                "loss-making miners, avg temperature."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": (
                "Latest telemetry + computed KPIs for a single miner. Returns "
                "hashrate, temperature, power, TE, ETE, operating mode, model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "e.g. 'miner_007'",
                    }
                },
                "required": ["device_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_miners_at_risk",
            "description": (
                "Miners whose avg temperature or error count crossed risk "
                "thresholds. Already pre-ranked by severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of miners to return.",
                        "default": 10,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_action",
            "description": (
                "Given a device_id, returns a suggested control action "
                "('underclock', 'noop', 'investigate') with plain-language "
                "reasoning. This is a SOFT recommendation - the DecisionEngine "
                "safety layer still gates execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                },
                "required": ["device_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_operator_alert",
            "description": (
                "Draft and send an email alert about a miner. Used when a "
                "critical failure is predicted. Falls back to logging if "
                "SMTP env vars are not configured (safe for PoC)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["device_id", "subject", "body"],
            },
        },
    },
]


@dataclass
class ToolResult:
    name: str
    content: str


def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(settings.DUCKDB_PATH), read_only=True)


def _kpi_table_exists() -> bool:
    try:
        conn = _conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'kpi'"
        ).fetchone()
        conn.close()
        return bool(row and row[0] > 0)
    except Exception:  # noqa: BLE001
        return False


def get_fleet_summary() -> ToolResult:
    if not _kpi_table_exists():
        return ToolResult(
            name="get_fleet_summary",
            content=json.dumps({"error": "KPI table empty. Run `python -m app.run_all` first."}),
        )
    conn = _conn()
    df = conn.execute("""
        WITH per_device AS (
            SELECT
                device_id,
                AVG(asic_hashrate_th)    AS dev_hashrate_th,
                AVG(chip_temperature_c)  AS dev_temp_c,
                AVG(asic_power_w)        AS dev_power_w,
                AVG(true_efficiency_jth) AS dev_te_jth,
                AVG(economic_te)         AS dev_ete
            FROM kpi
            WHERE true_efficiency_jth IS NOT NULL
            GROUP BY device_id
        )
        SELECT
            COUNT(*)                                              AS total_miners,
            AVG(dev_hashrate_th)                                  AS avg_hashrate_th,
            AVG(dev_temp_c)                                       AS avg_temp_c,
            AVG(dev_power_w)                                      AS avg_power_w,
            AVG(dev_te_jth)                                       AS avg_te_jth,
            AVG(dev_ete)                                          AS avg_ete,
            COUNT(CASE WHEN dev_ete <  1.0 THEN 1 END)            AS profitable,
            COUNT(CASE WHEN dev_ete >= 1.0 THEN 1 END)            AS loss_making
        FROM per_device
    """).fetchdf()
    conn.close()
    return ToolResult(
        name="get_fleet_summary",
        content=df.to_json(orient="records"),
    )


def get_device_status(device_id: str) -> ToolResult:
    if not _kpi_table_exists():
        return ToolResult(
            name="get_device_status",
            content=json.dumps({"error": "KPI table empty"}),
        )
    conn = _conn()
    df = conn.execute(
        """
        SELECT
            device_id, model, operating_mode,
            asic_hashrate_th, asic_power_w, chip_temperature_c,
            fan_speed_rpm, error_count,
            true_efficiency_jth, economic_te, profit_density,
            timestamp
        FROM kpi
        WHERE device_id = $1
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        [device_id],
    ).fetchdf()
    conn.close()
    if df.empty:
        return ToolResult(
            name="get_device_status",
            content=json.dumps({"error": f"unknown device_id={device_id}"}),
        )
    return ToolResult(
        name="get_device_status",
        content=df.to_json(orient="records", date_format="iso"),
    )


def list_miners_at_risk(limit: int = 10) -> ToolResult:
    if not _kpi_table_exists():
        return ToolResult(
            name="list_miners_at_risk",
            content=json.dumps({"error": "KPI table empty"}),
        )
    conn = _conn()
    df = conn.execute(
        """
        SELECT device_id, model,
               AVG(chip_temperature_c) AS avg_temp,
               AVG(error_count)        AS avg_errors,
               AVG(true_efficiency_jth) AS avg_te
        FROM kpi
        WHERE true_efficiency_jth IS NOT NULL
        GROUP BY device_id, model
        HAVING AVG(chip_temperature_c) > 80 OR AVG(error_count) > 50
        ORDER BY avg_errors DESC
        LIMIT $1
        """,
        [int(limit)],
    ).fetchdf()
    conn.close()
    return ToolResult(
        name="list_miners_at_risk",
        content=df.to_json(orient="records"),
    )


def recommend_action(device_id: str) -> ToolResult:
    """Pure-Python rule on top of ML outputs. The LLM wraps this in prose."""
    status = json.loads(get_device_status(device_id).content)
    if isinstance(status, dict) and "error" in status:
        return ToolResult(name="recommend_action", content=json.dumps(status))

    row = status[0] if isinstance(status, list) else status
    temp = float(row.get("chip_temperature_c", 0.0))
    errors = int(row.get("error_count", 0))
    ete = row.get("economic_te")

    if temp >= settings.TEMP_THROTTLE:
        action, reason = "underclock", "thermal_throttle"
    elif errors > 30:
        action, reason = "investigate", "high_error_rate"
    elif ete is not None and ete > 1.0:
        action, reason = "investigate", "unprofitable_ete_gt_1"
    else:
        action, reason = "noop", "nominal"

    return ToolResult(
        name="recommend_action",
        content=json.dumps(
            {
                "device_id": device_id,
                "recommended_action": action,
                "reason": reason,
                "note": "soft recommendation; DecisionEngine gates execution",
                "snapshot": row,
            }
        ),
    )


def send_operator_alert(device_id: str, subject: str, body: str) -> ToolResult:
    """Send email via SMTP if configured, otherwise append to a local log file."""
    import os

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    to_addr = os.getenv("OPERATOR_EMAIL")

    if not all([host, user, password, to_addr]):
        log_file = Path(settings.data_dir) / "alerts.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"[ALERT][{device_id}] {subject}\n{body}\n---\n")
        logger.warning(
            f"SMTP not configured -> alert appended to {log_file} instead"
        )
        return ToolResult(
            name="send_operator_alert",
            content=json.dumps(
                {
                    "status": "logged_locally",
                    "path": str(log_file),
                    "reason": "SMTP env vars missing",
                }
            ),
        )

    msg = MIMEText(body)
    msg["Subject"] = f"[MDK][{device_id}] {subject}"
    msg["From"] = user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        logger.info(f"Alert email sent to {to_addr} for {device_id}")
        return ToolResult(
            name="send_operator_alert",
            content=json.dumps({"status": "sent", "to": to_addr}),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"SMTP send failed: {exc}")
        return ToolResult(
            name="send_operator_alert",
            content=json.dumps({"status": "smtp_error", "error": str(exc)}),
        )


TOOL_REGISTRY = {
    "get_fleet_summary": lambda **_: get_fleet_summary(),
    "get_device_status": lambda **kw: get_device_status(**kw),
    "list_miners_at_risk": lambda **kw: list_miners_at_risk(**kw),
    "recommend_action": lambda **kw: recommend_action(**kw),
    "send_operator_alert": lambda **kw: send_operator_alert(**kw),
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Run the tool requested by the LLM and return its JSON payload."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return ToolResult(name=name, content=json.dumps({"error": f"unknown tool {name}"}))
    try:
        return fn(**arguments)
    except TypeError as exc:
        return ToolResult(name=name, content=json.dumps({"error": f"bad args: {exc}"}))
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Tool {name} crashed")
        return ToolResult(name=name, content=json.dumps({"error": str(exc)}))
