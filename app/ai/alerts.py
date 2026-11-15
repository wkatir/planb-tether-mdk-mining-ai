"""Operator alert delivery: SMTP when configured, local log file otherwise."""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import settings


@dataclass
class AlertConfig:
    host: str | None = None
    port: int = 587
    user: str | None = None
    password: str | None = None
    to_addr: str | None = None
    log_file: Path = Path(settings.data_dir) / "alerts.log"

    @classmethod
    def from_env(cls) -> AlertConfig:
        return cls(
            host=os.getenv("SMTP_HOST"),
            port=int(os.getenv("SMTP_PORT", "587")),
            user=os.getenv("SMTP_USER"),
            password=os.getenv("SMTP_PASS"),
            to_addr=os.getenv("OPERATOR_EMAIL"),
        )

    @property
    def smtp_ready(self) -> bool:
        return all([self.host, self.user, self.password, self.to_addr])


class AlertSender:
    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig.from_env()

    def send(self, device_id: str, subject: str, body: str) -> dict[str, Any]:
        """Send email via SMTP if configured, otherwise append to a local log file."""
        cfg = self.config
        if not cfg.smtp_ready:
            cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            with cfg.log_file.open("a", encoding="utf-8") as fh:
                fh.write(f"[ALERT][{device_id}] {subject}\n{body}\n---\n")
            logger.warning(f"SMTP not configured -> alert appended to {cfg.log_file} instead")
            return {
                "status": "logged_locally",
                "path": str(cfg.log_file),
                "reason": "SMTP env vars missing",
            }

        msg = MIMEText(body)
        msg["Subject"] = f"[MDK][{device_id}] {subject}"
        msg["From"] = cfg.user
        msg["To"] = cfg.to_addr

        try:
            with smtplib.SMTP(cfg.host, cfg.port) as smtp:
                smtp.starttls()
                smtp.login(cfg.user, cfg.password)
                smtp.send_message(msg)
        except (smtplib.SMTPException, OSError) as exc:
            logger.error(f"SMTP send failed: {exc}")
            return {"status": "smtp_error", "error": str(exc)}

        logger.info(f"Alert email sent to {cfg.to_addr} for {device_id}")
        return {"status": "sent", "to": cfg.to_addr}
