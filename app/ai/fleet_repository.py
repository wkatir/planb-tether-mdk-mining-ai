"""Read-only data access for the agent layer: every SQL query over the `kpi` table lives here."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from app.config import settings

Record = dict[str, Any]


class KpiUnavailableError(RuntimeError):
    """The `kpi` table does not exist yet (pipeline not run, or wrong database)."""


class FleetRepository:
    """Queries over pre-computed KPIs. Returns plain records, never raw telemetry."""

    def __init__(
        self,
        db_path: Path | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.db_path = Path(db_path or settings.DUCKDB_PATH)
        # A shared connection is only used for in-memory databases (demo / tests).
        self._shared = connection

    @classmethod
    def from_csv(cls, csv_path: Path) -> FleetRepository:
        """In-memory repository seeded from a KPI export (see examples/kpi_sample.csv)."""
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE kpi AS SELECT * FROM read_csv_auto(?)", [str(csv_path)])
        return cls(connection=conn)

    @contextmanager
    def _connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if self._shared is not None:
            yield self._shared
            return
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
        except duckdb.Error as exc:
            raise KpiUnavailableError(
                f"Cannot open {self.db_path} ({exc}). "
                "If it does not exist yet, run `python -m app.run_all` first."
            ) from exc
        try:
            yield conn
        finally:
            conn.close()

    def _records(self, sql: str, params: list[Any] | None = None) -> list[Record]:
        with self._connect() as conn:
            try:
                df = conn.execute(sql, params or []).fetchdf()
            except duckdb.CatalogException as exc:
                raise KpiUnavailableError(
                    "KPI table empty. Run `python -m app.run_all` first."
                ) from exc
        # Round-trip through pandas' JSON writer: NaN -> null, timestamps -> ISO.
        return json.loads(df.to_json(orient="records", date_format="iso"))

    def fleet_summary(self, ete_breakeven: float | None = None) -> Record:
        ete_breakeven = settings.ETE_BREAKEVEN if ete_breakeven is None else ete_breakeven
        rows = self._records(
            """
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
                COUNT(*)                                   AS total_miners,
                AVG(dev_hashrate_th)                       AS avg_hashrate_th,
                AVG(dev_temp_c)                            AS avg_temp_c,
                AVG(dev_power_w)                           AS avg_power_w,
                AVG(dev_te_jth)                            AS avg_te_jth,
                AVG(dev_ete)                               AS avg_ete,
                COUNT(CASE WHEN dev_ete <  $1 THEN 1 END)  AS profitable,
                COUNT(CASE WHEN dev_ete >= $1 THEN 1 END)  AS loss_making
            FROM per_device
            """,
            [ete_breakeven],
        )
        return rows[0]

    def device_status(self, device_id: str) -> Record | None:
        """Latest telemetry + KPIs for one miner, or None if the id is unknown."""
        rows = self._records(
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
        )
        return rows[0] if rows else None

    def miners_at_risk(
        self,
        limit: int = 10,
        temp_c: float | None = None,
        avg_errors: float | None = None,
    ) -> list[Record]:
        temp_c = settings.RISK_TEMP_C if temp_c is None else temp_c
        avg_errors = settings.RISK_AVG_ERRORS if avg_errors is None else avg_errors
        return self._records(
            """
            SELECT device_id, model,
                   AVG(chip_temperature_c)  AS avg_temp,
                   AVG(error_count)         AS avg_errors,
                   AVG(true_efficiency_jth) AS avg_te
            FROM kpi
            WHERE true_efficiency_jth IS NOT NULL
            GROUP BY device_id, model
            HAVING AVG(chip_temperature_c) > $2 OR AVG(error_count) > $3
            ORDER BY avg_errors DESC, avg_temp DESC
            LIMIT $1
            """,
            [int(limit), temp_c, avg_errors],
        )

    def unprofitable_miners(
        self, limit: int = 10, ete_breakeven: float | None = None
    ) -> list[Record]:
        """Miners whose average ETE is at or above break-even, worst first."""
        ete_breakeven = settings.ETE_BREAKEVEN if ete_breakeven is None else ete_breakeven
        return self._records(
            """
            SELECT device_id, model,
                   AVG(economic_te)         AS avg_ete,
                   AVG(profit_density)      AS avg_profit_density,
                   AVG(true_efficiency_jth) AS avg_te,
                   AVG(asic_power_w)        AS avg_power_w
            FROM kpi
            WHERE economic_te IS NOT NULL
            GROUP BY device_id, model
            HAVING AVG(economic_te) >= $2
            ORDER BY avg_ete DESC
            LIMIT $1
            """,
            [int(limit), ete_breakeven],
        )
