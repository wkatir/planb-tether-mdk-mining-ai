"""
app/pipeline/kpi.py — True Efficiency (TE) and Economic True Efficiency (ETE) KPI engine.
"""

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from app.config import settings


class KPIEngine:
    """Compute True Efficiency and Economic True Efficiency KPIs."""

    def __init__(self, duckdb_path: Path = settings.DUCKDB_PATH):
        self.duckdb_path = Path(duckdb_path)
        self.conn = duckdb.connect(str(self.duckdb_path))

    def compute_te(self, cooling_power_kw: float = 50.0) -> None:
        """Compute True Efficiency for all records."""
        logger.info("Computing True Efficiency (TE)...")

        cooling_w = cooling_power_kw * 1000.0
        eta_env_min = settings.TE_ETA_ENV_MIN
        te_beta = settings.TE_BETA
        temp_reference = settings.TEMP_REFERENCE
        aux_power_pct = settings.AUX_POWER_PCT

        self.conn.execute("DROP TABLE IF EXISTS kpi")
        self.conn.execute(f"""
            CREATE TABLE kpi AS
            SELECT
                fe.*,

                GREATEST(
                    {eta_env_min},
                    1.0 - {te_beta} * (ambient_temperature_c - {temp_reference})
                ) AS eta_env,

                CASE operating_mode
                    WHEN 'normal'    THEN 1.00
                    WHEN 'low_power' THEN 1.10
                    WHEN 'overclock' THEN 0.85
                    ELSE 1.00
                END AS eta_mode,

                {cooling_w} * (asic_power_w / SUM(asic_power_w) OVER wtime)
                    AS p_cooling_alloc,

                asic_power_w * {aux_power_pct} AS p_aux

            FROM features_enriched fe
            WINDOW wtime AS (PARTITION BY DATE_TRUNC('minute', timestamp))
        """)

        self.conn.execute("""
            ALTER TABLE kpi ADD COLUMN true_efficiency_jth DOUBLE;
            ALTER TABLE kpi ADD COLUMN economic_te DOUBLE;
            ALTER TABLE kpi ADD COLUMN profit_density DOUBLE;
        """)

        self.conn.execute("""
            UPDATE kpi SET
                true_efficiency_jth = CASE
                    WHEN asic_hashrate_th > 0 AND eta_env > 0 AND eta_mode > 0
                    THEN (asic_power_w + p_cooling_alloc + p_aux)
                         / (asic_hashrate_th * eta_env * eta_mode)
                    ELSE NULL
                END,
                economic_te = CASE
                    WHEN asic_hashrate_th > 0 AND hashprice_ph_day > 0
                         AND eta_env > 0 AND eta_mode > 0
                    THEN (
                        0.024 * (asic_power_w + p_cooling_alloc + p_aux)
                              / (asic_hashrate_th * eta_env * eta_mode)
                              * energy_price_kwh
                    ) / (
                        hashprice_ph_day / 1000.0
                    )
                    ELSE NULL
                END,
                profit_density = CASE
                    WHEN asic_power_w > 0 AND hashprice_ph_day > 0
                         AND eta_env > 0 AND eta_mode > 0
                    THEN (
                        (hashprice_ph_day / 1000.0 * asic_hashrate_th)
                        - (0.024 * (asic_power_w + p_cooling_alloc + p_aux)
                           / (eta_env * eta_mode) * energy_price_kwh)
                    ) / (asic_power_w + p_cooling_alloc + p_aux)
                    ELSE NULL
                END
        """)

        stats = self.query("""
            SELECT
                AVG(true_efficiency_jth) as avg_te,
                MIN(true_efficiency_jth) as min_te,
                MAX(true_efficiency_jth) as max_te,
                AVG(economic_te) as avg_ete,
                AVG(profit_density) as avg_pd
            FROM kpi
            WHERE true_efficiency_jth IS NOT NULL
        """)

        logger.info(
            f"KPI Stats — TE avg: {stats.iloc[0]['avg_te']:.1f} J/TH, "
            f"ETE avg: {stats.iloc[0]['avg_ete']:.4f}, "
            f"PD avg: {stats.iloc[0]['avg_pd']:.6f} $/W/day"
        )

    def get_device_kpi_summary(self) -> pd.DataFrame:
        """Get KPI summary per device."""
        return self.query("""
            SELECT
                device_id,
                model,
                operating_mode,
                AVG(true_efficiency_jth) as avg_te,
                AVG(economic_te) as avg_ete,
                AVG(asic_hashrate_th) as avg_hashrate,
                AVG(chip_temperature_c) as avg_temp,
                SUM(error_count) as total_errors,
                COUNT(*) as sample_count
            FROM kpi
            WHERE true_efficiency_jth IS NOT NULL
            GROUP BY device_id, model, operating_mode
            ORDER BY avg_te
        """)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return as DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    kpi = KPIEngine()
    kpi.compute_te()
    summary = kpi.get_device_kpi_summary()
    print(summary.head(10))
    kpi.close()
