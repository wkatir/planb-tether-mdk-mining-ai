"""
app/pipeline/features.py — Feature engineering for ML models.
"""

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from app.config import settings


class FeatureEngineering:
    """Compute ML features from raw telemetry stored in DuckDB."""

    def __init__(self, duckdb_path: Path = settings.DUCKDB_PATH):
        self.duckdb_path = Path(duckdb_path)
        self.conn = duckdb.connect(str(self.duckdb_path))

    def compute_rolling_features(self) -> None:
        """Add rolling window features using DuckDB window functions."""
        logger.info("Computing rolling features...")

        self.conn.execute("DROP TABLE IF EXISTS features")
        self.conn.execute("""
            CREATE TABLE features AS
            SELECT
                t.*,

                AVG(chip_temperature_c) OVER w5 AS temp_mean_5m,
                AVG(asic_power_w) OVER w5 AS power_mean_5m,
                AVG(asic_hashrate_th) OVER w5 AS hashrate_mean_5m,

                STDDEV_SAMP(chip_temperature_c) OVER w5 AS temp_std_5m,
                STDDEV_SAMP(asic_power_w) OVER w5 AS power_std_5m,
                STDDEV_SAMP(asic_hashrate_th) OVER w5 AS hashrate_std_5m,

                AVG(chip_temperature_c) OVER w60 AS temp_mean_1h,
                AVG(asic_power_w) OVER w60 AS power_mean_1h,
                AVG(asic_hashrate_th) OVER w60 AS hashrate_mean_1h,

                STDDEV_SAMP(chip_temperature_c) OVER w60 AS temp_std_1h,
                STDDEV_SAMP(asic_power_w) OVER w60 AS power_std_1h,
                STDDEV_SAMP(asic_voltage_mv) OVER w60 AS voltage_std_1h,

                chip_temperature_c - LAG(chip_temperature_c, 1) OVER wdev AS dtemp_dt,
                asic_power_w - LAG(asic_power_w, 1) OVER wdev AS dpower_dt,
                asic_hashrate_th - LAG(asic_hashrate_th, 1) OVER wdev AS dhashrate_dt,

                CASE
                    WHEN asic_hashrate_th > 0
                    THEN asic_power_w / asic_hashrate_th
                    ELSE NULL
                END AS actual_efficiency_jth,

                SUM(CAST(error_count AS INTEGER)) OVER w60 AS errors_1h,

                fan_speed_rpm / 6000.0 AS fan_utilization

            FROM telemetry t
            WINDOW
                wdev AS (PARTITION BY device_id ORDER BY timestamp),
                w5 AS (PARTITION BY device_id ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                w60 AS (PARTITION BY device_id ORDER BY timestamp ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        logger.info(f"Feature table created with {count:,} rows")

    def compute_cross_device_features(self) -> None:
        """Add features comparing each device to fleet average."""
        logger.info("Computing cross-device z-scores...")

        self.conn.execute("DROP TABLE IF EXISTS features_enriched")
        self.conn.execute("""
            CREATE TABLE features_enriched AS
            SELECT
                f.*,

                AVG(chip_temperature_c) OVER wtime AS fleet_temp_mean,
                STDDEV_SAMP(chip_temperature_c) OVER wtime AS fleet_temp_std,
                AVG(asic_hashrate_th) OVER wtime AS fleet_hashrate_mean,
                AVG(actual_efficiency_jth) OVER wtime AS fleet_efficiency_mean,

                CASE
                    WHEN STDDEV_SAMP(chip_temperature_c) OVER wtime > 0
                    THEN (chip_temperature_c - AVG(chip_temperature_c) OVER wtime)
                         / STDDEV_SAMP(chip_temperature_c) OVER wtime
                    ELSE 0
                END AS temp_zscore,

                CASE
                    WHEN STDDEV_SAMP(asic_hashrate_th) OVER wtime > 0
                    THEN (asic_hashrate_th - AVG(asic_hashrate_th) OVER wtime)
                         / STDDEV_SAMP(asic_hashrate_th) OVER wtime
                    ELSE 0
                END AS hashrate_zscore

            FROM features f
            WINDOW wtime AS (PARTITION BY DATE_TRUNC('minute', timestamp))
        """)

        logger.info("Cross-device features computed")

    def export_features(self, output_path: str | None = None) -> str:
        """Export feature table to Parquet for ML consumption."""
        if output_path is None:
            output_path = str(settings.processed_data_dir / "features.parquet")

        df = self.query("SELECT * FROM features_enriched")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, compression="snappy")
        logger.info(f"Features exported to {output_path}")
        return output_path

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return as DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.compute_rolling_features()
    fe.compute_cross_device_features()
    fe.export_features()
    fe.close()
    print("Feature engineering complete.")
