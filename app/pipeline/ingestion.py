"""
app/pipeline/ingestion.py — Data ingestion and validation pipeline.
"""

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator

from app.config import settings


class TelemetryRecord(BaseModel):
    """Pydantic model for validating a single telemetry record."""

    device_id: str
    timestamp: str
    asic_clock_freq_mhz: float
    asic_voltage_mv: float
    asic_power_w: float
    chip_temperature_c: float
    fan_speed_rpm: float
    asic_hashrate_th: float
    error_count: int
    ambient_temperature_c: float
    energy_price_kwh: float

    @field_validator("asic_clock_freq_mhz")
    @classmethod
    def validate_clock(cls, v: float) -> float:
        if not (50.0 <= v <= 1000.0):
            raise ValueError(
                f"Clock frequency {v} MHz outside physical bounds [50, 1000]"
            )
        return v

    @field_validator("asic_voltage_mv")
    @classmethod
    def validate_voltage(cls, v: float) -> float:
        if not (100.0 <= v <= 600.0):
            raise ValueError(f"Voltage {v} mV outside physical bounds [100, 600]")
        return v

    @field_validator("chip_temperature_c")
    @classmethod
    def validate_temp(cls, v: float) -> float:
        if not (-10.0 <= v <= 150.0):
            raise ValueError(f"Temperature {v}C outside physical bounds [-10, 150]")
        return v

    @field_validator("asic_hashrate_th")
    @classmethod
    def validate_hashrate(cls, v: float) -> float:
        if not (0.0 <= v <= 600.0):
            raise ValueError(f"Hashrate {v} TH/s outside physical bounds [0, 600]")
        return v

    @field_validator("asic_power_w")
    @classmethod
    def validate_power(cls, v: float) -> float:
        if not (0.0 <= v <= 10000.0):
            raise ValueError(f"Power {v} W outside physical bounds [0, 10000]")
        return v


class DataIngestion:
    """Ingests Parquet data into DuckDB with validation."""

    def __init__(self, duckdb_path: Path = settings.DUCKDB_PATH):
        self.duckdb_path = Path(duckdb_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.duckdb_path))
        logger.info("DuckDB connection initialized")

    def ingest_parquet(self, parquet_path: Path) -> int:
        """Ingest a Parquet file into DuckDB."""
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.conn.execute("DROP TABLE IF EXISTS telemetry")
        self.conn.execute("""
            CREATE TABLE telemetry AS
            SELECT
                timestamp,
                device_id,
                model,
                operating_mode,
                asic_clock_freq_mhz::DOUBLE AS asic_clock_freq_mhz,
                asic_voltage_mv::DOUBLE AS asic_voltage_mv,
                asic_power_w::DOUBLE AS asic_power_w,
                chip_temperature_c::DOUBLE AS chip_temperature_c,
                fan_speed_rpm::DOUBLE AS fan_speed_rpm,
                asic_hashrate_th::DOUBLE AS asic_hashrate_th,
                error_count::INTEGER AS error_count,
                ambient_temperature_c::DOUBLE AS ambient_temperature_c,
                energy_price_kwh::DOUBLE AS energy_price_kwh,
                hashprice_ph_day::DOUBLE AS hashprice_ph_day,
                is_healthy::BOOLEAN AS is_healthy,
                failure_type::VARCHAR AS failure_type,
                is_pre_failure::BOOLEAN AS is_pre_failure,
                TRUE AS is_valid
            FROM read_parquet(?)
        """, [str(parquet_path)])

        count = self.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        logger.info(f"Ingested {count:,} rows from {parquet_path}")
        return count

    def validate_bounds(self) -> dict:
        """Flag records with out-of-bounds values."""
        self.conn.execute(
            """
            UPDATE telemetry
            SET is_valid = FALSE
            WHERE chip_temperature_c > $1
               OR chip_temperature_c < -10
               OR asic_power_w <= 0
               OR asic_hashrate_th < 0
            """,
            [settings.TEMP_EMERGENCY],
        )

        invalid_count = self.conn.execute(
            "SELECT COUNT(*) FROM telemetry WHERE is_valid = FALSE"
        ).fetchone()[0]

        total_count = self.conn.execute(
            "SELECT COUNT(*) FROM telemetry"
        ).fetchone()[0]

        result = {
            "total_records": total_count,
            "valid_records": total_count - invalid_count,
            "invalid_records": invalid_count,
            "validation_rate": (total_count - invalid_count) / total_count
            if total_count > 0
            else 0,
        }
        logger.info(f"Validation complete: {result}")
        return result

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return as DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    settings.ensure_dirs()
    ingestion = DataIngestion()
    parquet_path = settings.raw_data_dir / "fleet_telemetry.parquet"
    ingestion.ingest_parquet(parquet_path)
    result = ingestion.validate_bounds()
    print(f"Validation result: {result}")
    sample = ingestion.query("SELECT * FROM telemetry LIMIT 5")
    print(sample)
    ingestion.close()
