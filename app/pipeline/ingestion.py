"""
src/pipeline/ingestion.py — Data ingestion and validation pipeline.
"""

from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator
from sqlalchemy import create_engine, text

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
    """Ingests Parquet data into PostgreSQL with validation."""

    def __init__(self, database_url: str = settings.DATABASE_URL):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        create_telemetry_sql = text("""
            CREATE TABLE IF NOT EXISTS telemetry (
                timestamp TIMESTAMP,
                device_id VARCHAR,
                model VARCHAR,
                operating_mode VARCHAR,
                asic_clock_freq_mhz FLOAT,
                asic_voltage_mv FLOAT,
                asic_power_w FLOAT,
                chip_temperature_c FLOAT,
                fan_speed_rpm FLOAT,
                asic_hashrate_th FLOAT,
                error_count SMALLINT,
                ambient_temperature_c FLOAT,
                energy_price_kwh FLOAT,
                hashprice_ph_day FLOAT,
                is_healthy BOOLEAN,
                failure_type VARCHAR,
                is_pre_failure BOOLEAN,
                is_valid BOOLEAN DEFAULT TRUE
            )
        """)
        with self.engine.connect() as conn:
            conn.execute(create_telemetry_sql)
            conn.commit()
        logger.info("PostgreSQL telemetry table initialized")

    def ingest_parquet(self, parquet_path: Path) -> int:
        """Ingest a Parquet file into PostgreSQL."""
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df["is_valid"] = True

        with self.engine.connect() as conn:
            df.to_sql("telemetry", conn, if_exists="append", index=False)
            conn.commit()

        count = self.query("SELECT COUNT(*) FROM telemetry").iloc[0, 0]
        logger.info(f"Ingested {count:,} rows from {parquet_path}")
        return count

    def validate_bounds(self) -> dict:
        """Flag records with out-of-bounds values."""
        update_sql = text(f"""
            UPDATE telemetry
            SET is_valid = FALSE
            WHERE chip_temperature_c > :temp_emergency
               OR chip_temperature_c < -10
               OR asic_power_w <= 0
               OR asic_hashrate_th < 0
        """)
        with self.engine.connect() as conn:
            conn.execute(update_sql, {"temp_emergency": settings.TEMP_EMERGENCY})
            conn.commit()

        invalid_count = self.query(
            "SELECT COUNT(*) FROM telemetry WHERE is_valid = FALSE"
        ).iloc[0, 0]

        total_count = self.query("SELECT COUNT(*) FROM telemetry").iloc[0, 0]

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
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()


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
