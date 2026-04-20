import duckdb
import pytest
from pathlib import Path


class TestFeatureEngineering:
    def test_feature_engineering_instantiation(self, tmp_path):
        from app.pipeline.features import FeatureEngineering

        duckdb_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                model VARCHAR,
                operating_mode VARCHAR,
                asic_clock_freq_mhz DOUBLE,
                asic_voltage_mv DOUBLE,
                asic_power_w DOUBLE,
                chip_temperature_c DOUBLE,
                fan_speed_rpm DOUBLE,
                asic_hashrate_th DOUBLE,
                error_count INTEGER,
                ambient_temperature_c DOUBLE,
                energy_price_kwh DOUBLE,
                hashprice_ph_day DOUBLE,
                is_healthy BOOLEAN,
                failure_type VARCHAR,
                is_pre_failure BOOLEAN,
                is_valid BOOLEAN
            )
        """)
        conn.close()

        fe = FeatureEngineering(duckdb_path=duckdb_path)
        assert fe is not None
        fe.close()

    def test_duckdb_table_creation(self, tmp_path):
        duckdb_path = tmp_path / "test_features.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                model VARCHAR,
                operating_mode VARCHAR,
                asic_clock_freq_mhz DOUBLE,
                asic_voltage_mv DOUBLE,
                asic_power_w DOUBLE,
                chip_temperature_c DOUBLE,
                fan_speed_rpm DOUBLE,
                asic_hashrate_th DOUBLE,
                error_count INTEGER,
                ambient_temperature_c DOUBLE,
                energy_price_kwh DOUBLE,
                hashprice_ph_day DOUBLE,
                is_healthy BOOLEAN,
                failure_type VARCHAR,
                is_pre_failure BOOLEAN,
                is_valid BOOLEAN
            )
        """)
        conn.execute("""
            INSERT INTO telemetry VALUES (
                'miner_001', '2024-01-01 00:00:00', 'S21', 'normal',
                500.0, 340.0, 3500.0, 65.0, 3000.0, 200.0, 0, 25.0, 0.04, 50.0,
                TRUE, NULL, FALSE, TRUE
            )
        """)
        result = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        conn.close()

        assert result == 1, "Table should contain 1 record"

    def test_compute_rolling_features(self, tmp_path):
        from app.pipeline.features import FeatureEngineering

        duckdb_path = tmp_path / "test_rolling.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                model VARCHAR,
                operating_mode VARCHAR,
                asic_clock_freq_mhz DOUBLE,
                asic_voltage_mv DOUBLE,
                asic_power_w DOUBLE,
                chip_temperature_c DOUBLE,
                fan_speed_rpm DOUBLE,
                asic_hashrate_th DOUBLE,
                error_count INTEGER,
                ambient_temperature_c DOUBLE,
                energy_price_kwh DOUBLE,
                hashprice_ph_day DOUBLE,
                is_healthy BOOLEAN,
                failure_type VARCHAR,
                is_pre_failure BOOLEAN,
                is_valid BOOLEAN
            )
        """)
        for i in range(10):
            conn.execute(f"""
                INSERT INTO telemetry VALUES (
                    'miner_001', '2024-01-01 00:{i:02d}:00', 'S21', 'normal',
                    500.0, 340.0, 3500.0, {65.0 + i}, 3000.0, {200.0 + i}, 0, 25.0, 0.04, 50.0,
                    TRUE, NULL, FALSE, TRUE
                )
            """)
        conn.close()

        fe = FeatureEngineering(duckdb_path=duckdb_path)
        fe.compute_rolling_features()
        df = fe.query("SELECT * FROM features")
        fe.close()

        assert len(df) == 10, "Should return 10 records"
        assert "temp_mean_5m" in df.columns, "Should have rolling average column"
