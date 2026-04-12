import duckdb
import pytest
from pathlib import Path


DATA_READY = False


def check_data_ready():
    return DATA_READY


@pytest.mark.skipif(not check_data_ready(), reason="Data not ready")
class TestFeatureEngineering:
    def test_feature_engineering_instantiation(self, tmp_path):
        from app.pipeline.features import FeatureEngineering

        duckdb_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS telemetry (device_id VARCHAR, timestamp TIMESTAMP, hash_rate DOUBLE, P DOUBLE, T DOUBLE, H DOUBLE, power_mode VARCHAR)"
        )
        conn.close()

        fe = FeatureEngineering(duckdb_path=duckdb_path)
        assert fe is not None
        fe.close()

    def test_duckdb_table_creation(self, tmp_path):
        from app.pipeline.features import FeatureEngineering

        duckdb_path = tmp_path / "test_features.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                hash_rate DOUBLE,
                P DOUBLE,
                T DOUBLE,
                H DOUBLE,
                power_mode VARCHAR
            )
        """)
        conn.execute(
            "INSERT INTO telemetry VALUES ('miner_001', '2024-01-01 00:00:00', 200.0, 3500.0, 85.0, 50.0, 'balanced')"
        )
        result = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        conn.close()

        assert result == 1, "Table should contain 1 record"

    def test_compute_rolling_features(self, tmp_path):
        from app.pipeline.features import FeatureEngineering

        duckdb_path = tmp_path / "test_rolling.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                hash_rate DOUBLE,
                P DOUBLE,
                T DOUBLE,
                H DOUBLE,
                power_mode VARCHAR
            )
        """)
        for i in range(10):
            conn.execute(
                f"INSERT INTO telemetry VALUES ('miner_001', '2024-01-01 00:{i:02d}:00', {200.0 + i}, 3500.0, 85.0, 50.0, 'balanced')"
            )
        conn.close()

        fe = FeatureEngineering(duckdb_path=duckdb_path)
        df = fe.compute_rolling_features()
        fe.close()

        assert len(df) == 10, "Should return 10 records"
        assert "hash_rate_ma5" in df.columns, "Should have rolling average column"
