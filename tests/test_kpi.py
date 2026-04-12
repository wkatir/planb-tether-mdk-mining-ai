import duckdb
import pytest
from pathlib import Path


class TestKPIEngine:
    def test_kpi_engine_instantiation(self, tmp_path):
        from app.pipeline.kpi import KPIEngine

        duckdb_path = tmp_path / "test_kpi.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS telemetry AS SELECT * FROM (VALUES (1)) t WHERE 1=0"
        )
        conn.close()

        kpi = KPIEngine(duckdb_path=duckdb_path)
        assert kpi is not None
        kpi.close()

    def test_te_calculation_formula(self, tmp_path):
        from app.pipeline.kpi import KPIEngine

        duckdb_path = tmp_path / "test_te.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                P DOUBLE,
                P_cooling DOUBLE,
                P_aux DOUBLE,
                H DOUBLE,
                η_env DOUBLE,
                η_mode DOUBLE,
                hash_price DOUBLE,
                energy_price DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO telemetry VALUES (
                'miner_001',
                '2024-01-01 00:00:00',
                3500.0,
                500.0,
                100.0,
                50.0,
                0.9,
                0.95,
                0.05,
                0.08
            )
        """)
        conn.close()

        kpi = KPIEngine(duckdb_path=duckdb_path)
        df = kpi.compute_te()
        kpi.close()

        expected_te = (3500.0 + 500.0 + 100.0) / (50.0 * 0.9 * 0.95)
        assert abs(df["TE"].iloc[0] - expected_te) < 0.01, (
            f"TE calculation incorrect: expected {expected_te}, got {df['TE'].iloc[0]}"
        )

    def test_ete_interpretation_profitable(self, tmp_path):
        from app.pipeline.kpi import KPIEngine

        duckdb_path = tmp_path / "test_ete_prof.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                P DOUBLE,
                P_cooling DOUBLE,
                P_aux DOUBLE,
                H DOUBLE,
                η_env DOUBLE,
                η_mode DOUBLE,
                hash_price DOUBLE,
                energy_price DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO telemetry VALUES (
                'miner_profitable',
                '2024-01-01 00:00:00',
                3500.0,
                500.0,
                100.0,
                50.0,
                0.9,
                0.95,
                50.0,
                0.01
            )
        """)
        conn.close()

        kpi = KPIEngine(duckdb_path=duckdb_path)
        df = kpi.compute_te()
        kpi.close()

        assert df["ETE"].iloc[0] < 1.0, (
            f"High hash_price/low energy_price should give ETE < 1, got {df['ETE'].iloc[0]}"
        )

    def test_ete_interpretation_unprofitable(self, tmp_path):
        from app.pipeline.kpi import KPIEngine

        duckdb_path = tmp_path / "test_ete_unprof.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                P DOUBLE,
                P_cooling DOUBLE,
                P_aux DOUBLE,
                H DOUBLE,
                η_env DOUBLE,
                η_mode DOUBLE,
                hash_price DOUBLE,
                energy_price DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO telemetry VALUES (
                'miner_unprofitable',
                '2024-01-01 00:00:00',
                3500.0,
                500.0,
                100.0,
                50.0,
                0.9,
                0.95,
                0.02,
                0.50
            )
        """)
        conn.close()

        kpi = KPIEngine(duckdb_path=duckdb_path)
        df = kpi.compute_te()
        kpi.close()

        assert df["ETE"].iloc[0] > 1.0, (
            f"Low hash_price/high energy_price should give ETE > 1, got {df['ETE'].iloc[0]}"
        )

    def test_device_kpi_summary(self, tmp_path):
        from app.pipeline.kpi import KPIEngine

        duckdb_path = tmp_path / "test_summary.duckdb"
        conn = duckdb.connect(str(duckdb_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                device_id VARCHAR,
                timestamp TIMESTAMP,
                P DOUBLE,
                P_cooling DOUBLE,
                P_aux DOUBLE,
                H DOUBLE,
                η_env DOUBLE,
                η_mode DOUBLE,
                hash_price DOUBLE,
                energy_price DOUBLE,
                hash_rate DOUBLE,
                T DOUBLE,
                failure_probability DOUBLE,
                uptime_hours DOUBLE,
                fan_speed DOUBLE,
                power_mode VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO telemetry VALUES
                ('miner_001', '2024-01-01 00:00:00', 3500.0, 500.0, 100.0, 50.0, 0.9, 0.95, 0.05, 0.08, 200.0, 85.0, 0.01, 100.0, 60.0, 'balanced'),
                ('miner_001', '2024-01-01 00:01:00', 3500.0, 500.0, 100.0, 50.0, 0.9, 0.95, 0.05, 0.08, 200.0, 85.0, 0.01, 101.0, 60.0, 'balanced')
        """)
        conn.close()

        kpi = KPIEngine(duckdb_path=duckdb_path)
        df = kpi.get_device_kpi_summary(device_id="miner_001")
        kpi.close()

        assert len(df) == 1, "Should return 1 device summary"
        assert df["record_count"].iloc[0] == 2, "Should have 2 records"
        assert df["avg_hash_rate"].iloc[0] == 200.0, "Average hash rate should be 200"
