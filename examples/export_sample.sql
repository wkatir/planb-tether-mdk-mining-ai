-- Regenerates examples/kpi_sample.csv from a pipeline run that contains failures:
--
--   FAILURE_INJECTION_RATE=0.4 python -m app.run_all --fleet-size 8 --days 30 --skip-training
--   duckdb data/mining.duckdb < examples/export_sample.sql
--
-- 48 hours at one reading every 15 minutes, ending while miner_007 (thermal
-- failure) is at its peak and miner_000 (hashboard failure) is degrading.
-- Only the columns the agent layer reads are exported.
COPY (
    SELECT timestamp, device_id, model, operating_mode,
           round(asic_hashrate_th, 2)     AS asic_hashrate_th,
           round(asic_power_w, 1)         AS asic_power_w,
           round(chip_temperature_c, 2)   AS chip_temperature_c,
           round(fan_speed_rpm)::INTEGER  AS fan_speed_rpm,
           error_count,
           round(true_efficiency_jth, 3)  AS true_efficiency_jth,
           round(economic_te, 4)          AS economic_te,
           round(profit_density, 7)       AS profit_density
    FROM kpi
    WHERE timestamp BETWEEN TIMESTAMP '2026-04-22 13:00:00' AND TIMESTAMP '2026-04-24 13:00:00'
      AND minute(timestamp) % 15 = 0
    ORDER BY device_id, timestamp
) TO 'examples/kpi_sample.csv' (HEADER);
