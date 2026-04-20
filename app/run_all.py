"""
app/run_all.py — Run the complete MDK Mining AI pipeline end-to-end.

Usage:
    python -m app.run_all                    # Full pipeline (50 miners, 30 days)
    python -m app.run_all --fleet-size 5 --days 1   # Quick demo
"""

import argparse
import time

from loguru import logger

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="MDK Mining AI — Full Pipeline")
    parser.add_argument("--fleet-size", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true", help="Skip ML model training")
    args = parser.parse_args()

    fleet_size = args.fleet_size or settings.fleet_size
    days = args.days or settings.simulation_days

    logger.info("=" * 60)
    logger.info("MDK Mining AI — Full Pipeline")
    logger.info(f"Fleet: {fleet_size} miners | Duration: {days} days")
    logger.info("=" * 60)

    t0 = time.time()

    # --- Step 1: Generate synthetic telemetry ---
    logger.info("\n[1/6] Generating synthetic telemetry...")
    settings.ensure_dirs()
    from app.data.generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(fleet_size=fleet_size, days=days)
    output_path = gen.generate()
    logger.info(f"  → {output_path}")

    # --- Step 2: Ingest into DuckDB ---
    logger.info("\n[2/6] Ingesting into DuckDB...")
    from app.pipeline.ingestion import DataIngestion

    ingestion = DataIngestion()
    count = ingestion.ingest_parquet(settings.raw_data_dir / "fleet_telemetry.parquet")
    validation = ingestion.validate_bounds()
    ingestion.close()
    logger.info(f"  → {count:,} rows, {validation['validation_rate']:.1%} valid")

    # --- Step 3: Feature engineering ---
    logger.info("\n[3/6] Computing features...")
    from app.pipeline.features import FeatureEngineering

    fe = FeatureEngineering()
    fe.compute_rolling_features()
    fe.compute_cross_device_features()
    fe.export_features()
    fe.close()

    # --- Step 4: KPI computation ---
    logger.info("\n[4/6] Computing KPIs (TE, ETE, PD)...")
    from app.pipeline.kpi import KPIEngine

    kpi = KPIEngine()
    kpi.compute_te()
    summary = kpi.get_device_kpi_summary()
    kpi.close()
    logger.info(f"  → {len(summary)} devices with KPIs")

    # --- Step 5: Train ML models ---
    if not args.skip_training:
        logger.info("\n[5/6] Training ML models...")
        from app.models.train_models import train_all

        train_all()
    else:
        logger.info("\n[5/6] Skipping ML training (--skip-training)")

    # --- Step 6: Summary ---
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"DuckDB: {settings.DUCKDB_PATH}")
    logger.info(f"Features: {settings.processed_data_dir / 'features.parquet'}")
    logger.info(f"Models: {settings.models_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  streamlit run app/dashboard/dashboard.py")
    logger.info("  python -m app.rl.train_agent")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
