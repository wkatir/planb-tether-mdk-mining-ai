"""
app/models/train_models.py — End-to-end ML model training pipeline.

Trains all three models from DuckDB feature data:
  1. LSTM Autoencoder (anomaly detection)
  2. Isolation Forest (fast anomaly detection)
  3. XGBoost Failure Classifier

Usage:
    python -m app.models.train_models
"""

from pathlib import Path

import duckdb
import numpy as np
from loguru import logger

from app.config import settings
from app.models.anomaly_detector import AnomalyDetector
from app.models.failure_classifier import FailureClassifier, FAILURE_CLASSES
from app.models.isolation_forest import IsolationForestDetector


FEATURE_COLS = [
    "chip_temperature_c",
    "asic_power_w",
    "asic_hashrate_th",
    "asic_voltage_mv",
    "fan_speed_rpm",
    "error_count",
    "ambient_temperature_c",
]


def load_training_data(duckdb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load healthy data and labeled data from DuckDB features table."""
    conn = duckdb.connect(str(duckdb_path), read_only=True)

    cols = ", ".join(FEATURE_COLS)

    healthy_df = conn.execute(f"""
        SELECT {cols}
        FROM features_enriched
        WHERE is_healthy = TRUE AND is_valid = TRUE
        ORDER BY timestamp
    """).fetchdf()

    all_df = conn.execute(f"""
        SELECT {cols}, failure_type, is_healthy
        FROM features_enriched
        WHERE is_valid = TRUE
        ORDER BY timestamp
    """).fetchdf()

    conn.close()

    healthy_data = healthy_df.values.astype(np.float32)

    failure_map = {"normal": 0, "thermal": 1, "hashboard": 2, "psu": 3}
    labels = []
    for _, row in all_df.iterrows():
        if row["is_healthy"]:
            labels.append(0)
        else:
            ft = row["failure_type"]
            labels.append(failure_map.get(ft, 0))

    all_features = all_df[FEATURE_COLS].values.astype(np.float32)
    labels = np.array(labels, dtype=np.int32)

    logger.info(
        f"Loaded {len(healthy_data)} healthy samples, "
        f"{len(all_features)} total samples ({(labels > 0).sum()} failures)"
    )
    return healthy_data, all_features, labels


def train_all(duckdb_path: Path | None = None) -> None:
    """Train all ML models end-to-end."""
    if duckdb_path is None:
        duckdb_path = settings.DUCKDB_PATH

    if not Path(duckdb_path).exists():
        logger.error(f"DuckDB not found at {duckdb_path}. Run the pipeline first.")
        return

    settings.ensure_dirs()
    healthy_data, all_features, labels = load_training_data(duckdb_path)

    if len(healthy_data) < 100:
        logger.warning(f"Only {len(healthy_data)} healthy samples — models may underfit")

    # --- 1. LSTM Autoencoder ---
    logger.info("=" * 50)
    logger.info("Training LSTM Autoencoder...")
    anomaly_detector = AnomalyDetector()
    anomaly_detector.train(healthy_data, n_samples=min(10000, len(healthy_data)))
    anomaly_detector.save()

    # --- 2. Isolation Forest ---
    logger.info("=" * 50)
    logger.info("Training Isolation Forest...")
    iso_forest = IsolationForestDetector(contamination=0.05)
    iso_forest.train(healthy_data)
    iso_forest.save()

    # --- 3. XGBoost Failure Classifier ---
    logger.info("=" * 50)
    logger.info("Training XGBoost Failure Classifier...")

    n_features = all_features.shape[1]
    n_rates = n_features
    n_ma = n_features
    n_std = n_features

    rates = np.diff(all_features, axis=0, prepend=all_features[:1])

    from pandas import DataFrame
    df_feat = DataFrame(all_features, columns=FEATURE_COLS)
    ma5 = df_feat.rolling(5, min_periods=1).mean().values
    std5 = df_feat.rolling(5, min_periods=1).std().fillna(0).values

    X_classifier = np.hstack([all_features, rates, ma5, std5])

    classifier = FailureClassifier()
    classifier.feature_names = (
        FEATURE_COLS
        + [f"{c}_rate" for c in FEATURE_COLS]
        + [f"{c}_ma5" for c in FEATURE_COLS]
        + [f"{c}_std5" for c in FEATURE_COLS]
    )
    classifier.train(X_classifier, labels)
    classifier.save()

    # --- Summary ---
    logger.info("=" * 50)
    logger.info("All models trained and saved:")
    logger.info(f"  LSTM Autoencoder  → {anomaly_detector.threshold:.6f} threshold")
    logger.info(f"  Isolation Forest  → {iso_forest.is_trained}")
    logger.info(f"  XGBoost Classifier → {classifier.is_trained}")


if __name__ == "__main__":
    train_all()
