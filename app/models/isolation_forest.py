"""
app/models/isolation_forest.py — Isolation Forest anomaly detector.

Lightweight unsupervised anomaly detector using scikit-learn.
Complements the LSTM Autoencoder: IF catches global outliers fast,
LSTM catches subtle temporal drift. Ensemble agreement = high confidence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest as SklearnIF

MODEL_DIR: Path = Path("data/models")
MODEL_PATH: Path = MODEL_DIR / "isolation_forest.joblib"

FEATURE_NAMES: list[str] = [
    "temp",
    "power",
    "hashrate",
    "voltage",
    "fan",
    "errors",
    "ambient",
]


@dataclass
class IFAnomalyResult:
    score: float
    is_anomaly: bool
    raw_score: float


class IsolationForestDetector:
    """Isolation Forest for fast, unsupervised anomaly detection."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.model = SklearnIF(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained: bool = False
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def train(self, healthy_data: np.ndarray) -> None:
        """Train on healthy fleet data. Expects shape (n_samples, n_features)."""
        logger.info(f"Training Isolation Forest on {healthy_data.shape[0]} samples")
        if healthy_data.shape[0] == 0:
            logger.warning("No healthy data available for training")
            return

        self.scaler_mean = np.mean(healthy_data, axis=0)
        self.scaler_std = np.std(healthy_data, axis=0) + 1e-8
        normalized = (healthy_data - self.scaler_mean) / self.scaler_std

        self.model.fit(normalized)
        self.is_trained = True

        scores = self.model.decision_function(normalized)
        logger.info(
            f"Training complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]"
        )

    def detect(self, data: np.ndarray) -> IFAnomalyResult:
        """Detect anomaly for a single sample or batch."""
        if not self.is_trained:
            logger.warning("Model not trained yet, returning default")
            return IFAnomalyResult(score=0.0, is_anomaly=False, raw_score=0.0)

        X = data.reshape(1, -1) if data.ndim == 1 else data
        normalized = (X - self.scaler_mean) / self.scaler_std

        raw_score = float(self.model.decision_function(normalized)[0])
        prediction = int(self.model.predict(normalized)[0])

        anomaly_score = max(0.0, -raw_score)

        return IFAnomalyResult(
            score=anomaly_score,
            is_anomaly=prediction == -1,
            raw_score=raw_score,
        )

    def detect_batch(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly labels for a batch: 1=normal, -1=anomaly."""
        if not self.is_trained:
            return np.ones(data.shape[0])
        normalized = (data - self.scaler_mean) / self.scaler_std
        return self.model.predict(normalized)

    def save(self, path: Path | None = None) -> None:
        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_trained": self.is_trained,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
            },
            save_path,
        )
        logger.info(f"Isolation Forest saved to {save_path}")

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        load_path = path or MODEL_PATH
        if not load_path.exists():
            logger.warning(f"Model not found at {load_path}, returning untrained")
            return cls()

        data = joblib.load(load_path)
        instance = cls()
        instance.model = data["model"]
        instance.is_trained = data["is_trained"]
        instance.scaler_mean = data["scaler_mean"]
        instance.scaler_std = data["scaler_std"]
        logger.info(f"Isolation Forest loaded from {load_path}")
        return instance
