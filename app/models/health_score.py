"""
app/models/health_score.py — Composite health score combining all detectors.

Formula: score = 1.0 - (0.4 * anomaly_score + 0.6 * max_failure_prob)
The 0.4/0.6 weighting prioritizes failure prediction (actionable) over
anomaly detection (exploratory).
"""

from dataclasses import dataclass

from loguru import logger

from app.models.anomaly_detector import AnomalyDetector, AnomalyResult
from app.models.failure_classifier import FailureClassifier, FailurePrediction
from app.models.isolation_forest import IsolationForestDetector


@dataclass
class HealthStatus:
    score: float
    status: str
    is_anomaly: bool
    is_if_anomaly: bool
    predicted_failure: str | None
    anomaly_score: float
    max_failure_prob: float


class HealthScore:
    def __init__(self) -> None:
        self.anomaly_detector: AnomalyDetector | None = None
        self.failure_classifier: FailureClassifier | None = None
        self.isolation_forest: IsolationForestDetector | None = None

    def compute_health_score(
        self,
        anomaly_result: AnomalyResult,
        failure_result: FailurePrediction,
    ) -> float:
        max_failure_prob = max(failure_result.probabilities.values())
        raw_score = 1.0 - (0.4 * anomaly_result.score + 0.6 * max_failure_prob)
        return float(max(0.0, min(1.0, raw_score)))

    def get_health_status(
        self,
        anomaly_result: AnomalyResult,
        failure_result: FailurePrediction,
        if_anomaly: bool = False,
    ) -> HealthStatus:
        health = self.compute_health_score(anomaly_result, failure_result)
        max_failure_prob = max(failure_result.probabilities.values())

        if health < 0.3:
            status = "critical"
        elif health < 0.6:
            status = "warning"
        elif health < 0.8:
            status = "caution"
        else:
            status = "healthy"

        predicted_failure = None
        if failure_result.class_id != 0:
            predicted_failure = failure_result.class_name

        return HealthStatus(
            score=health,
            status=status,
            is_anomaly=anomaly_result.is_anomaly,
            is_if_anomaly=if_anomaly,
            predicted_failure=predicted_failure,
            anomaly_score=anomaly_result.score,
            max_failure_prob=max_failure_prob,
        )

    def evaluate(self, data: dict) -> HealthStatus:
        if self.anomaly_detector is None or not self.anomaly_detector.is_trained:
            logger.warning("Anomaly detector not loaded")
            return HealthStatus(
                score=0.0,
                status="unknown",
                is_anomaly=False,
                is_if_anomaly=False,
                predicted_failure=None,
                anomaly_score=0.0,
                max_failure_prob=0.0,
            )

        import numpy as np

        features = np.array(
            [
                data.get("temp", 0),
                data.get("power", 0),
                data.get("hashrate", 0),
                data.get("voltage", 0),
                data.get("fan", 0),
                data.get("errors", 0),
                data.get("ambient", 0),
            ]
        )

        anomaly_result = self.anomaly_detector.detect(features)

        if_anomaly = False
        if self.isolation_forest and self.isolation_forest.is_trained:
            if_result = self.isolation_forest.detect(features)
            if_anomaly = if_result.is_anomaly

        failure_result = FailurePrediction(
            class_id=0,
            class_name="normal",
            confidence=0.0,
            probabilities={0: 1.0},
        )
        if self.failure_classifier and self.failure_classifier.is_trained:
            failure_result = self.failure_classifier.predict(features)

        return self.get_health_status(anomaly_result, failure_result, if_anomaly)
