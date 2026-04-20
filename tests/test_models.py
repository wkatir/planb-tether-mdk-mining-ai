"""Tests for ML models — Isolation Forest and Health Score."""

import numpy as np
import pytest


class TestIsolationForest:
    def test_instantiation(self):
        from app.models.isolation_forest import IsolationForestDetector

        detector = IsolationForestDetector()
        assert not detector.is_trained

    def test_train_and_detect(self):
        from app.models.isolation_forest import IsolationForestDetector

        rng = np.random.default_rng(42)
        healthy = rng.normal(loc=[65, 3500, 200, 340, 3000, 0, 25], scale=[5, 100, 10, 10, 200, 1, 3], size=(500, 7))

        detector = IsolationForestDetector(contamination=0.05)
        detector.train(healthy.astype(np.float32))
        assert detector.is_trained

        normal_sample = np.array([65, 3500, 200, 340, 3000, 0, 25], dtype=np.float32)
        result = detector.detect(normal_sample)
        assert not result.is_anomaly, "Normal sample should not be flagged"

        extreme = np.array([150, 9000, 0, 600, 6000, 100, 60], dtype=np.float32)
        result = detector.detect(extreme)
        assert result.is_anomaly, "Extreme sample should be flagged"

    def test_batch_detect(self):
        from app.models.isolation_forest import IsolationForestDetector

        rng = np.random.default_rng(42)
        healthy = rng.normal(loc=[65, 3500, 200, 340, 3000, 0, 25], scale=[5, 100, 10, 10, 200, 1, 3], size=(500, 7))

        detector = IsolationForestDetector(contamination=0.05)
        detector.train(healthy.astype(np.float32))

        batch = healthy[:10].astype(np.float32)
        labels = detector.detect_batch(batch)
        assert len(labels) == 10
        assert (labels == 1).sum() >= 8, "Most healthy samples should be normal"

    def test_save_load(self, tmp_path):
        from app.models.isolation_forest import IsolationForestDetector

        rng = np.random.default_rng(42)
        healthy = rng.normal(size=(200, 7)).astype(np.float32)

        detector = IsolationForestDetector()
        detector.train(healthy)

        save_path = tmp_path / "test_if.joblib"
        detector.save(save_path)

        loaded = IsolationForestDetector.load(save_path)
        assert loaded.is_trained

        sample = healthy[0]
        r1 = detector.detect(sample)
        r2 = loaded.detect(sample)
        assert abs(r1.raw_score - r2.raw_score) < 1e-6


class TestHealthScore:
    def test_weighted_formula(self):
        from app.models.anomaly_detector import AnomalyResult
        from app.models.failure_classifier import FailurePrediction
        from app.models.health_score import HealthScore

        hs = HealthScore()

        anomaly = AnomalyResult(score=0.5, is_anomaly=True, threshold=0.3)
        failure = FailurePrediction(
            class_id=1, class_name="thermal", confidence=0.8,
            probabilities={0: 0.2, 1: 0.8}
        )

        score = hs.compute_health_score(anomaly, failure)
        expected = 1.0 - (0.4 * 0.5 + 0.6 * 0.8)
        assert abs(score - max(0.0, expected)) < 1e-6

    def test_healthy_device_high_score(self):
        from app.models.anomaly_detector import AnomalyResult
        from app.models.failure_classifier import FailurePrediction
        from app.models.health_score import HealthScore

        hs = HealthScore()

        anomaly = AnomalyResult(score=0.02, is_anomaly=False, threshold=0.3)
        failure = FailurePrediction(
            class_id=0, class_name="normal", confidence=0.95,
            probabilities={0: 0.95, 1: 0.03, 2: 0.01, 3: 0.01}
        )

        # max_failure_prob = 0.95 (class 0, "normal")
        # score = 1.0 - (0.4*0.02 + 0.6*0.95) = 1.0 - 0.578 = 0.422
        status = hs.get_health_status(anomaly, failure)
        assert status.score > 0.3
        assert status.predicted_failure is None, "Normal class should not predict failure"
