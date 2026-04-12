from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import xgboost as xgb
from loguru import logger
from shap import TreeExplainer

FAILURE_CLASSES: dict[int, str] = {0: "normal", 1: "thermal", 2: "hashboard", 3: "psu"}
MODEL_DIR: Path = Path("data/models")
MODEL_PATH: Path = MODEL_DIR / "xgb_classifier.json"


@dataclass
class FailurePrediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[int, float]


class FailureClassifier:
    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.explainer: TreeExplainer | None = None
        self.is_trained: bool = False
        self.feature_names: list[str] = [
            "temp",
            "power",
            "hashrate",
            "voltage",
            "fan",
            "errors",
            "ambient",
            "temp_rate",
            "power_rate",
            "hashrate_rate",
            "voltage_rate",
            "fan_rate",
            "error_rate",
            "ambient_rate",
            "temp_ma5",
            "power_ma5",
            "hashrate_ma5",
            "voltage_ma5",
            "fan_ma5",
            "error_ma5",
            "ambient_ma5",
            "temp_std5",
            "power_std5",
            "hashrate_std5",
            "voltage_std5",
            "fan_std5",
            "error_std5",
            "ambient_std5",
        ]

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Training XGBoost Failure Classifier")
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.warning("Empty training data provided")
            return

        n_classes = len(np.unique(y))
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.explainer = TreeExplainer(self.model)
        self.is_trained = True
        logger.info(f"Training complete. Classes: {n_classes}")

    def predict(self, X: np.ndarray) -> FailurePrediction:
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, returning default prediction")
            return FailurePrediction(
                class_id=0, class_name="normal", confidence=0.0, probabilities={0: 1.0}
            )

        proba = self.model.predict_proba(X)
        probs = proba[0] if proba.ndim > 1 else proba
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        prob_dict = {i: float(probs[i]) for i in range(len(probs))}

        return FailurePrediction(
            class_id=class_id,
            class_name=FAILURE_CLASSES.get(class_id, "unknown"),
            confidence=confidence,
            probabilities=prob_dict,
        )

    def explain(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.explainer is None:
            logger.warning("Model not trained, cannot compute explanations")
            return np.zeros(len(self.feature_names))

        return self.explainer.shap_values(X)

    def save(self, path: Path | None = None) -> None:
        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_model(str(save_path))
            logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        load_path = path or MODEL_PATH
        if not load_path.exists():
            logger.warning(
                f"Model file not found at {load_path}, returning untrained model"
            )
            return cls()

        instance = cls()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(load_path))
        instance.explainer = TreeExplainer(instance.model)
        instance.is_trained = True
        logger.info(f"Model loaded from {load_path}")
        return instance
