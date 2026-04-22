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
        self._encoded_to_canonical: dict[int, int] = {}
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

        canonical_classes = sorted(int(c) for c in np.unique(y))
        self._encoded_to_canonical = {
            enc: canonical for enc, canonical in enumerate(canonical_classes)
        }
        canonical_to_encoded = {c: e for e, c in self._encoded_to_canonical.items()}
        y_encoded = np.array([canonical_to_encoded[int(v)] for v in y], dtype=np.int32)

        n_classes = len(canonical_classes)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=-1,
        )
        self.model.fit(X, y_encoded)
        self.explainer = TreeExplainer(self.model)
        self.is_trained = True
        logger.info(
            f"Training complete. Canonical classes seen: {canonical_classes} "
            f"(encoded as {list(range(n_classes))})"
        )

    def predict(self, X: np.ndarray) -> FailurePrediction:
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, returning default prediction")
            return FailurePrediction(
                class_id=0, class_name="normal", confidence=0.0, probabilities={0: 1.0}
            )

        X = X.reshape(1, -1) if X.ndim == 1 else X
        proba = self.model.predict_proba(X)
        probs = proba[0] if proba.ndim > 1 else proba
        encoded_class = int(np.argmax(probs))
        confidence = float(probs[encoded_class])

        canonical_id = self._encoded_to_canonical.get(encoded_class, encoded_class)
        prob_dict = {
            self._encoded_to_canonical.get(i, i): float(probs[i])
            for i in range(len(probs))
        }

        return FailurePrediction(
            class_id=canonical_id,
            class_name=FAILURE_CLASSES.get(canonical_id, "unknown"),
            confidence=confidence,
            probabilities=prob_dict,
        )

    def explain(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.explainer is None:
            logger.warning("Model not trained, cannot compute explanations")
            return np.zeros(len(self.feature_names))

        return self.explainer.shap_values(X)

    def save(self, path: Path | None = None) -> None:
        import json as _json

        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_model(str(save_path))
            meta_path = save_path.with_suffix(".meta.json")
            with meta_path.open("w", encoding="utf-8") as fh:
                _json.dump(
                    {
                        "encoded_to_canonical": {
                            str(k): v for k, v in self._encoded_to_canonical.items()
                        },
                        "feature_names": self.feature_names,
                    },
                    fh,
                )
            logger.info(f"Model saved to {save_path} (+ {meta_path.name})")

    def export_onnx(self, path: Path | None = None) -> str:
        """Export to ONNX for MOS edge deployment."""
        export_path = path or (MODEL_DIR / "classifier.onnx")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            logger.warning("No trained model to export")
            return ""
        try:
            import onnxmltools
            from skl2onnx.common.data_types import FloatTensorType

            initial_type = [
                ("features", FloatTensorType([None, len(self.feature_names)]))
            ]
            onnx_model = onnxmltools.convert_xgboost(
                self.model.get_booster(), initial_types=initial_type
            )
            onnxmltools.utils.save_model(onnx_model, str(export_path))
            logger.info(f"ONNX model exported: {export_path}")
            return str(export_path)
        except ImportError:
            logger.warning("Install onnxmltools and skl2onnx for ONNX export")
            return ""

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        load_path = path or MODEL_PATH
        if not load_path.exists():
            logger.warning(
                f"Model file not found at {load_path}, returning untrained model"
            )
            return cls()

        import json as _json

        instance = cls()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(load_path))
        instance.explainer = TreeExplainer(instance.model)
        instance.is_trained = True

        meta_path = load_path.with_suffix(".meta.json")
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as fh:
                meta = _json.load(fh)
            instance._encoded_to_canonical = {
                int(k): int(v) for k, v in meta.get("encoded_to_canonical", {}).items()
            }
            if meta.get("feature_names"):
                instance.feature_names = meta["feature_names"]
            logger.info(
                f"Model loaded from {load_path} "
                f"(encoder: {instance._encoded_to_canonical})"
            )
        else:
            logger.warning(
                f"Model loaded from {load_path} but no .meta.json found - "
                f"predictions will return encoded class IDs"
            )
        return instance
