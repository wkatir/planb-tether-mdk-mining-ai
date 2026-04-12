from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

FEATURE_NAMES: list[str] = [
    "temp",
    "power",
    "hashrate",
    "voltage",
    "fan",
    "errors",
    "ambient",
]
FEATURE_DIM: int = len(FEATURE_NAMES)
MODEL_DIR: Path = Path("data/models")
MODEL_PATH: Path = MODEL_DIR / "lstm_autoencoder.pt"


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LSTM(input_dim, 64, batch_first=True),
            nn.LSTM(64, 32, batch_first=True),
        )
        self.bottleneck = nn.Linear(32, 16)
        self.decoder = nn.Sequential(
            nn.LSTM(16, 32, batch_first=True),
            nn.LSTM(32, 64, batch_first=True),
        )
        self.output = nn.Linear(64, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, _ = self.encoder(x)
        enc = enc[:, -1, :]
        bottleneck = self.bottleneck(enc)
        bottleneck = bottleneck.unsqueeze(1).repeat(1, 2, 1)
        dec, _ = self.decoder(bottleneck)
        dec = dec[:, -1, :]
        return self.output(dec)


@dataclass
class AnomalyResult:
    score: float
    is_anomaly: bool
    threshold: float


class AnomalyDetector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMAutoencoder(FEATURE_DIM).to(self.device)
        self.threshold: float = 0.0
        self.is_trained: bool = False
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def train(self, healthy_data: np.ndarray, n_samples: int = 10000) -> None:
        logger.info("Training LSTM Autoencoder on healthy data")
        if healthy_data.shape[0] == 0:
            logger.warning("No healthy data available for training")
            return

        n = min(n_samples, healthy_data.shape[0])
        indices = np.random.choice(healthy_data.shape[0], n, replace=False)
        data = healthy_data[indices]

        self.scaler_mean = np.mean(data, axis=0)
        self.scaler_std = np.std(data, axis=0) + 1e-8
        normalized = (data - self.scaler_mean) / self.scaler_std

        sequences = []
        for i in range(len(normalized) - 2):
            seq = normalized[i : i + 3]
            sequences.append(seq)
        sequences = np.array(sequences)

        X = torch.FloatTensor(sequences).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(50):
            total_loss = 0.0
            for batch in X:
                batch = batch.unsqueeze(0)
                recon = self.model(batch)
                loss = criterion(recon, batch.squeeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss / len(X):.6f}")

        self.model.eval()
        with torch.no_grad():
            errors: list[float] = []
            for seq in X:
                seq = seq.unsqueeze(0)
                recon = self.model(seq)
                err = criterion(recon, seq.squeeze(0)).item()
                errors.append(err)

        self.threshold = float(np.percentile(errors, 95))
        self.is_trained = True
        logger.info(
            f"Training complete. Threshold (95th percentile): {self.threshold:.6f}"
        )

    def detect(self, data: np.ndarray) -> AnomalyResult:
        if not self.is_trained:
            logger.warning("Model not trained yet, returning default")
            return AnomalyResult(score=0.0, is_anomaly=False, threshold=0.0)

        normalized = (data - self.scaler_mean) / self.scaler_std
        seq = torch.FloatTensor(normalized[:3]).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(seq)
            score = float(torch.mean((recon - seq) ** 2).item())

        return AnomalyResult(
            score=score,
            is_anomaly=score > self.threshold,
            threshold=self.threshold,
        )

    def save(self, path: Path | None = None) -> None:
        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "threshold": self.threshold,
                "is_trained": self.is_trained,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
            },
            save_path,
        )
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        load_path = path or MODEL_PATH
        if not load_path.exists():
            logger.warning(
                f"Model file not found at {load_path}, returning untrained model"
            )
            return cls()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(load_path, map_location=device)

        instance = cls()
        instance.model.load_state_dict(checkpoint["model_state"])
        instance.threshold = checkpoint["threshold"]
        instance.is_trained = checkpoint["is_trained"]
        instance.scaler_mean = checkpoint["scaler_mean"]
        instance.scaler_std = checkpoint["scaler_std"]
        logger.info(f"Model loaded from {load_path}")
        return instance
