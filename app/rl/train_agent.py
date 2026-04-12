import os
import numpy as np
from loguru import logger
import duckdb

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
except ImportError as e:
    logger.error(f"Missing RL dependencies: {e}")
    raise

from app.rl.mining_env import MiningEnv


def load_training_data(duckdb_path: str = "./data/mining.duckdb"):
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
        query = """
            SELECT 
                hashrate,
                power,
                temp,
                fan_speed as fan,
                voltage,
                errors,
                ambient_temp,
                energy_price,
                efficiency
            FROM kpi
            ORDER BY timestamp
        """
        df = conn.execute(query).fetchdf()
        conn.close()

        if df.empty:
            logger.warning("KPI table is empty, using placeholder data")
            return None

        data_dict = df.to_dict(into=list)
        logger.info(f"Loaded {len(df)} training samples from DuckDB")
        return data_dict

    except Exception as e:
        logger.warning(f"Could not load training data: {e}")
        return None


def create_placeholder_env():
    logger.info("Creating placeholder environment for testing")
    dummy_data = []
    for i in range(100):
        dummy_data.append(
            {
                "hashrate": np.random.uniform(80, 120),
                "power": np.random.uniform(2200, 2800),
                "temp": np.random.uniform(55, 75),
                "fan": np.random.uniform(40, 70),
                "voltage": np.random.uniform(0.9, 1.1),
                "errors": np.random.randint(0, 5),
                "ambient_temp": np.random.uniform(20, 30),
                "energy_price": np.random.uniform(0.08, 0.15),
                "efficiency": np.random.uniform(35, 45),
            }
        )
    return dummy_data


def train_ppo(
    data=None,
    total_timesteps: int = 500_000,
    save_path: str = "./data/models/ppo_mining.zip",
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
):
    logger.info("Initializing PPO training...")

    if data is None:
        logger.warning("No data provided, using placeholder data")
        logger.warning("Run modules 4-6 first for real training data")
        data = create_placeholder_env()

    env = MiningEnv(data=data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=0,
        tensorboard_log="./data/models/tensorboard/",
    )

    logger.info(f"Training PPO for {total_timesteps} timesteps...")
    logger.info(
        f"Hyperparameters: n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}"
    )

    progress_interval = max(1, total_timesteps // 20)
    for step in range(0, total_timesteps, n_steps):
        model.learn(
            total_timesteps=n_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        current_step = step + n_steps
        if current_step % progress_interval == 0 or current_step >= total_timesteps:
            logger.info(
                f"Training progress: {current_step}/{total_timesteps} ({100 * current_step / total_timesteps:.1f}%)"
            )

    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

    return model


if __name__ == "__main__":
    logger.info("=== Module 12: PPO Training ===")

    duckdb_path = "./data/mining.duckdb"
    if os.path.exists(duckdb_path):
        logger.info("Loading data from DuckDB...")
        training_data = load_training_data(duckdb_path)
    else:
        logger.warning(f"DuckDB not found at {duckdb_path}")
        logger.warning("Run modules 4-6 first to generate training data")
        training_data = None

    save_path = "./data/models/ppo_mining.zip"

    model = train_ppo(
        data=training_data,
        total_timesteps=100_000,
        save_path=save_path,
    )

    logger.info("Training complete!")
    logger.info(f"Model saved at: {save_path}")
