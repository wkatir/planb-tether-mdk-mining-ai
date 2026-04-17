import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger


class MiningEnv(gym.Env):
    metadata = {"render_modes": []}

    CLOCK_LEVELS = [-0.10, -0.05, 0.00, 0.05, 0.10]
    VOLTAGE_OPTIONS = [0.85, 1.00, 1.15]

    BASE_CLOCK = 2000
    BASE_VOLTAGE = 1.00
    THERMAL_THRESHOLD = 78.0
    MAX_TEMP = 95.0

    def __init__(self, data=None, hashrate_model=None):
        super().__init__()
        self.data = data
        self.hashrate_model = hashrate_model

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0.7, 0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [150, 4000, 110, 100, 1.3, 100, 50, 0.5, 50], dtype=np.float32
            ),
            shape=(9,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(15)

        self.current_step = 0
        self.max_steps = len(data) if data is not None else 1000
        self.current_obs = None

    def _get_default_state(self):
        return np.array(
            [
                100.0,
                2500.0,
                60.0,
                50.0,
                1.0,
                0.0,
                25.0,
                0.10,
                0.04,
            ],
            dtype=np.float32,
        )

    def _compute_hashrate(self, clock_mhz: float, voltage: float, temp: float) -> float:
        if temp >= self.THERMAL_THRESHOLD:
            throttle_factor = max(0.1, 1.0 - (temp - self.THERMAL_THRESHOLD) / 15.0)
        else:
            throttle_factor = 1.0

        clock_factor = clock_mhz / self.BASE_CLOCK
        voltage_factor = (voltage / self.BASE_VOLTAGE) ** 2

        base_hashrate = 100.0
        return base_hashrate * clock_factor * voltage_factor * throttle_factor

    def _compute_power(
        self, clock_mhz: float, voltage: float, fan_speed: float
    ) -> float:
        dynamic_power = (clock_mhz / 1000) * (voltage**2) * (fan_speed / 100 + 0.5)
        return 2500 + dynamic_power * 100

    def _compute_temp(
        self, current_temp: float, power: float, fan_speed: float, ambient_temp: float
    ) -> float:
        heat_generation = power * 0.005
        cooling = (fan_speed / 100) * 0.8 + 0.2
        ambient_influence = (ambient_temp - 25) * 0.05

        new_temp = current_temp + heat_generation - cooling + ambient_influence
        return np.clip(new_temp, 20.0, self.MAX_TEMP)

    def _compute_fan(self, temp: float) -> float:
        if temp < 50:
            return 30.0
        elif temp < 70:
            return 50.0 + (temp - 50) * 1.5
        elif temp < 85:
            return 80.0 + (temp - 70) * 1.33
        else:
            return 100.0

    def _compute_efficiency(self, hashrate: float, power: float) -> float:
        if power <= 0:
            return 0.0
        return hashrate / power * 1000

    def _compute_reward(self, obs: np.ndarray) -> float:
        (
            hashrate,
            power,
            temp,
            fan,
            voltage,
            errors,
            ambient_temp,
            energy_price,
            efficiency,
        ) = obs

        if temp >= self.THERMAL_THRESHOLD:
            temp_penalty = -5.0 * ((temp - self.THERMAL_THRESHOLD) / 5)
        else:
            temp_penalty = 0.0

        hashprice = energy_price * 1000
        revenue = hashprice * hashrate / 1e6
        energy_cost = energy_price * power / 1000

        reward = revenue - energy_cost + temp_penalty
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.current_obs = self._get_default_state()

        info = {"step": self.current_step}
        return self.current_obs.astype(np.float32), info

    def step(self, action: int):
        clock_idx = action // 3
        voltage_idx = action % 3

        clock_delta = self.CLOCK_LEVELS[clock_idx]
        voltage_mult = self.VOLTAGE_OPTIONS[voltage_idx]

        clock_mhz = self.BASE_CLOCK * (1 + clock_delta)
        voltage = self.BASE_VOLTAGE * voltage_mult

        (
            hashrate,
            power,
            temp,
            fan,
            voltage_current,
            errors,
            ambient_temp,
            energy_price,
            efficiency,
        ) = self.current_obs

        new_hashrate = self._compute_hashrate(clock_mhz, voltage, temp)
        new_power = self._compute_power(clock_mhz, voltage, fan)
        new_temp = self._compute_temp(temp, new_power, fan, ambient_temp)
        new_fan = self._compute_fan(new_temp)
        new_voltage = voltage
        new_errors = errors + (1.0 if new_temp >= self.THERMAL_THRESHOLD else 0.0)
        new_efficiency = self._compute_efficiency(new_hashrate, new_power)

        if self.data is not None and self.current_step < len(self.data):
            row = self.data[self.current_step]
            ambient_temp = row.get("ambient_temp", ambient_temp)
            energy_price = row.get("energy_price", energy_price)

        self.current_obs = np.array(
            [
                new_hashrate,
                new_power,
                new_temp,
                new_fan,
                new_voltage,
                new_errors,
                ambient_temp,
                energy_price,
                new_efficiency,
            ],
            dtype=np.float32,
        )

        reward = self._compute_reward(self.current_obs)
        self.current_step += 1

        terminated = bool(new_temp >= self.MAX_TEMP)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "step": self.current_step,
            "thermal_throttle": new_temp >= self.THERMAL_THRESHOLD,
        }

        return self.current_obs, reward, terminated, truncated, info

    def close(self):
        pass


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    logger.info("Validating MiningEnv with stable-baselines3 check_env...")
    env = MiningEnv()
    try:
        check_env(env, warn=True)
        logger.info("MiningEnv validation passed!")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise
