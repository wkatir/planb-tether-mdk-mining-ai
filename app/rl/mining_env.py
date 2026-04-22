"""Gymnasium environment for the RL control agent; physics driven by ASICSpec."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

from app.config import settings
from app.data.asic_specs import ASICSpec, ANTMINER_S21_PRO


class MiningEnv(gym.Env):
    metadata = {"render_modes": []}

    # Action encodings (unchanged API).
    CLOCK_LEVELS = [-0.10, -0.05, 0.00, 0.05, 0.10]     # fractional delta on nominal clock
    VOLTAGE_OPTIONS = [0.85, 1.00, 1.15]                 # multiplier on nominal voltage

    # Safety thresholds mirror app.config.
    THERMAL_THROTTLE = settings.TEMP_THROTTLE            # 78 C
    MAX_TEMP = settings.TEMP_EMERGENCY                   # 95 C

    def __init__(
        self,
        data: list[dict] | None = None,
        asic_spec: ASICSpec = ANTMINER_S21_PRO,
    ) -> None:
        super().__init__()
        self.spec = asic_spec
        self.data = data

        # Physical baselines pulled from the ASICSpec (single source of truth
        # shared with app.data.generator.SyntheticDataGenerator).
        self.BASE_CLOCK_MHZ = float(asic_spec.nominal_clock_mhz)      # e.g. 530 for S21 Pro
        self.BASE_VOLTAGE_MV = float(asic_spec.nominal_voltage_mv)    # e.g. 330 for S21 Pro
        self.BASE_HASHRATE_TH = float(asic_spec.hashrate_th)          # e.g. 234 for S21 Pro
        self.BASE_POWER_W = float(asic_spec.power_watts)              # e.g. 3510 for S21 Pro

        # Observation: [hashrate_th, power_w, temp_c, fan_pct, voltage_mv,
        #               errors, ambient_c, energy_price_kwh, efficiency_jth]
        obs_low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.7 * self.BASE_VOLTAGE_MV, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.BASE_HASHRATE_TH * 1.5,
                self.BASE_POWER_W * 2.0,
                self.MAX_TEMP + 10.0,
                100.0,
                1.3 * self.BASE_VOLTAGE_MV,
                100.0,
                50.0,
                0.50,
                100.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(
            len(self.CLOCK_LEVELS) * len(self.VOLTAGE_OPTIONS)
        )

        self.current_step = 0
        self.max_steps = len(data) if data is not None else 1000
        self.current_obs: np.ndarray | None = None

    def _get_default_state(self) -> np.ndarray:
        """Sensible starting state: nominal clock/voltage at 25 C ambient."""
        return np.array(
            [
                self.BASE_HASHRATE_TH,
                self.BASE_POWER_W,
                60.0,
                50.0,
                self.BASE_VOLTAGE_MV,
                0.0,
                25.0,
                0.04,
                self.BASE_POWER_W / max(self.BASE_HASHRATE_TH, 1e-6),
            ],
            dtype=np.float32,
        )

    # --- Physics (consistent with SyntheticDataGenerator) ---

    def _compute_hashrate(self, clock_mhz: float, voltage_mv: float, temp: float) -> float:
        """Hashrate scales linearly with clock; throttles above THERMAL_THROTTLE."""
        if temp >= self.THERMAL_THROTTLE:
            throttle = max(0.1, 1.0 - (temp - self.THERMAL_THROTTLE) / 20.0)
        else:
            throttle = 1.0
        f_norm = clock_mhz / self.BASE_CLOCK_MHZ
        return self.BASE_HASHRATE_TH * f_norm * throttle

    def _compute_power(self, clock_mhz: float, voltage_mv: float) -> float:
        """CMOS dynamic power: P proportional to V^2 * f."""
        v_norm = voltage_mv / self.BASE_VOLTAGE_MV
        f_norm = clock_mhz / self.BASE_CLOCK_MHZ
        return float(self.BASE_POWER_W * (v_norm**2) * f_norm)

    def _compute_temp(
        self,
        current_temp: float,
        power_w: float,
        fan_pct: float,
        ambient_c: float,
    ) -> float:
        """Discrete-time RC thermal model, same shape as the generator."""
        r_thermal = 50.0 / self.BASE_POWER_W      # C per W (nominal)
        fan_cooling = (fan_pct / 100.0) * 15.0    # up to 15 C headroom at full fan
        heat_input = power_w * r_thermal
        new_temp = 0.9 * current_temp + 0.1 * (ambient_c + heat_input - fan_cooling)
        return float(np.clip(new_temp, 10.0, self.MAX_TEMP))

    def _compute_fan(self, temp: float) -> float:
        """Fan speed as a function of chip temperature, in %."""
        if temp < 50:
            return 30.0
        if temp < 70:
            return 50.0 + (temp - 50) * 1.5
        if temp < 85:
            return 80.0 + (temp - 70) * 1.33
        return 100.0

    def _compute_efficiency_jth(self, hashrate_th: float, power_w: float) -> float:
        return 0.0 if hashrate_th <= 0 else float(power_w / hashrate_th)

    # --- Reward (revenue - cost - thermal penalty) ---

    def _compute_reward(self, obs: np.ndarray) -> float:
        (
            hashrate_th,
            power_w,
            temp,
            _fan,
            _voltage,
            _errors,
            _ambient,
            energy_price_kwh,
            _eff,
        ) = obs

        # Hashprice placeholder in $/PH/s/day (use config-driven value in prod).
        hashprice_ph_day = 50.0
        revenue = (hashprice_ph_day / 1000.0) * hashrate_th          # $/day
        energy_cost = energy_price_kwh * (power_w / 1000.0) * 24.0   # $/day

        if temp >= self.THERMAL_THROTTLE:
            thermal_penalty = -5.0 * (temp - self.THERMAL_THROTTLE) / 5.0
        else:
            thermal_penalty = 0.0

        return float(revenue - energy_cost + thermal_penalty)

    # --- Gym API ---

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.current_obs = self._get_default_state()
        return self.current_obs.astype(np.float32), {"step": self.current_step}

    def step(self, action: int):
        clock_idx, voltage_idx = divmod(int(action), len(self.VOLTAGE_OPTIONS))
        clock_delta = self.CLOCK_LEVELS[clock_idx]
        voltage_mult = self.VOLTAGE_OPTIONS[voltage_idx]

        new_clock_mhz = self.BASE_CLOCK_MHZ * (1.0 + clock_delta)
        new_voltage_mv = self.BASE_VOLTAGE_MV * voltage_mult

        (
            _hashrate,
            _power,
            temp,
            fan,
            _voltage,
            errors,
            ambient_c,
            energy_price,
            _eff,
        ) = self.current_obs

        new_power = self._compute_power(new_clock_mhz, new_voltage_mv)
        new_temp = self._compute_temp(temp, new_power, fan, ambient_c)
        new_fan = self._compute_fan(new_temp)
        new_hashrate = self._compute_hashrate(new_clock_mhz, new_voltage_mv, new_temp)
        new_errors = errors + (1.0 if new_temp >= self.THERMAL_THROTTLE else 0.0)
        new_eff = self._compute_efficiency_jth(new_hashrate, new_power)

        if self.data is not None and self.current_step < len(self.data):
            row = self.data[self.current_step]
            ambient_c = float(row.get("ambient_temp", ambient_c))
            energy_price = float(row.get("energy_price", energy_price))

        self.current_obs = np.array(
            [
                new_hashrate,
                new_power,
                new_temp,
                new_fan,
                new_voltage_mv,
                new_errors,
                ambient_c,
                energy_price,
                new_eff,
            ],
            dtype=np.float32,
        )

        reward = self._compute_reward(self.current_obs)
        self.current_step += 1
        terminated = bool(new_temp >= self.MAX_TEMP)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "step": self.current_step,
            "thermal_throttle": new_temp >= self.THERMAL_THROTTLE,
            "model": self.spec.model_name,
        }
        return self.current_obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    logger.info("Validating MiningEnv (unified with S21 Pro spec)...")
    env = MiningEnv()
    try:
        check_env(env, warn=True)
        logger.info(
            f"MiningEnv validation passed - "
            f"base {env.BASE_CLOCK_MHZ:.0f} MHz @ {env.BASE_VOLTAGE_MV:.0f} mV, "
            f"{env.BASE_HASHRATE_TH:.0f} TH/s, {env.BASE_POWER_W:.0f} W"
        )
    except Exception as exc:
        logger.error(f"Validation failed: {exc}")
        raise
