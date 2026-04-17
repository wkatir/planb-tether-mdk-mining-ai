"""
src/data/generator.py — Synthetic ASIC telemetry data generator.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from app.config import settings
from app.data.asic_specs import ASICSpec, ASIC_REGISTRY


@dataclass
class MinerState:
    """Mutable state for a single simulated miner."""

    device_id: str
    model_key: str
    spec: ASICSpec
    age_days: float
    r_thermal: float
    base_error_rate: float
    is_healthy: bool
    failure_type: str | None
    failure_onset_step: int | None
    operating_mode: str


class SyntheticDataGenerator:
    """
    Generates synthetic telemetry for a mining fleet.
    """

    def __init__(
        self,
        fleet_size: int = settings.fleet_size,
        days: int = settings.simulation_days,
        interval_minutes: int = settings.sample_interval_minutes,
        failure_rate: float = settings.failure_injection_rate,
        output_dir: Path = settings.raw_data_dir,
        seed: int = 42,
    ):
        self.fleet_size = fleet_size
        self.days = days
        self.interval_minutes = interval_minutes
        self.failure_rate = failure_rate
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)

        self.steps_per_day = (24 * 60) // interval_minutes
        self.total_steps = self.steps_per_day * days
        self.miners = self._init_fleet()

        logger.info(
            f"Generator initialized: {fleet_size} miners, {days} days, "
            f"{self.total_steps} steps, {interval_minutes}-min intervals"
        )

    def _init_fleet(self) -> list[MinerState]:
        """Initialize fleet of miners with varied specs and ages."""
        miners = []
        model_keys = list(ASIC_REGISTRY.keys())
        num_failures = int(self.fleet_size * self.failure_rate)
        failure_indices = set(
            self.rng.choice(self.fleet_size, size=num_failures, replace=False)
        )
        failure_types = ["thermal", "hashboard", "psu"]

        for i in range(self.fleet_size):
            model_key = self.rng.choice(model_keys, p=[0.20, 0.35, 0.20, 0.25])
            spec = ASIC_REGISTRY[model_key]
            age = self.rng.uniform(0, 365)
            months = age / 30.0
            r_thermal = 1.0 + 0.005 * months
            base_error_rate = 0.001 * (1 + 0.02 * months)

            is_healthy = i not in failure_indices
            failure_type = None
            failure_onset = None
            if not is_healthy:
                failure_type = self.rng.choice(failure_types)
                failure_onset = self.rng.integers(
                    self.steps_per_day * 10, self.steps_per_day * 25
                )

            operating_mode = self.rng.choice(
                ["normal", "low_power", "overclock"], p=[0.7, 0.15, 0.15]
            )

            miners.append(
                MinerState(
                    device_id=f"miner_{i:03d}",
                    model_key=model_key,
                    spec=spec,
                    age_days=age,
                    r_thermal=r_thermal,
                    base_error_rate=base_error_rate,
                    is_healthy=is_healthy,
                    failure_type=failure_type,
                    failure_onset_step=failure_onset,
                    operating_mode=operating_mode,
                )
            )

        logger.info(f"Fleet initialized: {num_failures} miners with injected failures")
        return miners

    def _generate_ambient(self) -> np.ndarray:
        """Generate ambient temperature time series."""
        t = np.arange(self.total_steps)
        hours = (t * self.interval_minutes / 60.0) % 24.0
        diurnal = 10.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
        day_frac = t / self.steps_per_day
        seasonal = 3.0 * np.sin(2 * np.pi * day_frac / 30.0)
        noise = self.rng.normal(0, 2.0, size=self.total_steps)
        ambient = 25.0 + diurnal + seasonal + noise
        return np.clip(ambient, 5.0, 50.0)

    def _generate_energy_price(self) -> np.ndarray:
        """Generate energy price time series ($/kWh)."""
        t = np.arange(self.total_steps)
        hours = (t * self.interval_minutes / 60.0) % 24.0
        price = np.full(self.total_steps, 0.04)
        peak_mask = (hours >= 14) & (hours <= 20)
        price[peak_mask] += 0.03
        shoulder_mask = ((hours >= 7) & (hours < 14)) | ((hours > 20) & (hours <= 22))
        price[shoulder_mask] += 0.01
        spike_mask = self.rng.random(self.total_steps) < 0.001
        spike_multiplier = self.rng.uniform(2.0, 4.0, size=self.total_steps)
        price[spike_mask] *= spike_multiplier[spike_mask]
        price += self.rng.normal(0, 0.003, size=self.total_steps)
        return np.clip(price, 0.02, 0.50)

    def _generate_hashprice(self) -> np.ndarray:
        """Generate hashprice time series ($/PH/s/day)."""
        dt = self.interval_minutes / (24 * 60)
        mu = 0.0
        sigma = 0.15
        z = self.rng.standard_normal(self.total_steps)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        log_price = np.cumsum(log_returns)
        hashprice = 50.0 * np.exp(log_price)
        return np.clip(hashprice, 20.0, 120.0)

    def _simulate_miner(
        self, miner: MinerState, ambient: np.ndarray, energy_price: np.ndarray
    ) -> pd.DataFrame:
        """Simulate telemetry for a single miner."""
        spec = miner.spec
        n = self.total_steps
        fail_duration = 0

        mode_factors = {
            "normal": {"clock_mult": 1.00, "voltage_mult": 1.00},
            "low_power": {"clock_mult": 0.85, "voltage_mult": 0.92},
            "overclock": {"clock_mult": 1.12, "voltage_mult": 1.08},
        }
        mode = mode_factors[miner.operating_mode]

        clock_base = spec.nominal_clock_mhz * mode["clock_mult"]
        voltage_base = spec.nominal_voltage_mv * mode["voltage_mult"]

        clock = clock_base + self.rng.normal(0, 5.0, size=n)
        voltage = voltage_base + self.rng.normal(0, 3.0, size=n)

        v_norm = voltage / spec.nominal_voltage_mv
        f_norm = clock / spec.nominal_clock_mhz
        power = spec.power_watts * (v_norm**2) * f_norm
        power += self.rng.normal(0, power * 0.01)

        r_base = 50.0 / spec.power_watts
        r_thermal = r_base * miner.r_thermal

        chip_temp = np.zeros(n)
        fan_speed = np.zeros(n)

        for t in range(n):
            if t == 0:
                chip_temp[t] = ambient[t] + power[t] * r_thermal
            else:
                fan_cooling = fan_speed[t - 1] / 6000.0 * 15.0
                heat_input = power[t] * r_thermal
                chip_temp[t] = 0.9 * chip_temp[t - 1] + 0.1 * (
                    ambient[t] + heat_input - fan_cooling
                )
            temp_error = chip_temp[t] - 75.0
            fan_speed[t] = np.clip(3000 + temp_error * 100, 1500, 6000)

        hashrate = spec.hashrate_th * f_norm
        throttle_mask = chip_temp > settings.TEMP_THROTTLE
        throttle_factor = np.clip(
            1.0 - (chip_temp - settings.TEMP_THROTTLE) / 20.0, 0.3, 1.0
        )
        hashrate = np.where(throttle_mask, hashrate * throttle_factor, hashrate)
        hashrate += self.rng.normal(0, hashrate * 0.01)

        error_rate = miner.base_error_rate
        errors = self.rng.poisson(error_rate, size=n)

        if not miner.is_healthy and miner.failure_onset_step is not None:
            onset = miner.failure_onset_step
            if miner.failure_type == "thermal":
                fail_duration = self.rng.integers(
                    self.steps_per_day, 3 * self.steps_per_day
                )
                for t in range(onset, min(onset + fail_duration, n)):
                    progress = (t - onset) / fail_duration
                    chip_temp[t] += progress * 25.0
                    fan_speed[t] = min(fan_speed[t] + progress * 1500, 6000)
                    errors[t] += int(progress * 5)
            elif miner.failure_type == "hashboard":
                fail_duration = self.rng.integers(
                    self.steps_per_day // 2, 2 * self.steps_per_day
                )
                for t in range(onset, min(onset + fail_duration, n)):
                    progress = (t - onset) / fail_duration
                    hashrate[t] *= 1.0 - progress * 0.33
                    errors[t] += int(progress * 3)
            elif miner.failure_type == "psu":
                fail_duration = self.rng.integers(
                    2 * self.steps_per_day, 7 * self.steps_per_day
                )
                for t in range(onset, min(onset + fail_duration, n)):
                    progress = (t - onset) / fail_duration
                    voltage[t] += self.rng.normal(0, 15.0 * progress)
                    power[t] *= 1.0 + self.rng.normal(0, 0.05 * progress)
                    errors[t] += int(progress * 2)

        clock = np.clip(clock, 100.0, 800.0)
        voltage = np.clip(voltage, 200.0, 500.0)
        power = np.clip(power, 500.0, 8000.0)
        chip_temp = np.clip(chip_temp, 10.0, 130.0)
        fan_speed = np.clip(fan_speed, 0.0, 6500.0)
        hashrate = np.clip(hashrate, 0.0, 600.0)
        errors = np.clip(errors, 0, 50).astype(int)

        timestamps = pd.date_range(
            start="2026-04-01", periods=n, freq=f"{self.interval_minutes}min"
        )

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "device_id": miner.device_id,
                "model": miner.spec.model_name,
                "operating_mode": miner.operating_mode,
                "asic_clock_freq_mhz": clock.astype(np.float32),
                "asic_voltage_mv": voltage.astype(np.float32),
                "asic_power_w": power.astype(np.float32),
                "chip_temperature_c": chip_temp.astype(np.float32),
                "fan_speed_rpm": fan_speed.astype(np.float32),
                "asic_hashrate_th": hashrate.astype(np.float32),
                "error_count": errors.astype(np.int16),
                "ambient_temperature_c": ambient.astype(np.float32),
                "energy_price_kwh": energy_price.astype(np.float32),
                "is_healthy": miner.is_healthy,
                "failure_type": miner.failure_type if not miner.is_healthy else "none",
                "is_pre_failure": False,
            }
        )

        if not miner.is_healthy and miner.failure_onset_step is not None:
            onset = miner.failure_onset_step
            pre_start = max(0, onset - self.steps_per_day)
            df.loc[pre_start : max(pre_start, onset - 1), "is_pre_failure"] = True

        return df

    def generate(self) -> Path:
        """Generate full fleet telemetry and write to Parquet."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating site-level signals...")
        ambient = self._generate_ambient()
        energy_price = self._generate_energy_price()
        hashprice = self._generate_hashprice()

        all_dfs = []
        for i, miner in enumerate(self.miners):
            if (i + 1) % 10 == 0:
                logger.info(f"Generating miner {i + 1}/{self.fleet_size}...")
            df = self._simulate_miner(miner, ambient, energy_price)
            all_dfs.append(df)

        logger.info("Concatenating fleet data...")
        fleet_df = pd.concat(all_dfs, ignore_index=True)

        hashprice_series = pd.Series(
            np.tile(hashprice, self.fleet_size), name="hashprice_ph_day"
        )
        fleet_df["hashprice_ph_day"] = hashprice_series.values.astype(np.float32)

        output_path = self.output_dir / "fleet_telemetry.parquet"
        fleet_df.to_parquet(
            output_path, engine="pyarrow", compression="snappy", index=False
        )

        logger.info(
            f"Generated {len(fleet_df):,} rows → {output_path} "
            f"({output_path.stat().st_size / 1e6:.1f} MB)"
        )
        return output_path


if __name__ == "__main__":
    settings.ensure_dirs()
    gen = SyntheticDataGenerator()
    path = gen.generate()
    print(f"Data generated: {path}")
