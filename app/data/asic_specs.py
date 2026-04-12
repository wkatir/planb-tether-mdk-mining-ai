"""ASIC hardware specifications from manufacturer datasheets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ASICSpec:
    """Immutable specification for a single ASIC miner model."""

    model_name: str
    manufacturer: str
    algorithm: str
    hashrate_th: float
    power_watts: float
    efficiency_jth: float
    cooling_type: str
    chip_process_nm: int
    num_hashboards: int
    chips_per_board: int
    nominal_voltage_mv: float
    nominal_clock_mhz: float
    max_temp_c: float
    weight_kg: float
    dimensions_mm: tuple[int, int, int]
    noise_db: float
    release_year: int

    @property
    def total_chips(self) -> int:
        return self.num_hashboards * self.chips_per_board

    @property
    def hashrate_per_chip_th(self) -> float:
        return self.hashrate_th / self.total_chips

    @property
    def power_per_chip_w(self) -> float:
        return self.power_watts / self.total_chips


# Source: https://miningnow.com/asic-miner/bitmain-antminer-s21-200th-s/
ANTMINER_S21 = ASICSpec(
    model_name="Antminer S21",
    manufacturer="Bitmain",
    algorithm="SHA-256",
    hashrate_th=200.0,
    power_watts=3500.0,
    efficiency_jth=17.5,
    cooling_type="air",
    chip_process_nm=5,
    num_hashboards=3,
    chips_per_board=86,
    nominal_voltage_mv=340.0,
    nominal_clock_mhz=500.0,
    max_temp_c=45.0,
    weight_kg=15.4,
    dimensions_mm=(400, 195, 290),
    noise_db=75.0,
    release_year=2024,
)

# Source: https://miningnow.com/asic-miner/bitmain-antminer-s21-xp-270th-s/
ANTMINER_S21_XP = ASICSpec(
    model_name="Antminer S21 XP",
    manufacturer="Bitmain",
    algorithm="SHA-256",
    hashrate_th=270.0,
    power_watts=3645.0,
    efficiency_jth=13.5,
    cooling_type="air",
    chip_process_nm=5,
    num_hashboards=3,
    chips_per_board=90,
    nominal_voltage_mv=320.0,
    nominal_clock_mhz=550.0,
    max_temp_c=45.0,
    weight_kg=18.7,
    dimensions_mm=(449, 219, 293),
    noise_db=75.0,
    release_year=2024,
)

# Source: https://hashrateindex.com/rigs/microbt-whatsminer-186-m60s
WHATSMINER_M60S = ASICSpec(
    model_name="WhatsMiner M60S",
    manufacturer="MicroBT",
    algorithm="SHA-256",
    hashrate_th=186.0,
    power_watts=3441.0,
    efficiency_jth=18.5,
    cooling_type="air",
    chip_process_nm=3,
    num_hashboards=3,
    chips_per_board=78,
    nominal_voltage_mv=350.0,
    nominal_clock_mhz=480.0,
    max_temp_c=35.0,
    weight_kg=13.5,
    dimensions_mm=(155, 226, 430),
    noise_db=75.0,
    release_year=2023,
)

ASIC_REGISTRY: dict[str, ASICSpec] = {
    "antminer_s21": ANTMINER_S21,
    "antminer_s21_xp": ANTMINER_S21_XP,
    "whatsminer_m60s": WHATSMINER_M60S,
}

DEFAULT_ASIC = ANTMINER_S21
