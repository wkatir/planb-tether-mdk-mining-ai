# Failure-mode signatures

Derived from the failure physics in `app/data/generator.py` (the same patterns
the ML models are trained on).

## Nominal values per model

| Model | Hashrate (TH/s) | Power (W) | Efficiency (J/TH) |
|---|---|---|---|
| Antminer S21 | 200 | 3500 | 17.5 |
| Antminer S21 Pro | 234 | 3510 | 15.0 |
| Antminer S21 XP | 270 | 3645 | 13.5 |
| WhatsMiner M60S | 186 | 3441 | 18.5 |

`overclock` mode legitimately raises hashrate, power and temperature;
`low_power` lowers them. Judge a reading against its operating mode.

## Signatures

| Mode | What you see in `get_device_status` | Develops over | Typical action |
|---|---|---|---|
| Thermal | `chip_temperature_c` climbing toward `temp_warning_c`; `fan_speed_rpm` well above the fleet's usual level and rising with temperature; errors creeping up; hashrate and power roughly nominal | 1-3 days | Underclock (Decision Engine), then inspect airflow, dust, fan, thermal paste |
| Hashboard | `asic_hashrate_th` well below nominal (up to about a third lost) while power and temperature stay normal; TE gets worse; a few extra errors | 0.5-2 days | Investigate: reseat / replace the hashboard |
| PSU | `asic_power_w` off nominal with no matching change in hashrate; TE unstable between readings; few errors | 2-7 days | Investigate on site: measure voltage before swapping the PSU |

## Reading it honestly

- Fan speed climbing **and** temperature still rising is the strongest single
  signal available: cooling is losing headroom. 6000 rpm is the fan's ceiling.
- Low hashrate with high temperature is thermal throttling, not a hashboard fault.
- ETE above `ete_breakeven` with healthy telemetry is an economics problem
  (energy price, hashprice), not a hardware failure. Do not diagnose hardware.
- One snapshot cannot show a trend. If the cause depends on "rising" or
  "unstable", confidence is at most medium.
