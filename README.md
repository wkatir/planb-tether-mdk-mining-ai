# MDK Mining AI

AI-Driven Mining Optimization & Predictive Maintenance System for Tether's Mining Development Kit (MDK).

## Overview

This system simulates a fleet of ASIC miners (Antminer S21, S21 XP, M60S) and provides:
- **Predictive Maintenance**: ML models detect anomalies and predict failures before they occur
- **Optimal Control**: RL agent optimizes clock frequency and voltage for maximum efficiency
- **Fleet Monitoring**: Real-time dashboard with health scores and KPIs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MDK Mining AI                            │
├─────────────────────────────────────────────────────────────┤
│  Synthetic Data    │  Data Pipeline  │  ML Models          │
│  Generator          │  DuckDB         │  LSTM Autoencoder   │
│  (Parquet)         │  Features      │  XGBoost Classifier │
│                    │  KPIs          │  Health Score       │
├─────────────────────────────────────────────────────────────┤
│  RL Agent (PPO)    │  Decision Engine (Safety Layer)        │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard — Fleet Monitoring                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+

### Setup & Run

```bash
# Install dependencies
uv sync

# Set environment
cp .env.example .env

# Option A: Run entire pipeline in one command
python -m app.run_all

# Option B: Quick demo (5 miners, 1 day)
python -m app.run_all --fleet-size 5 --days 1

# Option C: Run steps individually
python -m app.data.generator          # Generate synthetic telemetry
python -m app.pipeline.ingestion      # Ingest into DuckDB
python -m app.pipeline.features       # Compute rolling features
python -m app.pipeline.kpi            # Compute KPIs (TE, ETE, PD)
python -m app.models.train_models     # Train ML models (LSTM, IF, XGBoost)

# Start dashboard
streamlit run app/dashboard/dashboard.py
```

## Project Structure

```
app/
├── config.py              # Settings (DuckDB path, physical constants)
├── control/
│   └── decision_engine.py # Safety checks + RL integration
├── dashboard/
│   └── app.py             # Streamlit monitoring dashboard
├── data/
│   ├── asic_specs.py     # ASIC miner specifications
│   └── generator.py       # Synthetic data generation
├── models/
│   ├── anomaly_detector.py  # LSTM autoencoder (PyTorch)
│   ├── isolation_forest.py  # Isolation Forest (scikit-learn)
│   ├── failure_classifier.py # XGBoost multi-class
│   ├── health_score.py    # Combined anomaly + failure score
│   └── train_models.py    # End-to-end ML training pipeline
├── pipeline/
│   ├── ingestion.py       # Parquet → DuckDB
│   ├── features.py        # Rolling statistics, z-scores
│   └── kpi.py            # True Efficiency & ETE KPIs
└── rl/
    ├── mining_env.py      # Gymnasium environment
    └── train_agent.py     # PPO training (Stable-Baselines3)

data/
├── raw/                   # Synthetic telemetry (Parquet)
└── processed/            # Engineered features

pyproject.toml            # Dependencies
.env.example               # Environment template
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DUCKDB_PATH` | `./data/mining.duckdb` | DuckDB database file path |
| `APP_ENV` | `development` | Environment (development/production) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `FLEET_SIZE` | `50` | Number of miners to simulate |
| `SIMULATION_DAYS` | `30` | Days of historical data |

## ML Models

### Anomaly Detector (LSTM Autoencoder)
- Architecture: 64→32→16→32→64 neurons with LSTM layers
- Trained on normal operating data
- Reconstruction error threshold detects anomalies
- Input: 48-timestep sequences of 7 features (temp, power, hashrate, voltage, fan, errors, ambient)

### Anomaly Detector (Isolation Forest)
- Unsupervised ensemble detector (200 estimators, 5% contamination)
- Complements LSTM: catches global multivariate outliers fast
- Ensemble agreement between LSTM + IF = high confidence anomaly

### Failure Classifier (XGBoost)
- Multi-class: `normal`, `overheating`, `power_issue`, `hashboard_failure`, `fan_failure`
- Features: rolling statistics, z-scores, temperature trends
- Probability output for pre-failure warning

### Health Score
- Combined score: `1.0 - (0.4 * anomaly_score + 0.6 * failure_probability)`
- Aggregated at device and fleet level

## RL Agent (PPO)

- Environment: `MiningEnv` (Gymnasium)
- State: Device telemetry + energy price + fleet constraints
- Actions: `{overclock +5%, maintain, underclock -5%}` per device
- Reward: `efficiency * hashrate - electricity_cost`
- Constraints: Temperature < 85°C, power deviation < 10%

## KPIs

### True Efficiency (TE)
```
TE = (P_asic + P_cooling + P_aux) / (H * eta_env * eta_mode)
eta_env = max(0.70, 1.0 - 0.008 * (T_ambient - 25))
```

### Economic True Efficiency (ETE)
```
ETE = (0.024 * TE * energy_price) / (hashprice / 1000)
```

### Profit Density (PD)
```
PD = (daily_revenue - daily_cost) / P_total  [$/W/day]
```

## ASIC Specifications

| Model | Hashrate (TH/s) | Power (W) | Efficiency (J/TH) | Source |
|-------|-----------------|-----------|-------------------|--------|
| Antminer S21 | 200 | 3,500 | 17.5 | [miningnow](https://miningnow.com/asic-miner/bitmain-antminer-s21-200th-s/) |
| Antminer S21 Pro | 234 | 3,510 | 15.0 | [miningnow](https://miningnow.com/asic-miner/bitmain-antminer-s21-pro-234th-s/) |
| Antminer S21 XP | 270 | 3,645 | 13.5 | [miningnow](https://miningnow.com/asic-miner/bitmain-antminer-s21-xp-270th-s/) |
| WhatsMiner M60S | 186 | 3,441 | 18.5 | [hashrateindex](https://hashrateindex.com/rigs/microbt-whatsminer-186-m60s) |

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check app/

# Format
ruff format app/
```

## License

Apache License 2.0 — matching MDK and MOS ecosystem. See [LICENSE](LICENSE).
