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
│  Generator          │  PostgreSQL     │  LSTM Autoencoder   │
│  (Parquet)         │  Features      │  XGBoost Classifier │
│                    │  KPIs          │  Health Score       │
├─────────────────────────────────────────────────────────────┤
│  RL Agent (PPO)    │  Decision Engine (Safety Layer)        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI REST API  │  Streamlit Dashboard                  │
│  /api/v1/telemetry │  Fleet Monitoring                     │
│  /api/v1/kpi      │  Anomaly Detection                   │
│  /api/v1/health    │  Control Recommendations             │
│  /api/v1/control   │                                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16+
- Docker (optional)

### Local Development

```bash
# Install dependencies
uv sync

# Start PostgreSQL
docker-compose up -d db

# Set environment
cp .env.example .env

# Generate synthetic telemetry (50 miners, 30 days)
python -m app.data.generator

# Ingest into PostgreSQL
python -m app.pipeline.ingestion

# Compute rolling features
python -m app.pipeline.features

# Compute KPIs (True Efficiency, ETE)
python -m app.pipeline.kpi

# Start API
uvicorn app.api.main:app --reload --port 8000

# Start dashboard (new terminal)
streamlit run app.dashboard.app --server.port 8501
```

### Docker

```bash
# Build and start all services
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Project Structure

```
app/
├── api/                    # FastAPI REST API
│   ├── main.py            # App entry point, routes
│   ├── schemas.py         # Pydantic request/response models
│   ├── dependencies.py    # Database session management
│   └── routes/
│       ├── telemetry.py   # /api/v1/telemetry
│       ├── kpi.py         # /api/v1/kpi
│       ├── health.py      # /api/v1/health
│       └── control.py     # /api/v1/control
├── config.py              # Settings (DB URL, physical constants)
├── control/
│   └── decision_engine.py # Safety checks + RL integration
├── dashboard/
│   └── app.py             # Streamlit monitoring dashboard
├── data/
│   ├── asic_specs.py     # ASIC miner specifications
│   └── generator.py       # Synthetic data generation
├── models/
│   ├── anomaly_detector.py  # LSTM autoencoder (PyTorch)
│   ├── failure_classifier.py # XGBoost multi-class
│   └── health_score.py    # Combined anomaly + failure score
├── pipeline/
│   ├── ingestion.py       # Parquet → PostgreSQL
│   ├── features.py        # Rolling statistics, z-scores
│   └── kpi.py            # True Efficiency & ETE KPIs
└── rl/
    ├── mining_env.py      # Gymnasium environment
    └── train_agent.py     # PPO training (Stable-Baselines3)

data/
├── raw/                   # Synthetic telemetry (Parquet)
└── processed/            # Engineered features

docker-compose.yml         # API + PostgreSQL services
Dockerfile                 # API container image
alembic/                  # Database migrations
pyproject.toml            # Dependencies
.env.example               # Environment template
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/app` | PostgreSQL connection |
| `APP_ENV` | `development` | Environment (development/production) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `FLEET_SIZE` | `50` | Number of miners to simulate |
| `SIMULATION_DAYS` | `30` | Days of historical data |
| `API_PORT` | `8000` | FastAPI server port |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/telemetry` | Fleet telemetry with pagination |
| GET | `/api/v1/telemetry/{device_id}` | Single device history |
| GET | `/api/v1/kpi` | Fleet KPIs (efficiency, hashrate) |
| GET | `/api/v1/kpi/fleet` | Aggregate fleet statistics |
| GET | `/api/v1/health` | Overall fleet health score |
| GET | `/api/v1/health/{device_id}` | Single device health |
| POST | `/api/v1/control/recommend` | Get overclock/underclock recommendation |

## ML Models

### Anomaly Detector (LSTM Autoencoder)
- Architecture: 64→32→16→32→64 neurons with LSTM layers
- Trained on normal operating data
- Reconstruction error threshold detects anomalies
- Input: 10-feature rolling window (temp, voltage, power, hashrate, etc.)

### Failure Classifier (XGBoost)
- Multi-class: `normal`, `overheating`, `power_issue`, `hashboard_failure`, `fan_failure`
- Features: rolling statistics, z-scores, temperature trends
- Probability output for pre-failure warning

### Health Score
- Combined score: `0.4 * anomaly_score + 0.6 * failure_probability`
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
TE = hashrate / (power * (1 + temp_factor))
temp_factor = beta * max(0, temp - reference_temp)
```

### Energy-to-Efficiency (ETE)
```
ETE = (power * electricity_price) / hashrate
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
