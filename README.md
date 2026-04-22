# MDK Mining AI

AI-driven mining optimization and predictive maintenance for Tether's Mining Development Kit (MDK). Two-layer design: **ML first** (DuckDB + LSTM-AE + Isolation Forest + XGBoost + PPO), **LLM on top** (Gemma 4 31B via NVIDIA NIM, tool-calling).

## What it does

- **Predictive Maintenance**: LSTM Autoencoder + Isolation Forest detect anomalies; XGBoost predicts failure type (thermal, hashboard, PSU) 12-72 h in advance.
- **Optimal Control**: PPO RL agent proposes clock/voltage actions; Decision Engine (safety > AI > operator) gates every command.
- **Custom KPIs**: True Efficiency (TE), Economic True Efficiency (ETE), Profit Density (PD) — capture cooling overhead, environmental derating, operating-mode penalties that plain J/TH misses.
- **LLM Fleet Assistant**: Gemma 4 31B on NVIDIA NIM answers natural-language questions using 5 Python tools. LLM consumes pre-computed ML outputs only — never raw telemetry.
- **Synthetic Data Generator in the UI**: configure fleet size, days, failure rate, seed from the dashboard — no CLI required.

## Architecture (layers)

```
Layer 0 Hardware  ->  ASIC fleet + site sensors
Layer 1 MDK Core  ->  JS workers / Hyperbee / HRPC (Tether, out of scope)
Layer 2 Ingestion ->  Parquet + Pydantic -> DuckDB (embedded, local-first)
Layer 3 Features  ->  DuckDB window functions (rolling, z-scores, dt)
Layer 4 ML        ->  KPI engine + IF + LSTM-AE + XGBoost + PPO
Layer 5 LLM       ->  Gemma 4 31B (NVIDIA NIM), tool-calling agent
Layer 6 Safety    ->  3-tier Decision Engine, hard thermal/voltage limits
Layer 7 Outputs   ->  Commands to MDK workers / Streamlit / ONNX artefacts
```

See [docs/architecture.mmd](docs/architecture.mmd) and [docs/technical_report.md](docs/technical_report.md).

## Quick Start

### Prerequisites
- Python 3.11+
- Optional: free NVIDIA NIM key (`nvapi-...`) from https://build.nvidia.com for the AI Assistant tab.

### Setup

```bash
pip install -e .
cp .env.example .env
# Edit .env and paste NVIDIA_API_KEY if you want the AI Assistant
```

### Run

```bash
# Option A: full CLI pipeline (50 miners, 30 days)
python -m app.run_all

# Option B: quick demo (5 miners, 1 day, skip ML training)
python -m app.run_all --fleet-size 5 --days 1 --skip-training

# Option C: skip the CLI entirely - open the dashboard and use the
# 'Synthetic Data' tab to generate data interactively.
streamlit run app/dashboard/dashboard.py
```

### Dashboard tabs

| Tab | What it shows |
|-----|---------------|
| Fleet Overview | Miner count, avg hashrate / TE / ETE, profitable vs loss-making. |
| Device Detail | Latest telemetry + hashrate/temperature/power time series per miner. |
| KPI Trends | Fleet-wide TE and ETE over time. |
| AI Insights | Health score distribution, miners at risk, rule-based recommendations. |
| **Synthetic Data** | Interactive form to regenerate telemetry with custom fleet size, days, failure rate, seed. Invalidates query cache on completion. |
| **AI Assistant** | Chat with Gemma 4 31B (NVIDIA NIM) over your fleet via tool-calling. |

## Project Structure

```
app/
├── run_all.py                # Full pipeline end-to-end
├── config.py                 # Pydantic-settings config (thermal/voltage/KPI constants)
├── ai/
│   ├── llm_client.py         # NVIDIA NIM / OpenAI-compatible client (Gemma)
│   ├── tools.py              # 5 Python tools exposed to the LLM
│   └── agent.py              # Tool-calling loop (FleetAgent)
├── control/
│   └── decision_engine.py    # 3-tier safety gate (dataclass commands)
├── dashboard/
│   └── dashboard.py          # Streamlit app (6 tabs)
├── data/
│   ├── asic_specs.py         # S21 / S21 Pro / S21 XP / M60S spec registry
│   └── generator.py          # Physics-grounded synthetic telemetry
├── models/
│   ├── anomaly_detector.py   # LSTM Autoencoder (PyTorch)
│   ├── isolation_forest.py   # sklearn IF, 200 estimators
│   ├── failure_classifier.py # XGBoost multi-class + SHAP + ONNX export
│   ├── health_score.py       # Ensemble: 0.4*anomaly + 0.6*failure
│   └── train_models.py       # End-to-end training pipeline
├── pipeline/
│   ├── ingestion.py          # Parquet -> DuckDB with Pydantic bounds
│   ├── features.py           # Rolling window SQL (5m / 1h, z-scores)
│   └── kpi.py                # TE / ETE / Profit Density
└── rl/
    ├── mining_env.py         # Gymnasium env, 15 discrete actions
    └── train_agent.py        # PPO via Stable-Baselines3

tests/                        # 6 test files, incl. test_decision_engine.py
docs/
├── technical_report.md       # Full 10-section report with references
└── architecture.mmd          # Mermaid layered architecture diagram
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DUCKDB_PATH` | `./data/mining.duckdb` | DuckDB database file |
| `APP_ENV` | `development` | development / production |
| `LOG_LEVEL` | `INFO` | logging verbosity |
| `FLEET_SIZE` | `50` | default miners to simulate |
| `SIMULATION_DAYS` | `30` | default days of history |
| `FAILURE_INJECTION_RATE` | `0.15` | share of miners with injected pre-failures |
| `NVIDIA_API_KEY` | — | NVIDIA NIM key (`nvapi-...`), for AI Assistant |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `google/gemma-4-31b-it` | model id (switchable to local Ollama) |
| `LLM_TEMPERATURE` | `0.2` | sampling temp for the assistant |
| `LLM_MAX_TOKENS` | `1024` | max completion tokens |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `OPERATOR_EMAIL` | — | optional; alerts fall back to `./data/alerts.log` when unset |

## KPIs

### True Efficiency (TE)
```
TE = (P_asic + P_cooling + P_aux) / (H * eta_env * eta_mode)   [J/TH]
eta_env  = max(0.70, 1.0 - 0.008 * (T_ambient - 25))
eta_mode = {normal: 1.00, low_power: 1.10, overclock: 0.85}
```

### Economic True Efficiency (ETE)
```
ETE = (0.024 * TE * energy_price) / (hashprice / 1000)         [dimensionless]
ETE < 1.0 -> profitable
ETE = 1.0 -> breakeven
ETE > 1.0 -> losing money
```

### Profit Density (PD)
```
PD = (daily_revenue - daily_cost) / P_total                     [$/W/day]
```

## Safety Constraints

| Constraint | Threshold | Action |
|-----------|-----------|--------|
| Chip temperature | >= 78 C | Throttle (underclock 20%) |
| Chip temperature | >= 95 C | Emergency shutdown |
| Voltage deviation | > +/-10% | Underclock + alert |
| Command rate | < 5 min/device | Queue / reject |
| Fleet overclock | > 20% simultaneous | Block new overclock |

Override hierarchy: **Safety Layer > AI Recommendations > Operator Commands**.

## ASIC Specifications

| Model | Hashrate (TH/s) | Power (W) | Efficiency (J/TH) | Source |
|-------|-----------------|-----------|-------------------|--------|
| Antminer S21 | 200 | 3,500 | 17.5 | [miningnow](https://miningnow.com/asic-miner/bitmain-antminer-s21-200th-s/) |
| Antminer S21 Pro | 234 | 3,510 | 15.0 | [miningnow](https://miningnow.com/asic-miner/bitmain-antminer-s21-pro-234th-s/) |
| Antminer S21 XP | 270 | 3,645 | 13.5 | [miningnow](https://miningnow.com/asic-miner/bitmain-antminer-s21-xp-270th-s/) |
| WhatsMiner M60S | 186 | 3,441 | 18.5 | [hashrateindex](https://hashrateindex.com/rigs/microbt-whatsminer-186-m60s) |

## Development

```bash
pytest tests/ -v        # unit tests
ruff check app/
ruff format app/
```

## License

Apache License 2.0 — matching the MDK / MOS ecosystem. See [LICENSE](LICENSE).
