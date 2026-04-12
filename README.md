# MDK Mining AI

AI-Driven Mining Optimization & Predictive Maintenance — Tether MDK Assignment.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
docker-compose up -d db

# Generate synthetic data
python -m app.data.generator

# Ingest into PostgreSQL
python -m app.pipeline.ingestion

# Compute features
python -m app.pipeline.features

# Compute KPIs
python -m app.pipeline.kpi

# Start API
uvicorn app.api.main:app --reload --port 8000

# Start dashboard
streamlit run app.dashboard.app --server.port 8501
```

## Architecture

- **Synthetic Data Generator**: Physics-based ASIC telemetry simulation
- **Data Pipeline**: PostgreSQL, feature engineering, KPI computation
- **ML Models**: LSTM autoencoder (anomaly detection), XGBoost (failure classification)
- **RL Agent**: PPO-based optimal control policy
- **API**: FastAPI for telemetry, KPIs, health, and control endpoints
- **Dashboard**: Streamlit UI for fleet monitoring and AI insights
