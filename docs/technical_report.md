# AI-Driven Mining Optimization & Predictive Maintenance
## MDK Assignment — Full Technical Report

**Author:** Wilmer Salazar
**Mentor:** Gio Galt, Head of MOS @ Tether
**Program:** Plan B Network CUBO+ 2026
**Date:** April 2026
**Repository:** `planb-tether-mdk-mining-ai`
**License:** Apache 2.0 (matching MDK/MOS)

> This is a single consolidated document. It covers the problem, the two-layer
> design, every library used and why, hardware/CPU limitations and what better
> compute unlocks, the full Gemma model tier map, MDK/MOS integration contract,
> KPI derivations, the ML and LLM layers, the decision engine, the synthetic
> data generator, the testing strategy, deployment paths, and future work.
> Nothing is split across separate files — if it matters, it lives here.

---

## Table of Contents

0. [Executive Summary](#0-executive-summary)
1. [Problem Statement](#1-problem-statement)
2. [Design Philosophy — ML First, LLM On Top](#2-design-philosophy--ml-first-llm-on-top)
3. [Layered Architecture (8 Layers)](#3-layered-architecture-8-layers)
4. [KPI Design — TE, ETE, PD](#4-kpi-design--te-ete-pd)
5. [Data Pipeline](#5-data-pipeline)
6. [Machine Learning Layer](#6-machine-learning-layer)
7. [Libraries & Dependencies — What, Why, Alternatives](#7-libraries--dependencies--what-why-alternatives)
8. [Hardware & CPU Limitations — What Works Now, What Better Compute Unlocks](#8-hardware--cpu-limitations--what-works-now-what-better-compute-unlocks)
9. [LLM Layer Deep Dive — Gemma via NVIDIA NIM](#9-llm-layer-deep-dive--gemma-via-nvidia-nim)
10. [Gemma Local Deployment Tier Map](#10-gemma-local-deployment-tier-map)
11. [Decision Engine & Safety Model](#11-decision-engine--safety-model)
12. [MDK / MOS Integration Contract](#12-mdk--mos-integration-contract)
13. [Dashboard & Interactive Synthetic Generator](#13-dashboard--interactive-synthetic-generator)
14. [Testing Strategy](#14-testing-strategy)
15. [Deployment Scenarios](#15-deployment-scenarios)
16. [Limitations & Future Work](#16-limitations--future-work)
17. [References](#17-references)
18. [Appendix A — Environment Variables](#appendix-a--environment-variables)
19. [Appendix B — ASIC Specifications](#appendix-b--asic-specifications)
20. [Appendix C — Directory Structure](#appendix-c--directory-structure)

---

## 0. Executive Summary

This PoC delivers an **AI-driven controller** for a fleet of Bitcoin mining ASICs. It ingests (synthetic) telemetry, computes novel profitability-aware KPIs (TE / ETE / Profit Density), trains three complementary ML models for anomaly detection and failure classification, runs a PPO RL agent for dynamic clock/voltage control, and exposes a natural-language **AI assistant** backed by Gemma 4 31B on NVIDIA NIM.

The architecture is **layered** (8 layers) and sits *adjacent* to Tether's MDK/MOS JavaScript stack — not inside it. This is deliberate: MDK's lean-core JS runtime should not carry PyTorch, and Python ML should not break the mining control hot path.

The two-layer principle that runs through the whole system:

> **ML owns numbers. LLM owns language.**
>
> The LLM never reads raw telemetry rows. It calls Python tools that return
> JSON already reduced by the ML layer (KPIs, SHAP, health score). This
> sidesteps the documented numerical / time-series / tabular weaknesses of
> LLMs [[1](#ref-1)–[5](#ref-5)].

The codebase covers Tier 1 (required) and both Tier 2 options (RL and predictive maintenance) of the assignment. Tests cover the safety-critical Decision Engine with 10 regression tests that lock the priority ordering.

---

## 1. Problem Statement

From Gio Galt's *Bitcoin Mining Economics* presentation (slides 8–12 [[15](#ref-15)]):

> **Gross Profit = (Hash Price − Hash Cost) × Miner Hash Rate**

- **Hash Price** is market-determined: BTC price, block subsidy, transaction fees, network difficulty. Operators cannot control it.
- **Hash Cost** = `Electricity Cost × Hashing Efficiency`. This is the **only** lever.

Standard J/TH efficiency numbers, however, miss:

- Cooling overhead (easily 5–10% of facility power, higher in hot climates).
- Auxiliary power (network, control, lighting).
- Environmental derating: a miner's nameplate efficiency assumes ~25 °C ambient; at 35 °C it degrades measurably.
- Operating-mode penalties: overclocking is CMOS-quadratic (`P ∝ V² · f`), so aggressive modes *worsen* efficiency even though they raise raw hashrate.

The assignment asks for two deliverables:

- **Tier 1 (required):** telemetry ingestion + mapping + exploratory analysis, and a **True Efficiency (TE)** KPI richer than plain J/TH.
- **Tier 2 (stretch):** either (A) dynamic over/underclock via RL, or (B) supervised pre-failure detection.

This repository delivers **Tier 1 in full**, plus **both Tier 2 options**, plus an explicit LLM layer that sits on top of the ML outputs for operator-facing explanations and soft action recommendations. It also goes beyond the TE request by adding **Economic True Efficiency (ETE)** and **Profit Density (PD)**.

---

## 2. Design Philosophy — ML First, LLM On Top

The dominant failure mode of LLM + data pipelines is *pasting the CSV into the prompt*. That is the wrong boundary, and it fails for reasons that are now well documented:

1. **Numerical precision degrades without external tools.** LLMs tokenise numerals; arithmetic beyond a few digits is unreliable [[1](#ref-1)].
2. **Time series is out-of-distribution for pre-training.** Continuous temporal patterns are not naturally present in the data LLMs train on, so zero-shot forecasting and anomaly detection underperform task-specific models [[2](#ref-2)].
3. **Table reasoning benchmarks (TReB, TableBench) show that aggregation and multi-hop numerical reasoning still lag specialised tools**, and benchmark datasets contain noise that makes zero-shot fragile [[3](#ref-3), [4](#ref-4)].
4. **Tool-augmented / retrieval-augmented approaches consistently outperform direct prompting** on tabular tasks, which is exactly the pattern the assistant in this project uses [[5](#ref-5)].

**Hard boundary enforced in code:**

| Concern | Owner | Example output |
|---|---|---|
| Compute KPIs | ML layer (DuckDB SQL + Python) | `TE = 16.8 J/TH, ETE = 0.32, PD = $0.0025/W/day` |
| Detect anomalies | ML layer (IF + LSTM-AE + XGBoost) | `health_score = 0.42, predicted_failure = thermal` |
| Explain a miner's status | LLM | *"Miner 047 is running hot — temperature has drifted up 0.4 °C/min for the last hour, fan is maxed at 92%, error rate is 3× baseline."* |
| Draft an operator email | LLM | *"Subject: Thermal pre-failure on miner_047 within 24h …"* |
| Decide which tool to call | LLM | `get_fleet_summary()` → `list_miners_at_risk(limit=3)` → `recommend_action("miner_047")` |
| Send that email | Python `smtplib` (tool), not LLM | — |
| Issue a clock command | **Decision Engine only**, never LLM | — |

The system prompt for the LLM explicitly forbids computing KPIs from raw numbers. See [app/ai/agent.py](../app/ai/agent.py) `SYSTEM_PROMPT`.

---

## 3. Layered Architecture (8 Layers)

```
  Layer 0 — HARDWARE
    ASIC fleet (S21 / S21 Pro / S21 XP / M60S) + site sensors
  ──────────────────────────────────────────────────────────
  Layer 1 — MDK CORE (Tether, JavaScript, out of scope here)
    Device adapters (workers) → Orchestrator → API layer
    HRPC for authenticated inter-process RPC
    Hyperbee for append-only time-series storage
    Holepunch P2P for replication across sites
  ──────────────────────────────────────────────────────────
  Layer 2 — INGESTION (Python, this project)
    Parquet / future HRPC → Pydantic validation → DuckDB
  ──────────────────────────────────────────────────────────
  Layer 3 — FEATURE ENGINEERING
    DuckDB window functions: 5m / 1h rolling stats,
    z-scores, dT/dt, dP/dt, cross-device comparisons
  ──────────────────────────────────────────────────────────
  Layer 4 — ML ANALYTICS
    KPI engine (TE / ETE / PD)
    Isolation Forest (fast global outlier)
    LSTM Autoencoder (temporal anomaly, optional on CPU)
    XGBoost failure classifier (thermal / hashboard / PSU)
    Health Score = 0.4 · anomaly + 0.6 · max_failure_prob
    PPO RL agent (clock × voltage optimiser)
  ──────────────────────────────────────────────────────────
  Layer 5 — AI ASSISTANT (Gemma 4 31B via NVIDIA NIM)
    Tool-calling loop with 5 Python tools.
    The LLM consumes ML outputs, never raw telemetry.
  ──────────────────────────────────────────────────────────
  Layer 6 — DECISION ENGINE (3-tier safety gate)
    Priority: Safety > AI recommendation > Operator command.
    Hard limits on temperature, voltage, rate, fleet-wide overclock.
  ──────────────────────────────────────────────────────────
  Layer 7 — OUTPUTS
    Control commands → MDK workers (JS) over HTTP/JSON
    Streamlit dashboard (6 tabs)
    ONNX artefacts for edge deployment on MOS lib-workers
```

**Why layers?** Each layer has a narrow contract with the next. You can:

- Swap **Layer 5** (the LLM) from NIM to local Ollama by changing one env var.
- Swap **Layer 2** (storage) from DuckDB to Hyperbee by replacing one class.
- Swap **Layer 4** individual models without touching Layer 5.

This mirrors MDK's own *composable, lean-core* philosophy [[6](#ref-6)].

Full Mermaid diagram: [docs/architecture.mmd](architecture.mmd).

---

## 4. KPI Design — TE, ETE, PD

### 4.1 True Efficiency (TE)

    TE = (P_asic + P_cooling_alloc + P_aux) / (H_actual × η_env × η_mode)   [J/TH]

| Component | Meaning | Formula / default |
|---|---|---|
| `P_asic` | ASIC board power | telemetry `asic_power_w` |
| `P_cooling_alloc` | Per-miner share of site cooling power | `P_asic / ΣP_all · P_cooling_total` |
| `P_aux` | Network + control overhead | `2% × P_asic` |
| `H_actual` | Measured hashrate, post-throttle | telemetry `asic_hashrate_th` |
| `η_env` | Environmental derating | `max(0.70, 1.0 − 0.008 · (T_ambient − 25))` |
| `η_mode` | Operating mode factor | `normal=1.00, low_power=1.10, overclock=0.85` |

**Dimensional analysis:** `[W] / [TH/s · 1 · 1] = [W · s / TH] = [J/TH]`. ✓

**Worked example — Antminer S21 Pro at 28 °C ambient, normal mode:**

    P_asic = 3,510 W
    P_cooling = 3,510 × 0.07 = 245.7 W   (7% cooling overhead)
    P_aux     = 3,510 × 0.02 =  70.2 W
    P_total   = 3,825.9 W
    H         = 234.0 TH/s
    η_env     = max(0.70, 1.0 − 0.008 × (28 − 25)) = 0.976
    η_mode    = 1.00
    TE        = 3,825.9 / (234.0 × 0.976 × 1.00) = 16.75 J/TH

Spec is 15.0 J/TH. The 1.75 J/TH delta is real operator cost that plain J/TH hides.

### 4.2 Economic True Efficiency (ETE)

    daily_cost_per_TH/s    = TE × energy_price × 0.024        [$/TH/s/day]
    daily_revenue_per_TH/s = hashprice / 1,000                [$/TH/s/day]
    ETE                    = daily_cost / daily_revenue       [dimensionless]

*The 0.024 factor = 1 kWh / 3,600,000 J × 86,400 s/day.*

- `ETE < 1.0` → profitable
- `ETE = 1.0` → breakeven
- `ETE > 1.0` → losing money

**Example** — S21 Pro at $0.04/kWh, hashprice $50/PH/s/day:

    daily_cost    = 0.024 × 16.75 × 0.04 = $0.01608/TH/s/day
    daily_revenue = 50 / 1,000            = $0.05/TH/s/day
    ETE           = 0.32   → earns ~3× its electricity cost

### 4.3 Profit Density (PD)

    PD = (daily_revenue − daily_cost) / P_total   [$/W/day]

This normalises across heterogeneous miners, answering the operator's real question: *"per watt I'm buying, which miner makes me the most money?"*

| Model | TE (J/TH) | ETE at $0.04 | PD rank |
|-------|-----------|--------------|---------|
| S21 XP  | ~14.7 | ~0.28 | 1 (best) |
| S21 Pro | ~16.8 | ~0.32 | 2 |
| S21     | ~19.1 | ~0.37 | 3 |
| M60S    | ~20.1 | ~0.39 | 4 |

---

## 5. Data Pipeline

### 5.1 Synthetic Telemetry Generator — [app/data/generator.py](../app/data/generator.py)

Physics-grounded, not just noise:

1. **CMOS dynamic power:** `P = α · C_load · V² · f` — quadratic voltage scaling makes overclocking penalties visible.
2. **Thermal RC model:** `T_chip(t+1) = 0.9 · T_chip(t) + 0.1 · (T_ambient + P · R_thermal − fan_cooling)`.
3. **Hashrate is proportional to clock** until thermal throttle kicks in at 78 °C.
4. **Degradation over time:** `R_thermal` grows +0.5 %/month; base error rate grows +2 %/month.
5. **Failure injection:** configurable share of fleet (default 15%) with `thermal`, `hashboard`, or `psu` pre-failure curves.
6. **Ambient temperature** with diurnal + seasonal + noise components.
7. **Energy price** with peak / shoulder / off-peak tiers + rare spikes.
8. **Hashprice** as geometric Brownian motion (`μ=0`, `σ=0.15`).

Outputs a single Parquet file (`fleet_telemetry.parquet`) with all miners concatenated, ready for DuckDB ingestion.

### 5.2 Ingestion — [app/pipeline/ingestion.py](../app/pipeline/ingestion.py)

- **Pydantic `TelemetryRecord`** validates physical bounds (temperature ∈ [−10, 150] °C, power ∈ [0, 10,000] W, etc.). Adversarial telemetry that crosses these bounds gets `is_valid = FALSE`.
- **DuckDB `telemetry` table** persists cast-to-`DOUBLE` columns.
- `validate_bounds()` runs a second pass marking records beyond `TEMP_EMERGENCY` or with non-positive power as invalid.

### 5.3 Feature Engineering — [app/pipeline/features.py](../app/pipeline/features.py)

Pure SQL, no Pandas loops — DuckDB window functions:

- **Rolling stats (5-sample / 60-sample):** `AVG`, `STDDEV_SAMP` over `chip_temperature_c`, `asic_power_w`, `asic_hashrate_th`, `asic_voltage_mv`.
- **Rates of change:** `dT/dt`, `dP/dt`, `dH/dt` via `LAG()`.
- **`actual_efficiency_jth`:** raw `asic_power_w / asic_hashrate_th` for direct comparison to TE.
- **`errors_1h`:** rolling sum of error count.
- **Fleet z-scores:** each miner vs. fleet mean at the same timestamp.

### 5.4 KPI Computation — [app/pipeline/kpi.py](../app/pipeline/kpi.py)

Builds a `kpi` table with `true_efficiency_jth`, `economic_te`, `profit_density`, and preserves all feature columns for downstream ML and dashboard use.

---

## 6. Machine Learning Layer

### 6.1 Model roster and why each exists

| Model | Purpose | Speed | Cost on CPU | Role in the ensemble |
|---|---|---|---|---|
| **Isolation Forest** ([app/models/isolation_forest.py](../app/models/isolation_forest.py)) | Unsupervised global outlier | Very fast | Low | First-pass anomaly detector |
| **LSTM Autoencoder** ([app/models/anomaly_detector.py](../app/models/anomaly_detector.py)) | Unsupervised temporal drift | Slow to train on CPU | Medium–High | Second opinion for temporal anomalies |
| **XGBoost multi-class** ([app/models/failure_classifier.py](../app/models/failure_classifier.py)) | Supervised failure type | Fast train / fast predict | Low | What is failing (thermal / hashboard / PSU) |
| **Health Score** ([app/models/health_score.py](../app/models/health_score.py)) | Single 0–1 number for dashboards and LLM | Trivial | — | `score = 1 − (0.4 · anomaly + 0.6 · max_failure_prob)` |
| **PPO (Stable-Baselines3)** ([app/rl/train_agent.py](../app/rl/train_agent.py)) | Dynamic clock × voltage control | Slow to train on CPU | High | Proposes soft actions; Decision Engine gates them |

### 6.2 Why the 0.4 / 0.6 weighting in Health Score

Failure prediction is **actionable** — *"this miner is likely a thermal failure in 24 h"*. Anomaly detection is **exploratory** — *"something looks off"*. The weighting puts more weight on the actionable signal so operators don't chase ghost anomalies, but the anomaly stays visible enough to surface edge cases the XGBoost hasn't seen.

### 6.3 SHAP explainability

`FailureClassifier.explain()` returns SHAP values per prediction, so the LLM assistant can answer *"why is miner_047 flagged thermal?"* with the top-3 features that drove the probability up. SHAP is only called on `explain`, not on hot-path `predict`, because TreeExplainer is not cheap.

### 6.4 ONNX export

`FailureClassifier.export_onnx()` converts the XGBoost booster via `onnxmltools` + `skl2onnx`. This is the production path: training stays in Python, inference moves to **`onnxruntime-node` inside MDK workers**, so there is no Python on the mining control hot path.

---

## 7. Libraries & Dependencies — What, Why, Alternatives

This is the complete dependency rationale. For each library, I state what it does, why it's here, what it costs (disk / CPU), and what could replace it if compute gets tighter.

### 7.1 Core data layer

| Library | Role | Size | Why chosen | Alternative if tighter |
|---|---|---|---|---|
| `numpy >= 1.26` | Array math backbone | ~25 MB | Ubiquitous, all other libs consume it | None — unavoidable |
| `pandas >= 2.2` | Dataframe API for ML prep | ~45 MB | Standard for rolling windows, SHAP input | `polars` is ~15 MB and faster, but XGBoost + SHAP examples assume pandas |
| `pyarrow >= 15` | Parquet I/O | ~60 MB | MDK/MOS stores history — Parquet is the portable edge | None — Parquet is the right on-disk format |
| `duckdb >= 1.0` | Embedded SQL engine | ~20 MB | Local-first, in-process, SQL window functions, closest analogue to Hyperbee's embedded philosophy | SQLite (smaller but no window functions over large data) |

**Rationale:** DuckDB is the single most important choice in the Python side. It gives us Hyperbee's "embedded, local-first" philosophy while we prototype, with SQL window functions powerful enough to compute every rolling feature in one query. When this moves to production, **Hyperbee replaces DuckDB**, but the row schema stays the same.

### 7.2 Validation & config

| Library | Role | Size | Why chosen |
|---|---|---|---|
| `pydantic >= 2.7` | Record validation | ~15 MB | Adversarial telemetry protection at the ingestion boundary |
| `pydantic-settings >= 2.2` | Env-driven config | ~1 MB | Clean `.env` → `Settings` class in [app/config.py](../app/config.py) |
| `python-dotenv >= 1.0` | Load `.env` in dev | ~100 KB | Bundled behaviour — zero ceremony |
| `loguru >= 0.7` | Structured logs | ~1 MB | Replaces `logging` boilerplate; good enough for PoC, swappable in prod |

### 7.3 Classical ML

| Library | Role | Size | Why chosen | Alternative |
|---|---|---|---|---|
| `scikit-learn >= 1.5` | Isolation Forest, scalers | ~45 MB | Mature, well-documented, IF is perfect for "fast first-pass anomaly" | None needed — IF is a sklearn staple |
| `xgboost >= 2.1` | Failure classifier | ~150 MB | Tabular SOTA for tree ensembles, native ONNX export via `onnxmltools` | `lightgbm` (~70 MB, comparable accuracy); `sklearn.RandomForest` (no ONNX export) |
| `shap >= 0.44` | Per-prediction feature importance | ~100 MB | Operator trust: *"why this miner flagged thermal?"* | Captum (torch-only), lime (less mature) |

**Rationale:** XGBoost with SHAP is the standard stack for interpretable tabular ML. The disk cost (~250 MB for both) is a known trade-off and it's paid once.

### 7.4 Deep learning

| Library | Role | Size | Why chosen | Alternative if tighter |
|---|---|---|---|---|
| `torch >= 2.3` | LSTM Autoencoder | ~800 MB (CPU wheel) | De-facto standard; easiest LSTM implementation | **Drop torch entirely** and lean only on Isolation Forest — the Health Score can re-weight to `1.0 × IF + 0.0 × LSTM` and still work. Losing LSTM is the single biggest memory win on a CPU-only machine. |

**This is the most honest size call in the stack.** If the CPU is tight, removing torch drops deploy size by ~60%. The PoC keeps LSTM as an *optional* branch: the pipeline still runs without it because IF is trained on the same data and the Health Score has graceful fallback logic.

### 7.5 Reinforcement learning

| Library | Role | Size | Why chosen | Alternative |
|---|---|---|---|---|
| `stable-baselines3 >= 2.3` | PPO implementation | ~80 MB | Includes `check_env`, tensorboard integration, saved-model format | `rllib` (much heavier), hand-rolled PPO (pedagogical but error-prone) |
| `gymnasium >= 1.0` | Env API | ~10 MB | SB3's counterpart, replaces legacy `gym` | None — this is the standard |

**Rationale:** RL training is the most expensive step. On a CPU-only laptop, 100k steps takes minutes; 10M steps takes hours. For the PoC we ship a 100k-step model. With a GPU, the same PPO script trains orders of magnitude faster without code changes.

### 7.6 Visualisation / UI

| Library | Role | Size | Why chosen |
|---|---|---|---|
| `streamlit >= 1.37` | Dashboard | ~100 MB | Pure Python UI. No React build step. Operator-appropriate. |
| `plotly >= 5.22` | Interactive charts | ~60 MB | Interactive zoom, matches Streamlit `st.plotly_chart` |
| `matplotlib >= 3.9` | Static charts for reports | ~45 MB | Kept for completeness; not heavily used |

### 7.7 AI / LLM layer

| Library | Role | Size | Why chosen | Alternative |
|---|---|---|---|---|
| `openai >= 1.40` | OpenAI-compatible HTTP client | ~2 MB | NVIDIA NIM, Ollama, vLLM and most local servers expose the same API surface — one client talks to all [[7](#ref-7), [8](#ref-8)] | `httpx` directly (hand-rolled tool-calling parsing; more error-prone) |
| `jinja2 >= 3.1` | Email body templates | ~1 MB | Clean separation between prompt logic and formatted output | Python f-strings (works but messier for multi-line) |

**Why not `langchain` / `llama-index`?** Both add 100+ MB and strong opinions. The tool-calling loop in [app/ai/agent.py](../app/ai/agent.py) is ~80 lines of Python and does exactly what this project needs. Fewer abstractions, fewer upgrade treadmills.

### 7.8 Testing

| Library | Role | Size | Why chosen |
|---|---|---|---|
| `pytest >= 8.2` | Test runner | ~10 MB | Standard |
| `pytest-asyncio >= 0.23` | Async test support | small | Future-proof for streaming LLM tests |

### 7.9 What's intentionally NOT in the stack

- **FastAPI / uvicorn.** The PoC surface is Streamlit + CLI. When this becomes a service, FastAPI + one `/fleet/*` blueprint is the right addition — but not yet.
- **Redis / Celery.** No async task queues at PoC scale.
- **MLflow / Weights & Biases.** Tensorboard is already wired for SB3. Adding an MLOps plane is premature.
- **LangChain / Haystack.** See 7.7 — unneeded abstractions.
- **Hugging Face `transformers`.** We do not run a local LLM in this PoC (CPU constraint). NIM handles the model; `transformers` would add ~2 GB for zero benefit here.

### 7.10 Total install footprint

Rough disk sizes (CPU wheels):

| Profile | Deps | Total |
|---|---|---|
| **Minimal** (DuckDB + sklearn + XGBoost + openai, no torch, no SB3) | ~10 libs | ~500 MB |
| **Full PoC** (everything in this repo) | ~20 libs | ~1.7 GB |
| **Full + Ollama local Gemma 3 270M INT4** | + Ollama binary + model | ~2.2 GB |

For a CPU-constrained laptop the **Minimal** profile is perfectly sufficient to demonstrate Tier 1 + Tier 2 option B + the LLM assistant. Opt into torch + SB3 only if Tier 2 option A is the focus.

---

## 8. Hardware & CPU Limitations — What Works Now, What Better Compute Unlocks

This PoC was developed and tested on a CPU-only Windows laptop. The following is a candid map of what scaled down, what didn't, and what better compute changes.

### 8.1 What runs comfortably on a CPU-only laptop (baseline tier)

| Step | Typical runtime | Notes |
|---|---|---|
| Synthetic data: 50 miners × 30 days, 1-min intervals | ~30 s | 2.16M rows, 60 MB Parquet |
| DuckDB ingestion + Pydantic validation | ~1 s | In-process, embedded |
| Feature engineering (DuckDB window functions) | ~1 s | Pure SQL, vectorised |
| KPI computation (TE / ETE / PD) | <1 s | SQL only |
| Isolation Forest training | ~5 s | 200 estimators, n_jobs=-1 |
| XGBoost multi-class training | ~10 s | 28 features, 100 trees |
| Streamlit dashboard | Instant after cached query | TTL=60 s cache |
| AI assistant one-shot chat + 2 tool calls | ~3–8 s | Limited by NIM latency and token count |

### 8.2 What hurts on CPU

| Step | Symptom | Cause | Mitigation |
|---|---|---|---|
| **LSTM Autoencoder training** (50 epochs, batch size 1) | ~10–30 min for 10k samples | Code uses `for batch in X: batch.unsqueeze(0)` — effectively SGD | **Fix:** wrap in `DataLoader(batch_size=32)`. Drops training time by ~20× |
| **PPO training** (100k steps) | ~5–15 min | Env is cheap but PPO does many policy-network passes | **Fix:** shorter rollouts for PoC demo (20k steps); move to GPU for real training |
| **Local LLM inference** (e.g., Gemma 3 4B on CPU) | ~2–5 tokens/s | No GPU matmul acceleration | **Fix in PoC:** use NVIDIA NIM's hosted Gemma 4 31B (free tier). **Fix local:** use Gemma 3 270M INT4 via llama.cpp (see §10) |
| **SHAP TreeExplainer on batches > 10k** | Seconds to minutes | Matrix expansion | **Fix:** only call `explain()` per-row when the operator clicks "why" |

### 8.3 What better compute unlocks

| Hardware tier | What becomes tractable |
|---|---|
| **CPU-only laptop (8 GB RAM, no GPU)** — *this PoC* | Full pipeline end-to-end at 50 miners × 30 days. LLM via cloud NIM. Local LLM limited to Gemma 3 270M INT4 for demos only. |
| **Laptop with 8 GB GPU (RTX 3060 / M1 Max class)** | LSTM-AE batch training in seconds. PPO 1M steps in ~1 h. Local Gemma 3 4B INT4 at 20–40 tok/s. ONNX Runtime GPU inference. |
| **Desktop with 16–24 GB GPU (RTX 4080 / 4090 / A4000)** | Local Gemma 3 12B at fp16 or Gemma 3 27B INT4. PPO 10M steps in hours. Full fleet simulation at 10× scale. |
| **Server with 48–80 GB GPU (A100 / H100)** | Fine-tune Gemma on operator corpus. Train bigger RL agents (multi-miner joint action spaces). Run fleet-level simulations with 10 000+ miners. |
| **Cluster (multi-node)** | Geographically distributed RL policy, federated training across mining sites, Hyperbee P2P replication of telemetry and models. |

### 8.4 What the PoC proves without better hardware

1. **Layering works.** Every layer runs end-to-end. The bottleneck is training speed, not correctness.
2. **Cloud LLM is a legitimate PoC choice.** NIM's free tier ([[7](#ref-7), [8](#ref-8)]) removes the GPU prerequisite entirely and keeps the portability story intact (swap to local Ollama with one env var).
3. **ONNX export closes the production loop.** Training stays in Python (wherever compute lives); inference moves to `onnxruntime-node` *inside* MDK workers. CPU constraints on the dev laptop don't affect the production path.

---

## 9. LLM Layer Deep Dive — Gemma via NVIDIA NIM

### 9.1 Why NVIDIA NIM for the PoC

**Constraint:** no local GPU, 3-week deadline, must demo reliably on any laptop.

**Decision:** NVIDIA NIM exposes Gemma 4 31B Instruct (`google/gemma-4-31b-it`) over an **OpenAI-compatible REST endpoint** at `https://integrate.api.nvidia.com/v1`, with:

- A **free developer tier** — 1,000 inference credits on signup, up to 5,000 by request [[7](#ref-7), [8](#ref-8)].
- A **40 req/min rate limit** on the free tier — plenty for operator-facing assistant traffic.
- **`nvapi-` prefixed API keys** managed at [build.nvidia.com](https://build.nvidia.com).
- **OpenAI SDK compatibility** — no new client library, just `base_url` + `api_key`.

The same `openai` Python client talks to:

- NVIDIA NIM cloud (this PoC).
- Self-hosted NIM container (enterprise deployment).
- Ollama (fully local).
- vLLM, LM Studio, llama-server, any OpenAI-compatible server.

**One env var switches between them** — see [.env.example](../.env.example):

```
LLM_BASE_URL=https://integrate.api.nvidia.com/v1   # NIM cloud
LLM_BASE_URL=http://localhost:11434/v1             # Ollama
LLM_BASE_URL=http://localhost:8000/v1              # self-hosted NIM container
```

### 9.2 Tool-calling contract

Five Python tools wrap the ML layer ([app/ai/tools.py](../app/ai/tools.py)):

| Tool | Signature | Backed by | What the LLM uses it for |
|---|---|---|---|
| `get_fleet_summary()` | `-> JSON` | DuckDB `kpi` aggregate | "how is my fleet doing?" |
| `get_device_status(device_id)` | `(str) -> JSON` | DuckDB `kpi` latest row | "what's miner_047 doing?" |
| `list_miners_at_risk(limit)` | `(int=10) -> JSON` | DuckDB ranked by temp/errors | "which miners need attention?" |
| `recommend_action(device_id)` | `(str) -> JSON` | Rule on top of ML outputs | "what should I do with miner_047?" |
| `send_operator_alert(device_id, subject, body)` | `(str, str, str) -> JSON` | `smtplib` or local log | Actually send the email the LLM drafted |

The agent ([app/ai/agent.py](../app/ai/agent.py)) runs a bounded tool-calling loop (`max_steps = 4`), returns both the final answer and a debug trace of which tools were called with which arguments. The trace is visible in the Streamlit **AI Assistant** tab inside an expander.

### 9.3 What the LLM is forbidden from doing

The `SYSTEM_PROMPT` in [app/ai/agent.py](../app/ai/agent.py) codifies the boundary:

1. Never compute KPIs from raw numbers — always call a tool.
2. Never bypass the Decision Engine. Its tool vocabulary contains `send_operator_alert` (email) but *not* `set_clock` or `set_voltage`. Hardware actions only flow through the Decision Engine.
3. If the KPI table is empty, tell the user to run `python -m app.run_all` first.
4. Prefer concise, bullet-heavy operator-oriented answers.

---

## 10. Gemma Local Deployment Tier Map

Detailed map of which Gemma variant runs where. This is the reference for migrating away from cloud NIM.

### 10.1 Gemma 3 270M — **Edge & CPU-only**

- **Size:** 270 million parameters. 170M embeddings + 100M transformer.
- **Disk (INT4 quantised):** ~300 MB.
- **RAM needed:** ~500 MB runtime.
- **Speed on CPU-only laptop:** ~15–30 tok/s with `llama.cpp` [[9](#ref-9)].
- **Power:** Google's internal test on a Pixel 9 Pro SoC showed INT4 Gemma 3 270M used **0.75% of battery for 25 conversations** [[9](#ref-9)].
- **Strengths:** instruction following, text structuring, task-specific fine-tuning.
- **Weaknesses:** weak at complex tool-calling out of the box — needs fine-tuning for reliable function calls.
- **Runs on Ollama with:** `ollama run gemma3:270m` [[20](#ref-20)].
- **Use case here:** air-gapped MOS site demos; embedded board co-located with miners.

### 10.2 Gemma 3 1B — **Strong CPU tier**

- **Size:** 1 billion parameters.
- **Disk (INT4):** ~1 GB.
- **RAM:** ~1.5 GB runtime.
- **Speed on modern laptop CPU:** ~10–20 tok/s.
- **Strengths:** noticeably better instruction following and multi-turn coherence than 270M; still runs on CPU.
- **Use case here:** the realistic upgrade target once the laptop has 16 GB RAM free.

### 10.3 Gemma 3 4B — **Modest GPU / strong CPU tier**

- **Size:** 4 billion parameters.
- **Disk (INT4):** ~3 GB.
- **RAM/VRAM:** ~4–6 GB.
- **Speed:** 30–60 tok/s on an 8 GB GPU; 5–10 tok/s on CPU (usable but slow).
- **Strengths:** native tool-calling works well, multi-turn reasoning noticeably sharper, multimodal in some releases.
- **Use case here:** the sweet spot for a self-hosted MOS site with a modest GPU.

### 10.4 Gemma 3 12B / 27B — **Dedicated GPU tier**

- **Disk (INT4):** ~8 GB / ~15 GB.
- **VRAM needed:** 12 GB (12B) / 24 GB (27B).
- **Strengths:** approaches the cloud-hosted assistants on complex reasoning.
- **Use case here:** production MOS sites with dedicated inference boxes.

### 10.5 Gemma 4 31B Instruct — **This PoC's cloud choice (NIM)**

- Not practical to self-host without ~40–48 GB VRAM at fp16 (smaller with INT4 but still demanding).
- **On NVIDIA NIM** with `nvapi-...` key: `google/gemma-4-31b-it` over OpenAI-compatible endpoint [[7](#ref-7)].
- **Why this size for the PoC?** Higher quality for complex tool-calling chains without operating any inference infrastructure. Trivially downgradable later by editing one env var.

### 10.6 Decision matrix

| Available hardware | Recommended Gemma variant | Deployment |
|---|---|---|
| Laptop CPU-only, no GPU | **Gemma 4 31B via NIM** (cloud) + optional **Gemma 3 270M INT4** via Ollama for offline demos | Cloud SDK + local fallback |
| Laptop 8 GB GPU | **Gemma 3 4B INT4** via Ollama | Local first, NIM fallback |
| Desktop 12 GB GPU | **Gemma 3 12B INT4** | Local |
| Desktop 24 GB GPU | **Gemma 3 27B INT4** | Local |
| Server ≥ 48 GB GPU | **Gemma 4 27B fp16** or fine-tuned domain model | Self-hosted NIM container |

### 10.7 Alternatives if Gemma doesn't fit

- **Phi-4-mini (~3.8B)** — Microsoft, strong instruction following, Apache-2 compatible licensing available.
- **Qwen 3 0.6B / 1.7B / 4B** — Alibaba, strong tool-calling in small sizes.
- **Llama 3.2 1B / 3B** — Meta, widely supported in Ollama.

All are drop-in through the same `LLM_MODEL` env var.

---

## 11. Decision Engine & Safety Model

### 11.1 Hard constraints (non-negotiable, enforced in code)

Source: [app/control/decision_engine.py](../app/control/decision_engine.py).

| Constraint | Threshold | Action |
|---|---|---|
| Chip temperature | ≥ 78 °C | Throttle (underclock 20%) |
| Chip temperature | ≥ 95 °C | Emergency shutdown |
| Voltage deviation | > ±10 % | Underclock + alert |
| Command rate | < 5 min since last command on same device | Queue / reject |
| Fleet overclock | > 20 % of fleet simultaneously | Block new overclock |
| AI step size | > 5 % clock change in one command | Clip to 5 % |

### 11.2 Priority order

    Safety Layer > AI Recommendations > Operator Commands

Even manual operator commands are gated through Safety. Thermal and voltage limits cannot be bypassed from any layer above.

### 11.3 Three execution modes

1. **Autonomous** — AI adjusts within ±5 % clock, no human approval.
2. **Supervised** — changes > 5 % emit a recommendation with a 5-minute confirmation window. Default action on timeout: no change.
3. **Emergency** — any safety violation suspends AI recommendations, reverts to manufacturer defaults, requires manual re-enable.

### 11.4 Adversarial telemetry protection

Before telemetry reaches the RL agent or the LLM:

1. **Pydantic** — physical bound check (T ∈ [−10, 150] °C, P ∈ [0, 10 000] W, etc.).
2. **Isolation Forest** — statistical plausibility against the fleet.
3. **LSTM Autoencoder** — temporal plausibility against the device's own history.

When both detectors agree, the reading is quarantined and the agent receives the last known-good state.

---

## 12. MDK / MOS Integration Contract

### 12.1 What MDK is

Tether's **Mining Development Kit** [[6](#ref-6), [11](#ref-11)] is an **open standard** for mining infrastructure. Composed of:

- **MDK Core (JavaScript)** — backend runtime with device adapters (workers), orchestrator (brain), and API layer (interface).
- **MDK UI Kit (React)** — frontend components for dashboards.
- **MOS** — Tether's production reference implementation [[12](#ref-12), [13](#ref-13)], open-sourced under Apache 2.0 at the 2026 Plan B Forum in San Salvador.
- **Hyperbee** — append-only, crash-resilient embedded KV store, optimised for time series and P2P replication via Holepunch.
- **HRPC** — authenticated RPC between MDK processes (single-process, PM2 cluster, or Docker/K8s).

### 12.2 Why this project is *adjacent*, not *inside* MDK

| Reason | Explanation |
|---|---|
| **Language isolation** | MDK Core is intentionally lean JavaScript. Adding PyTorch + scikit-learn + SHAP to the core would contaminate its runtime with ~1 GB of deps unrelated to mining control. |
| **Crash isolation** | An XGBoost SIGSEGV, a torch OOM, or an LLM API timeout must **never** interfere with a running miner. A sidecar process cannot touch the hot path. |
| **Security** | MDK's security model is *explicit interfaces, no implicit network, authenticated RPC, least privilege* [[6](#ref-6)]. An adjacent HTTP/JSON service with a fixed schema is auditable and granularly permissioned; a monolithic in-tree import is not. |
| **Upgrade path** | Training stays in Python; inference moves to `onnxruntime-node` *inside* MDK workers. No Python on the production hot path. |

### 12.3 Integration boundaries

| Concern | MDK side (JS) | This project (Python) | Transport |
|---|---|---|---|
| Telemetry collection | `@mdk/worker-*` device adapters | Parquet reader → DuckDB (PoC); HRPC topic subscriber (prod) | Parquet file (PoC) / HRPC (prod) |
| Time-series storage | Hyperbee | DuckDB (local-first, embedded — same philosophy) | — |
| Analytics requests | HRPC client call | HTTP/JSON service with `/fleet/*` endpoints | Loopback HTTP or stdio subprocess |
| ML edge inference | `onnxruntime-node` on a lib-worker | Train in Python → export ONNX | File artefact (`data/models/*.onnx`) |
| LLM | Cloud NIM (PoC) or local Ollama | `openai` SDK against OpenAI-compatible endpoint | HTTPS / loopback HTTP |
| Operator UI | MDK UI Kit (React) | Streamlit (PoC) | UI Kit could embed a widget calling the Python AI service over HTTP |

### 12.4 Schema contract (Python → MDK)

Every Python output consumed by MDK conforms to a fixed JSON schema:

```json
{
  "device_id": "miner_007",
  "ts": "2026-04-21T18:30:00Z",
  "kpis": {
    "te_jth": 16.8,
    "ete": 0.32,
    "pd_usd_per_w_day": 0.0025
  },
  "health": {
    "score": 0.42,
    "status": "warning",
    "predicted_failure": "thermal",
    "shap_top3": [
      {"feature": "temp_rate", "contribution": 0.31},
      {"feature": "fan_std5",  "contribution": 0.18},
      {"feature": "errors_1h", "contribution": 0.11}
    ]
  },
  "recommendation": {
    "action": "underclock",
    "clock_multiplier": 0.96,
    "reason": "low_health",
    "advisory": true
  },
  "explanation": "Temperature trending up 0.4 °C/min; fan at 92%; error rate 3× baseline."
}
```

The `recommendation.advisory` flag is always `true` — the MDK worker must still pass every command through the Decision Engine before writing to the device.

### 12.5 Deployment topology

```
[ASIC / PSU / Cooler]
      │  vendor protocol
      ▼
┌─────────────────────┐  HRPC  ┌──────────────────────┐  HRPC  ┌──────────────────────┐
│ MDK Worker (JS)     │ ─────► │ MDK Orchestrator(JS) │ ─────► │ MDK API Layer (JS)   │
└──────┬──────────────┘        └──────┬───────────────┘        └──────┬───────────────┘
       │ Hyperbee                     │                                │
       │                              │                                │ HTTP/JSON
       │                              │                                ▼
       │                              │                      ┌─────────────────────────┐
       │                              │                      │ Python AI Service       │
       │                              │                      │  ← THIS PROJECT         │
       │                              │                      │ ┌─ ONNX models ────────┐│
       │                              │                      │ ├─ DuckDB / Parquet ───┤│
       │                              │                      │ └─ NVIDIA NIM / Ollama ┘│
       │                              │                      └─────────────────────────┘
       ▼
   Hyperbee (append-only TS, Holepunch-replicable across MOS sites)
```

---

## 13. Dashboard & Interactive Synthetic Generator

Streamlit app at [app/dashboard/dashboard.py](../app/dashboard/dashboard.py), launched with:

    streamlit run app/dashboard/dashboard.py

**Six tabs:**

1. **Fleet Overview** — miner count, avg hashrate / TE / ETE, profitable vs. loss-making, top-20 TE bar chart, operating-mode donut.
2. **Device Detail** — select a miner, see latest telemetry + hashrate/temperature/power time series.
3. **KPI Trends** — fleet-wide TE and ETE over time.
4. **AI Insights** — health-score histogram, miners-at-risk table, rule-based recommended actions.
5. **Synthetic Data** — *interactive* form with fleet size (5–200), days (1–30), failure rate (0–30%), seed, optional ML training toggle. Runs the full pipeline (generate → ingest → features → KPI → optional training) and invalidates the Streamlit cache on completion, so every other tab refreshes automatically.
6. **AI Assistant** — chat with Gemma 4 31B via NIM. Warns if `NVIDIA_API_KEY` is unset. Shows the tool-call trace in an expander.

**Why put the generator in the UI?** The assignment PoC has no real telemetry source. Regenerating fleets with different parameters (more/fewer failures, bigger fleets, different seeds) is the fastest way to probe ML behaviour. Baking it into the dashboard removes one CLI context switch for reviewers.

---

## 14. Testing Strategy

| File | What it covers | Count |
|---|---|---|
| [tests/test_config.py](../tests/test_config.py) | Settings loading, thermal threshold ordering, KPI parameters | 7 |
| [tests/test_safety.py](../tests/test_safety.py) | Thermal threshold ordering, ASIC spec registry integrity | 9 |
| [tests/test_decision_engine.py](../tests/test_decision_engine.py) | Safety > AI > Operator priority, rate limiting, `ControlCommand` contract | 10 |
| [tests/test_features.py](../tests/test_features.py) | DuckDB window feature correctness | depends on env |
| [tests/test_generator.py](../tests/test_generator.py) | Synthetic telemetry physical bounds | depends on env |
| [tests/test_kpi.py](../tests/test_kpi.py) | Hand-calculated TE / ETE / PD | depends on env |
| [tests/test_models.py](../tests/test_models.py) | Isolation Forest + Health Score integration | depends on env |

The first three files — the **safety-critical core** — run without heavy deps (no torch/duckdb). **26/26 pass** under `pytest tests/test_{config,safety,decision_engine}.py -v`.

The other four require the full install (`pip install -e .`) and exercise DuckDB / sklearn / torch paths.

---

## 15. Deployment Scenarios

| Scenario | LLM host | Storage | Compute | Readiness |
|---|---|---|---|---|
| **PoC demo (this repo)** | NVIDIA NIM cloud | DuckDB | CPU-only laptop | ready |
| **Self-hosted MOS site** | Local NIM container or Ollama (Gemma 3 4B) | DuckDB → Hyperbee migration | Desktop with GPU | 1–2 weeks work |
| **Air-gapped MOS site** | Ollama (Gemma 3 270M INT4) | Hyperbee | Embedded board | 2–4 weeks work |
| **Distributed fleet across sites** | Federated (per-site local Ollama) | Hyperbee + Holepunch replication | GPU node per site | Production hardening path |

The code is deployment-agnostic today because:

- Every external endpoint is configured via env vars.
- Every model is serialised to disk (joblib / ONNX / SB3 zip / torch checkpoint).
- DuckDB → Hyperbee is a swap of one class (the row schema already matches Hyperbee's append-only semantics).

---

## 16. Limitations & Future Work

This is an honest list of what is scaled down, why, and what better compute or more time unlocks.

1. **LSTM Autoencoder batch size is 1.** With `DataLoader(batch_size=32)`, training drops ~20× in time and lets us train over full weeks of data instead of sampled windows.
2. **PPO trains for 100k steps.** Production would be 1–10M steps with the true TE/ETE reward (not the simplified placeholder) and multi-miner joint action spaces.
3. **RL env is single-spec at a time.** `MiningEnv` takes one `ASICSpec` in its constructor (default `ANTMINER_S21_PRO`). A fleet-level agent that trains simultaneously across heterogeneous miners (S21 + S21 Pro + S21 XP + M60S) would require a multi-spec vectorised env or domain-randomised training. Single-spec is fine for PoC reward-shape demonstration; multi-spec is a production requirement.
4. **LLM is hosted.** The portability story is intact (local Ollama works out of the box), but the demo depends on NVIDIA NIM's free tier. With a GPU, move to local Gemma 3 4B.
5. **Hyperbee migration.** DuckDB is the PoC proxy. Production swaps to Hyperbee so the AI service participates in MDK's P2P replication directly.
6. **Live HRPC ingestion.** Parquet on disk is a PoC boundary. The production path is an HRPC subscription to MDK telemetry topics.
7. **Heat-reuse revenue.** Miners are near-100% interruptible loads — all electricity becomes heat. Counting heat as revenue would refine TE/ETE. Matches Gio's "Heat Reuse" slide.
8. **Halving-aware strategy.** Automatic fleet-wide mode changes when block subsidy halvings shift Hash Price.
9. **Demand-response integration.** Tie the RL reward to grid-level signals so miners automatically idle during peaks and full-throttle during curtailment. Matches Gio's "Demand Response" slide.
10. **Gemma fine-tuning.** Fine-tune Gemma 3 4B on Tether's own operator-conversation corpus for domain-specific fluency and lower-latency on-device inference.
11. **SMTP is optional.** Alert email falls back to `./data/alerts.log` when SMTP is not configured. Production should wire Postmark / SendGrid / in-cluster SMTP.
12. **No auth on the HTTP boundary.** When this becomes a service, add bearer-token auth matching MDK's HRPC pattern.

---

## 17. References

<a id="ref-1"></a>[1] Fang et al. *Large Language Models on Tabular Data — A Survey.* arXiv:2402.17944 (Feb 2024) — numerical precision and tabular reasoning limits. https://arxiv.org/html/2402.17944v1

<a id="ref-2"></a>[2] Jin et al. *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.* ICLR 2024 — LLMs lack temporal priors. https://proceedings.iclr.cc/paper_files/paper/2024/file/680b2a8135b9c71278a09cafb605869e-Paper-Conference.pdf

<a id="ref-3"></a>[3] Li, Liu et al. *TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models.* arXiv:2506.18421 (Jun 2025) — aggregation / multi-hop numerical reasoning gaps. https://arxiv.org/abs/2506.18421

<a id="ref-4"></a>[4] *AnoLLM: Large Language Models for Tabular Anomaly Detection.* ICLR 2025 — reduce to structured features first. https://assets.amazon.science/f3/e4/9033ae94402eb468072da852f55c/anollm-large-language-models-for-tabular-anomaly-detection.pdf

<a id="ref-5"></a>[5] *Tabular Data Understanding with LLMs.* arXiv:2508.00217 (Aug 2025) — tool-augmented approaches outperform direct prompting. https://arxiv.org/pdf/2508.00217

<a id="ref-6"></a>[6] Tether MDK Architecture — https://docs.mdk.tether.io/architecture/

<a id="ref-7"></a>[7] *Query the Gemma 4 31B Instruct API* — NVIDIA NIM docs. https://docs.nvidia.com/nim/vision-language-models/latest/examples/gemma-4-31b-it/api.html

<a id="ref-8"></a>[8] *NVIDIA NIM API Explained: Free AI Inference in 2026.* https://decodethefuture.org/en/nvidia-nim-api-explained/

<a id="ref-9"></a>[9] *Introducing Gemma 3 270M: the compact model for hyper-efficient AI.* Google Developers Blog, 2025. https://developers.googleblog.com/en/introducing-gemma-3-270m/

<a id="ref-10"></a>[10] *Tether Launches Open-Source MiningOS.* Bitcoin Magazine, 2026-02-03. https://bitcoinmagazine.com/news/tether-launches-open-bitcoin-mining-system

<a id="ref-11"></a>[11] Tether MDK Documentation — https://docs.mdk.tether.io

<a id="ref-12"></a>[12] Tether MOS — https://mos.tether.io/

<a id="ref-13"></a>[13] *Bitcoin miners get an open-source alternative as Tether launches MiningOS.* CoinDesk, 2026-02-03. https://www.coindesk.com/tech/2026/02/03/bitcoin-miners-get-an-open-source-alternative-as-tether-launches-miningos

<a id="ref-14"></a>[14] Gio Galt. *Introduction to Bitcoin Mining.* CUBO+ 2026 presentation.

<a id="ref-15"></a>[15] Gio Galt. *Bitcoin Mining Economics.* CUBO+ 2026 presentation (slides 8–12: Hash Price vs Hash Cost; slide 11: Hash Cost measurement).

<a id="ref-16"></a>[16] Bitmain Antminer S21 Pro specifications — https://miningnow.com/asic-miner/bitmain-antminer-s21-pro-234th-s/

<a id="ref-17"></a>[17] Bitmain Antminer S21 XP specifications — https://miningnow.com/asic-miner/bitmain-antminer-s21-xp-270th-s/

<a id="ref-18"></a>[18] MicroBT WhatsMiner M60S specifications — https://hashrateindex.com/rigs/microbt-whatsminer-186-m60s

<a id="ref-19"></a>[19] Stable-Baselines3 documentation — https://stable-baselines3.readthedocs.io/

<a id="ref-20"></a>[20] Gemma 3 on Ollama — https://ollama.com/library/gemma3:270m

<a id="ref-21"></a>[21] Gemma model family overview — https://ai.google.dev/gemma/docs/core

<a id="ref-22"></a>[22] DuckDB documentation — https://duckdb.org/docs/

<a id="ref-23"></a>[23] XGBoost documentation — https://xgboost.readthedocs.io/

<a id="ref-24"></a>[24] SHAP (SHapley Additive exPlanations) — https://shap.readthedocs.io/

---

## Appendix A — Environment Variables

Full list. See [.env.example](../.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Env tag for logging / feature flags |
| `LOG_LEVEL` | `INFO` | Loguru verbosity |
| `DATA_DIR` | `./data` | Base data directory |
| `RAW_DATA_DIR` | `./data/raw` | Synthetic Parquet output |
| `PROCESSED_DATA_DIR` | `./data/processed` | Exported features |
| `MODELS_DIR` | `./data/models` | Trained model artefacts (joblib / ONNX / torch / SB3 zip) |
| `DUCKDB_PATH` | `./data/mining.duckdb` | Embedded DB file |
| `FLEET_SIZE` | `50` | Default miners in CLI generator |
| `SIMULATION_DAYS` | `30` | Default days of history |
| `SAMPLE_INTERVAL_MINUTES` | `1` | Telemetry sampling cadence |
| `FAILURE_INJECTION_RATE` | `0.15` | Share of miners with injected pre-failures |
| `NVIDIA_API_KEY` | — | NIM API key (`nvapi-...`). Optional if the AI Assistant tab is unused. |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `google/gemma-4-31b-it` | Model id (switchable to local Ollama names) |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Max completion tokens |
| `SMTP_HOST`, `SMTP_PORT` | —, `587` | Optional — alerts fall back to `./data/alerts.log` if unset |
| `SMTP_USER`, `SMTP_PASS` | — | SMTP credentials |
| `OPERATOR_EMAIL` | — | Destination address for alerts |

---

## Appendix B — ASIC Specifications

Source registry: [app/data/asic_specs.py](../app/data/asic_specs.py). Each miner is a `@dataclass(frozen=True)` with full manufacturer datasheet fields.

| Model | Hashrate (TH/s) | Power (W) | Efficiency (J/TH) | Hashboards × chips | Nominal clock (MHz) | Nominal voltage (mV) | Source |
|---|---|---|---|---|---|---|---|
| Antminer S21 | 200 | 3,500 | 17.5 | 3 × 86 | 500 | 340 | [[16](#ref-16)] |
| Antminer S21 Pro | 234 | 3,510 | 15.0 | 3 × 90 | 530 | 330 | [[16](#ref-16)] |
| Antminer S21 XP | 270 | 3,645 | 13.5 | 3 × 90 | 550 | 320 | [[17](#ref-17)] |
| WhatsMiner M60S | 186 | 3,441 | 18.5 | 3 × 78 | 480 | 350 | [[18](#ref-18)] |

The registry `ASIC_REGISTRY` assigns a mix-weighted probability to each model when initialising the fleet: S21 20%, S21 Pro 35%, S21 XP 20%, M60S 25%. `test_safety.py::TestASICSpecs::test_efficiency_matches_power_over_hashrate` asserts `|P/H − spec.efficiency_jth| < 0.5` for every model.

---

## Appendix C — Directory Structure

```
planb-tether-mdk-mining-ai/
├── README.md                      ← user-facing quickstart
├── LICENSE                        ← Apache 2.0
├── pyproject.toml                 ← dependency manifest + hatchling build
├── requirements.txt               ← mirror of pyproject deps
├── .env.example                   ← full env var template
├── app/
│   ├── run_all.py                 ← CLI: full pipeline end-to-end
│   ├── config.py                  ← Pydantic Settings (thermal/voltage/KPI constants)
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── llm_client.py          ← OpenAI-compatible client (NIM / Ollama / vLLM)
│   │   ├── tools.py               ← 5 tools exposed to the LLM
│   │   └── agent.py               ← tool-calling loop (FleetAgent)
│   ├── control/
│   │   └── decision_engine.py     ← 3-tier safety gate (@dataclass commands)
│   ├── dashboard/
│   │   └── dashboard.py           ← Streamlit app (6 tabs)
│   ├── data/
│   │   ├── asic_specs.py          ← S21 / S21 Pro / S21 XP / M60S registry
│   │   └── generator.py           ← physics-grounded synthetic telemetry
│   ├── models/
│   │   ├── anomaly_detector.py    ← LSTM Autoencoder (PyTorch)
│   │   ├── isolation_forest.py    ← sklearn IF, 200 estimators
│   │   ├── failure_classifier.py  ← XGBoost multi-class + SHAP + ONNX export
│   │   ├── health_score.py        ← Ensemble 0.4·AD + 0.6·FC
│   │   └── train_models.py        ← End-to-end ML training
│   ├── pipeline/
│   │   ├── ingestion.py           ← Parquet → DuckDB with Pydantic bounds
│   │   ├── features.py            ← DuckDB window SQL (5m/1h, z-scores)
│   │   └── kpi.py                 ← TE / ETE / Profit Density
│   └── rl/
│       ├── mining_env.py          ← Gymnasium env, 15 discrete actions
│       └── train_agent.py         ← PPO via Stable-Baselines3
├── tests/
│   ├── test_config.py             ← Settings + thresholds
│   ├── test_safety.py             ← Thermal ordering + ASIC registry
│   ├── test_decision_engine.py    ← Safety > AI > Operator, rate limiting
│   ├── test_features.py           ← DuckDB feature correctness
│   ├── test_generator.py          ← Synthetic data bounds
│   ├── test_kpi.py                ← Hand-calculated KPI verification
│   └── test_models.py             ← IF + Health Score integration
└── docs/
    ├── technical_report.md        ← THIS DOCUMENT
    └── architecture.mmd           ← Mermaid layered architecture
```

---

*End of report. Open issues tracked in the repository README's "Future Work" section mirror the items in §16.*
