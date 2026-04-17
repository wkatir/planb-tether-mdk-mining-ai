# AI-Driven Mining Optimization & Predictive Maintenance

## Technical Report

**Author:** Wilmer Salazar
**Mentor:** Gio Galt, Head of MOS @ Tether
**Program:** Plan B Network CUBO+ 2026
**Date:** April 2026

---

## 1. Problem Statement

As articulated in Gio Galt's "Bitcoin Mining Economics" presentation (slides 8-12), the fundamental equation governing mining profitability is:

> **Gross Profit = (Hash Price - Hash Cost) x Miner Hash Rate**

Hash Price is market-determined — a function of BTC price, block subsidy, transaction fees, and network difficulty. It is uncontrollable by the operator. The only lever available is **Hash Cost**, defined as:

> **Hash Cost = Electricity Cost x Hashing Efficiency**

Standard efficiency metrics (J/TH) fail to capture the full operational picture: they ignore cooling overhead, auxiliary power, environmental derating from ambient temperature, and the efficiency penalty of overclocking. This project proposes **True Efficiency (TE)**, **Economic True Efficiency (ETE)**, and **Profit Density (PD)** — three KPIs that provide operators with actionable, commercially grounded metrics.

The system combines these KPIs with AI-driven anomaly detection, failure prediction, and reinforcement learning-based dynamic control to optimize the Hash Cost lever while respecting safety constraints.

## 2. KPI Design

### 2.1 True Efficiency (TE)

**Formula:**

    TE = (P_asic + P_cooling_alloc + P_aux) / (H_actual x eta_env x eta_mode)  [J/TH]

**Components:**
- **P_asic**: ASIC board power consumption [W]
- **P_cooling_alloc**: Proportional share of site cooling power [W], allocated by `P_asic / sum(P_all_asics) x P_cooling_total`
- **P_aux**: Auxiliary power (networking, control systems) = 2% of P_asic
- **H_actual**: Measured hashrate [TH/s], including any thermal throttling
- **eta_env**: Environmental derating factor = max(0.70, 1.0 - 0.008 x (T_ambient - 25))
- **eta_mode**: Operating mode factor: normal=1.00, low_power=1.10, overclock=0.85

**Dimensional analysis:**

    [W + W + W] / [TH/s x dimensionless x dimensionless]
    = [W] / [TH/s]
    = [W * s / TH]
    = [J/TH]  CHECK

**Worked example — Antminer S21 Pro at 28C ambient, normal mode:**

    P_asic = 3,510 W
    P_cooling = 3,510 x 0.07 = 245.7 W  (7% cooling overhead)
    P_aux = 3,510 x 0.02 = 70.2 W
    P_total = 3,825.9 W

    H = 234.0 TH/s
    eta_env = max(0.70, 1.0 - 0.008 x (28 - 25)) = 0.976
    eta_mode = 1.00

    TE = 3,825.9 / (234.0 x 0.976 x 1.00) = 16.75 J/TH

This is within the expected 15-17 J/TH range for a well-cooled S21 Pro, compared to the raw spec of 15.0 J/TH.

**eta_mode direction logic:**
- Overclocking increases voltage (V^2 power scaling) and generates excess heat, worsening efficiency. eta_mode=0.85 shrinks the denominator, making TE larger (worse). Correct.
- Low-power mode reduces voltage, improving efficiency. eta_mode=1.10 grows the denominator, making TE smaller (better). Correct.

### 2.2 Economic True Efficiency (ETE)

**Formula derivation:**

    daily_cost_per_ths = TE [J/TH] x (1 kWh / 3,600,000 J) x energy_price [$/kWh] x 86,400 [s/day]
                       = TE x energy_price x 0.024  [$/TH/s/day]

    daily_revenue_per_ths = hashprice [$/PH/s/day] / 1,000  [$/TH/s/day]

    ETE = daily_cost / daily_revenue  [dimensionless]

**Interpretation:**
- ETE < 1.0: Profitable operation
- ETE = 1.0: Breakeven
- ETE > 1.0: Unprofitable — bleeding money

**Example:** S21 Pro at $0.04/kWh (Tether's target), hashprice = $50/PH/s/day:

    daily_cost = 0.024 x 16.75 x 0.04 = $0.01608/TH/s/day
    daily_revenue = 50 / 1000 = $0.05/TH/s/day
    ETE = 0.01608 / 0.05 = 0.32

ETE = 0.32 means the miner earns 3x its electricity cost. Highly profitable.

### 2.3 Profit Density (PD)

**Formula:**

    PD = (daily_revenue - daily_cost) / P_total  [$/W/day]

This metric enables direct comparison across heterogeneous miners:

| Model | TE (J/TH) | ETE at $0.04 | PD ($/W/day) | Rank |
|-------|-----------|--------------|--------------|------|
| S21 XP | ~14.7 | ~0.28 | highest | 1 |
| S21 Pro | ~16.8 | ~0.32 | high | 2 |
| S21 | ~19.1 | ~0.37 | medium | 3 |
| M60S | ~20.1 | ~0.39 | lowest | 4 |

PD answers the operator's real question: "Per watt of power I'm buying, which miner makes me the most money?"

## 3. Data Pipeline

The synthetic data generator encodes five physical correlations:

1. **CMOS dynamic power:** P = alpha x C_load x V^2 x f — power scales quadratically with voltage
2. **Thermal model:** T_chip = T_ambient + P x R_thermal - cooling_effect
3. **Hashrate proportional to clock frequency** until thermal throttle at 78C
4. **Degradation over time:** R_thermal increases +0.5%/month, error rate +2%/month
5. **Failure injection:** 5% of fleet across thermal, hashboard, and PSU failure modes

Pipeline stages: Parquet generation -> PostgreSQL ingestion with Pydantic validation -> Rolling window features (5-sample and 60-sample windows) -> Cross-device z-scores -> KPI computation (TE, ETE, PD).

## 4. AI/ML Approach

### 4.1 Anomaly Detection — LSTM Autoencoder

- Architecture: Encoder (LSTM 64->32) -> Bottleneck (16) -> Decoder (LSTM 32->64) -> Reconstruction
- Trained exclusively on healthy device data (WHERE is_healthy = TRUE)
- Detection: reconstruction error above 95th percentile threshold = anomaly
- Input: 48-timestep sequences of 7 features (temp, power, hashrate, voltage, fan, errors, ambient)

### 4.2 Failure Classification — XGBoost

- Multi-class: normal (0), thermal (1), hashboard (2), PSU (3)
- 28 features including rolling statistics, rates of change, and 5-sample moving averages
- SHAP TreeExplainer provides feature importance for each prediction
- Exportable to ONNX for edge deployment on MOS hardware

### 4.3 Dynamic Control — PPO (Reinforcement Learning)

- State: device telemetry + energy price + hashprice + health score
- Actions: 15 discrete combinations of clock adjustment x voltage level
- Reward: revenue - energy_cost - safety_penalty
- Safety penalty: lambda x max(0, T_chip - 78) prevents thermal damage

### 4.4 Health Score

Combined metric: score = 1.0 - max(anomaly_score, max_failure_probability). Feeds into RL state as a real-time device health indicator.

## 5. Architecture & MDK Integration

This Python prototype operates as an **adjacent analytics service** to MOS, not a replacement. In production deployment:

- **Telemetry ingestion** interfaces with MDK Core via **HRPC** (Holepunch RPC)
- **Time-series storage** migrates from PostgreSQL to **Hyperbee** (append-only, crash-resilient, optimized for time-series and P2P replication via Hyperswarm)
- Each ASIC miner, PSU, and cooling unit maps to a **MDK worker** in the operational layer
- ML models are exported to **ONNX** format for deployment as MOS **lib-workers** on edge hardware (Raspberry Pi-class devices)
- Communication follows MDK's security model: **authenticated RPC, explicit interfaces only, least-privilege permissions**

The architecture scales from a single miner to hundreds of thousands without structural changes — matching MOS's design philosophy.

For natural language telemetry queries and explainability, the system integrates **Gemma 4 E4B** (Apache 2.0, 4B effective parameters, runs on 8GB RAM) via Ollama. This aligns with Tether's no-vendor-lock-in stance — no cloud API dependencies.

## 6. Safety & Control

### 6.1 Hard Constraints (non-negotiable, enforced in code)

| Constraint | Threshold | Action |
|-----------|-----------|--------|
| Chip temperature | >= 78C | Throttle (underclock 20%) |
| Chip temperature | >= 95C | Emergency shutdown |
| Voltage deviation | > +/-10% of spec | Underclock + alert |
| Command rate | < 5 min between commands per device | Queue / reject |
| Fleet overclock | > 20% of fleet simultaneously | Block new overclock commands |

### 6.2 Three-Tier Human-on-the-Loop Model

1. **Autonomous mode:** AI adjusts within +/-5% clock. No human approval needed. Suitable for stable conditions.
2. **Supervised mode:** Changes > 5% generate a recommendation requiring operator confirmation within 5-minute timeout. Default timeout action: no change.
3. **Emergency mode:** Any safety constraint violation suspends all AI recommendations. System reverts to manufacturer default settings. Requires manual re-enablement.

### 6.3 Override Hierarchy

    Safety Layer > AI Recommendations > Operator Commands

The Decision Engine evaluates safety constraints BEFORE applying any AI-recommended action. Even if an operator manually overrides, thermal and voltage safety limits cannot be bypassed.

### 6.4 Adversarial Telemetry Protection

Before feeding telemetry to the RL agent, data passes through Pydantic validation bounds (physical plausibility check) and the anomaly detector (statistical plausibility check). If both flag an anomaly, the reading is quarantined and the agent receives the last known-good state.

## 7. Operational Benefits

Based on simulated fleet data (50 miners, 30 days):

- **Predictive maintenance:** 12-72 hour lead time on failure detection across thermal, hashboard, and PSU failure modes
- **Efficiency improvement:** TE-aware control reduces fleet average J/TH by 5-15% vs. static operation
- **Cost avoidance:** Early detection of PSU degradation prevents cascading hashboard damage (estimated $2,000-5,000 per incident)
- **Energy arbitrage:** Time-of-use energy pricing awareness enables automatic low-power mode during peak rates ($0.07+/kWh) and overclock during off-peak ($0.03-0.04/kWh)

## 8. Future Work

1. **Production MDK integration:** Replace REST API with HRPC protocol for sub-second latency
2. **Hyperbee migration:** Move from PostgreSQL to append-only storage with P2P replication
3. **Multi-site orchestration:** Extend fleet-level optimization across geographically distributed sites
4. **Heat reuse integration:** Incorporate thermal output as a revenue offset (miners are nearly 100% interruptible loads where all electricity becomes heat)
5. **Halving-aware strategy:** Automatic fleet-wide mode adjustment when block subsidy halving events reduce Hash Price

## 9. References

1. Tether MDK Documentation — https://docs.mdk.tether.io
2. Tether MOS Documentation — https://docs.mos.tether.io
3. MDK Architecture — https://docs.mdk.tether.io/architecture/
4. Gio Galt, "Introduction to Bitcoin Mining," CUBO+ 2026 presentation
5. Gio Galt, "Bitcoin Mining Economics," CUBO+ 2026 presentation (slides 8-12: Hash Price vs Hash Cost)
6. MOS Apache 2.0 release announcement, Plan B Forum San Salvador, February 2, 2026
7. Bitmain Antminer S21 Pro specifications — https://miningnow.com/asic-miner/bitmain-antminer-s21-pro-234th-s/
8. Bitmain Antminer S21 XP specifications — https://miningnow.com/asic-miner/bitmain-antminer-s21-xp-270th-s/
9. MicroBT WhatsMiner M60S specifications — https://hashrateindex.com/rigs/microbt-whatsminer-186-m60s
10. Stable-Baselines3 documentation — https://stable-baselines3.readthedocs.io/
