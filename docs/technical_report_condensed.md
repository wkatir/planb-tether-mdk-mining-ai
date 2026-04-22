# AI-Driven Mining Optimization & Predictive Maintenance — Condensed Report

**Author:** Wilmer Salazar · **Mentor:** Gio Galt (Head of MOS @ Tether) · **Program:** Plan B Network CUBO+ 2026 · **Date:** April 2026

> 4-page summary. The full technical report (with library rationale, CPU-tier analysis, complete Gemma model map, MDK integration contract, and references) lives at [`docs/technical_report.md`](technical_report.md).

## 1. Problem & Thesis

From Gio's *Bitcoin Mining Economics* (slides 8–12): **Gross Profit = (Hash Price − Hash Cost) × Hash Rate**. Hash Price is market-given; the only lever is Hash Cost = `Electricity Cost × Hashing Efficiency`. Plain J/TH misses cooling, auxiliary power, environmental derating, and overclock penalties, so operators optimise a metric that doesn't match the bill they pay.

The PoC delivers a layered controller that (a) computes profitability-aware KPIs, (b) predicts failures and anomalies, (c) proposes clock/voltage actions via RL, and (d) exposes a natural-language assistant — while enforcing a strict **ML-first, LLM-on-top** boundary: the LLM never reads raw telemetry, only pre-computed JSON from the ML layer. This sidesteps the well-documented numerical / tabular / time-series weaknesses of LLMs (TReB 2025, AnoLLM 2025, Time-LLM 2024).

## 2. Architecture (8 Layers)

```
L0 Hardware        ASIC fleet + site sensors
L1 MDK Core (JS)   workers · HRPC · Hyperbee · Holepunch P2P         ← Tether, out of scope
L2 Ingestion       Parquet → Pydantic validation → DuckDB (embedded)
L3 Features        DuckDB window SQL: rolling 5m/1h, z-scores, dT/dt
L4 ML              KPI (TE/ETE/PD) · IF · LSTM-AE · XGBoost · PPO
L5 LLM Assistant   Gemma 4 31B via NVIDIA NIM · 5-tool agent
L6 Decision Engine Safety > AI > Operator (hard thermal/voltage limits)
L7 Outputs         control cmds → MDK workers · Streamlit · ONNX
```

Each layer has a narrow contract. Swapping L5 from NIM cloud to local Ollama changes one env var. Swapping L2 DuckDB for Hyperbee changes one class. **Python sits *adjacent* to MDK, not inside it**: language/crash isolation, auditable HTTP schema, ONNX export for edge inference in `onnxruntime-node` inside JS workers.

## 3. KPIs — TE, ETE, PD

**True Efficiency:** `TE = (P_asic + P_cooling_alloc + P_aux) / (H · η_env · η_mode)` in J/TH, with `η_env = max(0.70, 1 − 0.008·(T_amb − 25))` and `η_mode ∈ {normal 1.00, low_power 1.10, overclock 0.85}`.

Worked S21 Pro @ 28 °C ambient, normal mode: `P_total = 3,825.9 W`, `η_env = 0.976` → **TE = 16.75 J/TH** vs. nameplate 15.0 (the 1.75 delta is real operator cost plain J/TH hides).

**Economic TE:** `ETE = (0.024 · TE · energy_price) / (hashprice/1000)` — dimensionless. `<1` profitable, `=1` breakeven, `>1` bleeding money. Same miner at $0.04/kWh, hashprice $50/PH/s/day → **ETE = 0.32**, earns ~3× electricity cost.

**Profit Density:** `PD = (daily_revenue − daily_cost) / P_total` in $/W/day — the only metric that compares heterogeneous miners fairly per watt purchased.

## 4. ML Layer

| Model | Role | Why |
|---|---|---|
| Isolation Forest (200 est.) | Fast global outlier | Runs on CPU, first-pass detector |
| LSTM Autoencoder (PyTorch) | Temporal drift / degradation | Optional — catches what IF misses |
| XGBoost multi-class + SHAP | Failure type: thermal / hashboard / PSU | Tabular SOTA + explanations + ONNX export |
| Health Score | Single 0–1 number | `1 − (0.4·anomaly + 0.6·max_failure_prob)` weights actionable over exploratory |
| PPO (Stable-Baselines3) | Clock × voltage optimiser | 15 discrete actions, reward = revenue − energy − thermal penalty |

The `MiningEnv` pulls its baselines directly from `ASICSpec` (nominal clock, voltage, hashrate, power) — physics is unified with the synthetic generator (`P ∝ V²·f`, RC thermal model).

## 5. LLM Layer — Gemma 4 31B on NVIDIA NIM

No GPU, 3-week PoC → NVIDIA NIM's OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`, 1000 free credits on signup, `nvapi-` key) hosts `google/gemma-4-31b-it`. The same `openai` SDK talks to local Ollama / vLLM / self-hosted NIM by changing one env var — the portability story is intact.

The LLM calls 5 Python tools; all return JSON already reduced by the ML layer:

1. `get_fleet_summary()` — aggregate KPIs
2. `get_device_status(device_id)` — latest miner state
3. `list_miners_at_risk(limit)` — pre-ranked by severity
4. `recommend_action(device_id)` — soft advisory
5. `send_operator_alert(device_id, subject, body)` — SMTP or local log

**Forbidden by system prompt:** compute KPIs from raw numbers, bypass the Decision Engine, issue hardware commands. Hardware actions only flow through L6.

**Local deployment tiers** (compute you have → Gemma you run):

| Hardware | Recommended Gemma | Notes |
|---|---|---|
| CPU-only laptop (this PoC) | NIM cloud + optional Gemma 3 270M INT4 for offline demo | 300 MB RAM, 15–30 tok/s |
| Laptop 8 GB GPU | Gemma 3 4B INT4 (Ollama) | 30–60 tok/s, solid tool-calling |
| Desktop 12–24 GB GPU | Gemma 3 12B / 27B INT4 | Production-grade local |
| ≥ 48 GB GPU | Gemma 4 27B fp16 or fine-tuned domain model | Operator-corpus fine-tune |

## 6. Safety — Decision Engine

Priority **Safety > AI > Operator**, implemented as immutable `@dataclass` commands in [`app/control/decision_engine.py`](../app/control/decision_engine.py).

| Hard constraint | Threshold | Action |
|---|---|---|
| Chip temperature | ≥ 78 °C | Throttle (−20%) |
| Chip temperature | ≥ 95 °C | Emergency shutdown |
| Voltage deviation | > ±10 % | Underclock + alert |
| Command rate | < 5 min / device | Queue / reject |
| Fleet overclock | > 20 % simultaneous | Block new overclocks |
| AI step size | > 5 % clock change | Clip to 5 % |

The priority ordering, rate limit, and step-limit behaviour are locked by 10 regression tests. `pytest tests/test_{config,safety,decision_engine}.py` → **26/26 passing**.

## 7. Data Pipeline & Dashboard

Synthetic generator encodes CMOS dynamic power, RC thermal response, age-based degradation, injected thermal/hashboard/PSU failures, diurnal ambient temperature, tiered energy prices, and GBM hashprice. Ingestion validates physical bounds with Pydantic; DuckDB window functions compute rolling features and z-scores; the KPI engine writes TE/ETE/PD.

The Streamlit dashboard (6 tabs) includes a **Synthetic Data** tab that regenerates fleets interactively (fleet size 5–200, days 1–30, failure rate 0–30%, seed, optional ML training) and invalidates the query cache on completion — no CLI context switch needed during review. The **AI Assistant** tab chats with Gemma via NIM and shows tool-call traces.

## 8. MDK / MOS Integration

**MDK** is a JavaScript open standard [docs.mdk.tether.io](https://docs.mdk.tether.io) — device workers → orchestrator → API layer, Hyperbee time-series, HRPC over Holepunch. **MOS** is the Apache-2.0 reference implementation launched at Plan B Forum, San Salvador, Feb 2026.

Python sits *adjacent* for four reasons: (1) language isolation keeps MDK's lean JS core lean, (2) crash isolation keeps an LLM timeout or torch SIGSEGV away from the mining hot path, (3) MDK's security model prefers explicit HTTP/JSON interfaces over in-tree imports, (4) ONNX export lets inference migrate into MDK workers via `onnxruntime-node` — Python trains, JS infers.

The Python → MDK contract is a fixed JSON schema (`device_id`, `ts`, `kpis{te_jth,ete,pd_usd_per_w_day}`, `health{score,status,predicted_failure,shap_top3}`, `recommendation{action,clock_multiplier,reason,advisory:true}`, `explanation`). The `advisory:true` flag forces every recommendation through the Decision Engine before it can touch hardware.

## 9. Limitations Worth Flagging

- **LSTM-AE trains at batch size 1** — fine for PoC, 20× slower than necessary. `DataLoader(batch_size=32)` is a one-line fix when compute improves.
- **PPO at 100k steps with a simplified reward** — production would use 1–10 M steps with the actual TE/ETE reward and multi-miner joint action spaces.
- **LLM hosted on NIM** — portability to local Ollama is demonstrated (one env var), but the demo depends on the free tier.
- **DuckDB not Hyperbee** — the local-first philosophy matches, but P2P replication requires the Hyperbee migration. Same row schema.
- **Parquet not HRPC** — PoC ingestion boundary; production subscribes to MDK telemetry topics over HRPC.

## 10. Deliverables Checklist

| Assignment ask | Delivered |
|---|---|
| Technical report 2–4 pages | ✅ This condensed report (4 pp). Full report: `docs/technical_report.md`. |
| Functional data pipeline | ✅ `python -m app.run_all` or the dashboard's Synthetic Data tab. |
| Tier 1 — ingestion, KPI design | ✅ TE + ETE + PD (beyond the ask). |
| Tier 2A — RL dynamic control | ✅ PPO + Gymnasium env unified with `ASICSpec`. |
| Tier 2B — predictive maintenance | ✅ IF + LSTM-AE + XGBoost + Health Score + ONNX. |
| Architecture diagram | ✅ `docs/architecture.mmd` (8 layers, Mermaid). |

---

*Report fits on 4 printed A4 pages at 10 pt. Detailed library rationale, full CPU tier analysis, complete Gemma local-deployment map, and all 24 references are in [`technical_report.md`](technical_report.md).*
