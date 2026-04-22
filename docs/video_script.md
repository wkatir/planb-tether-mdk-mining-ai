# Guión — Video walkthrough MDK Mining AI (8–10 min)

> Para demo day / sustentación. Formato `[tiempo] · [qué mostrar] · [qué decir]` con el **por qué** del assignment entre paréntesis. No leas literal — usá los bullets como punteros.

---

## [0:00 – 0:45] Apertura y contexto

**Mostrar:** Portada del technical_report.md o una slide con el título del proyecto.

**Qué decir (bullets):**
- Proyecto del MDK assignment de Tether, mentoreado por Gio Galt.
- Gio lo resume así: *Gross Profit = (Hash Price − Hash Cost) × Hash Rate*.
- Hash Price es de mercado, el único lever es Hash Cost.
- Mi PoC ataca ese lever con una arquitectura en capas: ML computa los números, LLM los traduce a lenguaje operador.

*(El assignment §1 pide literalmente "bridge the gap between raw hardware telemetry and intelligent automation". Abrir con el framing de Gio demuestra que leíste la economía del negocio, no solo la hoja técnica.)*

---

## [0:45 – 2:00] El problema — por qué J/TH no alcanza

**Mostrar:** Slide del pitch de Gio "Bitcoin Mining Economics" (slide 11 — Hash Cost = Electricity × Efficiency) o el §2 del technical report.

**Qué decir:**
- J/TH del datasheet ignora cooling (5–10% del consumo del sitio), auxiliary power, y el derating ambiental.
- También ignora el penalty del overclock: CMOS escala como `P ∝ V² · f`, así que subir voltaje para más hashrate empeora la eficiencia real.
- Mi KPI **True Efficiency (TE)** incorpora estos factores. Y además definí **ETE** (Economic TE) para saber si el miner es rentable, y **Profit Density (PD)** para comparar modelos heterogéneos.

*(Assignment §3.1.b pide literalmente "a True Efficiency (TE) ratio that goes beyond simple J/TH metrics and incorporates additional real-world variables such as cooling system power consumption, chip voltage, environmental conditions, device operating mode". TE cubre los 4. ETE y PD son "más allá" del ask — originalidad.)*

---

## [2:00 – 3:15] Arquitectura en 8 capas

**Mostrar:** `docs/architecture.mmd` renderizado (o el ASCII de §3 del technical report).

**Qué decir — recorré de abajo hacia arriba:**
- **L0–L1** Hardware y MDK Core (JS de Tether, fuera del scope).
- **L2–L4** El corazón: ingestion con Pydantic → DuckDB → feature engineering SQL → KPI + ML (Isolation Forest, LSTM Autoencoder, XGBoost, PPO).
- **L5** AI Assistant — Gemma vía NVIDIA NIM con tool-calling.
- **L6** Decision Engine con 3-tier safety: Safety > AI > Operador.
- **L7** Outputs — comandos a MDK workers, dashboard, ONNX.
- La tesis que atraviesa todo: **ML-first, LLM-on-top** — el LLM nunca ve filas crudas, recibe JSON ya reducido.

*(Assignment §4.3 pide un architecture diagram "Hardware → Telemetry Pipeline → Feature Processing → AI Controller → Command Execution". Mi diagrama tiene las 5 etapas pero desglosadas en 8 capas con contratos explícitos entre capas. Eso aplica directo al criterio de evaluación "clarity and quality of the technical reasoning".)*

---

## [3:15 – 4:30] Demo en vivo — pipeline end-to-end

**Mostrar:** Terminal corriendo `python -m app.run_all --fleet-size 50 --days 3 --skip-training` y luego el dashboard.

**Qué decir:**
- 50 miners × 3 días = 216,000 filas generadas sintéticamente en ~6 segundos.
- El generator no es ruido aleatorio — tiene física real: modelo térmico RC, escala CMOS `P ∝ V²·f`, degradación por edad +0.5%/mes, inyección de pre-fallos.
- DuckDB es embedded: misma filosofía local-first que Hyperbee (el storage de MDK). Cuando migremos a prod, es reemplazo directo sin cambiar schema.
- KPIs que salen: **TE ≈ 24 J/TH** (vs 15–18 del datasheet — refleja la realidad operativa), **ETE 0.43** (rentable), **PD $0.0024/W/día**.

*(Assignment §3.1.a pide "ingest and structure telemetry data" con clock, voltage, hashrate, temperature, power — los cinco están. Evalúa "quality of the data pipeline" — DuckDB + Parquet + Pydantic es production-grade local-first y lo justifico con el argumento de Hyperbee. "Understanding of mining operations and constraints" queda claro por la física, no por adornos.)*

---

## [4:30 – 6:00] Machine Learning — detección de fallos con explicabilidad

**Mostrar:** Tab "AI Insights" del dashboard (Health score distribution, Miners at risk).

**Qué decir:**
- Tres modelos complementarios:
  - **Isolation Forest** — detector rápido de outliers globales, flagged 4.8% (objetivo 5%).
  - **LSTM Autoencoder** — drift temporal, segundo par de ojos.
  - **XGBoost multiclase** — clasifica el tipo: thermal / hashboard / PSU. Accuracy **98% overall, 99% en healthy**.
- **Health Score** = `1 − (0.4·anomaly + 0.6·max_failure_prob)`. El 0.6 va a failure porque es *accionable* (te dice *qué* falla), el 0.4 a anomaly porque es *exploratorio*.
- SHAP da el "por qué" por predicción. XGBoost exporta a **ONNX** → inferencia en MDK workers JS vía `onnxruntime-node`. Python entrena, JavaScript infiere — cero Python en el hot-path de producción.

*(Assignment §3.2 Option B pide "supervised learning prototype using Random Forest, XGBoost, or LSTM". Tengo los tres. El criterio "awareness of security, safety, and control risks" lo ataco con ONNX — mover inferencia fuera de Python es menos superficie de ataque y menos puntos de falla.)*

---

## [6:00 – 7:15] Control dinámico y capa de seguridad

**Mostrar:** `app/control/decision_engine.py` y `app/rl/mining_env.py`.

**Qué decir:**
- PPO (Stable-Baselines3) sobre `MiningEnv` con 15 acciones discretas (5 clocks × 3 voltages). El env saca sus baselines directo de `ASICSpec` — misma fuente de verdad que el generator.
- Pero el RL **no toca hardware directamente**. Toda acción pasa por el Decision Engine con 3 niveles:
  - Safety (thermal ≥78°C → throttle, ≥95°C → shutdown, voltage ±10%, rate 5min/device, fleet overclock cap 20%)
  - AI (PPO o LLM, recomiendan)
  - Operator (manual)
- La prioridad es **Safety > AI > Operador**. Aún un comando manual pasa por safety.
- Telemetría adversarial: antes de que llegue al RL o al LLM, pasa por Pydantic + IF + LSTM-AE. Si IF y LSTM coinciden en que es anómala, se cuarentena y el agent recibe el last-known-good.

*(Assignment §3.2 Option A pide RL para over/underclocking "while respecting safety and operational constraints". Asignment §4.1 en el Technical Report pide explícitamente "security and safety considerations of an autonomous optimization agent" — 3-tier + adversarial protection responde directo a ese criterio.)*

---

## [7:15 – 8:30] Capa AI — Gemma por NIM, hablando con el fleet

**Mostrar:** Tab "AI Assistant" del dashboard. Escribir algo como "give me the worst 3 miners and why".

**Qué decir:**
- Modelo: Gemma 3 27B vía NVIDIA NIM (OpenAI-compatible, free tier, 0.3s de latencia típica).
- Elegí hosted para el PoC porque no tengo GPU local. El código es portable — cambiar el `LLM_BASE_URL` apunta a Ollama o a un NIM self-hosted sin tocar código.
- El LLM **no recibe filas crudas**. Recibe JSON pre-computado por la ML layer: `get_fleet_summary`, `list_miners_at_risk`. Esto es la tesis *ML-first, LLM-on-top*.
- La referencia académica está en el reporte: TReB (arXiv 2025), Time-LLM (ICLR 2024), AnoLLM (ICLR 2025) — los LLMs tienen debilidades numéricas y temporales documentadas; reducir a features estructuradas antes es la arquitectura correcta.
- Mostrar la respuesta real: identifica miner_006 a 85°C, da contexto, propone acción como **advisory** — no ejecuta nada sin pasar por el Decision Engine.

*(Este bloque no es Tier 1 ni Tier 2 explícito — es originalidad pura. Los criterios de evaluación premian "creativity and originality of the technical or creative approach". La justificación académica de por qué poner LLM encima y no al centro diferencia de cualquier pitch "LLM hace todo".)*

---

## [8:30 – 9:30] MDK alignment y por qué Python "adjacent"

**Mostrar:** §12 del technical report (la tabla de integration boundaries).

**Qué decir:**
- MDK es JS, Holepunch P2P, Hyperbee, HRPC. No vamos a meter PyTorch ahí.
- Mi servicio Python vive *adjacent*: HTTP/JSON con schema fijo al MDK worker, crash isolation, security auditable, y el upgrade path es ONNX — inferencia migra a `onnxruntime-node` dentro de los workers.
- El contrato JSON (§12.4 del reporte): device_id, ts, kpis, health, recommendation con flag `advisory:true`. El MDK worker decide si ejecuta.
- En producción DuckDB → Hyperbee, Parquet → HRPC. Cero cambios de lógica.

*(El assignment entrega bonus implícito por entender el stack de Tether. Menciona docs.mdk.tether.io como recurso. Demuestro que leí MDK / MOS y que el diseño respeta su security model — "explicit interfaces, authenticated RPC, least privilege".)*

---

## [9:30 – 10:00] Cierre y limitaciones honestas

**Mostrar:** §16 del technical report (Limitations).

**Qué decir:**
- Lo que quedó escaldado: LSTM-AE entrena con batch=1 (fix de una línea cuando tenga GPU), PPO corre 100k steps (prod sería 10M), LLM es cloud (portabilidad a Ollama está demostrada con env var).
- Lo que funciona: 51/51 tests verdes, pipeline end-to-end reproducible en <10s, XGBoost 98% accuracy, agent LLM con respuestas reales.
- Lo que entregaría siguiente: migración Hyperbee, ingestion HRPC live, heat-reuse como revenue en ETE, fine-tune Gemma con corpus propio.
- Cierre: *Gross Profit = (Hash Price − Hash Cost) × Hash Rate*. Este PoC demuestra cómo atacar Hash Cost con ingeniería en cada una de las 8 capas, respetando seguridad en cada una.

*(El criterio "clarity and quality of the technical reasoning" se mide también por admitir qué no funcionó. Listar limitaciones honestamente vale más que venderlo perfecto.)*

---

## Q&A — preguntas probables y respuestas cortas

| Pregunta probable | Respuesta corta |
|---|---|
| ¿Por qué DuckDB y no SQLite o PostgreSQL? | Local-first, embedded, en-proceso, window functions vectorizadas. Es el proxy más cercano a Hyperbee en el ecosistema Python. |
| ¿Por qué Gemma si el PDF no lo pide? | El PDF no pide LLM — es originalidad. Gemma porque es open weights, Apache 2.0, alineado con el stance open-source de Tether/MOS. |
| ¿El RL es realmente útil con 100k steps? | Para demostrar reward shaping y safety gating, sí. Para producción, no — necesita 1–10M steps y GPU. Lo declaro en §16 del reporte. |
| ¿Por qué ONNX si ya tenés XGBoost en Python? | Porque MDK es JS. ONNX + onnxruntime-node = inferencia dentro del worker sin Python en el hot-path. Es el único camino real a producción. |
| ¿Y la seguridad de la API key de NIM? | `.env` está gitignored. En prod, secrets manager. Para air-gapped MOS sites, Ollama local — cero dependencia externa. |
| ¿Por qué no entrenaste un modelo propio desde cero en vez de usar Gemma hosted? | Tres semanas, sin GPU. El diseño prioriza el boundary (ML computa, LLM traduce); el modelo específico es intercambiable. |
| ¿Qué pasa si Gemma alucina? | Por eso el LLM no emite comandos de hardware. Solo puede llamar tools de lectura y `send_operator_alert` (email). Clock/voltage solo por Decision Engine. |
| ¿Cuál es la ganancia cuantificable? | TE-aware control reduce J/TH de fleet 5–15% en simulación. Predictive maintenance anticipa fallos 12–72h. Energy arbitrage (off-peak overclock, peak underclock) documentado en §8 del reporte. |

---

## Setup previo al demo (5 min antes de grabar)

```bash
# 1. API key en .env
echo "NVIDIA_API_KEY=nvapi-..." >> .env

# 2. Datos frescos
python -m app.run_all --fleet-size 50 --days 3 --skip-training

# 3. Modelos entrenados (IF + XGBoost, rapido)
python -c "from app.models.train_models import train_all; train_all()"

# 4. Dashboard
streamlit run app/dashboard/dashboard.py --server.port 8501
```

Si se va el internet, Streamlit funciona completo menos la tab AI Assistant.
