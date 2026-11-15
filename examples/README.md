# Archivos de entrada

| Archivo | Qué es | Cómo se usa |
|---|---|---|
| `kpi_sample.csv` | 768 filas de KPIs ya calculados (8 mineros, 1 día, una lectura cada 15 min). Exportado de la tabla `kpi` tras `python -m app.run_all --fleet-size 8 --days 1 --skip-training`. | `--demo` en el CLI y fixture de `tests/test_agent_skills.py` |
| `questions.md` | Preguntas de ejemplo por agente, con la skill que cada una debería activar. | Entrada manual para `--ask` |
| `severity_payload.json` | Entrada de ejemplo para el script de la skill `fleet-triage`. | `python skills/fleet-triage/scripts/rank_severity.py < examples/severity_payload.json` |

Columnas de `kpi_sample.csv`: `timestamp, device_id, model, operating_mode,
asic_hashrate_th, asic_power_w, chip_temperature_c, fan_speed_rpm, error_count,
true_efficiency_jth, economic_te, profit_density`. Son las únicas que la capa de
agentes lee; la telemetría cruda nunca llega al LLM.
