# Archivos de entrada

| Archivo | Qué es | Cómo se usa |
|---|---|---|
| `kpi_sample.csv` | 1 544 filas de KPIs ya calculados: 8 mineros, 48 horas, una lectura cada 15 minutos. | `--demo` en el CLI y fixture de `tests/test_agent_skills.py` |
| `export_sample.sql` | Consulta exacta con la que se exportó `kpi_sample.csv`, y los comandos para regenerarlo. | Reproducir la muestra |
| `severity_payload.json` | Entrada de ejemplo para el script de la skill `fleet-triage` (los mineros en riesgo de la muestra). | `python skills/fleet-triage/scripts/rank_severity.py examples/severity_payload.json` |
| `questions.md` | Preguntas de ejemplo por agente, con la skill que cada una debería activar y lo que los datos permiten responder. | Entrada manual para `--ask` |
| `prompts/*.txt` | Salida de `--show-prompt --demo` para cada agente: el system prompt real y las tools expuestas. | Ver la personalización sin instalar nada |

## Qué contiene la muestra

Sale de una corrida de 30 días con fallas inyectadas
(`FAILURE_INJECTION_RATE=0.4`, 8 mineros). La ventana exportada va del
2026-04-22 13:00 al 2026-04-24 13:00 y termina en un momento con casos reales
para las skills:

| Minero | Situación en la muestra | Skill que lo ejercita |
|---|---|---|
| `miner_007` | Falla térmica en curso. Promedio 80.5 °C (P2), pero última lectura 98.2 °C, sobre `temp_emergency_c`. | fleet-triage, miner-diagnosis, operator-alert |
| `miner_000` | Falla de hashboard en curso y en overclock: 158 TH/s frente a 270 nominales, 86 °C. No aparece en la lista de riesgo porque su promedio queda bajo el umbral. | miner-diagnosis |
| `miner_002` | Sano pero en overclock: promedio 80.1 °C (P2) y ETE 1.35. | fleet-triage, profitability-analysis |
| resto | Operación normal; 6 de 8 mineros son rentables. | profitability-analysis |

Columnas: `timestamp, device_id, model, operating_mode, asic_hashrate_th,
asic_power_w, chip_temperature_c, fan_speed_rpm, error_count,
true_efficiency_jth, economic_te, profit_density`. Son las únicas que la capa de
agentes lee; la telemetría cruda nunca llega al LLM.
