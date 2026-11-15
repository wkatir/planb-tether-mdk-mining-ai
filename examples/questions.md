# Preguntas de ejemplo por agente

Ejecutar con `python -m app.ai.cli --agent <agente> --ask "<pregunta>" --demo`.
La columna "Qué deberían mostrar los datos" se obtuvo llamando a las tools sobre
`kpi_sample.csv`; sirve para comprobar si la respuesta del modelo es correcta.

## copilot

| Pregunta | Skill esperada | Qué deberían mostrar los datos |
|---|---|---|
| What needs my attention right now? | fleet-triage | 2 mineros en riesgo, ambos P2 por promedio: `miner_007` (80.5 °C) y `miner_002` (80.1 °C) |
| How many miners are losing money? | profitability-analysis | 2 de 8: `miner_002` (ETE 1.35) y `miner_000` (ETE 1.17) |
| Send an alert about miner_007. | ninguna | Debe decir que no tiene esa capacidad y remitir al agente maintenance |

## maintenance

| Pregunta | Skill esperada (vía `load_skill`) | Qué deberían mostrar los datos |
|---|---|---|
| Rank the at-risk miners. | fleet-triage (+ `scripts/rank_severity.py`) | P1: 0, P2: 2 |
| Why is miner_007 running hot? | miner-diagnosis | Última lectura 98.2 °C, ventilador 4 374 rpm, 4 errores, hashrate nominal: consistente con falla térmica. Acción: `underclock` |
| Diagnose miner_007 and alert the operator if it cannot wait. | miner-diagnosis, luego operator-alert | La lectura supera `temp_warning_c`, así que la alerta procede. Sin SMTP queda en `data/alerts.log` |
| Why is miner_000 slow? | miner-diagnosis | 158 TH/s frente a 270 nominales con 86 °C en overclock: debe nombrar primero el estrangulamiento térmico y declarar confianza media o baja |
| Is miner_004 profitable? | ninguna | Debe remitir al agente finance |

## finance

| Pregunta | Skill esperada | Qué deberían mostrar los datos |
|---|---|---|
| Which miners should we keep running? | profitability-analysis | 6 rentables; `miner_002` en banda "loss-making" (1.2x a 2x) y `miner_000` en "marginal" |
| How much money do we lose per month? | profitability-analysis | Debe negarse a convertir a dólares: no tiene la serie de precios |
| Underclock miner_002. | ninguna | No tiene `recommend_action`; debe decirlo |
