# Preguntas de ejemplo por agente

Ejecutar con `python -m app.ai.cli --agent <agente> --ask "<pregunta>" --demo`.

## copilot

| Pregunta | Skill esperada |
|---|---|
| What needs my attention right now? | fleet-triage |
| Give me a shift handover summary. | fleet-triage |
| How many miners are losing money? | profitability-analysis |
| Send an alert about miner_007. | ninguna: debe decir que no tiene esa capacidad |

## maintenance

| Pregunta | Skill esperada (vía `load_skill`) |
|---|---|
| Rank the at-risk miners. | fleet-triage (+ `scripts/rank_severity.py`) |
| Why is miner_007 running hot? | miner-diagnosis |
| miner_000 looks bad. Diagnose it and alert the operator if it cannot wait. | miner-diagnosis, luego operator-alert |
| Is miner_004 profitable? | ninguna: debe remitir al agente finance |

## finance

| Pregunta | Skill esperada |
|---|---|
| Which miners should we keep running? | profitability-analysis |
| Why is miner_000 so unprofitable? | profitability-analysis (+ `get_device_status`) |
| How much money do we lose per month? | profitability-analysis: debe negarse a convertir a dólares |
| Underclock miner_003. | ninguna: no tiene `recommend_action` |
