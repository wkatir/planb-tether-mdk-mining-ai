# KPI glossary

## Definitions (computed in `app/pipeline/kpi.py`)

| KPI | Formula | Unit | Better is |
|---|---|---|---|
| TE - True Efficiency | `(P_asic + P_cooling + P_aux) / (H x eta_env x eta_mode)` | J/TH | lower |
| ETE - Economic TE | `(0.024 x TE x energy_price) / (hashprice / 1000)` | ratio | lower; `< ete_breakeven` is profitable |
| PD - Profit Density | `(daily_revenue - daily_cost) / P_total` | $/W/day | higher; negative means losing money |

TE is higher than the nameplate J/TH of the model because it charges the miner
for its share of cooling and auxiliary power and derates for ambient
temperature and operating mode. How much higher depends on the fleet: the
cooling load is shared, so a small fleet carries more cooling per miner than a
large one, and `overclock` miners take the `eta_mode` penalty on top. Do not
judge a TE against the nameplate; compare miners of the same model and mode
within the same fleet.

## ETE bands

Relative to `ete_breakeven` (read its value from "Active thresholds").

| Band | avg ETE | Reading | Option to propose |
|---|---|---|---|
| Profitable | below break-even | Earns more than it costs | Keep. Candidate for overclock only if temperature allows |
| Marginal | break-even to 1.2x | Loses a little; flips with small market moves | Switch to `low_power` mode; re-evaluate when prices change |
| Loss-making | 1.2x to 2x | Loses money in any realistic short-term scenario | Run only in cheap-energy hours, or underclock |
| Deep loss | above 2x | Costs more than double what it earns | Candidate to power down; check for a hardware fault first |

## Why a miner is unprofitable

- **Overclock mode**: `eta_mode` penalizes it (0.85), so TE and ETE rise.
- **Hot site**: `eta_env` drops as ambient temperature rises above 25 C.
- **Degraded hardware**: lost hashrate at the same power raises TE directly.
- **Market**: high energy price or low hashprice moves the whole fleet at once.
  If almost every miner is above break-even, it is the market, not the miners.
