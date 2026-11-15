---
name: profitability-analysis
description: Interpret TE, ETE and Profit Density to say which miners make or lose money and what to do about it. Use when the operator asks about profitability, efficiency, energy cost, break-even, or whether to keep a miner running.
allowed-tools: get_fleet_summary list_unprofitable_miners get_device_status
metadata:
  version: "1.0"
  domain: economics
---

# Profitability analysis

Translate efficiency KPIs into a keep / tune / stop decision the operator can defend.

## Procedure

1. Start from the fleet level: `GET_FLEET_SUMMARY` (snapshot, or call
   `get_fleet_summary`) gives `avg_ete`, `profitable` and `loss_making`.
2. Get the loss-making miners from `LIST_UNPROFITABLE_MINERS` (snapshot, or
   call `list_unprofitable_miners`). They come sorted worst first.
3. For each one, classify the gap to `ete_breakeven` with the bands in
   `references/kpi-glossary.md` and pick the matching option.
4. Present a table: `device_id | model | avg ETE | profit density | band | option`.
5. End with the fleet-level takeaway in one sentence.

## Rules

- ETE is a ratio of cost to revenue: **below** `ete_breakeven` is profitable.
  Never describe a high ETE as "good efficiency".
- Profit density is in $/W/day and is small by nature. Report it in scientific
  or 6-decimal form; do not round it to zero.
- Do not convert KPIs into dollars per day or per month. You do not have the
  hashprice or energy price series; only the ML layer does.
- Profitability depends on market inputs that move daily. Phrase conclusions
  as "at the current hashprice and energy price".
- A loss-making miner that is also hot or erroring is a maintenance case first:
  point to it, do not recommend an economic action on top.
