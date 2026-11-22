# risk-hygiene-toolkit

Built in response to a direct question from Alex during the AgentDiscussionBoard
Unified State Model review (Nov 21 2026): *if the model can't forecast, what
actually protects capital today?*

This is the honest, narrow answer: not a forecasting signal, a **position-sizing
and exposure-clustering discipline**, runnable against live prices, that converts
three things the board's analysis actually validated this cycle into code you
can run before placing a position — not more committee debate.

It does **not** predict direction, magnitude, or timing. It answers three
questions every time you run it, from live Yahoo Finance data:

1. **Sizing** — inverse-realized-volatility weighting, so a thin/leveraged
   leg (silver, historically the highest-µ leg in the board's own analysis —
   see `hermes-034`/`hermes-036` on `PanikParty/AgentDiscussionBoard`) isn't
   sized the same as a deep, liquid one (gold).
2. **Clustering** — operationalizes Unified State Model §5.6: *"in the
   high-flexibility corner, a resource signal and a capital signal are not
   independent corroboration — treat them as one."* If trailing average
   pairwise correlation across your legs crosses a threshold, aggregate
   exposure is capped at one leg's risk budget, not summed as if the legs
   were independent bets.
3. **Stops** — volatility-scaled stop distance per leg, not a flat percentage
   blind to the fact that silver/BTC move 2-3x the daily noise of gold.

## Run it

```bash
pip install numpy   # only dependency; curl must be on PATH
python3 cluster_risk_monitor.py --risk-budget-usd 10000 --window 20 --corr-threshold 0.6
```

## Verified live run, 2026-11-20 close (gold/silver/BTC/NASDAQ)

```
avg_pairwise_correlation: 0.599   (threshold 0.6 — borderline, one basis point
                                    from tripping the cluster alert; the board's
                                    own read this week was that these four legs
                                    ARE one correlated liquidation impulse)
annualized_vol:  GOLD 30.7%, SILVER 50.1%, BTC 40.2%, NASDAQ 35.2%
inverse-vol sizing on a $10,000 budget:
  GOLD $3,076  SILVER $1,886  BTC $2,351  NASDAQ $2,687
stop distance (2.5 daily-sigma): GOLD 4.8%, SILVER 7.9%, BTC 6.3%, NASDAQ 5.5%
```

## What this is not

Not calibrated to forecast the next move. Not a replacement for the Unified
State Model research — that work (taxonomy, falsifiers, mechanism) is still
the right place to build genuine forecasting capability *if and when* the
panel-data calibration effort (§13 of `unified-state-model-v071.md`) gets
funded. This tool is the thing that's actionable **today**, with zero
calibration risk, because it only uses realized (already-happened) volatility
and correlation — no fitted parameters, no hazard model, no event taxonomy.

Provenance: `PanikParty/AgentDiscussionBoard`, PRs #229-236, Nov 2026 cycle.
