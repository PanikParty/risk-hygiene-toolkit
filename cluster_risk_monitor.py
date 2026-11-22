#!/usr/bin/env python3
"""
cluster_risk_monitor.py

Portfolio-hygiene tool derived directly from the AgentDiscussionBoard Unified
State Model work (Nov 2026 cycle, PRs #229-236). Converts three validated
findings from that exercise into actionable, runnable risk controls -- NOT a
forecasting signal. It does not predict direction, magnitude, or timing.
It answers three questions every session, from live data:

  1. SIZING    - what position size per leg keeps realized-volatility-adjusted
                 risk equal across legs (inverse-vol weighting against a
                 stated risk budget), so a thin, leveraged-prone leg (e.g.
                 silver) is not sized the same as a deep, liquid one (gold).
  2. CLUSTERING - are the legs currently moving as one correlated bet?
                 (operationalizes Unified State Model Section 5.6: "in the
                 high-flexibility corner, a resource signal and a capital
                 signal are not independent corroboration -- treat them as
                 one"). If avg pairwise correlation over the trailing window
                 exceeds a threshold, the tool caps AGGREGATE exposure across
                 the clustered legs to a single leg's risk budget, not N legs'
                 worth, regardless of how diversified the position list looks
                 on paper.
  3. STOPS     - a volatility-scaled stop distance per leg (ATR-style, from
                 realized vol, since this fetches daily closes not full OHLC),
                 so stop discipline is not a fixed percentage blind to how
                 much noisier silver/BTC are than gold/equities.

Data: Yahoo Finance chart API, unauthenticated, no key required.
Usage:
    python3 cluster_risk_monitor.py
    python3 cluster_risk_monitor.py --risk-budget-usd 10000 --window 20 --corr-threshold 0.6
"""
import argparse
import datetime
import json
import subprocess
import sys

import numpy as np

DEFAULT_TICKERS = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "BTC": "BTC-USD",
    "NASDAQ": "^IXIC",
}


def fetch_series(symbol: str, range_: str = "1y") -> dict:
    """Fetch {date: close} from Yahoo Finance chart API via curl (no deps)."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={range_}&interval=1d"
    )
    r = subprocess.run(
        ["curl", "-s", "-m", "20", "-A", "Mozilla/5.0", url],
        capture_output=True, text=True,
    )
    j = json.loads(r.stdout)
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    out = {}
    for t, c in zip(ts, closes):
        if c is None:
            continue
        day = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date()
        out[day] = c
    return out


def load_all(tickers: dict, range_: str = "1y") -> dict:
    series = {}
    for name, sym in tickers.items():
        try:
            series[name] = fetch_series(sym, range_)
        except Exception as e:
            print(f"WARNING: failed to fetch {name} ({sym}): {e}", file=sys.stderr)
    return series


def align(series: dict) -> tuple:
    legs = list(series.keys())
    common_days = sorted(set.intersection(*[set(series[l].keys()) for l in legs]))
    prices = {l: np.array([series[l][d] for d in common_days]) for l in legs}
    return legs, common_days, prices


def annualized_vol(log_returns: np.ndarray, window: int) -> float:
    tail = log_returns[-window:]
    return float(np.std(tail, ddof=1) * np.sqrt(252))


def pairwise_corr_matrix(rets: dict, legs: list, window: int) -> np.ndarray:
    n = len(legs)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = rets[legs[i]][-window:]
            b = rets[legs[j]][-window:]
            c = float(np.corrcoef(a, b)[0, 1])
            M[i, j] = M[j, i] = c
    return M


def run(risk_budget_usd: float, window: int, corr_threshold: float,
        tickers: dict = None, range_: str = "1y") -> dict:
    tickers = tickers or DEFAULT_TICKERS
    series = load_all(tickers, range_)
    legs, days, prices = align(series)
    rets = {l: np.diff(np.log(prices[l])) for l in legs}
    vols = {l: annualized_vol(rets[l], window) for l in legs}

    # 1. Inverse-vol sizing: equal risk contribution per leg, before clustering.
    inv_vol = {l: 1.0 / vols[l] for l in legs}
    total_inv = sum(inv_vol.values())
    raw_weight = {l: inv_vol[l] / total_inv for l in legs}
    raw_usd = {l: raw_weight[l] * risk_budget_usd for l in legs}

    # 2. Clustering: trailing-window pairwise correlation.
    corr = pairwise_corr_matrix(rets, legs, window)
    n = len(legs)
    off_diag = [corr[i, j] for i in range(n) for j in range(n) if i != j]
    avg_corr = float(np.mean(off_diag))
    clustered = avg_corr >= corr_threshold

    if clustered:
        # Aggregate exposure across the clustered legs capped at ONE leg's
        # worth of the risk budget (Sec 5.6: correlated legs are one bet).
        scale = (1.0 / n) / max(raw_weight.values())  # so sum(raw_weight)*scale -> largest single-leg weight
        adj_usd = {l: raw_usd[l] * scale for l in legs}
    else:
        adj_usd = dict(raw_usd)

    # 3. Vol-scaled stop distance (in % of price), using realized daily vol
    #    over the window, not fixed-% stops.
    stop_pct = {l: float(np.std(rets[l][-window:], ddof=1) * 2.5) for l in legs}  # ~2.5 daily sigma

    report = {
        "as_of": str(days[-1]),
        "window_days": window,
        "risk_budget_usd": risk_budget_usd,
        "annualized_vol": {l: round(vols[l], 4) for l in legs},
        "avg_pairwise_correlation": round(avg_corr, 3),
        "correlation_matrix": {legs[i]: {legs[j]: round(corr[i, j], 3) for j in range(n)} for i in range(n)},
        "clustered_as_one_bet": clustered,
        "raw_inverse_vol_sizing_usd": {l: round(raw_usd[l], 2) for l in legs},
        "cluster_adjusted_sizing_usd": {l: round(adj_usd[l], 2) for l in legs},
        "stop_distance_pct": {l: round(stop_pct[l] * 100, 2) for l in legs},
        "last_price": {l: round(float(prices[l][-1]), 4) for l in legs},
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risk-budget-usd", type=float, default=10000.0,
                     help="Total USD risk budget to allocate across legs (default 10000)")
    ap.add_argument("--window", type=int, default=20,
                     help="Trailing window in trading days for vol/correlation (default 20)")
    ap.add_argument("--corr-threshold", type=float, default=0.6,
                     help="Avg pairwise correlation above which legs are treated as one bet (default 0.6)")
    args = ap.parse_args()

    report = run(args.risk_budget_usd, args.window, args.corr_threshold)
    print(json.dumps(report, indent=2))

    if report["clustered_as_one_bet"]:
        print(
            f"\n*** CLUSTER ALERT: avg pairwise corr {report['avg_pairwise_correlation']} "
            f">= threshold {args.corr_threshold}. Legs are moving as ONE bet. "
            f"Aggregate exposure capped accordingly, not summed as if independent. ***",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
