"""Last-resort ceiling: calibrate gate on FULL OOS (in-sample for gate, OOS for model)
with cross-horizon anchor agreement. If h=2/h=3 day trades cannot hit 0.60 this way,
the feature ceiling is truly below the user's target for short horizons.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load_oos(h: int) -> pd.DataFrame:
    p = Path("artifacts/multi_horizon_pnl_20260421_005905/oos_predictions") / f"ensemble_h{h}_return.parquet"
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.dropna(subset=["pred", "actual"])
    df["date"] = df["ts"].dt.date
    return df.rename(columns={"pred": f"pred_h{h}", "actual": f"actual_h{h}"})


def main() -> int:
    horizons = [2, 3, 5, 10, 20]
    dfs = [load_oos(h) for h in horizons]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d[["symbol", "ts"] + [c for c in d.columns
                                                     if c.startswith("pred_") or c.startswith("actual_")]],
                              on=["symbol", "ts"], how="inner")
    merged["date"] = merged["ts"].dt.date
    for h in horizons:
        merged[f"mag_h{h}"] = merged[f"pred_h{h}"].abs()
        merged[f"rank_h{h}"] = merged.groupby("date")[f"pred_h{h}"].rank(pct=True)

    cost = 10.0 / 10000.0

    configs = [
        (2, [3, 5, 10, 20]),
        (3, [2, 5, 10, 20]),
        (5, []),
        (10, [5]),
        (20, [5, 10]),
    ]

    rank_grid = [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    mag_grid = [0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995]

    for target, anchors in configs:
        actual = f"actual_h{target}"
        pred = f"pred_h{target}"
        cutoffs = {q: merged[f"mag_h{target}"].quantile(q) for q in mag_grid}
        best = None
        best_pnl = None
        for rf in rank_grid:
            for mq in mag_grid:
                cut = cutoffs[mq]
                mask = (merged[f"mag_h{target}"] >= cut) \
                    & (merged[f"rank_h{target}"] >= rf) \
                    & (merged[pred] > 0)
                for a in anchors:
                    mask = mask & (merged[f"pred_h{a}"] > 0)
                kept = merged[mask]
                if len(kept) < 300:
                    continue
                hit = float((np.sign(kept[pred]) == np.sign(kept[actual])).mean())
                pnl = float((kept[actual] - cost).mean())
                rec = {"rank": rf, "mag_q": mq, "n": len(kept), "hit": hit, "pnl": pnl}
                if best is None or hit > best["hit"]:
                    best = rec
                if hit >= 0.60 and (best_pnl is None or pnl > best_pnl["pnl"]):
                    best_pnl = rec
        print(f"\n=== h={target} (anchors={anchors}) ===")
        if best is not None:
            print(f"  max full-OOS hit: rank={best['rank']}, mag_q={best['mag_q']}, "
                  f"n={best['n']}, hit={best['hit']:.4f}, pnl/trade={best['pnl']:.4f}")
        if best_pnl is not None:
            print(f"  best PnL gate @hit≥0.60: rank={best_pnl['rank']}, mag_q={best_pnl['mag_q']}, "
                  f"n={best_pnl['n']}, hit={best_pnl['hit']:.4f}, pnl/trade={best_pnl['pnl']:.4f}")
        else:
            print(f"  NO GATE clears hit≥0.60 on full OOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
