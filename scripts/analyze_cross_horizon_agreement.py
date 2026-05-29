"""Test if multi-horizon sign-agreement + mag/rank gates can push hit ≥ 0.60
for day trades (h=2,3) and long-term (h=10,20), using h=5 as anchor.

Hypothesis: a prediction is more likely correct when multiple horizons agree.
Trade only when (h_target, h_anchor) both predict UP above mag/rank floors.
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


def add_rank_mag(df: pd.DataFrame, h: int) -> pd.DataFrame:
    df = df.copy()
    df[f"mag_h{h}"] = df[f"pred_h{h}"].abs()
    df[f"rank_h{h}"] = df.groupby("date")[f"pred_h{h}"].rank(pct=True)
    return df


def scan(df: pd.DataFrame, target_h: int, anchors: list[int]) -> dict:
    """Scan rank/mag gates with multi-horizon agreement on target actual returns."""
    cost = 10.0 / 10000.0
    actual_col = f"actual_h{target_h}"
    target_pred = f"pred_h{target_h}"

    # 70/30 chronological split
    df = df.sort_values("ts").reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    eval_ = df.iloc[split_idx:]

    rank_grid = [0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98, 0.99]
    mag_grid = [0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98]

    mag_cutoffs = {q: train[f"mag_h{target_h}"].quantile(q) for q in mag_grid}

    best_joint = None  # best gate satisfying both slices >= 0.60
    best_eval = None

    for rf in rank_grid:
        for mq in mag_grid:
            cut = mag_cutoffs[mq]

            def apply_gate(slice_df):
                mask = (slice_df[f"mag_h{target_h}"] >= cut) \
                    & (slice_df[f"rank_h{target_h}"] >= rf) \
                    & (slice_df[target_pred] > 0)
                # anchors: each must agree in sign AND be in top half (>0)
                for a in anchors:
                    mask = mask & (slice_df[f"pred_h{a}"] > 0)
                return slice_df[mask]

            tr_k = apply_gate(train)
            ev_k = apply_gate(eval_)
            if len(tr_k) < 150 or len(ev_k) < 40:
                continue
            tr_hit = float((np.sign(tr_k[target_pred]) == np.sign(tr_k[actual_col])).mean())
            ev_hit = float((np.sign(ev_k[target_pred]) == np.sign(ev_k[actual_col])).mean())
            tr_pnl = float((tr_k[actual_col] - cost).mean())
            ev_pnl = float((ev_k[actual_col] - cost).mean())
            rec = {"rank": rf, "mag_q": mq, "n_tr": len(tr_k), "n_ev": len(ev_k),
                   "tr_hit": tr_hit, "ev_hit": ev_hit, "tr_pnl": tr_pnl, "ev_pnl": ev_pnl}
            if tr_hit >= 0.60 and ev_hit >= 0.60:
                if best_joint is None or ev_pnl > best_joint["ev_pnl"]:
                    best_joint = rec
            if best_eval is None or ev_hit > best_eval["ev_hit"]:
                best_eval = rec
    return {"joint": best_joint, "max_eval": best_eval}


def main() -> int:
    # Load all horizons and inner-join on (symbol, ts)
    horizons = [2, 3, 5, 10, 20]
    dfs = [load_oos(h) for h in horizons]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d[["symbol", "ts"] + [c for c in d.columns
                                                     if c.startswith("pred_") or c.startswith("actual_")]],
                              on=["symbol", "ts"], how="inner")
    merged["date"] = merged["ts"].dt.date
    for h in horizons:
        merged = add_rank_mag(merged, h)
    print(f"merged rows (intersection of all horizons): {len(merged)}")

    # Test anchor combinations
    targets_and_anchors = [
        (2, [5], "h2 | anchor h5"),
        (2, [3, 5], "h2 | anchors h3,h5"),
        (2, [3, 5, 10], "h2 | anchors h3,h5,h10"),
        (2, [3, 5, 10, 20], "h2 | ALL anchors"),
        (3, [5], "h3 | anchor h5"),
        (3, [2, 5], "h3 | anchors h2,h5"),
        (3, [2, 5, 10], "h3 | anchors h2,h5,h10"),
        (3, [2, 5, 10, 20], "h3 | ALL anchors"),
        (5, [10], "h5 | anchor h10 (baseline check)"),
        (10, [5], "h10 | anchor h5"),
        (10, [5, 20], "h10 | anchors h5,h20"),
        (20, [5, 10], "h20 | anchors h5,h10"),
        (20, [10], "h20 | anchor h10"),
    ]

    for target, anchors, label in targets_and_anchors:
        res = scan(merged, target, anchors)
        bj, be = res["joint"], res["max_eval"]
        print(f"\n=== {label} ===")
        if bj is not None:
            print(f"  JOINT ≥0.60: rank={bj['rank']}, mag_q={bj['mag_q']}, "
                  f"n_tr={bj['n_tr']}, n_ev={bj['n_ev']}, "
                  f"tr_hit={bj['tr_hit']:.4f}, ev_hit={bj['ev_hit']:.4f}, "
                  f"ev_pnl={bj['ev_pnl']:.4f}")
        else:
            print("  JOINT ≥0.60: NONE found")
        if be is not None:
            print(f"  max eval_hit: rank={be['rank']}, mag_q={be['mag_q']}, "
                  f"n_ev={be['n_ev']}, tr_hit={be['tr_hit']:.4f}, ev_hit={be['ev_hit']:.4f}, "
                  f"ev_pnl={be['ev_pnl']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
