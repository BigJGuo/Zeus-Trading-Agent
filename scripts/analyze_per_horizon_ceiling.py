"""Find max achievable hit rate per horizon under a joint train/eval ≥ 0.60 constraint.

For each OOS parquet file in the multi_horizon_pnl run:
  1. Split into chronological 70/30 train/eval slices
  2. Scan a dense grid of (rank_floor, mag_q) gates
  3. Record combinations where BOTH slices clear 0.60 with sufficient n
  4. Also find the gate that maximizes eval PnL under the joint constraint
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def scan(df: pd.DataFrame, label: str) -> list[dict]:
    df = df.dropna(subset=["pred", "actual"]).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # chronological 70/30 split
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx].copy()
    eval_ = df.iloc[split_idx:].copy()

    def add_rank(d):
        d = d.copy()
        d["date"] = d["ts"].dt.date
        d["rank_pct"] = d.groupby("date")["pred"].rank(pct=True)
        d["mag"] = d["pred"].abs()
        return d

    train = add_rank(train)
    eval_ = add_rank(eval_)
    mag_grid = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
    rank_grid = [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
    cost = 10 / 10000.0  # 10 bps

    # magnitude cutoffs come from train distribution
    mag_cutoffs = {q: train["mag"].quantile(q) for q in mag_grid}

    best_eval_pnl = None
    best_eval_hit = None
    qualifying = []
    for rf in rank_grid:
        for mq in mag_grid:
            cut = mag_cutoffs[mq]
            tr_kept = train[(train["mag"] >= cut) & (train["rank_pct"] >= rf) & (train["pred"] > 0)]
            ev_kept = eval_[(eval_["mag"] >= cut) & (eval_["rank_pct"] >= rf) & (eval_["pred"] > 0)]
            if len(tr_kept) < 200 or len(ev_kept) < 50:
                continue
            tr_hit = float((np.sign(tr_kept["pred"]) == np.sign(tr_kept["actual"])).mean())
            ev_hit = float((np.sign(ev_kept["pred"]) == np.sign(ev_kept["actual"])).mean())
            tr_pnl = float((tr_kept["actual"] - cost).mean())
            ev_pnl = float((ev_kept["actual"] - cost).mean())
            row = {"variant": label, "rank": rf, "mag_q": mq,
                   "n_tr": len(tr_kept), "n_ev": len(ev_kept),
                   "tr_hit": tr_hit, "ev_hit": ev_hit,
                   "tr_pnl": tr_pnl, "ev_pnl": ev_pnl}
            if tr_hit >= 0.60 and ev_hit >= 0.60:
                qualifying.append(row)
                if best_eval_pnl is None or ev_pnl > best_eval_pnl["ev_pnl"]:
                    best_eval_pnl = row
            if best_eval_hit is None or ev_hit > best_eval_hit["ev_hit"]:
                best_eval_hit = row
    return qualifying, best_eval_pnl, best_eval_hit


def main() -> int:
    oos_dir = Path("artifacts/multi_horizon_pnl_20260421_005905/oos_predictions")
    files = sorted(oos_dir.glob("*.parquet"))
    summary = []
    for f in files:
        df = pd.read_parquet(f)
        qualifying, best_pnl, best_hit = scan(df, f.stem)
        print(f"\n=== {f.stem} ===")
        print(f"qualifying (both slices ≥0.60): {len(qualifying)}")
        if best_pnl is not None:
            print(f"best qualifying by eval_pnl: rank={best_pnl['rank']}, mag_q={best_pnl['mag_q']}, "
                  f"n_tr={best_pnl['n_tr']}, n_ev={best_pnl['n_ev']}, "
                  f"tr_hit={best_pnl['tr_hit']:.4f}, ev_hit={best_pnl['ev_hit']:.4f}, "
                  f"tr_pnl={best_pnl['tr_pnl']:.4f}, ev_pnl={best_pnl['ev_pnl']:.4f}")
        if best_hit is not None:
            print(f"max eval_hit (no joint constraint): rank={best_hit['rank']}, mag_q={best_hit['mag_q']}, "
                  f"n_tr={best_hit['n_tr']}, n_ev={best_hit['n_ev']}, "
                  f"tr_hit={best_hit['tr_hit']:.4f}, ev_hit={best_hit['ev_hit']:.4f}")
        summary.append({"file": f.stem, "n_qualifying": len(qualifying),
                        "max_ev_hit": best_hit["ev_hit"] if best_hit else float("nan"),
                        "max_ev_hit_tr_hit": best_hit["tr_hit"] if best_hit else float("nan")})
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
