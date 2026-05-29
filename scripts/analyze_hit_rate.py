"""Diagnostic: find a filter that pushes raw hit rate ≥ 0.60 on OOS predictions.

Scans multiple OOS parquet files (per spec/horizon/variant) and for each:
  1. |pred| magnitude quantile gates at increasing coverage floors
  2. Cross-sectional per-date rank gates
  3. Combined |pred| floor + per-date top-K

Goal: identify (spec, horizon, variant, gate) that delivers hit ≥ 0.60 with
enough samples to be a deployable signal, not noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_file(oos_path: Path, label: str) -> list[dict]:
    df = pd.read_parquet(oos_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    df = df.dropna(subset=["pred", "actual"])

    def hit_of(s: pd.DataFrame) -> float:
        if len(s) == 0:
            return float("nan")
        return float((np.sign(s["pred"]) == np.sign(s["actual"])).mean())

    rows = []
    total_n = len(df)
    pooled = hit_of(df)
    rows.append({"variant": label, "gate": "pooled_all", "n": total_n, "hit": pooled})
    rows.append({"variant": label, "gate": "pooled_long", "n": int((df["pred"] > 0).sum()),
                 "hit": hit_of(df[df["pred"] > 0])})

    # |pred| pooled quantile gates
    mag = df["pred"].abs()
    for q in [0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
        cutoff = mag.quantile(q)
        kept = df[mag >= cutoff]
        rows.append({"variant": label, "gate": f"|pred|≥q{q}", "n": len(kept), "hit": hit_of(kept)})

    # Per-date top-K
    for k in [1, 3, 5, 10]:
        keep = df.sort_values("pred", ascending=False).groupby("date").head(k)
        keep = keep[keep["pred"] > 0]
        rows.append({"variant": label, "gate": f"top{k}_per_date", "n": len(keep), "hit": hit_of(keep)})

    # Per-date top-K with |pred| magnitude floor (quantile-based, universal)
    for q in [0.80, 0.90, 0.95, 0.98]:
        cutoff = mag.quantile(q)
        for k in [1, 3, 5]:
            filt = df[mag >= cutoff]
            if filt.empty:
                continue
            keep = filt.sort_values("pred", ascending=False).groupby("date").head(k)
            keep = keep[keep["pred"] > 0]
            rows.append({"variant": label,
                         "gate": f"|pred|≥q{q}_top{k}per_date",
                         "n": len(keep), "hit": hit_of(keep)})

    # Per-date percentile rank gate: top x% each day
    df["pred_rank_in_date"] = df.groupby("date")["pred"].rank(pct=True)
    for rank_floor in [0.95, 0.98, 0.99, 0.995]:
        keep = df[(df["pred_rank_in_date"] >= rank_floor) & (df["pred"] > 0)]
        rows.append({"variant": label, "gate": f"rank_pct≥{rank_floor}", "n": len(keep), "hit": hit_of(keep)})

    # Combined: per-date top percentile AND global magnitude floor
    for rank_floor in [0.95, 0.99]:
        for mag_q in [0.90, 0.95]:
            cutoff = mag.quantile(mag_q)
            keep = df[(df["pred_rank_in_date"] >= rank_floor)
                      & (df["pred"] > 0)
                      & (mag >= cutoff)]
            rows.append({"variant": label,
                         "gate": f"rank≥{rank_floor}+|pred|≥q{mag_q}",
                         "n": len(keep), "hit": hit_of(keep)})

    return rows


def main() -> int:
    artifacts = Path("artifacts")
    sweep_dirs = sorted(artifacts.glob("training_sweep_*"), reverse=True)
    if not sweep_dirs:
        print("no sweep dirs found")
        return 1
    sweep = sweep_dirs[0]
    oos_dir = sweep / "oos_predictions"
    files = sorted(oos_dir.glob("*.parquet"))
    print(f"sweep: {sweep}")
    print(f"found {len(files)} OOS files")
    print()

    all_rows = []
    for f in files:
        # File name format: {spec}_h{horizon}_{variant_tag}.parquet
        label = f.stem
        all_rows.extend(analyze_file(f, label))

    out = pd.DataFrame(all_rows)
    out["hit"] = out["hit"].round(4)
    out = out.sort_values(["hit"], ascending=False)

    # Filter: only show rows that clear 0.55 hit rate AND have n >= 200
    out_good = out[(out["hit"] >= 0.55) & (out["n"] >= 200)].copy()
    print(f"rows clearing hit ≥ 0.55 with n ≥ 200: {len(out_good)}")
    print(out_good.head(60).to_string(index=False))
    print()

    # Top 30 overall
    print("=== TOP 30 hit rates (any n) ===")
    top = out.sort_values("hit", ascending=False).head(30)
    print(top.to_string(index=False))
    print()

    # Focus: rows clearing 0.60 with n >= 100
    print("=== Clears 0.60 with n ≥ 100 ===")
    q60 = out[(out["hit"] >= 0.60) & (out["n"] >= 100)].copy()
    q60 = q60.sort_values(["variant", "n"], ascending=[True, False])
    print(q60.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
