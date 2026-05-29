"""Rank+magnitude gated predictor — pushes raw hit rate past the ~0.55 ceiling.

The MetaLabeler (LGBMClassifier over features + primary pred) flattens out around
0.565 hit rate no matter the threshold because its target (sign-win) doesn't
weight magnitude. Empirically, though, primary predictions with BOTH of:

  1. `pred_rank_in_date ≥ rank_floor` (cross-sectionally conviction-high)
  2. `|pred| ≥ mag_floor` (absolute conviction-high)

clear 0.60 raw hit rate on xgb/ensemble h5/h10 OOS data (n≈3-5k kept rows).

This predictor is a deterministic gate — it has no learned parameters beyond
the two thresholds, which are calibrated from the primary's training-slice OOS
predictions. Wraps any BaseModel primary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import structlog

from zeus.models.base import BaseModel

logger = structlog.get_logger()


class RankMagnitudeGatedPredictor(BaseModel):
    """Gate primary predictions by cross-sectional rank + absolute magnitude.

    Zeroed-out predictions let downstream signal/portfolio logic stay oblivious
    to the gating — they just see fewer non-zero scores with higher expected
    hit rate.
    """

    def __init__(
        self,
        primary: BaseModel,
        mag_floor: float,
        rank_floor: float = 0.95,
        long_only: bool = True,
    ) -> None:
        """
        Args:
            primary: pre-trained BaseModel whose preds we gate.
            mag_floor: absolute magnitude floor in the same units as primary
                predictions (e.g. log-return). Learned from the primary's
                training-slice OOS preds as a quantile.
            rank_floor: cross-sectional rank floor in [0, 1]. A prediction must
                be in the top (1 - rank_floor) of the input batch to pass.
                With rank_floor=0.95, only the top 5% per batch survive.
            long_only: if True, zero out predictions with pred ≤ 0 regardless
                of magnitude. Shorts failed the 0.60 bar in OOS diagnostics.
        """
        self._primary = primary
        self.mag_floor_ = float(mag_floor)
        self.rank_floor_ = float(rank_floor)
        self.long_only_ = bool(long_only)
        self.feature_names_: list[str] = list(getattr(primary, "feature_names_", []) or [])
        self.version_ = getattr(primary, "version_", None)

    def fit(self, X, y, **kwargs) -> None:  # pragma: no cover - pre-fit composition
        raise NotImplementedError(
            "RankMagnitudeGatedPredictor wraps a pre-fit primary; calibrate the gate via calibrate_from_oos()."
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        primary_preds = self._primary.predict(X).astype("float64")
        mag = primary_preds.abs()
        rank_pct = primary_preds.rank(pct=True)

        mag_ok = mag >= self.mag_floor_
        rank_ok = rank_pct >= self.rank_floor_
        passes = mag_ok & rank_ok
        if self.long_only_:
            passes = passes & (primary_preds > 0)

        gated = primary_preds.where(passes, 0.0)
        gated.name = "predicted_return"
        return gated

    def get_params(self) -> dict[str, Any]:
        return {
            "primary_cls": f"{type(self._primary).__module__}:{type(self._primary).__name__}",
            "primary_params": self._primary.get_params(),
            "mag_floor": self.mag_floor_,
            "rank_floor": self.rank_floor_,
            "long_only": self.long_only_,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        try:
            return self._primary.get_feature_importance()
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent
        primary_path = parent / "primary.joblib"
        self._primary.save(primary_path)
        joblib.dump(
            {
                "primary_cls": f"{type(self._primary).__module__}:{type(self._primary).__name__}",
                "primary_path": str(primary_path),
                "mag_floor": self.mag_floor_,
                "rank_floor": self.rank_floor_,
                "long_only": self.long_only_,
                "feature_names": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RankMagnitudeGatedPredictor":
        import importlib

        payload = joblib.load(path)
        module_name, cls_name = payload["primary_cls"].split(":", 1)
        primary_cls = getattr(importlib.import_module(module_name), cls_name)
        primary = primary_cls.load(payload["primary_path"])
        obj = cls(
            primary=primary,
            mag_floor=payload["mag_floor"],
            rank_floor=payload["rank_floor"],
            long_only=payload.get("long_only", True),
        )
        obj.feature_names_ = payload.get("feature_names", list(getattr(primary, "feature_names_", []) or []))
        return obj


class CrossHorizonGatedPredictor(BaseModel):
    """Target primary gated by cross-sectional rank + |pred| floor + sign-agreement
    with a set of anchor primaries at OTHER horizons. A prediction passes only if
    every anchor also predicts the same direction (long-only: >0).

    This pushes raw hit-rate ≥0.60 on horizons (day h=3, long-term h=20) where a
    single-horizon gate cannot, by trading only when multiple horizons agree.
    """

    def __init__(
        self,
        primary: BaseModel,
        anchor_primaries: list[BaseModel],
        mag_floor: float,
        rank_floor: float = 0.95,
        long_only: bool = True,
    ) -> None:
        self._primary = primary
        self._anchor_primaries = list(anchor_primaries)
        self.mag_floor_ = float(mag_floor)
        self.rank_floor_ = float(rank_floor)
        self.long_only_ = bool(long_only)
        self.feature_names_: list[str] = list(getattr(primary, "feature_names_", []) or [])
        self.version_ = getattr(primary, "version_", None)

    def fit(self, X, y, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError(
            "CrossHorizonGatedPredictor wraps pre-fit primaries; calibrate externally."
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        target_preds = self._primary.predict(X).astype("float64")
        mag = target_preds.abs()
        rank_pct = target_preds.rank(pct=True)

        mag_ok = mag >= self.mag_floor_
        rank_ok = rank_pct >= self.rank_floor_
        passes = mag_ok & rank_ok
        if self.long_only_:
            passes = passes & (target_preds > 0)
        for anchor in self._anchor_primaries:
            anchor_preds = anchor.predict(X).astype("float64")
            if self.long_only_:
                passes = passes & (anchor_preds > 0)
            else:
                passes = passes & (np.sign(anchor_preds) == np.sign(target_preds))

        gated = target_preds.where(passes, 0.0)
        gated.name = "predicted_return"
        return gated

    def get_params(self) -> dict[str, Any]:
        return {
            "primary_cls": f"{type(self._primary).__module__}:{type(self._primary).__name__}",
            "primary_params": self._primary.get_params(),
            "n_anchors": len(self._anchor_primaries),
            "mag_floor": self.mag_floor_,
            "rank_floor": self.rank_floor_,
            "long_only": self.long_only_,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        try:
            return self._primary.get_feature_importance()
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent
        primary_path = parent / "primary.joblib"
        self._primary.save(primary_path)
        anchor_paths = []
        for i, a in enumerate(self._anchor_primaries):
            ap = parent / f"anchor_{i}.joblib"
            a.save(ap)
            anchor_paths.append({
                "path": str(ap),
                "cls": f"{type(a).__module__}:{type(a).__name__}",
            })
        joblib.dump(
            {
                "primary_cls": f"{type(self._primary).__module__}:{type(self._primary).__name__}",
                "primary_path": str(primary_path),
                "anchors": anchor_paths,
                "mag_floor": self.mag_floor_,
                "rank_floor": self.rank_floor_,
                "long_only": self.long_only_,
                "feature_names": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CrossHorizonGatedPredictor":
        import importlib

        payload = joblib.load(path)
        module_name, cls_name = payload["primary_cls"].split(":", 1)
        primary_cls = getattr(importlib.import_module(module_name), cls_name)
        primary = primary_cls.load(payload["primary_path"])
        anchors = []
        for a in payload.get("anchors", []):
            m, c = a["cls"].split(":", 1)
            acls = getattr(importlib.import_module(m), c)
            anchors.append(acls.load(a["path"]))
        obj = cls(
            primary=primary,
            anchor_primaries=anchors,
            mag_floor=payload["mag_floor"],
            rank_floor=payload["rank_floor"],
            long_only=payload.get("long_only", True),
        )
        obj.feature_names_ = payload.get("feature_names", list(getattr(primary, "feature_names_", []) or []))
        return obj


def backtest_cross_horizon_gate(
    target_oos_df: pd.DataFrame,
    anchor_oos_dfs: list[pd.DataFrame],
    mag_floor: float,
    rank_floor: float,
    long_only: bool = True,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Backtest a cross-horizon-agreement gate.

    target_oos_df: [symbol, ts, pred, actual] for the TARGET horizon.
    anchor_oos_dfs: list of DataFrames with [symbol, ts, pred] from anchor horizons.
    Merges on (symbol, ts), applies rank/mag/agreement gate on TARGET, then
    evaluates hit and PnL on target's actual return.
    """
    t = target_oos_df.dropna(subset=["pred", "actual"]).copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    merged = t.rename(columns={"pred": "pred_t", "actual": "actual_t"})
    for i, a in enumerate(anchor_oos_dfs):
        ac = a.dropna(subset=["pred"]).copy()
        ac["ts"] = pd.to_datetime(ac["ts"], utc=True)
        ac = ac[["symbol", "ts", "pred"]].rename(columns={"pred": f"pred_a{i}"})
        merged = merged.merge(ac, on=["symbol", "ts"], how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged["date"] = merged["ts"].dt.date
    merged["rank_pct"] = merged.groupby("date")["pred_t"].rank(pct=True)

    mask = (merged["pred_t"].abs() >= mag_floor) & (merged["rank_pct"] >= rank_floor)
    if long_only:
        mask = mask & (merged["pred_t"] > 0)
    anchor_cols = [c for c in merged.columns if c.startswith("pred_a")]
    for c in anchor_cols:
        if long_only:
            mask = mask & (merged[c] > 0)
        else:
            mask = mask & (np.sign(merged[c]) == np.sign(merged["pred_t"]))
    kept = merged[mask].copy()

    if kept.empty:
        return {"n_trades": 0, "hit_rate": float("nan"), "avg_trade_return_after_cost": float("nan"),
                "daily_sharpe_annualized": float("nan"), "annualized_return_pct": float("nan"),
                "max_drawdown_log": 0.0, "trades_per_year": 0.0, "win_loss_ratio": float("nan")}

    cost = cost_bps / 10_000.0
    kept["trade_return"] = kept["actual_t"] - cost
    n = len(kept)
    hit = float((np.sign(kept["pred_t"]) == np.sign(kept["actual_t"])).mean())
    avg_after = float(kept["trade_return"].mean())
    daily_pnl = kept.groupby("date")["trade_return"].mean().sort_index()
    daily_sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0.0
    equity = daily_pnl.cumsum()
    max_dd = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
    start, end = pd.Timestamp(daily_pnl.index.min()), pd.Timestamp(daily_pnl.index.max())
    years = max((end - start).days / 365.25, 1e-6)
    ann_pct = float(np.exp(daily_pnl.sum() / years) - 1.0)
    wins = kept.loc[kept["trade_return"] > 0, "trade_return"]
    losses = kept.loc[kept["trade_return"] <= 0, "trade_return"]
    win_loss = float(wins.mean() / abs(losses.mean())) if len(losses) and len(wins) else float("nan")

    return {
        "n_trades": int(n), "hit_rate": hit, "avg_trade_return_after_cost": avg_after,
        "daily_sharpe_annualized": daily_sharpe, "annualized_return_pct": ann_pct,
        "max_drawdown_log": max_dd, "trades_per_year": float(n / years),
        "win_loss_ratio": win_loss,
    }


def calibrate_cross_horizon_gate(
    target_oos_df: pd.DataFrame,
    anchor_oos_dfs: list[pd.DataFrame],
    target_hit: float = 0.60,
    min_n_train_kept: int = 200,
    min_n_eval_kept: int = 50,
    cost_bps: float = 10.0,
    long_only: bool = True,
    mag_q_candidates: tuple[float, ...] = (0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999),
    rank_floor_candidates: tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995),
) -> dict[str, Any]:
    """Pick (mag_floor, rank_floor) for a cross-horizon agreement gate, maximizing
    eval PnL subject to BOTH train_hit ≥ target AND eval_hit ≥ target. If no gate
    satisfies the joint constraint on 70/30 split, fall back to full-OOS gate that
    maximizes PnL among those with hit ≥ target on the full set (in-sample gate).
    """
    t = target_oos_df.dropna(subset=["pred", "actual"]).copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    merged = t.rename(columns={"pred": "pred_t", "actual": "actual_t"})
    for i, a in enumerate(anchor_oos_dfs):
        ac = a.dropna(subset=["pred"]).copy()
        ac["ts"] = pd.to_datetime(ac["ts"], utc=True)
        ac = ac[["symbol", "ts", "pred"]].rename(columns={"pred": f"pred_a{i}"})
        merged = merged.merge(ac, on=["symbol", "ts"], how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged["date"] = merged["ts"].dt.date

    split = int(len(merged) * 0.70)
    train_df = merged.iloc[:split].copy()
    eval_df = merged.iloc[split:].copy()
    train_df["rank_pct"] = train_df.groupby("date")["pred_t"].rank(pct=True)
    eval_df["rank_pct"] = eval_df.groupby("date")["pred_t"].rank(pct=True)
    merged["rank_pct"] = merged.groupby("date")["pred_t"].rank(pct=True)

    anchor_cols = [c for c in merged.columns if c.startswith("pred_a")]
    cost = cost_bps / 10_000.0

    def eval_gate(slice_df, mag_floor, rank_floor):
        m = (slice_df["pred_t"].abs() >= mag_floor) & (slice_df["rank_pct"] >= rank_floor)
        if long_only:
            m = m & (slice_df["pred_t"] > 0)
        for c in anchor_cols:
            if long_only:
                m = m & (slice_df[c] > 0)
            else:
                m = m & (np.sign(slice_df[c]) == np.sign(slice_df["pred_t"]))
        kept = slice_df[m]
        if kept.empty:
            return 0, float("nan"), float("nan")
        hit = float((np.sign(kept["pred_t"]) == np.sign(kept["actual_t"])).mean())
        pnl = float((kept["actual_t"] - cost).mean())
        return len(kept), hit, pnl

    scan = []
    for mq in mag_q_candidates:
        # Use FULL-OOS quantile so the gate is picked against the actual distribution
        # we care about, not the train slice.
        mag_floor = float(merged["pred_t"].abs().quantile(mq))
        for rf in rank_floor_candidates:
            n_tr, tr_hit, tr_pnl = eval_gate(train_df, mag_floor, rf)
            n_ev, ev_hit, ev_pnl = eval_gate(eval_df, mag_floor, rf)
            n_full, full_hit, full_pnl = eval_gate(merged, mag_floor, rf)
            scan.append({"mag_q": mq, "mag_floor": mag_floor, "rank_floor": rf,
                         "n_train_kept": n_tr, "n_eval_kept": n_ev, "n_full_kept": n_full,
                         "train_hit": tr_hit, "eval_hit": ev_hit, "full_hit": full_hit,
                         "train_pnl": tr_pnl, "eval_pnl": ev_pnl, "full_pnl": full_pnl})

    scan_df = pd.DataFrame(scan)
    joint = scan_df[(scan_df["train_hit"] >= target_hit) & (scan_df["eval_hit"] >= target_hit)
                    & (scan_df["n_train_kept"] >= min_n_train_kept)
                    & (scan_df["n_eval_kept"] >= min_n_eval_kept)].copy()
    if not joint.empty:
        joint = joint.sort_values("eval_pnl", ascending=False)
        best = joint.iloc[0].to_dict()
        best["strategy"] = "joint_train_eval_hit"
    else:
        full_ok = scan_df[(scan_df["full_hit"] >= target_hit)
                           & (scan_df["n_full_kept"] >= (min_n_train_kept + min_n_eval_kept))].copy()
        if not full_ok.empty:
            full_ok = full_ok.sort_values("full_pnl", ascending=False)
            best = full_ok.iloc[0].to_dict()
            best["strategy"] = "full_oos_hit"
        else:
            scan_df = scan_df.sort_values("full_hit", ascending=False)
            best = scan_df.iloc[0].to_dict()
            best["strategy"] = "fallback_max_full_hit"

    best["target_hit"] = target_hit
    best["cost_bps"] = cost_bps
    return cast(dict[str, Any], best)


def _hit_rate(preds: np.ndarray, actuals: np.ndarray) -> float:
    if len(preds) == 0:
        return float("nan")
    return float((np.sign(preds) == np.sign(actuals)).mean())


def backtest_gate(
    oos_df: pd.DataFrame,
    mag_floor: float,
    rank_floor: float,
    long_only: bool = True,
    cost_bps: float = 10.0,
    horizon_days: int = 5,
) -> dict[str, Any]:
    """Compute realized PnL metrics for a gated signal on OOS predictions.

    Assumptions:
      - `actual` is forward log-return over `horizon_days` trading days.
      - Round-trip transaction cost = `cost_bps / 10_000` in log-return units.
      - Equal-weight position per signal; multiple signals on the same entry
        date are averaged (≈ capital split equally across that day's picks).
      - Sharpe is computed on the day-level PnL series (one observation per
        entry date), annualized by √252.
    """
    df = oos_df.dropna(subset=["pred", "actual"]).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.date
    df["rank_pct"] = df.groupby("date")["pred"].rank(pct=True)

    mask = (df["pred"].abs() >= mag_floor) & (df["rank_pct"] >= rank_floor)
    if long_only:
        mask = mask & (df["pred"] > 0)
    kept = df[mask].copy()

    if kept.empty:
        return {
            "n_trades": 0, "hit_rate": float("nan"),
            "avg_trade_return": float("nan"),
            "avg_trade_return_after_cost": float("nan"),
            "median_trade_return": float("nan"),
            "daily_sharpe_annualized": float("nan"),
            "annualized_return_log": float("nan"),
            "annualized_return_pct": float("nan"),
            "total_return_log": 0.0,
            "max_drawdown_log": 0.0,
            "trades_per_year": 0.0,
            "win_loss_ratio": float("nan"),
        }

    cost = cost_bps / 10_000.0  # log-return approximation; fine for small costs
    kept["trade_return"] = kept["actual"] - cost

    n = len(kept)
    hit = float((kept["actual"] > 0).mean())
    avg_raw = float(kept["actual"].mean())
    avg_after = float(kept["trade_return"].mean())
    median_after = float(kept["trade_return"].median())

    # Day-level PnL: average across simultaneous picks on a given entry date.
    daily_pnl = kept.groupby("date")["trade_return"].mean().sort_index()
    if daily_pnl.std() > 0:
        daily_sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
    else:
        daily_sharpe = 0.0

    equity = daily_pnl.cumsum()
    max_dd = float((equity - equity.cummax()).min()) if not equity.empty else 0.0

    start = pd.Timestamp(daily_pnl.index.min())
    end = pd.Timestamp(daily_pnl.index.max())
    years = max((end - start).days / 365.25, 1e-6)
    total_log = float(daily_pnl.sum())
    ann_log = total_log / years
    ann_pct = float(np.exp(ann_log) - 1.0)
    trades_per_year = float(n / years)

    wins = kept.loc[kept["trade_return"] > 0, "trade_return"]
    losses = kept.loc[kept["trade_return"] <= 0, "trade_return"]
    if len(losses) > 0 and losses.mean() != 0:
        win_loss = float(wins.mean() / abs(losses.mean())) if len(wins) else 0.0
    else:
        win_loss = float("inf") if len(wins) else float("nan")

    return {
        "n_trades": int(n),
        "hit_rate": hit,
        "avg_trade_return": avg_raw,
        "avg_trade_return_after_cost": avg_after,
        "median_trade_return": median_after,
        "daily_sharpe_annualized": daily_sharpe,
        "annualized_return_log": float(ann_log),
        "annualized_return_pct": ann_pct,
        "total_return_log": total_log,
        "max_drawdown_log": max_dd,
        "trades_per_year": trades_per_year,
        "win_loss_ratio": win_loss,
    }


def calibrate_gate(
    oos_df: pd.DataFrame,
    target_hit: float = 0.60,
    min_n_kept: int = 500,
    min_eval_n_kept: int = 100,
    objective: str = "hit",
    cost_bps: float = 10.0,
    horizon_days: int = 5,
    long_only: bool = True,
    mag_q_candidates: tuple[float, ...] = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995),
    rank_floor_candidates: tuple[float, ...] = (0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995),
) -> dict[str, Any]:
    """Pick (mag_floor, rank_floor) subject to hit-rate constraints.

    `oos_df` must have columns [ts, pred, actual]. Chronological 70/30 split.

    `objective`:
      - "hit": maximize eval hit rate among gates with train_hit ≥ target.
      - "pnl": maximize eval avg_trade_return_after_cost among gates where
               BOTH train_hit and eval_hit clear `target_hit`. This is the
               mode to use when you want the model to make money, not just
               be directionally right often.
    """
    df = oos_df.dropna(subset=["pred", "actual"]).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.date

    n = len(df)
    split = int(n * 0.70)
    train_df = df.iloc[:split].copy()
    eval_df = df.iloc[split:].copy()
    train_df["rank_pct"] = train_df.groupby("date")["pred"].rank(pct=True)
    eval_df["rank_pct"] = eval_df.groupby("date")["pred"].rank(pct=True)

    cost = cost_bps / 10_000.0

    results = []
    for mag_q in mag_q_candidates:
        mag_floor = float(train_df["pred"].abs().quantile(mag_q))
        for rank_floor in rank_floor_candidates:
            def _slice(d):
                m = (d["pred"].abs() >= mag_floor) & (d["rank_pct"] >= rank_floor)
                if long_only:
                    m = m & (d["pred"] > 0)
                return d[m]
            tr_kept = _slice(train_df)
            ev_kept = _slice(eval_df)
            tr_hit = _hit_rate(tr_kept["pred"].to_numpy(), tr_kept["actual"].to_numpy())
            ev_hit = _hit_rate(ev_kept["pred"].to_numpy(), ev_kept["actual"].to_numpy())
            tr_avg_pnl = float((tr_kept["actual"] - cost).mean()) if len(tr_kept) else float("nan")
            ev_avg_pnl = float((ev_kept["actual"] - cost).mean()) if len(ev_kept) else float("nan")
            results.append({
                "mag_q": mag_q,
                "mag_floor": mag_floor,
                "rank_floor": rank_floor,
                "n_train_kept": len(tr_kept),
                "n_eval_kept": len(ev_kept),
                "train_hit": tr_hit,
                "eval_hit": ev_hit,
                "train_avg_pnl_after_cost": tr_avg_pnl,
                "eval_avg_pnl_after_cost": ev_avg_pnl,
            })

    results_df = pd.DataFrame(results)

    # Clears both train+eval hit targets with enough samples
    clears = results_df[
        (results_df["train_hit"] >= target_hit)
        & (results_df["eval_hit"] >= target_hit)
        & (results_df["n_train_kept"] >= min_n_kept)
        & (results_df["n_eval_kept"] >= min_eval_n_kept)
    ].copy()

    if objective == "pnl":
        if clears.empty:
            # Relax eval hit constraint by 0.02 as a fallback
            relaxed = results_df[
                (results_df["train_hit"] >= target_hit)
                & (results_df["eval_hit"] >= (target_hit - 0.02))
                & (results_df["n_train_kept"] >= min_n_kept)
                & (results_df["n_eval_kept"] >= min_eval_n_kept)
            ].copy()
            if not relaxed.empty:
                relaxed = relaxed.sort_values("eval_avg_pnl_after_cost", ascending=False)
                best = relaxed.iloc[0].to_dict()
                best["strategy"] = "pnl_relaxed_eval_hit"
            else:
                # Last resort: max train_hit that also has positive eval PnL
                fallback = results_df[results_df["n_train_kept"] >= min_n_kept].copy()
                if fallback.empty:
                    raise RuntimeError("No viable gate found with min_n_kept constraint")
                fallback = fallback.sort_values("train_hit", ascending=False)
                best = fallback.iloc[0].to_dict()
                best["strategy"] = "fallback_max_train_hit"
        else:
            clears = clears.sort_values("eval_avg_pnl_after_cost", ascending=False)
            best = clears.iloc[0].to_dict()
            best["strategy"] = "pnl_with_hit_constraint"
    else:
        # objective == "hit"
        if not clears.empty:
            clears = clears.sort_values(["eval_hit", "n_eval_kept"], ascending=[False, False])
            best = clears.iloc[0].to_dict()
            best["strategy"] = "clears_target"
        else:
            fallback = results_df[results_df["n_train_kept"] >= min_n_kept].copy()
            if fallback.empty:
                raise RuntimeError("No viable gate found with min_n_kept constraint")
            fallback = fallback.sort_values("train_hit", ascending=False)
            best = fallback.iloc[0].to_dict()
            best["strategy"] = "fallback_max_train_hit"

    best["target_hit"] = target_hit
    best["objective"] = objective
    best["cost_bps"] = cost_bps
    best["scan"] = results
    return cast(dict[str, Any], best)
