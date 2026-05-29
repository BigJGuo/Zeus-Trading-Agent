"""Zeus 2015-2025 backtest — pure ML, no broker, no LLM, no DB.

This script runs a self-contained historical backtest of the Zeus multi-strategy
system from January 1, 2015 through December 31, 2025 starting with $100,000.

It bypasses the live broker (Alpaca), the LLM agent layer, and the SQLAlchemy
storage layer entirely. Signal generation comes only from the LightGBM return
predictor, trained walk-forward over the period.

Outputs are written under artifacts/backtest_2015_2025/.
"""
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportCallIssue=false, reportReturnType=false
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings

# Force UTF-8 stdout so the box-drawing/arrow glyphs render on Windows cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# Allow running as `python scripts/run_backtest_2015_2025.py` from project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zeus.backtesting.metrics import (
    calmar_ratio,
    hit_rate,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from zeus.backtesting.walk_forward import WalkForwardValidator
from zeus.features.cross_sectional import compute_cross_sectional_features
from zeus.features.interactions import add_macro_interactions
from zeus.features.technical import compute_technical_features
from zeus.models.labels import compute_triple_barrier_labels
from zeus.models.lgbm_return_predictor import LGBMReturnPredictor
from zeus.risk.drawdown_guard import DrawdownGuard, DrawdownLevel
from zeus.risk.limits import RiskLimits
from zeus.risk.position_sizer import (
    SizingInputs,
    kelly_position_size,
    shares_from_target,
)
from zeus.risk.stop_logic import (
    compute_initial_stops,
    should_exit_on_stop,
    update_trailing_stop,
)

# ── Run configuration ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_START = "2013-07-01"        # Extra warmup for 18mo first fold + 200d features.
DATA_END = "2025-12-31"
TRADE_START = pd.Timestamp("2015-01-01", tz="UTC")
TRADE_END = pd.Timestamp("2025-12-31", tz="UTC")
INITIAL_CAPITAL = 100_000.0
TXN_COST_BPS = 5.0               # One-way transaction cost (bps).
RF_ANNUAL = 0.015                # 1.5% risk-free for Sharpe.
PROMOTE_IC = 0.03
PROMOTE_HIT = 0.52
PROMOTE_SHARPE = 0.90
ARTIFACTS = _PROJECT_ROOT / "artifacts" / "backtest_2015_2025_4"

np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────
# Canonical ticker form across this script is the dot-form used by Alpaca
# (e.g. BRK.B, BF.B) — matching zeus.data.ingestion.universe_builder. At the
# yfinance boundary we convert to dash-form (BRK-B). first_trade_date below
# is keyed by the canonical (dot) form.

BENCHMARK = "SPY"
MACRO_TICKERS = ["^VIX", "^TNX", "^IRX"]   # VIX, 10Y, 13W treasury yields.


def fetch_sp500_universe() -> List[str]:
    """Pull the current S&P 500 from Wikipedia.

    Returns the constituent ticker list normalized to Alpaca dot-form
    (BRK.B, BF.B). This mirrors `UniverseBuilder.build_base_universe` —
    yfinance/Wikipedia ship dashes, the rest of the Zeus stack uses dots.
    Survivorship is one-sided (delisted names absent); we use IPO-date
    awareness downstream to mitigate look-ahead, but a fully
    point-in-time membership list would need a paid feed.

    Wikipedia rejects pandas's default User-Agent with HTTP 403, so we
    fetch the page through `requests` first (same workaround as
    YFinanceClient._fetch_html_tables) and hand the HTML to read_html.
    """
    import requests
    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        timeout=15,
    )
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), header=0)
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    raw = [str(s).strip().upper() for s in df[col].tolist()]
    # Wikipedia uses dashes; convert to Alpaca dot-form for canonical use.
    dot = sorted({t.replace("-", ".") for t in raw if t and t != "NAN"})
    print(f"  → Wikipedia S&P 500: {len(dot)} tickers")
    return dot


def _to_yf(sym: str) -> str:
    """Canonical (dot) → yfinance/Wikipedia (dash) ticker form."""
    return sym.replace(".", "-")


# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion
# ─────────────────────────────────────────────────────────────────────────────
def download_ohlcv(
    symbols: List[str], start: str, end: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Timestamp]]:
    """Download OHLCV for each symbol via yfinance.

    `symbols` are canonical (dot-form) tickers. At the yfinance boundary we
    convert to dash-form; results are keyed back to the canonical form so the
    rest of the pipeline only ever sees dots.

    Returns:
        (ohlcv_by_symbol, first_trade_date_by_symbol)
        - ohlcv_by_symbol: symbol → DataFrame indexed by date with
          [open, high, low, close, volume, adj_close].
        - first_trade_date_by_symbol: symbol → earliest date yfinance
          returned data for that ticker (proxy for IPO date within the
          requested window).
        Empty / failed downloads are dropped.
    """
    import yfinance as yf

    print(f"  → Downloading {len(symbols)} symbols from {start} to {end}...")
    batch_size = 25
    out: Dict[str, pd.DataFrame] = {}
    first_trade: Dict[str, pd.Timestamp] = {}

    # Build dash-form list for yfinance + a reverse map back to canonical.
    yf_to_canon = {_to_yf(s): s for s in symbols}
    yf_symbols = list(yf_to_canon.keys())

    for i in range(0, len(yf_symbols), batch_size):
        batch = yf_symbols[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker",
            )
        except Exception as exc:
            print(f"    ! batch {i//batch_size} failed: {exc}")
            continue
        if raw is None or raw.empty:
            continue
        # group_by="ticker" gives a MultiIndex column with (symbol, field).
        if isinstance(raw.columns, pd.MultiIndex):
            for yf_sym in batch:
                try:
                    df = raw[yf_sym].copy()
                except KeyError:
                    continue
                df = _normalize_ohlcv(df)
                if df is None or df.empty:
                    continue
                canon = yf_to_canon.get(yf_sym, yf_sym)
                out[canon] = df
                first_trade[canon] = pd.Timestamp(df.index.min())
        else:
            # Single-symbol batch — columns are flat.
            yf_sym = batch[0]
            df = _normalize_ohlcv(raw.copy())
            if df is not None and not df.empty:
                canon = yf_to_canon.get(yf_sym, yf_sym)
                out[canon] = df
                first_trade[canon] = pd.Timestamp(df.index.min())
    print(f"  → Got data for {len(out)} symbols")
    return out, first_trade


def _normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    rename = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
    }
    df = df.rename(columns=rename)
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df.dropna(subset=needed)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def apply_universe_filters(
    ohlcv: Dict[str, pd.DataFrame],
    min_price: float = 5.0,
    max_price: float = 2000.0,
    min_adv_usd: float = 5_000_000.0,
) -> Dict[str, pd.DataFrame]:
    """Drop symbols that never satisfy price / ADV filters in the window."""
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv.items():
        if df.empty:
            continue
        last_close = float(df["close"].iloc[-1])
        if not (min_price <= last_close <= max_price):
            continue
        adv = (df["close"] * df["volume"]).rolling(20).mean()
        if adv.dropna().max() < min_adv_usd:
            continue
        out[sym] = df
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature & label engineering
# ─────────────────────────────────────────────────────────────────────────────
# Fundamentals (PE, PB, etc.) require paid APIs to get point-in-time history.
# We zero them out so the feature shape is preserved; the model trains on the
# technical + cross-sectional + macro slice. This is documented in summary.txt.
_ZEROED_FUNDAMENTAL_FEATURES = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "earnings_yield",
    "revenue_growth_yoy", "earnings_growth_yoy", "gross_margin",
    "operating_margin", "debt_to_equity", "current_ratio", "roe", "roa",
    "short_interest_ratio", "market_cap_log",
]
# Fed funds rate / put-call ratio also gated behind premium feeds — zeroed.
_ZEROED_MACRO_FEATURES = ["fed_funds_rate", "put_call_ratio", "put_call_ratio_5d_change"]


def compute_macro_panel(macro_ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute daily macro features from yfinance series.

    Returns a DataFrame indexed by date with columns vix_level, vix_5d_change,
    yield_10y, yield_2y, yield_curve_spread, spy_return_5d, spy_return_20d, and
    zeroed columns for fed_funds_rate / put_call_ratio.
    """
    idx = None
    for sym in (BENCHMARK,) + tuple(MACRO_TICKERS):
        if sym in macro_ohlcv:
            i = macro_ohlcv[sym].index
            idx = i if idx is None else idx.union(i)
    if idx is None:
        return pd.DataFrame()

    panel = pd.DataFrame(index=idx)
    # VIX
    if "^VIX" in macro_ohlcv:
        vix = macro_ohlcv["^VIX"]["close"].reindex(idx).ffill()
        panel["vix_level"] = vix
        panel["vix_5d_change"] = vix.diff(5)
    else:
        panel["vix_level"] = np.nan
        panel["vix_5d_change"] = np.nan
    # Treasury yields
    if "^TNX" in macro_ohlcv:
        panel["yield_10y"] = macro_ohlcv["^TNX"]["close"].reindex(idx).ffill() / 100.0
    else:
        panel["yield_10y"] = np.nan
    if "^IRX" in macro_ohlcv:
        # ^IRX is 13-week T-bill; substitute for 2y as yfinance has no free 2y series.
        panel["yield_2y"] = macro_ohlcv["^IRX"]["close"].reindex(idx).ffill() / 100.0
    else:
        panel["yield_2y"] = np.nan
    panel["yield_curve_spread"] = panel["yield_10y"] - panel["yield_2y"]
    # SPY returns
    if BENCHMARK in macro_ohlcv:
        spy = macro_ohlcv[BENCHMARK]["close"].reindex(idx).ffill()
        panel["spy_return_5d"] = spy.pct_change(5)
        panel["spy_return_20d"] = spy.pct_change(20)
    else:
        panel["spy_return_5d"] = np.nan
        panel["spy_return_20d"] = np.nan
    # Zeroed (documented)
    for col in _ZEROED_MACRO_FEATURES:
        panel[col] = 0.0

    panel.index.name = "ts"
    return panel


def build_feature_panel(
    ohlcv: Dict[str, pd.DataFrame],
    macro_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the full per-(symbol, date) feature panel.

    Returns a long-form DataFrame with [symbol, ts, ...features..., close, volume,
    high, low, open] sorted by ts. All features are point-in-time as of `ts`:
    rolling/diff calculations use only data at or before ts, never after.
    """
    print("  → Computing technical features per symbol...")
    rows: List[pd.DataFrame] = []
    for sym, df in ohlcv.items():
        tech = compute_technical_features(df[["open", "high", "low", "close", "volume"]])
        merged = pd.concat([df[["open", "high", "low", "close", "volume"]], tech], axis=1)
        merged = merged.reset_index().rename(columns={merged.index.name or "index": "ts"})
        if "ts" not in merged.columns:
            # Older pandas — index name may be 'Date'
            merged = merged.rename(columns={merged.columns[0]: "ts"})
        merged["symbol"] = sym
        # Cross-sectional precomputes
        merged["return_5d"] = merged["close"].pct_change(5)
        merged["return_20d"] = merged["close"].pct_change(20)
        merged["vol_20d"] = np.log(merged["close"] / merged["close"].shift(1)).rolling(20).std()
        rows.append(merged)

    panel = pd.concat(rows, ignore_index=True)
    panel["ts"] = pd.to_datetime(panel["ts"]).dt.tz_localize(None)

    # Cross-sectional ranks (no sector data → "Unknown" sector for all).
    print("  → Computing cross-sectional features...")
    sector_map = {s: "Unknown" for s in ohlcv}
    panel = compute_cross_sectional_features(panel, sector_map)

    # Zeroed fundamentals.
    for col in _ZEROED_FUNDAMENTAL_FEATURES:
        panel[col] = 0.0

    # Macro broadcast.
    if not macro_panel.empty:
        macro_to_join = macro_panel.copy()
        macro_to_join.index = pd.to_datetime(macro_to_join.index).tz_localize(None)
        panel = panel.merge(
            macro_to_join.reset_index().rename(columns={"index": "ts"}),
            on="ts", how="left",
        )
    else:
        for col in (
            "vix_level", "vix_5d_change", "yield_10y", "yield_2y",
            "yield_curve_spread", "spy_return_5d", "spy_return_20d",
            *_ZEROED_MACRO_FEATURES,
        ):
            panel[col] = 0.0

    # Interaction terms.
    panel = add_macro_interactions(panel)

    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return panel


def compute_forward_returns(
    ohlcv: Dict[str, pd.DataFrame], horizons: Tuple[int, ...] = (3, 5, 20)
) -> pd.DataFrame:
    """Forward log-returns per (symbol, ts) for each horizon. No look-ahead:
    the value at ts is log(close[ts + h] / close[ts]) and is filled with NaN
    for ts within the last h rows of each symbol's series.
    """
    rows: List[pd.DataFrame] = []
    for sym, df in ohlcv.items():
        close = df["close"]
        sub = pd.DataFrame({"ts": df.index.tz_localize(None) if df.index.tz else df.index})
        sub["symbol"] = sym
        for h in horizons:
            sub[f"fwd_{h}d_return"] = np.log(close.shift(-h).values / close.values)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def select_feature_columns(panel: pd.DataFrame) -> List[str]:
    """Numeric columns the model should train on. Excludes IDs, label-like, and
    raw-price columns (which would leak via constant per-symbol scale).

    Crucial: all label-family columns must be filtered out here, not just the
    one we happen to be training on. Otherwise sibling labels become features.
    For triple-barrier labels (added later than `fwd_*`), `tb_return_h{H}` and
    `tb_days_to_hit_h{H}` are computed from the same future window as
    `tb_label_h{H}` and leak the target perfectly when included in X.
    """
    exclude = {
        "symbol", "ts", "open", "high", "low", "close", "volume", "adj_close",
        "dollar_volume",
    }
    cols: List[str] = []
    for c in panel.columns:
        if c in exclude:
            continue
        # Any future-looking label column, regardless of family.
        if c.startswith("fwd_"):
            continue
        if c.startswith("tb_label_") or c.startswith("tb_return_") or c.startswith("tb_days_to_hit_"):
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            cols.append(c)
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward training
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FoldRecord:
    horizon: int
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    ic: float
    hit_rate: float
    sharpe_on_signals: float
    promoted: bool


def _signal_returns(preds: np.ndarray, actuals: np.ndarray, dates: np.ndarray) -> pd.Series:
    df = pd.DataFrame({"pred": preds, "actual": actuals, "date": dates}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    out: List[Tuple[pd.Timestamp, float]] = []
    for dt, grp in df.groupby("date"):
        if len(grp) < 5:
            continue
        thr = grp["pred"].quantile(0.8)
        longs = grp[grp["pred"] >= thr]["actual"]
        if len(longs) == 0:
            continue
        out.append((dt, float(longs.mean())))
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series([v for _, v in out], index=[d for d, _ in out])
    return s.sort_index()


def run_walk_forward_for_horizon(
    panel: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    horizon: int,
    first_trade_date: Optional[Dict[str, pd.Timestamp]] = None,
) -> Tuple[pd.DataFrame, List[FoldRecord]]:
    """Train one LGBM model per fold for a given horizon.

    Args:
        first_trade_date: per-symbol earliest available bar date. If provided,
            any symbol whose first_trade_date is *after* the fold's train_start
            is silently excluded from that fold — i.e. the model never sees a
            stock that didn't exist by the start of its training period. This
            keeps the cross-section clean of look-ahead from IPO timing.

    Returns:
        predictions_df: long-form [symbol, ts, prediction] over all test
            windows in TRADE_START..TRADE_END. Folds that failed the
            promotion gate fall back to predictions from the last promoted
            model (if any).
        fold_records: per-fold diagnostics.
    """
    panel_sorted = panel.sort_values("ts").reset_index(drop=True)
    panel_sorted["ts"] = pd.to_datetime(panel_sorted["ts"])

    dates = pd.DatetimeIndex(panel_sorted["ts"].unique())
    validator = WalkForwardValidator(
        train_window_months=24, validation_window_months=3, step_months=1,
    )
    folds = validator.generate_folds(dates)
    if not folds:
        return pd.DataFrame(columns=["symbol", "ts", "prediction"]), []

    # Filter folds: only run folds whose test window touches the trade horizon.
    folds = [f for f in folds if f[3] >= TRADE_START.date() and f[2] <= TRADE_END.date()]

    last_promoted_model: Optional[LGBMReturnPredictor] = None
    last_promoted_features: List[str] = []
    records: List[FoldRecord] = []
    pred_frames: List[pd.DataFrame] = []

    feature_cols_in_panel = [c for c in feature_cols if c in panel_sorted.columns]

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        try:
            tr_mask = (panel_sorted["ts"].dt.date >= tr_s) & (panel_sorted["ts"].dt.date <= tr_e)
            te_mask = (panel_sorted["ts"].dt.date >= te_s) & (panel_sorted["ts"].dt.date <= te_e)

            # IPO-date awareness: any symbol that didn't exist by train_start
            # is silently excluded from this fold (both train and test slices)
            # so models never see a stock whose first bar lands mid-window.
            if first_trade_date:
                tr_s_ts = pd.Timestamp(tr_s)
                eligible = {
                    s for s, ft in first_trade_date.items()
                    if pd.Timestamp(ft) <= tr_s_ts
                }
                sym_ok = panel_sorted["symbol"].isin(eligible)
                tr_mask = tr_mask & sym_ok
                te_mask = te_mask & sym_ok

            train = panel_sorted.loc[tr_mask].copy()
            test = panel_sorted.loc[te_mask].copy()
            if train.empty or test.empty:
                continue

            train = train.dropna(subset=[label_col])
            if train.empty:
                continue

            X_train = train[feature_cols_in_panel].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
            y_train = train[label_col].astype("float32")
            X_test = test[feature_cols_in_panel].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

            model = LGBMReturnPredictor(
                n_estimators=500,
                max_depth=-1,
                learning_rate=0.01,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=50,
                early_stopping_rounds=0,
            )
            model.fit(X_train, y_train)

            preds_test = model.predict(X_test).values

            # Validation metrics
            actuals_test = test[label_col].values if label_col in test.columns else np.full(len(test), np.nan)
            dates_test = test["ts"].dt.date.values
            mask = ~np.isnan(actuals_test)
            if mask.sum() >= 30:
                ic = information_coefficient(
                    pd.Series(preds_test[mask]),
                    pd.Series(actuals_test[mask]),
                    group=pd.Series(dates_test[mask]),
                )
                hr = hit_rate(pd.Series(preds_test[mask]), pd.Series(actuals_test[mask]))
                sig_ret = _signal_returns(preds_test[mask], actuals_test[mask], dates_test[mask])
                shp = sharpe_ratio(sig_ret) if not sig_ret.empty else 0.0
            else:
                ic, hr, shp = 0.0, 0.0, 0.0

            promoted = (ic >= PROMOTE_IC and hr >= PROMOTE_HIT and shp >= PROMOTE_SHARPE)
            if promoted:
                last_promoted_model = model
                last_promoted_features = list(feature_cols_in_panel)

            # Deploy the *promoted* model (latest one) for this test window.
            if last_promoted_model is not None:
                X_pred = test[last_promoted_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
                deployed_preds = last_promoted_model.predict(X_pred).values
            else:
                deployed_preds = np.zeros(len(test))

            frame = pd.DataFrame({
                "symbol": test["symbol"].values,
                "ts": test["ts"].values,
                "prediction": deployed_preds,
                "fold": fold_idx,
            })
            pred_frames.append(frame)

            records.append(FoldRecord(
                horizon=horizon,
                fold=fold_idx,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                n_train=len(train),
                n_test=len(test),
                ic=ic,
                hit_rate=hr,
                sharpe_on_signals=shp,
                promoted=promoted,
            ))
            print(
                f"    fold {fold_idx:3d} h={horizon} "
                f"train={tr_s}..{tr_e} test={te_s}..{te_e} "
                f"IC={ic:+.4f} hit={hr:.3f} sharpe={shp:+.2f} "
                f"{'PROMOTED' if promoted else '----'}",
                flush=True,
            )
        except Exception as exc:
            import traceback
            print(f"    fold {fold_idx} h={horizon} FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue

    if not pred_frames:
        return pd.DataFrame(columns=["symbol", "ts", "prediction"]), records

    # For overlapping fold test windows, prefer predictions from the latest fold
    # (most recent model). Dedupe on (symbol, ts) keeping the last entry.
    preds = pd.concat(pred_frames, ignore_index=True)
    preds = preds.sort_values(["symbol", "ts", "fold"]).drop_duplicates(
        subset=["symbol", "ts"], keep="last"
    )
    return preds[["symbol", "ts", "prediction"]], records


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio simulation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StrategyCfg:
    id: str
    horizon: int
    max_hold_days: int
    weight: float
    max_positions: int
    max_position_pct: float
    max_exposure_pct: float


@dataclass
class OpenPosition:
    symbol: str
    strategy_id: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    peak_price: float
    hard_stop: float
    trailing_activation: float
    trailing_pct: float


@dataclass
class TradeRecord:
    strategy: str
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str


def _next_trading_day(prices_by_date: pd.DatetimeIndex, today: pd.Timestamp) -> Optional[pd.Timestamp]:
    nxt_idx = prices_by_date.searchsorted(today, side="right")
    if nxt_idx >= len(prices_by_date):
        return None
    return prices_by_date[nxt_idx]


def simulate_portfolio(
    panel: pd.DataFrame,
    predictions_by_horizon: Dict[int, pd.DataFrame],
    strategies: List[StrategyCfg],
    benchmark_prices: pd.Series,
    universe_ohlcv: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, List[TradeRecord], Dict[str, List[TradeRecord]]]:
    """Run the daily portfolio simulation across all three strategies.

    Returns:
        daily_nav: DataFrame indexed by date with columns
            [cash, positions_value, nav, day_nav, day_swing_nav, day_long_term_nav,
             spy_nav].
        all_trades: full list of closed trade records.
        trades_by_strategy: trades partitioned by strategy id.
    """
    # Build a per-symbol calendar lookup.
    sym_data: Dict[str, pd.DataFrame] = {}
    for sym, df in universe_ohlcv.items():
        d = df.copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        sym_data[sym] = d

    # Build a panel indexed by ts for fast daily slicing of features.
    panel = panel.copy()
    panel["ts"] = pd.to_datetime(panel["ts"]).dt.tz_localize(None)
    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)

    # ATR per (symbol, ts) — used for stop_logic. atr_14_pct is in the panel.
    atr_lookup = panel.set_index(["symbol", "ts"])[
        ["close", "atr_14_pct", "hist_vol_20d"]
    ]

    # Precompute rolling 20-day dollar volume per symbol once. The previous
    # implementation re-sliced sym_data[sym].loc[:today, ...] inside the daily
    # loop, which triggered pandas C-level access violations under heavy
    # repetition (Windows 0x80000003). One-shot rolling is faster too.
    adv_lookup: Dict[str, pd.Series] = {}
    for _sym, _df in sym_data.items():
        adv_lookup[_sym] = (_df["close"] * _df["volume"]).rolling(20).mean().fillna(0.0)

    # Trading calendar = union of all symbol date indexes in [TRADE_START, TRADE_END].
    all_dates = sorted({
        d for df in sym_data.values()
        for d in df.index
        if TRADE_START.tz_localize(None) <= d <= TRADE_END.tz_localize(None)
    })
    all_dates = pd.DatetimeIndex(all_dates)

    # Predictions joined on (symbol, ts) per horizon for quick lookup per day.
    preds_map: Dict[int, Dict[pd.Timestamp, pd.DataFrame]] = {}
    for h, pred_df in predictions_by_horizon.items():
        pdf = pred_df.copy()
        pdf["ts"] = pd.to_datetime(pdf["ts"]).dt.tz_localize(None)
        preds_map[h] = {
            ts: grp[["symbol", "prediction"]].reset_index(drop=True)
            for ts, grp in pdf.groupby("ts")
        }

    # State
    cash = INITIAL_CAPITAL
    positions: Dict[str, OpenPosition] = {}  # keyed by (strategy_id, symbol)
    open_trade_keys: Dict[Tuple[str, str], OpenPosition] = {}
    closed_trades: List[TradeRecord] = []
    closed_by_strategy: Dict[str, List[TradeRecord]] = {s.id: [] for s in strategies}

    limits = RiskLimits()
    dd_guard = DrawdownGuard(starting_capital=INITIAL_CAPITAL, limits=limits)

    nav_history: List[Dict] = []
    strategy_nav: Dict[str, float] = {s.id: INITIAL_CAPITAL * s.weight for s in strategies}
    strategy_cash: Dict[str, float] = {s.id: INITIAL_CAPITAL * s.weight for s in strategies}
    strategy_positions: Dict[str, Dict[str, OpenPosition]] = {s.id: {} for s in strategies}

    # Pending orders to execute at next-day open.
    pending_buys: List[Tuple[str, str, int, float]] = []  # (strategy_id, symbol, shares, target_notional)
    pending_sells: List[Tuple[str, str, str]] = []        # (strategy_id, symbol, reason)

    bench_series = benchmark_prices.copy()
    bench_series.index = pd.to_datetime(bench_series.index).tz_localize(None)
    # Anchor SPY benchmark at INITIAL_CAPITAL on the first simulation day.
    if not bench_series.empty and len(all_dates) > 0:
        anchor_date = all_dates[0]
        anchor_price = float(bench_series.asof(anchor_date))
        if anchor_price > 0:
            bench_norm = bench_series / anchor_price * INITIAL_CAPITAL
        else:
            bench_norm = bench_series * 0.0 + INITIAL_CAPITAL
    else:
        bench_norm = pd.Series(dtype=float)

    prev_nav = INITIAL_CAPITAL
    day_loss_pause = False
    print(f"  → Simulating {len(all_dates)} trading days...")

    for i, today in enumerate(all_dates):
        today_open_prices: Dict[str, float] = {}
        today_close_prices: Dict[str, float] = {}
        today_high: Dict[str, float] = {}
        today_low: Dict[str, float] = {}
        for sym, df in sym_data.items():
            if today in df.index:
                row = df.loc[today]
                today_open_prices[sym] = float(row["open"])
                today_close_prices[sym] = float(row["close"])
                today_high[sym] = float(row["high"])
                today_low[sym] = float(row["low"])

        # ─── 1) Execute pending orders at TODAY's open. ───────────────────────
        # Sells first to free capital.
        for strat_id, sym, reason in pending_sells:
            sp = strategy_positions[strat_id].get(sym)
            if sp is None or sym not in today_open_prices:
                continue
            fill_price = today_open_prices[sym] * (1 - TXN_COST_BPS / 10_000.0)
            proceeds = sp.shares * fill_price
            pnl = (fill_price - sp.entry_price) * sp.shares
            pnl_pct = (fill_price / sp.entry_price - 1) if sp.entry_price > 0 else 0.0
            hold = (today - sp.entry_date).days
            rec = TradeRecord(
                strategy=strat_id, symbol=sym,
                entry_date=sp.entry_date, entry_price=sp.entry_price,
                exit_date=today, exit_price=fill_price,
                shares=sp.shares, pnl=pnl, pnl_pct=pnl_pct,
                hold_days=hold, exit_reason=reason,
            )
            closed_trades.append(rec)
            closed_by_strategy[strat_id].append(rec)
            strategy_cash[strat_id] += proceeds
            del strategy_positions[strat_id][sym]
        pending_sells = []

        for strat_id, sym, shares, target_notional in pending_buys:
            if sym not in today_open_prices or shares <= 0:
                continue
            fill_price = today_open_prices[sym] * (1 + TXN_COST_BPS / 10_000.0)
            cost = shares * fill_price
            if cost > strategy_cash[strat_id]:
                shares = int(strategy_cash[strat_id] // fill_price)
                cost = shares * fill_price
                if shares <= 0:
                    continue
            # ATR + vol estimate for stop computation, from panel at *prior* date.
            try:
                feat_row = atr_lookup.loc[(sym, today)]
                atr_pct = float(feat_row["atr_14_pct"])
                vol_ann = float(feat_row["hist_vol_20d"])
            except KeyError:
                atr_pct, vol_ann = 2.0, 0.20
            if not np.isfinite(atr_pct) or atr_pct <= 0:
                atr_pct = 2.0
            if not np.isfinite(vol_ann) or vol_ann <= 0:
                vol_ann = 0.20
            atr_abs = fill_price * atr_pct / 100.0
            try:
                stops = compute_initial_stops(
                    entry_price=fill_price, atr=atr_abs, vol_annual=vol_ann,
                    atr_multiple=2.0, vol_daily_multiple=1.5,
                    trailing_activation_pct=0.02, trailing_pct=0.50,
                )
            except ValueError:
                continue
            pos = OpenPosition(
                symbol=sym, strategy_id=strat_id, shares=shares,
                entry_price=fill_price, entry_date=today,
                peak_price=fill_price,
                hard_stop=stops.hard_stop,
                trailing_activation=stops.trailing_activation_price,
                trailing_pct=stops.trailing_pct,
            )
            strategy_positions[strat_id][sym] = pos
            strategy_cash[strat_id] -= cost
        pending_buys = []

        # ─── 2) Mark-to-market, stops, time exits ─────────────────────────────
        for strat_cfg in strategies:
            to_close: List[Tuple[str, str]] = []
            for sym, pos in strategy_positions[strat_cfg.id].items():
                px = today_close_prices.get(sym)
                if px is None:
                    continue
                # Update trailing stop using today's high.
                hi = today_high.get(sym, px)
                new_stop, new_peak = update_trailing_stop(
                    current_price=hi,
                    entry_price=pos.entry_price,
                    peak_price=pos.peak_price,
                    hard_stop=pos.hard_stop,
                    trailing_pct=pos.trailing_pct,
                    trailing_activation_price=pos.trailing_activation,
                )
                pos.hard_stop = new_stop
                pos.peak_price = new_peak
                # Stop hit using today's low.
                lo = today_low.get(sym, px)
                if should_exit_on_stop(lo, pos.hard_stop):
                    to_close.append((sym, "stop_loss"))
                    continue
                # Time-based exit.
                hold = (today - pos.entry_date).days
                if hold >= strat_cfg.max_hold_days:
                    to_close.append((sym, "max_hold"))
            for sym, reason in to_close:
                pending_sells.append((strat_cfg.id, sym, reason))

        # ─── 3) Compute NAV at today's close ──────────────────────────────────
        total_positions_value = 0.0
        total_cash = 0.0
        per_strat_nav: Dict[str, float] = {}
        for strat_cfg in strategies:
            pos_val = 0.0
            for sym, pos in strategy_positions[strat_cfg.id].items():
                px = today_close_prices.get(sym, pos.entry_price)
                pos_val += pos.shares * px
            strat_nav_v = strategy_cash[strat_cfg.id] + pos_val
            per_strat_nav[strat_cfg.id] = strat_nav_v
            total_positions_value += pos_val
            total_cash += strategy_cash[strat_cfg.id]
        nav = total_cash + total_positions_value
        day_pnl = nav - prev_nav
        prev_nav = nav

        # ─── 4) Risk gates (drawdown, daily loss) ─────────────────────────────
        dd_state = dd_guard.update(nav)
        if day_pnl <= -5_000.0 and not day_loss_pause:
            day_loss_pause = True  # hard stop new entries for the day
        # Reset daily loss at next day.
        block_new_entries = dd_state.stop_new_entries or day_loss_pause
        if dd_state.close_all:
            # Force liquidation of everything next day.
            for strat_cfg in strategies:
                for sym in list(strategy_positions[strat_cfg.id].keys()):
                    pending_sells.append((strat_cfg.id, sym, "kill_switch"))

        bench_val = float(bench_norm.asof(today)) if not bench_norm.empty else np.nan
        nav_history.append({
            "date": today,
            "cash": total_cash,
            "positions_value": total_positions_value,
            "nav": nav,
            "spy_nav": bench_val,
            "day_pnl": day_pnl,
            "drawdown_level": int(dd_state.level.value),
            **{f"nav_{s.id}": per_strat_nav[s.id] for s in strategies},
        })

        # Reset day loss flag at end of day (next session starts fresh).
        day_loss_pause = False

        # ─── 5) Construct new orders for execution at next day's open ─────────
        if block_new_entries:
            continue
        nxt = _next_trading_day(all_dates, today)
        if nxt is None:
            continue

        for strat_cfg in strategies:
            pred_df = preds_map.get(strat_cfg.horizon, {}).get(today)
            if pred_df is None or pred_df.empty:
                continue
            # Top-quintile filter.
            thr = pred_df["prediction"].quantile(0.9)
            longs = pred_df[pred_df["prediction"] >= thr].sort_values(
                "prediction", ascending=False
            ).reset_index(drop=True)
            if longs.empty:
                continue

            current_positions = strategy_positions[strat_cfg.id]
            strat_nav_v = per_strat_nav[strat_cfg.id]
            strat_cash_v = strategy_cash[strat_cfg.id]
            max_new_notional = strat_nav_v * strat_cfg.max_exposure_pct
            current_positions_notional = sum(
                pos.shares * today_close_prices.get(sym, pos.entry_price)
                for sym, pos in current_positions.items()
            )
            available_notional = max(0.0, max_new_notional - current_positions_notional)

            n_open = len(current_positions)
            for _, row in longs.iterrows():
                sym = str(row["symbol"])
                if sym in current_positions:
                    continue
                if sym not in today_close_prices:
                    continue
                if n_open >= strat_cfg.max_positions:
                    break
                pred = float(row["prediction"])
                if pred <= 0:
                    continue
                # Volatility estimate from panel.
                try:
                    vol = float(atr_lookup.loc[(sym, today)]["hist_vol_20d"])
                except KeyError:
                    vol = 0.20
                if not np.isfinite(vol) or vol <= 0:
                    vol = 0.20
                # Liquidity proxy — precomputed rolling 20d $-volume.
                _adv_series = adv_lookup.get(sym)
                if _adv_series is not None and today in _adv_series.index:
                    liq_adv = float(_adv_series.at[today])
                else:
                    liq_adv = 5_000_000.0
                if not np.isfinite(liq_adv) or liq_adv <= 0:
                    liq_adv = 5_000_000.0
                # Convert predicted forward log-return to 5d-equivalent for Kelly.
                er_5d = pred * (5 / strat_cfg.horizon)
                # Confidence proxy: rank percentile.
                rank_pct = float(longs.loc[longs["symbol"] == sym].index[0]) / max(
                    1, len(longs) - 1
                )
                confidence = 1.0 - rank_pct  # top-ranked → high confidence
                sizing = kelly_position_size(
                    SizingInputs(
                        expected_return_5d=er_5d,
                        vol_estimate_annual=vol,
                        confidence=confidence,
                        liquidity_adv_usd=liq_adv,
                        regime="BULL",
                        recent_ic=0.03,
                    ),
                    max_position_pct=min(strat_cfg.max_position_pct, 0.18),
                    kelly_fraction=0.25,
                )
                target_pct = sizing.target_pct
                if target_pct <= 0:
                    continue
                target_notional = strat_nav_v * target_pct
                # Cap by remaining exposure and available cash.
                target_notional = min(target_notional, available_notional, strat_cash_v)
                if target_notional < 2_000.0:
                    continue
                # Use today's close as price estimate for share calc; fill at next open.
                price_est = today_close_prices[sym]
                shares = shares_from_target(target_notional / strat_nav_v, strat_nav_v, price_est)
                if shares <= 0:
                    continue
                pending_buys.append((strat_cfg.id, sym, shares, target_notional))
                available_notional -= shares * price_est
                strat_cash_v -= shares * price_est
                n_open += 1
                if available_notional <= 2_000.0 or strat_cash_v <= 2_000.0:
                    break

    nav_df = pd.DataFrame(nav_history)
    if not nav_df.empty:
        nav_df = nav_df.set_index("date")
    return nav_df, closed_trades, closed_by_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def annualize_return(nav_series: pd.Series) -> float:
    if nav_series.empty:
        return 0.0
    start, end = float(nav_series.iloc[0]), float(nav_series.iloc[-1])
    if start <= 0:
        return 0.0
    n_days = (nav_series.index[-1] - nav_series.index[0]).days
    if n_days <= 0:
        return 0.0
    return (end / start) ** (365.25 / n_days) - 1.0


def compute_portfolio_metrics(
    nav_series: pd.Series, trades: List[TradeRecord]
) -> Dict[str, "float | str | int"]:
    nav_series = nav_series.dropna()
    if nav_series.empty:
        return {}
    daily_ret = nav_series.pct_change().dropna()
    rf_daily = RF_ANNUAL / 252.0
    excess = daily_ret - rf_daily
    cagr = annualize_return(nav_series)
    total_ret = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1.0)
    mdd = max_drawdown(nav_series)
    peak = nav_series.cummax()
    dd_curve = (nav_series - peak) / peak
    mdd_date = dd_curve.idxmin()
    shp = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    sortino = sortino_ratio(excess)
    calmar = calmar_ratio(daily_ret)

    pnls = pd.Series([t.pnl for t in trades])
    wr = win_rate(pnls) if not pnls.empty else 0.0
    avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0

    return {
        "total_return_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": mdd * 100,
        "max_drawdown_date": str(mdd_date.date()) if mdd_date is not None else "n/a",
        "sharpe": shp,
        "sortino": float(sortino),
        "calmar": float(calmar),
        "win_rate_pct": float(wr * 100),
        "avg_hold_days": avg_hold,
        "n_trades": len(trades),
    }


def annual_returns_table(
    nav_series: pd.Series, benchmark: pd.Series
) -> pd.DataFrame:
    rows = []
    for yr, grp in nav_series.groupby(nav_series.index.year):
        if len(grp) < 2:
            continue
        z_ret = float(grp.iloc[-1] / grp.iloc[0] - 1.0)
        bench_year = benchmark[benchmark.index.year == yr]
        b_ret = float(bench_year.iloc[-1] / bench_year.iloc[0] - 1.0) if len(bench_year) >= 2 else np.nan
        rows.append({
            "year": int(yr),
            "zeus_return_pct": z_ret * 100,
            "spy_return_pct": b_ret * 100 if not np.isnan(b_ret) else None,
            "alpha_pct": (z_ret - b_ret) * 100 if not np.isnan(b_ret) else None,
        })
    return pd.DataFrame(rows)


def save_outputs(
    nav_df: pd.DataFrame,
    trades_all: List[TradeRecord],
    trades_by_strategy: Dict[str, List[TradeRecord]],
    fold_records: List[FoldRecord],
    strategies: List[StrategyCfg],
) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Daily NAV.
    out_nav = nav_df.reset_index().rename(columns={"date": "date"})
    out_nav["date"] = pd.to_datetime(out_nav["date"]).dt.strftime("%Y-%m-%d")
    out_nav.to_csv(ARTIFACTS / "results.csv", index=False)

    # Trades.
    trade_rows = [{
        "strategy": t.strategy, "symbol": t.symbol,
        "entry_date": t.entry_date.strftime("%Y-%m-%d"),
        "entry_price": t.entry_price,
        "exit_date": t.exit_date.strftime("%Y-%m-%d"),
        "exit_price": t.exit_price,
        "shares": t.shares, "pnl": t.pnl, "pnl_pct": t.pnl_pct * 100,
        "hold_days": t.hold_days, "exit_reason": t.exit_reason,
    } for t in trades_all]
    pd.DataFrame(trade_rows).to_csv(ARTIFACTS / "trades.csv", index=False)

    # Walk-forward folds.
    fold_rows = [{
        "horizon": f.horizon, "fold": f.fold,
        "train_start": f.train_start.isoformat(),
        "train_end": f.train_end.isoformat(),
        "test_start": f.test_start.isoformat(),
        "test_end": f.test_end.isoformat(),
        "n_train": f.n_train, "n_test": f.n_test,
        "ic": f.ic, "hit_rate": f.hit_rate,
        "sharpe_on_signals": f.sharpe_on_signals,
        "promoted": f.promoted,
    } for f in fold_records]
    pd.DataFrame(fold_rows).to_csv(ARTIFACTS / "fold_metrics.csv", index=False)

    # Equity curve PNG.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(nav_df.index, nav_df["nav"], label="Zeus NAV", linewidth=1.5)
        if "spy_nav" in nav_df.columns:
            ax.plot(nav_df.index, nav_df["spy_nav"], label="SPY (buy-hold)", linewidth=1.5, alpha=0.7)
        ax.set_title("Zeus Backtest: 2015-2025 (initial $100,000)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "equity_curve.png", dpi=120)
        plt.close()
    except Exception as exc:
        print(f"  ! Equity curve plot failed: {exc}")

    # Summary text.
    bench = nav_df["spy_nav"].dropna() if "spy_nav" in nav_df.columns else pd.Series(dtype=float)
    nav_series = nav_df["nav"].dropna()
    port_metrics = compute_portfolio_metrics(nav_series, trades_all)
    yearly = annual_returns_table(nav_series, bench)

    promoted_count = sum(1 for f in fold_records if f.promoted)
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(" ZEUS BACKTEST 2015-2025 — SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  Initial capital      : ${INITIAL_CAPITAL:,.0f}")
    lines.append(f"  Final NAV            : ${float(nav_series.iloc[-1]):,.0f}")
    lines.append(f"  Period               : {nav_series.index[0].date()} → {nav_series.index[-1].date()}")
    lines.append("")
    lines.append("  Portfolio-Level Metrics")
    lines.append("  " + "-" * 60)
    for k, v in port_metrics.items():
        if isinstance(v, float):
            lines.append(f"    {k:24s} : {v:>10.3f}")
        else:
            lines.append(f"    {k:24s} : {v}")
    lines.append("")
    lines.append("  Per-Strategy Breakdown")
    lines.append("  " + "-" * 60)
    for s in strategies:
        col = f"nav_{s.id}"
        if col not in nav_df.columns:
            continue
        sub = nav_df[col].dropna()
        if sub.empty:
            continue
        sm = compute_portfolio_metrics(sub, trades_by_strategy.get(s.id, []))
        lines.append(f"    [{s.id}]  weight={s.weight:.2f}  horizon={s.horizon}d")
        for k in (
            "total_return_pct", "cagr_pct", "max_drawdown_pct",
            "sharpe", "sortino", "calmar",
            "win_rate_pct", "avg_hold_days", "n_trades",
        ):
            if k in sm:
                v = sm[k]
                fmt = f"{v:>10.3f}" if isinstance(v, float) else f"{v}"
                lines.append(f"      {k:22s} : {fmt}")
        lines.append("")
    lines.append("  Annual Returns")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'year':>6s}  {'zeus%':>10s}  {'spy%':>10s}  {'alpha%':>10s}")
    for _, row in yearly.iterrows():
        spy_v = row["spy_return_pct"] if row["spy_return_pct"] is not None else float("nan")
        alpha_v = row["alpha_pct"] if row["alpha_pct"] is not None else float("nan")
        lines.append(
            f"    {int(row['year']):>6d}  {row['zeus_return_pct']:>10.2f}  "
            f"{spy_v:>10.2f}  {alpha_v:>10.2f}"
        )
    lines.append("")
    lines.append("  Walk-Forward Summary")
    lines.append("  " + "-" * 60)
    lines.append(f"    Total folds          : {len(fold_records)}")
    lines.append(f"    Folds promoted       : {promoted_count}")
    by_h: Dict[int, List[FoldRecord]] = {}
    for f in fold_records:
        by_h.setdefault(f.horizon, []).append(f)
    for h, fs in sorted(by_h.items()):
        prom = sum(1 for f in fs if f.promoted)
        mean_ic = float(np.mean([f.ic for f in fs])) if fs else 0.0
        mean_hit = float(np.mean([f.hit_rate for f in fs])) if fs else 0.0
        lines.append(
            f"    h={h:>3d}d  folds={len(fs):>3d}  promoted={prom:>3d}  "
            f"mean_IC={mean_ic:+.4f}  mean_hit={mean_hit:.3f}"
        )
    lines.append("")
    lines.append("  Notes")
    lines.append("  " + "-" * 60)
    lines.append("    • Fundamentals zeroed: " + ", ".join(_ZEROED_FUNDAMENTAL_FEATURES))
    lines.append("    • Macro zeroed: " + ", ".join(_ZEROED_MACRO_FEATURES))
    lines.append("    • Macro sourced from yfinance: ^VIX, ^TNX, ^IRX, SPY")
    lines.append("    • Sector limits disabled (no point-in-time sector data)")
    lines.append("    • Random seed: 42  (deterministic)")
    lines.append("    • Fills modeled at next-day open + 5 bps one-way cost")
    lines.append("=" * 78)

    summary_text = "\n".join(lines)
    (ARTIFACTS / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Zeus 2015-2025 ML-only backtest")
    parser.add_argument(
        "--max-symbols", type=int, default=None,
        help="Cap universe size to this many symbols (default: full S&P 500 ~503).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: 20 symbols, abbreviated date range.",
    )
    args = parser.parse_args()

    if args.quick:
        global TRADE_START, TRADE_END
        TRADE_START = pd.Timestamp("2022-01-01", tz="UTC")
        TRADE_END = pd.Timestamp("2023-06-30", tz="UTC")
        data_start = "2020-01-01"
    else:
        data_start = DATA_START

    print("[1/6] Fetching S&P 500 universe from Wikipedia")
    universe = fetch_sp500_universe()
    if args.quick:
        universe = universe[:20]
    elif args.max_symbols is not None:
        universe = universe[: args.max_symbols]

    print(f"  → Universe size: {len(universe)}; data window: {data_start} → {DATA_END}")
    ohlcv, first_trade_date = download_ohlcv(universe, data_start, DATA_END)
    macro_ohlcv, _ = download_ohlcv([BENCHMARK] + MACRO_TICKERS, data_start, DATA_END)
    if not ohlcv:
        print("ERROR: no OHLCV downloaded; aborting.")
        return 1

    print("[2/6] Filtering universe (price, ADV)")
    ohlcv = apply_universe_filters(ohlcv)
    first_trade_date = {s: first_trade_date[s] for s in ohlcv if s in first_trade_date}
    print(f"  → {len(ohlcv)} symbols pass filter")
    if not ohlcv:
        print("ERROR: universe empty after filter.")
        return 1

    print("[3/6] Feature engineering")
    macro_panel = compute_macro_panel(macro_ohlcv)
    panel = build_feature_panel(ohlcv, macro_panel)

    # Raw forward log-return labels (3d / 5d / 20d). Reverted from
    # triple-barrier — that experiment ran into both Windows-specific heap
    # crashes (math.log inner loop fixed) AND a sibling-label leakage path
    # where tb_return_h{H} / tb_days_to_hit_h{H} were entering the feature
    # matrix as proxies for the target (driving fake IC 0.78 / Sharpe 7).
    # Raw returns give honest, comparable metrics to run #3.
    labels = compute_forward_returns(ohlcv, horizons=(3, 5, 20))
    labels["ts"] = pd.to_datetime(labels["ts"]).dt.tz_localize(None)
    panel = panel.merge(labels, on=["symbol", "ts"], how="left")
    feature_cols = select_feature_columns(panel)
    print(f"  → panel rows: {len(panel):,}  features: {len(feature_cols)}")

    print("[4/6] Walk-forward training (per horizon)")
    import gc
    predictions_by_horizon: Dict[int, pd.DataFrame] = {}
    all_fold_records: List[FoldRecord] = []
    for horizon in (3, 5, 20):
        label_col = f"fwd_{horizon}d_return"
        if label_col not in panel.columns:
            continue
        print(f"  ── Horizon {horizon}d ──", flush=True)
        preds, records = run_walk_forward_for_horizon(
            panel, feature_cols, label_col, horizon,
            first_trade_date=first_trade_date,
        )
        predictions_by_horizon[horizon] = preds
        all_fold_records.extend(records)
        # Persist partial outputs so we don't lose results on a later crash.
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        preds.to_csv(ARTIFACTS / f"predictions_h{horizon}.csv", index=False)
        pd.DataFrame([{
            "horizon": f.horizon, "fold": f.fold,
            "train_start": f.train_start.isoformat(),
            "train_end": f.train_end.isoformat(),
            "test_start": f.test_start.isoformat(),
            "test_end": f.test_end.isoformat(),
            "n_train": f.n_train, "n_test": f.n_test,
            "ic": f.ic, "hit_rate": f.hit_rate,
            "sharpe_on_signals": f.sharpe_on_signals,
            "promoted": f.promoted,
        } for f in all_fold_records]).to_csv(ARTIFACTS / "fold_metrics.csv", index=False)
        gc.collect()
        print(f"  ── Horizon {horizon}d done ({len(records)} folds, {len(preds)} preds) ──", flush=True)

    print("[5/6] Portfolio simulation")
    strategies = [
        StrategyCfg("day", horizon=3, max_hold_days=5, weight=0.10,
                    max_positions=8, max_position_pct=0.08, max_exposure_pct=0.45),
        StrategyCfg("swing", horizon=5, max_hold_days=10, weight=0.60,
                    max_positions=10, max_position_pct=0.10, max_exposure_pct=0.50),
        StrategyCfg("long_term", horizon=20, max_hold_days=170, weight=0.30,
                    max_positions=8, max_position_pct=0.12, max_exposure_pct=0.40),
    ]
    bench_close = macro_ohlcv.get(BENCHMARK, pd.DataFrame()).get("close", pd.Series(dtype=float))
    nav_df, trades_all, trades_by_strat = simulate_portfolio(
        panel, predictions_by_horizon, strategies, bench_close, ohlcv,
    )
    print(f"  → simulation done. {len(nav_df)} trading days, {len(trades_all)} closed trades.")

    print("[6/6] Writing outputs")
    save_outputs(nav_df, trades_all, trades_by_strat, all_fold_records, strategies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
