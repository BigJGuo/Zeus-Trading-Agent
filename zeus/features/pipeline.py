"""FeaturePipeline: orchestrates technical, fundamental, macro, and cross-sectional features."""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Optional, cast

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.orm import Session

from zeus.config.settings import get_settings
from zeus.data.storage.database import FeaturesDaily, FundamentalsCache, OHLCVDaily
from zeus.features.cross_sectional import compute_cross_sectional_features
from zeus.features.fundamental import compute_fundamental_features
from zeus.features.interactions import add_macro_interactions
from zeus.features.macro import compute_macro_features
from zeus.features.technical import compute_technical_features

logger = structlog.get_logger()

_LOOKBACK_DAYS = 250


class FeaturePipeline:
    def __init__(self, feature_version: str = "v1"):
        self.feature_version = feature_version
        self._settings = get_settings()

    def compute_features_for_date(
        self,
        session: Session,
        as_of: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Compute full feature set for all symbols on a given date.

        Loads OHLCV from DB, computes technical features, merges fundamentals
        and macro features, adds cross-sectional ranks, saves parquet snapshot,
        and upserts to FeaturesDaily table.

        Returns:
            DataFrame with one row per symbol and all features as columns.
        """
        as_of_dt = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)
        symbol_frames: list[dict] = []

        n_failed = 0
        for symbol in symbols:
            # One bad symbol (e.g. newly-listed with insufficient history,
            # corrupt OHLCV, etc.) must not kill the whole batch — otherwise
            # the planner falls back to a stale parquet snapshot for days.
            try:
                tech_row = self._compute_latest_technical(session, symbol, as_of_dt)
            except Exception as e:
                n_failed += 1
                logger.warning(
                    "technical_features_failed", symbol=symbol,
                    as_of=str(as_of), error=str(e),
                )
                continue
            if tech_row is not None:
                tech_row["symbol"] = symbol
                tech_row["ts"] = as_of_dt
                symbol_frames.append(tech_row)
            else:
                logger.warning("no_ohlcv_data", symbol=symbol, as_of=str(as_of))
        if n_failed:
            logger.warning(
                "feature_pipeline_partial",
                as_of=str(as_of), n_failed=n_failed, n_ok=len(symbol_frames),
            )

        if not symbol_frames:
            logger.warning("no_technical_features_computed", as_of=str(as_of))
            return pd.DataFrame()

        panel = pd.DataFrame(symbol_frames).reset_index(drop=True)

        # ── Sector map from fundamentals cache ────────────────────────────────
        sector_map = self._load_sector_map(session, symbols)

        # ── Cross-sectional ranks ─────────────────────────────────────────────
        panel = compute_cross_sectional_features(panel, sector_map)

        # ── Fundamental features ──────────────────────────────────────────────
        fund_rows = self._load_fundamentals(session, symbols)
        if fund_rows:
            fund_df = pd.DataFrame(fund_rows)
            panel = panel.merge(fund_df, on="symbol", how="left")

        # ── Macro features (broadcast to all rows) ────────────────────────────
        macro = compute_macro_features(as_of)
        for k, v in macro.items():
            panel[k] = v

        # ── Macro × per-symbol interactions ────────────────────────────────────
        # The bare macro columns are constants across symbols on a given day —
        # trees can't split them cross-sectionally. The interactions are the
        # actual signal: e.g. momentum_20d × vix_level lets the model learn
        # that momentum reverses in vol spikes.
        panel = add_macro_interactions(panel)

        # ── Validation & logging ──────────────────────────────────────────────
        nan_counts = panel.isna().sum()
        high_nan = nan_counts[nan_counts > len(panel) * 0.5]
        if not high_nan.empty:
            logger.warning("high_nan_columns", columns=high_nan.to_dict(), as_of=str(as_of))

        n_before = len(panel)
        required_cols = ["symbol", "ts", "close", "rsi_14", "sma_20"]
        existing_required = [c for c in required_cols if c in panel.columns]
        panel_clean = panel.dropna(subset=existing_required)
        n_dropped = n_before - len(panel_clean)
        if n_dropped > 0:
            logger.warning("rows_dropped_missing_required", n_dropped=n_dropped, as_of=str(as_of))

        logger.info(
            "features_computed",
            as_of=str(as_of),
            n_symbols=len(panel_clean),
            n_features=len(panel_clean.columns),
            version=self.feature_version,
        )

        # ── Save parquet snapshot ─────────────────────────────────────────────
        self._save_parquet(panel_clean, as_of)

        # ── Upsert to DB ──────────────────────────────────────────────────────
        self._upsert_features_daily(session, panel_clean, as_of_dt)

        return panel_clean

    def _compute_latest_technical(
        self,
        session: Session,
        symbol: str,
        as_of_dt: datetime,
    ) -> Optional[dict]:
        rows = (
            session.query(OHLCVDaily)
            .filter(
                OHLCVDaily.symbol == symbol,
                OHLCVDaily.ts <= as_of_dt,
            )
            .order_by(OHLCVDaily.ts.desc())
            .limit(_LOOKBACK_DAYS)
            .all()
        )
        if not rows:
            return None

        rows_sorted = sorted(rows, key=lambda r: r.ts)
        df = pd.DataFrame(
            {
                "open": [r.open for r in rows_sorted],
                "high": [r.high for r in rows_sorted],
                "low": [r.low for r in rows_sorted],
                "close": [r.close for r in rows_sorted],
                "volume": [r.volume for r in rows_sorted],
            },
            index=[r.ts for r in rows_sorted],
        )
        df = df.dropna(subset=["close"])
        if df.empty:
            return None

        tech_df = compute_technical_features(df)
        if tech_df.empty:
            return None

        latest = tech_df.iloc[-1].to_dict()
        latest["close"] = df["close"].iloc[-1]
        latest["volume"] = df["volume"].iloc[-1]

        # Precompute return columns needed for cross-sectional ranking
        if len(df) >= 6:
            latest["return_5d"] = float((df["close"].iloc[-1] / df["close"].iloc[-6]) - 1)
        if len(df) >= 21:
            latest["return_20d"] = float((df["close"].iloc[-1] / df["close"].iloc[-21]) - 1)

        return latest

    def _load_sector_map(self, session: Session, symbols: list[str]) -> dict[str, str]:
        rows = (
            session.query(FundamentalsCache.symbol, FundamentalsCache.sector)
            .filter(FundamentalsCache.symbol.in_(symbols))
            .all()
        )
        return {r.symbol: (r.sector or "Unknown") for r in rows}

    def _load_fundamentals(self, session: Session, symbols: list[str]) -> list[dict]:
        rows = session.query(FundamentalsCache).filter(FundamentalsCache.symbol.in_(symbols)).all()
        result = []
        for row in rows:
            symbol = cast(str, row.symbol)
            raw = cast(dict, row.raw_info or {})
            feats = compute_fundamental_features(symbol, raw)
            feats["symbol"] = symbol
            result.append(feats)
        return result

    def _save_parquet(self, df: pd.DataFrame, as_of: date) -> None:
        artifacts_path = self._settings.artifacts_path
        out_dir = os.path.join(artifacts_path, "features", str(as_of))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"features_{self.feature_version}.parquet")
        df.to_parquet(path, engine="pyarrow", index=True)
        logger.info("parquet_snapshot_saved", path=path, n_rows=len(df))

    def load_snapshot(self, features_dir: str) -> pd.DataFrame:
        """Load the parquet snapshot written by `compute_features_for_date`.

        `features_dir` is the directory for a given as_of date (the caller
        typically passes `artifacts/features/YYYY-MM-DD`). Returns an empty
        DataFrame if the file is missing — callers should branch on .empty.
        """
        path = os.path.join(features_dir, f"features_{self.feature_version}.parquet")
        if not os.path.exists(path):
            logger.warning("feature_snapshot_missing", path=path)
            return pd.DataFrame()
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("feature_snapshot_loaded", path=path, n_rows=len(df))
        return df

    def _upsert_features_daily(
        self,
        session: Session,
        df: pd.DataFrame,
        as_of_dt: datetime,
    ) -> None:
        for _, row in df.iterrows():
            symbol = row.get("symbol")
            if not symbol:
                continue

            feature_dict = {
                k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in row.items()
                if k not in ("symbol", "ts")
            }

            existing = (
                session.query(FeaturesDaily)
                .filter(
                    FeaturesDaily.symbol == symbol,
                    FeaturesDaily.feature_date == as_of_dt,
                    FeaturesDaily.feature_version == self.feature_version,
                )
                .first()
            )
            if existing:
                setattr(existing, "features", feature_dict)
            else:
                session.add(
                    FeaturesDaily(
                        symbol=symbol,
                        feature_date=as_of_dt,
                        feature_version=self.feature_version,
                        features=feature_dict,
                    )
                )

        session.commit()
        logger.info("features_daily_upserted", n_symbols=len(df), as_of=str(as_of_dt))
