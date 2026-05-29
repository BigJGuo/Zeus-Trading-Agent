"""Candidate universe builders used by the research agents.

Each research agent consumes a `List[Dict[str, Any]]` of per-symbol feature
rows (close, gap, 20d_change, rvol, etc.). These are built here from the
existing `ohlcv_daily` table — no new ingestion needed for tomorrow's
session.

Cadence (all ET):
  - 04:30: `build_day_premarket_candidates` → input to day_research premarket
  - 19:00: `build_swing_eod_candidates`      → input to swing_research EOD
  - Sunday 09:30: `build_swing_weekly_universe` → input to weekly watchlist
  - Ad-hoc: `build_long_term_universe` → input to LT deep-dive target picker

If the database has no recent bars the builders return an empty list —
callers handle the no-candidates case gracefully (agent records an alert).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zeus.data.storage.database import OHLCVDaily, Position, get_session_factory

log = structlog.get_logger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _recent_bars(
    session: Session,
    *,
    symbols: Optional[List[str]] = None,
    days: int = 30,
    min_volume: float = 0.0,
) -> Dict[str, List[OHLCVDaily]]:
    """Return {symbol: [bars DESC by ts]} for the last `days` trading days."""
    since = datetime.now(timezone.utc) - timedelta(days=days + 5)
    stmt = (
        select(OHLCVDaily)
        .where(OHLCVDaily.ts >= since)
        .order_by(OHLCVDaily.symbol, OHLCVDaily.ts.desc())
    )
    if symbols:
        stmt = stmt.where(OHLCVDaily.symbol.in_(symbols))
    rows = session.execute(stmt).scalars().all()

    by_sym: Dict[str, List[OHLCVDaily]] = {}
    for r in rows:
        if r.volume is not None and r.volume < min_volume:
            continue
        by_sym.setdefault(r.symbol, []).append(r)
    return by_sym


def _active_universe(session: Session, lookback_days: int = 5) -> List[str]:
    """Symbols with at least one bar in the last `lookback_days`."""
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    stmt = (
        select(OHLCVDaily.symbol)
        .where(OHLCVDaily.ts >= since)
        .group_by(OHLCVDaily.symbol)
    )
    return [row[0] for row in session.execute(stmt).all()]


# ─── Day-research: pre-market candidates ──────────────────────────────────────


def build_day_premarket_candidates(
    *, max_candidates: int = 40, min_gap_pct: float = 0.02,
    min_dollar_volume: float = 5_000_000.0,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """Gap scanner: rank yesterday's close vs day-before close, flag RVOL
    spikes. LLM then narrows to a Top-10 brief list."""

    def _run(s: Session) -> List[Dict[str, Any]]:
        universe = _active_universe(s)
        bars = _recent_bars(s, symbols=universe, days=25)
        out: List[Dict[str, Any]] = []
        for symbol, series in bars.items():
            if len(series) < 2:
                continue
            today = series[0]
            prior = series[1]
            if not (today.close and prior.close and today.volume):
                continue
            gap_pct = (float(today.close) - float(prior.close)) / float(prior.close)
            dollar_vol = float(today.close) * float(today.volume)
            if dollar_vol < min_dollar_volume:
                continue
            if abs(gap_pct) < min_gap_pct:
                continue
            # 20d avg volume for RVOL
            vols = [float(b.volume) for b in series[:20] if b.volume]
            avg_vol = (sum(vols) / len(vols)) if vols else 0.0
            rvol = (float(today.volume) / avg_vol) if avg_vol > 0 else 0.0
            out.append({
                "symbol": symbol,
                "close": float(today.close),
                "gap_pct": round(gap_pct, 4),
                "rvol": round(rvol, 2),
                "dollar_volume": round(dollar_vol, 0),
                "ts": today.ts.isoformat() if today.ts else None,
            })
        out.sort(key=lambda r: (abs(r["gap_pct"]), r["rvol"]), reverse=True)
        return out[:max_candidates]

    if session is not None:
        return _run(session)
    with get_session_factory()() as s:
        return _run(s)


# ─── Swing-research: EOD candidates ──────────────────────────────────────────


def build_swing_eod_candidates(
    *, max_candidates: int = 60, min_dollar_volume: float = 3_000_000.0,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """End-of-day scan: 20d % change + distance from 20d high as a crude
    trend/pullback signal for the swing research brief."""

    def _run(s: Session) -> List[Dict[str, Any]]:
        universe = _active_universe(s)
        bars = _recent_bars(s, symbols=universe, days=40)
        out: List[Dict[str, Any]] = []
        for symbol, series in bars.items():
            if len(series) < 20:
                continue
            today = series[0]
            ref = series[19]
            if not (today.close and ref.close and today.volume):
                continue
            dollar_vol = float(today.close) * float(today.volume)
            if dollar_vol < min_dollar_volume:
                continue
            change_20d = (float(today.close) - float(ref.close)) / float(ref.close)
            highs = [float(b.high or b.close or 0) for b in series[:20]]
            window_high = max(highs) if highs else float(today.close)
            pullback_pct = (window_high - float(today.close)) / window_high if window_high > 0 else 0.0
            setup_hint = "breakout" if change_20d > 0.05 and pullback_pct < 0.02 else (
                "pullback" if change_20d > 0.02 and pullback_pct >= 0.03 else "watch"
            )
            out.append({
                "symbol": symbol,
                "close": float(today.close),
                "change_20d": round(change_20d, 4),
                "pullback_from_high": round(pullback_pct, 4),
                "setup_hint": setup_hint,
                "dollar_volume": round(dollar_vol, 0),
            })
        out.sort(
            key=lambda r: (abs(r["change_20d"]) + r["pullback_from_high"]),
            reverse=True,
        )
        return out[:max_candidates]

    if session is not None:
        return _run(session)
    with get_session_factory()() as s:
        return _run(s)


def build_swing_weekly_universe(
    *, limit: int = 150, session: Optional[Session] = None,
) -> List[str]:
    """Sunday: active universe sorted by 30-day dollar volume."""

    def _run(s: Session) -> List[str]:
        since = datetime.now(timezone.utc) - timedelta(days=35)
        stmt = (
            select(
                OHLCVDaily.symbol,
                func.avg(OHLCVDaily.close * OHLCVDaily.volume).label("adv"),
            )
            .where(OHLCVDaily.ts >= since)
            .group_by(OHLCVDaily.symbol)
            .order_by(func.avg(OHLCVDaily.close * OHLCVDaily.volume).desc())
            .limit(limit)
        )
        return [row[0] for row in s.execute(stmt).all()]

    if session is not None:
        return _run(session)
    with get_session_factory()() as s:
        return _run(s)


# ─── Long-term research: deep-dive picker ────────────────────────────────────


def pick_long_term_deep_dive_target(
    *,
    force_symbol: Optional[str] = None,
    session: Optional[Session] = None,
) -> Optional[str]:
    """Heuristic target for the next deep-dive:

      1. If `force_symbol` is passed, use that.
      2. Else pick a currently open long_term position with no memo in the
         last 10 days (thesis refresh).
      3. Else pick the most-liquid name from the universe that has not been
         deep-dived in the last 30 days.
    """
    from zeus.agents.journal import query_journal

    def _run(s: Session) -> Optional[str]:
        if force_symbol:
            return force_symbol
        # (2) stale holdings
        open_lt = s.execute(
            select(Position.symbol, Position.entry_ts)
            .where(Position.strategy_id == "long_term")
            .where(Position.strategy_shares > 0)
        ).all()
        now = datetime.now(timezone.utc)
        for sym, _ in open_lt:
            recent = query_journal(
                agent_id="long_term_research", kind="memo",
                symbol=sym, since=now - timedelta(days=10), limit=1,
                session=s,
            )
            if not recent:
                return sym
        # (3) fallback — top-ADV name with no memo in 30d.
        since = now - timedelta(days=35)
        stmt = (
            select(
                OHLCVDaily.symbol,
                func.avg(OHLCVDaily.close * OHLCVDaily.volume).label("adv"),
            )
            .where(OHLCVDaily.ts >= since)
            .group_by(OHLCVDaily.symbol)
            .order_by(func.avg(OHLCVDaily.close * OHLCVDaily.volume).desc())
            .limit(50)
        )
        for sym, _adv in s.execute(stmt).all():
            recent = query_journal(
                agent_id="long_term_research", kind="memo",
                symbol=sym, since=now - timedelta(days=30), limit=1,
                session=s,
            )
            if not recent:
                return sym
        return None

    if session is not None:
        return _run(session)
    with get_session_factory()() as s:
        return _run(s)


# ─── Predictions fetch for trader agents ─────────────────────────────────────


def build_trader_predictions(
    strategy_context,
    *,
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    """Predictions the trader agent will weigh against its paired research.

    Preferred source: the `ctx.current_plan` that the 21:00 model-based
    planning job has already built (entries already carry shares / target /
    stop / signal_score from PortfolioConstructor). We simply expose them
    in the shape the LLM expects. If no plan is attached (e.g. agents start
    before the first 21:00 tick) we fall back to an empty list so the trader
    records an alert instead of blindly opening positions.
    """
    plan = getattr(strategy_context, "current_plan", None)
    if plan is None or not getattr(plan, "entries", None):
        return []
    out: List[Dict[str, Any]] = []
    for e in plan.entries[:top_k]:
        out.append({
            "symbol": e.get("symbol"),
            "shares_model_proposed": int(e.get("shares") or 0),
            "target_price": float(e.get("target_price") or 0.0),
            "stop_price": float(e.get("stop") or 0.0),
            "score": float(e.get("signal_score") or 0.0),
            "sector": e.get("sector"),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out
