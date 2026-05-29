"""Tool definitions exposed to LLM agents.

Tools are declared in two parts:

  1. A JSONSchema spec (`<tool>_SPEC`) passed to Anthropic as part of the
     `tools=` argument. Describes name/args/description so the model knows
     when to invoke it.
  2. A Python handler callable that implements the tool. Handlers get the
     parsed `input` dict from the model as kwargs and return a Python value
     (dict/list/str). The client serializes non-str returns to JSON.

Tools are constructed by `build_tool_bundle(context)` — the bundle binds
per-call dependencies (broker, paired_research_id, ...) so the resulting
handlers can stay context-free from the AgentLoop's perspective.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from zeus.agents.journal import query_journal

log = structlog.get_logger(__name__)


# ─── Tool spec objects ────────────────────────────────────────────────────────
# Exported as module-level constants so tests / prompts can reference them.

GET_OHLCV_SPEC: Dict[str, Any] = {
    "name": "get_ohlcv",
    "description": (
        "Fetch OHLCV bars for a symbol. `timeframe` is '1D' (daily) or "
        "'5Min'/'15Min' (intraday). `lookback_bars` caps how many recent "
        "bars to return (max 500)."
    ),
    "input_schema": {
        "type": "object",
        "required": ["symbol", "timeframe"],
        "properties": {
            "symbol": {"type": "string"},
            "timeframe": {"type": "string", "enum": ["1D", "5Min", "15Min"]},
            "lookback_bars": {"type": "integer", "default": 60, "maximum": 500},
        },
    },
}

GET_NEWS_SPEC: Dict[str, Any] = {
    "name": "get_news",
    "description": (
        "Fetch news headlines for a symbol in the last `hours` hours "
        "(default 24h). Returns list of {ts, headline, source, url}."
    ),
    "input_schema": {
        "type": "object",
        "required": ["symbol"],
        "properties": {
            "symbol": {"type": "string"},
            "hours": {"type": "integer", "default": 24, "maximum": 168},
        },
    },
}

GET_FUNDAMENTALS_SPEC: Dict[str, Any] = {
    "name": "get_fundamentals",
    "description": (
        "Fetch cached fundamentals (PE, PB, margins, growth, ...) for a "
        "symbol from the FundamentalsCache table."
    ),
    "input_schema": {
        "type": "object",
        "required": ["symbol"],
        "properties": {"symbol": {"type": "string"}},
    },
}

GET_REGIME_SPEC: Dict[str, Any] = {
    "name": "get_regime",
    "description": (
        "Return the current rule-based market regime "
        "('bull' | 'correction' | 'high_vol' | 'neutral' | ...)."
    ),
    "input_schema": {"type": "object", "properties": {}},
}

QUERY_JOURNAL_SPEC: Dict[str, Any] = {
    "name": "query_journal",
    "description": (
        "Read recent journal entries. The agent can read its own journal or "
        "the paired-research journal (but NOT unrelated agents). `kind` "
        "optionally filters by entry kind (brief/memo/rationale/...)."
    ),
    "input_schema": {
        "type": "object",
        "required": ["agent_id"],
        "properties": {
            "agent_id": {"type": "string"},
            "kind": {"type": "string"},
            "symbol": {"type": "string"},
            "since_hours": {"type": "integer", "default": 24, "maximum": 720},
            "limit": {"type": "integer", "default": 20, "maximum": 100},
        },
    },
}

RUN_BACKTEST_SPEC: Dict[str, Any] = {
    "name": "run_backtest",
    "description": (
        "Run a walk-forward backtest on a hypothesis: a filter rule + a "
        "signal score recipe on the project's existing features. Returns "
        "{sharpe, hit_rate, n_trades}. Heavy — budget it sparingly."
    ),
    "input_schema": {
        "type": "object",
        "required": ["hypothesis"],
        "properties": {
            "hypothesis": {"type": "string"},
            "lookback_days": {"type": "integer", "default": 180},
        },
    },
}

PROPOSE_TRADE_SPEC: Dict[str, Any] = {
    "name": "propose_trade",
    "description": (
        "Emit a structured trade proposal. Does NOT place the order — the "
        "AgentLoop collects all proposals and hands them to StrategyManager "
        "for risk-checked execution. Only callable by trader agents.\n\n"
        "For action='enter' you MUST include BOTH `notional_usd` (USD to "
        "allocate) AND `target_price` (your entry reference price, which "
        "must be within ±30% of the symbol's most recent close — proposals "
        "outside that band are dropped). Proposals with a null / missing "
        "notional_usd are dropped before they reach the risk engine."
    ),
    "input_schema": {
        "type": "object",
        "required": ["action", "symbol", "rationale"],
        "properties": {
            "action": {"type": "string", "enum": ["enter", "hold", "exit"]},
            "symbol": {"type": "string"},
            "notional_usd": {
                "type": "number",
                "description": (
                    "USD to allocate. REQUIRED for action='enter'. Sizing "
                    "is derived as floor(notional_usd / target_price)."
                ),
            },
            "shares": {"type": "integer"},
            "target_price": {
                "type": "number",
                "description": (
                    "Reference entry price. REQUIRED for action='enter'. "
                    "Must be within ±30% of the symbol's most recent "
                    "close; otherwise the proposal is rejected."
                ),
            },
            "stop_price": {"type": "number"},
            "take_profit_price": {"type": "number"},
            "rationale": {"type": "string"},
            "brief_id": {"type": "integer"},
            "memo_id": {"type": "integer"},
        },
    },
}

# ─── Tool bundles per agent role ──────────────────────────────────────────────

RESEARCH_TOOLS = (
    GET_OHLCV_SPEC,
    GET_NEWS_SPEC,
    GET_FUNDAMENTALS_SPEC,
    GET_REGIME_SPEC,
    QUERY_JOURNAL_SPEC,
    RUN_BACKTEST_SPEC,
)

TRADER_TOOLS = (
    GET_OHLCV_SPEC,
    GET_NEWS_SPEC,
    GET_REGIME_SPEC,
    QUERY_JOURNAL_SPEC,
    PROPOSE_TRADE_SPEC,
)

OVERSEER_TOOLS = (
    GET_REGIME_SPEC,
    QUERY_JOURNAL_SPEC,
)


# ─── Bundle that AgentLoop receives ───────────────────────────────────────────


@dataclass
class ToolBundle:
    """Specs + handlers for one agent's run. Build with `build_tool_bundle`."""
    specs: List[Dict[str, Any]]
    handlers: Dict[str, Callable[..., Any]]
    proposals: List[Dict[str, Any]]   # populated by `propose_trade` handler


def build_tool_bundle(
    *,
    agent_id: str,
    role: str,                     # 'research' | 'trader' | 'overseer'
    allowed_journal_agents: Optional[List[str]] = None,
    broker: Optional[Any] = None,   # duck-typed: get_news, get_latest_quote
    feature_pipeline: Optional[Any] = None,
    regime_fn: Optional[Callable[[], str]] = None,
) -> ToolBundle:
    """Build a role-appropriate `ToolBundle`.

    The LLM sees the spec list; the handlers dict maps spec names to their
    Python implementations. `allowed_journal_agents` bounds what
    `query_journal` can read (trader sees its own + paired research;
    overseer sees everyone).
    """
    allowed_journal_agents = list(allowed_journal_agents or [agent_id])
    proposals: List[Dict[str, Any]] = []

    specs: List[Dict[str, Any]]
    if role == "research":
        specs = list(RESEARCH_TOOLS)
    elif role == "trader":
        specs = list(TRADER_TOOLS)
    elif role == "overseer":
        specs = list(OVERSEER_TOOLS)
    else:
        raise ValueError(f"unknown role {role!r}")

    handlers: Dict[str, Callable[..., Any]] = {}

    # ─── get_ohlcv ──────────────────────────────────────────────────────────
    def _get_ohlcv(symbol: str, timeframe: str = "1D", lookback_bars: int = 60):
        from zeus.data.storage.database import OHLCVDaily, OHLCVIntraday, get_session_factory
        from sqlalchemy import select
        lookback_bars = min(int(lookback_bars), 500)
        with get_session_factory()() as s:
            if timeframe == "1D":
                rows = s.execute(
                    select(OHLCVDaily.ts, OHLCVDaily.open, OHLCVDaily.high,
                           OHLCVDaily.low, OHLCVDaily.close, OHLCVDaily.volume)
                    .where(OHLCVDaily.symbol == symbol)
                    .order_by(OHLCVDaily.ts.desc())
                    .limit(lookback_bars)
                ).all()
            else:
                rows = s.execute(
                    select(OHLCVIntraday.ts, OHLCVIntraday.open, OHLCVIntraday.high,
                           OHLCVIntraday.low, OHLCVIntraday.close, OHLCVIntraday.volume)
                    .where(OHLCVIntraday.symbol == symbol)
                    .where(OHLCVIntraday.timeframe == timeframe.lower().replace("min", "m"))
                    .order_by(OHLCVIntraday.ts.desc())
                    .limit(lookback_bars)
                ).all()
        rows = list(reversed(rows))
        return {
            "symbol": symbol, "timeframe": timeframe, "n_bars": len(rows),
            "bars": [
                {"ts": r[0].isoformat() if r[0] else None,
                 "open": r[1], "high": r[2], "low": r[3],
                 "close": r[4], "volume": r[5]}
                for r in rows
            ],
        }
    handlers["get_ohlcv"] = _get_ohlcv

    # ─── get_news ───────────────────────────────────────────────────────────
    def _get_news(symbol: str, hours: int = 24):
        if broker is None:
            return {"symbol": symbol, "items": [], "note": "no broker bound"}
        try:
            items = broker.get_news(symbols=[symbol], limit=30)
        except Exception as e:
            return {"symbol": symbol, "items": [], "error": str(e)}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out: List[Dict[str, Any]] = []
        for n in items:
            ts = getattr(n, "created_at", None) or getattr(n, "updated_at", None)
            if ts is not None and ts < cutoff:
                continue
            out.append({
                "ts": ts.isoformat() if ts else None,
                "headline": getattr(n, "headline", ""),
                "source": getattr(n, "source", ""),
                "url": getattr(n, "url", ""),
            })
        return {"symbol": symbol, "hours": hours, "items": out}
    handlers["get_news"] = _get_news

    # ─── get_fundamentals ───────────────────────────────────────────────────
    def _get_fundamentals(symbol: str):
        from zeus.data.storage.database import FundamentalsCache, get_session_factory
        from sqlalchemy import select
        with get_session_factory()() as s:
            row = s.execute(
                select(FundamentalsCache).where(FundamentalsCache.symbol == symbol)
            ).scalar_one_or_none()
        if row is None:
            return {"symbol": symbol, "found": False}
        return {
            "symbol": symbol, "found": True,
            "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None,
            "pe": row.pe_ratio, "pb": row.pb_ratio, "ps": row.ps_ratio,
            "ev_ebitda": row.ev_ebitda, "earnings_yield": row.earnings_yield,
            "revenue_growth_yoy": row.revenue_growth_yoy,
            "earnings_growth_yoy": row.earnings_growth_yoy,
            "gross_margin": row.gross_margin, "operating_margin": row.operating_margin,
            "debt_to_equity": row.debt_to_equity, "roe": row.roe, "roa": row.roa,
            "sector": row.sector, "industry": row.industry,
            "market_cap": row.market_cap,
        }
    handlers["get_fundamentals"] = _get_fundamentals

    # ─── get_regime ─────────────────────────────────────────────────────────
    def _get_regime():
        if regime_fn is not None:
            return {"regime": regime_fn()}
        from zeus.data.storage.database import SystemMetrics, get_session_factory
        from sqlalchemy import select
        with get_session_factory()() as s:
            row = s.execute(
                select(SystemMetrics.regime)
                .order_by(SystemMetrics.ts.desc())
                .limit(1)
            ).scalar_one_or_none()
        return {"regime": row or "unknown"}
    handlers["get_regime"] = _get_regime

    # ─── query_journal ──────────────────────────────────────────────────────
    def _query_journal(
        agent_id: str,
        kind: Optional[str] = None,
        symbol: Optional[str] = None,
        since_hours: int = 24,
        limit: int = 20,
    ):
        if agent_id not in allowed_journal_agents:
            return {
                "error": (
                    f"agent {agent_id!r} is outside the read allowlist "
                    f"{allowed_journal_agents!r}"
                )
            }
        since = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))
        entries = query_journal(
            agent_id=agent_id, kind=kind, symbol=symbol,
            since=since, limit=int(limit),
        )
        return {
            "agent_id": agent_id,
            "n": len(entries),
            "entries": [
                {
                    "id": e.id, "ts": e.ts.isoformat(),
                    "kind": e.kind, "symbol": e.symbol,
                    "title": e.title, "body": e.body,
                    "tags": e.tags, "confidence": e.confidence,
                } for e in entries
            ],
        }
    handlers["query_journal"] = _query_journal

    # ─── run_backtest (stub) ───────────────────────────────────────────────
    # A full walk-forward run is expensive; keep the tool as a stub that
    # returns a "not-yet-wired" payload until Phase 4 research agents
    # actually need it. The spec is already exposed so the LLM knows it exists.
    def _run_backtest(hypothesis: str, lookback_days: int = 180):
        return {
            "status": "deferred",
            "hypothesis": hypothesis,
            "note": (
                "run_backtest is registered but returns deferred — wire in "
                "zeus.backtesting.walk_forward for Phase 4 hypothesis testing."
            ),
        }
    handlers["run_backtest"] = _run_backtest

    # ─── propose_trade (trader-only) ───────────────────────────────────────
    def _propose_trade(**payload):
        proposals.append({"ts": datetime.now(timezone.utc).isoformat(), **payload})
        return {"accepted": True, "proposals_so_far": len(proposals)}
    if role == "trader":
        handlers["propose_trade"] = _propose_trade

    return ToolBundle(specs=specs, handlers=handlers, proposals=proposals)
