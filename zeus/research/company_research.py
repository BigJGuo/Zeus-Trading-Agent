"""Overnight company research: enrich tomorrow's plan with news + fundamentals.

Runs AFTER `next_session_planning_job` (21:00 ET) produces the raw entries, and
BEFORE the next market open. For each planned entry:
  - Fetch recent Alpaca news headlines (last 72h)
  - Score keyword sentiment (simple lexicon)
  - Pull yfinance fundamentals snapshot into fundamentals_cache
  - Detect upcoming earnings within the next 5 trading days
  - Flag risk conditions (earnings imminent, extreme P/E, heavy short interest,
    negative-dominant news)
The enriched plan is written back to the same session_plans JSON so that
market_open_job reads research-aware entries. Nothing is removed automatically;
risk_flags is informational — the execution layer can choose to skip flagged
names via a separate policy change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from zeus.config.settings import get_settings
from zeus.data.storage.database import FundamentalsCache, get_session_factory

if TYPE_CHECKING:
    from zeus.research.after_hours import AfterHoursContext

log = structlog.get_logger(__name__)

_NEWS_LOOKBACK_HOURS = 72
_NEWS_PER_SYMBOL = 25
_EARNINGS_WINDOW_DAYS = 5

_POSITIVE_WORDS = {
    "beat", "beats", "exceeds", "upgrade", "upgraded", "buy", "outperform",
    "record", "surge", "surges", "rally", "rallies", "gain", "gains", "grow",
    "growth", "strong", "bullish", "raises", "dividend", "acquire", "acquires",
    "expansion", "profit", "profitable", "breakthrough",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "sell", "underperform",
    "plunge", "plunges", "slump", "slumps", "drop", "drops", "fall", "falls",
    "weak", "bearish", "cuts", "slashes", "lawsuit", "investigation", "probe",
    "fraud", "bankruptcy", "recall", "warning", "concern", "concerns", "decline",
    "loss", "losses", "disappointing", "layoff", "layoffs", "strike",
}


@dataclass
class SymbolResearch:
    symbol: str
    news_count: int
    sentiment: float          # -1.0 .. +1.0
    top_headlines: list[str]
    earnings_date: str | None
    earnings_days_out: int | None
    pe_ratio: float | None
    short_interest_ratio: float | None
    market_cap: float | None
    risk_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "news_count": self.news_count,
            "sentiment": round(self.sentiment, 3),
            "top_headlines": self.top_headlines,
            "earnings_date": self.earnings_date,
            "earnings_days_out": self.earnings_days_out,
            "pe_ratio": self.pe_ratio,
            "short_interest_ratio": self.short_interest_ratio,
            "market_cap": self.market_cap,
            "risk_flags": self.risk_flags,
        }


def _score_sentiment(text: str) -> float:
    if not text:
        return 0.0
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t.strip(".,!?:;\"'()") in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t.strip(".,!?:;\"'()") in _NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _fetch_news(broker, symbol: str, cutoff: datetime) -> tuple[int, float, list[str]]:
    articles = broker.get_news(symbols=[symbol], limit=_NEWS_PER_SYMBOL)
    if not articles:
        return 0, 0.0, []
    recent = []
    for a in articles:
        ts = getattr(a, "created_at", None) or getattr(a, "updated_at", None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            recent.append(a)
    if not recent:
        return 0, 0.0, []
    scores = []
    headlines = []
    for a in recent:
        headline = getattr(a, "headline", "") or ""
        summary = getattr(a, "summary", "") or ""
        scores.append(_score_sentiment(headline + " " + summary))
        if headline:
            headlines.append(headline)
    avg_sentiment = sum(scores) / len(scores) if scores else 0.0
    return len(recent), avg_sentiment, headlines[:5]


def _fetch_fundamentals(yf_client, symbol: str) -> dict[str, Any]:
    try:
        info = yf_client.get_info(symbol) if hasattr(yf_client, "get_info") else None
    except Exception as e:
        log.warning("fundamentals_fetch_failed", symbol=symbol, error=str(e))
        info = None
    if not info:
        import yfinance as yf
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as e:
            log.warning("yfinance_info_failed", symbol=symbol, error=str(e))
            info = {}
    return info or {}


def _earnings_days_out(info: dict[str, Any], as_of: date) -> tuple[str | None, int | None]:
    eps_dates = info.get("earningsDate") or info.get("earnings_dates") or []
    if isinstance(eps_dates, (int, float)):
        return None, None
    if isinstance(eps_dates, str):
        eps_dates = [eps_dates]
    earliest_future = None
    for d in eps_dates:
        try:
            if isinstance(d, (int, float)):
                ed = datetime.fromtimestamp(float(d), tz=timezone.utc).date()
            elif isinstance(d, datetime):
                ed = d.date()
            elif isinstance(d, date):
                ed = d
            else:
                ed = datetime.fromisoformat(str(d)).date()
        except Exception:
            continue
        if ed >= as_of and (earliest_future is None or ed < earliest_future):
            earliest_future = ed
    if earliest_future is None:
        return None, None
    return earliest_future.isoformat(), (earliest_future - as_of).days


def _upsert_fundamentals_cache(session, symbol: str, info: dict[str, Any]) -> None:
    from sqlalchemy import select

    existing = session.execute(
        select(FundamentalsCache).where(FundamentalsCache.symbol == symbol).limit(1)
    ).scalar_one_or_none()

    fields = dict(
        fetched_at=datetime.now(timezone.utc),
        pe_ratio=info.get("trailingPE"),
        pb_ratio=info.get("priceToBook"),
        ps_ratio=info.get("priceToSalesTrailing12Months"),
        ev_ebitda=info.get("enterpriseToEbitda"),
        earnings_yield=info.get("earningsYield") or (1.0 / info["trailingPE"] if info.get("trailingPE") else None),
        revenue_growth_yoy=info.get("revenueGrowth"),
        earnings_growth_yoy=info.get("earningsGrowth"),
        gross_margin=info.get("grossMargins"),
        operating_margin=info.get("operatingMargins"),
        debt_to_equity=info.get("debtToEquity"),
        current_ratio=info.get("currentRatio"),
        roe=info.get("returnOnEquity"),
        roa=info.get("returnOnAssets"),
        short_interest_ratio=info.get("shortPercentOfFloat"),
        market_cap=info.get("marketCap"),
        sector=info.get("sector"),
        industry=info.get("industry"),
    )
    if existing is not None:
        for k, v in fields.items():
            setattr(existing, k, v)
    else:
        session.add(FundamentalsCache(symbol=symbol, **fields))
    session.commit()


def _compute_risk_flags(r: SymbolResearch) -> list[str]:
    flags: list[str] = []
    if r.earnings_days_out is not None and r.earnings_days_out <= 2:
        flags.append("earnings_imminent")
    if r.sentiment <= -0.4 and r.news_count >= 3:
        flags.append("negative_news_cluster")
    if r.short_interest_ratio is not None and r.short_interest_ratio >= 0.20:
        flags.append("high_short_interest")
    if r.pe_ratio is not None and (r.pe_ratio > 80 or r.pe_ratio < 0):
        flags.append("pe_extreme")
    if r.news_count >= 15:
        flags.append("unusual_news_volume")
    return flags


def research_symbol(broker, yf_client, symbol: str, as_of: date) -> SymbolResearch:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_NEWS_LOOKBACK_HOURS)
    news_count, sentiment, headlines = _fetch_news(broker, symbol, cutoff)

    info = _fetch_fundamentals(yf_client, symbol)
    earnings_date, earnings_days = _earnings_days_out(info, as_of)

    try:
        with get_session_factory()() as session:
            _upsert_fundamentals_cache(session, symbol, info)
    except Exception as e:
        log.warning("fundamentals_cache_write_failed", symbol=symbol, error=str(e))

    r = SymbolResearch(
        symbol=symbol,
        news_count=news_count,
        sentiment=sentiment,
        top_headlines=headlines,
        earnings_date=earnings_date,
        earnings_days_out=earnings_days,
        pe_ratio=info.get("trailingPE"),
        short_interest_ratio=info.get("shortPercentOfFloat"),
        market_cap=info.get("marketCap"),
        risk_flags=[],
    )
    r.risk_flags = _compute_risk_flags(r)
    return r


def run_company_research(ctx: "AfterHoursContext", as_of: date | None = None) -> dict[str, Any]:
    """Enrich today's session plan file with per-symbol research + risk flags."""
    as_of = as_of or date.today()
    settings = get_settings()

    from zeus.scheduler.market_schedule import next_trading_day
    plan_for = next_trading_day(as_of)
    plan_path = Path(settings.artifacts_path) / "knowledge" / "session_plans" / f"{plan_for.isoformat()}.json"

    if not plan_path.exists():
        log.warning("research_plan_not_found", path=str(plan_path))
        return {"status": "no_plan", "path": str(plan_path)}

    with plan_path.open() as f:
        plan = json.load(f)

    entries = plan.get("entries", [])
    if not entries:
        log.info("research_no_entries")
        return {"status": "no_entries", "plan_date": plan.get("plan_date")}

    log.info("company_research_start", plan_date=plan.get("plan_date"), n_entries=len(entries))

    results: list[dict[str, Any]] = []
    enriched_entries = []
    for entry in entries:
        symbol = entry["symbol"]
        try:
            research = research_symbol(ctx.broker, ctx.yf_client, symbol, plan_for)
            entry["research"] = research.to_dict()
            results.append({"symbol": symbol, **research.to_dict()})
            log.info("symbol_researched",
                     symbol=symbol, news_count=research.news_count,
                     sentiment=research.sentiment, flags=research.risk_flags)
        except Exception as e:
            log.warning("symbol_research_failed", symbol=symbol, error=str(e))
            entry["research"] = {"error": str(e)}
        enriched_entries.append(entry)

    plan["entries"] = enriched_entries
    plan["research_completed_at"] = datetime.now(timezone.utc).isoformat()

    with plan_path.open("w") as f:
        json.dump(plan, f, indent=2)

    log.info("company_research_complete", n_symbols=len(results), path=str(plan_path))

    flagged = [r["symbol"] for r in results if r.get("risk_flags")]
    return {
        "status": "ok",
        "plan_date": plan.get("plan_date"),
        "n_symbols": len(results),
        "flagged_symbols": flagged,
        "path": str(plan_path),
    }
