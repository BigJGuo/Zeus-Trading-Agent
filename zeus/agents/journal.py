"""AgentJournal — typed CRUD for the `agent_journal` table.

Every agent instantiates one `AgentJournal(agent_id=...)` and uses it for:
  - Writing its reasoning/briefs/memos/rationales (record_* methods)
  - Reading its own recent context when the LLM assembles its prompt
    (recent, by_symbol, search)

The journal is the single source of truth for "why did the agent do that?"
The explainability invariant (every fill has a linked trade_rationale row)
is enforced at the reconciler / overseer layer; this module just provides
the write/read primitives.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, cast

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from zeus.data.storage.database import AgentJournal as JournalRow
from zeus.data.storage.database import get_session_factory

log = structlog.get_logger(__name__)


# Canonical agent_id set — enforced at write time so overseer audits can
# rely on a closed vocabulary.
TRADER_AGENTS = ("day", "swing", "long_term")
RESEARCH_AGENTS = ("day_research", "swing_research", "long_term_research")
OVERSEER_AGENT = "overseer"
# Pseudo-agent used to publish curated research-library content (PDF
# excerpts from the academic finance papers in `Trading Strategy
# Research/`). Read-only from the LLMs' perspective; populated by a
# one-shot ingestion script. Every live agent is allowed to query it.
LIBRARY_AGENT = "research_library"
KNOWN_AGENTS = frozenset(
    (*TRADER_AGENTS, *RESEARCH_AGENTS, OVERSEER_AGENT, LIBRARY_AGENT)
)

# Canonical kinds. 'review' used by long-term research for monthly holding re-examination.
# 'paper' is reserved for the research_library ingestion pipeline.
KNOWN_KINDS = frozenset((
    "trade_rationale",
    "hypothesis",
    "postmortem",
    "brief",
    "memo",
    "lesson",
    "decision",
    "alert",
    "review",
    "paper",
))

# Pairing map used by overseer audits: for every trader fill, verify a brief/memo
# from the paired research agent exists within the cadence window.
PAIRED_RESEARCH = {
    "day": "day_research",
    "swing": "swing_research",
    "long_term": "long_term_research",
}


@dataclass(frozen=True)
class JournalEntry:
    """Read-side snapshot of a journal row."""
    id: int
    ts: datetime
    agent_id: str
    kind: str
    title: str
    body: str
    symbol: Optional[str] = None
    related_trade_id: Optional[uuid.UUID] = None
    related_position_id: Optional[int] = None
    structured: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    confidence: Optional[float] = None

    @classmethod
    def from_row(cls, row: JournalRow) -> "JournalEntry":
        # SQLAlchemy legacy `Column(...)` columns are typed as `Column[X]` at
        # the class level. At runtime an ORM instance's attributes are the
        # underlying values, so cast to Any to dodge the descriptor types.
        r = cast(Any, row)
        return cls(
            id=r.id,
            ts=r.ts,
            agent_id=r.agent_id,
            kind=r.kind,
            symbol=r.symbol,
            related_trade_id=r.related_trade_id,
            related_position_id=r.related_position_id,
            title=r.title,
            body=r.body,
            structured=r.structured,
            tags=list(r.tags) if r.tags else None,
            confidence=r.confidence,
        )


class AgentJournal:
    """Per-agent wrapper: `AgentJournal(agent_id='day')` then `journal.record_*`.

    Not a repository pattern — intentionally thin so agents can also do
    ad-hoc SQL via the shared session_factory if they need to. Writes open a
    short-lived session each call; reads can optionally reuse a session the
    caller already holds.
    """

    def __init__(self, agent_id: str):
        if agent_id not in KNOWN_AGENTS:
            raise ValueError(
                f"unknown agent_id {agent_id!r}; expected one of {sorted(KNOWN_AGENTS)}"
            )
        self._agent_id = agent_id
        self._session_factory = get_session_factory()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # ─── Writes ───────────────────────────────────────────────────────────────
    def _write(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        symbol: Optional[str] = None,
        related_trade_id: Optional[uuid.UUID] = None,
        related_position_id: Optional[int] = None,
        structured: Optional[Dict[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
    ) -> int:
        if kind not in KNOWN_KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(KNOWN_KINDS)}")
        if not title or not body:
            raise ValueError("journal entries require non-empty title AND body")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {confidence}")

        row = JournalRow(
            ts=datetime.now(timezone.utc),
            agent_id=self._agent_id,
            kind=kind,
            symbol=symbol,
            related_trade_id=related_trade_id,
            related_position_id=related_position_id,
            title=title,
            body=body,
            structured=structured,
            tags=list(tags) if tags is not None else None,
            confidence=confidence,
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            new_id = int(cast(int, row.id))
        log.info(
            "journal_write", agent_id=self._agent_id, kind=kind, id=new_id,
            symbol=symbol, title=title[:80],
        )
        return new_id

    def record_rationale(
        self, *, symbol: str, title: str, body: str,
        related_trade_id: Optional[uuid.UUID] = None,
        related_position_id: Optional[int] = None,
        structured: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Trade rationale — linked 1:1 to a placed order. Required by the
        explainability invariant."""
        return self._write(
            kind="trade_rationale", symbol=symbol, title=title, body=body,
            related_trade_id=related_trade_id, related_position_id=related_position_id,
            structured=structured, confidence=confidence,
        )

    def record_hypothesis(
        self, *, title: str, body: str, symbol: Optional[str] = None,
        structured: Optional[Dict[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
    ) -> int:
        return self._write(
            kind="hypothesis", title=title, body=body, symbol=symbol,
            structured=structured, tags=tags, confidence=confidence,
        )

    def record_postmortem(
        self, *, title: str, body: str, symbol: Optional[str] = None,
        related_trade_id: Optional[uuid.UUID] = None,
        structured: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self._write(
            kind="postmortem", title=title, body=body, symbol=symbol,
            related_trade_id=related_trade_id, structured=structured,
        )

    def record_brief(
        self, *, symbol: str, title: str, body: str,
        structured: Optional[Dict[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Intraday/swing research brief. Read by the paired trader."""
        return self._write(
            kind="brief", symbol=symbol, title=title, body=body,
            structured=structured, tags=tags, confidence=confidence,
        )

    def record_memo(
        self, *, title: str, body: str, symbol: Optional[str] = None,
        structured: Optional[Dict[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Deep-dive memo (long-term research) or weekly overseer review."""
        return self._write(
            kind="memo", title=title, body=body, symbol=symbol,
            structured=structured, tags=tags, confidence=confidence,
        )

    def record_review(
        self, *, symbol: str, title: str, body: str,
        structured: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self._write(
            kind="review", symbol=symbol, title=title, body=body, structured=structured,
        )

    def record_lesson(
        self, *, title: str, body: str,
        tags: Optional[Sequence[str]] = None,
    ) -> int:
        return self._write(kind="lesson", title=title, body=body, tags=tags)

    def record_decision(
        self, *, title: str, body: str,
        structured: Optional[Dict[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> int:
        return self._write(
            kind="decision", title=title, body=body,
            structured=structured, tags=tags,
        )

    def record_alert(
        self, *, title: str, body: str, symbol: Optional[str] = None,
        structured: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self._write(
            kind="alert", title=title, body=body, symbol=symbol, structured=structured,
        )

    # ─── Reads ────────────────────────────────────────────────────────────────
    def recent(
        self,
        kind: Optional[str] = None,
        limit: int = 20,
        session: Optional[Session] = None,
    ) -> List[JournalEntry]:
        """Most recent N entries for this agent, optionally filtered by kind."""
        stmt = (
            select(JournalRow)
            .where(JournalRow.agent_id == self._agent_id)
            .order_by(JournalRow.ts.desc())
            .limit(limit)
        )
        if kind is not None:
            stmt = (
                select(JournalRow)
                .where(JournalRow.agent_id == self._agent_id)
                .where(JournalRow.kind == kind)
                .order_by(JournalRow.ts.desc())
                .limit(limit)
            )
        return [JournalEntry.from_row(r) for r in self._exec(stmt, session)]

    def by_symbol(
        self,
        symbol: str,
        window_days: int = 7,
        session: Optional[Session] = None,
    ) -> List[JournalEntry]:
        """All entries from this agent for `symbol` in the last `window_days`."""
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        stmt = (
            select(JournalRow)
            .where(JournalRow.agent_id == self._agent_id)
            .where(JournalRow.symbol == symbol)
            .where(JournalRow.ts >= since)
            .order_by(JournalRow.ts.desc())
        )
        return [JournalEntry.from_row(r) for r in self._exec(stmt, session)]

    def search(
        self,
        query: str,
        limit: int = 20,
        session: Optional[Session] = None,
    ) -> List[JournalEntry]:
        """Cheap substring search on title+body. For Phase 2 this is an
        `ILIKE '%q%'`; if `agent_journal` grows past ~100k rows we swap for
        full-text search (tsvector) or a vector index."""
        q = f"%{query}%"
        stmt = (
            select(JournalRow)
            .where(JournalRow.agent_id == self._agent_id)
            .where(or_(JournalRow.title.ilike(q), JournalRow.body.ilike(q)))
            .order_by(JournalRow.ts.desc())
            .limit(limit)
        )
        return [JournalEntry.from_row(r) for r in self._exec(stmt, session)]

    def _exec(self, stmt, session: Optional[Session]):
        if session is not None:
            return session.execute(stmt).scalars().all()
        with self._session_factory() as s:
            return s.execute(stmt).scalars().all()


# ─── Cross-agent reads (overseer audits) ──────────────────────────────────────


def query_journal(
    agent_id: str,
    *,
    kind: Optional[str] = None,
    symbol: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 50,
    session: Optional[Session] = None,
) -> List[JournalEntry]:
    """Cross-agent read used by the overseer + the LLM `query_journal` tool.
    Unlike `AgentJournal.recent`, this is agent-agnostic and keyed by argument."""
    stmt = select(JournalRow).where(JournalRow.agent_id == agent_id)
    if kind is not None:
        stmt = stmt.where(JournalRow.kind == kind)
    if symbol is not None:
        stmt = stmt.where(JournalRow.symbol == symbol)
    if since is not None:
        stmt = stmt.where(JournalRow.ts >= since)
    stmt = stmt.order_by(JournalRow.ts.desc()).limit(limit)

    if session is not None:
        rows = session.execute(stmt).scalars().all()
    else:
        with get_session_factory()() as s:
            rows = s.execute(stmt).scalars().all()
    return [JournalEntry.from_row(r) for r in rows]


def audit_research_coupling(
    trader_agent_id: str,
    symbol: str,
    trade_ts: datetime,
    *,
    window_hours: int = 24,
    session: Optional[Session] = None,
) -> Optional[JournalEntry]:
    """For an overseer audit: did the paired research agent emit a brief/memo
    for this symbol in the `window_hours` preceding the trade?
    Returns the most recent matching entry, or None if the coupling was broken.
    """
    paired = PAIRED_RESEARCH.get(trader_agent_id)
    if paired is None:
        raise ValueError(f"trader {trader_agent_id!r} has no paired research agent")
    since = trade_ts - timedelta(hours=window_hours)
    stmt = (
        select(JournalRow)
        .where(JournalRow.agent_id == paired)
        .where(JournalRow.symbol == symbol)
        .where(JournalRow.kind.in_(("brief", "memo", "review", "alert")))
        .where(JournalRow.ts >= since)
        .where(JournalRow.ts <= trade_ts)
        .order_by(JournalRow.ts.desc())
        .limit(1)
    )
    if session is not None:
        row = session.execute(stmt).scalar_one_or_none()
    else:
        with get_session_factory()() as s:
            row = s.execute(stmt).scalar_one_or_none()
    return JournalEntry.from_row(row) if row else None
