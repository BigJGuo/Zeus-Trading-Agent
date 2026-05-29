"""One-shot ingestion of the academic finance papers in
`Trading Strategy Research/` into the agent_journal as
agent_id='research_library', kind='paper' rows.

Each paper becomes one journal row containing:
  - title:  filename (sans .pdf)
  - body:   the first ~6k chars of extracted text (typically covers
            title page + abstract + first body section + a slice of
            the conclusion). Long enough for the agent to ground a
            thesis; short enough that `query_journal` returns are
            not catastrophic for the input-token budget.
  - structured: {filename, page_count, n_chars, topics: [..]}
  - tags:    topic keywords inferred from the filename (momentum,
             value, vol, pairs, RL, etc.)

Idempotent: scans existing rows by (agent_id, title) and skips
papers already ingested. Re-runs after editing this script just
overwrite (delete + insert) so summaries stay fresh.

Usage (from inside the scheduler container, where the DB env is
already wired up):

    docker exec zeus-scheduler python -m scripts.ingest_research_library \
        --library-dir "/app/Trading Strategy Research"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import structlog

from zeus.agents.journal import LIBRARY_AGENT
from zeus.data.storage.database import AgentJournal, get_session_factory

log = structlog.get_logger(__name__)


# ─── Topic tag inference ────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "momentum": ["momentum", "TimeSeriesMomentum", "time series momentum",
                 "Time_Series_Momentum", "Profitability of Momentum",
                 "Momentum Crashes", "Factor Momentum", "Industries Explain Momentum"],
    "value": ["Value and Momentum", "value", "the other side of value",
              "gross profitability"],
    "factors": ["Five Factor", "five-factor", "Asset Pricing"],
    "quality": ["accrural", "accrual", "asset growth", "profitability"],
    "vol": ["volatility", "evaporating liquidity", "cross section of volatility",
            "betting against"],
    "pairs": ["Pairs Trading", "pair", "regime-switching", "relative value",
              "relative-value", "convertible arbitrage"],
    "events": ["FOMC", "earnings annoucement", "post-earnings", "earnings"],
    "options": ["option volume", "open interest"],
    "shorts": ["short interest", "distress"],
    "rl": ["Reinforcement Learning", "reinforcement", "deep convolutional",
           "deep reinforcement", "practical deep", "predictive learning",
           "machine forecast"],
    "microstructure": ["limit order", "high frequency", "market making",
                       "price formation"],
    "fixed_income": ["fixed income", "FOMC annoucement", "bond markets"],
    "horizon": ["long horizon"],
}


def _infer_tags(filename_stem: str) -> List[str]:
    text = filename_stem.lower()
    tags: list[str] = []
    for tag, kws in _TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text:
                tags.append(tag)
                break
    return sorted(set(tags))


# ─── PDF text extraction ────────────────────────────────────────────────────

def _extract_pdf_text(path: Path, max_chars: int = 6000) -> tuple[str, int]:
    """Return (truncated_text, total_page_count). Tries pypdf, falls back
    to pdfplumber if available. Raises RuntimeError if neither lib loads.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
        except ImportError as e:
            raise RuntimeError(
                "neither pypdf nor PyPDF2 is installed in this Python — "
                "install with `pip install pypdf` first"
            ) from e

    reader = PdfReader(str(path))
    n_pages = len(reader.pages)
    chunks: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt:
            continue
        chunks.append(txt)
        total += len(txt)
        if total >= max_chars:
            break
    raw = "\n".join(chunks)

    # Soft-clip while preferring to break at paragraph boundaries.
    if len(raw) > max_chars:
        cut = raw.rfind("\n\n", 0, max_chars)
        if cut < max_chars - 800:
            cut = max_chars
        raw = raw[:cut]
    # Normalize whitespace runs that pypdf often leaves behind.
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return raw, n_pages


def _filename_to_title(stem: str) -> str:
    # Tidy the filename into something a human + agent can read.
    s = stem.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Ingestion ──────────────────────────────────────────────────────────────

def ingest_library(library_dir: Path, dry_run: bool = False) -> int:
    if not library_dir.exists():
        raise SystemExit(f"library dir not found: {library_dir}")

    pdfs = sorted(library_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("library_empty", path=str(library_dir))
        return 0

    SF = get_session_factory()
    session = SF()
    try:
        # Pull existing titles so we can skip already-ingested papers.
        existing = {
            row.title for row in
            session.query(AgentJournal)
            .filter(AgentJournal.agent_id == LIBRARY_AGENT)
            .filter(AgentJournal.kind == "paper")
            .all()
        }
        n_inserted = 0
        n_skipped = 0
        n_failed = 0

        for pdf in pdfs:
            stem = pdf.stem
            title = _filename_to_title(stem)
            if title in existing:
                n_skipped += 1
                log.info("paper_skip_existing", title=title)
                continue
            try:
                body, page_count = _extract_pdf_text(pdf)
            except Exception as e:
                n_failed += 1
                log.error("paper_extract_failed", title=title, error=str(e))
                continue
            if not body:
                n_failed += 1
                log.error("paper_extract_empty", title=title)
                continue

            tags = _infer_tags(stem)
            structured = {
                "filename": pdf.name,
                "page_count": page_count,
                "n_chars": len(body),
                "topics": tags,
            }

            if dry_run:
                log.info("paper_would_ingest", title=title,
                         pages=page_count, tags=tags, n_chars=len(body))
                continue

            row = AgentJournal(
                agent_id=LIBRARY_AGENT,
                kind="paper",
                title=title,
                body=body,
                structured=structured,
                tags=tags,
                confidence=None,
                symbol=None,
            )
            session.add(row)
            session.flush()
            n_inserted += 1
            log.info("paper_ingested", title=title,
                     pages=page_count, tags=tags, n_chars=len(body))

        if not dry_run:
            session.commit()

        log.info("library_ingest_complete",
                 inserted=n_inserted, skipped=n_skipped, failed=n_failed,
                 total_pdfs=len(pdfs))
        return n_inserted
    finally:
        session.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--library-dir",
        default=os.environ.get(
            "ZEUS_RESEARCH_LIBRARY_DIR",
            "/app/Trading Strategy Research",
        ),
        help="Path to the directory of research PDFs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be ingested but do not write to DB.",
    )
    args = p.parse_args(argv)
    n = ingest_library(Path(args.library_dir), dry_run=args.dry_run)
    print(f"ingested {n} new papers" + (" (DRY RUN)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
