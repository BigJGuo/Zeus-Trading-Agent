"""Append-only CSV trading log.

One row per fill event (partial or full). Columns:
  timestamp_utc, date_ct, time_ct, symbol, side, shares, fill_price,
  order_type, order_id, environment
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

_CT = ZoneInfo("America/Chicago")
_HEADER = (
    "timestamp_utc,date_ct,time_ct,symbol,side,shares,fill_price,"
    "order_type,order_id,environment\n"
)
_LOCK = Lock()


def _resolve_path() -> Path:
    return Path("logs") / "trading_log.csv"


def append_fill(
    symbol: str,
    side: str,
    shares: int,
    price: float | None,
    order_id: str,
    order_type: str,
    environment: str,
) -> None:
    if shares <= 0:
        return
    ts_utc = datetime.now(timezone.utc)
    ts_ct = ts_utc.astimezone(_CT)
    price_str = f"{price:.4f}" if price is not None else ""
    line = (
        f"{ts_utc.isoformat()},{ts_ct.strftime('%Y-%m-%d')},"
        f"{ts_ct.strftime('%H:%M:%S')},{symbol},{side},{shares},"
        f"{price_str},{order_type},{order_id},{environment}\n"
    )
    path = _resolve_path()
    with _LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            with path.open("a") as f:
                if new_file:
                    f.write(_HEADER)
                f.write(line)
        except Exception as exc:
            log.error("trading_log_write_failed", error=str(exc), symbol=symbol)
