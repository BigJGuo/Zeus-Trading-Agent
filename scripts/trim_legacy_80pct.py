"""One-off: queue market-on-open sells for 80% of each legacy position.

Submits 4 MOO (TimeInForce.OPG) sell orders to Alpaca. These sit until the
next regular-session opening auction (2026-04-24 09:30 ET / 08:30 CT) and
execute there. No scheduler restart required; orders live in Alpaca's queue,
not in our APScheduler.

Intent: free up portfolio exposure so the swing strategy's entries stop
getting rejected at `total_exposure_rejected`. 4 legacy positions from
2026-04-20 currently hold ~$73k and block every new swing entry.

After fills, the reconciler (which runs every ~5 min) will sync the new
share counts into `positions` automatically — no manual DB edits.

Run:
    docker exec zeus-scheduler python -m scripts.trim_legacy_80pct
"""
from __future__ import annotations

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from zeus.config.settings import get_settings

log = structlog.get_logger(__name__)

# (symbol, shares_to_sell) — 80% of each current legacy holding, rounded down
TRIMS: list[tuple[str, int]] = [
    ("ABT", 164),   # 206 * 0.80
    ("CVX", 64),    # 81  * 0.80
    ("NFLX", 164),  # 205 * 0.80
    ("WFC", 196),   # 245 * 0.80
]


def main() -> None:
    s = get_settings()
    client = TradingClient(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_secret_key,
        paper=True,
    )
    print(f"Submitting {len(TRIMS)} market-on-open sells for next session:")
    for symbol, qty in TRIMS:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.OPG,
        )
        try:
            order = client.submit_order(req)
            print(f"  {symbol:6s} sell {qty:4d}  MOO  id={order.id}  status={order.status}")
            log.info(
                "legacy_trim_moo_submitted",
                symbol=symbol, qty=qty, order_id=str(order.id),
                status=str(order.status),
            )
        except Exception as e:
            print(f"  {symbol:6s} FAILED: {e}")
            log.error("legacy_trim_moo_failed", symbol=symbol, qty=qty, error=str(e))


if __name__ == "__main__":
    main()
