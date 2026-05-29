"""
Alpaca Markets broker client for ZEUS.
Wraps alpaca-py SDK with retry logic, error handling, and paper/live switching.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, cast

import structlog
from alpaca.common.enums import Sort
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

_DAY_UNIT = cast(TimeFrameUnit, TimeFrameUnit.Day)

log = structlog.get_logger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class AlpacaBroker:
    """
    Unified Alpaca broker client.
    Handles authentication, order management, data fetching, and resilience.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)
        self._news = NewsClient(api_key, secret_key)
        log.info("AlpacaBroker initialized", paper=paper)

    # ─── Account ──────────────────────────────────────────────────────────────

    def get_account(self) -> TradeAccount:
        """Return current account state."""
        return cast(TradeAccount, self._retry(self._trading.get_account))

    def get_portfolio_value(self) -> float:
        account = self.get_account()
        return float(account.portfolio_value or 0)

    def get_buying_power(self) -> float:
        account = self.get_account()
        return float(account.buying_power or 0)

    def get_cash(self) -> float:
        account = self.get_account()
        return float(account.cash or 0)

    def is_account_active(self) -> bool:
        try:
            account = self.get_account()
            return account.status == "ACTIVE"
        except Exception:
            return False

    # ─── Positions ────────────────────────────────────────────────────────────

    def get_positions(self) -> List[Position]:
        """Return list of current positions."""
        return cast(List[Position], self._retry(self._trading.get_all_positions))

    def get_position(self, symbol: str):
        try:
            return self._trading.get_open_position(symbol)
        except Exception:
            return None

    # ─── Orders ───────────────────────────────────────────────────────────────

    def submit_market_order(self, symbol: str, qty: int, side: str) -> Order:
        """Submit a market order. side: 'buy' or 'sell'"""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = cast(Order, self._retry(lambda: self._trading.submit_order(req)))
        log.info("Market order submitted", symbol=symbol, qty=qty, side=side, order_id=order.id)
        return order

    def submit_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float
    ) -> Order:
        """Submit a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        )
        order = cast(Order, self._retry(lambda: self._trading.submit_order(req)))
        log.info("Limit order submitted", symbol=symbol, qty=qty, side=side,
                 limit_price=limit_price, order_id=order.id)
        return order

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        try:
            self._trading.cancel_order_by_id(order_id)
            log.info("Order cancelled", order_id=order_id)
        except Exception as e:
            log.warning("Cancel order failed", order_id=order_id, error=str(e))

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self._trading.cancel_orders()
        log.info("All orders cancelled")

    def get_order(self, order_id: str):
        return self._trading.get_order_by_id(order_id)

    def get_open_orders(self) -> List[Order]:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return cast(List[Order], self._retry(lambda: self._trading.get_orders(req)))

    def get_filled_orders_since(
        self,
        since: datetime,
        symbols: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List:
        """Closed orders filled at or after `since`, optionally filtered by symbol.

        Used by the reconciler to attribute realized P&L to broker-initiated
        closes (liquidations, manual closes, corporate actions) and by the
        backfill script to reconstruct Trade rows from historical fills.
        """
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=since,
                limit=limit,
                symbols=symbols,
                nested=False,
                direction=Sort.DESC,
            )
            orders = self._retry(lambda: self._trading.get_orders(req))
        except Exception as e:
            log.warning("get_filled_orders_since_failed", since=since.isoformat(), error=str(e))
            return []
        out = []
        for o in orders or []:
            filled_qty = int(float(getattr(o, "filled_qty", 0) or 0))
            if filled_qty <= 0:
                continue
            if getattr(o, "filled_avg_price", None) is None:
                continue
            out.append(o)
        return out

    def close_all_positions(self) -> None:
        """Market-close all positions immediately (kill switch)."""
        log.warning("Closing all positions")
        self._trading.close_all_positions(cancel_orders=True)

    def close_position(self, symbol: str) -> None:
        """Market-close a single position."""
        self._trading.close_position(symbol)
        log.info("Position closed", symbol=symbol)

    # ─── Market Data ──────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbols: List[str],
        start: date,
        end: Optional[date] = None,
    ):
        """Fetch daily OHLCV bars for a list of symbols.

        Uses the IEX feed because the paper-account subscription does not
        permit recent SIP queries (the SIP feed is the historical default
        but errors with `subscription does not permit querying recent SIP
        data` for any bar within the embargo window). IEX is consolidated
        from one exchange and slightly thinner, but it has no embargo and
        is sufficient for daily OHLCV.
        """
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, _DAY_UNIT),
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end or date.today(), datetime.min.time()) if end else None,
            adjustment=Adjustment.ALL,  # split + dividend adjusted
            feed=DataFeed.IEX,
        )
        return self._retry(lambda: self._data.get_stock_bars(req))

    def get_latest_quote(self, symbol: str):
        """Get latest bid/ask quote."""
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = self._retry(lambda: self._data.get_stock_latest_quote(req))
        return quotes.get(symbol)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest trade price for a symbol."""
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = self._data.get_stock_latest_trade(req)
            trade = trades.get(symbol)
            return float(trade.price) if trade else None
        except Exception as e:
            log.warning("get_latest_price failed", symbol=symbol, error=str(e))
            return None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

    # ─── News ────────────────────────────────────────────────────────────────

    def get_news(self, symbols: Optional[List[str]] = None, limit: int = 50) -> List:
        """Get recent news headlines via Alpaca's dedicated news endpoint."""
        try:
            from alpaca.data.requests import NewsRequest
            # alpaca-py's NewsRequest.symbols is a comma-joined string, not a list.
            symbols_arg: Optional[str]
            if symbols is None:
                symbols_arg = None
            elif isinstance(symbols, str):
                symbols_arg = symbols
            else:
                symbols_arg = ",".join(symbols)
            req = NewsRequest(
                symbols=symbols_arg,
                limit=limit,
                sort="desc",
            )
            resp = self._news.get_news(req)
            # NewsClient.get_news returns a NewsSet with a `.news` attribute, a
            # raw list, or occasionally a dict — normalize to a list of items.
            news_attr = getattr(resp, "news", None)
            if news_attr is not None:
                return list(news_attr)
            if isinstance(resp, dict):
                return list(resp.get("news") or [])
            return list(resp or [])
        except Exception as e:
            log.warning("get_news failed", error=str(e))
            return []

    # ─── Resilience ───────────────────────────────────────────────────────────

    def _retry(self, fn, *args, **kwargs):
        """Retry with exponential backoff on transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                # Check if retryable
                status = getattr(e, "status_code", None)
                if status and status not in RETRYABLE_STATUS_CODES:
                    raise
                wait = BACKOFF_BASE ** attempt
                log.warning("Alpaca API retry", attempt=attempt + 1, wait=wait, error=str(e))
                time.sleep(wait)
        if last_exc is None:
            raise RuntimeError("_retry exhausted with no exception captured")
        raise last_exc

    def ping(self) -> bool:
        """Check broker connectivity."""
        try:
            self._trading.get_account()
            return True
        except Exception:
            return False
