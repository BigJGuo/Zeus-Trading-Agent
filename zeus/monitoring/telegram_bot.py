"""
Telegram notification system for ZEUS.
Sends all material alerts: trades, risk, EOD reports, heartbeats, emergencies.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from telegram import Bot
from telegram.error import TelegramError

log = structlog.get_logger(__name__)

# Rate limiting: max 30 messages/hour
_message_count = 0
_rate_reset_ts = datetime.utcnow()
MAX_MESSAGES_PER_HOUR = 30


class TelegramNotifier:
    """
    Async Telegram notification client.
    All public send_* methods are safe to call from sync code via send_sync().
    """

    def __init__(self, bot_token: str | None, chat_id: str | None):
        self._chat_id = chat_id
        # No-op mode if either credential is missing or a placeholder
        if not bot_token or not chat_id or "PLACEHOLDER" in str(bot_token).upper():
            self._bot = None
            self._enabled = False
            log.warning("Telegram disabled — no credentials provided; running in no-op mode")
        else:
            self._bot = Bot(token=bot_token)
            self._enabled = True

    # ─── Core send ────────────────────────────────────────────────────────────

    async def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a raw message. Returns True on success."""
        if not self._enabled or self._bot is None or self._chat_id is None:
            return False
        if not self._check_rate_limit():
            log.warning("Telegram rate limit hit, message dropped")
            return False
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            return True
        except TelegramError as e:
            log.error("Telegram send failed", error=str(e))
            return False

    def send_sync(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Synchronous wrapper for send()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as a coroutine in the running loop
                asyncio.ensure_future(self.send(message, parse_mode))
                return True
            else:
                return loop.run_until_complete(self.send(message, parse_mode))
        except Exception as e:
            log.error("Telegram send_sync failed", error=str(e))
            return False

    # ─── Typed messages ───────────────────────────────────────────────────────

    def send_heartbeat(self, portfolio_value: float, drawdown_usd: float,
                       position_count: int, regime: str) -> None:
        dd_pct = drawdown_usd / 100_000 * 100 if drawdown_usd else 0
        msg = (
            f"💚 *ZEUS Heartbeat*\n"
            f"Portfolio: `${portfolio_value:,.0f}`\n"
            f"Drawdown: `${drawdown_usd:,.0f}` ({dd_pct:.1f}%)\n"
            f"Positions: `{position_count}`\n"
            f"Regime: `{regime}`\n"
            f"Time: `{datetime.utcnow().strftime('%H:%M UTC')}`"
        )
        self.send_sync(msg)

    def send_trade_open(self, symbol: str, shares: int, price: float,
                        signal_score: float, confidence: float,
                        stop: float, regime: str) -> None:
        notional = shares * price
        stop_pct = (stop - price) / price * 100
        msg = (
            f"📈 *TRADE OPEN*\n"
            f"Symbol: `{symbol}`\n"
            f"Shares: `{shares}` @ `${price:.2f}`\n"
            f"Notional: `${notional:,.0f}`\n"
            f"Signal: `{signal_score:.2f}` | Confidence: `{confidence*100:.0f}%`\n"
            f"Stop: `${stop:.2f}` (`{stop_pct:.1f}%`)\n"
            f"Regime: `{regime}`"
        )
        self.send_sync(msg)

    def send_trade_close(self, symbol: str, shares: int, entry_price: float,
                         exit_price: float, net_pnl: float,
                         hold_days: float, exit_reason: str) -> None:
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        emoji = "✅" if net_pnl > 0 else "❌"
        msg = (
            f"{emoji} *TRADE CLOSE*\n"
            f"Symbol: `{symbol}`\n"
            f"Shares: `{shares}` | Entry: `${entry_price:.2f}` → Exit: `${exit_price:.2f}`\n"
            f"P&L: `${net_pnl:+,.2f}` (`{pnl_pct:+.2f}%`)\n"
            f"Hold: `{hold_days:.1f} days`\n"
            f"Reason: `{exit_reason}`"
        )
        self.send_sync(msg)

    def send_daily_pnl(self, date_str: str, daily_pnl: float, portfolio_value: float,
                       drawdown_usd: float, open_positions: int,
                       win_count: int, loss_count: int, regime: str) -> None:
        dd_pct = drawdown_usd / 100_000 * 100
        emoji = "📊✅" if daily_pnl >= 0 else "📊❌"
        msg = (
            f"{emoji} *EOD Report — {date_str}*\n"
            f"Daily P&L: `${daily_pnl:+,.2f}`\n"
            f"Portfolio: `${portfolio_value:,.0f}`\n"
            f"Drawdown: `${drawdown_usd:,.0f}` ({dd_pct:.1f}%)\n"
            f"Open Positions: `{open_positions}`\n"
            f"Trades Today: `{win_count}W / {loss_count}L`\n"
            f"Regime: `{regime}`"
        )
        self.send_sync(msg)

    def send_risk_warning(self, level: int, drawdown_usd: float,
                          action: str) -> None:
        levels = {1: "⚠️ WARNING", 2: "🚨 ALERT", 3: "🔴 CRITICAL", 4: "🆘 EMERGENCY"}
        label = levels.get(level, "⚠️")
        msg = (
            f"{label} *Risk Level {level}*\n"
            f"Drawdown: `${drawdown_usd:,.0f}`\n"
            f"Action: {action}"
        )
        self.send_sync(msg)

    def send_kill_switch(self, reason: str, positions_closed: int) -> None:
        msg = (
            f"🆘 *KILL SWITCH ACTIVATED*\n"
            f"Reason: {reason}\n"
            f"Positions closed: `{positions_closed}`\n"
            f"System halted. Manual review required."
        )
        self.send_sync(msg)

    def send_model_retrain(self, model_name: str, version: str,
                           promoted: bool, metrics: Dict[str, Any]) -> None:
        status = "✅ PROMOTED" if promoted else "❌ NOT PROMOTED"
        ic = metrics.get("information_coefficient", 0)
        sharpe = metrics.get("sharpe_on_signals", 0)
        msg = (
            f"🤖 *Model Retrain — {model_name}*\n"
            f"Version: `{version}`\n"
            f"Status: {status}\n"
            f"IC: `{ic:.4f}` | Sharpe: `{sharpe:.2f}`"
        )
        self.send_sync(msg)

    def send_system_error(self, component: str, error: str,
                          action: str = "Retrying") -> None:
        msg = (
            f"🔧 *System Error*\n"
            f"Component: `{component}`\n"
            f"Error: {error[:200]}\n"
            f"Action: {action}"
        )
        self.send_sync(msg)

    def send_premarket_summary(self, planned_entries: int, planned_exits: int,
                                regime: str, portfolio_value: float,
                                top_signals: str) -> None:
        msg = (
            f"🌅 *Pre-Market Summary*\n"
            f"Portfolio: `${portfolio_value:,.0f}`\n"
            f"Regime: `{regime}`\n"
            f"Planned entries: `{planned_entries}` | Exits: `{planned_exits}`\n"
            f"Top signals:\n{top_signals}"
        )
        self.send_sync(msg)

    def send_startup(self, environment: str, portfolio_value: float) -> None:
        msg = (
            f"🚀 *ZEUS Started*\n"
            f"Environment: `{environment}`\n"
            f"Portfolio: `${portfolio_value:,.0f}`\n"
            f"Time: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`"
        )
        self.send_sync(msg)

    # ─── Rate limiting ────────────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        global _message_count, _rate_reset_ts
        now = datetime.utcnow()
        elapsed = (now - _rate_reset_ts).total_seconds()
        if elapsed >= 3600:
            _message_count = 0
            _rate_reset_ts = now
        if _message_count >= MAX_MESSAGES_PER_HOUR:
            return False
        _message_count += 1
        return True

    def ping(self) -> bool:
        """Check Telegram bot connectivity."""
        if not self._enabled or self._bot is None:
            return False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return True  # assume OK if loop already running
            info = loop.run_until_complete(self._bot.get_me())
            return info is not None
        except Exception:
            return False
