"""Telegram inbound command handler.

Polls Telegram for `/status`, `/positions`, `/pause`, `/resume`, `/kill`,
`/killstatus` from a single authorized chat ID. Dispatches to the live
TradingLoop / KillSwitch.

Designed to run alongside the APScheduler in `scheduler/runner.py` — call
`start()` after the loop is built and `stop()` on shutdown.
"""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Optional

import structlog
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

if TYPE_CHECKING:
    from zeus.live.kill_switch import KillSwitch
    from zeus.live.trading_loop import TradingLoop

log = structlog.get_logger(__name__)


def _auth(chat_id: int, allowed_chat_id: str) -> bool:
    return str(chat_id) == str(allowed_chat_id)


class TelegramCommandHandler:
    def __init__(
        self,
        bot_token: str,
        allowed_chat_id: str,
        loop: "TradingLoop",
        kill_switch: "KillSwitch",
    ) -> None:
        self._allowed = allowed_chat_id
        self._loop = loop
        self._kill = kill_switch
        self._bot_token = bot_token
        self._app: Optional[Application] = None
        self._thread: Optional[threading.Thread] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

    def _register_handlers(self) -> None:
        assert self._app is not None, "_register_handlers called before app build"
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("pause", self._cmd_pause))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("kill", self._cmd_kill))
        self._app.add_handler(CommandHandler("killstatus", self._cmd_killstatus))

    # ─── Command implementations ──────────────────────────────────────────────
    async def _cmd_status(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.message is None:
            return
        try:
            account = self._loop._broker.get_account()
            positions = self._loop._broker.get_positions()
            pv = float(account.portfolio_value or 0)
            peak = self._loop._risk._drawdown.peak_value
            dd = peak - pv if peak else 0.0
            msg = (
                f"📊 *Status*\n"
                f"Portfolio: `${pv:,.2f}`\n"
                f"Drawdown: `${dd:,.2f}`\n"
                f"Open positions: `{len(positions)}`\n"
                f"Paused: `{self._loop.is_paused}`\n"
                f"Kill switch: `{self._loop._risk.kill_switch_active}`"
            )
        except Exception as e:
            msg = f"❌ status failed: {e}"
        await update.message.reply_markdown(msg)

    async def _cmd_positions(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.message is None:
            return
        try:
            positions = self._loop._broker.get_positions()
            if not positions:
                await update.message.reply_text("No open positions.")
                return
            lines = ["📈 *Positions*"]
            for p in positions:
                qty = int(getattr(p, "qty", 0))
                upl = float(getattr(p, "unrealized_pl", 0) or 0)
                lines.append(f"• `{p.symbol}` x{qty}  P&L `${upl:+,.2f}`")
            await update.message.reply_markdown("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"❌ positions failed: {e}")

    async def _cmd_pause(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.message is None:
            return
        self._loop.pause()
        await update.message.reply_text("⏸️ Trading paused. Existing positions held.")

    async def _cmd_resume(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.message is None:
            return
        if self._loop._risk.kill_switch_active:
            await update.message.reply_text(
                "❌ Cannot resume: kill switch active. Manual review required."
            )
            return
        self._loop.resume()
        await update.message.reply_text("▶️ Trading resumed.")

    async def _cmd_kill(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.message is None:
            return
        result = self._kill.activate(reason="manual_telegram", session=None)
        await update.message.reply_text(
            f"🆘 Kill switch activated. "
            f"Cancelled {result['cancelled_orders']}, closed {result['closed_positions']}."
        )

    async def _cmd_killstatus(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.message is None:
            return
        from zeus.live.kill_switch import KillSwitch
        from zeus.config.settings import get_settings

        active = KillSwitch.is_tripped(get_settings().artifacts_path)
        await update.message.reply_text(
            f"Kill switch lockfile: {'PRESENT' if active else 'absent'}"
        )

    def _authorized(self, update: Update) -> bool:
        if update.effective_chat is None:
            return False
        if not _auth(update.effective_chat.id, self._allowed):
            log.warning(
                "unauthorized_telegram_command",
                chat_id=update.effective_chat.id,
                command=update.message.text if update.message else None,
            )
            return False
        return True

    # ─── Lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start polling in a background daemon thread. Non-blocking."""
        log.info("telegram_command_handler_starting")
        self._thread = threading.Thread(target=self._run, name="tg-cmd", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._asyncio_loop)
        self._app = Application.builder().token(self._bot_token).build()
        self._register_handlers()
        try:
            self._app.run_polling(close_loop=False, stop_signals=None)
        except Exception as e:
            log.error("telegram_command_handler_crashed", error=str(e))

    def stop(self) -> None:
        if self._app is None or self._asyncio_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._app.stop(), self._asyncio_loop).result(timeout=5)
            asyncio.run_coroutine_threadsafe(self._app.shutdown(), self._asyncio_loop).result(timeout=5)
        except Exception as e:
            log.warning("telegram_command_handler_shutdown_failed", error=str(e))
