"""Kill switch: cancels all orders, closes all positions, locks system."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker
    from zeus.monitoring.telegram_bot import TelegramNotifier
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)


class KillSwitch:
    """
    Executes emergency shutdown:
      1. Cancel all open orders
      2. Submit market-sell for every open position
      3. Persist risk event + heartbeat
      4. Send Telegram EMERGENCY message
      5. Set a local lockfile so subsequent runs refuse to trade
    """

    LOCKFILE_NAME = "kill_switch.lock"

    def __init__(
        self,
        broker: "AlpacaBroker",
        telegram: Optional["TelegramNotifier"],
        lockfile_dir: str = "./artifacts",
    ):
        self._broker = broker
        self._telegram = telegram
        self._lockfile_dir = lockfile_dir

    def activate(self, reason: str, session: Optional["Session"] = None) -> dict:
        log.critical("KILL_SWITCH_ACTIVATED", reason=reason)

        cancelled = 0
        closed = 0
        errors: list[str] = []

        # 1. Cancel open orders
        try:
            orders = self._broker.get_open_orders()
            cancelled_before = len(orders)
            self._broker.cancel_all_orders()
            cancelled = cancelled_before
        except Exception as e:
            errors.append(f"cancel_all:{e}")

        # 2. Close all positions (broker handles atomically)
        try:
            positions = self._broker.get_positions()
            closed = len(positions)
            self._broker.close_all_positions()
        except Exception as e:
            errors.append(f"close_all:{e}")

        # 3. Persist risk event
        if session is not None:
            try:
                from zeus.data.storage.database import RiskEvent
                evt = RiskEvent(
                    ts=datetime.now(timezone.utc),
                    event_type="kill_switch",
                    severity="EMERGENCY",
                    description=reason,
                    action_taken=f"cancelled={cancelled} closed={closed}",
                )
                session.add(evt)
                session.commit()
            except Exception as e:
                errors.append(f"db_persist:{e}")

        # 4. Telegram
        if self._telegram is not None:
            try:
                self._telegram.send_kill_switch(reason=reason, positions_closed=closed)
            except Exception as e:
                errors.append(f"telegram:{e}")

        # 5. Create lockfile
        import os
        os.makedirs(self._lockfile_dir, exist_ok=True)
        lockfile = os.path.join(self._lockfile_dir, self.LOCKFILE_NAME)
        with open(lockfile, "w") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}\n{reason}\n")

        result = {
            "cancelled_orders": cancelled,
            "closed_positions": closed,
            "errors": errors,
            "reason": reason,
            "lockfile": lockfile,
        }
        log.critical("KILL_SWITCH_COMPLETE", **result)
        return result

    @classmethod
    def is_tripped(cls, lockfile_dir: str = "./artifacts") -> bool:
        import os
        return os.path.exists(os.path.join(lockfile_dir, cls.LOCKFILE_NAME))

    @classmethod
    def clear(cls, lockfile_dir: str = "./artifacts") -> None:
        """Manual override — remove lockfile. Requires human review."""
        import os
        path = os.path.join(lockfile_dir, cls.LOCKFILE_NAME)
        if os.path.exists(path):
            os.remove(path)
