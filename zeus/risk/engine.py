"""Risk engine — pre-trade checks + real-time orchestration of sub-systems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from zeus.risk.daily_loss import DailyLossMonitor, DailyLossState
from zeus.risk.drawdown_guard import DrawdownGuard, DrawdownState, DrawdownLevel
from zeus.risk.limits import RiskLimits

log = structlog.get_logger(__name__)


@dataclass
class ProposedTrade:
    symbol: str
    shares: int
    price: float
    side: str  # 'buy' | 'sell'
    sector: Optional[str] = None
    avg_daily_dollar_volume: Optional[float] = None

    @property
    def notional(self) -> float:
        return self.shares * self.price


@dataclass
class PreTradeResult:
    approved: bool
    reason: str = ""
    adjusted_shares: Optional[int] = None
    adjusted_notional: Optional[float] = None


@dataclass
class PortfolioSnapshot:
    portfolio_value: float
    cash_balance: float
    buying_power: float
    positions: Dict[str, dict] = field(default_factory=dict)
    # positions[symbol] = {'shares': int, 'market_value': float, 'sector': str|None}

    def position_notional(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p["market_value"] if p else 0.0

    def sector_notional(self, sector: str) -> float:
        return sum(
            p.get("market_value", 0.0)
            for p in self.positions.values()
            if p.get("sector") == sector
        )


class RiskEngine:
    """
    Unified risk enforcement. Wraps drawdown guard + daily loss monitor + hard limits.
    """

    def __init__(
        self,
        limits: RiskLimits,
        starting_capital: float = 100_000.0,
        starting_peak: Optional[float] = None,
    ):
        self._limits = limits
        self._drawdown = DrawdownGuard(starting_capital, limits, starting_peak)
        self._daily_loss = DailyLossMonitor(limits)
        self._paused = False
        self._kill_switch_active = False
        # Per-agent halts (overseer-controlled). Key = strategy_id / agent_id,
        # value = reason string. Checked by StrategyManager before submitting
        # each strategy's entries.
        self._halted_agents: Dict[str, str] = {}

    # ─── State updates ────────────────────────────────────────────────────────
    def update_portfolio_value(self, portfolio_value: float) -> DrawdownState:
        state = self._drawdown.update(portfolio_value)
        if state.level == DrawdownLevel.KILL:
            self._kill_switch_active = True
        return state

    def update_daily_pnl(self, daily_pnl: float) -> DailyLossState:
        state = self._daily_loss.check(daily_pnl)
        if state.close_all:
            self._kill_switch_active = True
        return state

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        if not self._kill_switch_active:
            self._paused = False

    def trip_kill_switch(self) -> None:
        self._kill_switch_active = True

    # ─── Per-agent halts (overseer) ──────────────────────────────────────────
    def halt_agent(self, agent_id: str, reason: str) -> None:
        """Prevent `agent_id` from submitting new entries until resume_agent().

        Existing positions are untouched — the agent's time-exits and manual
        exits still run. Overseer uses this for mandate drift, research-trader
        decoupling, and per-strategy drawdown breaches.
        """
        self._halted_agents[agent_id] = reason
        log.warning("agent_halted", agent_id=agent_id, reason=reason)

    def resume_agent(self, agent_id: str) -> None:
        if self._halted_agents.pop(agent_id, None) is not None:
            log.info("agent_resumed", agent_id=agent_id)

    def is_agent_halted(self, agent_id: str) -> bool:
        return agent_id in self._halted_agents

    def halted_agents(self) -> Dict[str, str]:
        return dict(self._halted_agents)

    # ─── Pre-trade checks ─────────────────────────────────────────────────────
    def pre_trade_check(
        self,
        trade: ProposedTrade,
        portfolio: PortfolioSnapshot,
        drawdown_state: Optional[DrawdownState] = None,
        daily_loss_state: Optional[DailyLossState] = None,
    ) -> PreTradeResult:
        """
        Validates a proposed BUY (entry) against all hard limits.
        For exits, always approve — closing positions is always allowed.
        """
        if trade.side == "sell":
            return PreTradeResult(approved=True)

        if self._kill_switch_active:
            return PreTradeResult(False, "kill_switch_active")

        if self._paused:
            return PreTradeResult(False, "system_paused")

        if drawdown_state and drawdown_state.stop_new_entries:
            return PreTradeResult(False, f"drawdown_level_{drawdown_state.level.value}")

        if daily_loss_state and daily_loss_state.stop_new_entries:
            return PreTradeResult(False, "daily_loss_limit_hit")

        notional = trade.notional
        pv = portfolio.portfolio_value
        if pv <= 0:
            return PreTradeResult(False, "zero_portfolio_value")

        # Min position size
        if notional < self._limits.min_position_notional:
            return PreTradeResult(False, "below_min_position_size")

        # Max single order notional
        if notional > self._limits.max_single_order_notional:
            capped_shares = int(self._limits.max_single_order_notional // trade.price)
            return PreTradeResult(
                False,
                "exceeds_max_single_order",
                adjusted_shares=capped_shares,
                adjusted_notional=capped_shares * trade.price,
            )

        # Max position size
        max_position_notional = pv * self._limits.max_position_pct
        existing = portfolio.position_notional(trade.symbol)
        projected = existing + notional
        if projected > max_position_notional:
            available = max_position_notional - existing
            if available < self._limits.min_position_notional:
                return PreTradeResult(False, "position_limit_exceeded")
            capped_shares = int(available // trade.price)
            return PreTradeResult(
                False,
                "position_limit_exceeded",
                adjusted_shares=capped_shares,
                adjusted_notional=capped_shares * trade.price,
            )

        # Sector concentration
        if trade.sector:
            max_sector_notional = pv * self._limits.max_sector_pct
            projected_sector = portfolio.sector_notional(trade.sector) + notional
            if projected_sector > max_sector_notional:
                return PreTradeResult(False, "sector_limit_exceeded")

        # Buying power
        required_cash = notional
        available_cash = portfolio.buying_power * (1 - self._limits.broker_buying_power_buffer)
        if required_cash > available_cash:
            return PreTradeResult(False, "insufficient_buying_power")

        # Liquidity — position can't be > 5% of ADV
        if trade.avg_daily_dollar_volume:
            adv_cap = trade.avg_daily_dollar_volume * self._limits.min_liquidity_adv_multiple
            if notional > adv_cap:
                capped_shares = int(adv_cap // trade.price)
                return PreTradeResult(
                    False,
                    "liquidity_limit",
                    adjusted_shares=capped_shares,
                    adjusted_notional=capped_shares * trade.price,
                )

        return PreTradeResult(approved=True)
