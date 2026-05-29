"""Smart order-type selection based on urgency and spread."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import structlog

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker
    from zeus.execution.order_manager import OrderManager
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)

Urgency = Literal["low", "normal", "high"]


@dataclass
class ExecutionPlan:
    order_type: Literal["market", "limit"]
    limit_price: Optional[float]
    rationale: str


def plan_entry(
    bid: float, ask: float, urgency: Urgency = "normal",
    max_spread_bps: float = 30.0,
) -> ExecutionPlan:
    """
    Decide order type and price for a BUY entry.
      - urgency=high → market
      - tight spread → market
      - else → limit near mid/ask
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return ExecutionPlan("market", None, "invalid_quote")

    mid = (bid + ask) / 2
    spread_bps = (ask - bid) / mid * 10_000

    if urgency == "high":
        return ExecutionPlan("market", None, "high_urgency")

    if spread_bps <= 5:
        return ExecutionPlan("market", None, "tight_spread")

    if spread_bps > max_spread_bps:
        # Wide spread — limit at mid, patient
        return ExecutionPlan("limit", round(mid, 2), "wide_spread_patient")

    if urgency == "low":
        return ExecutionPlan("limit", round(mid, 2), "low_urgency_mid")

    # Normal urgency: limit at ask + 0.1% to cross at open but not fully market
    limit = round(ask * 1.001, 2)
    return ExecutionPlan("limit", limit, "normal_urgency_cross")


def plan_exit_signal(bid: float, ask: float) -> ExecutionPlan:
    """Exit on signal weakening — patient, limit at bid."""
    if bid <= 0:
        return ExecutionPlan("market", None, "invalid_quote")
    return ExecutionPlan("limit", round(bid, 2), "patient_exit")


def plan_exit_stop() -> ExecutionPlan:
    """Stop triggered — never miss a stop, always market."""
    return ExecutionPlan("market", None, "stop_trigger")


def execute_plan(
    om: "OrderManager",
    session: "Session",
    symbol: str,
    qty: int,
    side: str,
    plan: ExecutionPlan,
    *,
    strategy_id: str = "legacy",
) -> object:
    """Submit an order per the plan. Logs rationale.

    `strategy_id` is stamped on the persisted Order row so each of the three
    concurrent trader agents can be attributed to its fills.
    """
    log.info(
        "executing",
        symbol=symbol, side=side, qty=qty, strategy_id=strategy_id,
        type=plan.order_type, limit=plan.limit_price, why=plan.rationale,
    )
    if plan.order_type == "market" or plan.limit_price is None:
        return om.submit_market(session, symbol, qty, side, strategy_id=strategy_id)
    return om.submit_limit(
        session, symbol, qty, side, plan.limit_price, strategy_id=strategy_id,
    )
