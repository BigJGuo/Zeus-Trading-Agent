"""Portfolio construction from ranked signals."""
from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from zeus.risk.engine import PortfolioSnapshot, ProposedTrade, RiskEngine
from zeus.risk.limits import RiskLimits
from zeus.risk.position_sizer import SizingInputs, kelly_position_size, shares_from_target

log = structlog.get_logger(__name__)

_EXIT_SCORE_THRESHOLD = 0.0
_DEFAULT_PRICE = 100.0  # fallback when price not in signals


class PortfolioConstructor:
    def __init__(self, risk_engine: RiskEngine, limits: RiskLimits) -> None:
        self._risk = risk_engine
        self._limits = limits

    def construct(
        self,
        signals: pd.DataFrame,
        current_positions: dict[str, dict],
        portfolio_value: float,
        buying_power: float,
        regime: str,
        regime_params: dict,
        kelly_fraction: float = 0.25,
        recent_ic: float = 0.03,
    ) -> dict[str, Any]:
        max_positions: int = regime_params["max_positions"]
        max_exposure_pct: float = regime_params["max_exposure_pct"]
        max_new_notional = portfolio_value * max_exposure_pct

        snapshot = PortfolioSnapshot(
            portfolio_value=portfolio_value,
            cash_balance=buying_power,
            buying_power=buying_power,
            positions={
                sym: {
                    "shares": pos.get("shares", 0),
                    "market_value": pos.get("market_value", 0.0),
                    "sector": pos.get("sector"),
                }
                for sym, pos in current_positions.items()
            },
        )

        signal_symbols: set[str] = set(signals["symbol"].tolist())
        top_symbols: set[str] = set(signals.head(max_positions)["symbol"].tolist())

        exits: list[ProposedTrade] = []
        for sym, pos in current_positions.items():
            if sym not in signal_symbols:
                exits.append(
                    ProposedTrade(
                        symbol=sym,
                        shares=pos.get("shares", 0),
                        price=pos.get("price", _DEFAULT_PRICE),
                        side="sell",
                        sector=pos.get("sector"),
                    )
                )
                continue
            row = signals[signals["symbol"] == sym]
            if not row.empty and float(row.iloc[0]["blended_score"]) < _EXIT_SCORE_THRESHOLD:
                exits.append(
                    ProposedTrade(
                        symbol=sym,
                        shares=pos.get("shares", 0),
                        price=pos.get("price", _DEFAULT_PRICE),
                        side="sell",
                        sector=pos.get("sector"),
                    )
                )

        sector_exposure: dict[str, float] = {}
        for pos in current_positions.values():
            sec = pos.get("sector")
            if sec:
                sector_exposure[sec] = sector_exposure.get(sec, 0.0) + pos.get("market_value", 0.0)

        entries: list[ProposedTrade] = []
        holds: list[str] = []
        committed_notional = 0.0
        positions_added = 0
        n_considered = 0
        n_rejected = 0

        for _, row in signals.iterrows():
            sym = str(row["symbol"])

            if sym in current_positions and sym in top_symbols:
                holds.append(sym)
                continue

            if sym in current_positions:
                continue

            if positions_added >= max_positions:
                break

            n_considered += 1

            price = float(row.get("price", _DEFAULT_PRICE))
            vol = float(row.get("vol_estimate", 0.20)) or 0.20
            liq_score = float(row.get("liquidity_score", 0.5))
            liq_adv = liq_score * 50_000_000

            sizing = kelly_position_size(
                SizingInputs(
                    expected_return_5d=float(row["expected_return"]),
                    vol_estimate_annual=vol,
                    confidence=float(row["confidence"]),
                    liquidity_adv_usd=liq_adv,
                    regime=regime,
                    recent_ic=recent_ic,
                ),
                max_position_pct=self._limits.max_position_pct,
                kelly_fraction=kelly_fraction,
            )

            target_pct = sizing.target_pct
            if target_pct <= 0:
                n_rejected += 1
                continue

            notional = portfolio_value * target_pct
            if committed_notional + notional > max_new_notional:
                notional = max_new_notional - committed_notional
                if notional < self._limits.min_position_notional:
                    n_rejected += 1
                    continue
                target_pct = notional / portfolio_value

            shares = shares_from_target(target_pct, portfolio_value, price)
            if shares <= 0:
                n_rejected += 1
                continue

            sector = str(row.get("sector", "")) or None
            if sector:
                current_sector = sector_exposure.get(sector, 0.0)
                if current_sector + notional > portfolio_value * self._limits.max_sector_pct:
                    n_rejected += 1
                    log.debug("sector_cap_reject", symbol=sym, sector=sector)
                    continue

            trade = ProposedTrade(
                symbol=sym,
                shares=shares,
                price=price,
                side="buy",
                sector=sector,
                avg_daily_dollar_volume=liq_adv,
            )

            result = self._risk.pre_trade_check(trade, snapshot)
            if not result.approved:
                if result.adjusted_shares and result.adjusted_shares > 0:
                    adjusted_notional = result.adjusted_notional or result.adjusted_shares * price
                    if adjusted_notional >= self._limits.min_position_notional:
                        trade = ProposedTrade(
                            symbol=sym,
                            shares=result.adjusted_shares,
                            price=price,
                            side="buy",
                            sector=sector,
                            avg_daily_dollar_volume=liq_adv,
                        )
                        notional = adjusted_notional
                    else:
                        n_rejected += 1
                        log.debug("pre_trade_reject", symbol=sym, reason=result.reason)
                        continue
                else:
                    n_rejected += 1
                    log.debug("pre_trade_reject", symbol=sym, reason=result.reason)
                    continue

            entries.append(trade)
            committed_notional += notional
            positions_added += 1
            if sector:
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + notional

        log.info(
            "portfolio_constructed",
            regime=regime,
            n_considered=n_considered,
            n_rejected=n_rejected,
            n_entries=len(entries),
            n_exits=len(exits),
            n_holds=len(holds),
        )

        return {
            "entries": entries,
            "exits": exits,
            "holds": holds,
            "rationale": {
                "regime": regime,
                "max_positions": max_positions,
                "max_exposure_pct": max_exposure_pct,
                "committed_notional": committed_notional,
                "n_considered": n_considered,
                "n_rejected": n_rejected,
            },
        }
