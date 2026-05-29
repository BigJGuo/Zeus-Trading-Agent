"""Paper burn-in go/no-go dashboard.

Phase 6 promotion gate: every trader must clear all four tests below for
four consecutive weeks before real-money trading is authorized.

  1. Rolling 4-week annualized Sharpe > 1.0 (net of cost)
  2. Hit rate within ±3pp of its backtest projection
  3. Per-strategy max drawdown < 15%
  4. No research-trader decoupled fills in the rolling 14-day window

Run:  python -m scripts.paper_burnin_status
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from zeus.agents.journal import TRADER_AGENTS
from zeus.agents.metrics import (
    AgentMetrics,
    ResearchMetrics,
    compute_research_metrics,
    compute_trader_metrics,
)

# Backtest projections from the approved plan: day 0.67, swing 0.60, long-term 0.71.
BACKTEST_HIT_RATES: Dict[str, float] = {
    "day": 0.67,
    "swing": 0.60,
    "long_term": 0.71,
}

HIT_RATE_TOLERANCE = 0.03          # ±3pp
MIN_SHARPE = 1.0
MAX_DD = 0.15


@dataclass
class GateRow:
    agent_id: str
    sharpe_pass: bool
    sharpe: float
    hit_rate_pass: bool
    hit_rate: float
    hit_rate_target: float
    max_dd_pass: bool
    max_dd: float
    coupling_pass: bool
    decoupled_fills: int
    go: bool
    metrics: AgentMetrics
    research: Optional[ResearchMetrics] = None


def evaluate_agent(agent_id: str, window_days: int, environment: str) -> GateRow:
    m = compute_trader_metrics(agent_id, window_days=window_days, environment=environment)
    paired = {"day": "day_research", "swing": "swing_research",
              "long_term": "long_term_research"}[agent_id]
    r = compute_research_metrics(paired, window_days=14)

    target = BACKTEST_HIT_RATES[agent_id]
    sharpe_pass = m.sharpe_annualized >= MIN_SHARPE
    hit_pass = abs(m.hit_rate - target) <= HIT_RATE_TOLERANCE
    dd_pass = m.max_drawdown_pct < MAX_DD
    coupling_pass = r.trader_decoupled_trades == 0

    return GateRow(
        agent_id=agent_id,
        sharpe_pass=sharpe_pass,
        sharpe=m.sharpe_annualized,
        hit_rate_pass=hit_pass,
        hit_rate=m.hit_rate,
        hit_rate_target=target,
        max_dd_pass=dd_pass,
        max_dd=m.max_drawdown_pct,
        coupling_pass=coupling_pass,
        decoupled_fills=r.trader_decoupled_trades,
        go=(sharpe_pass and hit_pass and dd_pass and coupling_pass),
        metrics=m,
        research=r,
    )


def format_row(row: GateRow) -> str:
    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return (
        f"{row.agent_id:11} "
        f"sharpe={row.sharpe:>6.2f} [{_mark(row.sharpe_pass)}]  "
        f"hit={row.hit_rate:>5.2%} tgt={row.hit_rate_target:>5.2%} [{_mark(row.hit_rate_pass)}]  "
        f"max_dd={row.max_dd:>5.2%} [{_mark(row.max_dd_pass)}]  "
        f"coupling [{_mark(row.coupling_pass)}]  "
        f"→ {'GO' if row.go else 'NO-GO'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--environment", default="paper")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of the text table.")
    args = parser.parse_args()

    rows: List[GateRow] = [
        evaluate_agent(aid, args.window_days, args.environment)
        for aid in TRADER_AGENTS
    ]

    if args.json:
        print(json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "window_days": args.window_days,
                "environment": args.environment,
                "rows": [
                    {
                        **{k: v for k, v in asdict(r).items()
                           if k not in ("metrics", "research")},
                    } for r in rows
                ],
                "overall_go": all(r.go for r in rows),
            }, indent=2,
        ))
        return 0

    print(f"=== Paper burn-in status @ {datetime.now(timezone.utc).isoformat(timespec='seconds')} ===")
    print(f"Window: last {args.window_days} days ({args.environment})")
    print()
    for row in rows:
        print(format_row(row))
    print()
    overall = all(r.go for r in rows)
    print(f"OVERALL: {'GO — eligible for real-money promotion' if overall else 'NO-GO'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
