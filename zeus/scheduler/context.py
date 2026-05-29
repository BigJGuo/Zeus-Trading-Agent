"""Module-level registry for scheduler dependencies.

APScheduler's persistent jobstores (`SQLAlchemyJobStore`, etc.) pickle a
job's args at registration time. The TradingLoop and AgentOrchestrator
own non-picklable handles — DB engine pools, broker SDK clients,
background threads — so they can't be passed as job args once jobs are
persisted.

The fix is the same trick FastAPI / Flask use: stash the runtime objects
in a module-level slot and have each job function fetch them at call
time. From APScheduler's perspective the jobs are parameterless and
trivially picklable; from our code's perspective they still have access
to the full system.

Set once in `build_scheduler` before any job fires. `get_loop()` /
`get_orch()` return `None` if not yet configured; jobs short-circuit on
None so an early misfire doesn't crash the worker.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler
    from zeus.live.trading_loop import TradingLoop
    from zeus.scheduler.agent_jobs import AgentOrchestrator


_LOOP: Optional["TradingLoop"] = None
_ORCH: Optional["AgentOrchestrator"] = None
_SCHEDULER: Optional["BackgroundScheduler"] = None


def set_loop(loop: "TradingLoop") -> None:
    global _LOOP
    _LOOP = loop


def get_loop() -> Optional["TradingLoop"]:
    return _LOOP


def set_orch(orch: Optional["AgentOrchestrator"]) -> None:
    global _ORCH
    _ORCH = orch


def get_orch() -> Optional["AgentOrchestrator"]:
    return _ORCH


def set_scheduler(scheduler: "BackgroundScheduler") -> None:
    """Expose the live scheduler so execution-layer code (TWAP slicing,
    one-shot deferred sends) can schedule follow-up work without having
    to plumb a reference through every layer."""
    global _SCHEDULER
    _SCHEDULER = scheduler


def get_scheduler() -> Optional["BackgroundScheduler"]:
    return _SCHEDULER
