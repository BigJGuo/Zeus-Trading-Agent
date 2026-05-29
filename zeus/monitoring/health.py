"""
System health checks for ZEUS.
Runs every 5 minutes and alerts on failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import structlog

log = structlog.get_logger(__name__)


@dataclass
class HealthCheckResult:
    name: str
    healthy: bool
    message: str
    latency_ms: Optional[float] = None


class HealthMonitor:
    """Runs all registered health checks and aggregates results."""

    def __init__(self):
        self._checks: List[Callable[[], HealthCheckResult]] = []

    def register(self, check_fn: Callable[[], HealthCheckResult]) -> None:
        self._checks.append(check_fn)

    def run_all(self) -> List[HealthCheckResult]:
        results = []
        for check_fn in self._checks:
            try:
                import time
                start = time.time()
                result = check_fn()
                result.latency_ms = (time.time() - start) * 1000
                results.append(result)
                if not result.healthy:
                    log.warning("Health check FAILED", check=result.name, msg=result.message)
            except Exception as e:
                results.append(HealthCheckResult(
                    name=check_fn.__name__,
                    healthy=False,
                    message=f"Exception: {e}",
                ))
        return results

    def is_healthy(self) -> bool:
        results = self.run_all()
        return all(r.healthy for r in results)

    def summary(self) -> dict:
        results = self.run_all()
        return {
            "healthy": all(r.healthy for r in results),
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [
                {
                    "name": r.name,
                    "healthy": r.healthy,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }


def make_db_check(engine) -> Callable:
    def check_database() -> HealthCheckResult:
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return HealthCheckResult("database", True, "OK")
        except Exception as e:
            return HealthCheckResult("database", False, str(e))
    return check_database


def make_redis_check(cache) -> Callable:
    def check_redis() -> HealthCheckResult:
        try:
            ok = cache.ping()
            return HealthCheckResult("redis", ok, "OK" if ok else "PING failed")
        except Exception as e:
            return HealthCheckResult("redis", False, str(e))
    return check_redis


def make_alpaca_check(broker) -> Callable:
    def check_alpaca() -> HealthCheckResult:
        try:
            ok = broker.ping()
            return HealthCheckResult("alpaca", ok, "OK" if ok else "Unreachable")
        except Exception as e:
            return HealthCheckResult("alpaca", False, str(e))
    return check_alpaca


def make_data_freshness_check(db_session_factory, max_age_hours: float = 2.0) -> Callable:
    def check_data_freshness() -> HealthCheckResult:
        try:
            from sqlalchemy import text
            session = db_session_factory()
            result = session.execute(
                text("SELECT MAX(ts) FROM ohlcv_daily")
            ).scalar()
            session.close()
            if result is None:
                return HealthCheckResult("data_freshness", False, "No data in ohlcv_daily")
            age = (datetime.utcnow() - result.replace(tzinfo=None)).total_seconds() / 3600
            if age > max_age_hours:
                return HealthCheckResult("data_freshness", False,
                                         f"Stale: last data {age:.1f}h ago")
            return HealthCheckResult("data_freshness", True, f"Fresh ({age:.1f}h old)")
        except Exception as e:
            return HealthCheckResult("data_freshness", False, str(e))
    return check_data_freshness


def make_disk_space_check(path: str = "/", min_gb: float = 5.0) -> Callable:
    def check_disk_space() -> HealthCheckResult:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)
        if free_gb < min_gb:
            return HealthCheckResult("disk_space", False,
                                     f"Only {free_gb:.1f}GB free (min {min_gb}GB)")
        return HealthCheckResult("disk_space", True, f"{free_gb:.1f}GB free")
    return check_disk_space
