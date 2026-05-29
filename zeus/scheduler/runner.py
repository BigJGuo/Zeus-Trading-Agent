"""APScheduler entrypoint — wires jobs to TradingLoop instance."""
from __future__ import annotations

import signal
import sys
import time
import structlog
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone

from zeus.config.settings import get_settings
from zeus.config.strategies import get_strategies
from zeus.data.ingestion.ohlcv_ingester import OHLCVIngester
from zeus.data.ingestion.yfinance_client import YFinanceClient
from zeus.execution.alpaca_broker import AlpacaBroker
from zeus.execution.order_manager import OrderManager
from zeus.execution.reconciler import PositionReconciler
from zeus.features.pipeline import FeaturePipeline
from zeus.live.kill_switch import KillSwitch
from zeus.live.strategy import StrategyAllocator, StrategyContext
from zeus.live.strategy_manager import StrategyManager
from zeus.live.trading_loop import TradingLoop
from zeus.models.base import BaseModel
from zeus.models.regime_detector import RuleBasedRegimeDetector
from zeus.models.return_predictor import XGBReturnPredictor
from zeus.monitoring.telegram_bot import TelegramNotifier
from zeus.monitoring.telegram_commands import TelegramCommandHandler
from zeus.portfolio.constructor import PortfolioConstructor
from zeus.research.after_hours import AfterHoursContext
from zeus.risk.engine import RiskEngine
from zeus.risk.limits import RiskLimits
from zeus.scheduler.agent_jobs import AGENT_JOB_MANIFEST, AgentOrchestrator
from zeus.scheduler.jobs import JOB_MANIFEST
from zeus.signals.filter import SignalFilter
from zeus.signals.generator import SignalGenerator

log = structlog.get_logger(__name__)
ET = timezone("America/New_York")


class _NullPredictor(BaseModel):
    """Safe stand-in when no promoted model is registered.

    Returns zero predictions for every row so the signal generator, filter, and
    portfolio constructor all run without errors. A zero-signal plan means the
    constructor's fallback path (equal-weight across the filtered universe) is
    what ends up on the plan. This is intentional: if the model is garbage or
    missing, we want the agent to either stay flat or deploy a known-neutral
    fallback, not crash the scheduler.
    """

    version_ = "null_predictor"

    def fit(self, X, y, **kwargs) -> None:
        return None

    def predict(self, X):
        import pandas as pd
        return pd.Series(0.0, index=X.index, name="predicted_return")

    def get_params(self):
        return {}

    def get_feature_importance(self):
        import pandas as pd
        return pd.DataFrame(columns=["feature", "importance"])


def build_trading_loop() -> TradingLoop:
    settings = get_settings()
    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=settings.alpaca_paper,
    )
    env = "paper" if settings.is_paper() else "live"
    order_mgr = OrderManager(broker, environment=env)
    reconciler = PositionReconciler(broker, environment=env)
    telegram = TelegramNotifier(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )

    limits = RiskLimits()  # TODO: load from config.yaml
    risk_engine = RiskEngine(limits=limits, starting_capital=100_000.0)

    kill = KillSwitch(
        broker=broker,
        telegram=telegram,
        lockfile_dir=settings.artifacts_path,
    )

    loop = TradingLoop(
        broker=broker,
        order_manager=order_mgr,
        reconciler=reconciler,
        telegram=telegram,
        risk_engine=risk_engine,
        kill_switch=kill,
        environment=env,
    )

    yf_client = YFinanceClient()
    ingester = OHLCVIngester(broker, yf_client)
    feature_pipeline = FeaturePipeline(feature_version="v1")

    # Single-agent legacy model — kept for nightly_retrain + data refresh paths.
    legacy_model = _load_latest_model() or _NullPredictor()
    research_ctx = AfterHoursContext(
        broker=broker,
        yf_client=yf_client,
        ingester=ingester,
        feature_pipeline=feature_pipeline,
        model=legacy_model,
        signal_generator=SignalGenerator(model=legacy_model),
        signal_filter=SignalFilter(),
        portfolio_constructor=PortfolioConstructor(risk_engine=risk_engine, limits=limits),
        regime_detector=RuleBasedRegimeDetector(),
        risk_engine=risk_engine,
    )
    loop.attach_research_context(research_ctx)

    # Multi-strategy: one StrategyContext per enabled strategy.
    bundle = get_strategies()
    contexts: dict[str, StrategyContext] = {}
    for s in bundle.enabled():
        m = _load_model_by_name(s.model_name) or _NullPredictor()
        contexts[s.id] = StrategyContext(
            config=s,
            model=m,
            signal_generator=SignalGenerator(model=m),
            signal_filter=SignalFilter(),
            portfolio_constructor=PortfolioConstructor(risk_engine=risk_engine, limits=limits),
            feature_pipeline=feature_pipeline,
        )
        log.info(
            "strategy_context_built",
            strategy_id=s.id, model_name=s.model_name,
            model_version=getattr(m, "version_", "unknown"),
        )

    # Build the overseer-aware allocator so weekly reviews can actually
    # reshape capital. Falls back to static weights if the 7-agent system is
    # disabled (no ANTHROPIC_API_KEY configured).
    from zeus.risk.strategy_allocator import OverseerStrategyAllocator
    allocator = OverseerStrategyAllocator(
        strategies=bundle.strategies,
        globals_=bundle.globals,
        risk_engine=risk_engine,
    )
    manager = StrategyManager(
        contexts=contexts,
        allocator=allocator,
        broker=broker,
        order_manager=order_mgr,
        risk_engine=risk_engine,
        globals_=bundle.globals,
        environment=env,
    )
    loop.attach_strategy_manager(manager)
    loop._agent_orchestrator = None
    loop._agent_allocator = allocator
    return loop


# Marker keys → model class. Lets the loader recover when the registry
# row has no `__model_class__` config hint and no `model_class.txt` file
# alongside the artifact — older bundles written before those conventions
# were established. The list is ordered most-specific first so the
# detection doesn't return early on a generic key.
_PAYLOAD_KEY_TO_MODEL_PATH: tuple[tuple[str, str], ...] = (
    ("primary_cls", "zeus.models.rank_gated_predictor:CrossHorizonGatedPredictor"),
    ("rank_floor", "zeus.models.rank_gated_predictor:RankMagnitudeGatedPredictor"),
    ("meta_clf", "zeus.models.meta_labeler:MetaGatedPredictor"),
    ("members", "zeus.models.ensemble_return_predictor:EnsembleReturnPredictor"),
    ("params", "zeus.models.return_predictor:XGBReturnPredictor"),
)


def _detect_class_from_payload(artifact_path):
    """Peek at the joblib payload's top-level keys to identify the model
    class. Returns the import-string class hint or None.

    The legacy loader used to default to `XGBReturnPredictor` when no
    `__model_class__` hint was registered, which crashed on
    CrossHorizonGatedPredictor bundles (KeyError: 'params'). Sniffing
    payload structure is the only reliable way to recover from missing
    hints without manual data migration.
    """
    try:
        import joblib
        payload = joblib.load(artifact_path)
    except Exception as e:
        log.warning("payload_peek_failed", path=str(artifact_path), error=str(e))
        return None
    if not isinstance(payload, dict):
        return None
    for marker, cls_path in _PAYLOAD_KEY_TO_MODEL_PATH:
        if marker in payload:
            return cls_path
    return None


def _resolve_model_class(row, artifact_dir, artifact_path):
    """Resolve the model class for a `model_versions` row, in priority order:
    1. `config.__model_class__` field (newest convention).
    2. `model_class.txt` alongside the artifact (older convention).
    3. Sniffing the joblib payload's top-level keys (recovery for legacy
       rows where neither hint was written).
    Returns an import-string ``"module:ClassName"`` or None.
    """
    cfg = row.config or {}
    if isinstance(cfg, dict):
        hint = cfg.get("__model_class__")
        if hint:
            return hint
    txt_file = artifact_dir / "model_class.txt"
    if txt_file.exists():
        return txt_file.read_text().strip()
    return _detect_class_from_payload(artifact_path)


def _load_from_row(row, *, fallback_class):
    """Shared loader body used by both `_load_latest_model` and
    `_load_model_by_name`. Handles artifact-path resolution, class hint
    resolution (with payload-sniff fallback), and the actual load.

    `row.model_name` is already in the structured log line, so callers
    don't need to pass it through any sidechannel.
    """
    import importlib
    from pathlib import Path

    # `artifact_path` may be a dir (trainer.save_model) OR the .joblib
    # file itself (scripts/train_cross_horizon_bundle.py registers the
    # file path directly).
    raw = Path(row.artifact_path)
    if raw.is_file():
        artifact_dir = raw.parent
        artifact_path = raw
    else:
        artifact_dir = raw
        artifact_path = artifact_dir / "model.joblib"
    if not artifact_path.exists():
        log.warning("model_artifact_missing", version=row.version,
                    model_name=row.model_name, path=str(artifact_path))
        return None

    class_hint = _resolve_model_class(row, artifact_dir, artifact_path)

    try:
        if class_hint and ":" in class_hint:
            module_name, cls_name = class_hint.split(":", 1)
            klass = getattr(importlib.import_module(module_name), cls_name)
        else:
            klass = fallback_class
        m = klass.load(artifact_path)
        m.version_ = row.version
        log.info("model_loaded", version=row.version, status=row.status,
                 model_name=row.model_name, cls=klass.__name__)
        return m
    except Exception as e:
        log.warning("model_load_failed", version=row.version,
                    model_name=row.model_name, error=str(e))
        return None


def _load_model_by_name(model_name: str):
    """Load the newest staging/production row whose model_name matches.

    Each of the three trader agents points at a different `model_name`
    (cross_horizon_day_h3, cross_horizon_swing_h5, cross_horizon_long_term_h20).
    """
    from sqlalchemy import select, desc
    from zeus.data.storage.database import ModelVersion, get_session_factory
    from zeus.models.rank_gated_predictor import CrossHorizonGatedPredictor

    try:
        with get_session_factory()() as session:
            row = session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_name == model_name)
                .where(ModelVersion.status.in_(("staging", "production")))
                .order_by(desc(ModelVersion.created_at), desc(ModelVersion.id))
                .limit(1)
            ).scalar_one_or_none()
    except Exception as e:
        log.warning("model_db_query_failed", model_name=model_name, error=str(e))
        return None

    if row is None:
        log.warning("no_valid_model", model_name=model_name)
        return None

    # Per-strategy bundles are CrossHorizon by convention; that's the
    # fallback when neither config hint, model_class.txt, nor payload
    # sniffing resolves a class.
    return _load_from_row(row, fallback_class=CrossHorizonGatedPredictor)


def _load_latest_model():
    """Load newest promoted model (staging or production) from model_versions.

    Rejects status='failed' rows. Returns None if nothing is registered;
    the caller falls back to `_NullPredictor` so the agent stays flat.

    The fallback class here is the newer CrossHorizonGatedPredictor — the
    legacy XGBReturnPredictor default was a footgun that crashed on every
    promoted CrossHorizon bundle. Payload sniffing in `_resolve_model_class`
    still picks XGB correctly when an older bundle is genuinely registered.
    """
    from sqlalchemy import select, desc
    from zeus.data.storage.database import ModelVersion, get_session_factory
    from zeus.models.rank_gated_predictor import CrossHorizonGatedPredictor

    try:
        with get_session_factory()() as session:
            row = session.execute(
                select(ModelVersion)
                .where(ModelVersion.status.in_(("staging", "production")))
                .order_by(desc(ModelVersion.created_at), desc(ModelVersion.id))
                .limit(1)
            ).scalar_one_or_none()
    except Exception as e:
        log.warning("model_db_query_failed", error=str(e))
        return None

    if row is None:
        log.warning("no_valid_model_in_registry",
                    hint="all rows are status='failed' or model_versions is empty")
        return None

    return _load_from_row(row, fallback_class=CrossHorizonGatedPredictor)


def reload_models_if_promoted(loop: TradingLoop) -> dict:
    """Hot-swap any strategy's model when a newer staging/production version
    appears in model_versions.

    Bridges the gap between Sunday retraining (which writes a new row to
    model_versions) and the running scheduler — without this, a newly
    promoted model sits idle until the next process restart. Polled every
    30s by `reload_models_job`.

    Returns a per-strategy dict of `{strategy_id: new_version}` for any
    swaps that happened this tick.
    """
    swaps: dict[str, str] = {}
    manager = getattr(loop, "_strategy_mgr", None)
    if manager is None:
        return swaps
    # Each StrategyContext holds the SignalGenerator and the model — both
    # have to be updated atomically or the next predict() call would route
    # through the old model.
    try:
        strategy_ids = manager.strategy_ids()
    except Exception as e:
        log.warning("hot_reload_strategy_ids_failed", error=str(e))
        return swaps
    for sid in strategy_ids:
        try:
            ctx = manager.context(sid)
        except KeyError:
            continue
        current_version = getattr(getattr(ctx, "model", None), "version_", None)
        new_model = _load_model_by_name(ctx.config.model_name)
        if new_model is None:
            continue
        new_version = getattr(new_model, "version_", None)
        if new_version is None or new_version == current_version:
            continue
        # Swap both the context reference and the signal generator's cached
        # reference. Assignments are atomic under the GIL, so an in-flight
        # predict() call will complete on whichever model it had at the
        # moment of dereference.
        ctx.model = new_model
        sg = getattr(ctx, "signal_generator", None)
        if sg is not None and hasattr(sg, "_model"):
            sg._model = new_model
        swaps[sid] = str(new_version)
        log.info(
            "model_hot_swapped",
            strategy_id=sid,
            model_name=ctx.config.model_name,
            from_version=str(current_version) if current_version else None,
            to_version=str(new_version),
        )

    # Legacy single-model path used by the research context (nightly_retrain
    # + data_refresh fallbacks). Same swap pattern.
    research_ctx = getattr(loop, "_after_hours_ctx", None)
    if research_ctx is not None:
        current_legacy = getattr(getattr(research_ctx, "model", None), "version_", None)
        new_legacy = _load_latest_model()
        if new_legacy is not None:
            new_legacy_version = getattr(new_legacy, "version_", None)
            if new_legacy_version and new_legacy_version != current_legacy:
                research_ctx.model = new_legacy
                sg = getattr(research_ctx, "signal_generator", None)
                if sg is not None and hasattr(sg, "_model"):
                    sg._model = new_legacy
                swaps["__legacy__"] = str(new_legacy_version)
                log.info(
                    "legacy_model_hot_swapped",
                    from_version=str(current_legacy) if current_legacy else None,
                    to_version=str(new_legacy_version),
                )
    return swaps


def _refresh_if_stale(loop: TradingLoop, max_age_hours: float = 24.0) -> None:
    """On startup, auto-trigger a data refresh if ohlcv_daily is stale.

    Protects against the scheduled 16:30 ET refresh being missed due to downtime
    or restarts — without this, the system can sit on days-old bars until the
    next scheduled weekday refresh, causing the planner to run on stale data.
    """
    from datetime import datetime as _dt2
    from sqlalchemy import text
    from zeus.data.storage.database import get_session_factory

    try:
        with get_session_factory()() as session:
            last_ts = session.execute(text("SELECT MAX(ts) FROM ohlcv_daily")).scalar()
    except Exception as e:
        log.warning("startup_freshness_check_failed", error=str(e))
        return

    if last_ts is None:
        log.warning("startup_freshness_empty_table")
        return

    age_h = (_dt2.utcnow() - last_ts.replace(tzinfo=None)).total_seconds() / 3600.0
    if age_h <= max_age_hours:
        log.info("startup_data_fresh", age_hours=round(age_h, 2))
        return

    log.warning("startup_data_stale_triggering_refresh", age_hours=round(age_h, 2))
    try:
        loop.run_data_refresh()
        log.info("startup_data_refresh_complete")
    except Exception as e:
        log.error("startup_data_refresh_failed", error=str(e))


def _try_build_agent_orchestrator(loop: TradingLoop) -> "AgentOrchestrator | None":
    """Wire the 7-agent bundle if ANTHROPIC_API_KEY is present.

    Returns None (agents disabled) when the key is missing — scheduler still
    runs the existing single-model pipeline unaffected. Logs an explicit
    warning so it's obvious in production why the agents aren't firing.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        log.warning("agent_system_disabled_no_anthropic_key")
        return None
    try:
        from zeus.agents.runtime import build_agent_bundle
        from zeus.llm.client import AnthropicClient

        def _regime_fn() -> str:
            # Read the most recent SystemMetrics.regime row. Writers of that
            # row are the existing scheduled feature_engineering / EOD jobs.
            from sqlalchemy import select
            from zeus.data.storage.database import SystemMetrics, get_session_factory
            try:
                with get_session_factory()() as s:
                    regime = s.execute(
                        select(SystemMetrics.regime)
                        .order_by(SystemMetrics.ts.desc())
                        .limit(1)
                    ).scalar_one_or_none()
                return str(regime) if regime else "normal"
            except Exception:
                return "normal"

        budget = settings.llm_daily_budget_usd_per_agent
        opus = AnthropicClient(tier="opus", daily_budget_usd=budget)
        sonnet = AnthropicClient(tier="sonnet", daily_budget_usd=budget)
        haiku = AnthropicClient(tier="haiku", daily_budget_usd=budget)

        allocator = getattr(loop, "_agent_allocator", None)
        if allocator is None:
            log.error("agent_allocator_missing")
            return None

        bundle = build_agent_bundle(
            broker=getattr(loop, "_broker", None),
            risk_engine=loop._risk,
            allocator=allocator,
            opus_llm=opus, sonnet_llm=sonnet, haiku_llm=haiku,
            regime_fn=_regime_fn,
        )
        orch = AgentOrchestrator(
            bundle=bundle,
            trading_loop=loop,
            strategy_manager=loop._strategy_mgr,
            regime_fn=_regime_fn,
        )
        loop._agent_orchestrator = orch
        log.info("agent_orchestrator_built")
        return orch
    except Exception as e:
        log.error("agent_orchestrator_build_failed", error=str(e), exc_info=True)
        return None


def _startup_catchup(loop: TradingLoop, orch) -> None:
    """Run any agent jobs that should have already fired today to leave the
    system ready for the next market open.

    Idempotency: research agents safely write new journal rows on each run
    and the trader plan builder overwrites `ctx.current_plan`. Running an
    extra premarket scan or plan build costs LLM calls but does not corrupt
    state. Only fires on weekdays; weekends get the normal scheduled ticks.
    """
    from datetime import datetime as _dt
    from pytz import timezone as _tz
    from zeus.scheduler import agent_jobs

    now_et = _dt.now(_tz("America/New_York"))
    if now_et.weekday() >= 5:  # sat/sun
        return

    # 1. Make sure the model-plan exists for today — this is the LLM trader's
    # input. Fire the existing next-session planning job synchronously.
    need_plan = True
    try:
        sm = getattr(loop, "_strategy_mgr", None)
        if sm is not None:
            need_plan = not any(
                sm.context(sid).current_plan is not None
                and sm.context(sid).current_plan.plan_date.isoformat() == now_et.date().isoformat()
                for sid in sm.strategy_ids()
            )
    except Exception:
        pass
    if need_plan:
        log.info("startup_catchup_model_plan")
        try:
            loop.run_next_session_planning()
        except Exception as e:
            log.warning("startup_catchup_model_plan_failed", error=str(e))

    # Intraday backfill — if we boot mid-session, day_research needs fresh
    # 5-min bars immediately rather than waiting for the next cron tick.
    hour = now_et.hour
    minute = now_et.minute
    minutes_since_midnight = hour * 60 + minute
    if 9 * 60 + 30 <= minutes_since_midnight <= 16 * 60:
        log.info("startup_catchup_intraday_backfill")
        try:
            loop.run_intraday_backfill(lookback_minutes=240)
        except Exception as e:
            log.warning("startup_catchup_intraday_backfill_failed", error=str(e))

    if orch is None:
        return

    # Research briefs — fire if we missed last night's windows (before 09:30)
    if minutes_since_midnight < 9 * 60 + 30:
        log.info("startup_catchup_agents_premarket")
        try:
            agent_jobs.swing_research_eod_job()
        except Exception as e:
            log.warning("startup_catchup_swing_research_failed", error=str(e))
        try:
            agent_jobs.long_term_deep_dive_or_review_job()
        except Exception as e:
            log.warning("startup_catchup_lt_research_failed", error=str(e))
        try:
            agent_jobs.day_research_premarket_job()
        except Exception as e:
            log.warning("startup_catchup_day_research_failed", error=str(e))
        try:
            agent_jobs.trader_overnight_plan_job()
        except Exception as e:
            log.warning("startup_catchup_trader_plan_failed", error=str(e))
        try:
            agent_jobs.day_trader_morning_plan_job()
        except Exception as e:
            log.warning("startup_catchup_day_trader_failed", error=str(e))
    # Post-close boot (16:00–18:59 ET) — fire the trading-loop cascade
    # (data_refresh, feature_engineering) whose cron elapsed today.
    elif 16 * 60 <= minutes_since_midnight < 19 * 60:
        log.info("startup_catchup_postclose", now_et=now_et.isoformat())
        if minutes_since_midnight >= 16 * 60 + 30:
            try:
                loop.run_data_refresh()
            except Exception as e:
                log.warning("startup_catchup_data_refresh_failed", error=str(e))
        if minutes_since_midnight >= 17 * 60:
            try:
                loop.run_feature_engineering()
            except Exception as e:
                log.warning("startup_catchup_feature_eng_failed", error=str(e))
    # Evening boot (after 19:00 ET) — fire the full evening cascade whose
    # cron already elapsed tonight. Without this, a scheduler restart after
    # 19:00 silently skips data_refresh, feature_engineering,
    # next_session_planning, and the research/trader agent jobs until the
    # next business day. The order matters: data before features before
    # planning, then research briefs, then LLM trader decisions.
    elif minutes_since_midnight >= 19 * 60:
        log.info("startup_catchup_agents_evening", now_et=now_et.isoformat())
        # Trading-loop cascade (data → features → plans).
        try:
            loop.run_data_refresh()
        except Exception as e:
            log.warning("startup_catchup_data_refresh_failed", error=str(e))
        try:
            loop.run_feature_engineering()
        except Exception as e:
            log.warning("startup_catchup_feature_eng_failed", error=str(e))
        if minutes_since_midnight >= 21 * 60:
            try:
                loop.run_next_session_planning()
            except Exception as e:
                log.warning("startup_catchup_plan_failed", error=str(e))
        # Agent cascade.
        try:
            agent_jobs.swing_research_eod_job()
        except Exception as e:
            log.warning("startup_catchup_swing_research_failed", error=str(e))
        if minutes_since_midnight >= 19 * 60 + 15:
            try:
                agent_jobs.long_term_deep_dive_or_review_job()
            except Exception as e:
                log.warning("startup_catchup_lt_research_failed", error=str(e))
        if minutes_since_midnight >= 21 * 60 + 30:
            try:
                agent_jobs.trader_overnight_plan_job()
            except Exception as e:
                log.warning("startup_catchup_trader_plan_failed", error=str(e))


def build_scheduler(loop: TradingLoop) -> BackgroundScheduler:
    """Wire jobs into APScheduler with a SQLAlchemyJobStore backing.

    The persistent jobstore gives crash recovery — if the scheduler dies
    mid-day, restarting picks up the persisted `next_run_time` and fires
    missed crons within `misfire_grace_time`. Jobs are stored without args
    (they fetch `loop` / `orch` from `zeus.scheduler.context` at call
    time), because the live objects hold non-picklable handles (DB pool,
    broker SDK) and APScheduler must pickle args to persist them.

    `replace_existing=True` means a process restart with changed job
    definitions (new cron, new interval) wins over the persisted rows —
    so editing JOB_MANIFEST takes effect on the next boot, not the next
    week.
    """
    from zeus.scheduler.context import set_loop, set_orch

    # Make the loop visible to every (parameterless) job before the
    # scheduler starts firing.
    set_loop(loop)

    settings = get_settings()
    try:
        jobstore = SQLAlchemyJobStore(
            url=settings.database_url,
            tablename="apscheduler_jobs",
        )
        scheduler = BackgroundScheduler(
            timezone=ET,
            jobstores={"default": jobstore},
        )
        log.info("scheduler_jobstore", kind="sqlalchemy", table="apscheduler_jobs")
    except Exception as e:
        # Fall back to in-memory if the DB isn't reachable yet (early-boot
        # or local dev). Log loudly so the operator sees we lost crash
        # recovery for this process lifetime.
        log.error("scheduler_jobstore_fallback_to_memory", error=str(e))
        scheduler = BackgroundScheduler(timezone=ET)

    for job in JOB_MANIFEST:
        params = dict(job)
        trigger_type = params.pop("trigger")
        fn = params.pop("func")

        if trigger_type == "cron":
            trigger = CronTrigger(timezone=ET, **params)
        elif trigger_type == "interval":
            trigger = IntervalTrigger(**params)
        else:
            raise ValueError(f"unknown trigger: {trigger_type}")

        scheduler.add_job(
            fn, trigger=trigger,
            id=fn.__name__, misfire_grace_time=300, coalesce=True,
            replace_existing=True,
        )
        log.info("job_registered", name=fn.__name__, trigger=str(trigger))

    # 7-agent pipeline. Registered only when the orchestrator built — without
    # an Anthropic key the system still runs the single-model path.
    orch = _try_build_agent_orchestrator(loop)
    set_orch(orch)  # also wires the registry for direct startup_catchup calls.
    if orch is not None:
        for job in AGENT_JOB_MANIFEST:
            params = dict(job)
            trigger_type = params.pop("trigger")
            fn = params.pop("func")

            if trigger_type == "cron":
                trigger = CronTrigger(timezone=ET, **params)
            elif trigger_type == "interval":
                trigger = IntervalTrigger(**params)
            else:
                raise ValueError(f"unknown trigger: {trigger_type}")

            # Multiple rows share the same function name (day_research_premarket
            # fires three times). APScheduler needs unique job ids.
            job_id = f"{fn.__name__}_{trigger}"
            scheduler.add_job(
                fn, trigger=trigger,
                id=job_id, misfire_grace_time=300, coalesce=True,
                replace_existing=True,
            )
            log.info("agent_job_registered", name=fn.__name__, trigger=str(trigger))

    # Expose the scheduler so execution-layer code (TWAP slicing) can
    # schedule follow-up slice fires without plumbing a reference through
    # every layer.
    from zeus.scheduler.context import set_scheduler
    set_scheduler(scheduler)
    return scheduler


def main() -> None:
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI

    _CT = _ZI("America/Chicago")

    def _ct_ts(_logger, _method, event_dict):
        event_dict["timestamp"] = _dt.now(_CT).isoformat()
        return event_dict

    structlog.configure(
        processors=[
            _ct_ts,
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )

    loop = build_trading_loop()
    loop.startup()

    # Emit a startup heartbeat BEFORE the (potentially slow) data refresh
    # so the 10-min staleness window resets the moment the process is
    # alive. The scheduled `heartbeat_job` doesn't fire until 5 minutes
    # after `scheduler.start()`, and `_refresh_if_stale` + `_startup_catchup`
    # can together take several minutes — without this, every restart
    # produces a false-positive `heartbeat_stale` CRITICAL risk_event.
    try:
        loop.emit_heartbeat()
    except Exception as e:
        log.warning("startup_heartbeat_failed", error=str(e))

    _refresh_if_stale(loop)

    scheduler = build_scheduler(loop)

    # Fire any overnight agent jobs we missed before the regular cadence
    # picks up. Runs synchronously so the morning session inherits fresh
    # briefs/memos + model + LLM plan even if the process booted mid-day.
    try:
        _startup_catchup(loop, getattr(loop, "_agent_orchestrator", None))
    except Exception as e:
        log.error("startup_catchup_failed", error=str(e))

    scheduler.start()
    log.info("scheduler_started")
    # Second startup heartbeat — confirms transition from "booting" to
    # "running scheduled work". If `_startup_catchup` blew through 5+
    # minutes, the first startup heartbeat above could already be stale
    # by the time we reach this point.
    try:
        loop.emit_heartbeat()
    except Exception as e:
        log.warning("post_start_heartbeat_failed", error=str(e))

    try:
        from zeus.scheduler.telemetry import attach_scheduler_telemetry
        attach_scheduler_telemetry(scheduler)
    except Exception as e:
        log.warning("scheduler_telemetry_attach_failed", error=str(e))

    settings = get_settings()
    cmd_handler = None
    if settings.telegram_bot_token and settings.telegram_chat_id and "PLACEHOLDER" not in settings.telegram_bot_token.upper():
        cmd_handler = TelegramCommandHandler(
            bot_token=settings.telegram_bot_token,
            allowed_chat_id=settings.telegram_chat_id,
            loop=loop,
            kill_switch=loop._kill,
        )
        cmd_handler.start()
    else:
        log.warning("telegram_command_handler_disabled_no_credentials")

    stop_flag = {"stop": False}

    def _shutdown(*_):
        log.info("shutdown_signal_received")
        stop_flag["stop"] = True
        scheduler.shutdown(wait=True)
        if cmd_handler is not None:
            cmd_handler.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while not stop_flag["stop"]:
        time.sleep(5)


if __name__ == "__main__":
    main()
