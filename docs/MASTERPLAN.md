# Zeus Trading Agent — Masterplan

How the system collects data, trains models, makes trade decisions, and
learns from outcomes. Every claim is cited to a file and line; follow the
links to verify.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [System topology](#2-system-topology)
3. [Data pipeline](#3-data-pipeline)
4. [Feature engineering](#4-feature-engineering)
5. [Model training](#5-model-training)
6. [Walk-forward validation and bootstrapped IC](#6-walk-forward-validation-and-bootstrapped-ic)
7. [Model registry and hot-reload](#7-model-registry-and-hot-reload)
8. [The 7-agent system](#8-the-7-agent-system)
9. [Daily decision timeline](#9-daily-decision-timeline)
10. [Trade decision flow](#10-trade-decision-flow)
11. [Portfolio construction and sizing](#11-portfolio-construction-and-sizing)
12. [Risk engine and circuit breakers](#12-risk-engine-and-circuit-breakers)
13. [Execution: order types, TWAP, reconciliation](#13-execution-order-types-twap-reconciliation)
14. [Strategy differences: day vs swing vs long-term](#14-strategy-differences)
15. [Learning loops](#15-learning-loops)
16. [Memory: the agent journal](#16-memory-the-agent-journal)
17. [Failure modes and safety nets](#17-failure-modes-and-safety-nets)

---

## 1. Executive summary

Zeus is a multi-agent algorithmic trading system. Three strategy
"families" (day, swing, long-term) share one paper Alpaca account; each
family has its own cross-horizon-gated model and its own LLM trader
agent, plus a paired LLM research agent that produces written briefs.
A seventh agent — the overseer — audits the others against per-agent
mandates, issues halts when traders fill without research backing, and
proposes capital reallocations weekly.

The decision pipeline is **layered**:

1. **Numerical layer.** A tree-based predictor (XGB+LGBM ensemble,
   cross-horizon-gated) emits one log-return forecast per `(symbol,
   horizon)`. Predictions clear a deterministic rank+magnitude gate
   plus a cross-horizon agreement gate before they become candidates.
2. **Portfolio layer.** A Kelly-fraction sizer with vol-targeting
   converts candidate scores into share counts subject to per-strategy
   exposure caps and a global multi-agent cap.
3. **Conviction layer.** An LLM trader agent reviews the model's
   proposed plan against its paired research briefs and a written
   conviction-gate policy. It cuts or shrinks proposals that fail the
   gate — never overrides upward.
4. **Risk layer.** A drawdown-tiered risk engine pre-checks every
   submitted order against sector/liquidity/notional caps and
   halt-state, then routes the survivor through a TWAP-sliced
   execution algorithm.
5. **Reconciliation + learning.** The reconciler emits realized-P&L
   rows on every close; postmarket wrap-up jobs feed those into the
   research agents as `lesson` and `postmortem` journal entries;
   nightly Sunday retraining rebuilds the models off the updated
   feature snapshot and the freshest realized P&L.

Everything that is auditable is logged into one of three tables —
`agent_journal` (reasoning), `trades` (realized P&L), `risk_events`
(safety actions) — so every fill traces back to a model prediction +
a research brief + a written rationale.

---

## 2. System topology

Four long-running services, defined in
[docker-compose.yml](../docker-compose.yml):

- **zeus-postgres** — TimescaleDB 15 (Postgres + hypertable extensions).
  Holds OHLCV daily/intraday, features, signals, orders, trades,
  positions, agent_journal, risk_events, heartbeats, model_versions,
  apscheduler_jobs, and the new backup target.
- **zeus-redis** — Caching layer for macro features and short-lived
  pipeline state.
- **zeus-scheduler** — The trading process. Runs APScheduler with
  ~28 jobs, the trading loop, the broker client, and the agent
  orchestrator.
- **zeus-monitor** — FastAPI dashboard at `:8000` reading the same DB
  for status panels.

A separate `zeus-backup` compose service runs
[scripts/backup_db.sh](../scripts/backup_db.sh) on demand for nightly
pg_dump (also invokable from inside the scheduler via the
`backup_db_job` at 03:00 daily — see §15).

The scheduler is the only mutator of broker state. The monitor is
read-only.

---

## 3. Data pipeline

### Universe

[zeus/data/ingestion/universe_builder.py:20-101](../zeus/data/ingestion/universe_builder.py#L20) —
S&P 500 ∪ NASDAQ 100 tickers from yfinance, normalized for Alpaca
(`BRK-B` → `BRK.B`), gives ~600 base symbols. A 20-day rolling
`avg_dollar_volume` and `avg_spread_bps` are computed nightly and the
filtered subset is snapshotted into `UniverseSnapshot`
([database.py:286-296](../zeus/data/storage/database.py#L286)) with
`passes_filter` so downstream code can join against today's eligible
universe.

### OHLCV — daily

[zeus/data/ingestion/ohlcv_ingester.py:39-94](../zeus/data/ingestion/ohlcv_ingester.py#L39) —
yfinance is the primary source for backfills (split + dividend
adjusted), Alpaca IEX is the fallback for incremental refreshes.
Stored in `OHLCVDaily` ([database.py:39-52](../zeus/data/storage/database.py#L39)),
a TimescaleDB hypertable on `ts`. Refresh fires twice on weekdays:
06:00 ET (overnight catch-up) and 16:30 ET (today's close), both via
[`data_refresh_job`](../zeus/scheduler/jobs.py#L92).

### OHLCV — intraday 5-minute

[zeus/data/ingestion/intraday_alpaca.py:44-156](../zeus/data/ingestion/intraday_alpaca.py#L44) —
Alpaca IEX feed, capped at **25 symbols** (open positions + SPY/QQQ
+ explicit extras). Pulled every 5 minutes by
[`intraday_backfill_job`](../zeus/scheduler/jobs.py#L208) during
market hours with a 120-minute lookback. Stored in `OHLCVIntraday`
([database.py:55-67](../zeus/data/storage/database.py#L55)). These
bars feed day-research and the position-stop checker; the daily
predictor does *not* consume them.

### Fundamentals

[zeus/features/fundamental.py:46-76](../zeus/features/fundamental.py#L46) —
yfinance `.info` dict, 15 fields with sanity clips
(`pe_ratio` max 500, `ev_ebitda` max 200, etc.). Persisted in
`FundamentalsCache` ([database.py:251-274](../zeus/data/storage/database.py#L251)).

### Macro

[zeus/features/macro.py:93-201](../zeus/features/macro.py#L93) — 24-hour
Redis cache, FRED as primary (VIXCLS, DGS10, DGS2, FEDFUNDS), yfinance
as fallback (`^VIX`, `^TNX`, `^IRX`, SPY). Returns ten fields:

| Field | Source | Notes |
|---|---|---|
| `vix_level`, `vix_5d_change` | FRED VIXCLS / `^VIX` | fear gauge |
| `yield_10y`, `yield_2y`, `yield_curve_spread` | FRED DGS10/DGS2 / `^TNX`/`^IRX` | inversion indicator |
| `fed_funds_rate` | FRED FEDFUNDS | constant within FOMC window |
| `spy_return_5d`, `spy_return_20d` | yfinance SPY | market drift |
| `put_call_ratio`, `put_call_ratio_5d_change` | stooq.com `^cpc` (primary) / `^CPC` yfinance (fallback) | alt-data sentiment |

The put/call ratio is the alt-data signal added per the review;
[zeus/features/macro.py:_fetch_put_call_series](../zeus/features/macro.py#L36)
handles both sources with graceful degradation.

### News

[zeus/execution/alpaca_broker.py:207-234](../zeus/execution/alpaca_broker.py#L207) —
Alpaca news API, used by research agents for premarket scans. Not
persisted; queried fresh per call.

---

## 4. Feature engineering

Pipeline orchestrator:
[zeus/features/pipeline.py:30-125](../zeus/features/pipeline.py#L30).
Runs nightly at 17:00 ET via
[`feature_engineering_job`](../zeus/scheduler/jobs.py#L114) and
writes the full panel to both a parquet snapshot under
`artifacts/features/YYYY-MM-DD/` and the `FeaturesDaily` table
([database.py:70-82](../zeus/data/storage/database.py#L70)) keyed by
`(symbol, feature_date, feature_version)`. Feature version is `v1`
today; bump it when the schema changes so old models keep working.

### Technical features (~30)

[zeus/features/technical.py:15-128](../zeus/features/technical.py#L15) —
Computed per symbol from the 250-day OHLCV window:

- **Moving averages**: `sma_5`, `sma_20`, `sma_50`, `sma_200`,
  `ema_12`, `ema_26` + price-vs-MA % deviations
  (`price_vs_sma20_pct`, etc.).
- **Momentum / RSI**: `rsi_14`, `rsi_5`, `momentum_5d`,
  `momentum_10d`, `momentum_20d`, `momentum_60d`, `roc_5`, `roc_20`.
- **MACD**: `macd`, `macd_signal`, `macd_hist` (fast=12, slow=26,
  signal=9).
- **ADX**: `adx_14`, `di_plus`, `di_minus`.
- **Volatility**: `atr_14_pct` (ATR normalized to % of close),
  `bb_width_20` (Bollinger band width, length 20, std 2),
  `hist_vol_20d`, `hist_vol_5d` (log-return rolling vol annualized
  √252), `vol_ratio_5_20`.
- **Volume**: `volume_ratio_5d` (5d SMA / 20d SMA),
  `obv_slope_5d` (OBV diff, normalized), `vwap_deviation`.

Everything goes through `pandas-ta`. Warmup periods leave the early
rows NaN; the loader downstream drops them rather than imputing — no
look-ahead bias is introduced.

### Fundamental features (16)

PE, PB, PS, EV/EBITDA, earnings yield, revenue growth YoY, earnings
growth YoY, gross margin, operating margin, ROE, ROA, debt-to-equity,
current ratio, short interest ratio, log10 market cap, plus the
sector label. See
[fundamental.py:46-76](../zeus/features/fundamental.py#L46).

### Cross-sectional features

[zeus/features/cross_sectional.py:24-99](../zeus/features/cross_sectional.py#L24) —
Percentile ranks computed across the universe daily:
`return_5d_rank_universe`, `return_20d_rank_universe`,
`vol_20d_rank_universe`, `rsi_rank_universe`,
`momentum_rank_universe`, `dollar_volume_rank_universe`. Plus
sector-scoped ranks (`return_*_rank_sector`) and relative strength
vs SPY and vs sector median (`rs_vs_spy_20d`, `rs_vs_sector_20d`).

### Macro × symbol interactions

[zeus/features/interactions.py:37-96](../zeus/features/interactions.py#L37) —
The fix for "macro is broadcast as constants" identified in the audit.
Five base features (`momentum_20d`, `momentum_5d`, `rsi_14`,
`atr_14_pct`, `hist_vol_20d`) crossed with four macro features
(`vix_level`, `yield_curve_spread`, `spy_return_5d`, `put_call_ratio`)
produces 20 new columns of the form `{base}_x_{macro}`. The
multiplication makes the macro signal *cross-sectionally informative*
again: high-VIX days now read differently across high-momentum vs
low-momentum names, giving the trees actual splits to find.

### Macro

The ten macro fields from §3 are broadcast across the daily panel as
constants. Their *interaction* terms are where their signal lives;
the raw broadcasts are kept for backward compatibility with older
trained models.

---

## 5. Model training

### Model classes (the menagerie)

Each class lives under [zeus/models/](../zeus/models/) and exposes a
common `BaseModel` interface (`.fit`, `.predict`, `.save`, `.load`):

| Class | Purpose |
|---|---|
| [`XGBReturnPredictor`](../zeus/models/return_predictor.py#L19) | Single-model XGBoost regressor on forward log returns. Defaults: `n_estimators=500, max_depth=5, learning_rate=0.01, early_stopping_rounds=50`. |
| [`LGBMReturnPredictor`](../zeus/models/lgbm_return_predictor.py#L26) | LightGBM counterpart, used to diversify ensemble. |
| [`EnsembleReturnPredictor`](../zeus/models/ensemble_return_predictor.py#L23) | Average (or weighted average) of multiple base predictors. Empirically picks up 1–3 pp of IC vs single model. |
| [`MetaLabeler`](../zeus/models/meta_labeler.py#L33) | LGBM classifier predicting "will the primary be directionally right?" Trained on `(features, primary_pred) → win_label`. |
| [`MetaGatedPredictor`](../zeus/models/meta_labeler.py#L170) | Primary + meta: zeros out predictions where meta win-prob falls below threshold. Lifts hit rate from ~0.50 baseline toward 0.55–0.60. |
| [`RankMagnitudeGatedPredictor`](../zeus/models/rank_gated_predictor.py#L31) | Deterministic post-prediction gate: keep only when `abs(pred) ≥ mag_floor` AND `pred_rank_pct ≥ rank_floor` (default 0.95 — top 5%). `long_only=True` zeros shorts. |
| [`CrossHorizonGatedPredictor`](../zeus/models/rank_gated_predictor.py#L136) | Adds an *agreement* gate on top of the rank-magnitude gate: a target-horizon prediction survives only if every anchor primary (other horizons) agrees on direction. This is the model class actually deployed for all three strategies. |

### Horizons and per-strategy mapping

From [config/strategies.yaml](../config/strategies.yaml):

| Strategy | Model name | Horizon (days) |
|---|---|---|
| day | `cross_horizon_day_h3` | 3 |
| swing | `cross_horizon_swing_h5` | 5 |
| long_term | `cross_horizon_long_term_h20` | 20 |

Each strategy's model is independently trained on the same feature
panel but a different forward-return label (`r_h3`, `r_h5`, `r_h20`).
The anchor predictors used in the cross-horizon gate are the *other
two* horizons' primaries — so the long-term model is only allowed to
fire when the day and swing predictors agree on the sign.

### Training scripts

Living under [`scripts/`](../scripts/):

- [`train_multi_horizon_pnl.py`](../scripts/train_multi_horizon_pnl.py) —
  Trains XGB / LGBM / ensemble primaries across h3/h5/h10/h20. Outputs
  the OOS prediction tables that train_rank_gate.py and the
  cross-horizon trainer consume.
- [`train_rank_gate.py`](../scripts/train_rank_gate.py) — Grid-searches
  `(mag_floor, rank_floor)` against `TARGET_HIT_RATE = 0.60` on a
  70/30 OOS split, retrains the primary on the full sample, and
  registers a new `model_versions` row if hit rate clears 0.60.
- [`train_cross_horizon_bundle.py`](../scripts/train_cross_horizon_bundle.py) —
  Builds the deployed `CrossHorizonGatedPredictor` per strategy.
  Calibrates the gate via `calibrate_cross_horizon_gate`, maximizing
  out-of-sample PnL subject to `train_hit ≥ 0.60 AND eval_hit ≥ 0.60`.
  Registers a row only if both hit-rate AND PnL targets clear.
- [`train_meta.py`](../scripts/train_meta.py) — Trains the
  `MetaLabeler` classifier on the primary's OOS predictions.
- [`train_sweep.py`](../scripts/train_sweep.py) — Hyperparameter sweep.
- [`bootstrap_train.py`](../scripts/bootstrap_train.py) — Day-zero
  cold-start trainer.

### Nightly retrain

[`nightly_retrain_job`](../zeus/scheduler/jobs.py#L141) fires Sunday
02:00 ET. It delegates to
`zeus.research.after_hours.run_nightly_retrain`, which re-pulls 18
months of features, re-fits the per-horizon primaries, re-runs the
cross-horizon calibration, and writes new `ModelVersion` rows. The new
rows land as `status='staging'` (or `'failed'` if gates don't pass)
and the 30-second `reload_models_job` (§7) picks them up the next time
it polls. No restart is needed; the morning session uses the new
model.

---

## 6. Walk-forward validation and bootstrapped IC

### Walk-forward

[zeus/backtesting/walk_forward.py:24-139](../zeus/backtesting/walk_forward.py#L24) —
This is what real backtest discipline looks like:

- 18-month rolling train window.
- 3-month test window.
- 1-month step (so folds overlap, giving more measurements per
  parameter set).
- **20 trading days of embargo** between train_end and test_start
  ([line 21](../zeus/backtesting/walk_forward.py#L21)). This prevents
  the last few rows of training data from leaking forward-return
  information into the test window via target overlap — a common
  silent leak in 1-day, 5-day, 20-day label schemes.

Per fold, the validator computes IC, hit rate, and the Sharpe of an
equal-weight long-top-quintile signal portfolio.

### Bootstrapped IC lower bound

The audit identified that the gates were running on a point-estimate
IC. The fix:
[zeus/backtesting/metrics.py:bootstrapped_ic](../zeus/backtesting/metrics.py#L114)
resamples with replacement (1000 draws, seed=42) and reports the
**5th-percentile lower bound** alongside the mean and std. The
walk-forward output now writes `ic_lcb`, `ic_mean`, `ic_std`, and the
point `ic` per fold ([walk_forward.py:121-138](../zeus/backtesting/walk_forward.py#L121)).

When the group key is provided (typical: per-date), the resampling
happens at the *group* level — not the row level — because the right
unit of variation for cross-sectional rank forecasts is the *day*, not
the individual stock. Resampling rows would understate variance.

The point of the LCB is the gate criterion: a model whose IC point
estimate clears 0.05 but whose 5th-percentile LCB sits at -0.01 is
mostly noise; the LCB filters that case out. A separate trainer-side
change to gate on LCB rather than IC point is staged but not yet
enforced (see §17).

---

## 7. Model registry and hot-reload

### Registry schema

[`ModelVersion`](../zeus/data/storage/database.py#L191) holds one row
per trained-and-saved model bundle. Key columns:

- `model_name` — e.g. `cross_horizon_day_h3`.
- `version` — e.g. `v20260421_012744` (timestamp string).
- `status` — `staging`, `production`, `retired`, or `failed`.
- `metrics` (JSON) — IC, hit rate, PnL, Sharpe, etc.
- `config` (JSON) — hyperparameters and the `__model_class__`
  import-string hint.
- `artifact_path` — directory or `.joblib` file path.
- `feature_version` — pinned so a retrained model can't accidentally
  load against a v1 panel when it was trained on v2.

### Class resolution

The loader at
[`_resolve_model_class`](../zeus/scheduler/runner.py#L199) checks three
sources in priority order:

1. `row.config.__model_class__` (newest convention).
2. `model_class.txt` sidecar alongside the artifact (older
   convention).
3. **Payload sniffing** — peek at the joblib's top-level keys and
   match against the marker table at
   [`_PAYLOAD_KEY_TO_MODEL_PATH`](../zeus/scheduler/runner.py#L166):
   - `primary_cls` → `CrossHorizonGatedPredictor`
   - `rank_floor` → `RankMagnitudeGatedPredictor`
   - `meta_clf` → `MetaGatedPredictor`
   - `members` → `EnsembleReturnPredictor`
   - `params` → `XGBReturnPredictor`

The sniffing fix lets the system recover even when an old bundle was
registered without any class hint at all — which is exactly the case
that used to crash `_load_latest_model` with `KeyError: 'params'`.

### Hot-reload

[`reload_models_if_promoted`](../zeus/scheduler/runner.py#L307) fires
every 30 seconds via
[`reload_models_job`](../zeus/scheduler/jobs.py#L376). For each
strategy context, it queries `model_versions` for the newest
staging/production row matching that strategy's `model_name`. If the
loaded row's `version_` differs from the in-memory model, it atomically
swaps both `ctx.model` and the cached reference inside
`ctx.signal_generator._model`. Assignments are GIL-safe, so an
in-flight `predict()` call completes against whichever object it had
already dereferenced.

Net effect: when nightly retraining promotes a new model at Sunday
02:30 UTC, the running scheduler picks it up within 30 seconds — no
restart, no missed market days waiting for the next deploy window.

---

## 8. The 7-agent system

Each agent is an `AgentLoop`
([zeus/agents/base.py:76-180](../zeus/agents/base.py#L76)) bound to an
LLM tier, a system prompt, and a write/read scope on the agent
journal. They are tiered by cost vs reasoning depth in
[`build_agent_bundle`](../zeus/agents/runtime.py#L49):

| Agent | Class | Role | LLM tier | Prompt file |
|---|---|---|---|---|
| `day_research` | [DayResearchAgent](../zeus/agents/day_research.py#L32) | research | haiku | [day_research.md](../zeus/llm/prompts/day_research.md) |
| `swing_research` | [SwingResearchAgent](../zeus/agents/swing_research.py#L24) | research | sonnet | [swing_research.md](../zeus/llm/prompts/swing_research.md) |
| `long_term_research` | [LongTermResearchAgent](../zeus/agents/long_term_research.py#L30) | research | opus | [long_term_research.md](../zeus/llm/prompts/long_term_research.md) |
| `day` | [DayTraderAgent](../zeus/agents/day_trader.py#L7) | trader | sonnet | [day_trader.md](../zeus/llm/prompts/day_trader.md) |
| `swing` | [SwingTraderAgent](../zeus/agents/swing_trader.py#L7) | trader | sonnet | [swing_trader.md](../zeus/llm/prompts/swing_trader.md) |
| `long_term` | [LongTermTraderAgent](../zeus/agents/long_term_trader.py#L7) | trader | opus | [long_term_trader.md](../zeus/llm/prompts/long_term_trader.md) |
| `overseer` | [OverseerAgent](../zeus/agents/overseer.py#L74) | overseer | opus | [overseer.md](../zeus/llm/prompts/overseer.md) |

Plus a pseudo-agent `research_library` ([`LIBRARY_AGENT`](../zeus/agents/journal.py#L39))
that hosts curated paper excerpts from `Trading Strategy Research/`,
read-only from every other agent's perspective.

Each trader is *paired* with one research agent
([`PAIRED_RESEARCH`](../zeus/agents/journal.py#L61)). The overseer
audits this pairing: every fill from `day` must have at least one
recent `brief` or `memo` from `day_research`, or the overseer halts
`day` for a decoupling violation.

System prompts are loaded by
[`load_prompt`](../zeus/llm/prompts/__init__.py#L20) with `@lru_cache`,
and the principles preamble is rendered with
`cache_control: {"type": "ephemeral"}` on Anthropic's side
([zeus/llm/client.py:298-305](../zeus/llm/client.py#L298)) so the
shared text doesn't burn input tokens on every call.

---

## 9. Daily decision timeline

All times in `America/New_York`. Cron expressions live in
[`JOB_MANIFEST`](../zeus/scheduler/jobs.py#L378) and
[`AGENT_JOB_MANIFEST`](../zeus/scheduler/agent_jobs.py#L406). Jobs
are persisted in `apscheduler_jobs` (SQLAlchemyJobStore), so crash
restarts don't lose their schedules.

### Evening (previous trading day)

| Time | Job | What |
|---|---|---|
| 19:00 | [swing_research_eod_job](../zeus/scheduler/agent_jobs.py#L130) | Swing-research agent writes EOD briefs for tomorrow's swing setups. |
| 19:15 | [long_term_deep_dive_or_review_job](../zeus/scheduler/agent_jobs.py#L146) | Long-term research either deep-dives a new candidate or refreshes the thesis on an open position. |
| 20:00 | [overseer_daily_aggregate_job](../zeus/scheduler/agent_jobs.py#L285) | Overseer pulls today's trades + journal, computes per-agent metrics, issues halts if needed. Deterministic (no LLM). |
| 21:00 | [next_session_planning_job](../zeus/scheduler/jobs.py#L165) | Model-based signal generation: features → predictions → rank gate → cross-horizon gate → portfolio construction → SessionPlan. |
| 21:30 | [trader_overnight_plan_job](../zeus/scheduler/agent_jobs.py#L245) | Swing + long-term trader LLMs review the model plan against research briefs, emit final entry/exit lists, attach to strategy context. |
| 21:30 | [company_research_job](../zeus/scheduler/jobs.py#L187) | Fundamentals + news refresh for next-day candidates. |
| 03:00 | [backup_db_job](../zeus/scheduler/jobs.py#L299) | `pg_dump` to `./data/backups/`, optional off-machine upload via `BACKUP_REMOTE_CMD`. |

### Morning (current trading day)

| Time | Job | What |
|---|---|---|
| 05:00, 07:00, 09:00 | [day_research_premarket_job](../zeus/scheduler/agent_jobs.py#L306) | Day-research scans premarket movers, writes briefs. Three shots so 09:00 sees post-news premarket action. |
| 08:00 | [premarket_summary_job](../zeus/scheduler/jobs.py#L33) | Telegram summary push. |
| 09:10 | [day_trader_morning_plan_job](../zeus/scheduler/agent_jobs.py#L322) | Day trader LLM finalizes day plan. |
| 09:30 | [market_open_job](../zeus/scheduler/jobs.py#L45) | **Freshness-gated**: aborts if anchor OHLCV is stale (§17). Otherwise runs `loop.run_manual_trim()` then `loop.execute_entries()`, which TWAP-slices entries (§13). |

### Intraday (09:30–15:55)

Concurrent crons in market hours:

- **every 5 min** — `intraday_backfill_job` (5-min bars), `day_research_intraday_refresh_job` (LLM re-rank against open candidates), `heartbeat_job` (writes `Heartbeat` row with PV + drawdown).
- **every 30s** — `reload_models_job` (hot-swap if newer model promoted).
- **every 1 min** — `stale_heartbeat_check_job` (writes a CRITICAL `risk_event` if `trading_loop` heartbeat is ≥10 min stale).
- **every 15 min** — `intraday_monitor_job`. This is the consolidated risk-scan job. It runs `loop.check_stops_and_risk()` (refresh open-order state, evaluate hard stops + trailing stops + per-name P&L, trip risk levels), then folds in `orch.bundle.overseer.run_realtime_monitor()` (decoupled-fill detection, halt issuance for agents whose trades are missing paired research).

### Close

| Time | Job | What |
|---|---|---|
| 15:30 | [closing_job](../zeus/scheduler/jobs.py#L101) | Force-exit positions hitting their per-strategy `max_hold_days`. |
| 16:15 | [eod_report_job](../zeus/scheduler/jobs.py#L112) | Daily P&L + Telegram summary. |
| 16:15 | [day_research_postmarket_wrap_job](../zeus/scheduler/agent_jobs.py#L368) | Reads today's realized `Trade` rows, feeds them back to day-research as lesson material. |
| 16:30 | [data_refresh_job](../zeus/scheduler/jobs.py#L123) | OHLCV refresh + freshness self-check. |
| 17:00 | [feature_engineering_job](../zeus/scheduler/jobs.py#L148) | Feature pipeline runs (with freshness gate). |

### Weekly / monthly

- **Sun 02:00** — `nightly_retrain_job` (§5).
- **Sun 10:00** — `swing_weekly_watchlist_job`.
- **Sun 11:00** — `overseer_weekly_review_job` (capital reallocation proposals).
- **First Sat of month** — `long_term_monthly_review_job` (re-examine every long-term holding against its original memo).

---

## 10. Trade decision flow

This is the chain from prediction to fill, for one symbol in one
strategy.

### Step 1 — Numerical prediction

The 21:00 ET planning job runs the feature pipeline against today's
OHLCV, joins fundamentals, macro, cross-sectional ranks, and the new
macro×feature interactions. The result is one feature row per
eligible symbol.

The strategy's `CrossHorizonGatedPredictor.predict()` then runs in
three layers:

1. Primary predicts the horizon-h forward log return.
2. **Rank-magnitude gate**: zero out any row that isn't in the top
   5% by predicted magnitude (`rank_floor=0.95`) and whose absolute
   prediction is below `mag_floor`. With `long_only=True`, also zero
   out negative predictions.
3. **Cross-horizon gate**: for each surviving row, check the anchor
   models (the other horizons' primaries). If any anchor predicts
   the opposite sign, the row is zeroed.

Only rows that survive all three gates carry through as candidate
signals.

### Step 2 — Portfolio construction

[`PortfolioConstructor.construct()`](../zeus/portfolio/constructor.py#L24)
takes the candidate signals plus the current positions plus the
strategy's allocated portfolio_value/buying_power, and returns a
`SessionPlan`:

- For each candidate, compute shares via
  [`kelly_position_size()`](../zeus/risk/position_sizer.py) using the
  predicted return, the symbol's annualized vol estimate, confidence,
  and a Kelly fraction (default 0.25). Bound by `max_position_pct`
  (strategy-scoped, see table in §14) and a liquidity cap of 5% of
  20-day ADV.
- Aggregate sector exposure, reject if any sector exceeds
  `max_sector_pct = 0.35`.
- Aggregate total exposure, reject if it exceeds the strategy's
  `max_exposure_pct`.

The result is `SessionPlan.entries`, `SessionPlan.exits`,
`SessionPlan.holds`.

### Step 3 — Conviction gate (LLM)

The strategy's trader agent (e.g. `DayTraderAgent`) runs at 09:10
(or 21:30 for swing/LT). It loads:

- The model's `SessionPlan` from its `StrategyContext.current_plan`.
- Recent paired-research briefs (24h memory window for day, 72h for
  swing, 336h = 14 days for long_term — see
  [`TraderAgentBase._memory_window_hours`](../zeus/agents/trader_base.py#L79)).
- The system prompt's *conviction-gate policy*.

From [`day_trader.md`](../zeus/llm/prompts/day_trader.md#L24), every
proposed entry must clear ONE of:

- **Path A (model-led):** Score in the top decile + at least one
  corroborating observation (technical setup, regime, options flow,
  catalyst, sector momentum).
- **Path B (research-led):** A brief/alert from the paired research
  agent within the last 24h with `confidence ≥ 3`, plus at least
  one supportive observation.

Both paths additionally require citing at least one trading
principle and at least one paper from `research_library`.

The trader emits each surviving proposal via a `propose_trade` tool
call. `AgentLoop` collects these
([base.py:124-180](../zeus/agents/base.py#L124)) and records each one
as a `trade_rationale` row in `agent_journal`. The LLM does *not*
change share counts upward — it can shrink or veto, never lever up.
A zero-entries session is itself logged as a low-severity
`risk_event` so the overseer can audit whether the trader is being
appropriately selective vs simply broken.

### Step 4 — Risk pre-check

[`RiskEngine.pre_trade_check`](../zeus/risk/engine.py#L132) gates
every order, checking in this order:

- Kill switch active? (§12) → reject.
- Risk engine paused? → reject.
- Current drawdown level stops new entries? → reject.
- Today's loss-cap state stops new entries? → reject.
- Notional in `[min_position_notional=$2k, max_single_order_notional=$25k]`?
- `max_position_pct` not exceeded for this symbol across all
  strategies?
- Sector cap (`max_sector_pct=0.35`) not exceeded?
- Liquidity check: `notional ≤ 0.05 × avg_dollar_volume_20d`?
- Buying-power buffer left (`broker_buying_power_buffer=0.05`)?

A reject can either kill the order outright or come back with
`adjusted_shares` so the executor can submit a smaller line; the
trader path uses the latter when only the size is the issue.

### Step 5 — Execution

[`schedule_twap_entry`](../zeus/execution/twap.py#L165) splits the
approved share count into 5 slices over 12 minutes (configurable),
fires slice 1 inline, and schedules slices 2–5 as one-shot
DateTrigger jobs via the live scheduler. Each slice re-quotes the
spread and re-picks market vs limit via
[`plan_entry`](../zeus/execution/execution_algo.py#L26):

- spread ≤ 5 bps → market.
- spread > 30 bps → limit at mid (patient).
- otherwise → limit at ask + 0.1% (normal urgency, expects to cross).

The slicing only kicks in for orders ≥
[`MIN_SHARES_FOR_TWAP=50`](../zeus/execution/twap.py#L48); smaller
orders fire as a single fill.

### Step 6 — Reconciliation + journal

At every intraday monitor tick,
[`PositionReconciler.reconcile`](../zeus/execution/reconciler.py#L59)
pulls the broker's authoritative position snapshot, upserts local
rows, rescales `strategy_shares` proportionally on drift, and (the
new behavior) emits a `Trade` row when the broker has closed a
position the local DB still tracked. The reconciler also looks back
48h in the broker's filled-order history to attribute that close's
exit price and timestamp accurately.

The fill itself creates a chain: `Order` → `Trade` → and an
`agent_journal` entry of kind `trade_rationale` written by the
trader at proposal time. Every fill is therefore traceable to a
written reason; the overseer enforces this by *halting* any agent
that fills without a matching rationale.

---

## 11. Portfolio construction and sizing

### Sizing

[`kelly_position_size`](../zeus/risk/position_sizer.py) does a
fractional-Kelly bet on each candidate:

- Expected return: model's predicted forward log return for the
  strategy's horizon.
- Volatility: annualized historical vol estimate.
- Confidence: scales the bet down on low-confidence rows.
- Kelly fraction: 0.25 by default — quarter-Kelly is a standard
  defensive choice when expected returns and vol are estimated
  rather than known.
- Liquidity cap: position notional bounded to 5% of the symbol's
  20-day average dollar volume.

The output is `target_pct` of portfolio NAV; multiplied by
`portfolio_value/price` to get share count, then floored to an
integer (Alpaca rejects fractional equity orders).

### Per-strategy budgets

[`StrategyAllocator.effective_budgets`](../zeus/risk/strategy_allocator.py#L90)
slices the shared portfolio by `weight` and applies each strategy's
own `max_position_pct` / `max_exposure_pct` from
[`config/strategies.yaml`](../config/strategies.yaml):

| Strategy | weight | max_positions | max_position_pct | max_exposure_pct |
|---|---|---|---|---|
| day | 0.30 | 8 | 8% | 45% |
| swing | 0.40 | 10 | 10% | 50% |
| long_term | 0.30 | 8 | 12% | 40% |

A global cap of `max_total_exposure_pct = 95%` across all three
agents and a per-symbol cap of `global_max_position_pct = 18%`
(when two agents want the same name) prevent any single name or any
total exposure from running away.

### Sector concentration

`PortfolioConstructor` accumulates per-sector notional and rejects
candidates that would push the strategy above
`max_sector_pct = 0.35`. The sector label comes from the
`FundamentalsCache` row.

---

## 12. Risk engine and circuit breakers

[`zeus/risk/`](../zeus/risk/) houses three coordinated guards.

### Drawdown tiers

[`DrawdownGuard`](../zeus/risk/drawdown_guard.py#L37) tracks NAV
against its rolling peak and steps through five levels (USD
thresholds from
[`RiskLimits`](../zeus/risk/limits.py#L17)):

| Level | Trigger | Effect |
|---|---|---|
| NORMAL (0) | — | No constraints. |
| WARNING (1) | -$10k | Tighter sizing: max 12 positions, `max_position_pct=0.12`. |
| DEGRADED (2) | -$20k | Aggressive de-risk: max 6 positions, `max_position_pct=0.08`. |
| PRESERVATION (3) | -$35k | Stop new entries entirely, drain to max 2 positions, `max_position_pct=0.05`. |
| KILL (4) | -$50k | Activate kill switch, close all positions immediately. |

### Daily loss cap

[`DailyLossMonitor`](../zeus/risk/daily_loss.py#L26) — independent
of trailing drawdown, this guards intraday:

- WARN at -$3k (logged).
- STOP_NEW at -$5k (no new entries, existing positions run).
- KILL at -$7.5k (activate kill switch + close all).

### Kill switch

[`KillSwitch`](../zeus/risk/kill_switch.py) — a single boolean
`kill_switch_active` that, when set, makes
[`pre_trade_check`](../zeus/risk/engine.py#L146) reject every new
entry. Triggered by drawdown KILL, daily-loss KILL, or manual
activation. Closes all open positions via
[`AlpacaBroker.close_all_positions`](../zeus/execution/alpaca_broker.py#L142).

### Per-agent halts

[`RiskEngine.halt_agent`](../zeus/risk/engine.py#L111) — scoped halts.
The overseer triggers these when:

- Per-agent rolling drawdown exceeds
  `MAX_DD_HALT_PCT = 0.15` (15%).
- Mandate violations exceed `MAX_MANDATE_VIOLATIONS_HALT = 3`
  (e.g. a day-strategy fill held > 5 days, or a swing fill > 15
  days, or a long-term fill > 180 days).
- A trader files within
  `RESEARCH_COUPLING_WINDOW_HOURS = 24` without a matching brief/memo
  from the paired research agent.

A halted agent can't submit new entries but its existing positions
still get stop and time-exit treatment from the trading loop.

### Stop-loss management

[`compute_initial_stops`](../zeus/risk/stop_logic.py#L18) sets
`hard_stop = max(entry - 2.0×ATR, entry × (1 - 1.5×daily_vol),
entry × (1 + expected_downside × 1.5))`, never tighter than -3%.
Trailing stop activates at entry × 1.02 and tracks at 50% of the
gain above the activation point.

---

## 13. Execution: order types, TWAP, reconciliation

### Order routing

[`plan_entry`](../zeus/execution/execution_algo.py#L26) picks order
type based on spread + urgency:

| Condition | Order type | Why |
|---|---|---|
| urgency=high | market | Stop triggered or risk-driven, never miss. |
| spread ≤ 5 bps | market | Cost-free crossing. |
| spread > 30 bps (`max_spread_bps`) | limit at mid | Patient — wide spread means we'd pay both legs. |
| urgency=low | limit at mid | Same logic, slower. |
| else (normal) | limit at ask + 0.1% | Crosses the spread by a small margin to encourage fill but not full market price. |

Exits routed through
[`plan_exit_signal`](../zeus/execution/execution_algo.py#L60) (limit
at bid — patient) for normal exits, and
[`plan_exit_stop`](../zeus/execution/execution_algo.py#L67) (market)
for stop triggers.

### TWAP slicing

[`schedule_twap_entry`](../zeus/execution/twap.py#L165) — the
opening-spread saver:

- Total shares split into `DEFAULT_NUM_SLICES = 5` evenly-spaced
  slices over `DEFAULT_WINDOW_MINUTES = 12`.
- Any remainder rides on slice 1 (so the bulk lands quickly, before
  the spread widens further).
- Slice 1 fires inline; slices 2–5 are scheduled as one-shot
  `DateTrigger` jobs in the same persistent
  `SQLAlchemyJobStore` — they survive a scheduler restart.
- Each slice re-quotes the spread and re-picks market vs limit, so
  later slices adapt to the actual book at that moment.

Orders smaller than `MIN_SHARES_FOR_TWAP = 50` fall through to a
single fill — splitting 25 shares five ways isn't worth the
operational complexity.

### Reconciliation

[`PositionReconciler.reconcile`](../zeus/execution/reconciler.py#L59) —
runs at every intraday monitor tick and at process startup. Pulls
Alpaca's positions, upserts local `Position` rows by
`(symbol, strategy_id)`, and:

- If aggregate broker qty disagrees with the sum of our per-strategy
  `strategy_shares`, rescales proportionally to preserve relative
  ownership and emits a `strategy_share_drift` log event.
- If the broker has closed a symbol the local DB still tracks (a
  broker-side close — liquidation, manual close, corporate action),
  emits a `Trade` row with `exit_reason='broker_closed'`. To attribute
  the exit accurately, it looks back 48h in Alpaca's filled-order
  history for the matching sell fill; if not found, falls back to
  the cached `current_price` and `now`.

The `trades_emitted` field in the reconcile result is what feeds the
realized-P&L journal — every close, however it happened, now lands
in `trades`. Before the fix, broker-side closes silently dropped P&L
into the void.

---

## 14. Strategy differences

| Param | day | swing | long_term |
|---|---|---|---|
| Model | `cross_horizon_day_h3` | `cross_horizon_swing_h5` | `cross_horizon_long_term_h20` |
| Horizon (days) | 3 | 5 | 20 |
| Max hold (days) | 5 | 10 | 60 |
| Capital weight | 0.30 | 0.40 | 0.30 |
| Max positions | 8 | 10 | 8 |
| `max_position_pct` | 8% | 10% | 12% |
| `max_exposure_pct` | 45% | 50% | 40% |
| Trader LLM | sonnet | sonnet | opus |
| Research LLM | haiku | sonnet | opus |
| Memory window | 12 h | 72 h | 336 h (14 d) |

The day strategy is *narrow* and *fast*: paired with the cheapest
LLM tier (haiku for research), the tightest position size cap, the
shortest hold horizon. The long-term strategy is *deep* and *slow*:
opus for both research and trader, longest memory, deepest analyses
(`run_deep_dive`, `run_monthly_review`).

The mandate breaches that get an agent halted are scoped accordingly
— a day-strategy position held over 5 days counts as a mandate
violation; a long-term position held under 180 days is fine.

---

## 15. Learning loops

### Closed-loop realized-P&L (Task 1 fix)

Every position close — whether trader-initiated, stop-triggered,
broker-initiated, or time-exit-forced — now lands in the `trades`
table with `entry_price`, `exit_price`, `net_pnl`, `hold_days`,
`exit_reason`, `peak_price`, and (when attributable)
`alpaca_order_id`. This is what makes everything below possible —
without realized P&L, attribution is impossible.

### Postmarket research wrap

[`day_research_postmarket_wrap_job`](../zeus/scheduler/agent_jobs.py#L368)
reads today's closed trades, joins them against the research briefs
that preceded them, and feeds the pair into the day-research agent.
The agent writes `postmortem` journal entries explaining what
happened versus the brief's hypothesis. Pattern-recognition is the
intended payoff: if a hypothesis class consistently loses, the
agent's future briefs should stop citing it.

### Overseer daily aggregate

[`OverseerAgent.run_daily_aggregate`](../zeus/agents/overseer.py#L104)
(deterministic — no LLM) computes per-agent rolling metrics and
emits halts for the breaches in §12. Writes a `decision` journal
entry with the per-agent metrics structure as the audit trail.

### Overseer weekly review

[`OverseerAgent.run_weekly_review`](../zeus/agents/overseer.py#L138) —
opus-tier LLM call on Sundays. Reads the past week's trades,
postmortems, alerts. May propose:

- Reallocations: shift `weight` between strategies based on rolling
  Sharpe.
- Lessons: write `lesson` rows that future trader prompts will
  include.
- Halts or resumes for specific agents.

Reallocations go through
[`OverseerStrategyAllocator`](../zeus/risk/strategy_allocator.py#L130)
with safety bounds — proposals out of `[0.05, 0.60]` weight per
strategy are rejected as out-of-bounds.

### Long-term monthly review

[`LongTermResearchAgent.run_monthly_review`](../zeus/agents/long_term_research.py#L72) —
re-examines every open long-term holding against its original memo's
bull / base / bear targets. Writes a `review` row per position
either ratifying the thesis or marking it `thesis_invalidated`, which
the long-term trader uses as an exit trigger.

### Nightly retrain

The model itself learns once a week (§5). The interaction is:

1. Sunday 02:00: retrain runs on the freshest features.
2. Sunday 02:30 (typical): new `ModelVersion` rows lands with
   `status='staging'`.
3. `reload_models_job` (every 30s) sees the new version and
   hot-swaps in.
4. Monday's session uses the freshly trained model.

If the retrain doesn't clear gates (hit rate < 0.60 or PnL not
positive), the row is registered with `status='failed'` and the old
model continues to run. **The system never blindly upgrades to a
worse model** — but it also doesn't auto-roll-forward beyond
training-time validation, so a model that trained well but
out-of-sample collapses in week 1 is still trusted until the
following Sunday. This is one of the known weaknesses; see §17.

---

## 16. Memory: the agent journal

[`AgentJournal`](../zeus/agents/journal.py#L102) is the central
substrate every agent reads from and writes to. One table,
`agent_journal`, with these columns
([database.py:293-322](../zeus/data/storage/database.py#L293)):

- `id`, `ts`, `agent_id`, `kind`, `symbol`, `title`, `body`.
- `related_trade_id` (UUID, nullable) — links a `trade_rationale`
  row to its `Trade`.
- `related_position_id` (nullable) — links postmortems to positions.
- `structured` (JSON) — agent-specific structured fields (e.g. a
  research brief's `target_price`, `stop_price`, `confidence`).
- `tags` (text[] on Postgres, JSON on SQLite for tests) — search
  affordances.
- `confidence` (float) — agent's self-assessed confidence.

### Kinds

A frozenset enforced at write time
([journal.py:46-57](../zeus/agents/journal.py#L46)): `trade_rationale`,
`hypothesis`, `postmortem`, `brief`, `memo`, `lesson`, `decision`,
`alert`, `review`, `paper`.

### Write methods

[`record_rationale`](../zeus/agents/journal.py#L168),
[`record_postmortem`](../zeus/agents/journal.py#L194),
[`record_brief`](../zeus/agents/journal.py#L208),
[`record_memo`](../zeus/agents/journal.py#L218),
[`record_lesson`](../zeus/agents/journal.py#L236),
[`record_decision`](../zeus/agents/journal.py#L246),
[`record_alert`](../zeus/agents/journal.py#L256),
[`record_review`](../zeus/agents/journal.py#L228) — all delegate to
[`_write`](../zeus/agents/journal.py#L124), which validates kind in
`KNOWN_KINDS`, validates non-empty title/body, opens a short-lived
session, inserts and commits, returns the row id.

### Read methods

[`recent`](../zeus/agents/journal.py#L264) — last N for this agent.
[`by_symbol`](../zeus/agents/journal.py#L301) — rows about a specific
symbol. [`search`](../zeus/agents/journal.py#L313) — `ILIKE`
substring scan on `title || body` (a real semantic-search layer is a
known TODO — see §17).
[`query_journal`](../zeus/agents/journal.py#L330) — the general
filter, used by overseer audits and cross-agent reads.

### Coupling audit

The most consequential read is the overseer's coupling check: for
each trader fill in the audit window, look for a paired
`brief`/`memo` from the paired research agent within
`RESEARCH_COUPLING_WINDOW_HOURS = 24` hours. Missing → halt the
trader (§12).

This is what makes the system *explainable* — every fill traces to
a rationale, which traces to a research brief, which itself was
written against the journal's accumulated context. Pull any
`Trade.id` and you can reconstruct the full reasoning chain.

---

## 17. Failure modes and safety nets

### Data staleness

The 5-08 incident was 16 days of silent SIP-embargo failures: Alpaca
was returning empty bars for the daily refresh, the trader was
trading off bars that were 16 trading days old, and nothing alerted
because counts of "inserts" were never compared against expected
counts.

The fix lives in
[`zeus/data/freshness.py`](../zeus/data/freshness.py):

- `assert_fresh_or_halt`
  ([line 163](../zeus/data/freshness.py#L163)) — hard gate at the
  top of `market_open_job` and `feature_engineering_job`. If the
  anchor symbols (AAPL, MSFT, SPY) don't have a bar within
  `STALE_TRADING_DAYS_THRESHOLD = 2` trading days, the job aborts
  and writes a CRITICAL `risk_event`.
- `check_ohlcv_freshness` post-refresh self-check
  ([line 74](../zeus/data/freshness.py#L74)) — after
  `data_refresh_job` runs, it re-checks freshness; if still stale
  after refresh, writes a deduplicated `risk_event` (1-hour
  dedup window) so the dashboard surfaces it within an hour.

### Heartbeat staleness

The original 30-minute heartbeat could mask a 29-minute outage.
Tightened (Task 10) to:

- `heartbeat_job` every 5 minutes — writes a `Heartbeat` row.
- `stale_heartbeat_check_job` every 1 minute — if the latest
  `trading_loop` heartbeat is > 10 minutes old, writes a CRITICAL
  `heartbeat_stale` `risk_event`. Throttled by an "is there already
  an open stale-heartbeat alert in the window" check so a wedged
  process produces one alert, not ten.

### Scheduler crash

APScheduler runs against a
[`SQLAlchemyJobStore`](../zeus/scheduler/runner.py#L626) backed by
the same Postgres. All job definitions persist in
`apscheduler_jobs`. A scheduler crash restart picks up the persisted
`next_run_time` and fires missed crons within
`misfire_grace_time = 300s`. Jobs are *parameterless* and resolve
their dependencies (`loop`, `orch`, `scheduler`) at call time via
[`zeus.scheduler.context`](../zeus/scheduler/context.py) — the
trick that makes them picklable into the jobstore.

### Database loss

`backup_db_job` (Task 4) runs at 03:00 ET daily, `pg_dump`s to
`./data/backups/` (host-mounted volume), rotates locally past
`LOCAL_RETENTION_DAYS = 14`, and optionally streams to S3/restic/etc.
via `BACKUP_REMOTE_CMD`. Configure that env var if you want true
off-machine durability.

### Halted agent recovery

A halt sets `RiskEngine._halted_agents[agent_id] = reason`. No
auto-resume — the operator (or a future overseer prompt) must
manually call `resume_agent(agent_id)`. This is intentional: an
agent that decoupled or breached drawdown should not silently
restart.

### Known weaknesses (not yet fixed)

These are real gaps the code knows about:

- **No bandit attribution over realized trades.** The Trade rows now
  exist but nothing reads them to attribute realized P&L back to
  conviction-gate path, regime, paper citation, or LLM tier choice.
  Task 1 unblocked this; the implementation is downstream.
- **Bootstrapped IC LCB is *computed* but not yet *gated on*.** The
  trainer writes `ic_lcb` per fold but the promotion criteria still
  read from `ic` point. Future trainer change.
- **`feature_version` not bumped for the new macro × interaction
  columns.** A model trained on `v1` panel and running on a newly-
  computed `v1` panel that *now* includes interactions will see new
  columns the model wasn't trained on. Either bump version or skip
  interactions when loading models trained against the old schema.
  Until then, models retrain into the new feature set on the next
  Sunday cycle and pick it up via hot-reload.
- **Journal query is `ILIKE` substring** — no embeddings yet. At a
  year of journal entries this becomes the bottleneck. Adding a
  `pgvector` column + semantic retrieval is staged but not done.
- **No shadow trader for prompt changes.** A prompt edit is a live
  experiment with real (paper) capital. The audit identified this as
  the right way to safely A/B prompt changes; not yet built.
- **Universe is large-cap US long-only**, with the alpha most
  arbitraged. Diversification routes (pairs, sector hedges, options
  tail protection) are open avenues for future iteration.

Everything else either works, has explicit instrumentation in
`risk_events` for when it stops working, or is wrapped in a safety
net that fails loud rather than silent.
