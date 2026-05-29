# Zeus Training Guide — Ways to Train Right Now

> What training options are available given the data already loaded.
> As of 2026-04-20: 764,526 OHLCV daily rows, 633,977 feature rows, 0 trades, 0 signals, 4 `model_versions` rows (all `status=failed`).

---

## 1. The core training pipeline (already built)

**Entrypoint:** [scripts/bootstrap_train.py](../scripts/bootstrap_train.py)
**Orchestrator:** [zeus/models/trainer.py](../zeus/models/trainer.py)
**Model:** XGBoost regressor, 5-day forward log-return target → [zeus/models/return_predictor.py](../zeus/models/return_predictor.py)
**Labels:** `log(close[t+5] / close[t])`, direction buckets at ±1% → [zeus/models/labels.py](../zeus/models/labels.py)
**Features:** 50+ across technical / fundamental / macro / cross-sectional → [zeus/features/pipeline.py](../zeus/features/pipeline.py)

**Run it:**
```bash
docker compose run --rm zeus-scheduler python -m scripts.bootstrap_train
```

**Promotion gates** ([zeus/models/trainer.py:29-35](../zeus/models/trainer.py#L29)):
| Metric | Threshold |
|---|---|
| Information Coefficient | ≥ 0.03 |
| IC stability | ≥ 0.30 |
| Hit rate | ≥ 0.52 |
| Sharpe on signals | ≥ 0.80 |
| Max drawdown on signals | ≥ −0.25 |

Current `v20260419_f82782` fails on **hit_rate = 0.393**. Everything else passes.

---

## 2. Training options, ranked by effort

### Option A — Lower the forward horizon (cheapest, try first)

**Why:** Hit rate of 0.393 on 5-day returns suggests the model picks direction wrong more often than right over a 5-day window. Markets are noisier at longer horizons. Try 1-day or 3-day.

**How:** Edit `forward_return_days` in [zeus/config/config.yaml](../zeus/config/config.yaml) (the exploration agent confirmed this key exists around line 40). Rerun bootstrap_train.

**Expected:** Hit rate typically improves 2-5pp on shorter horizons because intraday momentum and overnight drift dominate.

### Option B — Walk-forward cross-validation (honest OOS)

**Why:** The current metrics come from a single 80/10/10 time split. That's one realization. `sharpe_on_signals=28.9` with `max_drawdown=0.0` is almost certainly artifact of a benign slice. Walk-forward averages across many train/test rolls and will show the real distribution.

**How:** [zeus/backtesting/walk_forward.py](../zeus/backtesting/walk_forward.py) — 18-month train / 3-month test / 1-month step with 20-day embargo. Wire it into a training loop that evaluates the same model spec across all folds and reports mean ± std of IC, hit rate, Sharpe.

**Run:**
```bash
docker compose run --rm zeus-scheduler python -c "
from zeus.backtesting.walk_forward import WalkForward
from zeus.models.return_predictor import XGBReturnPredictor
# wire into a script that iterates folds
"
```

### Option C — Hyperparameter search

**Why:** Defaults (`n_estimators=500, max_depth=5, lr=0.01`) are reasonable but not tuned. Grid or random search could push IC from 0.14 toward 0.20+.

**How:** Write a small driver that loops over `(max_depth ∈ [3,5,7], lr ∈ [0.005, 0.01, 0.03], n_estimators ∈ [200, 500, 1000])` = 27 combinations. Evaluate each on the 10% validation slice. Keep the one that passes all five gates.

Respect training time: full backfill takes ~10-20min per model. Use `early_stopping_rounds=50` (already set) and cap `n_estimators`.

### Option D — Ensemble XGB + LightGBM

**Why:** `config.yaml` already defines LightGBM params (exploration agent noted lines 50-55) but only XGB is trained. A simple (XGB + LGB)/2 average reliably improves IC by 1-3pp because the two models disagree on different subsets of the feature space.

**How:** Add a `LightGBMReturnPredictor` alongside `XGBReturnPredictor`, train both, save both, and have `SignalGenerator` average their predictions. New file `zeus/models/lightgbm_return_predictor.py` following the same `BaseModel` interface.

### Option E — Direction model as classifier

**Why:** Hit rate is directly about direction. Train on the `direction` label (−1/0/+1) as a classifier instead of regressing the return magnitude. Pair with a separate magnitude/vol model if you want sizing.

**How:** Replace `XGBRegressor` with `XGBClassifier` in a new model class. Label: already computed in `labels.py` as the ±1% bucketed field. Gate the `up_probability` threshold (e.g., only take long signals where `up_probability > 0.58`).

### Option F — Add intraday features (when `ohlcv_intraday` is populated)

**Why:** 0 intraday rows today. Once populated, opening-range, VWAP deviation intraday, first-hour volume, and gap-fill stats all strengthen next-day signals.

**How:** Extend [zeus/features/technical.py](../zeus/features/technical.py) with intraday-derived features, bump `feature_version` to `v2`, rebuild the feature snapshot, retrain.

### Option G — Meta-labeling (Lopez de Prado)

**Why:** Use the current model to produce a raw long/short signal, then train a second model to predict *whether the primary signal will win or lose*. The meta-model doesn't have to beat the market — it just has to improve when the primary fires. This is a well-known path from 50% → 55-60% hit rate.

**How:** Two-stage: (1) primary model gives direction, (2) meta-features = primary signal + confidence + regime + vol + recent autocorrelation, label = did the primary's call win. Train a binary classifier. Only execute primary signals where meta says "likely win."

### Option H — Train-on-simulated-trades (the bootstrap path)

**Why:** With 0 actual trades and 0 signals in the DB, the model has no closed-loop learning signal from the live agent. Run a backtest over the historical 764k bars that executes the current plan logic, logs every simulated trade outcome, and feeds the (features, outcome) pairs back as training data.

**How:** [zeus/backtesting/](../zeus/backtesting/) already has the machinery. Write a driver that (a) replays each historical day, (b) asks the current policy what it would have done, (c) records the realized outcome, (d) stores to `trades`-shaped parquet. Use this dataset as an additional training signal for meta-labeling (Option G).

---

## 3. Fix before training again

Two bugs were patched tonight and must be present before retrain:

1. **[zeus/scheduler/runner.py `_load_latest_model()`](../zeus/scheduler/runner.py)** now rejects `status=failed` rows. A retrained model that fails gates won't be used; the `_NullPredictor` fallback keeps the agent flat instead of trading with zeros.
2. **[zeus/models/trainer.py `save_model()`](../zeus/models/trainer.py)** now upserts on `(model_name, version)` instead of inserting a new row every time. Prevents the 4-duplicate rows pattern seen on `v20260419_f82782`.

---

## 4. Recommended path for this week

1. **Tonight:** let the scheduler run. The 21:00 ET planner will produce `2026-04-21.json` using `_NullPredictor` (zero signals → fallback equal-weight). 21:30 ET `company_research_job` will enrich it with news + fundamentals.
2. **Tomorrow:** run Option A (shorter horizon) + Option B (walk-forward) in parallel. Pick whichever clears the gates with better stability.
3. **Wednesday:** if neither clears gates, try Option D (ensemble) and/or Option E (classifier).
4. **Sunday:** `nightly_retrain_job` runs automatically at 02:00 ET — by then you want a model spec + feature version that you're confident about, since it will deploy whatever passes the gates.

---

## 5. Data coverage check before training

Before any serious retrain, confirm coverage is what you expect:

```bash
docker exec zeus-postgres psql -U zeus -d zeus_db -c "
  SELECT MIN(ts)::date AS first, MAX(ts)::date AS last,
         COUNT(*) AS rows, COUNT(DISTINCT symbol) AS symbols
  FROM ohlcv_daily;"
```

With ~765k rows and a ~500-symbol universe, that's roughly ~1,500 trading days per symbol = ~6 years. That's enough for walk-forward with 18-month windows. If `last` is stale (>24h old on a weekday), run the data refresh first — which, per the recent fix, now auto-triggers on scheduler startup if `ohlcv_daily` is >24h behind.
