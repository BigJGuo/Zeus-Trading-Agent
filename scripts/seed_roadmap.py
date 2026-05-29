"""Seed the `roadmap_tasks` table with the 9-week improvement plan.

Idempotent: re-running this script updates the *content* fields
(`title`, `description`, `acceptance_criteria`, `dependencies`,
`deliverables`) on existing rows but does NOT reset the agent-mutable
state (`status`, `progress_pct`, `blocked_reason`, `notes`,
`started_at`, `completed_at`). Safe to re-run after editing any task.

Run:
    docker compose exec zeus-scheduler python -m scripts.seed_roadmap
    # or from the host once deps are installed:
    python -m scripts.seed_roadmap
"""
from __future__ import annotations

import sys

from zeus.monitoring.dashboard_service import (
    RoadmapTaskError,
    upsert_roadmap_task,
)


# Tasks transcribed from the user's Part 3 roadmap. The id values are
# stable so dependency edges remain meaningful across re-seeds.
TASKS: list[dict] = [
    # ── Week 1 — Measurement infrastructure ────────────────────────────────
    {
        "id": "w1-attribution-table",
        "week": 1,
        "title": "Realized-attribution table",
        "category": "infrastructure",
        "description": (
            "Nightly job joining trades × agent_journal.trade_rationale × "
            "system_metrics (for regime at entry). Compute realized Sharpe "
            "segmented by regime, conviction path (A vs B), confidence "
            "bucket, paper citation, predictor-score decile, sector, and "
            "holding period. Write to a new `trade_attribution` table."
        ),
        "acceptance_criteria": (
            "Table populated for the last 90 days of trades. Dashboard "
            "panel 'Trade Attribution' shows per-segment Sharpe. Document "
            "at least 3 non-obvious findings in notes."
        ),
        "dependencies": [],
        "deliverables": [
            "zeus/research/attribution.py",
            "scripts/backfill_attribution.py",
            "Dashboard panel 'Trade Attribution'",
            "trade_attribution table (alembic migration)",
        ],
    },
    {
        "id": "w1-backtester-scaffold",
        "week": 1,
        "title": "Pipeline event-driven backtester",
        "category": "backtest",
        "description": (
            "Standalone vectorized replayer that simulates the full "
            "pipeline: features → predictions → gate → portfolio_constructor "
            "→ risk_engine → execution → realized P&L. First version with "
            "naive TC (half-spread on entry + half-spread on exit). Replays "
            "the deployed model against 2020–today."
        ),
        "acceptance_criteria": (
            "python -m zeus.backtesting.pipeline_replay --start 2020-01-01 "
            "--end yesterday --model cross_horizon_swing_h5 runs end-to-end "
            "and produces a CSV of daily equity, drawdown, position count. "
            "Establish a baseline portfolio Sharpe number that every future "
            "change will be compared against."
        ),
        "dependencies": [],
        "deliverables": [
            "zeus/backtesting/pipeline_replay.py",
            "zeus/backtesting/tc_model.py",
            "Baseline Sharpe documented in task notes",
        ],
    },
    {
        "id": "w1-pgvector-journal",
        "week": 1,
        "title": "Semantic search on agent journal",
        "category": "infrastructure",
        "description": (
            "Add embedding column (pgvector) to agent_journal. Backfill "
            "embeddings for all existing rows using a local model "
            "(bge-small) or text-embedding-3-small. Add a "
            "semantic_search(query, agent_id=None, kind=None, k=10) method "
            "to AgentJournal."
        ),
        "acceptance_criteria": (
            "All existing journal rows have embeddings. Query 'stocks that "
            "broke out on earnings' returns relevant entries ranked by "
            "cosine similarity. Embedding generation is automatic on "
            "_write."
        ),
        "dependencies": [],
        "deliverables": [
            "Alembic migration: pgvector column + ivfflat index on agent_journal",
            "Embedding generation utility",
            "AgentJournal.semantic_search() method",
            "Unit test covering semantic retrieval",
        ],
    },

    # ── Week 2 — Learn from what attribution revealed ──────────────────────
    {
        "id": "w2-attribution-review",
        "week": 2,
        "title": "Manual review of attribution findings",
        "category": "learning",
        "description": (
            "Spend a session reading the trade_attribution dashboard. "
            "Write findings into the task notes: which conviction path "
            "performs better, which regimes the strategies struggle in, "
            "which sectors lose money, whether the paper-citation "
            "correlation exists. This pre-dependency informs every "
            "subsequent decision."
        ),
        "acceptance_criteria": (
            "notes field contains at least 5 specific findings, each "
            "backed by a Sharpe/win-rate number from the attribution "
            "table."
        ),
        "dependencies": ["w1-attribution-table"],
        "deliverables": [
            "Documented findings in task notes",
            "Proposal of which Part 3 tasks to re-prioritize based on the data",
        ],
    },
    {
        "id": "w2-deflated-sharpe",
        "week": 2,
        "title": "Deflated Sharpe + bootstrap CI on backtests",
        "category": "backtest",
        "description": (
            "Implement Bailey/López de Prado's deflated Sharpe ratio in "
            "zeus/backtesting/metrics.py. Also add bootstrap CI (1000 "
            "resamples at the trade level) returning [5th, 50th, 95th] "
            "percentiles. Wire into the pipeline_replay output so every "
            "backtest reports both."
        ),
        "acceptance_criteria": (
            "Running the Week 1 backtest now reports sharpe, "
            "deflated_sharpe, sharpe_ci_5, sharpe_ci_95. Document the "
            "deflation factor assumed."
        ),
        "dependencies": ["w1-backtester-scaffold"],
        "deliverables": [
            "Updated zeus/backtesting/metrics.py with deflated_sharpe + bootstrap_ci",
            "Updated pipeline_replay CLI output",
        ],
    },

    # ── Week 3 — Better labels, better predictions ─────────────────────────
    {
        "id": "w3-triple-barrier",
        "week": 3,
        "title": "Triple-barrier labeling",
        "category": "training",
        "description": (
            "Replace forward-return regression labels with triple-barrier "
            "labels (López de Prado, AFML ch. 3): for each (symbol, date), "
            "label is +1 if entry + k×ATR hit before entry - k×ATR within "
            "H days, -1 if reverse, 0 if neither. Add "
            "zeus/labels/triple_barrier.py. Train a parallel classifier on "
            "these labels; ensemble with existing regressor."
        ),
        "acceptance_criteria": (
            "Backtester run with triple-barrier labels shows portfolio "
            "Sharpe delta vs Week 1 baseline. If delta is positive, gate "
            "this for promotion. If negative, document why and revert."
        ),
        "dependencies": ["w1-backtester-scaffold", "w2-attribution-review"],
        "deliverables": [
            "zeus/labels/triple_barrier.py",
            "Retrained model bundle in artifacts/",
            "Backtest comparison report in task notes",
        ],
    },
    {
        "id": "w3-sample-uniqueness",
        "week": 3,
        "title": "Sample-uniqueness weighting",
        "category": "training",
        "description": (
            "Implement getAvgUniqueness (AFML ch. 4): weight each training "
            "sample by inverse concurrency of its forward window. Pass as "
            "sample_weight to XGB/LGBM .fit."
        ),
        "acceptance_criteria": (
            "OOS IC improves by ≥ 1pp in walk-forward. Backtest portfolio "
            "Sharpe non-negative delta vs Week 1."
        ),
        "dependencies": ["w1-backtester-scaffold"],
        "deliverables": [
            "zeus/training/sample_weights.py",
            "Updated trainers in scripts/train_*.py",
            "Walk-forward report attached to notes",
        ],
    },
    {
        "id": "w3-probability-calibration",
        "week": 3,
        "title": "Platt/isotonic calibration",
        "category": "training",
        "description": (
            "Add CalibratedClassifierCV (sklearn) wrapping the MetaLabeler "
            "classifier inside walk-forward folds. Calibrate against OOS "
            "predictions."
        ),
        "acceptance_criteria": (
            "Calibration curve (reliability diagram) for the MetaLabeler "
            "shows monotonic improvement vs uncalibrated. Brier score "
            "improves."
        ),
        "dependencies": ["w3-triple-barrier"],
        "deliverables": [
            "Updated zeus/models/meta_labeler.py with calibration",
            "Calibration diagnostic notebook in notebooks/",
        ],
    },

    # ── Week 4 — Smarter decision layer ────────────────────────────────────
    {
        "id": "w4-magnitude-agreement",
        "week": 4,
        "title": "Cross-horizon magnitude agreement (replace binary sign-agreement)",
        "category": "decision",
        "description": (
            "Replace the cross-horizon gate's binary sign agreement with a "
            "continuous agreement_score = sign_agreement_fraction × "
            "geometric_mean(|preds|). Pass only signals where "
            "agreement_score is in the top quintile of historical scores."
        ),
        "acceptance_criteria": (
            "Backtester shows ≥ 5bps annualized Sharpe improvement vs "
            "Week 1 baseline."
        ),
        "dependencies": ["w1-backtester-scaffold"],
        "deliverables": [
            "Updated CrossHorizonGatedPredictor.predict()",
            "Backtest comparison report in notes",
        ],
    },
    {
        "id": "w4-regime-conditional-gates",
        "week": 4,
        "title": "Regime-conditional gate thresholds",
        "category": "decision",
        "description": (
            "Train separate (mag_floor, rank_floor) per regime. Pass "
            "regime as a categorical feature to the predictor (one-hot). "
            "Adjust Kelly fraction by regime: 0.10 in HIGH_VOL, 0.25 in "
            "BULL/RANGE, 0.15 in TRANSITION."
        ),
        "acceptance_criteria": (
            "Backtester shows positive Sharpe delta. Per-regime "
            "attribution shows reduced underperformance in the worst "
            "regime."
        ),
        "dependencies": ["w2-attribution-review", "w1-backtester-scaffold"],
        "deliverables": [
            "Per-regime gate calibration in scripts/train_cross_horizon_bundle.py",
            "Regime feature in zeus/features/pipeline.py",
            "Kelly fraction logic in zeus/risk/position_sizer.py",
        ],
    },

    # ── Week 5 — Two models live, not one ──────────────────────────────────
    {
        "id": "w5-champion-challenger",
        "week": 5,
        "title": "Champion-challenger model serving",
        "category": "infrastructure",
        "description": (
            "Hold the production model + one challenger live per strategy. "
            "Score both on every prediction. Track rolling 30-day OOS IC "
            "of each. Swap challenger to production when its rolling IC "
            "has been higher for ≥ 20 trading days AND has ≥ 30 trade "
            "observations."
        ),
        "acceptance_criteria": (
            "ModelVersion table has both rows live for each strategy. "
            "Dashboard shows side-by-side rolling IC. Swap mechanism "
            "tested in a forced scenario."
        ),
        "dependencies": ["w3-probability-calibration", "w4-magnitude-agreement"],
        "deliverables": [
            "Updated zeus/scheduler/runner.py reload logic",
            "New dashboard panel: champion vs challenger IC",
            "Swap criteria documented in notes",
        ],
    },

    # ── Week 6 — Faster learning ───────────────────────────────────────────
    {
        "id": "w6-daily-incremental",
        "week": 6,
        "title": "Daily incremental retrain for day strategy",
        "category": "training",
        "description": (
            "New job daily_incremental_retrain_job firing after 16:30 ET. "
            "Warm-starts XGB (xgb_model parameter) with the previous day's "
            "bundle, η = 0.001, ≤ 50 additional trees. Register as a new "
            "ModelVersion row that participates in champion-challenger."
        ),
        "acceptance_criteria": (
            "Job runs daily without errors. Incrementally retrained model "
            "accumulates ≥ 30 days of OOS observations and is compared "
            "against the weekly-retrained champion via the Week 5 "
            "mechanism."
        ),
        "dependencies": ["w5-champion-challenger"],
        "deliverables": [
            "New job in zeus/scheduler/jobs.py",
            "New training script scripts/incremental_retrain.py",
        ],
    },

    # ── Week 7 — Adaptive conviction gate ──────────────────────────────────
    {
        "id": "w7-online-conviction-threshold",
        "week": 7,
        "title": "Online conviction-gate threshold calibration",
        "category": "learning",
        "description": (
            "Weekly job that computes the conviction-gate threshold "
            "(currently confidence ≥ 3) that would have maximized realized "
            "Sharpe on accepted trades from the past 60 days, subject to a "
            "minimum trade count. Inject the new threshold into the trader "
            "prompts via a {conviction_threshold} placeholder."
        ),
        "acceptance_criteria": (
            "conviction_thresholds table populated weekly. Prompt "
            "rendering uses the latest value. Backtest with rolling "
            "threshold shows non-negative Sharpe delta."
        ),
        "dependencies": ["w1-attribution-table", "w2-attribution-review"],
        "deliverables": [
            "zeus/research/threshold_calibration.py",
            "New scheduler job",
            "Prompt template updates",
        ],
    },

    # ── Week 8 — Sizing learns from outcomes ───────────────────────────────
    {
        "id": "w8-thompson-bandit-size",
        "week": 8,
        "title": "Thompson Sampling bandit on sizing",
        "category": "learning",
        "description": (
            "Contextual bandit (Bayesian linear regression with Thompson "
            "Sampling) that picks among {full, half, quarter, skip} given "
            "a context vector [predictor_score, regime_one_hot, "
            "conviction_path_one_hot, agreement_score, "
            "strategy_30d_sharpe]. Reward = realized Sharpe of the trade "
            "over its actual hold period."
        ),
        "acceptance_criteria": (
            "Bandit posteriors update on every closed trade. Mean realized "
            "Sharpe after 60 days of bandit operation ≥ baseline "
            "within-period Sharpe."
        ),
        "dependencies": ["w1-attribution-table", "w7-online-conviction-threshold"],
        "deliverables": [
            "zeus/learning/thompson_bandit.py",
            "Bandit-state persistence table",
            "Integration into zeus/live/strategy_manager.py",
        ],
    },

    # ── Week 9+ — Safety, alt-data, prompt experimentation ─────────────────
    {
        "id": "w9-shadow-trader",
        "week": 9,
        "title": "Shadow trader for prompt A/B testing",
        "category": "safety",
        "description": (
            "Flag on trader agents that runs an alternate prompt in "
            "parallel, writing proposals to a shadow_proposals table "
            "instead of submitting. After 30 days, compute realized P&L "
            "of what would have been filled using next-tick prices."
        ),
        "acceptance_criteria": (
            "One full A/B comparison of a real prompt edit (e.g. removing "
            "paper-citation requirement) completed with documented P&L "
            "delta."
        ),
        "dependencies": ["w1-attribution-table"],
        "deliverables": [
            "shadow_proposals table (alembic migration)",
            "Flag on trader agents",
            "A/B comparison report in notes",
        ],
    },
    {
        "id": "w9-alt-data-shortinterest",
        "week": 9,
        "title": "Short-interest delta feature",
        "category": "training",
        "description": (
            "Add FINRA short-interest data (free, twice-monthly) as a "
            "feature. Compute 14-day delta. Add to the feature panel."
        ),
        "acceptance_criteria": (
            "Walk-forward IC contribution measured. Champion-challenger "
            "comparison run."
        ),
        "dependencies": ["w5-champion-challenger"],
        "deliverables": [
            "zeus/data/ingestion/short_interest.py",
            "New feature in zeus/features/",
            "Retrained challenger registered in model_versions",
        ],
    },
    {
        "id": "w9-event-features",
        "week": 9,
        "title": "Earnings/event distance features",
        "category": "training",
        "description": (
            "Add days_to_earnings, days_since_earnings, days_to_fomc, "
            "days_to_exdiv features."
        ),
        "acceptance_criteria": "Walk-forward IC contribution measured.",
        "dependencies": ["w5-champion-challenger"],
        "deliverables": [
            "zeus/features/events.py",
            "New features in pipeline",
            "Retrained challenger registered",
        ],
    },
    {
        "id": "w9-vol-scaled-labels",
        "week": 9,
        "title": "Volatility-scaled labels",
        "category": "training",
        "description": (
            "Predict forward_return / hist_vol_20d rather than raw forward "
            "return. Unscale at sizing time."
        ),
        "acceptance_criteria": (
            "Walk-forward IC and backtest Sharpe both reported."
        ),
        "dependencies": ["w3-triple-barrier"],
        "deliverables": [
            "Updated label generation in zeus/labels/",
            "Updated zeus/risk/position_sizer.py with unscaling",
        ],
    },
]


def main() -> int:
    inserted_or_updated = 0
    errors: list[tuple[str, str]] = []
    for task in TASKS:
        try:
            upsert_roadmap_task(task)
            inserted_or_updated += 1
        except RoadmapTaskError as e:
            errors.append((task["id"], str(e)))
            print(f"[error] {task['id']}: {e}", file=sys.stderr)

    print(f"seeded {inserted_or_updated}/{len(TASKS)} tasks")
    if errors:
        print(f"{len(errors)} errors", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
