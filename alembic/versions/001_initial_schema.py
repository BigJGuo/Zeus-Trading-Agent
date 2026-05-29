"""Initial schema with TimescaleDB hypertables.

Revision ID: 001
Revises:
Create Date: 2025-04-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ohlcv_daily (will become TimescaleDB hypertable)
    op.create_table(
        'ohlcv_daily',
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open', sa.Float()),
        sa.Column('high', sa.Float()),
        sa.Column('low', sa.Float()),
        sa.Column('close', sa.Float()),
        sa.Column('volume', sa.BigInteger()),
        sa.Column('vwap', sa.Float()),
        sa.Column('adj_close', sa.Float()),
        sa.Column('source', sa.String(20), server_default='alpaca'),
        sa.PrimaryKeyConstraint('symbol', 'ts'),
    )
    # Convert to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('ohlcv_daily', 'ts', if_not_exists => TRUE)")

    # ohlcv_intraday
    op.create_table(
        'ohlcv_intraday',
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('timeframe', sa.String(5), nullable=False),
        sa.Column('open', sa.Float()),
        sa.Column('high', sa.Float()),
        sa.Column('low', sa.Float()),
        sa.Column('close', sa.Float()),
        sa.Column('volume', sa.BigInteger()),
        sa.Column('vwap', sa.Float()),
        sa.PrimaryKeyConstraint('symbol', 'ts', 'timeframe'),
    )
    op.execute("SELECT create_hypertable('ohlcv_intraday', 'ts', if_not_exists => TRUE)")

    # features_daily
    op.create_table(
        'features_daily',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('feature_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('feature_version', sa.String(20), nullable=False, server_default='v1'),
        sa.Column('features', postgresql.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'feature_date', 'feature_version'),
    )
    op.create_index('ix_features_daily_symbol', 'features_daily', ['symbol'])
    op.create_index('ix_features_daily_date', 'features_daily', ['feature_date'])

    # signals
    op.create_table(
        'signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('signal_version', sa.String(20), nullable=False),
        sa.Column('expected_return', sa.Float()),
        sa.Column('expected_downside', sa.Float()),
        sa.Column('up_probability', sa.Float()),
        sa.Column('down_probability', sa.Float()),
        sa.Column('confidence', sa.Float()),
        sa.Column('blended_score', sa.Float()),
        sa.Column('vol_estimate', sa.Float()),
        sa.Column('liquidity_score', sa.Float()),
        sa.Column('regime', sa.String(20)),
        sa.Column('model_version', sa.String(50)),
        sa.Column('feature_version', sa.String(20)),
        sa.Column('raw_scores', postgresql.JSON()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_signals_ts', 'signals', ['ts'])
    op.create_index('ix_signals_symbol', 'signals', ['symbol'])

    # trades
    op.create_table(
        'trades',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('direction', sa.String(5), nullable=False),
        sa.Column('entry_ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('exit_ts', sa.DateTime(timezone=True)),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('exit_price', sa.Float()),
        sa.Column('shares', sa.Integer(), nullable=False),
        sa.Column('gross_pnl', sa.Float()),
        sa.Column('commission', sa.Float(), server_default='0.0'),
        sa.Column('net_pnl', sa.Float()),
        sa.Column('hold_days', sa.Float()),
        sa.Column('exit_reason', sa.String(50)),
        sa.Column('signal_id', sa.Integer(), sa.ForeignKey('signals.id'), nullable=True),
        sa.Column('regime_at_entry', sa.String(20)),
        sa.Column('predicted_return', sa.Float()),
        sa.Column('actual_return', sa.Float()),
        sa.Column('entry_stop', sa.Float()),
        sa.Column('peak_price', sa.Float()),
        sa.Column('environment', sa.String(10), nullable=False, server_default='paper'),
        sa.Column('alpaca_order_id', sa.String(50)),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_trades_symbol', 'trades', ['symbol'])

    # orders
    op.create_table(
        'orders',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('side', sa.String(5), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('limit_price', sa.Float()),
        sa.Column('submitted_at', sa.DateTime(timezone=True)),
        sa.Column('filled_at', sa.DateTime(timezone=True)),
        sa.Column('filled_qty', sa.Integer(), server_default='0'),
        sa.Column('filled_avg_price', sa.Float()),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('environment', sa.String(10), nullable=False, server_default='paper'),
        sa.Column('trade_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('trades.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_orders_symbol', 'orders', ['symbol'])
    op.create_index('ix_orders_status', 'orders', ['status'])

    # positions
    op.create_table(
        'positions',
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('avg_entry_price', sa.Float(), nullable=False),
        sa.Column('current_price', sa.Float()),
        sa.Column('market_value', sa.Float()),
        sa.Column('unrealized_pnl', sa.Float()),
        sa.Column('unrealized_pnl_pct', sa.Float()),
        sa.Column('hard_stop', sa.Float()),
        sa.Column('trailing_pct', sa.Float()),
        sa.Column('peak_price', sa.Float()),
        sa.Column('environment', sa.String(10), nullable=False, server_default='paper'),
        sa.Column('updated_at', sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint('symbol'),
    )

    # model_versions
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_name', sa.String(50), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('promoted_at', sa.DateTime(timezone=True)),
        sa.Column('status', sa.String(20), nullable=False, server_default='staging'),
        sa.Column('metrics', postgresql.JSON()),
        sa.Column('config', postgresql.JSON()),
        sa.Column('artifact_path', sa.Text()),
        sa.Column('training_start_date', sa.DateTime(timezone=True)),
        sa.Column('training_end_date', sa.DateTime(timezone=True)),
        sa.Column('n_training_samples', sa.Integer()),
        sa.Column('feature_version', sa.String(20)),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_model_versions_name', 'model_versions', ['model_name'])

    # risk_events
    op.create_table(
        'risk_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('portfolio_value', sa.Float()),
        sa.Column('drawdown_usd', sa.Float()),
        sa.Column('drawdown_pct', sa.Float()),
        sa.Column('action_taken', sa.Text()),
        sa.Column('environment', sa.String(10), server_default='paper'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_risk_events_ts', 'risk_events', ['ts'])

    # heartbeats
    op.create_table(
        'heartbeats',
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('component', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('details', postgresql.JSON()),
        sa.PrimaryKeyConstraint('ts'),
    )

    # system_metrics
    op.create_table(
        'system_metrics',
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('portfolio_value', sa.Float()),
        sa.Column('cash_balance', sa.Float()),
        sa.Column('unrealized_pnl', sa.Float()),
        sa.Column('realized_pnl_daily', sa.Float()),
        sa.Column('drawdown_from_peak_usd', sa.Float()),
        sa.Column('drawdown_from_start_usd', sa.Float()),
        sa.Column('drawdown_from_peak_pct', sa.Float()),
        sa.Column('open_position_count', sa.Integer()),
        sa.Column('model_ic_rolling20d', sa.Float()),
        sa.Column('regime', sa.String(20)),
        sa.Column('api_error_count_1h', sa.Integer(), server_default='0'),
        sa.Column('environment', sa.String(10), server_default='paper'),
        sa.PrimaryKeyConstraint('ts'),
    )

    # fundamentals_cache
    op.create_table(
        'fundamentals_cache',
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('pe_ratio', sa.Float()),
        sa.Column('pb_ratio', sa.Float()),
        sa.Column('ps_ratio', sa.Float()),
        sa.Column('ev_ebitda', sa.Float()),
        sa.Column('earnings_yield', sa.Float()),
        sa.Column('revenue_growth_yoy', sa.Float()),
        sa.Column('earnings_growth_yoy', sa.Float()),
        sa.Column('gross_margin', sa.Float()),
        sa.Column('operating_margin', sa.Float()),
        sa.Column('debt_to_equity', sa.Float()),
        sa.Column('current_ratio', sa.Float()),
        sa.Column('roe', sa.Float()),
        sa.Column('roa', sa.Float()),
        sa.Column('short_interest_ratio', sa.Float()),
        sa.Column('market_cap', sa.Float()),
        sa.Column('sector', sa.String(50)),
        sa.Column('industry', sa.String(100)),
        sa.Column('raw_info', postgresql.JSON()),
        sa.PrimaryKeyConstraint('symbol'),
    )

    # universe_snapshots
    op.create_table(
        'universe_snapshots',
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('snapshot_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('in_sp500', sa.Boolean(), server_default='false'),
        sa.Column('in_nasdaq100', sa.Boolean(), server_default='false'),
        sa.Column('avg_dollar_volume_20d', sa.Float()),
        sa.Column('avg_spread_bps', sa.Float()),
        sa.Column('passes_filter', sa.Boolean(), nullable=False),
        sa.Column('filter_reason', sa.String(100)),
        sa.Column('sector', sa.String(50)),
        sa.Column('industry', sa.String(100)),
        sa.PrimaryKeyConstraint('symbol', 'snapshot_date'),
    )


def downgrade() -> None:
    op.drop_table('universe_snapshots')
    op.drop_table('fundamentals_cache')
    op.drop_table('system_metrics')
    op.drop_table('heartbeats')
    op.drop_table('risk_events')
    op.drop_table('model_versions')
    op.drop_table('positions')
    op.drop_table('orders')
    op.drop_table('trades')
    op.drop_table('signals')
    op.drop_table('features_daily')
    op.drop_table('ohlcv_intraday')
    op.drop_table('ohlcv_daily')
