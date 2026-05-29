"""Multi-strategy support: strategy_id on positions/trades/orders/signals.

Revision ID: 002
Revises: 001
Create Date: 2026-04-20

Adds a `strategy_id` dimension so three concurrent trader agents (day, swing,
long_term) can share one broker account without clobbering each other's state.

Positions move from PK (symbol) to PK (symbol, strategy_id). The broker still
sees aggregate shares; `strategy_shares` on each row lets us partition those
aggregate shares between strategies. Drift between sum(strategy_shares) and
broker qty is surfaced by the reconciler as a `strategy_share_drift` event.
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


DEFAULT_STRATEGY = "legacy"


def upgrade() -> None:
    # ─── positions: add strategy_id + bookkeeping, switch PK ──────────────────
    op.add_column(
        "positions",
        sa.Column("strategy_id", sa.String(20), nullable=False, server_default=DEFAULT_STRATEGY),
    )
    op.add_column(
        "positions",
        sa.Column("strategy_shares", sa.Integer(), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("entry_ts", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("strategy_model_version", sa.String(50), nullable=True),
    )
    # Backfill: treat pre-existing rows as the 'legacy' strategy owning 100% of shares.
    op.execute(
        "UPDATE positions SET strategy_shares = qty WHERE strategy_shares IS NULL"
    )
    # Re-key: PK (symbol) -> PK (symbol, strategy_id)
    op.drop_constraint("positions_pkey", "positions", type_="primary")
    op.create_primary_key("positions_pkey", "positions", ["symbol", "strategy_id"])
    op.create_index("ix_positions_strategy_id", "positions", ["strategy_id"])

    # ─── trades ───────────────────────────────────────────────────────────────
    op.add_column(
        "trades",
        sa.Column("strategy_id", sa.String(20), nullable=False, server_default=DEFAULT_STRATEGY),
    )
    op.add_column(
        "trades",
        sa.Column("strategy_model_version", sa.String(50), nullable=True),
    )
    op.create_index("ix_trades_strategy_id", "trades", ["strategy_id"])

    # ─── orders ───────────────────────────────────────────────────────────────
    op.add_column(
        "orders",
        sa.Column("strategy_id", sa.String(20), nullable=False, server_default=DEFAULT_STRATEGY),
    )
    op.create_index("ix_orders_strategy_id", "orders", ["strategy_id"])

    # ─── signals ──────────────────────────────────────────────────────────────
    op.add_column(
        "signals",
        sa.Column("strategy_id", sa.String(20), nullable=False, server_default=DEFAULT_STRATEGY),
    )
    op.create_index("ix_signals_strategy_id", "signals", ["strategy_id"])

    # Drop the server_default once backfill is done — the application code always
    # sets strategy_id explicitly, and keeping the default silently masks bugs
    # where a caller forgot to pass it.
    op.alter_column("positions", "strategy_id", server_default=None)
    op.alter_column("trades", "strategy_id", server_default=None)
    op.alter_column("orders", "strategy_id", server_default=None)
    op.alter_column("signals", "strategy_id", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_signals_strategy_id", table_name="signals")
    op.drop_column("signals", "strategy_id")

    op.drop_index("ix_orders_strategy_id", table_name="orders")
    op.drop_column("orders", "strategy_id")

    op.drop_index("ix_trades_strategy_id", table_name="trades")
    op.drop_column("trades", "strategy_model_version")
    op.drop_column("trades", "strategy_id")

    op.drop_index("ix_positions_strategy_id", table_name="positions")
    op.drop_constraint("positions_pkey", "positions", type_="primary")
    op.create_primary_key("positions_pkey", "positions", ["symbol"])
    op.drop_column("positions", "strategy_model_version")
    op.drop_column("positions", "entry_ts")
    op.drop_column("positions", "strategy_shares")
    op.drop_column("positions", "strategy_id")
