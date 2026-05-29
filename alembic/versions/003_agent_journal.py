"""Agent journal: persistent memory for the 7-agent system.

Revision ID: 003
Revises: 002
Create Date: 2026-04-20

Adds `agent_journal`, the primary memory substrate every agent writes to:
trader rationales, research briefs/memos, overseer decisions, alerts, lessons.
Journal rows are the auditable trail linking every fill to the reasoning +
research brief + model version that produced it (the "explainability
invariant").
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_journal",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # 'day', 'swing', 'long_term', 'day_research', 'swing_research',
        # 'long_term_research', 'overseer'
        sa.Column("agent_id", sa.String(32), nullable=False),
        # 'trade_rationale' | 'hypothesis' | 'postmortem' | 'brief' | 'memo'
        # | 'lesson' | 'decision' | 'alert' | 'review'
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=True),
        sa.Column("related_trade_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("related_position_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("structured", JSONB(), nullable=True),
        sa.Column("tags", ARRAY(sa.Text()), nullable=True),
        sa.Column("confidence", sa.REAL(), nullable=True),
    )
    op.create_index(
        "ix_journal_agent_ts", "agent_journal", ["agent_id", sa.text("ts DESC")]
    )
    op.create_index("ix_journal_symbol", "agent_journal", ["symbol"])
    op.create_index("ix_journal_kind", "agent_journal", ["kind"])
    # Composite index used by overseer to audit research-trader coupling:
    # "for this trader's fill at ts, did the paired research agent emit a
    # brief for this symbol in the window?"
    op.create_index(
        "ix_journal_symbol_agent_ts",
        "agent_journal",
        ["symbol", "agent_id", sa.text("ts DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_journal_symbol_agent_ts", table_name="agent_journal")
    op.drop_index("ix_journal_kind", table_name="agent_journal")
    op.drop_index("ix_journal_symbol", table_name="agent_journal")
    op.drop_index("ix_journal_agent_ts", table_name="agent_journal")
    op.drop_table("agent_journal")
