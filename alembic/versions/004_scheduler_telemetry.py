"""Scheduler telemetry: per-job state, execution log, per-agent LLM spend.

Revision ID: 004
Revises: 003
Create Date: 2026-04-21

Three tables backing the dashboard's scheduler/agent-status panel:
- `scheduler_jobs` — one row per registered APScheduler job, upserted each
  time the job is submitted/executed/errored; carries `next_run_time` +
  last-run stats so the dashboard can show a countdown + health chip.
- `job_executions` — append-only execution log (last N shown on dashboard,
  older rows retained for audit + post-mortems).
- `agent_daily_spend` — per-agent/day token + USD totals, upserted by the
  LLM client. Drives the overseer's per-agent budget guardrail and the
  dashboard's daily cost tile.
"""
from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scheduler_jobs",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("func_name", sa.Text(), nullable=False),
        sa.Column("trigger_repr", sa.Text(), nullable=True),
        sa.Column("next_run_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_duration_ms", sa.Integer(), nullable=True),
        sa.Column("last_success", sa.Boolean(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_scheduler_jobs_next_run",
        "scheduler_jobs",
        ["next_run_time"],
    )

    op.create_table(
        "job_executions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("func_name", sa.Text(), nullable=False),
        sa.Column(
            "scheduled_ts", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "started_ts", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        # 'submitted' | 'success' | 'error' | 'missed'
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_job_executions_started",
        "job_executions",
        [sa.text("started_ts DESC")],
    )
    op.create_index(
        "ix_job_executions_job_started",
        "job_executions",
        ["job_id", sa.text("started_ts DESC")],
    )

    op.create_table(
        "agent_daily_spend",
        sa.Column("agent_id", sa.String(32), primary_key=True),
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column(
            "input_tokens", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "output_tokens", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "cache_read_tokens",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "cache_write_tokens",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "calls", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "cost_usd", sa.Numeric(10, 4), nullable=False, server_default="0"
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_agent_daily_spend_date",
        "agent_daily_spend",
        [sa.text("date DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_daily_spend_date", table_name="agent_daily_spend")
    op.drop_table("agent_daily_spend")
    op.drop_index("ix_job_executions_job_started", table_name="job_executions")
    op.drop_index("ix_job_executions_started", table_name="job_executions")
    op.drop_table("job_executions")
    op.drop_index("ix_scheduler_jobs_next_run", table_name="scheduler_jobs")
    op.drop_table("scheduler_jobs")
