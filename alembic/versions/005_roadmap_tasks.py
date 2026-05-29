"""Roadmap tasks: improvement-plan tracker the dashboard tab renders.

Revision ID: 005
Revises: 004
Create Date: 2026-05-17

Adds the `roadmap_tasks` table that backs the new dashboard "Roadmap"
tab. Each row is one work item from the 9-week improvement plan;
status transitions (`not_started → in_progress → complete`, plus
`blocked` as an off-ramp) are mutated by the agent itself as it works,
gated by dependency completeness.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "roadmap_tasks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("week", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("acceptance_criteria", sa.Text(), nullable=False),
        # Postgres ARRAY of text; tests use SQLite which gets JSON via the
        # model's `.with_variant(JSON(), "sqlite")` declaration.
        sa.Column("dependencies", ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("deliverables", ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="not_started"),
        sa.Column("progress_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("blocked_reason", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_roadmap_tasks_week", "roadmap_tasks", ["week"])
    op.create_index("ix_roadmap_tasks_category", "roadmap_tasks", ["category"])
    op.create_index("ix_roadmap_tasks_status", "roadmap_tasks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_roadmap_tasks_status", table_name="roadmap_tasks")
    op.drop_index("ix_roadmap_tasks_category", table_name="roadmap_tasks")
    op.drop_index("ix_roadmap_tasks_week", table_name="roadmap_tasks")
    op.drop_table("roadmap_tasks")
