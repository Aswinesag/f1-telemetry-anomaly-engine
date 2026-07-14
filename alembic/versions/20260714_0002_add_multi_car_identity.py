from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260714_0002"
down_revision = "20260714_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "telemetry_snapshots",
        sa.Column("session_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "telemetry_snapshots",
        sa.Column("car_id", sa.String(length=32), nullable=True),
    )
    op.execute("UPDATE telemetry_snapshots SET session_id = 'default' WHERE session_id IS NULL")
    op.execute("UPDATE telemetry_snapshots SET car_id = 'unknown' WHERE car_id IS NULL")
    op.alter_column("telemetry_snapshots", "session_id", nullable=False)
    op.alter_column("telemetry_snapshots", "car_id", nullable=False)
    op.create_index(
        "ix_telemetry_snapshots_session_car_time",
        "telemetry_snapshots",
        ["session_id", "car_id", "time_sec"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_telemetry_snapshots_session_car_time",
        table_name="telemetry_snapshots",
    )
    op.drop_column("telemetry_snapshots", "car_id")
    op.drop_column("telemetry_snapshots", "session_id")
