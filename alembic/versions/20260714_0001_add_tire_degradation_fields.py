from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260714_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "telemetry_snapshots",
        sa.Column("tire_compound", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "telemetry_snapshots",
        sa.Column("stint_lap_number", sa.Integer(), nullable=True),
    )
    op.add_column(
        "telemetry_snapshots",
        sa.Column("calculated_degradation_index", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_telemetry_snapshots_calculated_degradation_index",
        "telemetry_snapshots",
        ["calculated_degradation_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_telemetry_snapshots_calculated_degradation_index",
        table_name="telemetry_snapshots",
    )
    op.drop_column("telemetry_snapshots", "calculated_degradation_index")
    op.drop_column("telemetry_snapshots", "stint_lap_number")
    op.drop_column("telemetry_snapshots", "tire_compound")
