from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://f1_telemetry:f1_telemetry@localhost:5432/f1_telemetry",
)


class Base(DeclarativeBase):
    pass


class TelemetrySnapshot(Base):
    __tablename__ = "telemetry_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    time_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    throttle: Mapped[float] = mapped_column(Float, nullable=False)
    brake: Mapped[float] = mapped_column(Float, nullable=False)
    rpm: Mapped[float] = mapped_column(Float, nullable=False)
    gear: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    actual_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    alert_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="unknown",
    )
    raw_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    engineered_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    @classmethod
    def from_inference(
        cls,
        *,
        raw_snapshot: Mapping[str, Any],
        engineered_snapshot: Mapping[str, Any],
        result: Mapping[str, Any],
        model_version: str,
    ) -> "TelemetrySnapshot":
        return cls(
            captured_at=datetime.fromisoformat(str(result["CapturedAt"])),
            time_sec=float(result["TimeSec"]),
            speed=float(result["Speed"]),
            throttle=float(raw_snapshot.get("Throttle", 0.0)),
            brake=float(result["Brake"]),
            rpm=float(raw_snapshot.get("RPM", 0.0)),
            gear=int(raw_snapshot.get("nGear", 0)),
            predicted_temperature=float(result["Predicted_Temp"]),
            actual_temperature=float(result["Actual_Temp"]),
            anomaly_score=float(result["Anomaly_Score"]),
            alert_threshold=float(result["Alert_Threshold"]),
            is_anomaly=bool(result["Is_Anomaly"]),
            model_version=model_version,
            raw_snapshot=dict(raw_snapshot),
            engineered_snapshot=dict(engineered_snapshot),
        )


engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
    pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE_SECONDS", "1800")),
)

AsyncSessionFactory = async_sessionmaker[
    AsyncSession
](
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionFactory() as session:
        yield session


async def init_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def check_database_ready() -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def close_database() -> None:
    await engine.dispose()
