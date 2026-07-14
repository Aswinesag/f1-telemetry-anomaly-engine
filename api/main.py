from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, cast

from aiokafka import AIOKafkaConsumer
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import select

from api.database.models import (
    AsyncSessionFactory,
    TelemetrySnapshot,
    check_database_ready,
    close_database,
    init_database,
)
from api.schemas import ServiceHealth, TelemetryAccepted, TelemetryInput
from api.worker import InferenceResult, InferenceWorker


LOGGER = logging.getLogger(__name__)
KAFKA_TOPIC = os.getenv("KAFKA_TELEMETRY_TOPIC", "f1-telemetry-bus")
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WEBSOCKET_POLL_SECONDS = float(os.getenv("WEBSOCKET_POLL_SECONDS", "0.05"))


def _json_deserializer(payload: bytes) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(payload.decode("utf-8")))


async def _consume_kafka(
    consumer: AIOKafkaConsumer,
    worker: InferenceWorker,
) -> None:
    try:
        async for message in consumer:
            await worker.enqueue(message.value)
    except asyncio.CancelledError:
        raise
    except Exception:
        LOGGER.exception("Kafka ingestion task stopped unexpectedly")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis_client = Redis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        health_check_interval=30,
    )
    worker: InferenceWorker | None = None
    consumer: AIOKafkaConsumer | None = None
    kafka_task: asyncio.Task[None] | None = None

    try:
        await redis_client.ping()
        await init_database()

        worker = await InferenceWorker.create(
            redis_client=redis_client,
            session_factory=AsyncSessionFactory,
            queue_max_size=int(os.getenv("INFERENCE_QUEUE_MAX_SIZE", "2000")),
        )
        await worker.start()

        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=os.getenv("KAFKA_CONSUMER_GROUP", "inference-api"),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=_json_deserializer,
        )
        await consumer.start()
        kafka_task = asyncio.create_task(
            _consume_kafka(consumer, worker),
            name="kafka-telemetry-consumer",
        )

        app.state.redis = redis_client
        app.state.worker = worker
        app.state.kafka_consumer = consumer
        app.state.kafka_task = kafka_task
        yield
    finally:
        if kafka_task is not None:
            kafka_task.cancel()
            await asyncio.gather(kafka_task, return_exceptions=True)
        if consumer is not None:
            await consumer.stop()
        if worker is not None:
            await worker.stop()
        await redis_client.aclose()
        await close_database()


app = FastAPI(
    title="F1 Inference Engine",
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8502",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


def get_worker(request: Request) -> InferenceWorker:
    worker = getattr(request.app.state, "worker", None)
    if worker is None:
        raise HTTPException(
            status_code=503,
            detail="Inference worker is not initialized.",
        )
    return cast(InferenceWorker, worker)


def get_redis(request: Request) -> Redis:
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Redis client is not initialized.",
        )
    return cast(Redis, redis_client)


@app.get("/health", response_model=ServiceHealth)
async def health(request: Request) -> ServiceHealth:
    worker = getattr(request.app.state, "worker", None)
    return ServiceHealth(
        status="ok",
        service="inference-api",
        device=worker.device if worker is not None else None,
    )


@app.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    worker = getattr(request.app.state, "worker", None)
    redis_client = getattr(request.app.state, "redis", None)
    kafka_task = getattr(request.app.state, "kafka_task", None)

    try:
        redis_ready = bool(redis_client is not None and await redis_client.ping())
    except Exception:
        LOGGER.exception("Redis readiness check failed")
        redis_ready = False

    database_ready = await check_database_ready()
    kafka_ready = kafka_task is not None and not kafka_task.done()
    worker_ready = worker is not None and worker.is_running
    is_ready = redis_ready and database_ready and kafka_ready and worker_ready

    payload: dict[str, Any] = {
        "status": "ready" if is_ready else "not_ready",
        "checks": {
            "database": database_ready,
            "kafka": kafka_ready,
            "redis": redis_ready,
            "worker": worker_ready,
        },
        "queue_depth": worker.queue_depth if worker is not None else None,
    }
    if kafka_task is not None and kafka_task.done():
        kafka_error = kafka_task.exception()
        if kafka_error is not None:
            payload["checks"]["kafka_error"] = str(kafka_error)
    if not is_ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.post(
    "/telemetry",
    response_model=TelemetryAccepted,
    status_code=202,
)
async def ingest_telemetry(
    telemetry: TelemetryInput,
    request: Request,
) -> TelemetryAccepted:
    worker = get_worker(request)
    queue_depth = await worker.enqueue(telemetry.as_worker_payload())
    return TelemetryAccepted(status="accepted", queue_depth=queue_depth)


@app.get("/telemetry/latest", response_model=None)
async def latest_telemetry(request: Request) -> InferenceResult:
    latest = await get_worker(request).latest_result()
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail="No telemetry has been processed yet.",
        )
    return latest


@app.get("/telemetry/compare/{car_a}/{car_b}", response_model=None)
async def compare_telemetry(
    car_a: str,
    car_b: str,
    session_id: str = Query(..., min_length=1),
) -> dict[str, Any]:
    async with AsyncSessionFactory() as session:
        result = await session.execute(
            select(
                TelemetrySnapshot.car_id,
                TelemetrySnapshot.time_sec,
                TelemetrySnapshot.speed,
                TelemetrySnapshot.predicted_temperature,
                TelemetrySnapshot.actual_temperature,
                TelemetrySnapshot.anomaly_score,
                TelemetrySnapshot.calculated_degradation_index,
            )
            .where(
                TelemetrySnapshot.session_id == session_id,
                TelemetrySnapshot.car_id.in_((car_a, car_b)),
            )
            .order_by(TelemetrySnapshot.car_id, TelemetrySnapshot.time_sec)
        )
        rows = result.mappings().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No telemetry snapshots found for comparison.",
        )

    payload = await asyncio.to_thread(
        _build_comparison_payload,
        rows,
        car_a,
        car_b,
        session_id,
    )
    if not payload["samples"]:
        raise HTTPException(
            status_code=404,
            detail="Both cars must have telemetry in the requested session.",
        )
    return payload


def _build_comparison_payload(
    rows: list[Mapping[str, Any]],
    car_a: str,
    car_b: str,
    session_id: str,
) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(rows)
    car_a_frame = _comparison_car_frame(frame, car_a)
    car_b_frame = _comparison_car_frame(frame, car_b)

    if car_a_frame.empty or car_b_frame.empty:
        return {
            "session_id": session_id,
            "car_a": car_a,
            "car_b": car_b,
            "samples": [],
        }

    aligned = pd.merge_asof(
        car_a_frame,
        car_b_frame,
        on="time_sec",
        direction="nearest",
        tolerance=0.25,
        suffixes=("_a", "_b"),
    ).dropna(subset=["speed_b"])

    aligned["speed_delta"] = aligned["speed_a"] - aligned["speed_b"]
    aligned["predicted_temp_delta"] = (
        aligned["predicted_temperature_a"] - aligned["predicted_temperature_b"]
    )
    aligned["actual_temp_delta"] = (
        aligned["actual_temperature_a"] - aligned["actual_temperature_b"]
    )
    aligned["anomaly_score_delta"] = (
        aligned["anomaly_score_a"] - aligned["anomaly_score_b"]
    )
    aligned["degradation_index_delta"] = (
        aligned["calculated_degradation_index_a"].fillna(0.0)
        - aligned["calculated_degradation_index_b"].fillna(0.0)
    )

    samples = [
        {
            "time_sec": float(row.time_sec),
            "car_a": {
                "car_id": car_a,
                "speed": float(row.speed_a),
                "predicted_temperature": float(row.predicted_temperature_a),
                "actual_temperature": float(row.actual_temperature_a),
                "anomaly_score": float(row.anomaly_score_a),
                "degradation_index": _nullable_float(
                    row.calculated_degradation_index_a
                ),
            },
            "car_b": {
                "car_id": car_b,
                "speed": float(row.speed_b),
                "predicted_temperature": float(row.predicted_temperature_b),
                "actual_temperature": float(row.actual_temperature_b),
                "anomaly_score": float(row.anomaly_score_b),
                "degradation_index": _nullable_float(
                    row.calculated_degradation_index_b
                ),
            },
            "delta": {
                "speed": float(row.speed_delta),
                "predicted_temperature": float(row.predicted_temp_delta),
                "actual_temperature": float(row.actual_temp_delta),
                "anomaly_score": float(row.anomaly_score_delta),
                "degradation_index": float(row.degradation_index_delta),
            },
        }
        for row in aligned.itertuples(index=False)
    ]

    return {
        "session_id": session_id,
        "car_a": car_a,
        "car_b": car_b,
        "alignment": {
            "method": "merge_asof",
            "direction": "nearest",
            "tolerance_seconds": 0.25,
        },
        "samples": samples,
    }


def _comparison_car_frame(frame: pd.DataFrame, car_id: str) -> pd.DataFrame:
    car_frame = frame[frame["car_id"] == car_id].copy()
    if car_frame.empty:
        return car_frame
    return car_frame.sort_values("time_sec").reset_index(drop=True)


def _nullable_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    worker = getattr(websocket.app.state, "worker", None)
    if worker is None:
        await websocket.close(code=1013)
        return
    worker = cast(InferenceWorker, worker)
    last_payload: str | None = None

    try:
        while True:
            latest = await worker.latest_result()
            if latest is not None:
                serialized = json.dumps(latest, sort_keys=True)
                if serialized != last_payload:
                    await websocket.send_json(latest)
                    last_payload = serialized
            await asyncio.sleep(WEBSOCKET_POLL_SECONDS)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        LOGGER.exception("WebSocket telemetry stream failed")
        await websocket.close(code=1011)
