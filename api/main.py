from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from api.database.models import (
    AsyncSessionFactory,
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
    return cast(InferenceWorker, request.app.state.worker)


def get_redis(request: Request) -> Redis:
    return cast(Redis, request.app.state.redis)


@app.get("/health", response_model=ServiceHealth)
async def health(request: Request) -> ServiceHealth:
    worker = get_worker(request)
    return ServiceHealth(
        status="ok",
        service="inference-api",
        device=worker.device,
    )


@app.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    worker = get_worker(request)
    redis_client = get_redis(request)
    kafka_task = cast(asyncio.Task[None], request.app.state.kafka_task)

    redis_ready = bool(await redis_client.ping())
    database_ready = await check_database_ready()
    kafka_ready = not kafka_task.done()
    worker_ready = worker.is_running
    is_ready = redis_ready and database_ready and kafka_ready and worker_ready

    payload: dict[str, Any] = {
        "status": "ready" if is_ready else "not_ready",
        "checks": {
            "database": database_ready,
            "kafka": kafka_ready,
            "redis": redis_ready,
            "worker": worker_ready,
        },
        "queue_depth": worker.queue_depth,
    }
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


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    worker = cast(InferenceWorker, websocket.app.state.worker)
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
