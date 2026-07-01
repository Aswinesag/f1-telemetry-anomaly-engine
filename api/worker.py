from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import torch
import yaml
from redis.asyncio import Redis
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.database.models import TelemetrySnapshot
from src.engine.physics import PhysicsConfig, PhysicsEngine
from src.models.autoencoder import AnomalyAutoencoder
from src.models.virtual_sensor import HybridVirtualSensor


LOGGER = logging.getLogger(__name__)


class InferenceResult(TypedDict):
    CapturedAt: str
    TimeSec: float
    Speed: float
    Brake: float
    Predicted_Temp: float
    Actual_Temp: float
    Anomaly_Score: float
    Alert_Threshold: float
    Is_Anomaly: bool


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    virtual_sensor: HybridVirtualSensor
    autoencoder: AnomalyAutoencoder
    scaler: MinMaxScaler
    alert_threshold: float
    sequence_length: int
    feature_columns: tuple[str, ...]
    scaling_columns: tuple[str, ...]
    device: torch.device


class InferenceWorker:
    LATEST_RESULT_KEY = "telemetry:inference:latest"

    def __init__(
        self,
        *,
        artifacts: ModelArtifacts,
        physics_engine: PhysicsEngine,
        redis_client: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        queue_max_size: int = 2_000,
        model_version: str = "unknown",
    ) -> None:
        self._artifacts = artifacts
        self._physics_engine = physics_engine
        self._redis = redis_client
        self._session_factory = session_factory
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=queue_max_size
        )
        self._window: deque[dict[str, Any]] = deque(
            maxlen=artifacts.sequence_length
        )
        self._runner_task: asyncio.Task[None] | None = None
        self._model_version = model_version

    @classmethod
    async def create(
        cls,
        *,
        redis_client: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        config_path: str = "config/config.yaml",
        sensor_model_path: str = "data/virtual_sensor.pt",
        anomaly_model_path: str = "data/isolation_engine.pt",
        queue_max_size: int = 2_000,
    ) -> "InferenceWorker":
        artifacts, sample_rate_hz = await asyncio.to_thread(
            cls._load_artifacts,
            Path(config_path),
            Path(sensor_model_path),
            Path(anomaly_model_path),
        )
        return cls(
            artifacts=artifacts,
            physics_engine=PhysicsEngine(
                PhysicsConfig(sample_rate_hz=sample_rate_hz)
            ),
            redis_client=redis_client,
            session_factory=session_factory,
            queue_max_size=queue_max_size,
            model_version=os.getenv("MODEL_VERSION", "unknown"),
        )

    @property
    def device(self) -> str:
        return str(self._artifacts.device)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._runner_task is not None and not self._runner_task.done()

    async def start(self) -> None:
        if self.is_running:
            return
        self._runner_task = asyncio.create_task(
            self._run(),
            name="inference-worker",
        )

    async def stop(self) -> None:
        if self._runner_task is None:
            return
        await self._queue.put(None)
        await self._runner_task
        self._runner_task = None

    async def enqueue(self, telemetry: Mapping[str, Any]) -> int:
        await self._queue.put(dict(telemetry))
        return self.queue_depth

    async def latest_result(self) -> InferenceResult | None:
        payload = await self._redis.get(self.LATEST_RESULT_KEY)
        if payload is None:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload)

    async def _run(self) -> None:
        while True:
            telemetry = await self._queue.get()
            try:
                if telemetry is None:
                    return

                self._window.append(telemetry)
                if len(self._window) < self._artifacts.sequence_length:
                    continue

                result, raw_snapshot, engineered_snapshot = await asyncio.to_thread(
                    self._infer_sync,
                    tuple(self._window),
                )
                await self._redis.set(
                    self.LATEST_RESULT_KEY,
                    json.dumps(result, separators=(",", ":")),
                )
                await self._persist_result(
                    result=result,
                    raw_snapshot=raw_snapshot,
                    engineered_snapshot=engineered_snapshot,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Inference processing failed")
            finally:
                self._queue.task_done()

    def _infer_sync(
        self,
        window: tuple[dict[str, Any], ...],
    ) -> tuple[InferenceResult, dict[str, Any], dict[str, Any]]:
        raw_frame = pd.DataFrame.from_records(window)
        engineered_frame = self._physics_engine.transform(
            raw_frame,
            include_target=True,
        )
        scaled_frame = engineered_frame.copy(deep=True)
        scaled_frame[list(self._artifacts.scaling_columns)] = (
            self._artifacts.scaler.transform(
                engineered_frame[list(self._artifacts.scaling_columns)]
            )
        )

        input_tensor = torch.as_tensor(
            scaled_frame[list(self._artifacts.feature_columns)].to_numpy(
                dtype=np.float32,
                copy=True,
            ),
            dtype=torch.float32,
            device=self._artifacts.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            predicted_temperature = float(
                self._artifacts.virtual_sensor(input_tensor).item()
            )
            actual_temperature = float(
                engineered_frame.iloc[-1]["Brake_Temp_Target"]
            )
            residual_tensor = torch.tensor(
                [[abs(actual_temperature - predicted_temperature)]],
                dtype=torch.float32,
                device=self._artifacts.device,
            )
            anomaly_score_tensor, _ = (
                self._artifacts.autoencoder.calculate_reconstruction_loss(
                    residual_tensor
                )
            )
            anomaly_score = float(anomaly_score_tensor.item())

        latest_raw = self._to_json_mapping(raw_frame.iloc[-1].to_dict())
        latest_engineered = self._to_json_mapping(
            engineered_frame.iloc[-1].to_dict()
        )
        result: InferenceResult = {
            "CapturedAt": datetime.now(UTC).isoformat(),
            "TimeSec": float(latest_raw["TimeSec"]),
            "Speed": float(latest_raw["Speed"]),
            "Brake": float(latest_raw["Brake"]),
            "Predicted_Temp": predicted_temperature,
            "Actual_Temp": actual_temperature,
            "Anomaly_Score": anomaly_score,
            "Alert_Threshold": self._artifacts.alert_threshold,
            "Is_Anomaly": anomaly_score > self._artifacts.alert_threshold,
        }
        return result, latest_raw, latest_engineered

    async def _persist_result(
        self,
        *,
        result: InferenceResult,
        raw_snapshot: Mapping[str, Any],
        engineered_snapshot: Mapping[str, Any],
    ) -> None:
        snapshot = TelemetrySnapshot.from_inference(
            raw_snapshot=raw_snapshot,
            engineered_snapshot=engineered_snapshot,
            result=result,
            model_version=self._model_version,
        )
        async with self._session_factory.begin() as session:
            session.add(snapshot)

    @staticmethod
    def _load_artifacts(
        config_path: Path,
        sensor_model_path: Path,
        anomaly_model_path: Path,
    ) -> tuple[ModelArtifacts, float]:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sensor_payload = torch.load(
            sensor_model_path,
            map_location=device,
            weights_only=False,
        )
        virtual_sensor = HybridVirtualSensor(
            input_dim=int(sensor_payload["input_dim"]),
            hidden_dim=int(sensor_payload["hidden_dim"]),
            sequence_length=int(sensor_payload["sequence_length"]),
        )
        virtual_sensor.load_state_dict(sensor_payload["state_dict"])
        virtual_sensor.to(device).eval()

        anomaly_payload = torch.load(
            anomaly_model_path,
            map_location=device,
            weights_only=False,
        )
        autoencoder = AnomalyAutoencoder(
            input_dim=int(anomaly_payload["input_dim"])
        )
        autoencoder.load_state_dict(anomaly_payload["state_dict"])
        autoencoder.to(device).eval()

        raw_columns = tuple(config["features"]["raw_channels"])
        physics_columns = tuple(config["features"]["physics_engineered"])
        scaler = sensor_payload["scalar_metadata"]["scaler"]
        artifacts = ModelArtifacts(
            virtual_sensor=virtual_sensor,
            autoencoder=autoencoder,
            scaler=scaler,
            alert_threshold=float(anomaly_payload["alert_threshold"]),
            sequence_length=int(config["model_hyperparameters"]["sequence_length"]),
            feature_columns=raw_columns + physics_columns,
            scaling_columns=raw_columns + physics_columns,
            device=device,
        )
        return artifacts, float(config["system"]["target_frequency_hz"])

    @staticmethod
    def _to_json_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values.items():
            if pd.isna(value):
                result[str(key)] = None
            elif isinstance(value, np.generic):
                result[str(key)] = value.item()
            elif isinstance(value, pd.Timestamp):
                result[str(key)] = value.isoformat()
            else:
                result[str(key)] = value
        return result
