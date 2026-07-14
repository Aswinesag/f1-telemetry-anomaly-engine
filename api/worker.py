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
from src.ml.xai_engine import FeatureImportance, XAIEngine
from src.models.autoencoder import AnomalyAutoencoder
from src.models.virtual_sensor import HybridVirtualSensor


LOGGER = logging.getLogger(__name__)


class ExplanationResult(TypedDict):
    top_factor: str
    importance_score: float
    fault_type: str
    recommendation: str
    feature_importance: dict[str, float]


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
    anomaly_score: float
    tire_compound: str | None
    stint_lap_number: int | None
    degradation_index: float | None
    degradation_trend: float | None
    fault_type: str
    explanation: ExplanationResult | None


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    virtual_sensor: HybridVirtualSensor
    autoencoder: AnomalyAutoencoder
    scaler: MinMaxScaler
    alert_threshold: float
    autoencoder_input_dim: int
    degradation_slope_threshold: float
    degradation_index_threshold: float
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
        self._xai_trigger_threshold = float(
            os.getenv("XAI_TRIGGER_THRESHOLD", str(artifacts.alert_threshold))
        )
        self._xai_engine = self._build_xai_engine()

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
        degradation_index = self._optional_float(
            latest_engineered.get("Calculated_Degradation_Index")
        )
        degradation_trend = self._calculate_degradation_trend(engineered_frame)
        fault_type = self._classify_fault_type(
            anomaly_score=anomaly_score,
            degradation_index=degradation_index,
            degradation_trend=degradation_trend,
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
            "anomaly_score": anomaly_score,
            "tire_compound": self._extract_tire_compound(latest_raw),
            "stint_lap_number": self._extract_stint_lap_number(latest_raw),
            "degradation_index": degradation_index,
            "degradation_trend": degradation_trend,
            "fault_type": fault_type,
            "explanation": self._get_explanation(
                anomaly_score=anomaly_score,
                fault_type=fault_type,
                degradation_index=degradation_index,
                degradation_trend=degradation_trend,
                engineered_frame=engineered_frame,
                residual_tensor=residual_tensor,
                scaled_frame=scaled_frame,
            ),
        }
        return result, latest_raw, latest_engineered

    def _build_xai_engine(self) -> XAIEngine:
        if self._artifacts.autoencoder_input_dim == len(
            self._artifacts.feature_columns
        ):
            feature_names = self._artifacts.feature_columns
        elif self._artifacts.autoencoder_input_dim == 1:
            feature_names = ("Thermal_Residual",)
        else:
            feature_names = tuple(
                f"Autoencoder_Input_{index}"
                for index in range(self._artifacts.autoencoder_input_dim)
            )

        return XAIEngine(
            self._artifacts.autoencoder,
            feature_names=feature_names,
            n_steps=int(os.getenv("XAI_IG_STEPS", "32")),
        )

    def _get_explanation(
        self,
        *,
        anomaly_score: float,
        fault_type: str,
        degradation_index: float | None,
        degradation_trend: float | None,
        engineered_frame: pd.DataFrame,
        residual_tensor: torch.Tensor,
        scaled_frame: pd.DataFrame,
    ) -> ExplanationResult | None:
        if fault_type == "Tire Degradation":
            return self._summarize_tire_degradation(
                engineered_frame=engineered_frame,
                degradation_index=degradation_index,
                degradation_trend=degradation_trend,
            )

        if anomaly_score < 0.5 or anomaly_score < self._xai_trigger_threshold:
            return None

        try:
            attribution_input = self._build_attribution_input(
                residual_tensor=residual_tensor,
                scaled_frame=scaled_frame,
            )
            feature_importance = self._xai_engine.get_feature_importance(
                attribution_input
            )
        except Exception:
            LOGGER.exception("XAI feature attribution failed")
            return None

        return self._summarize_feature_importance(
            feature_importance,
            fault_type=fault_type,
        )

    def _build_attribution_input(
        self,
        *,
        residual_tensor: torch.Tensor,
        scaled_frame: pd.DataFrame,
    ) -> torch.Tensor:
        if self._artifacts.autoencoder_input_dim == len(
            self._artifacts.feature_columns
        ):
            return torch.as_tensor(
                scaled_frame.iloc[-1][list(self._artifacts.feature_columns)].to_numpy(
                    dtype=np.float32,
                    copy=True,
                ),
                dtype=torch.float32,
                device=self._artifacts.device,
            ).unsqueeze(0)

        return residual_tensor

    @staticmethod
    def _summarize_feature_importance(
        feature_importance: FeatureImportance,
        *,
        fault_type: str,
    ) -> ExplanationResult | None:
        if not feature_importance:
            return None

        top_factor, importance_score = max(
            feature_importance.items(),
            key=lambda item: item[1],
        )
        return {
            "top_factor": top_factor,
            "importance_score": float(importance_score),
            "fault_type": fault_type,
            "recommendation": (
                "Inspect brake, cooling, and mechanical telemetry for a sudden "
                "fault signature."
            ),
            "feature_importance": {
                feature_name: float(score)
                for feature_name, score in feature_importance.items()
            },
        }

    def _classify_fault_type(
        self,
        *,
        anomaly_score: float,
        degradation_index: float | None,
        degradation_trend: float | None,
    ) -> str:
        degradation_shift = (
            degradation_trend is not None
            and degradation_trend >= self._artifacts.degradation_slope_threshold
        )
        degradation_state = (
            degradation_index is not None
            and degradation_index >= self._artifacts.degradation_index_threshold
        )

        if degradation_shift or degradation_state:
            return "Tire Degradation"
        if anomaly_score > self._artifacts.alert_threshold:
            return "Mechanical Fault"
        return "Nominal"

    def _calculate_degradation_trend(
        self,
        engineered_frame: pd.DataFrame,
    ) -> float | None:
        if "Calculated_Degradation_Index" not in engineered_frame.columns:
            return None

        degradation_series = (
            engineered_frame["Calculated_Degradation_Index"]
            .tail(self._artifacts.sequence_length)
            .astype("float64")
            .dropna()
        )
        if len(degradation_series) < 2:
            return None

        sample_index = np.arange(len(degradation_series), dtype=np.float64)
        slope = np.polyfit(sample_index, degradation_series.to_numpy(), deg=1)[0]
        return float(slope)

    def _summarize_tire_degradation(
        self,
        *,
        engineered_frame: pd.DataFrame,
        degradation_index: float | None,
        degradation_trend: float | None,
    ) -> ExplanationResult:
        candidate_features = (
            "Calculated_Degradation_Index",
            "Thermal_Decay_Indicator",
            "Tire_Stress_Index",
            "Slip_Ratio",
            "Normal_Load_N",
        )
        scores: dict[str, float] = {}
        for feature_name in candidate_features:
            if feature_name not in engineered_frame.columns:
                continue
            series = engineered_frame[feature_name].tail(
                self._artifacts.sequence_length
            ).astype("float64")
            latest_value = abs(float(series.iloc[-1])) if not series.empty else 0.0
            trend_value = abs(self._series_slope(series))
            scores[feature_name] = latest_value + trend_value

        if not scores:
            scores = {"Calculated_Degradation_Index": abs(degradation_index or 0.0)}

        total_score = sum(scores.values())
        if total_score > 0.0:
            feature_importance = {
                feature_name: score / total_score
                for feature_name, score in scores.items()
            }
        else:
            equal_score = 1.0 / len(scores)
            feature_importance = {
                feature_name: equal_score
                for feature_name in scores
            }

        top_factor, importance_score = max(
            feature_importance.items(),
            key=lambda item: item[1],
        )
        trend = degradation_trend or 0.0
        return {
            "top_factor": top_factor,
            "importance_score": float(importance_score),
            "fault_type": "Tire Degradation",
            "recommendation": (
                "Manage slip and sustained tire load; reduce traction-zone "
                f"stress. Degradation trend slope: {trend:.6f}."
            ),
            "feature_importance": {
                feature_name: float(score)
                for feature_name, score in feature_importance.items()
            },
        }

    @staticmethod
    def _series_slope(series: pd.Series) -> float:
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0
        sample_index = np.arange(len(clean_series), dtype=np.float64)
        return float(np.polyfit(sample_index, clean_series.to_numpy(), deg=1)[0])

    @staticmethod
    def _extract_tire_compound(snapshot: Mapping[str, Any]) -> str | None:
        for key in ("tire_compound", "TireCompound", "Compound"):
            value = snapshot.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    @staticmethod
    def _extract_stint_lap_number(snapshot: Mapping[str, Any]) -> int | None:
        for key in ("stint_lap_number", "StintLapNumber", "LapNumber", "Lap"):
            value = snapshot.get(key)
            if value not in (None, ""):
                return int(value)
        return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

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
            autoencoder_input_dim=int(anomaly_payload["input_dim"]),
            degradation_slope_threshold=float(
                config["anomaly_detection"].get(
                    "degradation_slope_threshold",
                    0.0025,
                )
            ),
            degradation_index_threshold=float(
                config["anomaly_detection"].get(
                    "degradation_index_threshold",
                    0.15,
                )
            ),
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
