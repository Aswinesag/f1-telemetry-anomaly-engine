from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    sample_rate_hz: float = 50.0
    vehicle_mass_kg: float = 798.0
    air_density_kg_m3: float = 1.225
    drag_area_coefficient: float = 1.15
    downforce_area_coefficient: float = 3.5
    gravity_m_s2: float = 9.81
    brake_work_scale: float = 10_000.0
    brake_work_ema_alpha: float = 0.05
    cooling_coefficient: float = 0.05
    cooling_exponent: float = 0.8
    base_brake_temperature_c: float = 180.0
    heat_gain_coefficient: float = 2.2
    heat_loss_coefficient: float = 1.5

    @property
    def sample_interval_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be greater than zero")
        return 1.0 / self.sample_rate_hz


class PhysicsEngine:
    REQUIRED_COLUMNS = frozenset({"Speed", "Brake"})

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        self._config = config or PhysicsConfig()

    @property
    def config(self) -> PhysicsConfig:
        return self._config

    def transform(
        self,
        telemetry: pd.DataFrame,
        *,
        include_target: bool = True,
        copy: bool = True,
    ) -> pd.DataFrame:
        self._validate_columns(telemetry)
        frame = telemetry.copy(deep=True) if copy else telemetry
        frame = self.compute_kinematics(frame, copy=False)
        frame = self.compute_aerodynamics(frame, copy=False)
        frame = self.compute_thermodynamics(frame, copy=False)

        if include_target:
            frame = self.synthesize_brake_temperature(frame, copy=False)

        return frame

    def compute_kinematics(
        self,
        telemetry: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        self._validate_columns(telemetry)
        frame = telemetry.copy(deep=True) if copy else telemetry
        speed_ms = frame["Speed"].astype("float64").div(3.6)
        speed_squared = np.square(speed_ms)

        frame["Speed_ms"] = speed_ms
        frame["Delta_KE"] = speed_squared.diff().fillna(0.0)
        frame["Acceleration"] = speed_ms.diff().fillna(0.0).div(
            self._config.sample_interval_seconds
        )
        frame["Longitudinal_G"] = frame["Acceleration"].div(
            self._config.gravity_m_s2
        )
        return frame

    def compute_aerodynamics(
        self,
        telemetry: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        frame = telemetry.copy(deep=True) if copy else telemetry
        if "Speed_ms" not in frame.columns:
            frame = self.compute_kinematics(frame, copy=False)

        dynamic_pressure = (
            0.5
            * self._config.air_density_kg_m3
            * np.square(frame["Speed_ms"].astype("float64"))
        )
        frame["Aero_Drag_N"] = (
            dynamic_pressure * self._config.drag_area_coefficient
        )
        frame["Aero_Downforce_N"] = (
            dynamic_pressure * self._config.downforce_area_coefficient
        )
        frame["Effective_Weight_N"] = (
            self._config.vehicle_mass_kg * self._config.gravity_m_s2
            + frame["Aero_Downforce_N"]
        )
        return frame

    def compute_thermodynamics(
        self,
        telemetry: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        frame = telemetry.copy(deep=True) if copy else telemetry
        if "Effective_Weight_N" not in frame.columns:
            frame = self.compute_aerodynamics(frame, copy=False)

        brake_application = frame["Brake"].astype("float64")
        instantaneous_brake_work = (
            brake_application
            * frame["Speed_ms"]
            * frame["Effective_Weight_N"].div(self._config.brake_work_scale)
        )
        frame["Brake_Work_EMA"] = instantaneous_brake_work.ewm(
            alpha=self._config.brake_work_ema_alpha,
            adjust=False,
        ).mean()
        frame["Convective_Cooling_Factor"] = (
            np.power(frame["Speed_ms"].clip(lower=0.0), self._config.cooling_exponent)
            * self._config.cooling_coefficient
        )
        return frame

    def synthesize_brake_temperature(
        self,
        telemetry: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        frame = telemetry.copy(deep=True) if copy else telemetry
        if "Brake_Work_EMA" not in frame.columns:
            frame = self.compute_thermodynamics(frame, copy=False)

        frame["Brake_Temp_Target"] = (
            self._config.base_brake_temperature_c
            + frame["Brake_Work_EMA"] * self._config.heat_gain_coefficient
            - frame["Convective_Cooling_Factor"]
            * self._config.heat_loss_coefficient
        )
        return frame

    def _validate_columns(self, telemetry: pd.DataFrame) -> None:
        missing_columns = self.REQUIRED_COLUMNS.difference(telemetry.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Telemetry is missing required columns: {missing}")
