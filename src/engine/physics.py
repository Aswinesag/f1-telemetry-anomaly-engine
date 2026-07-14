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
    tire_decay_window_samples: int = 25
    tire_slip_threshold: float = 0.08
    tire_load_factor_threshold: float = 1.15
    tire_stress_ema_alpha: float = 0.08
    tire_stress_scale: float = 10_000.0

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
        frame = self.calculate_tire_dynamics(frame, copy=False)

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

    def calculate_tire_dynamics(
        self,
        telemetry: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        frame = telemetry.copy(deep=True) if copy else telemetry
        if "Aero_Downforce_N" not in frame.columns:
            frame = self.compute_aerodynamics(frame, copy=False)
        if self._config.tire_decay_window_samples <= 0:
            raise ValueError("tire_decay_window_samples must be greater than zero")

        vehicle_speed_ms = self._get_vehicle_speed_ms(frame)
        wheel_speed_ms = self._get_wheel_speed_ms(frame, vehicle_speed_ms)
        static_weight_n = self._config.vehicle_mass_kg * self._config.gravity_m_s2
        normal_load_n = static_weight_n + frame["Aero_Downforce_N"].astype("float64")

        slip_ratio = np.divide(
            wheel_speed_ms - vehicle_speed_ms,
            vehicle_speed_ms,
            out=np.zeros_like(vehicle_speed_ms, dtype="float64"),
            where=np.abs(vehicle_speed_ms) > 1e-6,
        )
        abs_slip_ratio = np.abs(slip_ratio)
        tire_load_factor = normal_load_n.div(static_weight_n)
        tire_stress_index = (
            abs_slip_ratio
            * tire_load_factor
            * normal_load_n.div(self._config.tire_stress_scale)
        )
        high_slip_load_event = (
            (abs_slip_ratio >= self._config.tire_slip_threshold)
            & (tire_load_factor >= self._config.tire_load_factor_threshold)
        )
        weighted_decay_event = tire_stress_index.where(
            high_slip_load_event,
            other=0.0,
        )
        rolling_decay = weighted_decay_event.rolling(
            window=self._config.tire_decay_window_samples,
            min_periods=1,
        ).mean()
        degradation_index = rolling_decay.ewm(
            alpha=self._config.tire_stress_ema_alpha,
            adjust=False,
        ).mean()

        frame["Wheel_Speed_ms"] = wheel_speed_ms
        frame["Slip_Ratio"] = slip_ratio
        frame["Normal_Load_N"] = normal_load_n
        frame["Tire_Load_Factor"] = tire_load_factor
        frame["Tire_Stress_Index"] = tire_stress_index
        frame["Thermal_Decay_Indicator"] = rolling_decay
        frame["Calculated_Degradation_Index"] = degradation_index
        frame["Sustained_Tire_Decay_Flag"] = (
            rolling_decay > 0.0
        ).astype("bool")
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

    @staticmethod
    def _get_vehicle_speed_ms(telemetry: pd.DataFrame) -> pd.Series:
        if "Speed_ms" in telemetry.columns:
            return telemetry["Speed_ms"].astype("float64")
        return telemetry["Speed"].astype("float64").div(3.6)

    @staticmethod
    def _get_wheel_speed_ms(
        telemetry: pd.DataFrame,
        vehicle_speed_ms: pd.Series,
    ) -> pd.Series:
        if "WheelSpeed" in telemetry.columns:
            return telemetry["WheelSpeed"].astype("float64").div(3.6)
        if "Wheel_Speed" in telemetry.columns:
            return telemetry["Wheel_Speed"].astype("float64").div(3.6)

        wheel_speed_columns = [
            column
            for column in ("WheelSpeedFL", "WheelSpeedFR", "WheelSpeedRL", "WheelSpeedRR")
            if column in telemetry.columns
        ]
        if wheel_speed_columns:
            return telemetry[wheel_speed_columns].astype("float64").mean(axis=1).div(3.6)

        return vehicle_speed_ms.copy(deep=True)
