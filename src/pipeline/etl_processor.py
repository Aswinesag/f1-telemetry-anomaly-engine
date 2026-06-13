import os
import yaml
import numpy as np
import pandas as pd
import fastf1 as ff1
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler

class F1TelemetryProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.hz = self.config["system"]["target_frequency_hz"]
        self.time_delta_step = 1.0 / self.hz
        
        ff1.Cache.enable_cache(self.config["system"]["cache_directory"])
        self.scaler = MinMaxScaler()

    def process_session_telemetry(self, year: int, location: str, session_type: str, driver: str) -> pd.DataFrame:
        """Loads F1 timing sheets and extracts continuous aligned telemetry arrays."""
        session = ff1.get_session(year, location, session_type)
        session.load(telemetry=True, laps=True, weather=False)
        
        fastest_lap = session.laps.pick_driver(driver).pick_fastest()
        raw_telemetry = fastest_lap.get_telemetry()
        
        # Isolate baseline dynamics and clean spatial indexes
        cleaned_df = raw_telemetry.drop(columns=["X", "Y", "Z", "Source"], errors="ignore").copy()
        cleaned_df["TimeSec"] = cleaned_df["Time"].dt.total_seconds()
        cleaned_df = cleaned_df.set_index("TimeSec")
        
        # Continuous Equidistant Temporal Realignment Array Grid Generation [3.1]
        start_time = cleaned_df.index.min()
        end_time = cleaned_df.index.max()
        uniform_time_grid = np.arange(start_time, end_time, self.time_delta_step)
        
        # Executing multi-channel alignment leveraging Cubic Splines [3.1]
        aligned_data = {}
        target_channels = self.config["features"]["raw_channels"]
        
        for channel in target_channels:
            # Drop NaN instances purely to capture stable mathematical boundary splines
            valid_subset = cleaned_df[channel].dropna()
            spline_interpolator = CubicSpline(valid_subset.index, valid_subset.values, extrapolate=False)
            aligned_data[channel] = spline_interpolator(uniform_time_grid)
            
        processed_df = pd.DataFrame(aligned_data, index=uniform_time_grid)
        processed_df.index.name = "TimeSec"
        processed_df = processed_df.reset_index()
        
        return self._inject_thermodynamic_features(processed_df)

    def _inject_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Injects advanced aero and thermodynamic physics constraints."""
        # 1. Base Kinematics
        df["Speed_ms"] = df["Speed"] / 3.6
        df["Delta_KE"] = df["Speed_ms"].pow(2).diff().fillna(0.0)
        
        # Calculate acceleration using the configured time delta (e.g., 0.02s for 50Hz)
        df["Acceleration"] = df["Speed_ms"].diff().fillna(0.0) / self.time_delta_step
        df["Longitudinal_G"] = df["Acceleration"] / 9.81
        
        # 2. Aerodynamic Profiling (F1 Constants approximation)
        df["Aero_Drag_N"] = 0.5 * 1.225 * (df["Speed_ms"] ** 2) * 1.15
        df["Aero_Downforce_N"] = 0.5 * 1.225 * (df["Speed_ms"] ** 2) * 3.5
        
        # 3. Dynamic Vehicle Weight (Static Mass ~798kg + Aero Load)
        df["Effective_Weight_N"] = (798 * 9.81) + df["Aero_Downforce_N"]
        
        # 4. Advanced Thermodynamics (Generation vs. Dissipation)
        instantaneous_brake_work = df["Brake"] * df["Speed_ms"] * (df["Effective_Weight_N"] / 10000)
        df["Brake_Work_EMA"] = instantaneous_brake_work.ewm(alpha=0.05, adjust=False).mean()
        
        # Convective cooling from airspeed through ducts
        df["Convective_Cooling_Factor"] = (df["Speed_ms"] ** 0.8) * 0.05
        
        # 5. Synthesize the Ground Truth Target
        base_temp = 180.0
        heat_added = df["Brake_Work_EMA"] * 2.2
        heat_extracted = df["Convective_Cooling_Factor"] * 1.5
        df["Brake_Temp_Target"] = base_temp + heat_added - heat_extracted
        
        # Dynamic Scaling Sequence Preparation
        scaling_features = self.config["features"]["raw_channels"] + self.config["features"]["physics_engineered"]
        df[scaling_features] = self.scaler.fit_transform(df[scaling_features])
        
        return df