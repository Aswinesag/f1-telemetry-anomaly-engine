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
        """Injects physics-informed constraints directly inside data matrices [3.2]."""
        # Convert Speed metrics from km/h parameters down to SI units (m/s)
        df["Speed_ms"] = df["Speed"] / 3.6
        
        # Kinetic Energy Dissipation Proxy Evaluation Matrix [3.2]
        df["Delta_KE"] = df["Speed_ms"].pow(2).diff().fillna(0.0)
        
        # Exponential Mechanical Work Estimation Module [3.2]
        instantaneous_work = df["Brake"] * df["Speed_ms"]
        df["Brake_Work_EMA"] = instantaneous_work.ewm(alpha=0.05, adjust=False).mean()
        
        # Establish structural synthetic training target matching volatile real-world behavior
        df["Brake_Temp_Target"] = 180.0 + (df["Brake_Work_EMA"] * 1.8) - (df["Delta_KE"] * 0.4)
        
        # Dynamic Scaling Sequence Preparation
        scaling_features = self.config["features"]["raw_channels"] + self.config["features"]["physics_engineered"]
        df[scaling_features] = self.scaler.fit_transform(df[scaling_features])
        
        return df