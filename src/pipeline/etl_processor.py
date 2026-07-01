import yaml
import numpy as np
import pandas as pd
import fastf1 as ff1
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler

from src.engine.physics import PhysicsConfig, PhysicsEngine

class F1TelemetryProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.hz = self.config["system"]["target_frequency_hz"]
        self.time_delta_step = 1.0 / self.hz
        
        ff1.Cache.enable_cache(self.config["system"]["cache_directory"])
        self.scaler = MinMaxScaler()
        self.physics_engine = PhysicsEngine(
            PhysicsConfig(sample_rate_hz=float(self.hz))
        )

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
        
        return self.physics_engine.transform(processed_df, include_target=True)

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_df = df.copy(deep=True)
        scaling_features = self.config["features"]["raw_channels"] + self.config["features"]["physics_engineered"]
        scaled_df[scaling_features] = self.scaler.fit_transform(
            scaled_df[scaling_features]
        )
        return scaled_df
