import yaml
import numpy as np
import pandas as pd
import fastf1 as ff1
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler
from fastf1.core import Session

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

    def process_session_telemetry(
        self,
        year: int,
        location: str,
        session_type: str,
        drivers: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Loads F1 timing sheets and extracts aligned telemetry for multiple cars."""
        if not drivers:
            raise ValueError("drivers must contain at least one driver identifier")

        session = ff1.get_session(year, location, session_type)
        session.load(telemetry=True, laps=True, weather=False)
        session_id = f"{year}-{location}-{session_type}".replace(" ", "_")

        cleaned_by_driver: dict[str, pd.DataFrame] = {}
        for driver_id in drivers:
            cleaned_driver_frame = self._extract_driver_fastest_lap(
                session=session,
                driver_id=driver_id,
            )
            if cleaned_driver_frame is not None:
                cleaned_by_driver[driver_id] = cleaned_driver_frame

        if not cleaned_by_driver:
            return {}

        uniform_time_grid = self._build_shared_time_grid(cleaned_by_driver)
        if uniform_time_grid.size == 0:
            return {}

        aligned_by_driver: dict[str, pd.DataFrame] = {}
        for driver_id, cleaned_df in cleaned_by_driver.items():
            aligned_df = self._align_driver_to_grid(
                cleaned_df=cleaned_df,
                uniform_time_grid=uniform_time_grid,
            )
            if aligned_df.empty:
                continue
            aligned_df["session_id"] = session_id
            aligned_df["car_id"] = driver_id
            aligned_by_driver[driver_id] = self._inject_thermodynamic_features(
                aligned_df
            )

        return self._synchronize_aligned_frames(aligned_by_driver)

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_df = df.copy(deep=True)
        scaling_features = self.config["features"]["raw_channels"] + self.config["features"]["physics_engineered"]
        scaled_df[scaling_features] = self.scaler.fit_transform(
            scaled_df[scaling_features]
        )
        return scaled_df

    def _extract_driver_fastest_lap(
        self,
        *,
        session: Session,
        driver_id: str,
    ) -> pd.DataFrame | None:
        try:
            driver_laps = session.laps.pick_driver(driver_id)
            if driver_laps.empty:
                return None

            fastest_lap = driver_laps.pick_fastest()
            if fastest_lap is None or pd.isna(fastest_lap.get("LapTime")):
                return None

            raw_telemetry = fastest_lap.get_telemetry()
            if raw_telemetry.empty:
                return None

            cleaned_df = raw_telemetry.drop(
                columns=["X", "Y", "Z", "Source"],
                errors="ignore",
            ).copy()
            if "Time" not in cleaned_df.columns:
                return None

            cleaned_df["TimeSec"] = cleaned_df["Time"].dt.total_seconds()
            cleaned_df = (
                cleaned_df.dropna(subset=["TimeSec"])
                .sort_values("TimeSec")
                .drop_duplicates(subset=["TimeSec"], keep="last")
                .set_index("TimeSec")
            )

            missing_channels = [
                channel
                for channel in self.config["features"]["raw_channels"]
                if channel not in cleaned_df.columns
            ]
            if missing_channels:
                return None

            return cleaned_df
        except Exception:
            return None

    def _build_shared_time_grid(
        self,
        cleaned_by_driver: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        start_time = max(frame.index.min() for frame in cleaned_by_driver.values())
        end_time = min(frame.index.max() for frame in cleaned_by_driver.values())
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            return np.array([], dtype=np.float64)
        if end_time <= start_time:
            return np.array([], dtype=np.float64)
        return np.arange(start_time, end_time, self.time_delta_step)

    def _align_driver_to_grid(
        self,
        *,
        cleaned_df: pd.DataFrame,
        uniform_time_grid: np.ndarray,
    ) -> pd.DataFrame:
        aligned_data: dict[str, np.ndarray] = {}
        for channel in self.config["features"]["raw_channels"]:
            valid_subset = cleaned_df[channel].astype("float64").dropna()
            valid_subset = valid_subset[~valid_subset.index.duplicated(keep="last")]
            if len(valid_subset) < 2:
                aligned_data[channel] = np.full(
                    shape=uniform_time_grid.shape,
                    fill_value=np.nan,
                    dtype=np.float64,
                )
                continue

            spline_interpolator = CubicSpline(
                valid_subset.index.to_numpy(dtype=np.float64),
                valid_subset.to_numpy(dtype=np.float64),
                extrapolate=False,
            )
            aligned_data[channel] = spline_interpolator(uniform_time_grid)

        processed_df = pd.DataFrame(aligned_data, index=uniform_time_grid)
        processed_df.index.name = "TimeSec"
        processed_df = processed_df.reset_index()
        return processed_df.dropna(
            subset=self.config["features"]["raw_channels"],
            how="any",
        ).reset_index(drop=True)

    def _inject_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.physics_engine.transform(df, include_target=True)

    @staticmethod
    def _synchronize_aligned_frames(
        aligned_by_driver: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        if not aligned_by_driver:
            return {}

        common_time_values: set[float] | None = None
        for frame in aligned_by_driver.values():
            time_values = set(frame["TimeSec"].astype("float64").to_numpy())
            common_time_values = (
                time_values
                if common_time_values is None
                else common_time_values.intersection(time_values)
            )

        if not common_time_values:
            return {}

        common_time_index = np.array(sorted(common_time_values), dtype=np.float64)
        synchronized: dict[str, pd.DataFrame] = {}
        for driver_id, frame in aligned_by_driver.items():
            synchronized_frame = (
                frame[frame["TimeSec"].isin(common_time_index)]
                .sort_values("TimeSec")
                .reset_index(drop=True)
            )
            if not synchronized_frame.empty:
                synchronized[driver_id] = synchronized_frame

        return synchronized
