import numpy as np
import pandas as pd

from src.pipeline.etl_processor import F1TelemetryProcessor


def test_align_driver_to_grid_preserves_uniform_time_grid() -> None:
    processor = F1TelemetryProcessor()
    raw_channels = processor.config["features"]["raw_channels"]
    time_index = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    cleaned_df = pd.DataFrame(
        {
            channel: np.linspace(1.0, 4.0, num=len(time_index))
            for channel in raw_channels
        },
        index=time_index,
    )
    cleaned_df.index.name = "TimeSec"
    uniform_time_grid = np.array([0.25, 0.75, 1.25], dtype=np.float64)

    aligned_df = processor._align_driver_to_grid(
        cleaned_df=cleaned_df,
        uniform_time_grid=uniform_time_grid,
    )

    np.testing.assert_allclose(
        aligned_df["TimeSec"].to_numpy(),
        uniform_time_grid,
    )
    assert set(raw_channels).issubset(aligned_df.columns)


def test_synchronize_aligned_frames_keeps_common_time_values_only() -> None:
    driver_a = pd.DataFrame(
        {
            "TimeSec": [1.0, 2.0, 3.0],
            "Speed": [100.0, 101.0, 102.0],
        }
    )
    driver_b = pd.DataFrame(
        {
            "TimeSec": [2.0, 3.0, 4.0],
            "Speed": [99.0, 100.0, 101.0],
        }
    )

    synchronized = F1TelemetryProcessor._synchronize_aligned_frames(
        {"VER": driver_a, "HAM": driver_b}
    )

    assert set(synchronized) == {"VER", "HAM"}
    np.testing.assert_allclose(
        synchronized["VER"]["TimeSec"].to_numpy(),
        np.array([2.0, 3.0]),
    )
    np.testing.assert_allclose(
        synchronized["HAM"]["TimeSec"].to_numpy(),
        np.array([2.0, 3.0]),
    )
