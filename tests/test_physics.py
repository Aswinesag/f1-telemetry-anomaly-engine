import numpy as np
import pandas as pd
import pytest

from src.engine.physics import PhysicsConfig, PhysicsEngine


def test_transform_is_vectorized_and_does_not_mutate_input() -> None:
    telemetry = pd.DataFrame(
        {
            "Speed": [0.0, 100.0, 200.0],
            "Brake": [0.0, 1.0, 1.0],
        }
    )
    original = telemetry.copy(deep=True)

    result = PhysicsEngine().transform(telemetry)

    pd.testing.assert_frame_equal(telemetry, original)
    assert {
        "Aero_Drag_N",
        "Brake_Work_EMA",
        "Brake_Temp_Target",
        "Convective_Cooling_Factor",
    }.issubset(result.columns)
    assert np.isfinite(result["Brake_Temp_Target"]).all()


def test_transform_rejects_missing_required_columns() -> None:
    telemetry = pd.DataFrame({"Speed": [100.0]})

    with pytest.raises(ValueError, match="Brake"):
        PhysicsEngine().transform(telemetry)


def test_sample_rate_must_be_positive() -> None:
    engine = PhysicsEngine(PhysicsConfig(sample_rate_hz=0.0))
    telemetry = pd.DataFrame({"Speed": [100.0], "Brake": [1.0]})

    with pytest.raises(ValueError, match="sample_rate_hz"):
        engine.transform(telemetry)
