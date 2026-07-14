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
        "Slip_Ratio",
        "Normal_Load_N",
        "Thermal_Decay_Indicator",
        "Calculated_Degradation_Index",
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


def test_tire_dynamics_compute_slip_load_and_decay_indicator() -> None:
    telemetry = pd.DataFrame(
        {
            "Speed": [100.0, 100.0, 100.0, 100.0],
            "WheelSpeed": [100.0, 112.0, 115.0, 118.0],
            "Brake": [0.0, 0.0, 0.0, 0.0],
        }
    )
    engine = PhysicsEngine(
        PhysicsConfig(
            tire_decay_window_samples=2,
            tire_slip_threshold=0.08,
            tire_load_factor_threshold=1.0,
        )
    )

    result = engine.transform(telemetry)

    np.testing.assert_allclose(
        result["Slip_Ratio"].to_numpy(),
        np.array([0.0, 0.12, 0.15, 0.18]),
        rtol=1e-6,
        atol=1e-6,
    )
    expected_static_weight = engine.config.vehicle_mass_kg * engine.config.gravity_m_s2
    assert (result["Normal_Load_N"] >= expected_static_weight).all()
    assert result["Thermal_Decay_Indicator"].iloc[0] == pytest.approx(0.0)
    assert result["Thermal_Decay_Indicator"].iloc[-1] > 0.0
    assert result["Calculated_Degradation_Index"].iloc[-1] > 0.0
    assert bool(result["Sustained_Tire_Decay_Flag"].iloc[-1]) is True


def test_tire_dynamics_falls_back_to_vehicle_speed_without_wheel_speed() -> None:
    telemetry = pd.DataFrame(
        {
            "Speed": [80.0, 120.0],
            "Brake": [0.0, 0.0],
        }
    )

    result = PhysicsEngine().transform(telemetry)

    np.testing.assert_allclose(result["Slip_Ratio"].to_numpy(), np.zeros(2))
    np.testing.assert_allclose(
        result["Wheel_Speed_ms"].to_numpy(),
        result["Speed_ms"].to_numpy(),
    )
    assert (result["Thermal_Decay_Indicator"] == 0.0).all()


def test_tire_decay_window_must_be_positive() -> None:
    engine = PhysicsEngine(PhysicsConfig(tire_decay_window_samples=0))
    telemetry = pd.DataFrame({"Speed": [100.0], "Brake": [0.0]})

    with pytest.raises(ValueError, match="tire_decay_window_samples"):
        engine.transform(telemetry)
