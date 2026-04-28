from __future__ import annotations

import numpy as np
import pytest

from dtcalib.simulation import (
    ExampleRCCircuitSimulator,
    LowPassR1CR2Simulator,
    SimulationResult,
)


# ============================================================
# Helpers
# ============================================================

def _basic_signal(n: int = 100):
    t = np.linspace(0.0, 1.0, n)
    u = np.ones_like(t)
    return t, u


# ============================================================
# ExampleRCCircuitSimulator
# ============================================================

def test_example_simulation_returns_correct_type() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=True)

    result = sim.simulate(t, u, theta=np.array([0.1], dtype=float))

    assert isinstance(result, SimulationResult)
    assert isinstance(result.y, np.ndarray)
    assert result.y.shape == u.shape
    assert "tau" in result.aux
    assert result.aux["tau"] == pytest.approx(0.1)


def test_example_simulation_converges_for_constant_input() -> None:
    t, u = _basic_signal(n=500)
    sim = ExampleRCCircuitSimulator(use_tau=True)

    tau = 0.1
    result = sim.simulate(t, u, theta=np.array([tau], dtype=float))

    assert result.y[-1] == pytest.approx(1.0, rel=1e-2)


def test_example_simulation_with_R_C_parameters() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=False)

    R = 2.0
    C = 0.5
    result = sim.simulate(t, u, theta=np.array([R, C], dtype=float))

    assert result.aux["tau"] == pytest.approx(R * C)


def test_example_error_if_theta_shape_wrong_tau_mode() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="Expected theta shape"):
        sim.simulate(t, u, theta=np.array([0.1, 0.2], dtype=float))


def test_example_error_if_theta_shape_wrong_RC_mode() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=False)

    with pytest.raises(ValueError, match="Expected theta shape"):
        sim.simulate(t, u, theta=np.array([0.1], dtype=float))


def test_example_error_if_tau_negative() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="tau must be > 0"):
        sim.simulate(t, u, theta=np.array([-1.0], dtype=float))


def test_example_error_if_time_not_increasing() -> None:
    t = np.array([0.0, 0.5, 0.4, 1.0], dtype=float)
    u = np.ones_like(t)

    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="strictly increasing"):
        sim.simulate(t, u, theta=np.array([0.1], dtype=float))


def test_example_error_if_shapes_mismatch() -> None:
    t = np.linspace(0.0, 1.0, 10)
    u = np.ones(9)

    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="same length"):
        sim.simulate(t, u, theta=np.array([0.1], dtype=float))


def test_example_error_if_less_than_two_samples() -> None:
    t = np.array([0.0], dtype=float)
    u = np.array([1.0], dtype=float)

    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="at least 2 samples"):
        sim.simulate(t, u, theta=np.array([0.1], dtype=float))


# ============================================================
# LowPassR1CR2Simulator
# ============================================================

def test_lowpass_simulation_returns_correct_type_and_aux() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="dc_from_u0",
    )

    C = 1e-6
    result = sim.simulate(t, u, theta=np.array([C], dtype=float))

    assert isinstance(result, SimulationResult)
    assert isinstance(result.y, np.ndarray)
    assert result.y.shape == u.shape

    assert result.aux["R1"] == pytest.approx(10_000.0)
    assert result.aux["R2"] == pytest.approx(10_000.0)
    assert result.aux["C"] == pytest.approx(C)
    assert result.aux["dc_gain"] == pytest.approx(0.5)
    assert "tau_eff" in result.aux


def test_lowpass_dc_gain_is_respected_at_steady_state() -> None:
    t = np.linspace(0.0, 0.5, 5000)
    u = np.ones_like(t)

    R1 = 10_000.0
    R2 = 10_000.0
    C = 1e-6

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": R1, "R2": R2},
        y0_mode="dc_from_u0",
    )

    result = sim.simulate(t, u, theta=np.array([C], dtype=float))

    expected_dc_gain = R2 / (R1 + R2)
    assert result.y[-1] == pytest.approx(expected_dc_gain, rel=1e-2)


def test_lowpass_y0_mode_zero() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="zero",
    )

    result = sim.simulate(t, u, theta=np.array([1e-6], dtype=float))
    assert result.y[0] == pytest.approx(0.0)


def test_lowpass_y0_mode_u0() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="u0",
    )

    result = sim.simulate(t, u, theta=np.array([1e-6], dtype=float))
    assert result.y[0] == pytest.approx(u[0])


def test_lowpass_y0_mode_dc_from_u0() -> None:
    t, u = _basic_signal()
    R1 = 10_000.0
    R2 = 20_000.0

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": R1, "R2": R2},
        y0_mode="dc_from_u0",
    )

    result = sim.simulate(t, u, theta=np.array([1e-6], dtype=float))

    expected_y0 = (R2 / (R1 + R2)) * u[0]
    assert result.y[0] == pytest.approx(expected_y0)


def test_lowpass_multi_parameter_theta_order_is_used() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("R2", "C"),
        fixed_params={"R1": 10_000.0},
        y0_mode="dc_from_u0",
    )

    result = sim.simulate(t, u, theta=np.array([20_000.0, 2e-6], dtype=float))

    assert result.aux["R1"] == pytest.approx(10_000.0)
    assert result.aux["R2"] == pytest.approx(20_000.0)
    assert result.aux["C"] == pytest.approx(2e-6)


def test_lowpass_error_if_theta_shape_wrong() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("R2", "C"),
        fixed_params={"R1": 10_000.0},
    )

    with pytest.raises(ValueError, match="Expected theta shape"):
        sim.simulate(t, u, theta=np.array([1e-6], dtype=float))


def test_lowpass_error_if_theta_not_1d() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
    )

    with pytest.raises(ValueError, match="theta must be a 1D array"):
        sim.simulate(t, u, theta=np.array([[1e-6]], dtype=float))


def test_lowpass_error_if_parameter_non_positive() -> None:
    t, u = _basic_signal()
    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
    )

    with pytest.raises(ValueError, match="must be > 0"):
        sim.simulate(t, u, theta=np.array([-1e-6], dtype=float))


def test_lowpass_error_if_time_not_increasing() -> None:
    t = np.array([0.0, 0.5, 0.4, 1.0], dtype=float)
    u = np.ones_like(t)

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        sim.simulate(t, u, theta=np.array([1e-6], dtype=float))


def test_lowpass_error_if_shapes_mismatch() -> None:
    t = np.linspace(0.0, 1.0, 10)
    u = np.ones(9)

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
    )

    with pytest.raises(ValueError, match="same length"):
        sim.simulate(t, u, theta=np.array([1e-6], dtype=float))


def test_lowpass_error_if_less_than_two_samples() -> None:
    t = np.array([0.0], dtype=float)
    u = np.array([1.0], dtype=float)

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
    )

    with pytest.raises(ValueError, match="at least 2 samples"):
        sim.simulate(t, u, theta=np.array([1e-6], dtype=float))


def test_lowpass_constructor_rejects_invalid_y0_mode() -> None:
    with pytest.raises(ValueError, match="y0_mode must be one of"):
        LowPassR1CR2Simulator(
            calibrated_params=("C",),
            fixed_params={"R1": 10_000.0, "R2": 10_000.0},
            y0_mode="invalid_mode",
        )


def test_lowpass_constructor_rejects_empty_calibrated_params() -> None:
    with pytest.raises(ValueError, match="calibrated_params cannot be empty"):
        LowPassR1CR2Simulator(
            calibrated_params=(),
            fixed_params={"R1": 10_000.0, "R2": 10_000.0, "C": 1e-6},
        )


def test_lowpass_constructor_rejects_duplicate_calibrated_params() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        LowPassR1CR2Simulator(
            calibrated_params=("C", "C"),
            fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        )


def test_lowpass_constructor_rejects_overlap_between_fixed_and_calibrated() -> None:
    with pytest.raises(ValueError, match="both fixed and calibrated"):
        LowPassR1CR2Simulator(
            calibrated_params=("C",),
            fixed_params={"R1": 10_000.0, "R2": 10_000.0, "C": 1e-6},
        )


def test_lowpass_constructor_rejects_missing_parameter_definition() -> None:
    with pytest.raises(ValueError, match="Missing parameter definition"):
        LowPassR1CR2Simulator(
            calibrated_params=("C",),
            fixed_params={"R1": 10_000.0},
        )


def test_lowpass_constructor_rejects_non_positive_fixed_param() -> None:
    with pytest.raises(ValueError, match="Fixed parameter R1 must be > 0"):
        LowPassR1CR2Simulator(
            calibrated_params=("C",),
            fixed_params={"R1": -10_000.0, "R2": 10_000.0},
        )

def test_lowpass_constructor_rejects_unknown_calibrated_param() -> None:
    with pytest.raises(ValueError, match="Unknown calibrated parameter"):
        LowPassR1CR2Simulator(
            calibrated_params=("R3",),
            fixed_params={"R1": 10_000.0, "R2": 10_000.0, "C": 1e-6},
        )


def test_lowpass_constructor_rejects_unknown_fixed_param() -> None:
    with pytest.raises(ValueError, match="Unknown fixed parameter"):
        LowPassR1CR2Simulator(
            calibrated_params=("C",),
            fixed_params={"R1": 10_000.0, "R2": 10_000.0, "R3": 5_000.0},
        )


def test_lowpass_properties_calibrated_params_and_n_parameters() -> None:
    sim = LowPassR1CR2Simulator(
        calibrated_params=("R1", "R2", "C"),
        fixed_params={},
    )

    assert sim.calibrated_params == ("R1", "R2", "C")
    assert sim.n_parameters == 3


def test_lowpass_full_parameter_theta_order_is_used() -> None:
    t, u = _basic_signal()

    sim = LowPassR1CR2Simulator(
        calibrated_params=("R1", "R2", "C"),
        fixed_params={},
        y0_mode="dc_from_u0",
    )

    result = sim.simulate(
        t,
        u,
        theta=np.array([5_000.0, 20_000.0, 2e-6], dtype=float),
    )

    assert result.aux["R1"] == pytest.approx(5_000.0)
    assert result.aux["R2"] == pytest.approx(20_000.0)
    assert result.aux["C"] == pytest.approx(2e-6)


def test_lowpass_tau_eff_matches_analytic_formula() -> None:
    t, u = _basic_signal()

    R1 = 10_000.0
    R2 = 20_000.0
    C = 2e-6

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": R1, "R2": R2},
        y0_mode="dc_from_u0",
    )

    result = sim.simulate(t, u, theta=np.array([C], dtype=float))

    expected_tau_eff = C / ((1.0 / R1) + (1.0 / R2))

    assert result.aux["tau_eff"] == pytest.approx(expected_tau_eff)


def test_lowpass_dc_gain_matches_analytic_formula() -> None:
    t, u = _basic_signal()

    R1 = 5_000.0
    R2 = 15_000.0
    C = 1e-6

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": R1, "R2": R2},
        y0_mode="dc_from_u0",
    )

    result = sim.simulate(t, u, theta=np.array([C], dtype=float))

    expected_dc_gain = R2 / (R1 + R2)

    assert result.aux["dc_gain"] == pytest.approx(expected_dc_gain)


def test_lowpass_output_is_finite_for_large_time_step() -> None:
    t = np.linspace(0.0, 10.0, 20)
    u = np.ones_like(t)

    sim = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="zero",
    )

    result = sim.simulate(t, u, theta=np.array([1e-6], dtype=float))

    assert np.all(np.isfinite(result.y))