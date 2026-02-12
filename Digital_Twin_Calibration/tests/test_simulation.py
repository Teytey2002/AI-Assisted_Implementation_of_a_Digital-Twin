from __future__ import annotations

import numpy as np
import pytest

from dtcalib.simulation import ExampleRCCircuitSimulator, SimulationResult


def _basic_signal(n: int = 100):
    t = np.linspace(0.0, 1.0, n)
    u = np.ones_like(t)
    return t, u


def test_simulation_returns_correct_type() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=True)

    result = sim.simulate(t, u, theta=np.array([0.1]))

    assert isinstance(result, SimulationResult)
    assert isinstance(result.y, np.ndarray)
    assert result.y.shape == u.shape
    assert "tau" in result.aux
    assert result.aux["tau"] == pytest.approx(0.1)


def test_simulation_converges_for_constant_input() -> None:
    t, u = _basic_signal(n=500)
    sim = ExampleRCCircuitSimulator(use_tau=True)

    tau = 0.1
    result = sim.simulate(t, u, theta=np.array([tau]))

    # For constant input = 1, steady-state should approach 1
    assert result.y[-1] == pytest.approx(1.0, rel=1e-2)


def test_simulation_with_R_C_parameters() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=False)

    R = 2.0
    C = 0.5
    result = sim.simulate(t, u, theta=np.array([R, C]))

    assert result.aux["tau"] == pytest.approx(R * C)


def test_error_if_theta_shape_wrong_tau_mode() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="Expected theta shape"):
        sim.simulate(t, u, theta=np.array([0.1, 0.2]))


def test_error_if_theta_shape_wrong_RC_mode() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=False)

    with pytest.raises(ValueError, match="Expected theta shape"):
        sim.simulate(t, u, theta=np.array([0.1]))


def test_error_if_tau_negative() -> None:
    t, u = _basic_signal()
    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="tau must be > 0"):
        sim.simulate(t, u, theta=np.array([-1.0]))


def test_error_if_time_not_increasing() -> None:
    t = np.array([0.0, 0.5, 0.4, 1.0])
    u = np.ones_like(t)

    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="strictly increasing"):
        sim.simulate(t, u, theta=np.array([0.1]))


def test_error_if_shapes_mismatch() -> None:
    t = np.linspace(0.0, 1.0, 10)
    u = np.ones(9)

    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="same length"):
        sim.simulate(t, u, theta=np.array([0.1]))


def test_error_if_less_than_two_samples() -> None:
    t = np.array([0.0])
    u = np.array([1.0])

    sim = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError, match="at least 2 samples"):
        sim.simulate(t, u, theta=np.array([0.1]))
