from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """
    Output of a simulation run.
    """
    y: np.ndarray
    aux: Dict[str, object]


class Simulator(ABC):
    """
    Abstract interface for a parametric simulator.

    Implementations:
      - FMU-based simulator
      - ODE-based analytic simulator
      - External tool wrapper

    Requirements:
      simulate(t, u, theta) returns y_pred aligned with t.
    """

    @abstractmethod
    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        raise NotImplementedError


class ExampleRCCircuitSimulator(Simulator):
    """
    Example placeholder: 1st-order RC low-pass discrete-time simulation.

    NOTE:
      This is just a *template* to show structure. Replace with your real simulator
      (FMU, Modelica, etc.) as soon as available.

    Model:
      dy/dt = (1/RC) * (u - y)
      theta = [R, C] or [tau] depending on your choice.
    """

    def __init__(self, *, use_tau: bool = True) -> None:
        self._use_tau = use_tau

    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1D arrays.")
        if t.shape[0] != u.shape[0]:
            raise ValueError("t and u must have same length.")
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to simulate.")

        # Convert theta to time constant tau
        if self._use_tau:
            if theta.shape != (1,):
                raise ValueError("Expected theta shape (1,) for tau.")
            tau = float(theta[0])
        else:
            if theta.shape != (2,):
                raise ValueError("Expected theta shape (2,) for [R, C].")
            R = float(theta[0])
            C = float(theta[1])
            tau = R * C

        if tau <= 0:
            raise ValueError("tau must be > 0.")

        y = np.zeros_like(u, dtype=float)
        y[0] = u[0]  # or 0, depending on initial condition assumption

        for k in range(1, len(t)):
            dt = float(t[k] - t[k - 1])
            if dt <= 0:
                raise ValueError("Time vector must be strictly increasing.")
            alpha = dt / tau
            # Forward Euler
            y[k] = y[k - 1] + alpha * (u[k - 1] - y[k - 1])

        return SimulationResult(y=y, aux={"tau": tau})
