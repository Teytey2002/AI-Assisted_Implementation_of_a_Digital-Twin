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
    y: np.ndarray           # signal simulate (y^^(t))
    aux: Dict[str, object]  # data supp (not usefull for calibration, more usefull for debugging)


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

    @abstractmethod    # Why an abstract class (ABC)? An ABC imposes a rule: any class that inherits from Simulator MUST implement simulate.
    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        raise NotImplementedError


class ExampleRCCircuitSimulator(Simulator):
    """
    Example placeholder: 1st-order RC low-pass discrete-time simulation.

    NOTE:
      This is just a template to show structure. Replace with your real simulator
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

class LowPassR1CR2Simulator(Simulator):
    """
    Simulator for the real circuit:
        u -- R1 -- v
                 |-- C -- GND
                 |-- R2 -- GND

    ODE:
        dv/dt = (1/(R1*C)) * u - ( (1/R1 + 1/R2)/C ) * v

    Discretization:
        Exact ZOH (u constant on each [t_{k-1}, t_k]):

        v_k = v_{k-1} * exp(-a*dt) + (b/a) * u_{k-1} * (1 - exp(-a*dt))
        where:
          a = (1/R1 + 1/R2) / C
          b = 1/(R1*C)

    theta convention:
      - theta = [C]  (Farads)  if use_C=True
      - theta = [tau] (seconds) if use_C=False, with tau = C*(R1*R2)/(R1+R2)
    """

    def __init__(self, *, R1: float, R2: float, use_C: bool = True, y0_mode: str = "dc_from_u0") -> None:
        if R1 <= 0 or R2 <= 0:
            raise ValueError("R1 and R2 must be > 0.")
        if y0_mode not in {"zero", "u0", "dc_from_u0"}:
            raise ValueError("y0_mode must be one of {'zero','u0','dc_from_u0'}.")
        self._R1 = float(R1)
        self._R2 = float(R2)
        self._use_C = bool(use_C)
        self._y0_mode = y0_mode

    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1D arrays.")
        if t.shape[0] != u.shape[0]:
            raise ValueError("t and u must have same length.")
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to simulate.")

        theta = np.asarray(theta, dtype=float)

        # Interpret theta
        if self._use_C:
            if theta.shape != (1,):
                raise ValueError("Expected theta shape (1,) for C.")
            C = float(theta[0])
        else:
            if theta.shape != (1,):
                raise ValueError("Expected theta shape (1,) for tau.")
            tau = float(theta[0])
            # tau = C * (R1*R2)/(R1+R2)  ->  C = tau * (R1+R2)/(R1*R2)
            C = tau * (self._R1 + self._R2) / (self._R1 * self._R2)

        if C <= 0:
            raise ValueError("C must be > 0.")

        invR1 = 1.0 / self._R1
        invR2 = 1.0 / self._R2

        a = (invR1 + invR2) / C
        b = invR1 / C

        # Diagnostics
        tau_eff = 1.0 / a
        dc_gain = self._R2 / (self._R1 + self._R2)

        y = np.zeros_like(u, dtype=float)

        # Initial condition options
        if self._y0_mode == "zero":
            y[0] = 0.0
        elif self._y0_mode == "u0":
            y[0] = float(u[0])
        else:  # "dc_from_u0"
            y[0] = dc_gain * float(u[0])

        for k in range(1, len(t)):
            dt = float(t[k] - t[k - 1])
            if dt <= 0:
                raise ValueError("Time vector must be strictly increasing.")

            exp_term = np.exp(-a * dt)
            y[k] = y[k - 1] * exp_term + (b / a) * float(u[k - 1]) * (1.0 - exp_term)

        return SimulationResult(
            y=y,
            aux={
                "R1": self._R1,
                "R2": self._R2,
                "C": C,
                "a": a,
                "tau_eff": tau_eff,
                "dc_gain": dc_gain,
            },
        )
