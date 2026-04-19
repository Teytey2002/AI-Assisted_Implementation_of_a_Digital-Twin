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
    Simulator for the circuit:
        u -- R1 -- v
                 |-- C -- GND
                 |-- R2 -- GND

    ODE:
        dv/dt = (1/(R1*C)) * u - ( (1/R1 + 1/R2)/C ) * v

    Exact ZOH discretization:
        v_k = v_{k-1} * exp(-a*dt) + (b/a) * u_{k-1} * (1 - exp(-a*dt))
        where:
            a = (1/R1 + 1/R2) / C
            b = 1/(R1*C)

    New theta convention:
      - calibrated_params defines which physical parameters are estimated
      - theta follows exactly the same order as calibrated_params

    Examples:
      - calibrate only C:
            LowPassR1CR2Simulator(
                calibrated_params=("C",),
                fixed_params={"R1": 10_000.0, "R2": 10_000.0},
            )
            theta = [C]

      - calibrate R2 and C:
            LowPassR1CR2Simulator(
                calibrated_params=("R2", "C"),
                fixed_params={"R1": 10_000.0},
            )
            theta = [R2, C]

      - calibrate R1, R2, C:
            LowPassR1CR2Simulator(
                calibrated_params=("R1", "R2", "C"),
                fixed_params={},
            )
            theta = [R1, R2, C]
    """

    _VALID_PARAM_NAMES = ("R1", "R2", "C")

    def __init__(
        self,
        *,
        calibrated_params: tuple[str, ...] = ("C",),
        fixed_params: Optional[Dict[str, float]] = None,
        y0_mode: str = "dc_from_u0",
    ) -> None:
        if y0_mode not in {"zero", "u0", "dc_from_u0"}:
            raise ValueError("y0_mode must be one of {'zero', 'u0', 'dc_from_u0'}.")

        calibrated_params = tuple(calibrated_params)
        if len(calibrated_params) == 0:
            raise ValueError("calibrated_params cannot be empty.")

        invalid = [p for p in calibrated_params if p not in self._VALID_PARAM_NAMES]
        if invalid:
            raise ValueError(
                f"Unknown calibrated parameter(s): {invalid}. "
                f"Valid names are {self._VALID_PARAM_NAMES}."
            )

        if len(set(calibrated_params)) != len(calibrated_params):
            raise ValueError("calibrated_params must not contain duplicates.")

        fixed_params = {} if fixed_params is None else dict(fixed_params)

        invalid_fixed = [p for p in fixed_params if p not in self._VALID_PARAM_NAMES]
        if invalid_fixed:
            raise ValueError(
                f"Unknown fixed parameter(s): {invalid_fixed}. "
                f"Valid names are {self._VALID_PARAM_NAMES}."
            )

        overlap = set(calibrated_params).intersection(fixed_params.keys())
        if overlap:
            raise ValueError(
                f"Parameters cannot be both fixed and calibrated: {sorted(overlap)}."
            )

        missing = set(self._VALID_PARAM_NAMES) - set(calibrated_params) - set(fixed_params.keys())
        if missing:
            raise ValueError(
                f"Missing parameter definition for: {sorted(missing)}. "
                "Each of R1, R2, C must be either fixed or calibrated."
            )

        for name, value in fixed_params.items():
            value = float(value)
            if value <= 0:
                raise ValueError(f"Fixed parameter {name} must be > 0.")
            fixed_params[name] = value

        self._calibrated_params = calibrated_params
        self._fixed_params = fixed_params
        self._y0_mode = y0_mode

    @property
    def calibrated_params(self) -> tuple[str, ...]:
        return self._calibrated_params

    @property
    def n_parameters(self) -> int:
        return len(self._calibrated_params)

    def _decode_theta(self, theta: np.ndarray) -> tuple[float, float, float]:
        theta = np.asarray(theta, dtype=float)

        if theta.ndim != 1:
            raise ValueError(f"theta must be a 1D array, got shape {theta.shape}.")
        if theta.shape != (len(self._calibrated_params),):
            raise ValueError(
                f"Expected theta shape {(len(self._calibrated_params),)} "
                f"for calibrated_params={self._calibrated_params}, got {theta.shape}."
            )

        params: Dict[str, float] = dict(self._fixed_params)
        for name, value in zip(self._calibrated_params, theta):
            value = float(value)
            if value <= 0:
                raise ValueError(f"Parameter {name} must be > 0, got {value}.")
            params[name] = value

        R1 = float(params["R1"])
        R2 = float(params["R2"])
        C = float(params["C"])

        return R1, R2, C

    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1D arrays.")
        if t.shape[0] != u.shape[0]:
            raise ValueError("t and u must have same length.")
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to simulate.")

        R1, R2, C = self._decode_theta(theta)

        invR1 = 1.0 / R1
        invR2 = 1.0 / R2

        a = (invR1 + invR2) / C
        b = invR1 / C

        tau_eff = 1.0 / a
        dc_gain = R2 / (R1 + R2)

        y = np.zeros_like(u, dtype=float)

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
                "R1": R1,
                "R2": R2,
                "C": C,
                "a": a,
                "b": b,
                "tau_eff": tau_eff,
                "dc_gain": dc_gain,
                "calibrated_params": self._calibrated_params,
                "fixed_params": self._fixed_params,
            },
        )
