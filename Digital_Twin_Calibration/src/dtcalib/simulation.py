from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import lambertw

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


class ThreeStageRCLadderSimulator(Simulator):
    """
    Simulator for the LTspice circuit:

        Vin -- R1 -- v1 -- R3 -- v2 -- R5 -- v3 = Vout
                    |             |           |
                    |             |           +-- R6 -- GND
                    |             |           |
                    |             |           +-- C3 -- GND
                    |             |           |
                    |             |           +-- R7 -- GND
                    |             |
                    |             +-- R4 -- GND
                    |             |
                    |             +-- C2 -- GND
                    |
                    +-- R2 -- GND
                    |
                    +-- C1 -- GND

    State vector:
        x = [v1, v2, v3]

    Output:
        y = v3 = Vout

    All components can be either fixed or calibrated:
        R1, R2, R3, R4, R5, R6, R7, C1, C2, C3
    """

    _VALID_PARAM_NAMES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "C1", "C2", "C3")

    _DEFAULT_PARAMS = {
        "R1": 10.0,
        "R2": 47.5,
        "R3": 22.1,
        "R4": 15.0,
        "R5": 33.2,
        "R6": 68.1,
        "R7": 100.0,
        "C1": 1e-6,
        "C2": 10e-6,
        "C3": 15e-6,
    }

    def __init__(
        self,
        *,
        calibrated_params: tuple[str, ...] = ("C1", "C2", "C3"),
        fixed_params: Optional[Dict[str, float]] = None,
        y0_mode: str = "zero",
    ) -> None:
        if y0_mode not in {"zero", "dc_from_u0"}:
            raise ValueError("y0_mode must be one of {'zero', 'dc_from_u0'}.")

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

        if fixed_params is None:
            fixed_params = {
                p: v for p, v in self._DEFAULT_PARAMS.items()
                if p not in calibrated_params
            }
        else:
            fixed_params = dict(fixed_params)

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
                "Each component must be either fixed or calibrated."
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

    def _decode_theta(self, theta: np.ndarray) -> Dict[str, float]:
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

        return params

    @staticmethod
    def _state_matrix(params: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        R1 = params["R1"]
        R2 = params["R2"]
        R3 = params["R3"]
        R4 = params["R4"]
        R5 = params["R5"]
        R6 = params["R6"]
        R7 = params["R7"]
        C1 = params["C1"]
        C2 = params["C2"]
        C3 = params["C3"]

        g1 = 1.0 / R1
        g2 = 1.0 / R2
        g3 = 1.0 / R3
        g4 = 1.0 / R4
        g5 = 1.0 / R5
        g6 = 1.0 / R6
        g7 = 1.0 / R7

        A = np.array(
            [
                [-(g1 + g2 + g3) / C1,  g3 / C1,               0.0],
                [ g3 / C2,              -(g3 + g4 + g5) / C2,  g5 / C2],
                [ 0.0,                   g5 / C3,              -(g5 + g6 + g7) / C3],
            ],
            dtype=float,
        )

        B = np.array(
            [
                g1 / C1,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        return A, B

    @staticmethod
    def _dc_state(params: Dict[str, float], u0: float) -> np.ndarray:
        A, B = ThreeStageRCLadderSimulator._state_matrix(params)

        # At DC: dx/dt = 0 => A x + B u = 0
        return np.linalg.solve(A, -B * float(u0))

    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        t = np.asarray(t, dtype=float)
        u = np.asarray(u, dtype=float)

        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1D arrays.")
        if t.shape[0] != u.shape[0]:
            raise ValueError("t and u must have same length.")
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to simulate.")

        params = self._decode_theta(theta)
        A, B = self._state_matrix(params)
        y = np.zeros(len(t), dtype=float)
        x = np.zeros((len(t), 3), dtype=float)

        if self._y0_mode == "zero":
            x[0, :] = 0.0
        else:
            x[0, :] = self._dc_state(params, float(u[0]))

        # RK4 integration, robust for arbitrary input sampling
        I = np.eye(3)

        for k in range(1, len(t)):
            dt = float(t[k] - t[k - 1])
            if dt <= 0:
                raise ValueError("Time vector must be strictly increasing.")

            uk = float(u[k - 1])

            Phi = expm(A * dt)

            # Exact ZOH discretization:
            # x[k] = Phi x[k-1] + A^{-1}(Phi - I) B u[k-1]
            Gamma = np.linalg.solve(A, (Phi - I) @ B)

            x[k] = Phi @ x[k - 1] + Gamma * uk

        y = x[:, 2].copy()

        return SimulationResult(
            y=y,
            aux={
                **params,
                "A": A,
                "B": B,
                "v1": x[:, 0],
                "v2": x[:, 1],
                "v3": x[:, 2],
                "calibrated_params": self._calibrated_params,
                "fixed_params": self._fixed_params,
                "y0_mode": self._y0_mode,
            },
        )
    



class ThreeStageRLCLadderSimulator(Simulator):
    """
    Simulator for the LTspice RLC ladder circuit:

    Vin ── R1 ── L1 ── v1 ── R3 ── L2 ── v2 ── R5 ── L3 ──●── Vout
                        |                 |                |
                        |                 |                +── R6 ── GND
                        |                 |                |
                        |                 |                +── C3 ── GND
                        |                 |                |
                        |                 |                +── R7 ── GND
                        |                 |
                        |                 +── R4 ── GND
                        |                 |
                        |                 +── C2 ── GND
                        |
                        +── R2 ── GND
                        |
                        +── C1 ── GND

    State vector:
        x = [iL1, iL2, iL3, v1, v2, v3]

    Output:
        y = v3 = Vout
    """

    _VALID_PARAM_NAMES = (
        "R1", "L1", "R2", "C1",
        "R3", "L2", "R4", "C2",
        "R5", "L3", "R6", "C3", "R7",
    )

    _DEFAULT_PARAMS = {
        "R1": 10.0,
        "L1": 10e-3,
        "R2": 42.2,
        "C1": 1e-6,

        "R3": 22.1,
        "L2": 22e-3,
        "R4": 15.0,
        "C2": 10e-6,

        "R5": 33.2,
        "L3": 33e-3,
        "R6": 68.1,
        "C3": 15e-6,
        "R7": 100.0,
    }

    def __init__(
        self,
        *,
        calibrated_params: tuple[str, ...] = ("L1", "L2", "L3"),
        fixed_params: Optional[Dict[str, float]] = None,
        y0_mode: str = "zero",
    ) -> None:
        if y0_mode not in {"zero", "dc_from_u0"}:
            raise ValueError("y0_mode must be one of {'zero', 'dc_from_u0'}.")

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

        if fixed_params is None:
            fixed_params = {
                p: v for p, v in self._DEFAULT_PARAMS.items()
                if p not in calibrated_params
            }
        else:
            fixed_params = dict(fixed_params)

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
                "Each component must be either fixed or calibrated."
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

    def _decode_theta(self, theta: np.ndarray) -> Dict[str, float]:
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

        return params

    @staticmethod
    def _state_matrix(params: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        R1 = params["R1"]
        L1 = params["L1"]
        R2 = params["R2"]
        C1 = params["C1"]

        R3 = params["R3"]
        L2 = params["L2"]
        R4 = params["R4"]
        C2 = params["C2"]

        R5 = params["R5"]
        L3 = params["L3"]
        R6 = params["R6"]
        C3 = params["C3"]
        R7 = params["R7"]

        A = np.array(
            [
                [-R1 / L1, 0.0,       0.0,       -1.0 / L1,  0.0,        0.0],
                [0.0,      -R3 / L2,  0.0,        1.0 / L2, -1.0 / L2,   0.0],
                [0.0,       0.0,     -R5 / L3,    0.0,       1.0 / L3,  -1.0 / L3],

                [1.0 / C1, -1.0 / C1, 0.0,       -1.0 / (R2 * C1), 0.0, 0.0],
                [0.0,       1.0 / C2, -1.0 / C2,  0.0, -1.0 / (R4 * C2), 0.0],
                [0.0,       0.0,       1.0 / C3,  0.0, 0.0, -(1.0 / R6 + 1.0 / R7) / C3],
            ],
            dtype=float,
        )

        B = np.array(
            [
                1.0 / L1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        return A, B

    @staticmethod
    def _dc_state(params: Dict[str, float], u0: float) -> np.ndarray:
        A, B = ThreeStageRLCLadderSimulator._state_matrix(params)
        return np.linalg.solve(A, -B * float(u0))

    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        t = np.asarray(t, dtype=float)
        u = np.asarray(u, dtype=float)

        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1D arrays.")
        if t.shape[0] != u.shape[0]:
            raise ValueError("t and u must have same length.")
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to simulate.")

        params = self._decode_theta(theta)
        A, B = self._state_matrix(params)

        x = np.zeros((len(t), 6), dtype=float)

        if self._y0_mode == "zero":
            x[0, :] = 0.0
        else:
            x[0, :] = self._dc_state(params, float(u[0]))

        I = np.eye(6)

        for k in range(1, len(t)):
            dt = float(t[k] - t[k - 1])
            if dt <= 0:
                raise ValueError("Time vector must be strictly increasing.")

            uk = float(u[k - 1])

            Phi = expm(A * dt)
            Gamma = np.linalg.solve(A, (Phi - I) @ B)

            x[k] = Phi @ x[k - 1] + Gamma * uk

        y = x[:, 5].copy()

        return SimulationResult(
            y=y,
            aux={
                **params,
                "A": A,
                "B": B,
                "iL1": x[:, 0],
                "iL2": x[:, 1],
                "iL3": x[:, 2],
                "v1": x[:, 3],
                "v2": x[:, 4],
                "v3": x[:, 5],
                "calibrated_params": self._calibrated_params,
                "fixed_params": self._fixed_params,
                "y0_mode": self._y0_mode,
            },
        )
    

class DiodeClippedRCSimulator(Simulator):
    """
    Simulator for nonlinear RC diode clipping circuit:

        Vin -- R1 -- v = Vout
                    |
                    +-- C1 -- GND
                    |
                    +-- D1 -- GND

    Diode orientation:
        anode at Vout, cathode at GND

    ODE:
        C dv/dt = (Vin - v)/R - Id(v)

    Diode model:
        Id = Is * (exp(Vd / (n*Vt)) - 1)
        with optional series resistance Rs:
        v = Vd + Rs * Id
    """

    _VALID_PARAM_NAMES = ("R1", "C1", "IS", "N", "VT", "RS")

    _DEFAULT_PARAMS = {
        "R1": 1_000.0,
        "C1": 10e-6,
        "IS": 2.52e-9,
        "N": 1.75,
        "VT": 25.85e-3,
        "RS": 0.568,
    }

    def __init__(
        self,
        *,
        calibrated_params: tuple[str, ...] = ("R1", "C1"),
        fixed_params: Optional[Dict[str, float]] = None,
        y0_mode: str = "zero",
        method: str = "BDF",
    ) -> None:
        if y0_mode not in {"zero", "dc_from_u0"}:
            raise ValueError("y0_mode must be one of {'zero', 'dc_from_u0'}.")

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

        if fixed_params is None:
            fixed_params = {
                p: v for p, v in self._DEFAULT_PARAMS.items()
                if p not in calibrated_params
            }
        else:
            fixed_params = dict(fixed_params)

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
                "Each component/diode parameter must be either fixed or calibrated."
            )

        for name, value in fixed_params.items():
            value = float(value)
            if value <= 0:
                raise ValueError(f"Fixed parameter {name} must be > 0.")
            fixed_params[name] = value

        self._calibrated_params = calibrated_params
        self._fixed_params = fixed_params
        self._y0_mode = y0_mode
        self._method = method

    @property
    def calibrated_params(self) -> tuple[str, ...]:
        return self._calibrated_params

    @property
    def n_parameters(self) -> int:
        return len(self._calibrated_params)

    def _decode_theta(self, theta: np.ndarray) -> Dict[str, float]:
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

        return params

    @staticmethod
    def _diode_current(v: float, params: Dict[str, float]) -> float:
        IS = params["IS"]
        N = params["N"]
        VT = params["VT"]
        RS = params["RS"]

        a = N * VT

        if RS <= 1e-15:
            arg = np.clip(v / a, -100.0, 100.0)
            return float(IS * np.expm1(arg))

        # Solves:
        #   i = IS * (exp((v - RS*i)/(N*VT)) - 1)
        #
        # Closed form:
        #   i = (a / RS) * W((RS*IS/a) * exp((v + RS*IS)/a)) - IS

        log_z = np.log(RS * IS / a) + (v + RS * IS) / a

        # Avoid numerical overflow in exp
        log_z = np.clip(log_z, -700.0, 700.0)

        z = np.exp(log_z)
        w = lambertw(z).real

        i = (a / RS) * w - IS

        return float(i)

    @staticmethod
    def _dc_state(params: Dict[str, float], u0: float) -> float:
        R1 = params["R1"]

        def f(v: float) -> float:
            return (u0 - v) / R1 - DiodeClippedRCSimulator._diode_current(v, params)

        return float(brentq(f, -10.0, max(10.0, float(u0) + 10.0)))

    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        t = np.asarray(t, dtype=float)
        u = np.asarray(u, dtype=float)

        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1D arrays.")
        if t.shape[0] != u.shape[0]:
            raise ValueError("t and u must have same length.")
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to simulate.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Time vector must be strictly increasing.")

        params = self._decode_theta(theta)

        R1 = params["R1"]
        C1 = params["C1"]

        if self._y0_mode == "zero":
            v0 = 0.0
        else:
            v0 = self._dc_state(params, float(u[0]))

        def rhs(ti: float, x: np.ndarray) -> np.ndarray:
            vin = float(np.interp(ti, t, u))
            v = float(x[0])
            id_ = self._diode_current(v, params)
            dvdt = ((vin - v) / R1 - id_) / C1
            return np.array([dvdt], dtype=float)

        sol = solve_ivp(
            rhs,
            t_span=(float(t[0]), float(t[-1])),
            y0=np.array([v0], dtype=float),
            t_eval=t,
            method=self._method,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"Diode RC simulation failed: {sol.message}")

        y = sol.y[0].astype(float)

        diode_current = np.array(
            [self._diode_current(float(v), params) for v in y],
            dtype=float,
        )

        return SimulationResult(
            y=y,
            aux={
                **params,
                "vout": y,
                "diode_current": diode_current,
                "calibrated_params": self._calibrated_params,
                "fixed_params": self._fixed_params,
                "y0_mode": self._y0_mode,
                "method": self._method,
            },
        )