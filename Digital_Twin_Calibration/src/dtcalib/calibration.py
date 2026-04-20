from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from .data import Experiment
from .simulation import Simulator
from .metrics import Metrics, MetricsResult

from pathlib import Path
import torch

from dtcalib.deep_learning.model import RCInverseCNN, ProbabilisticRCInverseCNN


@dataclass(frozen=True)
class CalibrationReport:
    theta_hat: np.ndarray       # Parameter estimated
    cost: float
    success: bool               # Info of convergence (from scipy)
    message: str                # Info of convergence (from scipy)
    nfev: int                   # Info of convergence (from scipy)
    per_experiment_metrics: List[tuple[str, MetricsResult]]     # [RMSE, NMSE, MSE]


class LeastSquaresCalibrator:
    """
    Nonlinear least squares parameter calibration:
      theta_hat = argmin_theta sum_i || y_i - sim(t_i,u_i;theta) ||^2

    Uses scipy.optimize.least_squares (Levenberg-Marquardt or trust-region).
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        method: str = "trf",
        loss: str = "linear",
        f_scale: float = 1.0,
    ) -> None:
        """
        Parameters:
          method: "trf", "dogbox", or "lm" (lm requires unconstrained)
          loss: robust losses supported by scipy ("linear", "soft_l1", "huber", "cauchy", "arctan")
        """
        self._sim = simulator
        self._method = method
        self._loss = loss
        self._f_scale = float(f_scale)

    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
        max_nfev: Optional[int] = None,
    ) -> CalibrationReport:
        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")
        theta0 = np.asarray(theta0, dtype=float)

        if weights is not None and len(weights) != len(experiments):
            raise ValueError("weights must match number of experiments.")
        w = np.ones(len(experiments), dtype=float) if weights is None else np.asarray(weights, dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            res_parts: List[np.ndarray] = []
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                # Weighted residuals: sqrt(w_i) * (y - yhat)
                res_parts.append(np.sqrt(w[i]) * (exp.y - sim_out))
            return np.concatenate(res_parts, axis=0)

        if bounds is None:
            lb = -np.inf * np.ones_like(theta0)
            ub = np.inf * np.ones_like(theta0)
            bounds = (lb, ub)

        result = least_squares(
            residuals,
            theta0,
            bounds=bounds,
            method=self._method,
            loss=self._loss,
            f_scale=self._f_scale,
            max_nfev=max_nfev,
            #verbose=2              # Add a lot of information but to much in the terminal. Mayby on log after
        )

        # Build per-experiment diagnostics
        per_metrics: List[tuple[str, MetricsResult]] = []
        for exp in experiments:
            yhat = self._sim.simulate(exp.t, exp.u, result.x).y
            per_metrics.append((exp.name, Metrics.compute(exp.y, yhat)))

        return CalibrationReport(
            theta_hat=result.x.astype(float),
            cost=float(result.cost),  # 0.5 * sum(residuals**2)
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            per_experiment_metrics=per_metrics,
        )


class BayesianMAPCalibrator(LeastSquaresCalibrator):
    """
    Nonlinear Bayesian MAP calibration.

    We assume:
      - Measurement model: y = sim(t,u;theta) + eps,  eps ~ N(0, sigma_y^2)
      - Prior on parameters: theta ~ N(prior_mean, diag(prior_std^2))

    MAP estimate:
      theta_hat = argmin_theta [ (1/(2*sigma_y^2)) * sum ||y - sim(...)||^2
                                + (1/2) * sum ||(theta - prior_mean)/prior_std||^2 ]

    Implementation trick:
      least_squares minimizes 0.5*sum(r(theta)^2).
      So we concatenate:
        r_data  = (y - yhat)/sigma_y
        r_prior = (theta - prior_mean)/prior_std
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        prior_mean: np.ndarray,
        prior_std: np.ndarray,
        sigma_y: float = 1.0,
        method: str = "trf",
        loss: str = "linear",
        f_scale: float = 1.0,
    ) -> None:
        super().__init__(simulator, method=method, loss=loss, f_scale=f_scale)

        self._prior_mean = np.asarray(prior_mean, dtype=float)
        self._prior_std = np.asarray(prior_std, dtype=float)

        if self._prior_mean.ndim != 1 or self._prior_std.ndim != 1:
            raise ValueError("prior_mean and prior_std must be 1D arrays.")
        if self._prior_mean.shape != self._prior_std.shape:
            raise ValueError("prior_mean and prior_std must have the same shape.")
        if np.any(self._prior_std <= 0):
            raise ValueError("prior_std must be strictly positive.")

        self._sigma_y = float(sigma_y)
        if self._sigma_y <= 0:
            raise ValueError("sigma_y must be strictly positive.")

    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
        max_nfev: Optional[int] = None,
    ) -> CalibrationReport:
        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")
        theta0 = np.asarray(theta0, dtype=float)

        if theta0.ndim != 1:
            raise ValueError("theta0 must be a 1D array.")
        if theta0.shape != self._prior_mean.shape:
            raise ValueError(
                f"theta0 shape {theta0.shape} must match prior_mean shape {self._prior_mean.shape}."
            )

        if weights is not None and len(weights) != len(experiments):
            raise ValueError("weights must match number of experiments.")
        w = np.ones(len(experiments), dtype=float) if weights is None else np.asarray(weights, dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            res_parts: List[np.ndarray] = []

            # Data residuals (whitened by sigma_y, weighted per experiment)
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                r = (exp.y - sim_out) / self._sigma_y
                res_parts.append(np.sqrt(w[i]) * r)

            # Prior residuals (MAP): (theta - mu)/sigma_prior
            r_prior = (theta - self._prior_mean) / self._prior_std
            res_parts.append(r_prior)

            return np.concatenate(res_parts, axis=0)

        if bounds is None:
            lb = -np.inf * np.ones_like(theta0)
            ub = np.inf * np.ones_like(theta0)
            bounds = (lb, ub)

        result = least_squares(
            residuals,
            theta0,
            bounds=bounds,
            method=self._method,
            loss=self._loss,
            f_scale=self._f_scale,
            max_nfev=max_nfev,
            # verbose=2
        )

        # Build per-experiment diagnostics (same as LeastSquaresCalibrator)
        per_metrics: List[tuple[str, MetricsResult]] = []
        for exp in experiments:
            yhat = self._sim.simulate(exp.t, exp.u, result.x).y
            per_metrics.append((exp.name, Metrics.compute(exp.y, yhat)))

        return CalibrationReport(
            theta_hat=result.x.astype(float),
            cost=float(result.cost),
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            per_experiment_metrics=per_metrics,
        )


class GeneticAlgorithmCalibrator:
    """
    Genetic algorithm for parameter calibration.

    Objective:
      theta_hat = argmin_theta sum_i w_i * || y_i - sim(t_i,u_i;theta) ||^2

    Notes:
      - derivative-free optimizer
      - explicit finite bounds are required
      - supports any number of parameters as long as theta0 and bounds match
      - returned ``cost`` follows the same convention as LeastSquaresCalibrator:
        0.5 * sum(residuals**2)
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        population_size: int = 80,
        n_generations: int = 120,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.1,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        init_near_theta0_fraction: float = 0.5,
        init_near_theta0_scale: float = 0.2,
        mutation_mode: str = "log",   # "log" or "relative"
        seed: Optional[int] = None,
        polish: bool = True,
        polish_method: str = "trf",
        polish_loss: str = "linear",
        polish_f_scale: float = 1.0,
    ) -> None:
        self._sim = simulator
        self._population_size = int(population_size)
        self._n_generations = int(n_generations)
        self._crossover_rate = float(crossover_rate)
        self._mutation_rate = float(mutation_rate)
        self._mutation_scale = float(mutation_scale)
        self._elite_fraction = float(elite_fraction)
        self._tournament_size = int(tournament_size)

        self._init_near_theta0_fraction = float(init_near_theta0_fraction)
        self._init_near_theta0_scale = float(init_near_theta0_scale)
        self._mutation_mode = str(mutation_mode)

        self._seed = seed
        self._polish = bool(polish)
        self._polish_method = str(polish_method)
        self._polish_loss = str(polish_loss)
        self._polish_f_scale = float(polish_f_scale)

        if self._population_size < 4:
            raise ValueError("population_size must be >= 4.")
        if self._n_generations < 1:
            raise ValueError("n_generations must be >= 1.")
        if not (0.0 <= self._crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1].")
        if not (0.0 <= self._mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1].")
        if self._mutation_scale < 0.0:
            raise ValueError("mutation_scale must be >= 0.")
        if not (0.0 < self._elite_fraction < 1.0):
            raise ValueError("elite_fraction must be in (0, 1).")
        if self._tournament_size < 2:
            raise ValueError("tournament_size must be >= 2.")
        if not (0.0 <= self._init_near_theta0_fraction <= 1.0):
            raise ValueError("init_near_theta0_fraction must be in [0, 1].")
        if self._init_near_theta0_scale < 0.0:
            raise ValueError("init_near_theta0_scale must be >= 0.")
        if self._mutation_mode not in {"log", "relative"}:
            raise ValueError("mutation_mode must be either 'log' or 'relative'.")

    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
        max_nfev: Optional[int] = None,
    ) -> CalibrationReport:
        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")

        theta0 = np.asarray(theta0, dtype=float)
        if theta0.ndim != 1:
            raise ValueError("theta0 must be a 1D array.")

        if bounds is None:
            raise ValueError("GeneticAlgorithmCalibrator requires explicit bounds.")

        lb = np.asarray(bounds[0], dtype=float)
        ub = np.asarray(bounds[1], dtype=float)

        if lb.shape != theta0.shape or ub.shape != theta0.shape:
            raise ValueError("bounds must have the same shape as theta0.")
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            raise ValueError("GeneticAlgorithmCalibrator requires finite bounds.")
        if np.any(lb >= ub):
            raise ValueError("Each lower bound must be strictly smaller than the upper bound.")
        if np.any(theta0 < lb) or np.any(theta0 > ub):
            raise ValueError("theta0 must lie inside bounds.")
        if np.any(lb <= 0.0):
            raise ValueError(
                "This GA implementation assumes strictly positive lower bounds "
                "to support log-scale mutation."
            )

        if weights is not None and len(weights) != len(experiments):
            raise ValueError("weights must match number of experiments.")
        w = np.ones(len(experiments), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")

        rng = np.random.default_rng(self._seed)
        dim = theta0.shape[0]
        span = ub - lb
        elite_count = max(1, int(round(self._elite_fraction * self._population_size)))
        elite_count = min(elite_count, self._population_size - 1)

        nfev = 0

        def objective(theta: np.ndarray) -> float:
            nonlocal nfev
            total = 0.0
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                err = exp.y - sim_out
                total += float(w[i]) * float(np.dot(err, err))
            nfev += 1
            return total

        def evaluate_population(pop: np.ndarray) -> np.ndarray:
            vals = np.empty(pop.shape[0], dtype=float)
            for i in range(pop.shape[0]):
                if max_nfev is not None and nfev >= max_nfev:
                    vals[i:] = np.inf
                    break
                vals[i] = objective(pop[i])
            return vals

        def tournament_select(pop: np.ndarray, fit: np.ndarray) -> np.ndarray:
            k = min(self._tournament_size, pop.shape[0])
            idx = rng.choice(pop.shape[0], size=k, replace=False)
            best_local = idx[np.argmin(fit[idx])]
            return pop[best_local].copy()

        def crossover(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if rng.random() >= self._crossover_rate:
                return p1.copy(), p2.copy()
            alpha = rng.random(dim)
            c1 = alpha * p1 + (1.0 - alpha) * p2
            c2 = alpha * p2 + (1.0 - alpha) * p1
            c1 = np.clip(c1, lb, ub)
            c2 = np.clip(c2, lb, ub)
            return c1, c2

        def mutate(child: np.ndarray) -> np.ndarray:
            mask = rng.random(dim) < self._mutation_rate
            if not np.any(mask):
                return child

            child = child.copy()

            if self._mutation_mode == "log":
                # multiplicative mutation: x <- x * exp(noise)
                log_child = np.log(child)
                noise = rng.normal(loc=0.0, scale=self._mutation_scale, size=dim)
                log_child[mask] += noise[mask]
                child = np.exp(log_child)
            else:
                # relative mutation: x <- x * (1 + noise)
                noise = rng.normal(loc=0.0, scale=self._mutation_scale, size=dim)
                child[mask] *= (1.0 + noise[mask])

            return np.clip(child, lb, ub)

        def initialize_population() -> np.ndarray:
            pop = np.empty((self._population_size, dim), dtype=float)

            # Always keep theta0
            pop[0] = np.clip(theta0, lb, ub)

            n_local = int(round(self._init_near_theta0_fraction * (self._population_size - 1)))
            n_local = max(0, min(n_local, self._population_size - 1))
            n_global = (self._population_size - 1) - n_local

            cursor = 1

            # Local initialization around theta0
            for _ in range(n_local):
                candidate = theta0.copy()

                if self._mutation_mode == "log":
                    log_candidate = np.log(candidate)
                    noise = rng.normal(loc=0.0, scale=self._init_near_theta0_scale, size=dim)
                    log_candidate += noise
                    candidate = np.exp(log_candidate)
                else:
                    noise = rng.normal(loc=0.0, scale=self._init_near_theta0_scale, size=dim)
                    candidate = candidate * (1.0 + noise)

                pop[cursor] = np.clip(candidate, lb, ub)
                cursor += 1

            # Global initialization over the full search box
            for _ in range(n_global):
                pop[cursor] = rng.uniform(lb, ub, size=dim)
                cursor += 1

            return pop

        # Initial population
        population = initialize_population()
        fitness = evaluate_population(population)

        best_idx = int(np.argmin(fitness))
        best_theta = population[best_idx].copy()
        best_value = float(fitness[best_idx])

        stopped_early = max_nfev is not None and nfev >= max_nfev

        for _gen in range(self._n_generations):
            if stopped_early:
                break

            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]

            if float(fitness[0]) < best_value:
                best_value = float(fitness[0])
                best_theta = population[0].copy()

            new_population = [population[i].copy() for i in range(elite_count)]

            while len(new_population) < self._population_size:
                parent1 = tournament_select(population, fitness)
                parent2 = tournament_select(population, fitness)

                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)

                new_population.append(child1)
                if len(new_population) < self._population_size:
                    new_population.append(child2)

            population = np.asarray(new_population, dtype=float)
            fitness = evaluate_population(population)
            stopped_early = max_nfev is not None and nfev >= max_nfev

        # Keep best final individual
        final_best_idx = int(np.argmin(fitness))
        if float(fitness[final_best_idx]) < best_value:
            best_value = float(fitness[final_best_idx])
            best_theta = population[final_best_idx].copy()

        # Optional local refinement
        if self._polish and not stopped_early:
            def residuals(theta: np.ndarray) -> np.ndarray:
                res_parts: List[np.ndarray] = []
                for i, exp in enumerate(experiments):
                    sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                    res_parts.append(np.sqrt(w[i]) * (exp.y - sim_out))
                return np.concatenate(res_parts, axis=0)

            remaining_nfev = None if max_nfev is None else max(1, max_nfev - nfev)

            result = least_squares(
                residuals,
                x0=best_theta,
                bounds=(lb, ub),
                method=self._polish_method,
                loss=self._polish_loss,
                f_scale=self._polish_f_scale,
                max_nfev=remaining_nfev,
            )
            nfev += int(result.nfev)

            polished_value = float(2.0 * result.cost)
            if polished_value < best_value:
                best_value = polished_value
                best_theta = result.x.astype(float)
                ga_success = bool(result.success)
                ga_message = f"GA + polish: {result.message}"
            else:
                ga_success = True
                ga_message = "GA finished; local polish did not improve the objective."
        else:
            ga_success = not stopped_early
            ga_message = (
                "GA stopped because max_nfev was reached."
                if stopped_early
                else "GA finished successfully."
            )

        per_metrics: List[tuple[str, MetricsResult]] = []
        for exp in experiments:
            yhat = self._sim.simulate(exp.t, exp.u, best_theta).y
            per_metrics.append((exp.name, Metrics.compute(exp.y, yhat)))

        return CalibrationReport(
            theta_hat=np.asarray(best_theta, dtype=float),
            cost=0.5 * float(best_value),
            success=bool(ga_success),
            message=ga_message,
            nfev=int(nfev),
            per_experiment_metrics=per_metrics,
        )


class ParticleSwarmCalibrator:
    """
    Particle Swarm Optimization (PSO) calibrator.

    Each particle is a candidate parameter vector theta.
    The swarm evolves using:
      - inertia        : keeps previous motion
      - cognitive term : attraction toward particle's personal best
      - social term    : attraction toward global best

    Objective:
      minimize 0.5 * sum_i w_i * || y_i - sim(t_i,u_i;theta) ||^2
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        swarm_size: int = 40,
        n_iterations: int = 100,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        velocity_clamp: Optional[float] = 0.2,
        seed: Optional[int] = None,
        polish: bool = True,
        polish_method: str = "trf",
        polish_loss: str = "linear",
        polish_f_scale: float = 1.0,
    ) -> None:
        self._sim = simulator
        self._swarm_size = int(swarm_size)
        self._n_iterations = int(n_iterations)
        self._inertia = float(inertia)
        self._cognitive = float(cognitive)
        self._social = float(social)
        self._velocity_clamp = None if velocity_clamp is None else float(velocity_clamp)
        self._seed = seed
        self._polish = bool(polish)

        self._polish_method = polish_method
        self._polish_loss = polish_loss
        self._polish_f_scale = float(polish_f_scale)

        if self._swarm_size < 2:
            raise ValueError("swarm_size must be >= 2.")
        if self._n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")
        if self._inertia < 0:
            raise ValueError("inertia must be >= 0.")
        if self._cognitive < 0:
            raise ValueError("cognitive must be >= 0.")
        if self._social < 0:
            raise ValueError("social must be >= 0.")
        if self._velocity_clamp is not None and self._velocity_clamp <= 0:
            raise ValueError("velocity_clamp must be > 0 when provided.")

    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
        max_nfev: Optional[int] = None,
    ) -> CalibrationReport:
        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")

        theta0 = np.asarray(theta0, dtype=float)
        if theta0.ndim != 1:
            raise ValueError("theta0 must be a 1D array.")

        if weights is not None and len(weights) != len(experiments):
            raise ValueError("weights must match number of experiments.")
        w = np.ones(len(experiments), dtype=float) if weights is None else np.asarray(weights, dtype=float)

        if bounds is None:
            raise ValueError("ParticleSwarmCalibrator requires finite bounds.")
        lb = np.asarray(bounds[0], dtype=float)
        ub = np.asarray(bounds[1], dtype=float)

        if lb.shape != theta0.shape or ub.shape != theta0.shape:
            raise ValueError("bounds must have the same shape as theta0.")
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            raise ValueError("ParticleSwarmCalibrator requires finite bounds.")
        if np.any(lb >= ub):
            raise ValueError("Each lower bound must be strictly smaller than upper bound.")
        if np.any(theta0 < lb) or np.any(theta0 > ub):
            raise ValueError("Initial guess is outside of provided bounds.")

        rng = np.random.default_rng(self._seed)
        dim = theta0.size
        span = ub - lb

        nfev = 0

        def objective(theta: np.ndarray) -> float:
            nonlocal nfev
            nfev += 1

            total = 0.0
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                r = exp.y - sim_out
                total += w[i] * float(np.dot(r, r))
            return 0.5 * total

        def residuals(theta: np.ndarray) -> np.ndarray:
            parts: List[np.ndarray] = []
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                parts.append(np.sqrt(w[i]) * (exp.y - sim_out))
            return np.concatenate(parts, axis=0)

        # --------------------------------------------------
        # Swarm initialization
        # --------------------------------------------------
        positions = rng.uniform(lb, ub, size=(self._swarm_size, dim))
        positions[0] = theta0.copy()

        velocities = rng.uniform(-span, span, size=(self._swarm_size, dim)) * 0.1

        if self._velocity_clamp is not None:
            vmax = self._velocity_clamp * span
            velocities = np.clip(velocities, -vmax, vmax)
        else:
            vmax = None

        pbest_positions = positions.copy()
        pbest_costs = np.full(self._swarm_size, np.inf, dtype=float)

        for i in range(self._swarm_size):
            if max_nfev is not None and nfev >= max_nfev:
                break
            pbest_costs[i] = objective(positions[i])

        # In rare case max_nfev is tiny
        if np.any(~np.isfinite(pbest_costs)):
            # fallback safe init
            for i in range(self._swarm_size):
                if not np.isfinite(pbest_costs[i]):
                    pbest_costs[i] = np.inf

        gbest_idx = int(np.argmin(pbest_costs))
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_cost = float(pbest_costs[gbest_idx])

        stop_reason = "Maximum iterations reached."
        success = True

        # --------------------------------------------------
        # Main PSO loop
        # --------------------------------------------------
        for _ in range(self._n_iterations):
            if max_nfev is not None and nfev >= max_nfev:
                stop_reason = "Maximum number of function evaluations reached."
                success = True
                break

            r1 = rng.random(size=(self._swarm_size, dim))
            r2 = rng.random(size=(self._swarm_size, dim))

            cognitive_term = self._cognitive * r1 * (pbest_positions - positions)
            social_term = self._social * r2 * (gbest_position[None, :] - positions)

            velocities = self._inertia * velocities + cognitive_term + social_term

            if vmax is not None:
                velocities = np.clip(velocities, -vmax, vmax)

            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            for i in range(self._swarm_size):
                if max_nfev is not None and nfev >= max_nfev:
                    stop_reason = "Maximum number of function evaluations reached."
                    break

                cost_i = objective(positions[i])

                if cost_i < pbest_costs[i]:
                    pbest_costs[i] = cost_i
                    pbest_positions[i] = positions[i].copy()

                    if cost_i < gbest_cost:
                        gbest_cost = float(cost_i)
                        gbest_position = positions[i].copy()

            if max_nfev is not None and nfev >= max_nfev:
                break

        theta_hat = gbest_position.copy()
        final_cost = gbest_cost
        message = f"PSO finished. {stop_reason}"

        # --------------------------------------------------
        # Optional local refinement
        # --------------------------------------------------
        if self._polish and (max_nfev is None or nfev < max_nfev):
            remaining_nfev = None if max_nfev is None else max(max_nfev - nfev, 1)

            result = least_squares(
                residuals,
                theta_hat,
                bounds=(lb, ub),
                method=self._polish_method,
                loss=self._polish_loss,
                f_scale=self._polish_f_scale,
                max_nfev=remaining_nfev,
            )

            nfev += int(result.nfev)

            if float(result.cost) < final_cost:
                theta_hat = result.x.astype(float)
                final_cost = float(result.cost)
                success = bool(result.success)
                message = f"PSO + polish finished. {result.message}"
            else:
                message = "PSO finished. Local polish did not improve the solution."

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------
        per_metrics: List[tuple[str, MetricsResult]] = []
        for exp in experiments:
            yhat = self._sim.simulate(exp.t, exp.u, theta_hat).y
            per_metrics.append((exp.name, Metrics.compute(exp.y, yhat)))

        return CalibrationReport(
            theta_hat=np.asarray(theta_hat, dtype=float),
            cost=float(final_cost),
            success=bool(success),
            message=str(message),
            nfev=int(nfev),
            per_experiment_metrics=per_metrics,
        )



# -----------------------------------------------------------
# ------------- Deep learning Calibration -------------------
# -----------------------------------------------------------

@dataclass(frozen=True)
class NormalizationStats:
    x_mean: torch.Tensor          # [3]
    x_std: torch.Tensor           # [3]
    y_mean: torch.Tensor          # [d]
    y_std: torch.Tensor           # [d]
    calibrated_params: tuple[str, ...]
    transform_map: Dict[str, str]

    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized target back to log(y).
        """
        return y_norm * self.y_std + self.y_mean
    
    def normalize_y(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw - self.y_mean) / self.y_std

    def inverse_target_transform(self, y_transformed: torch.Tensor) -> torch.Tensor:
        """
        y_transformed: [..., d]
        returns physical parameters [..., d]
        """
        out = y_transformed.clone()
        for i, p in enumerate(self.calibrated_params):
            mode = self.transform_map.get(p, "identity")
            if mode == "log":
                out[..., i] = torch.exp(out[..., i])
        return out
    
    def forward_target_transform(self, y_physical: torch.Tensor) -> torch.Tensor:
        out = y_physical.clone()
        for i, p in enumerate(self.calibrated_params):
            mode = self.transform_map.get(p, "identity")
            if mode == "log":
                out[..., i] = torch.log(out[..., i])
        return out
    
    def y_norm_to_physical(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_transformed = self.denormalize_y(y_norm)
        return self.inverse_target_transform(y_transformed)


class RCNeuralCalibrator:
    """
    Neural inverse calibrator:
      input  : time(t), Vin(t), Vout(t)
      output : theta_hat vector following calibrated_params order

    Assumption:
      - the model was trained on log(x) with natural logarithm
      - input normalization stats come from the training split
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stats: NormalizationStats,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.stats = NormalizationStats(
            x_mean=stats.x_mean.to(self.device),
            x_std=stats.x_std.to(self.device),
            y_mean=stats.y_mean.to(self.device),
            y_std=stats.y_std.to(self.device),
            calibrated_params=tuple(stats.calibrated_params),
            transform_map=dict(stats.transform_map),
        )

    @property
    def calibrated_params(self) -> tuple[str, ...]:
        return self.stats.calibrated_params
    
    @staticmethod
    def load(checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> "RCNeuralCalibrator":
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        model_class = ckpt["model_class"]
        output_dim = len(tuple(ckpt["calibrated_params"]))

        if model_class == "RCInverseCNN":
            model = RCInverseCNN(output_dim=output_dim)
        elif model_class == "ProbabilisticRCInverseCNN":
            model = ProbabilisticRCInverseCNN(output_dim=output_dim)
        else:
            raise ValueError(f"Unsupported model_class in checkpoint: {model_class}")

        model.load_state_dict(ckpt["model_state_dict"])

        stats = NormalizationStats(
            x_mean=ckpt["x_mean"].float(),
            x_std=ckpt["x_std"].float(),
            y_mean=ckpt["y_mean"].float(),
            y_std=ckpt["y_std"].float(),
            calibrated_params=tuple(ckpt["calibrated_params"]),
            transform_map=dict(ckpt["transform_map"]),
        )

        return RCNeuralCalibrator(model=model, stats=stats, device=device)

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [3, T] or [B, 3, T]
        """
        if x.ndim == 2:
            return (x - self.stats.x_mean[:, None]) / self.stats.x_std[:, None]
        if x.ndim == 3:
            return (x - self.stats.x_mean[None, :, None]) / self.stats.x_std[None, :, None]
        raise ValueError(f"Expected x with shape [3,T] or [B,3,T], got {tuple(x.shape)}")

    def predict_logC(
        self,
        time: Union[np.ndarray, torch.Tensor],
        vin: Union[np.ndarray, torch.Tensor],
        vout: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Returns predicted list of params in log(x) after de-normalization.
        Useful for debugging.
        """
        if isinstance(vin, np.ndarray):
            vin_t = torch.tensor(vin, dtype=torch.float32)
        else:
            vin_t = vin.float()

        if isinstance(vout, np.ndarray):
            vout_t = torch.tensor(vout, dtype=torch.float32)
        else:
            vout_t = vout.float()
        
        if isinstance(time, np.ndarray):
            time_t = torch.tensor(time, dtype=torch.float32)
        else:
            time_t = time.float()

        if vin_t.ndim != 1 or vout_t.ndim != 1 or time_t.ndim != 1:
            raise ValueError("vin and vout and time must be 1D arrays (shape [T]).")
        if not (vin_t.shape[0] == vout_t.shape[0] == time_t.shape[0]):
            raise ValueError("vin and vout and time must have the same length T.")

        x = torch.stack([time_t, vin_t, vout_t], dim=0).to(self.device)  # [3, T]
        x = self._normalize_x(x)
        x = x.unsqueeze(0)  # [1, 3, T]

        with torch.no_grad():
            if isinstance(self.model, ProbabilisticRCInverseCNN):
                y_norm, _log_var = self.model(x)  # [1, d]
            else:
                y_norm = self.model(x)           # [1, d]

        y_phys = self.stats.y_norm_to_physical(y_norm).squeeze(0)  # [d]
        return y_phys.detach().cpu().numpy().astype(float)

    def predict_distribution(
        self,
        time: Union[np.ndarray, torch.Tensor],
        vin: Union[np.ndarray, torch.Tensor],
        vout: Union[np.ndarray, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(self.model, ProbabilisticRCInverseCNN):
            raise TypeError("predict_distribution is only available for ProbabilisticRCInverseCNN.")

        if isinstance(vin, np.ndarray):
            vin_t = torch.tensor(vin, dtype=torch.float32)
        else:
            vin_t = vin.float()

        if isinstance(vout, np.ndarray):
            vout_t = torch.tensor(vout, dtype=torch.float32)
        else:
            vout_t = vout.float()

        if isinstance(time, np.ndarray):
            time_t = torch.tensor(time, dtype=torch.float32)
        else:
            time_t = time.float()

        if vin_t.ndim != 1 or vout_t.ndim != 1 or time_t.ndim != 1:
            raise ValueError("vin, vout and time must be 1D arrays (shape [T]).")
        if not (vin_t.shape[0] == vout_t.shape[0] == time_t.shape[0]):
            raise ValueError("vin, vout and time must have the same length T.")

        x = torch.stack([time_t, vin_t, vout_t], dim=0).to(self.device) # [3, T]
        x = self._normalize_x(x)
        x = x.unsqueeze(0)  # [1, 3, T]
        
        with torch.no_grad():
            mu_norm, log_var_norm = self.model(x)   # [1, d], [1, d]

        mu_transformed = self.stats.denormalize_y(mu_norm)
        std_transformed = torch.sqrt(torch.exp(log_var_norm)) * self.stats.y_std[None, :]

        mu_physical = self.stats.inverse_target_transform(mu_transformed).squeeze(0).cpu().numpy()

        # approximation simple de std en espace physique
        std_physical = std_transformed.squeeze(0).cpu().numpy()

        return mu_physical.astype(float), std_physical.astype(float)
        
    def predict(
        self,
        time: Union[np.ndarray, torch.Tensor],
        vin: Union[np.ndarray, torch.Tensor],
        vout: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        vin, vout: arrays of shape [T]
        returns: C_hat in Farads
        """
        pred_C, _ = self.predict_distribution(time, vin, vout)
        return pred_C

    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
        max_nfev: Optional[int] = None,
    ) -> CalibrationReport:
        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")

        preds = []
        for exp in experiments:
            preds.append(self.predict_vector(exp.t, exp.u, exp.y))

        theta_hat = np.mean(np.stack(preds, axis=0), axis=0)

        return CalibrationReport(
            theta_hat=theta_hat.astype(float),
            cost=0.0,
            success=True,
            message="Neural calibration by averaged inverse predictions.",
            nfev=0,
            per_experiment_metrics=[],
        )