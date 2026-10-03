"""Gaussian-filtered spectral and observable estimators.

Rows of phi and psi contain bra overlaps <phi_l|E_m> and <psi_r|E_m>.
The noise model is additive complex Gaussian noise, not circuit outcomes.
"""
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm


@dataclass
class Samples:
    values: np.ndarray
    times: np.ndarray
    total_time: float
    max_time: float


def sample_times(rng, size, T, sigma):
    """Sample a_T(t) proportional to exp(-t**2/(4*T**2)), |t| <= sigma*T."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    return truncnorm.rvs(-sigma / np.sqrt(2), sigma / np.sqrt(2),
                         scale=np.sqrt(2) * T, size=size, random_state=rng)


def time_cost(times, L, R):
    """Count every state-pair evolution, including repeated shared times."""
    factor = L * R if times.ndim == 1 else 1
    return float(factor * np.abs(times).sum())


def initial_states(rng, dimension, L, R, replicated=False, min_overlap=0.5):
    if min(L, R, dimension) < 1:
        raise ValueError("State counts and dimension must be positive")
    if replicated:
        # Fix the spectral weights across all trials and state counts.
        v = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
        return np.tile(v, (L, 1)), np.tile(v, (R, 1))
    if not 0 <= min_overlap < 1 / np.sqrt(dimension):
        raise ValueError("The strict minimum-overlap constraint is infeasible")
    rows = []
    for _ in range(L + R):
        for attempt in range(100000):
            v = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            v /= np.linalg.norm(v)
            if np.min(np.abs(v)) > min_overlap:
                rows.append(v)
                break
        else:
            raise RuntimeError("Initial-state rejection sampler exhausted its limit")
    return np.asarray(rows[:L]), np.asarray(rows[L:])


def observable_initial_states(L, R):
    """Deterministic frames with orthogonal target columns and fixed cross-cluster correlation."""
    def frame(n):
        phase = np.exp(2j * np.pi * np.arange(n) / n)
        return np.column_stack([np.ones(n), phase, np.ones(n)]) / np.sqrt(3)
    return frame(L), frame(R)


def observable_target_peak(samples, search_bounds=(-0.04, 0.04)):
    """Localize the target in its known neighborhood; return optimizer diagnostics."""
    return minimize_scalar(
        lambda theta: -np.linalg.norm(filtered_matrix(samples, theta), "fro"),
        bounds=search_bounds, method="bounded", options={"xatol": 1e-10})


def spectral_samples(rng, eigenvalues, phi, psi, N, T, sigma, scheme, noise_std):
    L, R = len(phi), len(psi)
    if scheme not in ("independent", "shared") or N < 2 or noise_std < 0:
        raise ValueError("Invalid sampling scheme, sample count, or noise scale")
    shape = (L, R, N)
    times = sample_times(rng, N if scheme == "shared" else shape, T, sigma)
    values = np.zeros(shape, dtype=complex)
    # Sum over eigenstates without materializing an L x R x M x N tensor.
    for m, value in enumerate(eigenvalues):
        coefficient = phi[:, m, None] * psi[None, :, m].conj()
        values += coefficient[..., None] * np.exp(-1j * value * times)
    values += noise_std * rng.normal(size=shape)
    values += 1j * noise_std * rng.normal(size=shape)
    return Samples(values, times, time_cost(times, L, R), float(np.max(np.abs(times))))


def filtered_matrix(samples, theta, chunk_size=512):
    """Evaluate G(theta), with bounded temporary arrays."""
    values, times = samples.values, samples.times
    result = np.zeros(values.shape[:2], dtype=complex)
    for start in range(0, values.shape[-1], chunk_size):
        sl = slice(start, start + chunk_size)
        result += np.sum(values[..., sl] * np.exp(1j * theta * times[..., sl]), axis=-1)
    return result / values.shape[-1]


def find_peaks(samples, T, count, bounds, grid_step=0.4, alpha=2.0,
               threshold=0.1, xatol=1e-10):
    """Coarse search plus local refinement of the same Frobenius objective.

    grid_step is dimensionless: the actual grid spacing is at most grid_step/T.
    threshold applies to ||G||_F / sqrt(LR), so it does not change with L or R.
    Bounds are a declared spectral search interval, not true-peak locations.
    """
    low, high = bounds
    if low >= high or grid_step <= 0 or alpha <= grid_step or count < 1:
        raise ValueError("Invalid peak-search parameters")
    grid = np.linspace(low, high, max(3, int(np.ceil((high - low) * T / grid_step)) + 1))
    scale = np.sqrt(np.prod(samples.values.shape[:2]))

    def score(theta):
        return np.linalg.norm(filtered_matrix(samples, theta), "fro") / scale

    scores = np.array([score(theta) for theta in grid])
    peaks = []
    for _ in range(count):
        available = np.ones(len(grid), dtype=bool)
        for peak in peaks:
            available &= np.abs(grid - peak) > alpha / T
        masked = np.where(available, scores, -np.inf)
        index = int(np.argmax(masked))
        if masked[index] < threshold:
            break
        a, b = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
        # Keep refinement in the same unblocked component as the grid maximum.
        for peak in peaks:
            if grid[index] < peak:
                b = min(b, peak - alpha / T)
            else:
                a = max(a, peak + alpha / T)
        if a >= b:
            break
        opt = minimize_scalar(lambda theta: -score(theta), bounds=(a, b),
                              method="bounded", options={"xatol": xatol})
        if not opt.success:
            break
        candidates = [grid[index], opt.x, a, b]
        peak = max(candidates, key=score)
        if score(peak) < threshold:
            break
        peaks.append(peak)
    return np.sort(peaks)


def observable_filtered(rng, eigenvalues, phi, psi, observable, N, T, sigma,
                        scheme, noise_std, theta, chunk_size=256):
    """Stream G^O and count both legs of each observable circuit.

    The two times are independent draws at scale T/sqrt(2). In the shared
    scheme each time pair is reused across (l,r), but noise remains independent.
    """
    if scheme not in ("shared", "independent") or N < 2 or noise_std < 0:
        raise ValueError("Invalid observable sampling parameters")
    L, R = len(phi), len(psi)
    total, maximum = 0.0, 0.0
    result = np.zeros((L, R), dtype=complex)
    for start in range(0, N, chunk_size):
        n = min(chunk_size, N - start)
        shape = (n,) if scheme == "shared" else (L, R, n)
        t = sample_times(rng, shape, T / np.sqrt(2), sigma)
        tp = sample_times(rng, shape, T / np.sqrt(2), sigma)
        values = np.zeros((L, R, n), dtype=complex)
        for m, lm in enumerate(eigenvalues):
            for k, lk in enumerate(eigenvalues):
                if observable[m, k] == 0:
                    continue
                coefficient = phi[:, m, None] * observable[m, k] * psi[None, :, k].conj()
                values += coefficient[..., None] * np.exp(1j * lm * tp - 1j * lk * t)
        values += noise_std * rng.normal(size=values.shape)
        values += 1j * noise_std * rng.normal(size=values.shape)
        result += np.sum(values * np.exp(1j * theta * (t - tp)), axis=-1)
        total += time_cost(t, L, R) + time_cost(tp, L, R)
        maximum = max(maximum, float(np.max(np.abs(t) + np.abs(tp))))
    return result / N, total, maximum


def observable_eigenvalues(base, filtered_observable, multiplicity):
    """Use the leading singular subspace and a generalized eigensolver."""
    u, s, vh = np.linalg.svd(base, full_matrices=False)
    m = multiplicity
    if not 1 <= m <= len(s):
        raise ValueError("Invalid target multiplicity")
    if s[m - 1] <= np.finfo(float).eps * max(base.shape) * s[0]:
        raise np.linalg.LinAlgError("Projected denominator is numerically singular")
    projected = u[:, :m].conj().T @ filtered_observable @ vh[:m].conj().T
    estimates = eigvals(projected, np.diag(s[:m]))
    if not np.all(np.isfinite(estimates)):
        raise np.linalg.LinAlgError("Nonfinite generalized eigenvalues")
    return estimates, float(s[0] / s[m - 1])


def matching_error(estimates, targets):
    """Bottleneck matching in the complex plane for the small toy spectra."""
    estimates, targets = np.asarray(estimates), np.asarray(targets)
    if len(estimates) != len(targets) or not np.all(np.isfinite(estimates)):
        return np.inf
    if len(targets) > 8:
        raise ValueError("Exact permutation matching is intended for small spectra")
    return float(min(np.max(np.abs(estimates[list(p)] - targets))
                     for p in permutations(range(len(targets)))))
