"""Checks for cost accounting, filter signs, and projected observable recovery."""
import unittest
import csv
from pathlib import Path
import tempfile
import numpy as np
from qfames.core import (Samples, filtered_matrix, initial_states, matching_error,
                         observable_eigenvalues, observable_filtered, sample_times,
                         spectral_samples, time_cost, find_peaks)


class NumericsTests(unittest.TestCase):
    def test_shared_cost_counts_all_state_pairs(self):
        times = np.array([-2.0, 1.0, -3.0])
        self.assertEqual(time_cost(times, 2, 4), 48)
        self.assertEqual(time_cost(np.broadcast_to(times, (2, 4, 3)), 2, 4), 48)

    def test_shared_and_independent_filter_shapes(self):
        rng = np.random.default_rng(1)
        t = rng.normal(size=7)
        z = rng.normal(size=(2, 3, 7)) + 1j * rng.normal(size=(2, 3, 7))
        expected = np.mean(z * np.exp(0.3j * t), axis=-1)
        for times in [t, np.broadcast_to(t, z.shape)]:
            np.testing.assert_allclose(filtered_matrix(Samples(z, times, 0, 0), .3, 3), expected)

    def test_truncated_sampler_and_known_characteristic_function(self):
        t = sample_times(np.random.default_rng(12), 100000, 2, 6)
        self.assertLessEqual(np.max(np.abs(t)), 12)
        self.assertAlmostEqual(np.mean(np.exp(.25j * t)).real, np.exp(-.25), delta=.008)

    def test_degenerate_observable_is_recovered_for_complex_overlaps(self):
        rng = np.random.default_rng(4)
        phi, psi = initial_states(rng, 3, 9, 8)
        target = np.array([[1.5, .5j], [-.5j, 1.5]])
        base = phi[:, :2] @ psi[:, :2].conj().T
        obs = phi[:, :2] @ target @ psi[:, :2].conj().T
        recovered, _ = observable_eigenvalues(base, obs, 2)
        self.assertLess(matching_error(recovered, [1, 2]), 1e-12)
        self.assertAlmostEqual(matching_error([1+.2j, 2], [1, 2]), .2)

    def test_observable_filter_and_two_leg_cost(self):
        # All energies coincide with theta: every phase cancels exactly.
        phi, psi = initial_states(np.random.default_rng(3), 2, 3, 4, replicated=True)
        observable = np.array([[1, .3j], [-.3j, 2]])
        for scheme in ["shared", "independent"]:
            seed = 9
            N, T, sigma = 11, 3., 6.
            result, cost, _ = observable_filtered(np.random.default_rng(seed), [.4, .4],
                phi, psi, observable, N, T, sigma, scheme, 0, .4, chunk_size=N)
            np.testing.assert_allclose(result, phi @ observable @ psi.conj().T, atol=1e-14)
            rng = np.random.default_rng(seed)
            shape = (N,) if scheme == "shared" else (3, 4, N)
            t = sample_times(rng, shape, T / np.sqrt(2), sigma)
            tp = sample_times(rng, shape, T / np.sqrt(2), sigma)
            self.assertAlmostEqual(cost, time_cost(t, 3, 4) + time_cost(tp, 3, 4))

    def test_noiseless_peak_search(self):
        phi, psi = initial_states(np.random.default_rng(1), 1, 2, 2, replicated=True)
        data = spectral_samples(np.random.default_rng(2), [.13], phi, psi, 100,
                                20, 6, "independent", 0)
        peaks = find_peaks(data, 20, 1, [-.2, .4])
        np.testing.assert_allclose(peaks, [.13], atol=1e-8)

    def test_failed_trials_remain_in_quantiles(self):
        from experiments.common import summarize
        rows = [dict(scheme="independent", N=10, L=4, trial=i,
                     status="ok" if i < 8 else "peak_failure",
                     error=.1 if i < 8 else np.inf, total_time=1,
                     spectral_time=1, observable_time=0, max_time=1,
                     **{f"sv_{c}_{k}": "" for c in range(2) for k in range(1, 6)})
                for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "trials.csv"
            with raw.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize(raw, Path(directory) / "summary.csv")[0]
            self.assertEqual(summary["failed"], 2)
            self.assertTrue(np.isinf(summary["error_q90"]))

    def test_infeasible_overlap_rejected(self):
        with self.assertRaises(ValueError):
            initial_states(np.random.default_rng(1), 4, 4, 4, min_overlap=.5)


if __name__ == "__main__":
    unittest.main()
