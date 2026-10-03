"""Regression checks for exact-center multiplicity evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.common import run_trial
from qfames.core import (Samples, initial_states, spectral_samples, filtered_matrix)
from qfames.plotting import multiplicity_plot


class MultiplicityTests(unittest.TestCase):
    def test_exact_centers_without_peak_search(self):
        cfg = json.loads(Path('configs/multiplicity.json').read_text())
        matrix = np.diag([8., 6., 4., 2.]).astype(complex)
        data = Samples(matrix[..., None], np.zeros((4, 4, 1)), 0., 0.)
        with patch('experiments.common.find_peaks', side_effect=AssertionError('peak search called')), patch('experiments.common.spectral_samples', return_value=data):
            row = run_trial(('multiplicity', cfg, 'independent', 500, 4, 0))
        self.assertEqual(row['evaluation_mode'], 'exact_cluster_centers')
        for c, theta in enumerate([0., .02]):
            self.assertEqual(row[f'evaluation_theta_{c}'], theta)
            self.assertEqual(row[f'peak_{c}'], '')
            np.testing.assert_allclose([row[f'sv_{c}_{k}'] for k in range(1, 5)], [2, 1.5, 1, .5])

    def test_quick_skips_reference_and_keeps_points(self):
        rows = [dict(L=L, failed=0, **{f'sv_{c}_{k}': (5-k)/L for c in range(2) for k in range(1,5)}) for L in [5, 8, 12]]
        with tempfile.TemporaryDirectory() as tmp, patch('qfames.plotting.finish') as finish:
            multiplicity_plot(rows, {'eigenvalues': [0,0,.02]}, Path(tmp), False)
            self.assertEqual(finish.call_count, 1)
            self.assertTrue(finish.call_args.args[1].name.endswith('_independent'))
            fig = finish.call_args.args[0]
            for ax in fig.axes:
                for line in ax.lines[:4]:
                    np.testing.assert_array_equal(line.get_xdata(), [5,8,12])
                self.assertEqual(len(ax.lines), 4)
                self.assertIn('multiplicity', ax.get_title())
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_scheme_curves_never_mix_and_reference_uses_tail(self):
        rows = [dict(scheme=s, L=L, failed=0,
                     **{f'sv_{c}_{k}': (1 if s == 'independent' else 10)*(5-k)/L
                        for c in range(2) for k in range(1,5)})
                for L in [50, 5, 24, 12] for s in ['shared', 'independent']]
        with patch('qfames.plotting.finish') as finish:
            multiplicity_plot(rows, {'eigenvalues': [0,0,.02]}, Path('.'), False)
            self.assertEqual(finish.call_count, 2)
            import matplotlib.pyplot as plt
            for call, scheme in zip(finish.call_args_list, ['independent', 'shared']):
                fig, path, _ = call.args
                self.assertEqual(path.name, f'multiplicity_singular_values_vs_state_count_{scheme}')
                self.assertEqual(len(fig.axes), 2)
                factor = 1 if scheme == 'independent' else 10
                for ax in fig.axes:
                    data = [line for line in ax.lines if line.get_marker() in ('o', 's')]
                    self.assertEqual(len(data), 4)
                    for line in data:
                        self.assertTrue(line.get_label().endswith(scheme))
                        np.testing.assert_array_equal(line.get_xdata(), [5,12,24,50])
                        k = int(line.get_label()[1])
                        np.testing.assert_allclose(line.get_ydata(), factor*(5-k)/line.get_xdata())
                    ref = next(line for line in ax.lines if line.get_linestyle() == ':')
                    x, y = ref.get_data()
                    np.testing.assert_array_equal(x, [24,50])
                    np.testing.assert_allclose(np.diff(np.log(y))/np.diff(np.log(x)), -.5)
                    self.assertAlmostEqual(np.mean(np.log(factor/x)-np.log(y)), 0)
                plt.close(fig)

    def test_paired_frames_and_distinct_sampling_streams(self):
        cfg = json.loads(Path('configs/multiplicity.json').read_text())
        frames = []
        rng_states = []
        for scheme in ['independent', 'shared']:
            def sample(rng, spectrum, phi, psi, *args):
                frames.append((phi.copy(), psi.copy()))
                rng_states.append(rng.bit_generator.state)
                return spectral_samples(rng, spectrum, phi, psi, *args)
            with patch('experiments.common.spectral_samples', side_effect=sample):
                run_trial(('multiplicity', cfg, scheme, 20, 5, 1))
        for a,b in zip(*frames):
            np.testing.assert_array_equal(a,b)
        self.assertNotEqual(rng_states[0], rng_states[1])

    def test_noiseless_shared_rank_at_most_three(self):
        phi, psi = initial_states(np.random.default_rng(12), 3, 8, 8)
        data = spectral_samples(np.random.default_rng(13), [0,0,.02], phi, psi, 31, 50, 4, 'shared', 0)
        for theta in [0,.02]:
            matrix = filtered_matrix(data,theta)
            self.assertLessEqual(np.linalg.matrix_rank(matrix, tol=1e-12),3)
            self.assertLess(np.linalg.svd(matrix, compute_uv=False)[3], 1e-12)
