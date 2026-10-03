"""Regression checks for exact-center multiplicity evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.common import run_trial
from qfames.core import Samples
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

    def test_sampling_noise_guide_fits_all_s4_and_keeps_points(self):
        rows = [dict(L=L, failed=0, **{f'sv_{c}_{k}': (5-k)/L for c in range(2) for k in range(1,5)}) for L in [5, 8, 12]]
        with tempfile.TemporaryDirectory() as tmp, patch('qfames.plotting.finish') as finish:
            multiplicity_plot(rows, {'eigenvalues': [0,0,.02]}, Path(tmp), False)
            fig = finish.call_args.args[0]
            for ax in fig.axes:
                for line in ax.lines[:4]:
                    np.testing.assert_array_equal(line.get_xdata(), [5,8,12])
                guide_x = ax.lines[-1].get_xdata()
                guide_y = ax.lines[-1].get_ydata()
                np.testing.assert_allclose(np.diff(np.log(guide_y)) / np.diff(np.log(guide_x)), -0.5)
                self.assertAlmostEqual(np.mean(np.log(1 / guide_x) - np.log(guide_y)), 0)
                self.assertIn('evaluated at the exact cluster centers', ax.get_title())
            import matplotlib.pyplot as plt
            plt.close(fig)
