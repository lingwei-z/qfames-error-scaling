"""Regression checks for estimated-center multiplicity evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.common import run_trial
from qfames.core import (Samples, initial_states, spectral_samples, filtered_matrix)
from qfames.plotting import multiplicity_plot, threshold_range


class MultiplicityTests(unittest.TestCase):
    def test_threshold_inset_band_and_fixed_value(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            x = np.array([5., 10., 20., 40.])
            lower = np.array([.01, .02, .03, np.nan])
            upper = np.array([.04, .08, .03, .09])
            threshold_range(ax, x, lower, upper)
            low = ax.lines[0].get_ydata()
            high = ax.lines[1].get_ydata()
            self.assertTrue(np.all(lower[:2] < low[:2]))
            self.assertTrue(np.all(low[:2] < high[:2]))
            self.assertTrue(np.all(high[:2] < upper[:2]))
            np.testing.assert_allclose(np.log(low[:2]/lower[:2])/np.log(upper[:2]/lower[:2]), .15)
            self.assertTrue(np.all(np.isnan(low[2:])))
            self.assertEqual(ax.lines[0].get_linestyle(), '--')
            self.assertEqual(ax.lines[1].get_linestyle(), '--')
            self.assertEqual(len(ax.collections), 1)
            handles = threshold_range(ax, x, lower, upper, tau=.04)
            np.testing.assert_allclose(handles[-1].get_ydata(), [.04]*4)
            # The true upper endpoint is excluded, not the inset boundary.
            np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [5.])
        finally:
            plt.close(fig)

    def test_svd_uses_estimated_peaks(self):
        cfg = json.loads(Path('configs/multiplicity.json').read_text())
        matrix = np.diag([10., 8., 6., 4., 2.]).astype(complex)
        data = Samples(matrix[..., None], np.zeros((5, 5, 1)), 0., 0.)
        with patch('experiments.common.find_peaks', return_value=np.array([.001, .019])), patch('experiments.common.spectral_samples', return_value=data), patch('experiments.common.filtered_matrix', return_value=matrix) as filtered:
            row = run_trial(('multiplicity', cfg, 'independent', 500, 5, 0))
        self.assertEqual(row['evaluation_mode'], 'estimated_cluster_centers')
        for c, theta in enumerate([.001, .019]):
            self.assertEqual(row[f'evaluation_theta_{c}'], theta)
            self.assertEqual(row[f'peak_{c}'], theta)
            self.assertEqual(filtered.call_args_list[c].args[1], theta)
            np.testing.assert_allclose([row[f'sv_{c}_{k}'] for k in range(1, 6)], [2, 1.6, 1.2, .8, .4])

    def test_quick_skips_reference_and_keeps_points(self):
        rows = [dict(L=L, failed=0, **{f'sv_{c}_{k}': (6-k)/L for c in range(2) for k in range(1,6)}) for L in [5, 8, 12]]
        with tempfile.TemporaryDirectory() as tmp, patch('qfames.plotting.finish') as finish:
            multiplicity_plot(rows, {'eigenvalues': [0,0,.02]}, Path(tmp), False)
            self.assertEqual(finish.call_count, 1)
            self.assertTrue(finish.call_args.args[1].name.endswith('_independent'))
            fig = finish.call_args.args[0]
            for ax in fig.axes:
                for line in ax.lines[:5]:
                    np.testing.assert_array_equal(line.get_xdata(), [5,8,12])
                self.assertEqual(sum(line.get_label().startswith("s") for line in ax.lines), 5)
                self.assertIn('multiplicity', ax.get_title())
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_scheme_curves_never_mix_and_reference_uses_tail(self):
        rows = [dict(scheme=s, L=L, failed=0,
                     **{f'sv_{c}_{k}': (1 if s == 'independent' else 10)*(6-k)/L
                        for c in range(2) for k in range(1,6)})
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
                    np.testing.assert_allclose(ax.get_ylim(), [2e-3, .4])
                    data = [line for line in ax.lines if line.get_marker() in ('o', 's')]
                    self.assertEqual(len(data), 5)
                    for line in data:
                        self.assertTrue(line.get_label().endswith(scheme))
                        np.testing.assert_array_equal(line.get_xdata(), [5,12,24,50])
                        k = int(line.get_label()[1])
                        np.testing.assert_allclose(line.get_ydata(), factor*(6-k)/line.get_xdata())
                    if scheme == 'shared':
                        self.assertEqual(sum(line.get_label().startswith("s") for line in ax.lines), 5)
                        self.assertIsNone(ax.get_legend())
                        continue
                    ref = next(line for line in ax.lines if line.get_linestyle() == ':')
                    x, y = ref.get_data()
                    np.testing.assert_array_equal(x, [24,50])
                    np.testing.assert_allclose(np.diff(np.log(y))/np.diff(np.log(x)), -.5)
                    self.assertAlmostEqual(np.mean(np.log(2*factor/x)-np.log(y)), 0)
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

    def test_five_value_trial_summary_and_plot(self):
        import csv
        from experiments.common import FIELDS, EVALUATION_FIELDS, summarize, validate_config
        from qfames.plotting import plot_run
        cfg = json.loads(Path('configs/multiplicity.json').read_text())
        validate_config('multiplicity', cfg)
        cfg['L_values'] = [4]
        with self.assertRaises(ValueError):
            validate_config('multiplicity', cfg)
        cfg['L_values'] = [5]
        rows = [run_trial(('multiplicity', cfg, scheme, 50, 5, i))
                for scheme in ['independent', 'shared'] for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with (path / 'trials.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS + EVALUATION_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize(path / 'trials.csv', path / 'summary.csv')
            for row in summary:
                group = [r for r in rows if r['scheme'] == row['scheme']]
                for c in range(2):
                    field = f'sv_{c}_5'
                    self.assertGreater(row[field], 0)
                    self.assertAlmostEqual(row[field], np.mean([r[field] for r in group]))
            (path / 'metadata.json').write_text(json.dumps(dict(
                experiment='multiplicity', config=cfg, quick=True,
                evaluation_mode='estimated_cluster_centers')))
            plot_run(path)
            self.assertEqual(len(list((path / 'figures').glob('*.png'))), 2)
            self.assertEqual(len(list((path / 'figures').glob('*.pdf'))), 2)
