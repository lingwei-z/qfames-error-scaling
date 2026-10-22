import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from qfames.core import observable_initial_states, spectral_samples, observable_filtered, observable_target_peak, Samples
from experiments.common import run_trial

class ObservableTests(unittest.TestCase):
    def test_frame_properties(self):
        for L,R in [(3,5),(8,12),(50,32)]:
            for n, frame in zip((L,R), observable_initial_states(L,R)):
                np.testing.assert_allclose(np.linalg.norm(frame,axis=1),1)
                np.testing.assert_allclose(frame[:,:2].conj().T @ frame[:,:2], n/3*np.eye(2),atol=1e-13)
                np.testing.assert_array_equal(frame[:,0],frame[:,2])

    def test_time_sharing_and_cost(self):
        phi,psi=observable_initial_states(3,5)
        for scheme in ['shared','independent']:
            d=spectral_samples(np.random.default_rng(1),[0,0,.1],phi,psi,7,50,6,scheme,.015)
            self.assertEqual(d.times.shape,(7,) if scheme=='shared' else (3,5,7))
            self.assertAlmostEqual(d.total_time,2*np.abs(d.times).sum()*(15 if scheme=='shared' else 1))
            if scheme=='independent': self.assertFalse(np.array_equal(d.times[0,0],d.times[1,0]))
            from qfames.core import sample_times
            with patch('qfames.core.sample_times', wraps=sample_times) as sample:
                observable_filtered(np.random.default_rng(2),[0,0,.1],phi,psi,np.diag([1,2,0]),7,50,6,scheme,.015,.001,7)
                self.assertEqual([c.args[1] for c in sample.call_args_list],[(7,) if scheme=='shared' else (3,5,7)]*2)

    def test_local_peak_can_be_nonzero(self):
        t=np.linspace(-100,100,301)
        data=Samples(np.exp(-.012j*t)[None,None,:],t,0,0)
        opt=observable_target_peak(data)
        self.assertTrue(opt.success)
        self.assertAlmostEqual(opt.x,.012,places=8)

    def test_configured_search_bounds(self):
        cfg=json.loads(Path('configs/observable.json').read_text())
        cfg['search_bounds'] = [-.02, .03]
        with patch('qfames.core.minimize_scalar', return_value=SimpleNamespace(success=False,x=0,fun=0,nfev=0,message='test')) as optimizer:
            run_trial(('observable',cfg,'shared',7,3,0))
        self.assertEqual(optimizer.call_args.kwargs['bounds'], [-.02, .03])
        self.assertEqual(optimizer.call_args.kwargs['options']['xatol'], 1e-10)

    def test_trial_uses_same_frames_and_estimated_theta(self):
        cfg=json.loads(Path('configs/observable.json').read_text())
        from experiments import common
        frames=[]
        for scheme in ['shared','independent']:
            with patch('experiments.common.initial_states',side_effect=AssertionError('generic states')), patch('experiments.common.find_peaks',side_effect=AssertionError('global search')), patch('experiments.common.observable_target_peak',return_value=SimpleNamespace(success=True,x=.003,fun=-1,nfev=5,message='ok')), patch('experiments.common.spectral_samples',wraps=common.spectral_samples) as spectral, patch('experiments.common.filtered_matrix',wraps=common.filtered_matrix) as matrix, patch('experiments.common.observable_filtered',wraps=common.observable_filtered) as numerator:
                row=run_trial(('observable',cfg,scheme,7,3,0))
                frames.append(spectral.call_args.args[2:4])
                self.assertEqual(matrix.call_args.args[1],.003)
                self.assertEqual(numerator.call_args.args[10],.003)
                self.assertEqual(row['peak_0'],.003)
                self.assertEqual(row['peak_1'],'')
                self.assertEqual(row['total_time'],row['spectral_time']+row['observable_time'])
        for a,b in zip(*frames): np.testing.assert_array_equal(a,b)

    def test_failed_local_search_preserves_spectral_cost(self):
        cfg=json.loads(Path('configs/observable.json').read_text())
        with patch('experiments.common.observable_target_peak',return_value=SimpleNamespace(success=False,x=0,fun=0,nfev=2,message='failed')), patch('experiments.common.observable_filtered') as obs:
            row=run_trial(('observable',cfg,'shared',7,3,0))
        self.assertEqual(row['status'],'peak_failure')
        self.assertTrue(np.isinf(row['error']))
        self.assertGreater(row['spectral_time'],0)
        self.assertEqual(row['total_time'],row['spectral_time'])
        obs.assert_not_called()
