"""CPU geometry/regression checks without pretrained model loading."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
import SimpleITK as sitk
import torch
from bms_deploy.geometry import Volume, require_same_grid
from bms_deploy.da_adapter import MODELS, DomainAdapter, prepare, restore, to_sitk, from_sitk, network_affine
from bms_deploy._vendor.sampling_util import space_indices


class DATests(unittest.TestCase):
    def test_lps_ras_roundtrip(self):
        # Native SAR axis order with distinct spacing and nonzero origin.
        affine = np.array([[0,0,-.7,40],[0,1.2,0,-20],[2,0,0,5],[0,0,0,1.]])
        v = Volume(np.arange(6*9*10, dtype=np.float32).reshape(6,9,10), affine)
        r = from_sitk(to_sitk(v))
        require_same_grid(v, r)
        np.testing.assert_array_equal(v.data, r.data)

    def test_prepare_sar_and_native_restore(self):
        affine = np.array([[0,0,-.7,40],[0,1.2,0,-20],[2,0,0,5],[0,0,0,1.]])
        v = Volume(np.arange(6*9*10, dtype=np.float32).reshape(6,9,10), affine)
        before = v.data.copy()
        native, work, x, norm = prepare(v)
        self.assertEqual(work.GetSize(), (10,9,12))
        self.assertEqual(work.GetSpacing(), (.7,1.2,1.))
        self.assertEqual(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(work.GetDirection()), 'RAS')
        self.assertEqual(tuple(x.shape), (12,256,256))
        self.assertGreaterEqual(x.min().item(), -1.000001)
        self.assertLessEqual(x.max().item(), 1.000001)
        require_same_grid(v, restore(x, work, native, v.affine))
        np.testing.assert_array_equal(v.data, before)
        self.assertEqual(norm['percentiles'], [.1,99.9])

    def test_legacy_normalization_and_half_pixel_grid(self):
        rng = np.random.default_rng(42)
        v = Volume(rng.uniform(1,1000,(40,32,3)).astype(np.float32), np.eye(4))
        native, work, x, norm = prepare(v)
        raw = v.data.transpose(2,1,0)
        lo, hi = np.percentile(raw,[.1,99.9])
        normalized = (np.clip(raw,lo,hi)-lo)/(hi-lo)*2-1
        expected = torch.nn.functional.interpolate(torch.from_numpy(normalized)[:,None],
            (256,256),mode='bilinear',align_corners=False)[:,0]
        torch.testing.assert_close(x,expected,rtol=0,atol=0)
        a = network_affine(work)
        np.testing.assert_allclose(np.diag(a)[:3],[40/256,32/256,1])
        np.testing.assert_allclose(a[:3,3],[(40/256-1)/2,(32/256-1)/2,0])
        out = restore(torch.full_like(x,2.),work,native,v.affine)
        np.testing.assert_array_equal(out.data,np.ones(v.data.shape))

    def test_fixed_seed_independent_of_volume_id_and_caller_rng(self):
        class RandomInference:
            def __init__(self, directory, device):
                self.config = {'training_job': 112788, 'step': 175000}

            def sample(self, x1, cond, nfe):
                return torch.randn_like(x1)[:, 0]

            def release_gpu(self):
                pass

        real = Volume(np.arange(24, dtype=np.float32).reshape(3, 4, 2), np.eye(4))
        adapter = DomainAdapter('/unused', device='cpu')
        self.addCleanup(adapter.close)
        before = torch.random.get_rng_state().clone()
        with patch('bms_deploy.da_network.I2SBInference', RandomInference):
            _, first = adapter.generate(real, 'T1CE', 'first_id')
            _, renamed = adapter.generate(real, 'T1CE', 'renamed_id')
        torch.testing.assert_close(first[1], renamed[1], rtol=0, atol=0)
        expected = torch.randn((2, 1, 256, 256), generator=torch.Generator().manual_seed(0))[:, 0]
        torch.testing.assert_close(first[1], expected, rtol=0, atol=0)
        self.assertEqual(first[3]['seed'], 0)
        self.assertEqual(renamed[3]['seed'], 0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        with self.assertRaisesRegex(ValueError, 'fixed at 0'):
            DomainAdapter('/unused', seed=42)

    @staticmethod
    def fake_model(sequence):
        model = Mock()
        _, _, job, step = MODELS[sequence]
        model.config = {'training_job': job, 'step': step}
        model.sample.side_effect = lambda x1, cond, nfe: torch.randn_like(x1)[:, 0]
        return model

    def test_direction_cache_survives_switches_and_preserves_outputs(self):
        real = Volume(np.arange(24, dtype=np.float32).reshape(3, 4, 2), np.eye(4))
        adapter = DomainAdapter('/unused', device='cpu')
        self.addCleanup(adapter.close)
        t1ce, bb = self.fake_model('T1CE'), self.fake_model('BB')
        with patch('bms_deploy.da_network.I2SBInference', side_effect=[t1ce, bb]) as load:
            results = [adapter.generate(real, seq, str(i))[1][1]
                       for i, seq in enumerate(('T1CE', 'BB', 't1ce', 'BB'))]
            self.assertEqual(load.call_count, 2)
            self.assertEqual([call.args[0].name for call in load.call_args_list],
                             [MODELS['T1CE'][0], MODELS['BB'][0]])
        torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)
        torch.testing.assert_close(results[1], results[3], rtol=0, atol=0)
        t1ce.release_gpu.assert_not_called()
        bb.release_gpu.assert_not_called()
        adapter.close()
        adapter.close()
        t1ce.release_gpu.assert_called_once()
        bb.release_gpu.assert_called_once()
        fresh = self.fake_model('T1CE')
        with patch('bms_deploy.da_network.I2SBInference', return_value=fresh) as load:
            adapter.generate(real, 'T1CE', 'after_close')
            load.assert_called_once()

    def test_failed_direction_is_reloaded_without_evicting_other_direction(self):
        real = Volume(np.arange(24, dtype=np.float32).reshape(3, 4, 2), np.eye(4))
        adapter = DomainAdapter('/unused', device='cpu')
        self.addCleanup(adapter.close)
        t1ce, bb, replacement = [self.fake_model(seq) for seq in ('T1CE', 'BB', 'T1CE')]
        with patch('bms_deploy.da_network.I2SBInference', side_effect=[t1ce, bb, replacement]) as load:
            adapter.generate(real, 'T1CE', 'first')
            adapter.generate(real, 'BB', 'second')
            t1ce.sample.side_effect = RuntimeError('sampling failure')
            with self.assertRaisesRegex(RuntimeError, 'sampling failure'):
                adapter.generate(real, 'T1CE', 'failed')
            t1ce.release_gpu.assert_called_once()
            bb.release_gpu.assert_not_called()
            adapter.generate(real, 'BB', 'still_cached')
            self.assertEqual(load.call_count, 2)
            adapter.generate(real, 'T1CE', 'retry')
            self.assertEqual(load.call_count, 3)

    def test_wrong_checkpoint_is_released_and_never_cached(self):
        real = Volume(np.arange(24, dtype=np.float32).reshape(3, 4, 2), np.eye(4))
        adapter = DomainAdapter('/unused', device='cpu')
        self.addCleanup(adapter.close)
        wrong, correct = self.fake_model('BB'), self.fake_model('T1CE')
        with patch('bms_deploy.da_network.I2SBInference', side_effect=[wrong, correct]) as load:
            with self.assertRaisesRegex(ValueError, 'Wrong direction/checkpoint'):
                adapter.generate(real, 'T1CE', 'wrong')
            wrong.release_gpu.assert_called_once()
            adapter.generate(real, 'T1CE', 'correct')
            self.assertEqual(load.call_count, 2)

    def test_sampling_schedule(self):
        steps=space_indices(1000,51)
        self.assertEqual(len(steps),51)
        self.assertEqual((steps[0],steps[-1]),(0,999))
        self.assertTrue(all(a<b for a,b in zip(steps,steps[1:])))


if __name__ == '__main__': unittest.main()
