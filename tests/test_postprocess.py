"""Native postprocessing and CLI wiring; no DA/lesion network inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from bms_deploy.geometry import Volume, load_nifti
from bms_deploy.postprocess import intersect, postprocess_native, SynthStrip
from bms_deploy import cli
from bms_deploy.dicom_input import output_id


class BrainGenerator:
    device = 'cpu'
    def generate(self, native, scratch, log_path, source_path=None):
        self.input = native.data.copy()
        self.source_path = source_path
        mask = np.ones(native.data.shape, dtype=np.uint8)
        mask[0] = 0
        Path(log_path).write_text('test brain generator\n')
        return Volume(mask, native.affine.copy())


class PostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.native = Volume(np.arange(120, dtype=np.float32).reshape(4, 5, 6),
                             np.array([[0, 0, -1.5, 30], [-.8, 0, 0, 20], [0, 1.2, 0, 10], [0, 0, 0, 1.]]))

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, debug=False):
        logs = self.root/'logs';logs.mkdir()
        raw = logs/'pending.nii.gz';nib.save(Volume(np.ones((4,5,6),dtype=np.uint8),self.native.affine).nifti(),raw)
        if debug:
            (logs/'debug').mkdir()
            nib.save(Volume(np.full((4,5,6),.75,dtype=np.float32),self.native.affine).nifti(),logs/'debug/native_probability.nii.gz')
        return raw,self.root/'final.nii.gz',logs

    def test_debug_off_filters_native_and_keeps_input_unchanged(self):
        raw,final,logs=self.stage();generator=BrainGenerator();before=self.native.data.copy()
        info=postprocess_native(self.native,raw,final,logs,generator)
        expected=np.ones((4,5,6),dtype=np.uint8);expected[0]=0
        np.testing.assert_array_equal(load_nifti(final).data,expected)
        np.testing.assert_array_equal(self.native.data,before)
        np.testing.assert_array_equal(generator.input,before)
        self.assertEqual(info['lesion_voxels_before'],120);self.assertEqual(info['lesion_voxels_after'],90)
        self.assertFalse(raw.exists());self.assertEqual(list(logs.rglob('*.nii.gz')),[])

    def test_debug_preserves_before_and_after_probability(self):
        raw,final,logs=self.stage(True)
        postprocess_native(self.native,raw,final,logs,BrainGenerator(),True)
        dbg=logs/'debug';after=load_nifti(final)
        np.testing.assert_array_equal(load_nifti(dbg/'native_probability.nii.gz').data>=.5,after.data>0)
        self.assertTrue((load_nifti(dbg/'native_mask_before_skullstrip.nii.gz').data==1).all())
        self.assertTrue((load_nifti(dbg/'native_probability_before_skullstrip.nii.gz').data==.75).all())
        np.testing.assert_array_equal(load_nifti(dbg/'native_mask.nii.gz').data,after.data)
        self.assertTrue((dbg/'brain_mask.nii.gz').is_file())

    def test_wrong_geometry_empty_or_nonbinary_mask_rejected(self):
        lesion=Volume(np.ones((4,5,6),dtype=np.uint8),self.native.affine)
        for brain in [Volume(np.ones((4,5,6)),np.eye(4)),Volume(np.zeros((4,5,6)),self.native.affine),Volume(np.full((4,5,6),.7),self.native.affine)]:
            with self.assertRaises(ValueError):intersect(lesion,brain)

    def test_failure_never_publishes_unfiltered_final_mask(self):
        raw,final,logs=self.stage();generator=BrainGenerator()
        with patch.object(generator,'generate',side_effect=RuntimeError('brain extraction failed')):
            with self.assertRaisesRegex(RuntimeError,'brain extraction failed'):
                postprocess_native(self.native,raw,final,logs,generator)
        self.assertFalse(final.exists());self.assertTrue(raw.exists())
        self.assertEqual(list(logs.glob('synthstrip_*')),[])

    def test_missing_optional_runtime_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError,'--skull-strip off'):
            SynthStrip(self.root/'missing')

    def run_cli(self, enabled, fail=False):
        inputs=self.root/'input';inputs.mkdir();nib.save(self.native.nifti(),inputs/'case.nii.gz')
        csv=inputs/'input.csv';csv.write_text('file_id,sequence\ncase,T1CE\n');out=self.root/'out'
        native=self.native
        class Adapter:
            def __init__(self,*a):pass
            def close(self):pass
            def generate(self, real, sequence, cid):
                np.testing.assert_array_equal(real.data,native.data)
                source=torch.from_numpy(real.data.transpose(2,1,0).copy())
                info=dict(transform={},training_job=112788,checkpoint_step=175000,nfe=50,seed=0)
                return Volume(np.zeros_like(real.data),real.affine),(source,source,real.affine,info)
        class Segmenter:
            def __init__(self,*a):pass
            def predict(self, real, synthetic, sequence, output, debug, **kwargs):
                logs=Path(output)/'logs';logs.mkdir(parents=True)
                path=kwargs.get('mask_output',Path(output).with_suffix('.nii.gz'))
                nib.save(Volume(np.ones_like(native.data,dtype=np.uint8),native.affine).nifti(),path)
                result=dict(status='completed',skull_stripping=False)
                (logs/'result.json').write_text(json.dumps(result));return result
        generator=BrainGenerator()
        if fail:generator.generate=lambda *a:(_ for _ in ()).throw(RuntimeError('brain extraction failed'))
        args=['run.sh','--input-root',str(inputs),'--csv',str(csv),'--input-format','nifti',
              '--weights',str(self.root),'--output',str(out),'--device','cpu','--skull-strip','on' if enabled else 'off']
        with patch('sys.argv',args),patch('bms_deploy.da_adapter.DomainAdapter',Adapter),patch('bms_deploy.segment.Segmenter',Segmenter),patch('bms_deploy.postprocess.SynthStrip',return_value=generator) as construct:
            if fail:
                with self.assertRaises(SystemExit):cli.main()
            else:cli.main()
            self.assertEqual(construct.call_count,int(enabled))
        return out,output_id('case'),generator

    def test_cli_off_needs_no_synthstrip(self):
        out,cid,_=self.run_cli(False)
        self.assertTrue((load_nifti(out/(cid+'.nii.gz')).data==1).all())
        self.assertEqual(json.loads((out/'manifest.json').read_text())['pipeline_version'],'v1')

    def test_cli_on_uses_original_file_and_filters_final(self):
        out,cid,generator=self.run_cli(True)
        self.assertEqual(generator.source_path,self.root/'input/case.nii.gz')
        self.assertEqual(int(load_nifti(out/(cid+'.nii.gz')).data.sum()),90)
        self.assertTrue(json.loads((out/cid/'logs/result.json').read_text())['skull_stripping'])

    def test_cli_failure_is_recorded_without_a_final_mask(self):
        out,cid,_=self.run_cli(True,True)
        self.assertFalse((out/(cid+'.nii.gz')).exists())
        self.assertEqual(json.loads((out/cid/'logs/result.json').read_text())['status'],'failed')
        self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()
