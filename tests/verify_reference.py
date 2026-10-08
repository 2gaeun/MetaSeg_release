"""Optional migration regression: compare in-memory input with exported nnU-Net files."""
import argparse,json
from pathlib import Path
import numpy as np
from bms_deploy.geometry import load_nifti,prepare_pair
from bms_deploy.segment import Segmenter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--reference-inputs',type=Path,required=True)
    p.add_argument('--paired-samples-root',type=Path,required=True,
                   help='External historical pair root; bundled samples now contain real only')
    a=p.parse_args()
    segmenter=Segmenter(a.bundle/'weights/segmentation_123195',device='cpu')
    print('CHECKPOINT_STRICT_LOAD_OK epoch=1000',flush=True)
    for r in json.loads((a.bundle/'samples/manifest.json').read_text()):
        if r['case_id'] not in ('BrainMet_test_00001','BrainMet_test_00048'):continue
        folder=a.paired_samples_root/r['folder']
        data,props,info=prepare_pair(load_nifti(folder/'real.nii.gz'),load_nifti(folder/'synthetic.nii.gz'),r['sequence'])
        new,_,newprops=segmenter.preprocessor.run_case_npy(data,None,props,segmenter.plans,segmenter.cfg,segmenter.dataset)
        paths=[str(a.reference_inputs/f"{r['case_id']}_{ch:04d}.nii.gz") for ch in (0,1)]
        ref,_,refprops=segmenter.preprocessor.run_case(paths,None,segmenter.plans,segmenter.cfg,segmenter.dataset)
        np.testing.assert_allclose(new,ref,atol=1e-5,rtol=1e-6)
        for key in ('spacing','shape_before_cropping','shape_after_cropping_and_before_resampling','bbox_used_for_cropping'):
            np.testing.assert_array_equal(newprops[key],refprops[key])
        print('PREPROCESS_REFERENCE_MATCH',r['case_id'],new.shape,float(np.max(np.abs(new-ref))),flush=True)


if __name__=='__main__':main()
