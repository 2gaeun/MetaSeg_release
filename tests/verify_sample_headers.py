"""Audit real sample headers; check synthetic geometry only if separately supplied."""
import argparse,json,sys
from pathlib import Path
import nibabel as nib
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.bundle/'samples/manifest.json').read_text())
    counts={'1mm':0,'LE0p5mm':0}
    for row in manifest:
        folder=a.bundle/'samples'/row['folder']
        real=nib.load(a.bundle/'samples'/row['file'])
        assert len(real.shape)==3,row['folder']
        assert np.isfinite(real.affine).all(),row['folder']
        if (folder/'synthetic.nii.gz').exists():
            synthetic=nib.load(folder/'synthetic.nii.gz')
            assert real.shape==synthetic.shape,row['folder']
            np.testing.assert_allclose(real.affine,synthetic.affine,atol=1e-4,rtol=0)
        counts[row['folder'].split('/')[0]]+=1
    assert counts=={'1mm':31,'LE0p5mm':47},counts
    print('ALL_REAL_SAMPLE_HEADERS_VALID',counts,flush=True)
    from bms_deploy.segment import Segmenter
    model=Segmenter(a.bundle/'weights/segmentation_123195',device='cpu')
    assert not any(k=='brainmetaseg' or k.startswith('brainmetaseg.') for k in sys.modules)
    print('NO_ORIGINAL_BRAINMETASEG_MODULE_IMPORTED',flush=True)


if __name__=='__main__':main()
