"""Validate portable inference outputs, optionally against historical native probabilities."""
import argparse,json
from pathlib import Path
import nibabel as nib
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-probabilities',type=Path)
    a=p.parse_args()
    manifest=json.loads((a.output/'manifest.json').read_text())
    assert manifest['status']=='completed' and len(manifest['volumes'])>0
    sample_rows=json.loads((a.bundle/'samples/manifest.json').read_text())
    samples_by_id={r['case_id']:r for r in sample_rows}
    summary=[]
    for r in manifest['volumes']:
        case_id=r.get('file_id') or Path(r['folder']).name
        real=nib.load(a.bundle/'samples'/samples_by_id[case_id]['file'])
        pred=nib.load(a.output/(r['id']+'.nii.gz'))
        assert pred.shape==real.shape
        np.testing.assert_allclose(pred.affine,real.affine,atol=1e-4,rtol=0)
        mask=np.asarray(pred.dataobj)>0
        assert set(np.unique(np.asarray(pred.dataobj))) <= {0,1}
        result={'file_id':case_id,'native_geometry_match':True,'foreground_voxels':int(mask.sum())}
        if a.reference_probabilities:
            ref=nib.load(a.reference_probabilities/(case_id+'.nii.gz'))
            assert ref.shape==real.shape
            np.testing.assert_allclose(ref.affine,real.affine,atol=1e-4,rtol=0)
            old=ref.get_fdata(dtype=np.float32)>=.5
            n=int(mask.sum()+old.sum())
            result.update(reference_mask_dice=float(2*np.count_nonzero(mask&old)/n) if n else 1.,
                          different_voxels=int(np.count_nonzero(mask!=old)))
        if manifest['debug']:
            debug=a.output/r['id']/'logs/debug'
            model=nib.load(debug/'model_mask.nii.gz')
            np.testing.assert_allclose(model.header.get_zooms(),[1,1,1],atol=1e-5)
            assert (debug/'geometry.json').is_file()
        else:assert not (a.output/r['id']/'logs/debug').exists()
        summary.append(result);print('OUTPUT_VERIFIED',json.dumps(result),flush=True)
    (a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
