"""GPU smoke: real DA weights + segmentation, synthetic classic-MR fixtures.

This is an engineering integration test, not clinical accuracy validation.
Only this bundle and a writable output directory are required.
"""
import argparse
import json
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from bms_deploy.dicom_input import scan_series, read_series, output_id


def fixture(root):
    yy, xx = np.mgrid[:64,:64]
    for series, sequence in enumerate(('T1CE','BB'),1):
        folder=root/sequence; folder.mkdir(parents=True)
        for z in range(3):
            pixels=(1500*np.exp(-((xx-32)**2+(yy-32)**2)/250)*(1+z*.05)).astype(np.int16)
            pixels[pixels<10]=0
            image=sitk.GetImageFromArray(pixels); image.SetSpacing([1.,1.])
            tags={'0008|0060':'MR','0008|0016':'1.2.840.10008.5.1.4.1.1.4',
                '0008|0018':f'1.2.826.0.1.3680043.10.999.42.{series}.{z+1}',
                '0020|000d':'1.2.826.0.1.3680043.10.999.42',
                '0020|000e':f'1.2.826.0.1.3680043.10.999.42.{series}',
                '0020|0032':f'10\\20\\{30+z}', '0020|0037':'1\\0\\0\\0\\1\\0',
                '0028|0030':'1\\1','0020|0013':str(z+1)}
            for key,value in tags.items(): image.SetMetaData(key,value)
            writer=sitk.ImageFileWriter(); writer.KeepOriginalImageUIDOn()
            writer.SetFileName(str(folder/f'{2-z}.dcm')); writer.Execute(image)
    (root/'input.csv').write_text('folder,sequence\nT1CE,T1CE\nBB,BB\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'dicom'; fixture(source)
    report={}
    for debug in (True,False):
        target=a.output/('debug_on' if debug else 'debug_off')
        command=['bash',str(a.bundle/'run.sh'),'--input-root',str(source),'--csv',str(source/'input.csv'),
                 '--output',str(target),'--da-batch-size','3','--da-nfe','50']
        if debug: command.append('--debug')
        subprocess.run(command,check=True)
        manifest=json.loads((target/'manifest.json').read_text())
        assert manifest['status']=='completed' and len(manifest['volumes'])==2
        for entry in manifest['volumes']:
            cid=entry['id']; folder=target/cid
            _,files=scan_series(source/entry['folder'])[0]; real=read_series(files)
            mask=nib.load(target/(cid+'.nii.gz'))
            assert mask.shape==real.data.shape
            np.testing.assert_allclose(mask.affine,real.affine,atol=1e-4,rtol=0)
            assert set(np.unique(np.asarray(mask.dataobj))) <= {0,1}
            expected=(112788,175000) if entry['sequence']=='T1CE' else (114599,150000)
            da=json.loads((folder/'logs/da_result.json').read_text())
            assert (da['training_job'],da['checkpoint_step'])==expected
            assert da['seed']==0
            if debug:
                d=folder/'logs/debug'
                for name in ['real_pseudo','synthetic_pseudo','seg_input_0000','seg_input_0001',
                             'model_input_0000','model_input_0001','model_mask','model_probability',
                             'da_input_network','da_output_network']:
                    im=nib.load(d/(name+'.nii.gz')); assert np.isfinite(im.get_fdata()).all()
                synthetic=nib.load(d/'synthetic_pseudo.nii.gz')
                assert synthetic.shape==(256,256,3)
                np.testing.assert_allclose(synthetic.affine,nib.load(d/'real_pseudo.nii.gz').affine,atol=1e-4,rtol=0)
                assert nib.load(d/'da_output_network.nii.gz').shape==(256,256,3)
                assert not np.allclose(synthetic.get_fdata(),0)
            else:
                assert not list(folder.rglob('*.nii.gz'))
                on=nib.load(a.output/'debug_on'/(cid+'.nii.gz'))
                np.testing.assert_array_equal(np.asarray(mask.dataobj),np.asarray(on.dataobj))
        report['debug_on' if debug else 'debug_off']={'volumes':2,'nfe':50,'passed':True}
    assert not list(source.rglob('*.nii*'))
    # The same voxel data and fixed seed must agree across input formats and IDs.
    nifti_source=a.output/'nifti'; nifti_source.mkdir()
    for sequence, extension in (('T1CE','.nii.gz'),('BB','.nii')):
        _,files=scan_series(source/sequence)[0]
        nib.save(read_series(files).nifti(),nifti_source/(sequence+extension))
    csv=nifti_source/'input.csv'; csv.write_text('file_id,sequence\nT1CE,T1CE\nBB,BB\n')
    target=a.output/'nifti_debug_on'
    subprocess.run(['bash',str(a.bundle/'run.sh'),'--input-format','nifti',
        '--input-root',str(nifti_source),'--csv',str(csv),'--output',str(target),
        '--da-batch-size','3','--da-nfe','50','--debug'],check=True)
    nifti_manifest=json.loads((target/'manifest.json').read_text())
    assert nifti_manifest['status']=='completed' and len(nifti_manifest['volumes'])==2
    for entry in nifti_manifest['volumes']:
        uid,_=scan_series(source/entry['file_id'])[0]
        reference=a.output/'debug_on'/output_id(entry['file_id'],uid)
        actual=target/entry['id']
        assert json.loads((actual/'logs/da_result.json').read_text())['seed']==0
        for old_path,new_path in ((reference.with_name(reference.name+'.nii.gz'), actual.with_name(actual.name+'.nii.gz')),
                                  (reference/'logs/debug/synthetic_pseudo.nii.gz', actual/'logs/debug/synthetic_pseudo.nii.gz')):
            expected=nib.load(old_path); observed=nib.load(new_path)
            np.testing.assert_array_equal(np.asarray(observed.dataobj),np.asarray(expected.dataobj))
            np.testing.assert_allclose(observed.affine,expected.affine,atol=1e-4,rtol=0)
    report['nifti_vs_dicom']={'volumes':2,'seed':0,'synthetic_and_masks_equal':True}
    report['scope']='Generated classic MR engineering fixture, not clinical DICOM accuracy validation'
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DA_PIPELINE_VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__': main()
