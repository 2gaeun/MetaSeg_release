"""Opt-in comparison with read-only research results; never used by deployment CLI."""
import argparse,csv,gc,json,pickle,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from bms_deploy.geometry import Volume,load_nifti
from bms_deploy.da_adapter import prepare,from_sitk
from bms_deploy.pseudo import make_transform,network_reference,prepare_pair,undo_crops,restore_native
from bms_deploy.segment import Segmenter,json_value


def difference(a,b):
    assert a.shape==b.shape
    n=changed=0;total=maximum=0.
    for z in range(a.shape[2]):
        x=np.asarray(a[:,:,z]);y=np.asarray(b[:,:,z]);d=np.abs(x.astype(np.float64)-y)
        n+=d.size;total+=float(d.sum());maximum=max(maximum,float(d.max()));changed+=int(np.count_nonzero(x!=y))
    return dict(shape=list(a.shape),different_voxels=changed,voxel_agreement=1-changed/n,mae=total/n,max_abs=maximum)


def masks(a,b):
    if a.shape!=b.shape or not np.allclose(a.affine,b.affine,atol=1e-4,rtol=0):raise ValueError('Compared mask geometry mismatch')
    x=np.asarray(a.dataobj)>0;y=np.asarray(b.dataobj)>0;d=difference(x,y)
    den=int(x.sum()+y.sum());d['dice']=2*int(np.count_nonzero(x&y))/den if den else 1.;d['geometry_match']=True
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--nas-home',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['pairs','e2e'],required=True);p.add_argument('--cases',default='1,6,11,18,48,59');p.add_argument('--resume-evaluation',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume_evaluation);torch.set_num_threads(4)
    home=args.nas_home
    def host(s):return home/Path(s).relative_to('/home') if s.startswith('/home/') else Path(s)
    manifest=home/'logs/brainmetaseg/dataset510_pseudo_1mm_real_channel_crop/model_space_export_manifest.csv'
    by_id={r['nnunet_case_id']:r for r in csv.DictReader(manifest.open()) if r['split']=='test'}
    reference=home/'logs/nnUNet_test_inference/test_seed42_dataset510_focal131329_epoch0950_job132408'
    predictions=reference/'131329_2ch_focal_dataset510_epoch0950/predictions';evaluation=reference/'evaluation_job132409'
    selected=[by_id[f'BrainMet_test_{int(x):05d}'] for x in args.cases.split(',')]
    results=[];seg=Segmenter(args.bundle/'weights/segmentation_131329',threads=4) if args.mode=='pairs' else None
    for row in selected:
        cid=row['nnunet_case_id'];seq=row['sequence_label'].upper();meta=json.loads(host(row['transform_metadata_path']).read_text());out=args.output/cid
        if args.mode=='pairs':
            real=load_nifti(host(row['model_space_real_path']));syn=load_nifti(host(row['model_space_synthetic_path']))
            raw,props,info=prepare_pair(real,syn,seq,meta)
            assert info['real_channel_index']==int(row['real_channel_index'])
            assert info['real_bbox_xyz']==[[int(row[f'crop_{a}_start']),int(row[f'crop_{a}_stop'])] for a in 'xyz']
            paths=[host(row[f'exported_ch{i}_path']) for i in range(2)]
            rw=seg.plans.image_reader_writer_class();reference_data,reference_props=rw.read_images([str(f) for f in paths])
            assert np.array_equal(raw,reference_data),'export pair values/channel order differ'
            computed,_,computed_props=seg.preprocessor.run_case_npy(raw,None,props,seg.plans,seg.cfg,seg.dataset)
            expected,_,expected_props=seg.preprocessor.run_case_npy(reference_data,None,reference_props,seg.plans,seg.cfg,seg.dataset)
            assert np.array_equal(computed,expected),'normalization/transpose/crop differs'
            with (predictions/(cid+'.model_grid.pkl')).open('rb') as f:oldprops=pickle.load(f)
            for key in ['shape_before_cropping','shape_after_cropping_and_before_resampling','bbox_used_for_cropping']:assert np.array_equal(computed_props[key],oldprops[key]),key
            with np.load(predictions/(cid+'.model_grid.npz')) as z:prob=np.asarray(z['probabilities'][1],dtype=np.float32)
            full=undo_crops(prob,oldprops,seg.plans,info);mask=Volume((full.data>=.5).astype(np.uint8),full.affine)
            inverse_pseudo=masks(mask.nifti(),nib.load(evaluation/'pseudo_predictions'/(cid+'.nii.gz')))
            inverse_native=masks(restore_native(mask,meta),nib.load(evaluation/'native_predictions'/(cid+'.nii.gz')))
            assert inverse_pseudo['different_voxels']==inverse_native['different_voxels']==0,'inverse differs with identical reference probability'
            del raw,computed,expected,reference_data,prob,full,mask;gc.collect()
            seg.predict(real,syn,seq,out,True,transform=meta)
            result=dict(case_id=cid,mode=args.mode,channels_crop_preprocessed_exact=True,inverse_reference_probability_pseudo=inverse_pseudo,inverse_reference_probability_native=inverse_native)
        else:
            inputs=args.output/(cid+'_input');inputs.mkdir(exist_ok=args.resume_evaluation);(inputs/'input.csv').write_text('file_id,sequence\n'+cid+','+seq+'\n')
            # Input contract rejects external symlinks. Copy only this unmodified image for the CLI check.
            import shutil
            shutil.copy2(host(row['native_image_path']),inputs/(cid+'.nii.gz'))
            native=load_nifti(inputs/(cid+'.nii.gz'));original,work,source,norm=prepare(native)
            generated_meta=make_transform(original,work,cid,seq)
            for key in ['original','ras','work_before_xy_resize','pseudo_model_space']:
                for field in ['shape_xyz','spacing_xyz','origin','direction','affine_lps']:
                    assert np.allclose(generated_meta[key][field],meta[key][field],atol=1e-4,rtol=0),(key,field)
            reference_real=load_nifti(host(row['model_space_real_path']));prepared=difference(source.numpy().transpose(2,1,0),reference_real.data)
            del native,source,reference_real;gc.collect()
            cliout=args.output/(cid+'_run')
            command=['bash',str(args.bundle/'run.sh'),'--input-format','nifti','--input-root',str(inputs),'--csv',str(inputs/'input.csv'),'--output',str(cliout),'--threads','4','--debug']
            if not (args.resume_evaluation and (cliout/'manifest.json').exists()):subprocess.run(command,check=True)
            run=json.loads((cliout/'manifest.json').read_text())
            assert run['status']=='completed' and len(run['volumes'])==1 and not run['errors']
            out=cliout/run['volumes'][0]['id']
            synthetic=load_nifti(out/'logs/debug/synthetic_pseudo.nii.gz');old=load_nifti(host(row['model_space_synthetic_path']))
            result=dict(case_id=cid,mode=args.mode,da_real_vs_stored=prepared,da_synthetic_vs_stored=difference(synthetic.data,old.data),transform_geometry_matches=True,seed=0,nfe=50,batch_size=8)
            del synthetic,old;gc.collect()
            seed0=home/'logs/nnUNet_test_inference/_dist_v1/testset_seed_confounder/seed_0/predictions'/meta['case_id']/'logs/debug/da_output_network.nii.gz'
            if seed0.exists():
                old_da=nib.load(seed0);new_da=nib.load(out/'logs/debug/da_output_network.nii.gz')
                result['da_network_vs_prior_seed0']=difference(np.asarray(new_da.dataobj),np.asarray(old_da.dataobj))
                result['prior_seed0_reference']=str(seed0)
                del old_da,new_da;gc.collect()
        dbg=out/'logs/debug'
        result['pseudo']=masks(nib.load(dbg/'pseudo_mask.nii.gz'),nib.load(evaluation/'pseudo_predictions'/(cid+'.nii.gz')))
        result['native']=masks(nib.load(out.with_name(out.name+'.nii.gz')),nib.load(evaluation/'native_predictions'/(cid+'.nii.gz')))
        result['original_geometry']=meta['original'];result['sequence']=seq
        (args.output/(cid+'_comparison.json')).write_text(json.dumps(result,default=json_value,indent=2)+'\n');results.append(result)
        print('COMPARED',json.dumps(result,default=json_value),flush=True)
        if args.mode=='pairs':del real,syn
        gc.collect()
    report=dict(status='completed',mode=args.mode,skull_stripping=False,cases=results,reference=str(reference),checkpoint='checkpoint_epoch_0950.pth',notes='Mask equality is explicitly measured. E2E synthetic may differ because existing DA seed0 is retained; stored reference pairs are used unchanged only in pairs mode.')
    (args.output/'summary.json').write_text(json.dumps(report,default=json_value,indent=2)+'\n')
if __name__=='__main__':main()
