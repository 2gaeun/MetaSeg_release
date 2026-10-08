"""nnUNetv2 2.8.1 inference without importing the original BrainMetaSeg repo."""
import json
import os
from pathlib import Path
import importlib.metadata
import nibabel as nib
import numpy as np
import torch
from .geometry import Volume, model_affine
from .pseudo import prepare_pair, restore_native, undo_crops


def json_value(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


class Segmenter:
    def __init__(self,weights,device='cuda',threads=2):
        if importlib.metadata.version('nnunetv2')!='2.8.1':raise RuntimeError('Requires nnunetv2==2.8.1')
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from .trainer import nnUNetTrainerBrainMetaFocalLR3e3Components as Trainer
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        torch.set_num_threads(threads);self.threads=threads
        os.environ['nnUNet_compile']='false'
        self.weights=Path(weights)
        self.dataset=json.loads((self.weights/'dataset.json').read_text())
        self.plans=PlansManager(str(self.weights/'plans.json'))
        self.cfg=self.plans.get_configuration('3d_fullres')
        raw=self.plans.plans
        if raw['dataset_name']!='Dataset510_BrainMet_Synthetic_2ch_Pseudo1mm_RealChannelCrop_DHW112x128x160' or self.dataset['channel_names']!={'0':'T1CE_like','1':'BB_like'}:
            raise ValueError('Expected Dataset510 two-channel metadata')
        if list(self.cfg.patch_size)!=[112,128,160] or list(self.cfg.spacing)!=[1,1,1]:raise ValueError('Unexpected plan')
        self.labels=self.plans.get_label_manager(self.dataset)
        self.network=Trainer.build_network_architecture(self.plans,self.cfg,
            2,self.labels.num_segmentation_heads,enable_deep_supervision=False)
        # Only load bundled, trusted checkpoint files. torch checkpoint is pickle-based.
        ck=torch.load(self.weights/'checkpoint_epoch_0950.pth',map_location='cpu',weights_only=False)
        if ck['trainer_name']!='nnUNetTrainerBrainMetaFocalLR3e3Components':raise ValueError('Wrong checkpoint trainer')
        self.stored_epoch=ck.get('current_epoch')
        params=ck['network_weights'];self.network.load_state_dict(params,strict=True)
        dev=torch.device(device)
        if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; CPU is explicit --device cpu')
        self.network.to(dev).eval()
        self.predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,
            perform_everything_on_device=dev.type=='cuda',device=dev,verbose=False,verbose_preprocessing=False,allow_tqdm=True)
        self.predictor.manual_initialization(self.network,self.plans,self.cfg,[params],self.dataset,
            ck['trainer_name'],ck['inference_allowed_mirroring_axes'])
        self.preprocessor=self.cfg.preprocessor_class(verbose=False)

    def predict(self,real,synthetic,sequence,output,debug=False,*,transform,mask_output=None):
        output=Path(output)
        mask_path=Path(mask_output) if mask_output is not None else output.parent/(output.name+'.nii.gz')
        if mask_path.exists():raise FileExistsError(mask_path)
        output.mkdir(parents=True,exist_ok=False)
        logs=output/'logs';logs.mkdir()
        raw,props,info=prepare_pair(real,synthetic,sequence,transform)
        preprocessed,_,props=self.preprocessor.run_case_npy(raw,None,props,self.plans,self.cfg,self.dataset)
        if tuple(preprocessed.shape[1:])!=tuple(props['shape_after_cropping_and_before_resampling']):
            raise ValueError('Dataset510 preprocessing changed spacing/shape')
        logits=self.predictor.predict_logits_from_preprocessed_data(torch.from_numpy(preprocessed))
        model_prob=self.labels.apply_inference_nonlin(logits).detach().cpu().numpy()
        if debug:
            dbg=logs/'debug';dbg.mkdir()
            aff=model_affine(info,props,self.plans,self.cfg.spacing)
            backward=self.plans.transpose_backward
            def model_xyz(a):return a.transpose(backward).transpose(2,1,0)
            nib.save(nib.Nifti1Image(model_xyz((model_prob[1]>=.5).astype(np.uint8)),aff),dbg/'model_mask.nii.gz')
            nib.save(nib.Nifti1Image(model_xyz(model_prob[1]),aff),dbg/'model_probability.nii.gz')
            for i in range(2):
                nib.save(nib.Nifti1Image(model_xyz(preprocessed[i]),aff),dbg/f'model_input_{i:04d}.nii.gz')
                nib.save(nib.Nifti1Image(raw[i].transpose(2,1,0),info['cropped_ras_affine']),dbg/f'seg_input_{i:04d}.nii.gz')
            nib.save(real.nifti(),dbg/'real_pseudo.nii.gz');nib.save(synthetic.nifti(),dbg/'synthetic_pseudo.nii.gz')
            (dbg/'geometry.json').write_text(json.dumps({'pair':info,'nnunet':props},default=json_value,indent=2)+'\n')
        full_probability=undo_crops(model_prob[1],props,self.plans,info)
        full_mask=Volume((full_probability.data>=.5).astype(np.uint8),full_probability.affine)
        native=restore_native(full_mask,transform)
        nib.save(native,mask_path)
        if debug:
            nib.save(full_mask.nifti(),logs/'debug/pseudo_mask.nii.gz')
            nib.save(full_probability.nifti(),logs/'debug/pseudo_probability.nii.gz')
            nib.save(native,logs/'debug/native_mask.nii.gz')
            nib.save(restore_native(full_probability,transform,probability=True),logs/'debug/native_probability.nii.gz')
        (logs/'transform.json').write_text(json.dumps(transform,default=json_value,indent=2)+'\n')
        (logs/'crop.json').write_text(json.dumps({'pair':info,'nnunet':props},default=json_value,indent=2)+'\n')
        record=dict(status='completed',training_job=131329,dataset=510,checkpoint_epoch=950,
            checkpoint_filename="checkpoint_epoch_0950.pth",stored_current_epoch=self.stored_epoch,
            trainer='nnUNetTrainerBrainMetaFocalLR3e3Components',space='pseudo-1mm model space',
            threshold=.5,threshold_rule='>=',skull_stripping=False,channel_order=['T1CE','BB'],
            real_channel_index=info['real_channel_index'],native_shape=list(native.shape),native_affine=native.affine.tolist(),
            model_shape_nnunet_axes=list(preprocessed.shape[1:]),debug=debug)
        (logs/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
