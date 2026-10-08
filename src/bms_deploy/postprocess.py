"""Optional native-space lesion filtering. Never changes DA/segmentation inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
from .geometry import Volume, load_nifti, require_same_grid

SCRIPT_SHA256 = 'fa18eb301568dc7c99bd9f69805545a7ce6d68e3fa3e1775c68053c9069cc4a1'
WEIGHT_SHA256 = '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33'


def binary(volume, name):
    volume.validate()
    if not np.isin(volume.data, [0, 1]).all():
        raise ValueError(name + ' must contain only 0 and 1')
    return volume.data.astype(bool)


def intersect(lesion, brain):
    require_same_grid(lesion, brain)
    mask = binary(brain, 'Brain mask')
    if not mask.any():
        raise ValueError('SynthStrip returned an empty brain mask')
    return Volume((binary(lesion, 'Lesion mask') & mask).astype(np.uint8), lesion.affine.copy())


class SynthStrip:
    """Run the unmodified official 1.8 script in a separate process."""
    def __init__(self, home=None, device='cpu', threads=2):
        self.home = Path(home or os.environ.get('BMS_SYNTHSTRIP_HOME', '/opt/synthstrip/1.8'))
        self.device, self.threads = device, threads
        if device not in ('cpu', 'cuda') or threads < 1:
            raise ValueError('Invalid SynthStrip device/threads')
        self.script = self.home / 'mri_synthstrip'
        self.weight = self.home / 'models/synthstrip.1.pt'
        for path, expected in ((self.script, SCRIPT_SHA256), (self.weight, WEIGHT_SHA256)):
            if not path.is_file():
                raise RuntimeError('SynthStrip 1.8 is unavailable. Use the v2 image, or explicitly select --skull-strip off with v1.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Unexpected SynthStrip 1.8 asset: ' + str(path))
        for module in ('surfa', 'xxhash'):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError('SynthStrip dependency missing: ' + module + '; use the v2 image')
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('SynthStrip CUDA requested but unavailable')

    def generate(self, native, scratch, log_path, source_path=None):
        scratch = Path(scratch)
        if source_path is None:
            source_path = scratch / 'input_native.nii.gz'
            nib.save(native.nifti(), source_path)
        else:
            # Keep original NIfTI intensities/dtype; no normalization or skull stripping.
            image = nib.load(str(source_path))
            if image.shape != native.data.shape or not np.allclose(image.affine, native.affine, atol=1e-4, rtol=0):
                raise ValueError('SynthStrip input reference geometry mismatch')
        output = scratch / 'brain_mask.nii.gz'
        command = [sys.executable, str(self.script), '-i', str(source_path), '-m', str(output),
                   '-t', str(self.threads)]
        if self.device == 'cuda':
            command.append('-g')
        env = os.environ.copy()
        env['FREESURFER_HOME'] = str(self.home)
        with Path(log_path).open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        brain = load_nifti(output)
        require_same_grid(native, brain)
        return brain


def postprocess_native(native, raw_mask_path, final_mask_path, logs, generator,
                       debug=False, source_path=None):
    """Publish the final mask only after successful brain extraction and validation."""
    logs, raw_mask_path, final_mask_path = Path(logs), Path(raw_mask_path), Path(final_mask_path)
    if final_mask_path.exists():
        raise FileExistsError(final_mask_path)
    raw = load_nifti(raw_mask_path)
    require_same_grid(native, raw)
    binary(raw, 'Lesion mask')
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='synthstrip_', dir=logs) as folder:
        brain = generator.generate(native, folder, logs / 'synthstrip.log', source_path)
        require_same_grid(native, brain)
        filtered = intersect(raw, brain)
        record = dict(tool='SynthStrip', version='1.8', model_version=1, device=generator.device,
                      border_mm=1, no_csf=False, input='original native MRI', operation='lesion AND brain',
                      space='native', geometry_verified=True, interpolation_after_synthstrip=False,
                      brain_voxels=int(np.count_nonzero(brain.data)),
                      lesion_voxels_before=int(np.count_nonzero(raw.data)),
                      lesion_voxels_after=int(np.count_nonzero(filtered.data)),
                      seconds=time.monotonic()-start, script_sha256=SCRIPT_SHA256, weight_sha256=WEIGHT_SHA256)
        if debug:
            dbg = logs / 'debug'
            dbg.mkdir(exist_ok=True)
            probability_path = dbg / 'native_probability.nii.gz'
            if probability_path.exists():
                probability = load_nifti(probability_path)
                require_same_grid(raw, probability)
                if not np.array_equal(probability.data >= .5, raw.data > 0):
                    raise ValueError('Unfiltered probability/mask disagree')
                probability.data *= brain.data > 0
                probability_path.rename(dbg / 'native_probability_before_skullstrip.nii.gz')
                nib.save(probability.nifti(), probability_path)
            nib.save(brain.nifti(), dbg / 'brain_mask.nii.gz')
            nib.save(raw.nifti(), dbg / 'native_mask_before_skullstrip.nii.gz')
            nib.save(filtered.nifti(), dbg / 'native_mask.nii.gz')
        (logs / 'postprocessing.json').write_text(json.dumps(record, indent=2)+'\n')
        temporary = Path(folder) / 'final_mask.nii.gz'
        nib.save(filtered.nifti(), temporary)
        os.replace(temporary, final_mask_path)
    raw_mask_path.unlink()
    return record
