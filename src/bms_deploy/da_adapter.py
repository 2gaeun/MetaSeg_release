"""112788/114599 native-to-pseudo DA boundary for Dataset510 segmentation."""
import gc
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from .geometry import Volume, require_same_grid

MODELS = {'T1CE': ('t1ce2bb_112788_step0175000', 'BB', 112788, 175000),
          'BB': ('bb2t1ce_114599_step0150000', 'T1CE', 114599, 150000)}


def to_sitk(volume):
    volume.validate()
    lps = np.diag([-1., -1., 1., 1.]) @ volume.affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    direction = lps[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('Sheared affine is unsupported by DA')
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume.data.transpose(2, 1, 0), dtype=np.float32))
    image.SetSpacing(tuple(spacing)); image.SetOrigin(tuple(lps[:3, 3]))
    image.SetDirection(tuple(direction.ravel()))
    return image


def from_sitk(image):
    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    return Volume(sitk.GetArrayFromImage(image).transpose(2, 1, 0),
                  np.diag([-1., -1., 1., 1.]) @ affine).validate()


def geometry(image):
    return dict(size_xyz=list(image.GetSize()), spacing_xyz=list(image.GetSpacing()),
                origin_lps=list(image.GetOrigin()), direction_lps=list(image.GetDirection()))


def prepare(real):
    original = to_sitk(real)
    work = sitk.DICOMOrient(original, 'RAS')
    if abs(work.GetSpacing()[2] - 1.) > 1e-4:
        size = list(work.GetSize()); size[2] = max(1, int(round(size[2] * work.GetSpacing()[2])))
        work = sitk.Resample(work, size, sitk.Transform(), sitk.sitkLinear,
            work.GetOrigin(), (work.GetSpacing()[0], work.GetSpacing()[1], 1.),
            work.GetDirection(), 0., sitk.sitkFloat32)
    raw = sitk.GetArrayFromImage(work).astype(np.float32)
    low, high = np.percentile(raw, [0.1, 99.9])
    norm = np.zeros_like(raw) if high <= low else ((np.clip(raw, low, high) - low) / (high - low) * 2 - 1).astype(np.float32)
    # Apply exactly the legacy per-slice interpolation used by the trained models.
    resized = torch.stack([F.interpolate(torch.from_numpy(x)[None, None], (256, 256),
        mode='bilinear', align_corners=False)[0, 0] for x in norm])
    return original, work, resized, dict(percentiles=[0.1, 99.9], low=float(low), high=float(high))


def restore(raw_output, work, original, native_affine):
    # Historical Dataset505 native synthetic helper; unused by Dataset510 inference.
    slices = F.interpolate(raw_output[:, None], size=(work.GetSize()[1], work.GetSize()[0]),
                           mode='bilinear', align_corners=False)[:, 0].numpy()
    image = sitk.GetImageFromArray(np.clip(slices, -1., 1.).astype(np.float32))
    image.CopyInformation(work)
    image = sitk.Resample(image, original, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32)
    return Volume(sitk.GetArrayFromImage(image).transpose(2, 1, 0), native_affine.copy()).validate()


def network_affine(work):
    # Historical physical debug affine; Dataset510 debug uses network_reference instead.
    image = from_sitk(work)
    scale = np.array([work.GetSize()[0] / 256, work.GetSize()[1] / 256, 1.])
    mapping = np.eye(4); mapping[:3, :3] = np.diag(scale)
    mapping[:3, 3] = (scale - 1) / 2
    return image.affine @ mapping


class DomainAdapter:
    def __init__(self, weights, device='cuda', batch_size=8, nfe=50, seed=0):
        if batch_size < 1 or not 0 < nfe < 1000:
            raise ValueError('DA batch size must be positive; nfe must be 1..999')
        if seed != 0:
            raise ValueError('DA seed is fixed at 0')
        self.weights = Path(weights); self.device = device
        self.batch_size, self.nfe, self.seed = batch_size, nfe, seed
        # Load each direction on first use and retain it on the selected device.
        self._models = {}

    def close(self):
        """Release all cached directions after the input batch has finished."""
        while self._models:
            _, model = self._models.popitem()
            model.release_gpu()
            del model
        gc.collect()

    def generate(self, real, input_sequence, volume_id):
        from .da_network import I2SBInference
        sequence = input_sequence.upper()
        if sequence not in MODELS: raise ValueError('Input sequence must be BB or T1CE')
        folder, target, job, step = MODELS[sequence]
        original, work, source, normalization = prepare(real)
        model = self._models.get(sequence)
        if model is None:
            model = I2SBInference(self.weights / folder, self.device)
            if (model.config['training_job'], model.config['step']) != (job, step):
                model.release_gpu()
                raise ValueError('Wrong direction/checkpoint in DA bundle')
            self._models[sequence] = model
        seed = self.seed
        outputs = []
        try:
            devices = [torch.cuda.current_device()] if self.device == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                for start in range(0, len(source), self.batch_size):
                    end = min(start + self.batch_size, len(source))
                    indices = [[max(0, min(len(source)-1, z+d)) for d in (-1, 0, 1)] for z in range(start, end)]
                    cond = source[torch.tensor(indices)]
                    outputs.append(model.sample(source[start:end, None], cond, self.nfe))
                    print(f'DA_SLICES {volume_id} {end}/{len(source)}', flush=True)
            generated = torch.cat(outputs)
            from .pseudo import make_transform, network_reference
            transform = make_transform(original, work, volume_id, sequence)
            affine = from_sitk(network_reference(work)).affine
            # Legacy keep_network_resolution output clips on the 256x256 pseudo grid.
            synthetic = Volume(generated.clamp(-1., 1.).numpy().transpose(2, 1, 0), affine).validate()
        except Exception:
            # Retry a failed direction with a fresh model on the next volume.
            self._models.pop(sequence)
            model.release_gpu(); del model; gc.collect()
            raise
        info = dict(training_job=job, checkpoint_step=step, input_sequence=sequence, target_sequence=target,
                    seed=seed, base_seed=self.seed, nfe=self.nfe, batch_size=self.batch_size,
                    original=geometry(original), work=geometry(work), normalization=normalization,
                    endpoint_channels=1, condition_offsets=[-1, 0, 1], clip_denoise=False,
                    native_background=0., array_order='XYZ externally; ZYX for torch',
                    debug_network_grid='pseudo-1mm model space', transform=transform,
                    output_contract='256x256xdepth clipped [-1,1]; no native XY restoration')
        return synthetic, (source, generated, affine, info)


def save_debug(payload, directory):
    source, generated, affine, info = payload
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(source.numpy().transpose(2, 1, 0), affine), directory / 'da_input_network.nii.gz')
    nib.save(nib.Nifti1Image(generated.numpy().transpose(2, 1, 0), affine), directory / 'da_output_network.nii.gz')
    (directory / 'da_geometry.json').write_text(json.dumps(info, indent=2) + '\n')
