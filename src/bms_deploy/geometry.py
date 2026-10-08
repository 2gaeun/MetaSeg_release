from dataclasses import dataclass
import nibabel as nib
import numpy as np


@dataclass
class Volume:
    """In-memory XYZ voxels; 4x4 voxel->RAS-mm affine (DICOM LPS is converted)."""
    data: np.ndarray
    affine: np.ndarray

    def validate(self):
        if self.data.ndim != 3 or any(n<1 for n in self.data.shape):
            raise ValueError('Expected nonempty 3D XYZ volume')
        if not np.isfinite(self.data).all() or not np.isfinite(self.affine).all():
            raise ValueError('Nonfinite volume/geometry')
        if self.affine.shape!=(4,4) or abs(np.linalg.det(self.affine[:3,:3]))<1e-10:
            raise ValueError('Invalid affine')
        if not np.allclose(self.affine[3],[0,0,0,1]): raise ValueError('Invalid homogeneous affine')
        return self

    def nifti(self):
        return nib.Nifti1Image(self.data,self.affine)


def load_nifti(path):
    im=nib.load(str(path))
    return Volume(im.get_fdata(dtype=np.float32),np.array(im.affine)).validate()


def require_same_grid(real, synthetic):
    real.validate();synthetic.validate()
    if real.data.shape!=synthetic.data.shape or not np.allclose(real.affine,synthetic.affine,atol=1e-4,rtol=0):
        raise ValueError('Real/synthetic geometry mismatch: automatic alignment/resampling is forbidden')


def prepare_pair(real, synthetic, sequence):
    """Historical Dataset505 helper; Dataset510 inference uses pseudo.prepare_pair."""
    require_same_grid(real,synthetic)
    sequence=sequence.upper()
    if sequence not in ('T1CE','BB'):raise ValueError('Sequence must be BB or T1CE')
    original=real.nifti()
    real_ras=nib.as_closest_canonical(original)
    syn_ras=nib.as_closest_canonical(synthetic.nifti())
    if not np.allclose(real_ras.affine,syn_ras.affine,atol=1e-4,rtol=0):raise ValueError('RAS mismatch')
    a=np.asarray(real_ras.dataobj,dtype=np.float32)
    b=np.asarray(syn_ras.dataobj,dtype=np.float32)
    coords=np.where(a!=0)
    if not coords[0].size:raise ValueError('Real volume is all-zero')
    bbox=tuple(slice(int(v.min()),int(v.max())+1) for v in coords)
    cropped=real_ras.slicer[bbox]
    channels=(a[bbox],b[bbox]) if sequence=='T1CE' else (b[bbox],a[bbox])
    # Exact NibabelIOWithReorient convention: RAS XYZ -> nnU-Net ZYX.
    data=np.stack([c.transpose(2,1,0) for c in channels]).astype(np.float32)
    props={'spacing':[float(v) for v in cropped.header.get_zooms()[::-1]],
           'nibabel_stuff':{'original_affine':cropped.affine.copy(),'reoriented_affine':cropped.affine.copy()}}
    info={'original_shape':list(real.data.shape),'original_affine':real.affine.copy(),
          'ras_shape':list(a.shape),'ras_affine':real_ras.affine.copy(),
          'real_bbox_xyz':[[s.start,s.stop] for s in bbox],
          'cropped_ras_affine':cropped.affine.copy(),'sequence':sequence,'real_channel_index':0 if sequence=='T1CE' else 1}
    return data,props,info


def restore_native(cropped_zyx, info):
    """Historical native-pair inverse, not the Dataset510 pseudo inverse."""
    canvas=np.zeros(info['ras_shape'],dtype=cropped_zyx.dtype)
    bbox=tuple(slice(a,b) for a,b in info['real_bbox_xyz'])
    if canvas[bbox].shape!=cropped_zyx.transpose(2,1,0).shape:raise ValueError('Restored crop shape mismatch')
    canvas[bbox]=cropped_zyx.transpose(2,1,0)
    orientation=nib.orientations.ornt_transform(nib.orientations.axcodes2ornt('RAS'),nib.orientations.io_orientation(info['original_affine']))
    im=nib.Nifti1Image(canvas,info['ras_affine']).as_reoriented(orientation)
    if list(im.shape)!=info['original_shape'] or not np.allclose(im.affine,info['original_affine'],atol=1e-4,rtol=0):
        raise ValueError('Native output geometry mismatch')
    return im


def model_affine(info,props,plans,spacing):
    # nnU-Net axes are transpose_forward(ZYX). Return affine for transpose_backward -> XYZ display.
    starts=np.asarray([b[0] for b in props['bbox_used_for_cropping']])[plans.transpose_backward][::-1]
    affine=np.array(info['cropped_ras_affine'],copy=True)
    origin=(affine@np.r_[starts,1])[:3]
    scales=np.asarray(spacing)[plans.transpose_backward][::-1]
    affine[:3,:3]=affine[:3,:3]/nib.affines.voxel_sizes(affine)*scales
    affine[:3,3]=origin
    return affine
