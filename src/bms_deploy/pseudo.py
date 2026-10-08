"""Dataset510 nominal grid and inverse, independent of research repositories/GT."""
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from .geometry import Volume, require_same_grid
from .da_adapter import to_sitk, from_sitk

SCHEMA = 'i2sb-pseudo-1mm-transform-v1'
SPACE = 'pseudo-1mm model space'


def describe(image):
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    return dict(shape_xyz=list(image.GetSize()), spacing_xyz=list(image.GetSpacing()),
                origin=list(image.GetOrigin()), direction=list(image.GetDirection()),
                affine_lps=affine.tolist(),
                orientation=sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection()))


def reference(g):
    size = g['shape_xyz']
    if len(size) != 3 or any(int(n) != n or n < 1 for n in size):
        raise ValueError('Invalid transform size')
    spacing = np.asarray(g['spacing_xyz'], dtype=float)
    direction = np.asarray(g['direction'], dtype=float).reshape(3, 3)
    origin = np.asarray(g['origin'], dtype=float)
    if spacing.shape != (3,) or origin.shape != (3,) or not np.isfinite(np.r_[spacing, origin, direction.ravel()]).all() or np.any(spacing <= 0):
        raise ValueError('Invalid transform geometry')
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Nonorthogonal transform direction')
    affine = np.eye(4); affine[:3, :3] = direction @ np.diag(spacing); affine[:3, 3] = origin
    if not np.allclose(affine, g['affine_lps'], atol=1e-4, rtol=0):
        raise ValueError('Transform affine disagrees with spacing/origin/direction')
    image = sitk.Image([int(n) for n in size], sitk.sitkUInt8)
    image.SetSpacing(tuple(spacing)); image.SetOrigin(tuple(origin)); image.SetDirection(tuple(direction.ravel()))
    if describe(image)['orientation'] != g['orientation']:
        raise ValueError('Transform orientation mismatch')
    return image


def require_geometry(image, ref, name):
    if image.GetSize() != ref.GetSize() or any(not np.allclose(a, b, atol=1e-4, rtol=0) for a, b in (
            (image.GetSpacing(), ref.GetSpacing()), (image.GetOrigin(), ref.GetOrigin()),
            (image.GetDirection(), ref.GetDirection()))):
        raise ValueError(name + ': geometry mismatch; automatic repair is forbidden')


def network_reference(work):
    # Legacy make_network_space_reference: nominal XY spacing, preserved center and direction.
    spacing = np.array([1., 1., work.GetSpacing()[2]])
    size = np.array([256, 256, work.GetSize()[2]])
    direction = np.asarray(work.GetDirection()).reshape(3, 3)
    center = np.asarray(work.GetOrigin()) + direction @ ((np.asarray(work.GetSize()) - 1) * np.asarray(work.GetSpacing()) / 2)
    origin = center - direction @ ((size - 1) * spacing / 2)
    ref = sitk.Image([int(n) for n in size], sitk.sitkUInt8)
    ref.SetSpacing(tuple(spacing)); ref.SetDirection(work.GetDirection()); ref.SetOrigin(tuple(origin))
    return ref


def make_transform(original, work, case_id, sequence):
    sequence = sequence.upper()
    if sequence not in ('T1CE', 'BB'): raise ValueError('Unknown input sequence')
    ras = sitk.DICOMOrient(original, 'RAS')
    return dict(schema_version=SCHEMA, space_name=SPACE, case_id=case_id,
        original=describe(original), ras=describe(ras), work_before_xy_resize=describe(work),
        pseudo_model_space=describe(network_reference(work)), target_orientation='RAS',
        z_resampled=abs(ras.GetSpacing()[2]-1.) > 1e-4,
        axis_order=dict(array='ZYX', geometry='XYZ', physical_coordinates='LPS'),
        xy_resize=dict(input_size_xyz=list(work.GetSize()), output_size_xyz=[256,256,work.GetSize()[2]],
                       image='torch bilinear align_corners=False', mask='torch nearest', crop_pad=False),
        native_mask_restore=dict(xy_resize='torch nearest to work_before_xy_resize size',
                                 resample='SimpleITK nearest to original geometry', background_fill_value=0),
        input_sequence=sequence, real_channel_index=0 if sequence=='T1CE' else 1,
        provenance_source='input volume geometry and supplied sequence; no GT or test manifest', skull_stripping=False)


def validate_transform(meta):
    if meta.get('schema_version') != SCHEMA or meta.get('space_name') != SPACE:
        raise ValueError('Expected dedicated Dataset510 pseudo transform')
    original, ras, work, pseudo = [reference(meta[k]) for k in ('original','ras','work_before_xy_resize','pseudo_model_space')]
    require_geometry(sitk.DICOMOrient(original, 'RAS'), ras, 'original -> RAS')
    z_resampled = abs(ras.GetSpacing()[2]-1.) > 1e-4
    size = list(ras.GetSize()); spacing = list(ras.GetSpacing())
    if z_resampled: size[2]=max(1,int(round(size[2]*spacing[2]))); spacing[2]=1.
    expected = sitk.Image(size, sitk.sitkUInt8); expected.SetSpacing(spacing)
    expected.SetDirection(ras.GetDirection()); expected.SetOrigin(ras.GetOrigin())
    if meta['z_resampled'] != z_resampled: raise ValueError('Incorrect z resampling flag')
    require_geometry(work, expected, 'work_before_xy_resize')
    require_geometry(pseudo, network_reference(work), 'pseudo grid')
    if not np.allclose(pseudo.GetSpacing(), [1,1,1], atol=1e-5, rtol=0):
        raise ValueError('Dataset510 requires nominal 1mm spacing')
    return original, work, pseudo


def prepare_pair(real, synthetic, sequence, transform):
    require_same_grid(real, synthetic)
    _, _, ref = validate_transform(transform)
    require_geometry(to_sitk(real), ref, 'pseudo pair vs transform')
    sequence = sequence.upper()
    if sequence not in ('T1CE','BB'): raise ValueError('Sequence must be BB or T1CE')
    real_index = 0 if sequence=='T1CE' else 1
    if transform.get('input_sequence', sequence) != sequence or transform.get('real_channel_index',real_index) != real_index:
        raise ValueError('Sequence/real-channel provenance mismatch')
    for image in (real, synthetic):
        if image.data.min() < -1.0001 or image.data.max() > 1.0001:
            raise ValueError('Expected normalized [-1,1] pseudo pair')
    if nib.aff2axcodes(real.affine) != ('R','A','S'):
        raise ValueError('Expected legacy RAS pseudo grid')
    coords = np.where(~np.isclose(real.data, -1., atol=1e-6, rtol=0))
    if not coords[0].size: raise ValueError('Real normalized background covers the entire image')
    bbox = tuple(slice(int(c.min()), int(c.max())+1) for c in coords)
    # Crop in the exported XYZ grid exactly as Dataset510's builder, then use the nnU-Net reader order.
    cropped = real.nifti().slicer[bbox]
    channels = (real.data[bbox], synthetic.data[bbox]) if real_index==0 else (synthetic.data[bbox],real.data[bbox])
    data = np.stack([a.transpose(2,1,0) for a in channels]).astype(np.float32)
    props = dict(spacing=[float(v) for v in cropped.header.get_zooms()[::-1]],
                 nibabel_stuff=dict(original_affine=cropped.affine.copy(), reoriented_affine=cropped.affine.copy()))
    info = dict(pseudo_shape=list(real.data.shape), pseudo_affine=real.affine.tolist(),
                real_bbox_xyz=[[s.start,s.stop] for s in bbox], cropped_ras_affine=cropped.affine.tolist(),
                sequence=sequence, real_channel_index=real_index, normalized_background=-1., background_tolerance=1e-6)
    return data, props, info


def undo_crops(probability, props, plans, info):
    if tuple(probability.shape) != tuple(props['shape_after_cropping_and_before_resampling']):
        raise ValueError('Dataset510 nnU-Net output changed shape: spacing interpolation forbidden')
    canvas = np.zeros(props['shape_before_cropping'], dtype=np.float32)
    canvas[tuple(slice(a,b) for a,b in props['bbox_used_for_cropping'])] = probability
    xyz = canvas.transpose(plans.transpose_backward).transpose(2,1,0)
    full = np.zeros(info['pseudo_shape'], dtype=np.float32)
    bbox = tuple(slice(a,b) for a,b in info['real_bbox_xyz'])
    if full[bbox].shape != xyz.shape: raise ValueError('Export crop shape mismatch')
    full[bbox] = xyz
    return Volume(full, np.asarray(info['pseudo_affine'])).validate()


def restore_native(pseudo, transform, probability=False):
    original, work, ref = validate_transform(transform)
    require_geometry(to_sitk(pseudo), ref, 'prediction vs pseudo transform')
    a = pseudo.data
    if probability:
        if a.min()<0 or a.max()>1: raise ValueError('Invalid probability')
    elif not np.isin(a,[0,1]).all(): raise ValueError('Expected binary mask')
    x = torch.from_numpy(np.ascontiguousarray(a.transpose(2,1,0),dtype=np.float32))[:,None]
    xy = F.interpolate(x, size=(work.GetSize()[1],work.GetSize()[0]),mode='nearest')[:,0].numpy()
    image = sitk.GetImageFromArray(xy if probability else xy.astype(np.uint8)); image.CopyInformation(work)
    restored = sitk.Resample(image, original, sitk.Transform(), sitk.sitkNearestNeighbor, 0., image.GetPixelID())
    require_geometry(restored, original, 'native prediction')
    return from_sitk(restored).nifti()
