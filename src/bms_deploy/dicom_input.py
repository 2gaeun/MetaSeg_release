"""Strict classic single-frame MR DICOM ingestion, without intermediate NIfTI files.

One DICOM CSV row is a relative single-series folder + BB/T1CE.
Multiple SeriesInstanceUIDs in one folder are rejected. Duplicate positions (4D/echo stacks),
enhanced multiframe, nonuniform spacing and gantry shear are rejected, not guessed.
"""
import csv
from pathlib import Path
import hashlib
import re
import numpy as np
import SimpleITK as sitk
from .geometry import Volume


def nifti_index(root):
    """Index NIfTI basenames across the input tree; duplicates stay ambiguous."""
    found = {}
    for path in sorted(root.rglob('*')):
        suffix = next((s for s in ('.nii.gz', '.nii') if path.name.endswith(s)), None)
        if suffix and path.is_file():
            found.setdefault(path.name[:-len(suffix)], []).append(path)
    return found


def read_input_csv(root, csv_path, input_format='dicom'):
    if input_format not in ('dicom', 'nifti', 'nifti-pairs'):
        raise ValueError(f'Unsupported input format: {input_format}')
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Input root must be a directory')
    rows, seen = [], set()
    index = nifti_index(root) if input_format == 'nifti' else None
    headers = ('file_id', 'series_id', 'id') if index is not None else ('folder',)
    first = True
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not s.strip() for s in row):
                continue
            columns = [v.strip() for v in row]
            if first and len(columns) == 2 and columns[0].lower() in headers and columns[1].lower() == 'sequence':
                first = False
                continue
            first = False
            if len(columns) != 2:
                raise ValueError(f'CSV line {line}: exactly two columns required')
            name, seq = columns[0], columns[1].upper()
            if seq not in ('BB', 'T1CE'):
                raise ValueError(f'CSV line {line}: invalid sequence {seq}')
            relative = Path(name)
            if not name or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Input must be a safe relative name')
            if index is not None:
                if '/' in name or '\\' in name or name in ('.', '..') or name.endswith(('.nii', '.nii.gz')):
                    raise ValueError('NIfTI file ID must be a basename without .nii or .nii.gz')
                matches = index.get(name, [])
                if not matches:
                    raise FileNotFoundError(f'NIfTI file ID {name}: no {name}.nii or {name}.nii.gz under {root}')
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous NIfTI file ID {name}: {len(matches)} matching files')
                path = matches[0].resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError('NIfTI file outside input root')
            else:
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root) or not path.is_dir():
                    raise ValueError('Input folder outside root or not directory')
            if path in seen:
                raise ValueError(f'Duplicate input: {name}')
            seen.add(path)
            rows.append((name, seq, path))
    if not rows:
        raise ValueError('Empty input CSV')
    return rows


def output_id(folder,series_uid=''):
    # Stable identifier; never use this to select model channel order.
    stem=re.sub(r'[^A-Za-z0-9_.-]','_',folder).strip('.')[:100] or 'volume'
    return stem+'_'+hashlib.sha256((folder+'\0'+series_uid).encode()).hexdigest()[:10]


def _tag(reader,key,default=None):
    return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else default


def scan_series(folder):
    found={}
    for directory in [Path(folder),*sorted(p for p in Path(folder).rglob('*') if p.is_dir())]:
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or ():
            files=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory),uid)
            found.setdefault(uid,set()).update(files)
    if not found:raise ValueError(f'No DICOM series in {folder}')
    if len(found) != 1:
        raise ValueError(f'Expected exactly one DICOM series per CSV folder; found {len(found)} in {folder}')
    return [(uid,sorted(paths)) for uid,paths in sorted(found.items())]


def read_series(files):
    entries=[];echoes=set();times=set();sops=set()
    for path in files:
        r=sitk.ImageFileReader();r.SetFileName(str(path));r.ReadImageInformation()
        if _tag(r,'0008|0060')!='MR':raise ValueError('Only MR DICOM is supported')
        if int(_tag(r,'0028|0008','1'))!=1:raise ValueError('Enhanced/multiframe DICOM not supported; do not split frames silently')
        try:
            direction=np.array([float(x) for x in _tag(r,'0020|0037').split('\\')])
            origin=np.array([float(x) for x in _tag(r,'0020|0032').split('\\')])
            pixel=np.array([float(x) for x in _tag(r,'0028|0030').split('\\')])
        except (AttributeError,ValueError):raise ValueError(f'Missing/invalid DICOM geometry: {path}')
        if direction.shape!=(6,) or origin.shape!=(3,) or pixel.shape!=(2,):raise ValueError('DICOM geometry dimensions invalid')
        sop=_tag(r,'0008|0018')
        if not sop or sop in sops:raise ValueError('Missing or duplicate SOPInstanceUID')
        sops.add(sop);echoes.add(_tag(r,'0018|0086',''));times.add(_tag(r,'0020|0100',''))
        entries.append((path,direction,origin,pixel,r.GetSize()[:2]))
    if len(entries)<2:raise ValueError('At least two slices required for unambiguous volume spacing')
    if len(echoes)>1 or len(times)>1:raise ValueError('Multiple echoes/timepoints in one series; split into unambiguous volume folders')
    _,direction,origin,pixel,size=entries[0]
    x,y=direction[:3],direction[3:];normal=np.cross(x,y)
    if not np.allclose([np.linalg.norm(x),np.linalg.norm(y),np.linalg.norm(normal)],[1,1,1],atol=1e-4):raise ValueError('Nonorthogonal DICOM axes')
    for _,d,_,p,s in entries:
        if not np.allclose(d,direction,atol=1e-5) or not np.allclose(p,pixel,atol=1e-5) or s!=size:raise ValueError('Inconsistent DICOM slice geometry')
    entries.sort(key=lambda e:float(e[2]@normal))
    positions=np.array([e[2] for e in entries]);distance=positions@normal;delta=np.diff(distance)
    if np.min(delta)<=1e-4 or not np.allclose(delta,np.median(delta),rtol=1e-3,atol=1e-3):raise ValueError('Duplicate/missing/nonuniform slice positions')
    if not np.allclose(np.diff(positions,axis=0),delta[:,None]*normal,atol=1e-3):raise ValueError('Sheared slice stack unsupported')
    reader=sitk.ImageSeriesReader();reader.SetFileNames([str(e[0]) for e in entries]);image=reader.Execute()
    affine=np.eye(4);affine[:3,:3]=np.asarray(image.GetDirection()).reshape(3,3)@np.diag(image.GetSpacing());affine[:3,3]=image.GetOrigin()
    affine=np.diag([-1.,-1.,1.,1.])@affine
    data=sitk.GetArrayFromImage(image).transpose(2,1,0).astype(np.float32)
    if data.shape[2]!=len(entries):raise ValueError('Unexpected DICOM dimensionality')
    return Volume(data,affine).validate()
