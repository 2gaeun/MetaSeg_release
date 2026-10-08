"""One-time asset copy: only needed to assemble dist, NOT deployment runtime.

Writes real regular files (no hard/symlinks). Never overwrites an existing
different asset. Existing matching files permit resuming an interrupted copy.
"""
import argparse,csv,hashlib,json,os
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def copy_asset(src,dst):
    src=Path(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():raise ValueError(f'Symlink in bundle: {dst}')
    if dst.exists():
        sha=digest(src)
        if src.stat().st_size!=dst.stat().st_size or digest(dst)!=sha:raise FileExistsError(dst)
        return dict(bytes=dst.stat().st_size,sha256=sha)
    tmp=dst.with_name(dst.name+f'.partial_{os.getpid()}');h=hashlib.sha256()
    with src.open('rb') as inp,tmp.open('xb') as out:
        for b in iter(lambda:inp.read(8*1024*1024),b''):h.update(b);out.write(b)
        out.flush();os.fsync(out.fileno())
    if tmp.stat().st_size!=src.stat().st_size or digest(tmp)!=h.hexdigest():raise IOError('Copy verification failed')
    tmp.rename(dst)
    return dict(bytes=dst.stat().st_size,sha256=h.hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dist','source-manifest','resolution-metadata','model-dir','docker-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();root=a.dist.resolve();assets={}
    records={r['nnunet_case_id']:r for r in csv.DictReader(a.source_manifest.open()) if r['split']=='test'}
    resolution=list(csv.DictReader(a.resolution_metadata.open()))
    if len(records)!=78 or {r['nnunet_case_id'] for r in resolution}!=set(records):raise ValueError('Expected test78')
    for src,rel in [(a.model_dir/'fold_0/checkpoint_final.pth','weights/segmentation_123195/checkpoint_final.pth'),
                    (a.model_dir/'plans.json','weights/segmentation_123195/plans.json'),
                    (a.model_dir/'dataset.json','weights/segmentation_123195/dataset.json'),
                    (a.docker_dir/'Dockerfile','docker/Dockerfile'),(a.docker_dir/'requirements.txt','docker/requirements.txt')]:
        assets[rel]=copy_asset(src,root/rel);print('COPIED',rel,flush=True)
    samples=[];counts={'1mm':0,'LE0p5mm':0}
    for idx,r in enumerate(resolution,1):
        cid=r['nnunet_case_id'];s=records[cid]
        one=r['is_native_1mm']=='1';fine=r['is_native_fine_inplane_le_0p5mm']=='1'
        if one==fine:raise ValueError('Unexpected resolution group')
        group='1mm' if one else 'LE0p5mm';counts[group]+=1
        real=[i for i in (0,1) if s[f'ch{i}_type'].startswith('real_')]
        if len(real)!=1:raise ValueError('Expected one real channel')
        real=real[0];folder=f'{group}/{cid}'
        for name,ch in [('real',real),('synthetic',1-real)]:
            rel=f'samples/{folder}/{name}.nii.gz'
            assets[rel]=copy_asset(Path(s[f'ch{ch}_original_path']),root/rel)
        samples.append(dict(folder=folder,sequence=s['sequence_label'].upper(),case_id=cid,
                            native_spacing_xyz=[float(r['native_spacing_'+a]) for a in 'xyz']))
        (root/'assets.partial.json').write_text(json.dumps(assets,indent=2)+'\n')
        print(f'SAMPLE_COPIED {idx}/78 {folder}',flush=True)
    if counts!={'1mm':31,'LE0p5mm':47}:raise ValueError(counts)
    for group in ('all','1mm','LE0p5mm'):
        with (root/'samples'/f'{group}.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['folder','sequence'])
            writer.writerows((r['folder'],r['sequence']) for r in samples if group=='all' or r['folder'].startswith(group+'/'))
    # First fine-resolution and first 1mm volume for portable smoke/regression checks.
    with (root/'samples/smoke.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['folder','sequence'])
        writer.writerows((r['folder'],r['sequence']) for g in ('1mm','LE0p5mm') for r in [next(x for x in samples if x['folder'].startswith(g+'/'))])
    (root/'samples/manifest.json').write_text(json.dumps(samples,indent=2)+'\n')
    (root/'assets.json').write_text(json.dumps(dict(status='completed',model_job=123195,checkpoint_epoch=1000,
        sample_counts=counts,files=assets,total_bytes=sum(v['bytes'] for v in assets.values())),indent=2)+'\n')
    (root/'assets.partial.json').unlink(missing_ok=True)
    print('BUNDLE_ASSETS_COMPLETE',counts,flush=True)


if __name__=='__main__':main()
