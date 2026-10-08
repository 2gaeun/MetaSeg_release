"""Refresh or verify the portable bundle file hashes after an intentional migration."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify',action='store_true')
    a=p.parse_args(); root=a.bundle
    path=root/'assets.json'; manifest=json.loads(path.read_text())
    if a.verify:
        for name,info in manifest['files'].items():
            f=root/name
            if f.is_symlink() or f.stat().st_size!=info['bytes'] or digest(f)!=info['sha256']:
                raise ValueError(f'Bundle mismatch: {name}')
        print('ASSETS_VERIFIED',len(manifest['files']))
        return
    selected=set(manifest['files'])
    for directory in ('src','docker','weights','tools','tests'):
        for f in (root/directory).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith('.pyc'):
                selected.add(str(f.relative_to(root)))
    selected.update(('run.sh','HANDOFF.md','.gitignore','.dockerignore'))
    result={}
    for name in sorted(selected):
        f=root/name
        if f.is_symlink(): raise ValueError(f'External symlink forbidden: {name}')
        result[name]={'bytes':f.stat().st_size,'sha256':digest(f)}
    manifest.update(model_job=131329, dataset=510, checkpoint_epoch=950,
                    checkpoint_filename='checkpoint_epoch_0950.pth',
                    trainer='nnUNetTrainerBrainMetaFocalLR3e3Components',
                    segmentation_weights='weights/segmentation_131329',
                    space='pseudo-1mm model space', skull_stripping=False)
    manifest['files']=result
    manifest['total_bytes']=sum(r['bytes'] for r in result.values())
    manifest['domain_adaptation']={'t1ce2bb':{'job':112788,'step':175000},
                                   'bb2t1ce':{'job':114599,'step':150000},'weights':'EMA float32'}
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('ASSETS_REFRESHED',len(result),manifest['total_bytes'])


if __name__=='__main__': main()
