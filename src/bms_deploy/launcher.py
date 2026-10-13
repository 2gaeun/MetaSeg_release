"""User-facing inference with fixed input/output folders."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def input_list(folder):
    source = folder / 'series.csv'
    with source.open(newline='', encoding='utf-8-sig') as f:
        rows = [[x.strip() for x in row] for row in csv.reader(f) if any(x.strip() for x in row)]
    if rows and len(rows[0]) == 2 and rows[0][0].lower() in ('id','file_id','series_id','folder') and rows[0][1].lower() == 'sequence':
        rows.pop(0)
    if not rows:
        raise ValueError('series.csv에 처리할 영상 목록을 입력하세요.')
    modes, names, seen = set(), {}, set()
    for row in rows:
        if len(row) != 2:
            raise ValueError('series.csv는 ID와 sequence 두 열이어야 합니다.')
        name, sequence = row
        relative = Path(name)
        if not name or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('ID에는 input 안의 파일 이름 또는 폴더 이름을 적으세요.')
        if sequence.upper() not in ('T1CE','BB'):
            raise ValueError(f'{name}: sequence는 T1CE 또는 BB로 적으세요.')
        if name in seen:
            raise ValueError(f'중복 ID: {name}')
        seen.add(name)
        if (folder / name).is_dir():
            mode = 'dicom'
        else:
            if name.endswith(('.nii','.nii.gz')) or '/' in name or '\\' in name:
                raise ValueError(f'{name}: NIfTI ID에는 경로와 .nii/.nii.gz를 제외하세요.')
            matches = list(folder.rglob(name+'.nii')) + list(folder.rglob(name+'.nii.gz'))
            if len(matches) != 1:
                raise ValueError(f'{name}: input에서 해당 영상을 하나만 찾을 수 있어야 합니다. 파일 이름과 CSV를 확인하세요.')
            mode = 'nifti'
        modes.add(mode)
        filename = re.sub(r'[^A-Za-z0-9_.-]', '_', name).strip('.')[:100] or 'volume'
        if filename in names.values():
            raise ValueError('결과 이름이 중복됩니다. 입력 ID를 서로 다르게 바꾸세요.')
        names[name] = filename
    if len(modes) != 1:
        raise ValueError('한 번에 NIfTI 또는 DICOM 한 종류만 처리하세요.')
    return rows, modes.pop(), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    try:
        rows, mode, names = input_list(root/'input')
    except (ValueError, OSError) as exc:
        print('입력 확인: '+str(exc), file=sys.stderr)
        return 2
    base = root/'output'
    base.mkdir(exist_ok=True)
    run = base/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    display = Path('output')/run.name
    logs = run/'logs'
    logs.mkdir(parents=True)
    listing = logs/'input.csv'
    with listing.open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    internal = run/'.work'
    command = [sys.executable, '-m', 'bms_deploy.cli',
               '--input-root', str(root/'input'), '--csv', str(listing), '--input-format', mode,
               '--output', str(internal), '--weights', str(root/'weights/segmentation_131329'),
               '--da-weights', str(root/'weights'), '--threads', '4',
               '--skull-strip', 'on', '--synthstrip-device', 'cpu']
    print(f'총 {len(rows)}개 영상 처리를 시작합니다. 결과: {display}', flush=True)
    done = 0
    last_update = time.monotonic()
    with (logs/'run.log').open('w') as logfile:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                logfile.write(line); logfile.flush()
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'VOLUME_START':
                    print(f'영상 처리 중 ({done+1}/{len(rows)}). 창을 닫지 마세요.', flush=True)
                elif parts[0] == 'VOLUME_DONE':
                    done += 1
                    print(f'영상 {done}/{len(rows)} 처리 완료', flush=True)
                elif parts[0] == 'VOLUME_ERROR':
                    print('처리하지 못한 영상이 있습니다. 마지막 안내를 확인하세요.', flush=True)
                elif time.monotonic()-last_update > 60:
                    print('계속 처리 중입니다. 잠시 기다려 주세요.', flush=True)
                    last_update = time.monotonic()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate(); process.wait()
            print(f'중단했습니다. 기록: {display}/logs', file=sys.stderr)
            return 130
    manifest_path = internal/'manifest.json'
    if not manifest_path.exists() and code == 0:
        code = 1
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'completed' or len(manifest['volumes']) != len(rows):
            code = code or 1
        for record in manifest['volumes']:
            original = record.get('file_id', record.get('folder'))
            filename = names[original]
            source = internal/(record['id']+'.nii.gz')
            source.rename(run/(filename+'.nii.gz'))
            (internal/record['id']/'logs').rename(logs/filename)
            (internal/record['id']).rmdir()
            record['output_file'] = filename+'.nii.gz'
        (logs/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        for p in internal.iterdir():
            if p.is_file() and p.suffix == '.json':
                p.rename(logs/('pipeline_'+p.name))
        if not any(internal.iterdir()):internal.rmdir()
    if code:
        print(f'일부 또는 전체 처리가 실패했습니다. 담당자에게 {display}/logs/run.log 를 전달하세요.', file=sys.stderr)
        return code
    print(f'모든 처리가 완료되었습니다. 병변 마스크는 {display} 폴더의 .nii.gz 파일입니다.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
