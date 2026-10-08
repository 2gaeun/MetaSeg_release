import argparse
import json
from pathlib import Path
from .dicom_input import read_input_csv, scan_series, read_series, output_id
from .geometry import load_nifti, require_same_grid


def main():
    p = argparse.ArgumentParser(description='In-memory DICOM -> I2SB DA -> Dataset510 two-channel segmentation')
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True, help='two columns: folder,sequence (DICOM) or file_id,sequence (NIfTI); header optional')
    p.add_argument('--input-format', choices=['dicom','nifti-pairs','nifti'], default='dicom')
    p.add_argument('--synthetic-root', type=Path, help='Precomputed pseudo synthetic: <output_id>.nii.gz; real/transform generated from native input')
    p.add_argument('--weights', type=Path, required=True, help='Bundled Dataset510 segmentation weights')
    p.add_argument('--da-weights', type=Path, help='Directory holding the two bundled I2SB EMA models')
    p.add_argument('--da-batch-size', type=int, default=8)
    p.add_argument('--da-nfe', type=int, default=50)
    p.add_argument('--seed', type=int, choices=[0], default=0, help='DA seed fixed at 0 for every volume')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--validate-inputs-only', action='store_true')
    args = p.parse_args()
    if args.threads < 1: raise ValueError('threads must be positive')
    rows = read_input_csv(args.input_root, args.csv, args.input_format)
    args.output.mkdir(parents=True, exist_ok=False)
    records, failed = [], []
    segmenter = adapter = None
    try:
        for folder, sequence, path in rows:
            try:
                series = scan_series(path) if args.input_format == 'dicom' else [('', None)]
                for uid, files in series:
                    cid = output_id(folder, uid)
                    print('VOLUME_START', cid, 'sequence='+sequence, flush=True)
                    real = read_series(files) if files is not None else load_nifti(
                        path if args.input_format == 'nifti' else path/'real.nii.gz')
                    identity = {'file_id': folder} if args.input_format == 'nifti' else {'folder': folder}
                    if args.validate_inputs_only:
                        records.append(dict(id=cid,**identity,sequence=sequence,series_uid=uid,
                            shape=list(real.data.shape),affine=real.affine.tolist(),status='input_validated'))
                        continue
                    native_real = real if args.input_format != 'nifti-pairs' else None
                    payload = None
                    if args.input_format == 'nifti-pairs':
                        synthetic = load_nifti(path/'synthetic.nii.gz')
                        transform = json.loads((path/'transform.json').read_text())
                    elif args.synthetic_root:
                        from .da_adapter import prepare
                        from .pseudo import make_transform, network_reference
                        from .da_adapter import from_sitk
                        from .geometry import Volume
                        original, work, source, _ = prepare(real)
                        transform = make_transform(original, work, cid, sequence)
                        real = Volume(source.numpy().transpose(2,1,0), from_sitk(network_reference(work)).affine)
                        synthetic = load_nifti(args.synthetic_root/(cid+'.nii.gz'))
                    else:
                        from .da_adapter import DomainAdapter
                        if adapter is None:
                            import torch
                            torch.set_num_threads(args.threads)
                            adapter = DomainAdapter(args.da_weights or args.weights.parent, args.device,
                                                    args.da_batch_size, args.da_nfe, args.seed)
                        synthetic, payload = adapter.generate(real, sequence, cid)
                        from .geometry import Volume
                        real = Volume(payload[0].numpy().transpose(2,1,0), payload[2])
                        transform = payload[3]['transform']
                    require_same_grid(real, synthetic)
                    if segmenter is None:
                        from .segment import Segmenter
                        segmenter = Segmenter(args.weights, args.device, args.threads)
                    result = segmenter.predict(real, synthetic, sequence, args.output/cid, args.debug, transform=transform)
                    if native_real is not None:
                        restored = load_nifti(args.output/(cid+'.nii.gz'))
                        require_same_grid(native_real, restored)
                    if payload is not None:
                        from .da_adapter import save_debug
                        info = payload[3]
                        (args.output/cid/'logs/da_result.json').write_text(json.dumps(info,indent=2)+'\n')
                        result['domain_adaptation'] = {k: info[k] for k in ('training_job','checkpoint_step','nfe','seed')}
                        if args.debug: save_debug(payload, args.output/cid/'logs/debug')
                        del payload
                    records.append(dict(id=cid,**identity,sequence=sequence,series_uid=uid,**result))
                    print('VOLUME_DONE', cid, flush=True)
            except Exception as exc:
                failed.append(dict(**({'file_id': folder} if args.input_format == 'nifti' else {'folder': folder}),
                                   error=str(exc),type=type(exc).__name__))
                (args.output/'errors.json').write_text(json.dumps(failed,indent=2)+'\n')
                print('VOLUME_ERROR',folder,type(exc).__name__,str(exc),flush=True)
            (args.output/'manifest.partial.json').write_text(json.dumps(records,indent=2)+'\n')
    finally:
        if adapter is not None:
            adapter.close()
    (args.output/'manifest.json').write_text(json.dumps(dict(status='failed' if failed else 'completed',
        volumes=records,errors=failed,debug=args.debug,input_format=args.input_format,
        validation_only=args.validate_inputs_only),indent=2)+'\n')
    if failed: raise SystemExit(1)


if __name__ == '__main__': main()
